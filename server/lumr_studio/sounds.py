"""Tell likely laughs from hesitations among the transcript's sound events.

Hammy marks every voiced sound in a gap between words as ``[vocalization]``.
Some are the creator laughing after a joke; most are a breath, an "uhh", or
the tail of a word. Claude reads the packed transcript without hearing it, so
without a label it cuts jokes short and trims the pause a laugh needs.

The rule, measured on a 21 minute talking-head video with 98 events (the
numbers behind each constant are next to it):

* A laugh is a train of short bursts, about 4 to 5 a second, and breathy: the
  "h" between the bursts has no pitch, so under half the sounding time is
  voiced. A hesitation is one steady voiced blob (voiced share 0.8 to 1.0).
* A laugh follows a sentence end, the punchline.

``classify_events`` applies the rule. ``measure_events`` gets the numbers from
the video with ffmpeg and numpy. ``load_or_measure_labels`` caches the result
beside the project, and ``protected_spans`` turns laughs into the source spans
an automatic trim must leave alone.

Without audio measures the labels fall back to length and position, which can
only ever say "possible laugh". A wrong laugh label protects dead air, so the
rule prefers missing a laugh to inventing one.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from lumr_studio import word_times
from lumr_studio.errors import StudioError
from lumr_studio.project import Project, write_json_atomic
from lumr_studio.transcript import _is_sentence_end, is_event

AudioMeasure = Callable[[Path, list[tuple[float, float]]], list[dict]]
Decoder = Callable[[Path, float, float, int], np.ndarray]

LAUGH = "laugh"
SOUND = "sound"
LIKELY = "likely"
POSSIBLE = "possible"

# ── Measuring ─────────────────────────────────────────────────────────────────

# Speech detail sits below 8 kHz, so 16 kHz mono keeps everything the rule uses
# and decodes fast.
SAMPLE_RATE = 16000
# Loudness envelope: 30 ms windows every 10 ms. Laugh bursts are 80 to 150 ms
# long, so each shows up as several frames with a clear dip between bursts.
ENVELOPE_WINDOW_SECONDS = 0.030
ENVELOPE_HOP_SECONDS = 0.010
# A burst is a loudness peak standing at least this far above the dips on
# either side. The measured laughs dip 6 to 15 dB between bursts; a held "uhh"
# wobbles by 1 to 3 dB.
BURST_PROMINENCE_DB = 6.0
# Only frames within this range of the event's loudest frame count as sounding,
# so room noise in a quiet gap is never counted as a burst or as voice.
SOUNDING_RANGE_DB = 20.0
# Anything quieter than this (dB relative to full scale) is silence whatever
# the event's own level.
SILENCE_FLOOR_DBFS = -50.0
# Burst peaks in one laugh sit 0.12 to 0.40 s apart (2.5 to 8 a second).
# Closer peaks are one burst with a ripple; wider ones are separate sounds.
LAUGH_SPACING_SECONDS = (0.12, 0.40)
# Pitch check: 40 ms frames every 20 ms, a voice between 75 and 450 Hz. A
# frame whose normalised autocorrelation peak reaches 0.6 is voiced (pitched).
PITCH_FRAME_SECONDS = 0.040
PITCH_HOP_SECONDS = 0.020
PITCH_RANGE_HZ = (75.0, 450.0)
VOICED_CORRELATION = 0.6

UNMEASURED: dict[str, Any] = {}


def _frames(samples: np.ndarray, size: int, hop: int) -> np.ndarray:
    """``(n, size)`` overlapping frames of ``samples``; zero rows when too short."""
    count = (len(samples) - size) // hop + 1 if len(samples) >= size else 0
    if count <= 0:
        return np.zeros((0, size), dtype=np.float64)
    index = np.arange(size)[None, :] + hop * np.arange(count)[:, None]
    return samples[index].astype(np.float64)


def _level_db(frames: np.ndarray) -> np.ndarray:
    """Mean power of each frame in dB relative to full scale."""
    return 10 * np.log10(np.mean(frames**2, axis=1) + 1e-10)


def _sounding_floor(levels: np.ndarray) -> float:
    return max(float(levels.max()) - SOUNDING_RANGE_DB, SILENCE_FLOOR_DBFS)


def _prominent_peaks(levels: np.ndarray, floor: float) -> list[int]:
    """Indexes of the loudness peaks that stand ``BURST_PROMINENCE_DB`` above both neighbouring dips.

    A peak's dip on each side is the lowest point before the envelope rises
    above the peak again (or the edge of the span). Peaks below ``floor`` are
    ignored.
    """
    peaks: list[int] = []
    n = len(levels)
    for i in range(1, n - 1):
        if not (levels[i] >= levels[i - 1] and levels[i] > levels[i + 1]) or levels[i] <= floor:
            continue
        left = i
        left_min = levels[i]
        while left > 0 and levels[left - 1] <= levels[i]:
            left -= 1
            left_min = min(left_min, levels[left])
        right = i
        right_min = levels[i]
        while right < n - 1 and levels[right + 1] <= levels[i]:
            right += 1
            right_min = min(right_min, levels[right])
        if levels[i] - max(left_min, right_min) >= BURST_PROMINENCE_DB:
            peaks.append(i)
    return peaks


def _merge_close(peaks: list[int], levels: np.ndarray, min_gap: int) -> list[int]:
    """Keep the louder of any two peaks closer than ``min_gap`` frames."""
    kept: list[int] = []
    for p in peaks:
        if kept and p - kept[-1] < min_gap:
            if levels[p] > levels[kept[-1]]:
                kept[-1] = p
            continue
        kept.append(p)
    return kept


def _longest_train(peaks: list[int], max_gap: int) -> list[int]:
    """The longest run of consecutive peaks each at most ``max_gap`` frames after the last."""
    best: list[int] = []
    current: list[int] = []
    for p in peaks:
        current = current + [p] if current and p - current[-1] <= max_gap else [p]
        if len(current) > len(best):
            best = current
    return best


def _voiced_share(samples: np.ndarray, sample_rate: int) -> float:
    """Share of the sounding pitch frames that carry a pitch. 0.0 when nothing sounds."""
    size = int(PITCH_FRAME_SECONDS * sample_rate)
    frames = _frames(samples, size, int(PITCH_HOP_SECONDS * sample_rate))
    if not len(frames):
        return 0.0
    levels = _level_db(frames)
    sounding = levels > _sounding_floor(levels)
    centred = frames - frames.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(centred, n=2 * size, axis=1)
    autocorr = np.fft.irfft(np.abs(spectrum) ** 2, axis=1)[:, :size]
    lo = int(sample_rate / PITCH_RANGE_HZ[1])
    hi = int(sample_rate / PITCH_RANGE_HZ[0])
    energy = np.maximum(autocorr[:, 0], 1e-12)
    pitched = autocorr[:, lo:hi].max(axis=1) / energy >= VOICED_CORRELATION
    return float((pitched & sounding).sum() / sounding.sum()) if sounding.any() else 0.0


def measure_samples(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict[str, Any]:
    """The laugh measures of one mono clip (floats in -1..1).

    Returns ``{bursts, burst_rate, voiced}``:

    * ``bursts``: peaks in the longest train spaced like laughter
      (``LAUGH_SPACING_SECONDS`` apart).
    * ``burst_rate``: that train's bursts per second, 0.0 under two bursts.
    * ``voiced``: share of the sounding time that has a pitch, 0.0 to 1.0.

    A clip too short for one envelope window measures as no bursts.
    """
    hop = int(ENVELOPE_HOP_SECONDS * sample_rate)
    frames = _frames(np.asarray(samples, dtype=np.float64), int(ENVELOPE_WINDOW_SECONDS * sample_rate), hop)
    if len(frames) < 3:
        return {"bursts": 0, "burst_rate": 0.0, "voiced": 0.0}
    levels = _level_db(frames)
    min_gap, max_gap = (round(s / ENVELOPE_HOP_SECONDS) for s in LAUGH_SPACING_SECONDS)
    peaks = _merge_close(_prominent_peaks(levels, _sounding_floor(levels)), levels, min_gap)
    train = _longest_train(peaks, max_gap)
    span = (train[-1] - train[0]) * ENVELOPE_HOP_SECONDS if len(train) > 1 else 0.0
    return {
        "bursts": len(train),
        "burst_rate": round((len(train) - 1) / span, 2) if span > 0 else 0.0,
        "voiced": round(_voiced_share(np.asarray(samples, dtype=np.float64), sample_rate), 2),
    }


def decode_span(video: Path, start: float, end: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Mono float samples of ``video`` from ``start`` to ``end`` seconds.

    Seeks on the input so only the span is decoded. Reads the video, never
    writes. Raises StudioError when ffmpeg fails.
    """
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin",
        "-ss", f"{start:.3f}", "-t", f"{max(end - start, 0.0):.3f}", "-i", str(video),
        "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=False)
    except FileNotFoundError:
        raise StudioError("ffmpeg is not installed or not on PATH. Install ffmpeg to measure sounds.") from None
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()[-300:]
        raise StudioError(
            f"ffmpeg could not read audio from {video.name} at {start:.2f}s: {detail} "
            "Check that the source video plays and has an audio track."
        )
    return np.frombuffer(result.stdout, dtype=np.int16).astype(np.float64) / 32768.0


def measure_events(
    video: Path,
    spans: list[tuple[float, float]],
    *,
    decode: Decoder = decode_span,
) -> list[dict]:
    """The production ``AudioMeasure``: one ``measure_samples`` result per span, in order.

    Decodes only the spans (one short ffmpeg read each). Raises StudioError
    when ffmpeg fails.
    """
    return [measure_samples(decode(video, float(a), float(b), SAMPLE_RATE), SAMPLE_RATE) for a, b in spans]


# ── Classifying ───────────────────────────────────────────────────────────────

# A likely laugh needs a train of at least this many bursts. The three events
# after the clearest punchlines had trains of 5, 9 and 12. Of the other 95,
# one reached 4 and it was a steady voiced sound mid-sentence; the rest had 3
# or fewer. 4 keeps a margin under the weakest real laugh.
LIKELY_MIN_BURSTS = 4
# Laughter runs at roughly 4 to 6 bursts a second; the measured laughs ran at
# 3.9 to 5.3.
LAUGH_RATE = (3.0, 7.0)
# Laughs are breathy but still pitched: the measured ones were voiced 0.31 to
# 0.44 of their sounding time. Held "uhh" sounds and word tails measured 0.75
# to 1.0; a plain breath or a near-silent gap measured 0.0.
LAUGH_VOICED = (0.15, 0.5)
# A possible laugh (measured): after a punchline, breathy, and at least two
# bursts in laugh spacing or this long. On the test video, breathy events
# without a punchline were a throat clear and mid-sentence breaths.
POSSIBLE_MIN_BURSTS = 2
POSSIBLE_MIN_SECONDS = 1.2
# Without audio, only a long event right after a sentence end is a possible
# laugh. On the test video 6 events qualified at 1.8 s and 3 of them measured
# as laughs; at 1.5 s it was 3 of 12, at 1.2 s 3 of 15.
UNMEASURED_MIN_SECONDS = 1.8
# A laugh can follow a short connecting word the transcriber heard in the
# laughter ("pie. And [laugh]"). The sentence end may sit this many spoken
# words back, if it ended no more than this long before the event.
PUNCHLINE_MAX_WORDS_BETWEEN = 1
PUNCHLINE_MAX_SECONDS_BEFORE = 1.5
# A run-on sentence before a laugh can last 16 seconds. The joke is its end:
# the punchline starts after the sentence's last pause of this length, and
# never more than PUNCHLINE_MAX_SECONDS before the sentence ends.
PUNCHLINE_PAUSE_SECONDS = 0.5
PUNCHLINE_MAX_SECONDS = 8.0


def _punchline(words: list[dict[str, Any]], index: int) -> list[float] | None:
    """``[start, end]`` of the sentence that ends just before event ``words[index]``.

    The sentence end must be the last spoken word, or at most
    ``PUNCHLINE_MAX_WORDS_BETWEEN`` words back and within
    ``PUNCHLINE_MAX_SECONDS_BEFORE`` of the event. None otherwise.

    Of a sentence longer than ``PUNCHLINE_MAX_SECONDS`` only the end counts:
    from its last pause of ``PUNCHLINE_PAUSE_SECONDS`` inside that limit, or
    from the first word inside the limit when it has no such pause.
    """
    event_start = float(words[index]["start"])
    spoken_before = [i for i in range(index - 1, -1, -1) if not is_event(words[i])]
    for between, i in enumerate(spoken_before[:PUNCHLINE_MAX_WORDS_BETWEEN + 1]):
        if not _is_sentence_end(words[i]):
            continue
        if between and event_start - float(words[i]["end"]) > PUNCHLINE_MAX_SECONDS_BEFORE:
            return None
        rest = spoken_before[between + 1:]
        end = float(words[i]["end"])
        run_on = _runs_past_limit(words, rest, end)
        first = i
        for j in rest:
            if _is_sentence_end(words[j]) or end - float(words[j]["start"]) > PUNCHLINE_MAX_SECONDS:
                break
            if run_on and float(words[first]["start"]) - float(words[j]["end"]) >= PUNCHLINE_PAUSE_SECONDS:
                break
            first = j
        return [float(words[first]["start"]), end]
    return None


def _runs_past_limit(words: list[dict[str, Any]], before: list[int], end: float) -> bool:
    """Whether the sentence ending at ``end`` started more than ``PUNCHLINE_MAX_SECONDS`` earlier.

    ``before`` are the indexes of the spoken words ahead of the sentence end, nearest first.
    """
    for j in before:
        if _is_sentence_end(words[j]):
            return False
        if end - float(words[j]["start"]) > PUNCHLINE_MAX_SECONDS:
            return True
    return False


def _judge(seconds: float, punchline: list[float] | None, measure: dict[str, Any]) -> tuple[str, str]:
    """``(kind, confidence)`` for one event from its length, position and measures."""
    if not measure:
        if punchline and seconds >= UNMEASURED_MIN_SECONDS:
            return LAUGH, POSSIBLE
        return SOUND, POSSIBLE
    bursts = int(measure.get("bursts", 0))
    rate = float(measure.get("burst_rate", 0.0))
    breathy = LAUGH_VOICED[0] <= float(measure.get("voiced", 1.0)) <= LAUGH_VOICED[1]
    laugh_train = bursts >= LIKELY_MIN_BURSTS and LAUGH_RATE[0] <= rate <= LAUGH_RATE[1]
    if not breathy:
        return SOUND, POSSIBLE
    if laugh_train:
        return LAUGH, LIKELY if punchline else POSSIBLE
    if punchline and (bursts >= POSSIBLE_MIN_BURSTS or seconds >= POSSIBLE_MIN_SECONDS):
        return LAUGH, POSSIBLE
    return SOUND, POSSIBLE


def classify_events(words: list[dict[str, Any]], *, measures: list[dict] | None = None) -> list[dict]:
    """Sound labels for every event in ``words``, sorted by start.

    ``words`` is a sorted transcript. ``measures``, when given, holds one
    ``measure_samples``-shaped dict per event in the same order; an empty dict
    means that event was not measured and is judged by length and position.
    Returns ``[{start, end, seconds, kind, confidence, voiced, bursts,
    burst_rate, punchline}]``. ``confidence`` only matters for a laugh and is
    ``possible`` for a sound. ``voiced``, ``bursts`` and ``burst_rate`` are
    the measure, or None when the event was not measured (``is_filler``
    reads all three). ``punchline`` is set for laughs only.

    Raises StudioError when ``measures`` does not have one entry per event.
    """
    indexes = [i for i, w in enumerate(words) if is_event(w)]
    if measures is not None and len(measures) != len(indexes):
        raise StudioError(
            f"Got {len(measures)} sound measures for {len(indexes)} events. "
            "Measure again from the same transcript."
        )
    labels = []
    for n, i in enumerate(indexes):
        start, end = float(words[i]["start"]), float(words[i]["end"])
        seconds = round(end - start, 2)
        punchline = _punchline(words, i)
        measure = measures[n] if measures is not None else UNMEASURED
        kind, confidence = _judge(seconds, punchline, measure)
        labels.append({
            "start": start,
            "end": end,
            "seconds": seconds,
            "kind": kind,
            "confidence": confidence,
            "voiced": round(float(measure["voiced"]), 2) if measure else None,
            "bursts": int(measure["bursts"]) if measure else None,
            "burst_rate": round(float(measure["burst_rate"]), 2) if measure else None,
            "punchline": punchline if kind == LAUGH else None,
        })
    return sorted(labels, key=lambda label: label["start"])


# ── Filler vocalizations ────────────────────────────────────────────────────────

# A "[vocalization]" this voiced or more is a held "umm" or "uhh", the way a
# reader hears it: a filler, not a breath. On a 21 minute take, 25 of its 87
# sounds sat near a pause cut: 8 measured voiced 0.8 and up (an unmistakable
# held vowel), 10 under 0.5 (breathy, no held pitch: an inhale or a sigh),
# and 7 between the two, still a filler by ear. The line sits at the low end
# of that middle band, so a breath never reads as one.
FILLER_VOICED_MIN = 0.5


def is_filler(label: dict[str, Any]) -> bool:
    """Whether a sound label is a held "um" or "uh": a ``SOUND`` voiced ``FILLER_VOICED_MIN`` or more, and no laugh train.

    Measured only: an event never measured (``voiced`` is None) answers
    False, so nothing is cut on a guess. ``kind`` already keeps out a laugh
    ClipForge would call likely or possible, but a giggle voiced 0.5 or more
    still judges as ``SOUND`` (``_judge`` needs a breathy voice, under
    ``LAUGH_VOICED``, to call one), so this reads the bursts and their rate
    itself: a train of ``LIKELY_MIN_BURSTS`` or more in ``LAUGH_RATE`` is a
    laugh whatever its voicing, and is never a filler.
    """
    if label.get("kind") != SOUND or label.get("voiced") is None or label["voiced"] < FILLER_VOICED_MIN:
        return False
    bursts, rate = label.get("bursts"), label.get("burst_rate")
    if bursts is not None and rate is not None and bursts >= LIKELY_MIN_BURSTS and LAUGH_RATE[0] <= rate <= LAUGH_RATE[1]:
        return False
    return True


# ── Protecting ────────────────────────────────────────────────────────────────

# After a laugh the creator often pauses before the next line; that beat is
# part of the joke. Protect up to this much of it.
PAUSE_AFTER_LAUGH_SECONDS = 1.0
# The pause before a punchline is the beat that sets it up. Protect up to this
# much of it.
PAUSE_BEFORE_PUNCHLINE_SECONDS = 1.0


def protected_spans(labels: list[dict], words: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """Source spans an automatic trim must leave alone, sorted and merged.

    For each laugh (likely or possible): from the pause before its punchline,
    capped at ``PAUSE_BEFORE_PUNCHLINE_SECONDS`` (or from the laugh itself
    when it has no punchline), to the laugh's end plus the pause after it,
    capped at ``PAUSE_AFTER_LAUGH_SECONDS``.
    """
    starts = sorted(float(w["start"]) for w in words if not is_event(w))
    ends = sorted(float(w["end"]) for w in words if not is_event(w))
    spans: list[tuple[float, float]] = []
    for label in labels:
        if label["kind"] != LAUGH:
            continue
        end = float(label["end"])
        next_word = next((s for s in starts if s >= end), end)
        begin = float(label["start"])
        if label.get("punchline"):
            begin = float(label["punchline"][0])
            spoken_before = max((e for e in ends if e <= begin), default=begin)
            begin -= min(begin - spoken_before, PAUSE_BEFORE_PUNCHLINE_SECONDS)
        spans.append((begin, end + min(max(next_word - end, 0.0), PAUSE_AFTER_LAUGH_SECONDS)))
    merged: list[tuple[float, float]] = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


# ── Caching ───────────────────────────────────────────────────────────────────

SOUNDS_FILE = "sounds.json"
# Bump when the measures or the rule change, so old caches are rebuilt.
SOUNDS_VERSION = 2


def _made_as_before(times: str) -> int | None:
    """How the word times of a cache that does not say were made: measured ones, before words had room."""
    return word_times.WITHOUT_ROOM_VERSION if times == word_times.MEASURED else None


def load_or_measure_labels(
    project: Project,
    words: list[dict[str, Any]],
    *,
    measure: AudioMeasure = measure_events,
) -> list[dict]:
    """Labels cached in ``<project.root>/sounds.json``, measured when the cache is stale.

    The cache is keyed on the transcript file's size and mtime, so a
    re-transcribe invalidates it. It also says which word times the labels
    were judged on, and how those were made: a punchline is found from the
    words before a laugh, so when the word times change the sounds are
    judged again. The measures
    already saved are used again when the sounds sit where they sat; sounds
    that moved are measured again. With no transcript file on disk the
    labels are measured and not cached. Raises StudioError when measuring
    fails.
    """
    events = [w for w in words if is_event(w)]
    if not events:
        return []
    key = word_times.transcript_key(project)
    times = word_times.source_of(project)
    made_as = word_times.made_as(times)
    spans = [[float(e["start"]), float(e["end"])] for e in events]
    cache_path = project.root / SOUNDS_FILE
    measures = None
    if key is not None and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cached = {}
        if cached.get("version") == SOUNDS_VERSION and cached.get("transcript") == key:
            same_times = cached.get("word_times", word_times.ESTIMATED) == times
            if same_times and cached.get("word_times_made_as", _made_as_before(times)) == made_as:
                return cached["labels"]
            if cached.get("spans", spans) == spans:
                measures = cached.get("measures")
    if not isinstance(measures, list) or len(measures) != len(events):
        measures = measure(project.video, [(a, b) for a, b in spans])
    labels = classify_events(words, measures=measures)
    if key is not None:
        write_json_atomic(cache_path, {
            "version": SOUNDS_VERSION,
            "transcript": key,
            "word_times": times,
            "word_times_made_as": made_as,
            "spans": spans,
            "labels": labels,
            "measures": measures,
        })
    return labels
