"""Find the sounds between a take's words, and tell laughs, ums and other sounds apart among them.

A sound is a transcript entry of ``type: "event"``: a laugh, a giggle, a
breath, an "uhh". Claude reads the packed transcript without hearing it, and
the pace reads the gap a sound sits in as a pause, so a sound that is not in
the transcript gets cut like silence. Three steps, each in this module:

1. Finding (``find_sounds``). The speech model writes words only. Its own
   check for sounds (``speech/transcribe._detect_vocalizations``) loads the
   video with librosa, which from 1.0 on opens only what libsndfile opens:
   no mp4, no mov. The load fails, the check swallows the failure and
   answers "no sounds", so every transcript made on Python 3.12 or later,
   where 1.0 is the locked version, has none. Where the load does work the
   check wants a pitch, which drops a breathy laugh. So the server looks
   itself, with numpy and ffmpeg alone: the loudness of the whole take,
   20 ms at a time, and every stretch between two words that is clearly
   louder than the room becomes a sound. The numbers behind each constant
   are next to it.
2. Labelling (``classify_events``, ``measure_events``). The rule, measured on
   two talking-head videos:

   * A laugh is a train of short bursts, about 4 to 5 a second. Some laughs
     are breathy (the "h" between the bursts has no pitch, so under half the
     sounding time is voiced), a giggle is pitched most of the way through.
     Neither is a steady sound: a hesitation is one voiced blob.
   * A laugh follows a sentence end, the punchline.
   * An um (``UM``) is a held hesitation, an "um" or "uh" the speech model
     wrote no word for: one steady voiced sound under a second, no louder
     than the speech around it (``is_filler``), and no short run-on of the
     word before it. The speech model writes almost none of them as words:
     on a 28 minute test take it wrote 3 and 10 more were sounds.

3. Protecting (``protected_spans``, ``whole_spans``). The rule every
   automatic cut goes by:

   * A laugh, likely or possible, and any sound that lasts
     ``LONG_SOUND_SECONDS`` or more is never taken, in whole or in part, by a
     pause cut, by the Filler words switch, by the Stutters switch, or by an
     edge of one of Claude's cuts that the server moved. Only a cut that was
     asked for over the whole of it takes it out.
   * A short plain sound (a breath, an um, under a second and no laugh) is
     never taken in part, and no pause cut is planned over one. No rule
     takes one whole either. An um is judged by reading, the way "like" is:
     ``find_words`` lists every one in its sentence, Claude picks the ones
     that do not help, and a pick is the one thing that takes an um out
     (``edit.parse_pick``). Some hold a sentence together, and no measure
     of the sound says which.

``load_or_measure_labels`` caches the labels beside the project.

Without audio measures the labels fall back to length and position, which can
only ever say "possible laugh". A wrong laugh label protects dead air and a
missed one lets a cut through a laugh, so where the measures leave room for
doubt the rule calls it a laugh.
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from lumr_studio import word_times
from lumr_studio.engine.microcut_pacing import GAP_LENGTH_MIN
from lumr_studio.errors import StudioError
from lumr_studio.project import Project, now_iso, write_json_atomic, write_text_atomic
from lumr_studio.transcript import EVENT_TYPE, UM, _is_sentence_end, is_event, load_transcript, plain_text

AudioMeasure = Callable[[Path, list[tuple[float, float]]], list[dict]]
Decoder = Callable[[Path, float, float, int], np.ndarray]

LAUGH = "laugh"
SOUND = "sound"
# ``UM`` (from ``transcript``) is the third kind: a held hesitation between
# two words, an "um" or "uh" with no word in the transcript. See ``is_um``.
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


# ── Finding ───────────────────────────────────────────────────────────────────

# The word every sound of the transcript carries, and what a sound found here
# says about itself, so a second look can tell its own finds from the speech
# model's and replace them.
SOUND_WORD = "[vocalization]"
FOUND_BY = "found_by"
BY_LOUDNESS = "loudness"
# A found sound also says how far its loudest window stands over the take's
# loud speech, in dB (under it when negative). ``is_filler`` reads it.
OVER_SPEECH = "over_speech_db"

# Loudness of the take: one level for every 20 ms, windows side by side. A
# laugh burst lasts 80 to 150 ms, four windows or more.
FIND_WINDOW_SECONDS = 0.020
# The room's own noise is the level the quietest tenth of the take sits
# under, and loud speech the level the loudest tenth of the time inside
# words sits over. On a 27 minute take in a quiet room they were -58 and
# -13 dB.
NOISE_PERCENTILE = 10.0
SPEECH_PERCENTILE = 90.0
# A window sounds when it is no more than this far under loud speech: as loud
# as soft speech. On that take the line fell at -30 dB, where the 14 sounds
# marked by hand between its words all stood out of their gaps, each within
# 30 ms of the marks, laughs and giggles peaking -7 to -10 dB and
# half-second "mm"s -14 to -17. A breath sits 25 dB or more under speech
# and stays out.
BELOW_SPEECH_DB = 17.0
# In a loud room the line rises with the noise: a sound stands this far over
# the room or it is not clearly a sound. A take with one steady level end to
# end (a tone, a music bed) has nothing over the line and gets no sounds.
ABOVE_NOISE_DB = 10.0
# A sound needs this much time over the line: less is a click, a lip smack or
# the stray end of a word. The half-second sounds on that take had 0.30 to
# 0.48 s.
MIN_LOUD_SECONDS = 0.3
# Loud stretches less than this apart are one sound. A long laugh comes in
# bouts with a breath between them: 0.86 s on that take, between a giggle's
# body and its last soft bout. It is also the pause protected after a laugh
# (``PAUSE_AFTER_LAUGH_SECONDS``), so a cut could not go between two closer
# sounds anyway.
SAME_SOUND_QUIET_SECONDS = 1.0
# A found sound runs from its first loud window to its last, and this much
# further each way for the soft start and end, never past the gap it is in.
FOUND_PAD_SECONDS = 0.04
# Audio is read from ffmpeg this many seconds at a time, so a two hour take
# never sits in memory: only its levels do, 50 numbers a second.
FIND_READ_SECONDS = 60
# Sound decoded from start to end is counted in samples, and a video's clock
# is its timestamps. They part where the sound has a hole: a reel copied
# together from clips has one at every join, 10 to 20 ms each. On a reel of
# four clips a laugh 25 minutes in was found 40 ms early, so the 40 ms before
# it was guarded and its last 40 ms was not. This filter pads every hole of
# 2 ms or more with silence, and the count keeps time with the clock.
KEEP_TIME = "aresample=async=1:min_hard_comp=0.002:first_pts=0"

FOUND_FILE = "sounds-found.json"
# Bump when the way sounds are found changes, so every transcript is looked
# through again.
FOUND_VERSION = 1

# What every tool that reads sounds says while a transcript was never looked
# through. Written for Claude to act on.
NOT_CHECKED = (
    "This transcript was never checked for laughs and other sounds. None are marked and an "
    "automatic cut can take one. Call transcribe on this video: it finds them without "
    "transcribing again."
)
# What the join check says when a look found nothing.
NONE_FOUND = "No laughs or other sounds were found between the words of this video."

Loudness = Callable[[Path], np.ndarray]


def loudness_of(video: Path) -> np.ndarray:
    """The loudness of the whole ``video``: one level in dB relative to full scale for every ``FIND_WINDOW_SECONDS``.

    Reads the video's sound through ffmpeg a minute at a time and keeps only
    the levels. Level ``i`` is the sound at ``i * FIND_WINDOW_SECONDS`` on
    the video's own clock, the one every cut is made on: ffmpeg fills each
    hole in the sound's timestamps with silence (``KEEP_TIME``). Reads the
    video, never writes. Raises StudioError when ffmpeg fails.
    """
    window = int(FIND_WINDOW_SECONDS * SAMPLE_RATE)
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-i", str(video),
        "-vn", "-af", KEEP_TIME, "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-",
    ]
    levels: list[np.ndarray] = []
    with tempfile.TemporaryFile() as said:
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=said)
        except FileNotFoundError:
            raise StudioError("ffmpeg is not installed or not on PATH. Install ffmpeg to find sounds.") from None
        left = b""
        with proc:
            while chunk := proc.stdout.read(2 * window * round(FIND_READ_SECONDS / FIND_WINDOW_SECONDS)):
                data = left + chunk
                whole = len(data) // (2 * window) * (2 * window)
                left = data[whole:]
                samples = np.frombuffer(data[:whole], dtype=np.int16).astype(np.float64) / 32768.0
                levels.append(_level_db(samples.reshape(-1, window)))
        if proc.returncode != 0:
            said.seek(0)
            detail = said.read().decode("utf-8", "replace").strip()[-300:]
            raise StudioError(
                f"ffmpeg could not read the sound of {video.name}: {detail} "
                "Check that the source video plays and has an audio track."
            )
    return np.concatenate(levels) if levels else np.zeros(0)


def _loud_runs(loud: np.ndarray, max_quiet: int) -> list[tuple[int, int, int]]:
    """``(first, last, count)`` of each run of the sorted window indexes ``loud``, a run ending at a quiet of ``max_quiet`` windows or more."""
    runs: list[tuple[int, int, int]] = []
    for i in (int(i) for i in loud):
        if runs and i - runs[-1][1] - 1 < max_quiet:
            runs[-1] = (runs[-1][0], i, runs[-1][2] + 1)
        else:
            runs.append((i, i, 1))
    return runs


def find_in_gaps(levels: np.ndarray, words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The sounds between the entries of ``words``, as new transcript entries sorted by start.

    ``levels`` is ``loudness_of`` the take and ``words`` its transcript on
    the best times there are, sounds it already holds included: nothing is
    found where an entry sits. A stretch of a gap is a sound when it is
    over the line (``BELOW_SPEECH_DB``, ``ABOVE_NOISE_DB``) for
    ``MIN_LOUD_SECONDS`` or more, loud stretches under
    ``SAME_SOUND_QUIET_SECONDS`` apart counting as one. Only gaps between
    two entries are read: what sounds before the first word or after the
    last is the camera being started and stopped more often than a laugh.
    Pure: reads no file.
    """
    tokens = sorted(words, key=lambda w: float(w["start"]))
    spoken = [w for w in tokens if not is_event(w)]
    if not len(levels) or not spoken:
        return []
    hop = FIND_WINDOW_SECONDS
    in_words = np.zeros(len(levels), dtype=bool)
    for w in spoken:
        in_words[round(float(w["start"]) / hop):round(float(w["end"]) / hop)] = True
    if not in_words.any():
        return []
    noise = float(np.percentile(levels, NOISE_PERCENTILE))
    speech = float(np.percentile(levels[in_words], SPEECH_PERCENTILE))
    line = max(speech - BELOW_SPEECH_DB, noise + ABOVE_NOISE_DB)
    max_quiet = round(SAME_SOUND_QUIET_SECONDS / hop)
    found = []
    at = float(tokens[0]["end"])
    for token in tokens[1:]:
        lo, hi = at, float(token["start"])
        at = max(at, float(token["end"]))
        if hi - lo < MIN_LOUD_SECONDS:
            continue
        first, last = math.ceil(lo / hop - 1e-6), math.floor(hi / hop + 1e-6)
        loud = np.flatnonzero(levels[first:last] > line) + first
        for a, b, count in _loud_runs(loud, max_quiet):
            if count * hop < MIN_LOUD_SECONDS - 1e-6:
                continue
            found.append({
                "word": SOUND_WORD,
                "start": round(max(lo, a * hop - FOUND_PAD_SECONDS), 3),
                "end": round(min(hi, (b + 1) * hop + FOUND_PAD_SECONDS), 3),
                "type": EVENT_TYPE,
                FOUND_BY: BY_LOUDNESS,
                OVER_SPEECH: round(float(levels[a:b + 1].max()) - speech, 1),
            })
    return found


def _found_before(project: Project) -> dict[str, Any] | None:
    """What the last look through this very transcript saved, or None when it was never looked through as it is now."""
    key = word_times.transcript_key(project)
    try:
        saved = json.loads((project.root / FOUND_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(saved, dict) or saved.get("version") != FOUND_VERSION:
        return None
    return saved if key is not None and saved.get("transcript") == key else None


def sounds_checked(project: Project) -> bool:
    """Whether the project's transcript, as it is now, was looked through for sounds."""
    return _found_before(project) is not None


def looked_on(project: Project) -> str | None:
    """The word times the last look read its gaps from (``word_times.MEASURED`` or ``ESTIMATED``), or None when there was none.

    A look on estimated times reads loose gaps: where a word's time runs
    over a laugh, the laugh is not in a gap. So once the times are measured
    the transcript is looked through again.
    """
    saved = _found_before(project)
    return str(saved.get("word_times")) if saved else None


def find_sounds(project: Project, *, loudness: Loudness = loudness_of) -> dict[str, Any]:
    """Look through the project's transcript for sounds and write the ones found into it.

    The gaps are read on the best word times there are
    (``word_times.read``): measured ones say where the words really sound,
    the transcript's own run long and hide half the sounds under words.
    Sounds an earlier look found are replaced; the speech model's own stay.
    The transcript file is written again only when its sounds changed, and
    then everything made from it (measured word times, sound labels, plans)
    is stale: the caller measures the word times again.

    Answers ``{found, on, changed}``: how many sounds this look found, the
    word times it read (``measured`` or ``estimated``), and whether the
    transcript file was written. Raises StudioError when there is no
    transcript or the video's sound can't be read; nothing is written then.
    """
    raw = load_transcript(project.words_path)
    times = word_times.read(project)
    theirs = [w for w in times.words if w.get(FOUND_BY) != BY_LOUDNESS]
    found = find_in_gaps(loudness(project.video), theirs)
    changed = [w for w in raw if w.get(FOUND_BY) == BY_LOUDNESS] != found
    if changed:
        kept = [w for w in raw if w.get(FOUND_BY) != BY_LOUDNESS]
        merged = sorted(kept + found, key=lambda w: float(w["start"]))
        write_text_atomic(project.words_path, json.dumps(merged, ensure_ascii=False, indent=2))
    write_json_atomic(project.root / FOUND_FILE, {
        "version": FOUND_VERSION, "made_at": now_iso(), "transcript": word_times.transcript_key(project),
        "word_times": times.source, "found": len(found),
    })
    return {"found": len(found), "on": times.source, "changed": changed}


# ── Classifying ───────────────────────────────────────────────────────────────

# A likely laugh needs a train of at least this many bursts. The three events
# after the clearest punchlines had trains of 5, 9 and 12. Of the other 95,
# one reached 4 and it was a steady voiced sound mid-sentence; the rest had 3
# or fewer. 4 keeps a margin under the weakest real laugh.
LIKELY_MIN_BURSTS = 4
# Laughter runs at roughly 4 to 6 bursts a second; the measured laughs ran at
# 3.9 to 5.3.
LAUGH_RATE = (3.0, 7.0)
# A laugh has a voice in it: a plain breath, a near-silent gap or something
# handled off camera measured 0.0 to 0.04 and is never a laugh, whatever its
# bursts. Up to the second number a laugh is breathy: the laughs of the first
# test video were voiced 0.31 to 0.44 of their sounding time, its held "uhh"
# sounds and word tails 0.75 to 1.0. A giggle is pitched further up: on the
# second video a 7 second one measured 0.69 and two short ones 0.66 and 0.73,
# all in bursts. So the voiced share alone no longer rules a laugh out; the
# bursts say it.
LAUGH_VOICED = (0.15, 0.5)
# A possible laugh (measured): after a punchline, and either at least two
# bursts in laugh spacing or, when breathy, this long. Two or three bursts
# of a pitched sound must also come at the rate of laughter. On the test
# video, breathy events without a punchline were a throat clear and
# mid-sentence breaths.
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
# A laugh belongs to a line only when it starts this soon after it, or this
# soon after a sound that does (the bouts of one long laugh). A noise half a
# minute into a wait is no laugh at the line before the wait: called one, it
# held the whole wait off the pace, and the join check told Claude to drop
# its cut of the wait. ``samples.sentence_spans`` holds the same limit for
# what a kept part and a sample take with a line.
PUNCHLINE_MAX_SECONDS_BEFORE = 1.5
# A run-on sentence before a laugh can last 16 seconds. The joke is its end:
# the punchline starts after the sentence's last pause of this length, and
# never more than PUNCHLINE_MAX_SECONDS before the sentence ends.
PUNCHLINE_PAUSE_SECONDS = 0.5
PUNCHLINE_MAX_SECONDS = 8.0


def _punchline(words: list[dict[str, Any]], index: int) -> list[float] | None:
    """``[start, end]`` of the sentence that ends just before event ``words[index]``.

    The sentence end must be the last spoken word, with the event starting
    within ``PUNCHLINE_MAX_SECONDS_BEFORE`` of it or of a sound between the
    two that does (``_follows_closely``), or at most
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
        if not between and not _follows_closely(words, i, index):
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


def _follows_closely(words: list[dict[str, Any]], said: int, index: int) -> bool:
    """Whether event ``words[index]`` follows spoken word ``words[said]`` with nothing but sounds between, each close behind the last.

    Each sound from ``said`` to ``index`` must start within
    ``PUNCHLINE_MAX_SECONDS_BEFORE`` of the end of what came before it, the
    word or the sounds so far, which is how ``samples.sentence_spans``
    decides that a sound belongs to the sentence before it.
    """
    last_end = float(words[said]["end"])
    for j in range(said + 1, index + 1):
        if round(float(words[j]["start"]) - last_end, 3) > PUNCHLINE_MAX_SECONDS_BEFORE:
            return False
        last_end = max(last_end, float(words[j]["end"]))
    return True


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
    voiced = float(measure.get("voiced", 1.0))
    if voiced < LAUGH_VOICED[0]:
        return SOUND, POSSIBLE
    breathy = voiced <= LAUGH_VOICED[1]
    at_laugh_rate = LAUGH_RATE[0] <= rate <= LAUGH_RATE[1]
    if bursts >= LIKELY_MIN_BURSTS and at_laugh_rate:
        return LAUGH, LIKELY if punchline else POSSIBLE
    if punchline and bursts >= POSSIBLE_MIN_BURSTS and (breathy or at_laugh_rate):
        return LAUGH, POSSIBLE
    if punchline and breathy and seconds >= POSSIBLE_MIN_SECONDS:
        return LAUGH, POSSIBLE
    return SOUND, POSSIBLE


def classify_events(words: list[dict[str, Any]], *, measures: list[dict] | None = None) -> list[dict]:
    """Sound labels for every event in ``words``, sorted by start.

    ``words`` is a sorted transcript. ``measures``, when given, holds one
    ``measure_samples``-shaped dict per event in the same order; an empty dict
    means that event was not measured and is judged by length and position.
    Returns ``[{start, end, seconds, kind, confidence, voiced, bursts,
    burst_rate, over_speech_db, punchline}]``. ``kind`` is ``LAUGH``, ``UM``
    (a held hesitation, see ``is_um``) or ``SOUND`` for anything else.
    ``confidence`` only matters for a laugh and is ``possible`` otherwise.
    ``voiced``, ``bursts`` and ``burst_rate`` are the measure, or None when
    the event was not measured. ``over_speech_db`` is what the finder wrote
    on the event (``OVER_SPEECH``), or None for a sound it did not find.
    ``is_filler`` reads all four. ``punchline`` is set for laughs only.

    Raises StudioError when ``measures`` does not have one entry per event.
    """
    indexes = [i for i, w in enumerate(words) if is_event(w)]
    if measures is not None and len(measures) != len(indexes):
        raise StudioError(
            f"Got {len(measures)} sound measures for {len(indexes)} events. "
            "Measure again from the same transcript."
        )
    labels = []
    spoken_ends = [float(w["end"]) for w in words if not is_event(w)]
    for n, i in enumerate(indexes):
        start, end = float(words[i]["start"]), float(words[i]["end"])
        seconds = round(end - start, 2)
        punchline = _punchline(words, i)
        measure = measures[n] if measures is not None else UNMEASURED
        kind, confidence = _judge(seconds, punchline, measure)
        label = {
            "start": start,
            "end": end,
            "seconds": seconds,
            "kind": kind,
            "confidence": confidence,
            "voiced": round(float(measure["voiced"]), 2) if measure else None,
            "bursts": int(measure["bursts"]) if measure else None,
            "burst_rate": round(float(measure["burst_rate"]), 2) if measure else None,
            OVER_SPEECH: words[i].get(OVER_SPEECH),
            "punchline": punchline if kind == LAUGH else None,
        }
        if is_um(label, spoken_ends):
            label["kind"] = UM
        labels.append(label)
    return sorted(labels, key=lambda label: label["start"])


# ── Ums ───────────────────────────────────────────────────────────────────────

# A "[vocalization]" this voiced or more is a held "umm" or "uhh", the way a
# reader hears it: a filler, not a breath. On a 21 minute take, 25 of its 87
# sounds sat near a pause cut: 8 measured voiced 0.8 and up (an unmistakable
# held vowel), 10 under 0.5 (breathy, no held pitch: an inhale or a sigh),
# and 7 between the two, still a filler by ear. The line sits at the low end
# of that middle band, so a breath never reads as one.
FILLER_VOICED_MIN = 0.5
# A sound this long or longer is never cut by a rule, laugh or not: a held
# "uhh" lasts a few tenths of a second, and what goes on for a second is a
# laugh the measures missed, a sigh, a cough, something the creator did. A
# person decides about those.
LONG_SOUND_SECONDS = 1.0
# A hesitation trails along under the voice. A sound whose loudest window
# stands more than this over the take's loud speech broke out of it: a short
# laugh, an exclamation, a cough. On the second test video the eight plain
# "mm"s and "uhh"s between sentences peaked 3.4 dB under loud speech to 1.6
# over, and every laugh, giggle and snort 2.3 to 6.6 over: two of those were
# one steady voiced burst of 0.6 s, which no other measure tells from an
# "uhh". Only a sound the finder wrote says how loud it is (``OVER_SPEECH``).
FILLER_MAX_OVER_SPEECH_DB = 2.0


def is_long(label: dict[str, Any]) -> bool:
    """Whether a sound lasts ``LONG_SOUND_SECONDS`` or more, by its own start and end."""
    if "start" not in label or "end" not in label:
        return float(label.get("seconds") or 0.0) >= LONG_SOUND_SECONDS
    return round(float(label["end"]) - float(label["start"]), 3) >= LONG_SOUND_SECONDS


def is_filler(label: dict[str, Any]) -> bool:
    """Whether a sound label has the shape of a held "um" or "uh", by its measures alone.

    All of: no laugh, likely or possible, voiced ``FILLER_VOICED_MIN`` or
    more, shorter than ``LONG_SOUND_SECONDS``, no louder than
    ``FILLER_MAX_OVER_SPEECH_DB`` over the take's loud speech, and one
    steady sound: under ``POSSIBLE_MIN_BURSTS`` bursts in laugh spacing. A
    giggle is voiced like an "uhh" and comes in bursts, and one too short
    or too far from a sentence end to be called a laugh still has two; on
    the second test video two laughs of under a second had 2 and 3.

    Measured only: an event never measured (``voiced`` is None) answers
    False, so nothing is called an um on a guess. ``is_um`` adds what the
    words around the sound say; ``classify_events`` names the sound by it.
    """
    if label.get("kind") not in (SOUND, UM) or label.get("voiced") is None or label["voiced"] < FILLER_VOICED_MIN:
        return False
    if is_long(label):
        return False
    over = label.get(OVER_SPEECH)
    if over is not None and over > FILLER_MAX_OVER_SPEECH_DB:
        return False
    bursts = label.get("bursts")
    return bursts is None or bursts < POSSIBLE_MIN_BURSTS


# A sound with no gap at all from the word right before it, and this short
# or shorter, is that word's own voice running on, not a fresh sound: on the
# real take "in" (aligned end 299.44) ran on 0.119 s into a "[vocalization]"
# logged right there, voiced 0.5, the edge of held. Longer than this a
# run-on is its own um, held or not: the measured "uhh" after "And" ran on
# 0.169 s and is one. Matches ``GAP_LENGTH_MIN``, the shortest pause any
# pace cuts and the same gap ``word_times.RUNS_ON_SECONDS`` calls no gap.
WORD_TAIL_MAX_SECONDS = GAP_LENGTH_MIN
# How close a sound's start must sit to the word before it to call it no gap
# at all. Times are saved to the millisecond; this gives a little past that.
WORD_TAIL_TOUCH_SECONDS = 0.002
# A sound shorter than this is no um: nobody hesitates for under a tenth of
# a second, and a cut that short removes nothing worth removing
# (``autocuts.SHORTEST_PIECE_SECONDS``).
SHORTEST_UM_SECONDS = 0.1


def _is_word_tail(sound: dict[str, Any], spoken_ends: list[float]) -> bool:
    """Whether ``sound`` is the trailing voice of the spoken word right before it, not a sound of its own.

    True when some word ends within ``WORD_TAIL_TOUCH_SECONDS`` of where
    ``sound`` starts, no gap at all, and ``sound`` is
    ``WORD_TAIL_MAX_SECONDS`` or shorter.
    """
    if float(sound["end"]) - float(sound["start"]) > WORD_TAIL_MAX_SECONDS:
        return False
    start = float(sound["start"])
    return any(abs(end - start) <= WORD_TAIL_TOUCH_SECONDS for end in spoken_ends)


def is_um(label: dict[str, Any], spoken_ends: list[float]) -> bool:
    """Whether a sound is an um: a held hesitation (``is_filler``) that stands on its own between the words.

    ``spoken_ends`` are the ends of the take's spoken words. Left out: a
    sound under ``SHORTEST_UM_SECONDS``, and a short run-on with no gap from
    the word before it, which is that word's own tail (``_is_word_tail``).
    A sound label already holds nothing but its audible span
    (``word_times._sound_left``), so its times are the um's own.
    """
    if not is_filler(label) or float(label["end"]) - float(label["start"]) < SHORTEST_UM_SECONDS:
        return False
    return not _is_word_tail(label, spoken_ends)


# The ways a speech model spells a hesitation when it does write one, each
# as ``plain_text`` gives it: the "um" family (lips closed or closing) and
# the "uh" family (mouth open). They are one thing to a viewer and to an
# editor, and one thing here: ``find_words`` lists every spelling, and every
# um sound, under ``UM``. "ah" and "mm" are on the list because a reader
# tells "ah, um, so" from "Ah, there it is" at a glance and no rule can.
# Not on it, because each says something: "mhm" (yes), "huh" (a question),
# "oh", "well" and "so".
UM_WORDS = frozenset({
    "um", "umm", "ummm", "uhm", "hm", "hmm", "mm", "mmm", "erm",
    "uh", "uhh", "uhhh", "er", "err", "ah", "ahh", "eh",
})


def is_um_word(entry: dict[str, Any]) -> bool:
    """Whether a spoken entry of the transcript is a hesitation the speech model wrote as a word (``UM_WORDS``)."""
    return not is_event(entry) and plain_text(str(entry.get("word", ""))) in UM_WORDS


def um_spans(labels: list[dict[str, Any]] | None) -> list[tuple[float, float]]:
    """The um sounds among ``labels``, as ``(start, end)`` sorted by start."""
    return sorted((float(lb["start"]), float(lb["end"])) for lb in labels or [] if lb.get("kind") == UM)


# ── Protecting ────────────────────────────────────────────────────────────────

# After a laugh the creator often pauses before the next line; that beat is
# part of the joke. Protect up to this much of it.
PAUSE_AFTER_LAUGH_SECONDS = 1.0
# The pause before a punchline is the beat that sets it up. Protect up to this
# much of it.
PAUSE_BEFORE_PUNCHLINE_SECONDS = 1.0


def whole_spans(labels: list[dict]) -> list[tuple[float, float]]:
    """The sounds no automatic cut may take any part of, as ``(start, end)`` sorted by start.

    Every laugh, likely or possible, and every sound of
    ``LONG_SOUND_SECONDS`` or more. Each span is the sound itself, as the
    transcript times it.
    """
    return sorted(
        (float(label["start"]), float(label["end"]))
        for label in labels if label["kind"] == LAUGH or is_long(label)
    )


def protected_spans(labels: list[dict], words: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """Source spans an automatic trim must leave alone, sorted and merged.

    For each laugh (likely or possible): from the pause before its punchline,
    capped at ``PAUSE_BEFORE_PUNCHLINE_SECONDS`` (or from the laugh itself
    when it has no punchline), to the laugh's end plus the pause after it,
    capped at ``PAUSE_AFTER_LAUGH_SECONDS``. For a sound of
    ``LONG_SOUND_SECONDS`` or more that is no laugh: the sound itself.
    """
    starts = sorted(float(w["start"]) for w in words if not is_event(w))
    ends = sorted(float(w["end"]) for w in words if not is_event(w))
    spans: list[tuple[float, float]] = []
    for label in labels:
        if label["kind"] != LAUGH:
            if is_long(label):
                spans.append((float(label["start"]), float(label["end"])))
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
# 3: a pitched laugh in bursts is a laugh, not only a breathy one.
# 4: a held hesitation is named ``UM``.
# 5: a sound is the laugh of a line only when it follows it closely (``_follows_closely``).
SOUNDS_VERSION = 5
# A cache from this version on holds the measures as they are made today: only
# the rule that reads them changed since, so they are judged again, not measured again.
MEASURES_SINCE_VERSION = 4


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

    The cache is keyed on the transcript file's size and content
    (``word_times.same_transcript``), so a re-transcribe invalidates it and
    a copy to another Mac does not. It also says which word times the labels
    were judged on, and how those were made: a punchline is found from the
    words before a laugh, so when the word times change the sounds are
    judged again, and so are they when the rule that judges them changed
    (``SOUNDS_VERSION``). The measures
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
        version = cached.get("version")
        known = isinstance(version, int) and MEASURES_SINCE_VERSION <= version <= SOUNDS_VERSION
        if known and word_times.same_transcript(cached.get("transcript"), project):
            same_times = cached.get("word_times", word_times.ESTIMATED) == times
            same_rule = version == SOUNDS_VERSION
            if same_rule and same_times and cached.get("word_times_made_as", _made_as_before(times)) == made_as:
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
