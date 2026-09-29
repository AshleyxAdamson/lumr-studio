"""Word times: the one place that chooses which word times the plugin reads.

The transcript carries raw times from the speech model. They are loose:
on the test footage 94% of neighbouring words touch, the median word is twice
as long as it sounds, and 427 s of measured silence sits inside a word's
start to end. The engine never cuts inside a word, so most of the real quiet
is off limits and the harder paces find little to cut.

Aligning pins each word to the sound (``aligner.Wav2Vec2Aligner``, a CTC
forced alignment). The text never changes, only the
times. A sound (a laugh, a breath, an "uhh") keeps the time the transcript
gave it, less any part a measured word turns out to sit in, and less any
part that never sounds at the soft level either: the transcript's span for
one runs long, and what is silent even there was not a voice
(``_sounds_clear_of_words``).

The aligner's own times are tight. It ends a word at the first measured
silence after its start, and the quiet of a closed mouth inside "take" is
such a silence, so on the test footage 29 of 3,231 words came out under 5 ms
long and 244 s of sound sat under no word at all. The engine never cuts
inside a word, but a word 1 ms long protects 1 ms. ``with_room`` gives each
word the sound that belongs to it, once, before the times are saved, so
every reader gets words that cover what was said. It reads the video's
silences twice: as the whole plugin measures them, and at the soft level
(``silences.SOFT_NOISE_DB``), where a word's fading "s" still sounds.

``read`` is the one source. Automatic cuts, the join check, Claude's cuts,
keeps, samples, clusters, the packed transcript and the words on the page all
get their words from it, so they can never disagree:

* aligned times saved with the project, when they were made from the
  transcript as it is now: ``measured``;
* else the raw transcript: ``estimated``, with a note that says why.

Aligning is slow and needs a model file of about 380 MB. Aligning never
downloads it: ``models.download`` is the one place a model arrives, after the
creator says yes. ``aligning_lacks`` says what is missing on this machine, and
without it the plugin runs on estimated times.

The aligned times live in the project folder, never in the video's folder
and never in LUMR_HOME.
"""

from __future__ import annotations

import bisect
import json
import logging
import math
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from lumr_studio import models
from lumr_studio.engine.audio_boundaries import Silence
from lumr_studio.engine.microcut_pacing import GAP_LENGTH_MIN
from lumr_studio.errors import StudioError
from lumr_studio.project import Project, now_iso, write_text_atomic
from lumr_studio.silences import measured_soft_silences
from lumr_studio.transcript import is_event, load_transcript, validate_words

log = logging.getLogger(__name__)

Words = list[dict[str, Any]]
# What measures the silences of a video.
SilencesFor = Callable[[Path], list[Silence]]

ALIGNED_FILE = "words-aligned.json"
# Bump when the saved shape or the way of aligning changes, so old files are made again.
# 2: each word has room for its sound (``with_room``).
# 3: sound the transcript heard inside a word is the word's, past a soft consonant.
# 4: a word keeps its soft start and end, measured at ``silences.SOFT_NOISE_DB``.
# 5: a word that runs into the next one ends where it starts.
# 6: a sound is narrowed to where it sounds at the soft level, and dropped
#    when it never does.
# 7: the times come from wav2vec 2.0 base 960h with a 0.5 s tail pad and starts
#    20 ms earlier, not from MMS_FA (docs/11.03).
# A file of any other version is not read as measured: the times are
# estimates until transcribe aligns again. No project keeps MMS_FA's times.
ALIGNED_VERSION = 7
# How measured times were made before ``word_times_made_as`` existed: the
# aligner's times as they came. A saved edit or sound cache that doesn't say
# is taken to be on these, and is placed again on the times as made now.
WITHOUT_ROOM_VERSION = 1

MEASURED, ESTIMATED = "measured", "estimated"
SOURCES = (MEASURED, ESTIMATED)

# What the state and the tools say when the times are estimated. Written for
# the creator: no file names, no model names.
NOTE_NOT_ALIGNED = (
    "The word times are estimates, so the harder paces find less to cut. "
    "Ask Claude to measure them."
)
# What aligning says when the soft ends of words could not be measured well.
# Written for the creator, like the notes above.
NOTE_NO_SOFT_QUIET = (
    "The room is never quiet enough here to hear where a word fades out, so a pause cut may take "
    "the soft end of a word, like the \"s\" of \"books\". A gentler pace clips fewer."
)
NOTE_SOFT_TOOK_MUCH = (
    "The room is loud here, so the soft ends of words took {share}% of the pauses. "
    "Every pace takes out less than it would in a quieter room."
)
# What a model that is missing or cut short says about getting it.
DOWNLOAD_ON_CONSENT = "Transcribe downloads it when the creator says yes."
NOTE_TRANSCRIPT_CHANGED = (
    "The transcript changed after the word times were measured, so they are estimates again. "
    "Ask Claude to measure them."
)

# The aligner's model, as torchaudio names it. The file is 377,664,473 bytes;
# anything much smaller is a broken download or another model under the same
# name, and loading it would fail or fetch.
MODEL_MIN_BYTES = 300_000_000
# A word that moved less than this kept its time, for the count in the report.
MOVED_SECONDS = 0.02
# A sound with less room than this left, once the measured words are taken
# out of it, was those words all along and is dropped.
MIN_SOUND_SECONDS = 0.1
# Each aligned entry keeps the times the transcript gave it under these keys,
# so a span chosen on the transcript's times can be found again by its words.
RAW_START, RAW_END = "raw_start", "raw_end"
# A word that was given room keeps the times the aligner gave it under these
# keys, so a span chosen on those can be found again by its words too.
ALIGNED_START, ALIGNED_END = "aligned_start", "aligned_end"

# The aligner (wav2vec 2.0) hears the sound in frames of 20 ms and gives every
# letter of a word at least one frame.
ALIGNER_FRAME_SECONDS = 0.02
# What the aligner tells letters from: ClipForge's own rule (``alignment.normalize_word``).
_NOT_A_LETTER = re.compile(r"[^a-z']")
# A quiet shorter than this inside or beside a word is part of the word: the
# closed mouth before the "t" of "take" (92 ms on the test footage), not a
# pause. It is the shortest pause any pace cuts, so giving a word this quiet
# takes nothing from the pace that it could have cut by itself.
RUNS_ON_SECONDS = GAP_LENGTH_MIN
# The longest quiet inside a word: a soft consonant beside a closed mouth,
# which the detector hears as silence. On the test footage the "s" and "t" in
# "positivity" measured 0.172 s, the "st" in "honestly" 0.194 s and the "s" in
# "person" 0.162 s. Sound after such a quiet is the word's when the transcript
# heard it inside the word. The margins are thin: the longest quiet bridged on
# that take is 0.227 s (after "these"), and the next longer one before sound
# the transcript heard in a word, 0.259 s, comes before a click, 9 ms past.
LONGEST_QUIET_IN_A_WORD = 0.25
# The most a word takes across such a quiet: the quiet and the stretch of
# sound after it together. The rest of a word is a syllable or two: on the
# test footage 0.16 to 0.35 s. A breath (0.58 s) and a laugh (1.71 s) that
# the transcript put inside a word's time come to more, and stay pauses.
LONGEST_REST_OF_A_WORD = 0.5
# How far a word's soft end reaches into the silence after it, and its soft
# start into the silence before it (``silences.SOFT_NOISE_DB``). The "s" of
# "books" sounds 0.279 s into the silence after the word on the test footage;
# a soft start is shorter. Room noise or a breath as loud as a soft "s" takes
# no more than this from a pause.
SOFT_END_SECONDS = 0.3
SOFT_START_SECONDS = 0.15
# The soft edges of words taking more than this share of a take's measured
# silence says the room is loud at the soft level. On the test footage they
# take 28%.
SOFT_SHARE_WARNED = 0.4
# Two measured silences this close are one: the detector often ends one and
# starts the next a tenth of a millisecond later.
SAME_SILENCE_SECONDS = 0.002
# A move of a word's edge smaller than this is rounding.
WORTH_MOVING_SECONDS = 0.001


# ── What aligning needs ───────────────────────────────────────────────────────


def model_path() -> Path:
    """Where torchaudio keeps the aligner's model on this machine.

    Raises ImportError when torch is not installed.
    """
    return models.ALIGNER.path_of(models.ALIGNER_FILE.name)


def aligning_lacks() -> str:
    """What this machine lacks for aligning, as a sentence, or "" when it can align.

    Looks for the model file and never fetches it.
    """
    try:
        path = model_path()
    except ImportError as why:
        return f"Measuring word times needs torch and torchaudio, and {why.name or 'one of them'} is not installed."
    if not path.exists():
        return (
            f"Measuring word times needs a model file that is not on this machine ({path}, about 380 MB). "
            f"{DOWNLOAD_ON_CONSENT}"
        )
    if path.stat().st_size < MODEL_MIN_BYTES:
        return f"The model file for measuring word times looks incomplete ({path}). {DOWNLOAD_ON_CONSENT}"
    return ""


def refuse_download(url: str, *_args: Any, **_kwargs: Any) -> None:
    """Stands in for torch's downloader while aligning, so a missing model can never be fetched from here."""
    raise StudioError(
        f"Measuring word times tried to download {url}. Only transcribe with download_models=true "
        f"fetches a model, once the creator agrees, so the word times stay estimates."
    )


class Aligner(Protocol):
    """What makes aligned times. Tests pass a fake."""

    def lacks(self) -> str:
        """What is missing for aligning, as a sentence, or "" when it can run."""

    def align(self, video: Path, words: Words, silences: list[Silence]) -> Words:
        """``words`` with each spoken word's start and end moved onto the sound."""


# ── Room for each word ────────────────────────────────────────────────────────

Span = tuple[float, float]


@dataclass(frozen=True)
class _Sound:
    """Where a take is not silent: the stretches between its measured silences."""

    quiet: list[Span]
    starts: list[float]

    @classmethod
    def between(cls, silences: list[Silence]) -> _Sound:
        quiet: list[Span] = []
        for a, b in sorted((float(s.start), float(s.end)) for s in silences):
            if quiet and a - quiet[-1][1] <= SAME_SILENCE_SECONDS:
                quiet[-1] = (quiet[-1][0], max(quiet[-1][1], b))
            elif b > a:
                quiet.append((a, b))
        return cls(quiet=quiet, starts=[a for a, _ in quiet])

    @property
    def known_until(self) -> float:
        """The end of the last measured silence. What sounds after it was never measured."""
        return self.quiet[-1][1] if self.quiet else 0.0

    def inside(self, lo: float, hi: float) -> list[Span]:
        """The stretches of sound inside ``[lo, hi]``, in order."""
        found, at = [], lo
        i = max(0, bisect.bisect_right(self.starts, lo) - 1)
        while i < len(self.quiet) and self.quiet[i][0] < hi:
            a, b = self.quiet[i]
            if b > at:
                if a > at:
                    found.append((at, min(a, hi)))
                at = b
            i += 1
        if at < hi:
            found.append((at, hi))
        return [(a, b) for a, b in found if b - a >= WORTH_MOVING_SECONDS]

    def quiet_at(self, t: float) -> Span | None:
        """The measured silence ``t`` sits in, its edges included to the millisecond, or None."""
        i = bisect.bisect_right(self.starts, t + WORTH_MOVING_SECONDS) - 1
        if i >= 0 and t <= self.quiet[i][1] + WORTH_MOVING_SECONDS:
            return self.quiet[i]
        return None


def letters_of(word: dict[str, Any]) -> int:
    """How many letters the aligner placed for ``word``, at least one."""
    return max(1, len(_NOT_A_LETTER.sub("", str(word.get("word", "")).lower())))


def cut_short(word: dict[str, Any]) -> bool:
    """Whether the aligner could not place ``word``: it is shorter than the aligner can make a word.

    The threshold is one frame of the aligner for each letter, 20 ms a
    letter: "take" under 80 ms. The aligner gives every letter a frame of
    its own, so a shorter word did not come from it. ClipForge ended the
    word at a measured silence that began inside it, and where the word
    really ends is no longer known. A flat threshold of one frame finds 121
    such words on the test footage and misses "experiencing" at 116 ms,
    whose other half a second went as a pause. This one finds 429 of 3,231.
    """
    length = float(word["end"]) - float(word["start"])
    return length < ALIGNER_FRAME_SECONDS * letters_of(word) - WORTH_MOVING_SECONDS / 2


def _sides(lo: float, hi: float, sounds: list[Span]) -> tuple[Span | None, Span | None]:
    """The parts of the gap ``[lo, hi]`` the word before it and the word after it may grow into.

    A sound of the transcript (a laugh, a breath) keeps its place: the word
    before grows up to the first one, the word after back to the last one.
    None for a word that has a sound right against it.
    """
    inside = [(a, b) for a, b in sounds if a < hi and b > lo]
    if not inside:
        return (lo, hi), (lo, hi)
    first, last = min(a for a, _ in inside), max(b for _, b in inside)
    return ((lo, first) if first > lo else None), ((last, hi) if last < hi else None)


def _runs_on(quiet: float, stretch: Span, heard_in_the_word: bool) -> bool:
    """Whether ``stretch``, sound after a quiet of this length, to the millisecond, is part of the word beside it.

    ``heard_in_the_word`` says the transcript heard the whole stretch inside the word.
    """
    quiet = round(quiet, 3)
    if quiet < RUNS_ON_SECONDS:
        return True
    rest = quiet + stretch[1] - stretch[0]
    return heard_in_the_word and quiet < LONGEST_QUIET_IN_A_WORD and round(rest, 3) <= LONGEST_REST_OF_A_WORD


def _shared_out(
    sound: _Sound, end: float, start: float, before: Span | None, after: Span | None, heard: Span,
) -> Span:
    """Where the word before a gap ends and the word after it starts, once each has the sound that runs on from it.

    Sound that starts less than ``RUNS_ON_SECONDS`` after a word ends is
    that word's, and so is the sound that follows it as closely. So is a
    stretch of sound after a quiet shorter than ``LONGEST_QUIET_IN_A_WORD``
    that the transcript heard inside the word, all of it before
    ``heard[0]``, the end it gave the word before the gap, when the quiet
    and the stretch come to no more than ``LONGEST_REST_OF_A_WORD``. The
    same holds before a word, with ``heard[1]``, the start it gave the word
    after.

    The two words grow in step, a stretch each at a time, and a stretch
    both reach in the same step goes to the nearer one, to the word before
    when they are as near. In step, a word keeps the rest of itself after a
    closed mouth. Grown one at a time, nearest first, the word after would
    first take its own lead-in and then, nearer now, the end of the word
    before: on the test footage 12 words lost their ends that way, the "st"
    of "most" to "a" and the "-ple" of "purple." to "I" among them.
    """
    shared = before is not None and before == after
    heard_until, heard_from = heard
    while True:
        ahead = sound.inside(end, start if shared else before[1]) if before else []
        behind = sound.inside(end if shared else after[0], start) if after else []
        quiet_ahead = ahead[0][0] - end if ahead else 0.0
        quiet_behind = start - behind[-1][1] if behind else 0.0
        reach_ahead = bool(ahead) and _runs_on(quiet_ahead, ahead[0], ahead[0][1] <= heard_until)
        reach_behind = bool(behind) and _runs_on(quiet_behind, behind[-1], behind[-1][0] >= heard_from)
        if reach_ahead and reach_behind and shared and ahead[0] == behind[-1]:
            if quiet_ahead <= quiet_behind:
                reach_behind = False
            else:
                reach_ahead = False
        if not (reach_ahead or reach_behind):
            return end, start
        if reach_ahead:
            end = ahead[0][1]
        if reach_behind:
            start = behind[-1][0]


@dataclass(frozen=True)
class _Beside:
    """The sounds of the transcript beside a word, as its soft edges meet them."""

    # Where those after it start and those before it end: its soft edges stop there.
    walls_after: list[float]
    walls_before: list[float]
    # Whether one was shortened to where the word ends, or starts.
    cut_to_its_end: bool
    cut_to_its_start: bool

    @classmethod
    def of(cls, events: Words, word: dict[str, Any]) -> _Beside:
        """What is beside ``word``.

        A sound the transcript heard begin inside the word, shortened to
        where the word ends (``_sounds_clear_of_words``), is no wall to its
        soft end: "in" sounds 0.14 s past the 887.644 a "[vocalization]" was
        shortened to. The same holds at its start.
        """
        start, end = float(word["start"]), float(word["end"])
        after, before, to_end, to_start = [], [], False, False
        for e in events:
            a, b = float(e["start"]), float(e["end"])
            heard_from, heard_to = float(e.get(RAW_START, a)), float(e.get(RAW_END, b))
            if a >= end - WORTH_MOVING_SECONDS:
                if start <= heard_from < end:
                    to_end = to_end or a - end < WORTH_MOVING_SECONDS
                else:
                    after.append(a)
            if b <= start + WORTH_MOVING_SECONDS:
                if start < heard_to <= end:
                    to_start = to_start or start - b < WORTH_MOVING_SECONDS
                else:
                    before.append(b)
        return cls(walls_after=after, walls_before=before, cut_to_its_end=to_end, cut_to_its_start=to_start)


def _soft_edges(spoken: Words, given: list[Span], events: Words, sound: _Sound, soft: _Sound) -> tuple[int, float]:
    """Give each of ``spoken``, in place, the soft sound at its end and at its start. Answers how many grew, and by how much.

    ``given`` are the times the words came with, ``sound`` where the take
    sounds by the measured silences and ``soft`` where it sounds at
    ``silences.SOFT_NOISE_DB``. A word that ends in a measured silence
    keeps the soft sound that runs on from its end, up to
    ``SOFT_END_SECONDS`` past where it fell silent: where that silence
    starts, or where the aligner ended it if later. A word that starts in
    one keeps the soft sound that leads into it, up to
    ``SOFT_START_SECONDS`` before where it began to sound: where that
    silence ends, or where the aligner heard it begin if earlier (it began
    "sending" on its soft "s", 0.146 s inside the silence). Neither
    reaches past the word beside it or into a sound of the transcript
    (``_Beside``); a word that a sound of the transcript was cut back to
    reaches into it even where it still sounds. The word before a gap takes
    its share first.
    """
    was = [(float(w["start"]), float(w["end"])) for w in spoken]
    for k, word in enumerate(spoken):
        end = float(word["end"])
        quiet, beside = sound.quiet_at(end), _Beside.of(events, word)
        if quiet is not None:
            silent_from, quiet_to = max(quiet[0], float(word.get(ALIGNED_END, given[k][1]))), quiet[1]
        elif beside.cut_to_its_end:
            silent_from, quiet_to = end, math.inf  # it ends in its own sound, where a sound of the transcript was cut
        else:
            continue
        hi = min(silent_from + SOFT_END_SECONDS, quiet_to, *beside.walls_after)
        if k + 1 < len(spoken):
            hi = min(hi, float(spoken[k + 1]["start"]))
        tail = soft.inside(end, hi) if hi > end else []
        if tail and tail[0][0] - end < WORTH_MOVING_SECONDS:
            word["end"] = round(tail[0][1], 3)
    for k, word in enumerate(spoken):
        start = float(word["start"])
        quiet, beside = sound.quiet_at(start), _Beside.of(events, word)
        if quiet is not None:
            sounding_from, quiet_from = min(quiet[1], float(word.get(ALIGNED_START, given[k][0]))), quiet[0]
        elif beside.cut_to_its_start:
            sounding_from, quiet_from = start, -math.inf
        else:
            continue
        lo = max(sounding_from - SOFT_START_SECONDS, quiet_from, *beside.walls_before)
        if k > 0:
            lo = max(lo, float(spoken[k - 1]["end"]))
        lead = soft.inside(lo, start) if start > lo else []
        if lead and start - lead[-1][1] < WORTH_MOVING_SECONDS:
            word["start"] = round(lead[-1][0], 3)
    grew = [(a - float(w["start"])) + (float(w["end"]) - b) for w, (a, b) in zip(spoken, was)]
    return sum(1 for g in grew if g > 0), round(sum(grew), 3)


def _apart(spoken: Words) -> None:
    """End each of ``spoken``, in place, where the next one starts when it runs into it.

    ClipForge aligns a take in windows of 40 s, and at a seam a word can
    run over the first words of the next window: on the test footage all
    five words that overlap the next one sit at a seam. The later word is
    right. "that" ran to 640.433 over "you" at 640.24, and "you" sounds
    from 640.21; "me" ran over all of "know" at 360.08. A double-click on
    "you" then left most of it playing, and one on "know" took "me" with
    it. A word that starts as the next one does is left as it is.
    """
    for word, after in zip(spoken, spoken[1:]):
        start = float(after["start"])
        if start - float(word["start"]) >= WORTH_MOVING_SECONDS and float(word["end"]) > start:
            word["end"] = start


def with_room(words: Words, silences: list[Silence], soft: list[Silence] | None = None) -> Words:
    """``words`` with each spoken word given the sound that belongs to it. The one place a word gets its room.

    Two rules, both read from the measured silences, and a third read from
    ``soft``, the silences measured at ``silences.SOFT_NOISE_DB``. What is
    not silent is sound. First, a word that runs into the next one ends
    where the next one starts (``_apart``).

    * Sound runs on. A word keeps the sound that follows its end and the
      sound that leads into its start, through any quiet shorter than
      ``RUNS_ON_SECONDS``. On the test footage 7 in 10 of the stretches of
      sound beside a word carry a pitch through half their length or more,
      and 1 in 8 carries none: most of it is voice, the rest of a word the
      aligner gave 0.12 s and the transcript 0.24. Where the transcript
      heard the sound inside the word, it runs on through a quiet shorter
      than ``LONGEST_QUIET_IN_A_WORD``, as far as ``LONGEST_REST_OF_A_WORD``:
      the soft "s" and closed mouth in "positivity" read as 0.172 s of
      silence, and without this the pace took "-tivity" as a pause.
    * A word the aligner could not place (``cut_short``) keeps the room the
      transcript gave it: the sound from the start the transcript or the
      aligner gave it, whichever is earlier, to the end the transcript gave
      it. Its edges stop at the sound, so measured silence at either end of
      the room stays outside the word. With no sound in its room it is a
      word said quietly, and gets back the frame a letter the aligner gave
      it before ClipForge cut it short.
    * A word keeps its soft edges (``_soft_edges``). The measured silences
      are measured at -25 dB, where the "s" of "books" is silent already:
      on the test footage pause cuts at Standard took 0.1 s or more of the
      soft sound running on from a word at 45 places, 0.28 s of that "s".
      Soft sound that runs on from a word's end into the silence after it
      is the word's, and so is soft sound that leads into its start.

    No rule takes a word past its neighbours' edges or into a sound of the
    transcript (a laugh stays whole), and no two words come to overlap that
    did not before. Without measured silences nothing is known about the
    sound and the words are returned as they are; without ``soft`` no word
    gets its soft edges.

    A word that grew keeps the aligner's times under ``ALIGNED_START`` and
    ``ALIGNED_END``, and room is always given from the aligner's times, so
    giving it again changes nothing. ``with_soft_room`` also says what the
    soft edges took.
    """
    return with_soft_room(words, silences, soft)[0]


@dataclass(frozen=True)
class SoftRoom:
    """What the soft edges of ``with_room`` gave a take's words, beside how much of it is measured silence.

    ``measured`` is False when no soft level was measured at all; then
    nothing is said.
    """

    measured: bool = False
    found_quiet: bool = False
    words: int = 0
    seconds: float = 0.0
    quiet_seconds: float = 0.0

    @property
    def share(self) -> float:
        """The share of the measured silence the soft edges took."""
        return self.seconds / self.quiet_seconds if self.quiet_seconds else 0.0

    def warning(self) -> str:
        """What to tell the creator when the soft level misread the room, or ""."""
        if not self.measured:
            return ""
        if not self.found_quiet:
            return NOTE_NO_SOFT_QUIET
        if self.share > SOFT_SHARE_WARNED:
            return NOTE_SOFT_TOOK_MUCH.format(share=round(self.share * 100))
        return ""

    def as_dict(self) -> dict[str, Any]:
        """``{words_given_soft_room, soft_room_seconds, soft_room_share}``, and ``warning`` when there is one."""
        out: dict[str, Any] = {
            "words_given_soft_room": self.words, "soft_room_seconds": round(self.seconds, 1),
            "soft_room_share": round(self.share, 2),
        }
        if self.warning():
            out["warning"] = self.warning()
        return out


def with_soft_room(
    words: Words, silences: list[Silence], soft: list[Silence] | None = None,
) -> tuple[Words, SoftRoom]:
    """``with_room``, and what its soft edges took (``SoftRoom``). ``soft`` None means the soft level was not measured.

    A sound of the transcript that the grown words leave too little of is
    dropped (``_sounds_clear_of_words``), and room is given again without
    it, so it walls off no word. Given room a second time, the words then
    meet the same sounds, and nothing changes.
    """
    base = _as_the_aligner_gave(words)
    sound = _Sound.between(silences)
    if not sound.quiet:
        return [dict(w) for w in words], SoftRoom()  # nothing is known about the sound, so nothing is said
    gone: set[int] = set()
    while True:
        kept = [i for i in range(len(base)) if i not in gone]
        out = [dict(base[i]) for i in kept]
        report = _room(out, sound, soft)
        spoken = Spoken.of(out)
        dropped = {i for i, w in zip(kept, out) if is_event(w) and _sound_left(w, spoken) is None}
        if not dropped:
            return sorted(out, key=lambda w: float(w["start"])), report
        gone |= dropped


def _room(out: Words, sound: _Sound, soft: list[Silence] | None) -> SoftRoom:
    """The room of ``with_room`` given to the words of ``out``, in place, sorted by start and at the aligner's times."""
    spoken = [w for w in out if not is_event(w)]
    events = [w for w in out if is_event(w)]
    sounds = sorted((float(w["start"]), float(w["end"])) for w in events)
    aligned = [(float(w["start"]), float(w["end"])) for w in spoken]
    _apart(spoken)
    unplaced = [cut_short(w) for w in spoken]
    given = [(float(w["start"]), float(w["end"])) for w in spoken]
    for k, word in enumerate(spoken):
        after = spoken[k + 1] if k + 1 < len(spoken) else None
        lo = given[k][1]
        hi = given[k + 1][0] if after is not None else max(lo, sound.known_until)
        if hi - lo < WORTH_MOVING_SECONDS:
            continue
        mine, theirs = _sides(lo, hi, sounds)
        if after is None:
            theirs = None
        end, start = lo, hi
        if unplaced[k] and mine:
            room = sound.inside(lo, min(mine[1], max(lo, float(word.get(RAW_END, lo)))))
            least = min(mine[1], given[k][0] + ALIGNER_FRAME_SECONDS * letters_of(word))
            end = max(lo, least, room[-1][1] if room else lo)
        if after is not None and unplaced[k + 1] and theirs:
            room = sound.inside(max(theirs[0], end, min(hi, float(after.get(RAW_START, hi)))), hi)
            if room:
                start = room[0][0]
        heard = (float(word.get(RAW_END, lo)), float(after.get(RAW_START, hi)) if after is not None else hi)
        end, start = _shared_out(sound, end, start, mine, theirs, heard)
        if end - lo >= WORTH_MOVING_SECONDS:
            word["end"] = round(end, 3)
        if after is not None and hi - start >= WORTH_MOVING_SECONDS:
            after["start"] = round(max(start, float(word["end"])), 3)
    quiet_below_soft = _Sound.between(soft or [])
    grew, seconds = 0, 0.0
    if quiet_below_soft.quiet:
        grew, seconds = _soft_edges(spoken, given, events, sound, quiet_below_soft)
    for word, (start, end) in zip(spoken, aligned):
        if (float(word["start"]), float(word["end"])) != (start, end):
            word[ALIGNED_START], word[ALIGNED_END] = start, end
    return SoftRoom(
        measured=soft is not None, found_quiet=bool(quiet_below_soft.quiet), words=grew, seconds=seconds,
        quiet_seconds=sum(b - a for a, b in sound.quiet),
    )


def given_room(words: Words) -> int:
    """How many of ``words`` were given room."""
    return sum(1 for w in words if ALIGNED_START in w)


# ── The saved aligned times ───────────────────────────────────────────────────


def aligned_path(project: Project) -> Path:
    return project.root / ALIGNED_FILE


def transcript_key(project: Project) -> dict[str, int] | None:
    """Size and mtime of the project's transcript, or None when it is missing.

    Everything made from the transcript (aligned times, sound labels) is
    saved with this, so a new transcript makes it stale.
    """
    try:
        stat = project.words_path.stat()
    except FileNotFoundError:
        return None
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


# The aligned file last read, by path, with the size and mtime it had. Every
# page request reads the words; parsing 3000 of them each time is wasted work.
_parsed: dict[Path, tuple[tuple[int, int], dict[str, Any]]] = {}
_parsed_lock = threading.Lock()
# A few videos in one session.
PARSED_FILES_KEPT = 4


def _parse_aligned(path: Path) -> dict[str, Any]:
    """The aligned file's content, parsed once for each state of the file. Raises as ``json`` and ``open`` do."""
    stat = path.stat()
    state = (stat.st_size, stat.st_mtime_ns)
    with _parsed_lock:
        hit = _parsed.get(path)
        if hit is not None and hit[0] == state:
            return hit[1]
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("words"), list):
        data["words"] = sorted(validate_words(data["words"]), key=lambda w: float(w["start"]))
    with _parsed_lock:
        _parsed[path] = (state, data)
        while len(_parsed) > PARSED_FILES_KEPT:
            _parsed.pop(next(iter(_parsed)))
    return data


# What can go wrong while the saved aligned file is read: it is gone, it isn't JSON,
# its words fail ``validate_words`` (a StudioError), or it isn't the shape ``_usable`` expects.
_CANT_READ = (OSError, ValueError, StudioError, KeyError, AttributeError, TypeError)


def _as_the_aligner_gave(words: Words) -> Words:
    """``words`` as they were before room was given: the words at the aligner's times, the sounds cleared of those.

    A sound is cleared of the words it overlaps (``_sounds_clear_of_words``),
    and those words then grow. Cleared again of the aligner's words, the
    sound is where it was before they grew, so a word beside it gets the
    same room the second time as the first.
    """
    out = []
    for w in words:
        if ALIGNED_START in w:
            w = {k: v for k, v in w.items() if k not in (ALIGNED_START, ALIGNED_END)} | {
                "start": w[ALIGNED_START], "end": w[ALIGNED_END],
            }
        elif is_event(w) and RAW_START in w and RAW_END in w:
            w = {**w, "start": w[RAW_START], "end": w[RAW_END]}
        out.append(w)
    return _sounds_clear_of_words(sorted(out, key=lambda w: float(w["start"])))


def _usable(project: Project) -> tuple[dict[str, Any] | None, str]:
    """The saved aligned file's content when it fits the transcript, else None and why not.

    A file of another version is set aside, as one that can't be read is.
    """
    path = aligned_path(project)
    if not path.exists():
        return None, NOTE_NOT_ALIGNED
    try:
        data = _parse_aligned(path)
        if data.get("version") != ALIGNED_VERSION or not isinstance(data.get("words"), list):
            return None, NOTE_NOT_ALIGNED
        if data.get("transcript") != transcript_key(project):
            return None, NOTE_TRANSCRIPT_CHANGED
    except _CANT_READ as why:
        log.warning("the saved word times at %s can't be read, using the transcript's own: %s", path, why)
        return None, NOTE_NOT_ALIGNED
    return data, ""


def _saved_aligned(project: Project) -> tuple[Words | None, str]:
    """The saved aligned words when they fit the transcript, else None and why not.

    The words are copies, so a caller may change them.
    """
    data, note = _usable(project)
    if data is None:
        return None, note
    return [dict(w) for w in data["words"]], ""


def _fits(project: Project) -> bool:
    """Whether saved aligned words fit the transcript, without copying them."""
    return _usable(project)[0] is not None


# ── The one source ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WordTimes:
    """The words everything reads, and where their times came from."""

    words: Words
    source: str  # MEASURED or ESTIMATED
    note: str    # why the times are estimated, for the creator; "" when measured


def read(project: Project) -> WordTimes:
    """The project's words with the best times there are. Raises StudioError when there is no transcript."""
    raw = load_transcript(project.words_path)
    aligned, note = _saved_aligned(project)
    if aligned is None:
        return WordTimes(words=raw, source=ESTIMATED, note=note)
    return WordTimes(words=aligned, source=MEASURED, note="")


def load_words(project: Project) -> Words:
    """``read(project).words``, for callers that only need the words."""
    return read(project).words


def words_if_any(project: Project) -> Words:
    """The project's words, or none when nothing was transcribed yet."""
    return load_words(project) if project.words_path.exists() else []


def source_of(project: Project) -> str:
    """``measured`` or ``estimated``, without loading the words."""
    return MEASURED if _fits(project) else ESTIMATED


def made_as(source: str) -> int | None:
    """How the word times of ``source`` were made, for a saved edit to say what it was placed on.

    ``ALIGNED_VERSION`` for measured times, None for estimated ones. An edit
    placed on measured times made another way holds its cuts on other word
    edges, and is placed again.
    """
    return ALIGNED_VERSION if source == MEASURED else None


# ── Aligning ──────────────────────────────────────────────────────────────────


def _moved(raw: Words, aligned: Words) -> int:
    return sum(
        1 for a, b in zip(raw, aligned)
        if not is_event(a) and (abs(float(a["start"]) - float(b["start"])) >= MOVED_SECONDS
                                or abs(float(a["end"]) - float(b["end"])) >= MOVED_SECONDS)
    )


def _same_text(raw: Words, aligned: Words) -> bool:
    return len(raw) == len(aligned) and all(
        a.get("word") == b.get("word") and a.get("type") == b.get("type") for a, b in zip(raw, aligned)
    )


def _with_sounds_as_they_were(raw: Words, aligned: Words) -> Words:
    """``aligned`` with every sound back at the times the transcript gave it.

    ClipForge ends each entry at the next measured silence, sounds included.
    A laugh is bursts with quiet between them, so that ends it after its
    first burst (3.5 s became 0.4 s on the test footage) and the laugh is no
    longer found. Only spoken words are re-timed here.
    """
    if len(raw) != len(aligned):
        return aligned
    return [
        {**b, "start": a["start"], "end": a["end"]} if is_event(a) else b
        for a, b in zip(raw, aligned)
    ]


def _sounds_clear_of_words(words: Words, soft: list[Silence] | None = None) -> Words:
    """``words`` with each sound shortened to where no measured word sits, and dropped when no room is left.

    The transcript marks a sound wherever it heard a voice between two words.
    Its word times are loose, so on the test footage 52 of 98 sounds overlap
    a word once the words are measured, and 32 hold a word's middle: what
    was heard there was the word. A sound keeps the longest stretch of its
    span that is free of words. A laugh after a punchline has no words in
    it and stays whole.

    With ``soft``, the silences measured at ``silences.SOFT_NOISE_DB``, a
    sound is also narrowed to where it sounds at that level, first to last
    (``_sound_left``): a "[vocalization]" runs long past where it stops
    sounding, and the rest was never a voice, whatever the transcript's span
    for it. Left out, a sound keeps the span it had.
    """
    spoken = Spoken.of(words)
    soft_sound = _Sound.between(soft) if soft is not None else None
    out = []
    for w in words:
        left = _sound_left(w, spoken, soft_sound) if is_event(w) else w
        if left is not None:
            out.append(left)
    return out


def _sound_left(
    sound: dict[str, Any], spoken: Spoken, soft_sound: _Sound | None = None,
) -> dict[str, Any] | None:
    """``sound`` shortened to the longest stretch of it no word sits in, then to where it sounds, or None when nothing is left.

    ``soft_sound``, when given and measured, narrows what is left again to
    its first and last sounding instant at the soft level: a sound with
    nothing above that level, end to end, was silence the transcript heard
    as a voice, and is dropped the same way a sound with no room free of
    words is.

    Known, not fixed: where a word's own voice runs straight into a sound
    with no quiet between them, the word's soft end (``_soft_edges``) has
    already claimed its share of that run-on, so the sound's first sounding
    instant here is where the word's claim stopped, not where the sound
    itself began. On the real take this leaves 50 to 100 ms of an "uhh"'s
    onset with the word before it ("And" into one, "while" into another).
    ``autocuts._is_word_tail`` catches the extreme of this, a short run-on
    with no gap at all, and never counts it as a filler to begin with.
    """
    start, end = float(sound["start"]), float(sound["end"])
    piece = spoken.longest_free(start, end)
    if piece is None:
        return None
    if soft_sound is not None and soft_sound.quiet:
        heard = soft_sound.inside(*piece)
        if not heard:
            return None
        piece = (heard[0][0], heard[-1][1])
    if piece == (start, end):
        return sound
    if piece[1] - piece[0] >= MIN_SOUND_SECONDS:
        return {**sound, "start": round(piece[0], 3), "end": round(piece[1], 3)}
    return None


def _with_raw_times(raw: Words, aligned: Words) -> Words:
    """``aligned`` with each entry's transcript times beside its measured ones."""
    return [{**b, RAW_START: a["start"], RAW_END: a["end"]} for a, b in zip(raw, aligned)]


def _held_before(start: float, end: float, words: Words, was_start: str, was_end: str) -> Words:
    """The words whose middle lay inside ``[start, end]`` on the times saved under the two keys, ends included."""
    held = []
    for w in words:
        a, b = w.get(was_start, w["start"]), w.get(was_end, w["end"])
        if start <= (float(a) + float(b)) / 2 <= end:
            held.append(w)
    return held


# An edge this close to a word's edge is on it: times are saved to the millisecond.
ON_A_WORD_EDGE = 0.001


@dataclass(frozen=True)
class Spoken:
    """A take's spoken words as spans, sorted once, so the word an instant sits in is found by binary search."""

    spans: list[Span]
    starts: list[float]
    longest: float

    @classmethod
    def of(cls, words: Words, sounds: bool = False) -> Spoken:
        """The spans of the spoken words of ``words``, and with ``sounds`` of the transcript's sounds too."""
        spans = sorted((float(w["start"]), float(w["end"])) for w in words if sounds or not is_event(w))
        return cls(spans=spans, starts=[a for a, _ in spans], longest=max((b - a for a, b in spans), default=0.0))

    def around(self, t: float) -> Span | None:
        """The word ``t`` sits inside, more than ``ON_A_WORD_EDGE`` from both its edges, or None."""
        first = bisect.bisect_left(self.starts, t - self.longest)
        last = bisect.bisect_left(self.starts, t)
        return next(((a, b) for a, b in self.spans[first:last] if a + ON_A_WORD_EDGE < t < b - ON_A_WORD_EDGE), None)

    def longest_free(self, start: float, end: float) -> Span | None:
        """The longest stretch of ``[start, end]`` that no word sits in, or None when words fill it."""
        free, at = [], start
        for a, b in self.spans[bisect.bisect_left(self.starts, start - self.longest):bisect.bisect_left(self.starts, end)]:
            if b <= at:
                continue
            if a > at:
                free.append((at, a))
            at = max(at, b)
        if at < end:
            free.append((at, end))
        return max(free, key=lambda piece: piece[1] - piece[0]) if free else None

    def clear(self, start: float, end: float, keep: Span | None = None) -> Span:
        """``[start, end]`` with each edge that sits inside a word moved to the word's far side. The one way a span is kept off words.

        The span then takes no part of a word. ``keep`` is a part of the span
        that must stay in it: no edge moves past it, so an edge of that part
        never moves. Without one the span may come to nothing (its end at or
        before its start).
        """
        first, last = keep if keep is not None else (math.inf, -math.inf)
        while start < min(end, first) and (word := self.around(start)) is not None:
            start = min(word[1], first)
        while end > max(start, last) and (word := self.around(end)) is not None:
            end = max(word[0], last)
        return start, end


def _clear_of_words(start: float, end: float, words: Words) -> tuple[float, float]:
    """A span that holds no word, shortened so that it takes no part of a word's room either.

    A cut placed in a pause stays a cut of that pause: a word whose room now
    reaches into the span is left whole (``Spoken.clear``). A span with
    nothing left is returned as it was.
    """
    a, b = Spoken.of(words).clear(start, end)
    return (round(a, 3), round(b, 3)) if b > a else (start, end)


def on_measured_times(start: float, end: float, words: Words, placed_on: str = ESTIMATED) -> tuple[float, float]:
    """A span chosen on other times than the measured ones, moved onto the measured times of the same words.

    ``placed_on`` says what the span was chosen on: ``ESTIMATED`` for the
    transcript's times, ``MEASURED`` for measured times made before words
    had the room they have now. The words inside are found the way every
    cut finds them, by their middle: on the transcript's times, or on the
    aligner's. A word's room only ever grew from the aligner's times, so its
    middle there lies inside any span that held it. The span then runs from
    the first one's measured start to the last one's measured end, so it
    holds the same words as before. A span that held no word holds none
    after (``_clear_of_words``).
    """
    if placed_on == ESTIMATED:
        inside = [
            w for w in words
            if RAW_START in w and RAW_END in w and start < (float(w[RAW_START]) + float(w[RAW_END])) / 2 < end
        ]
    else:
        inside = _held_before(start, end, words, ALIGNED_START, ALIGNED_END)
    if not inside:
        return _clear_of_words(start, end, words)
    return (round(min(float(w["start"]) for w in inside), 3), round(max(float(w["end"]) for w in inside), 3))


def align_project(
    project: Project, *, aligner: Aligner, silences: SilencesFor, soft: SilencesFor | None = None,
) -> dict[str, Any]:
    """Make aligned times for the project's transcript and save them with the project.

    ``soft`` measures the video's silences at ``silences.SOFT_NOISE_DB``;
    left out, they are measured the way the plugin measures them. Answers
    ``{word_times, seconds, words_moved, saved_to, sounds,
    sounds_that_were_words, words_given_room, words_given_soft_room,
    soft_room_seconds, soft_room_share}``, and ``warning`` when the soft
    level misread the room (``SoftRoom.warning``). Raises
    StudioError when the aligner can't run or gives back other words than it
    was given; nothing is saved then and the plugin stays on estimated times.
    """
    lacks = aligner.lacks()
    if lacks:
        raise StudioError(lacks)
    raw = load_transcript(project.words_path)
    key = transcript_key(project)
    quiet = silences(project.video)
    started = time.monotonic()
    aligned = _with_sounds_as_they_were(raw, validate_words(aligner.align(project.video, raw, quiet)))
    seconds = time.monotonic() - started
    if not _same_text(raw, aligned):
        raise StudioError(
            "Measuring word times gave back different words than the transcript holds, so the result "
            "was not saved. The word times stay estimates."
        )
    under_soft = (soft or measured_soft_silences)(project.video)
    roomy, soft_room = with_soft_room(_sounds_clear_of_words(_with_raw_times(raw, aligned)), quiet, under_soft)
    saved = _sounds_clear_of_words(roomy, under_soft)
    write_text_atomic(aligned_path(project), json.dumps({
        "version": ALIGNED_VERSION, "made_at": now_iso(), "transcript": key,
        "seconds": round(seconds, 1), "words": saved, "soft_room": soft_room.as_dict(),
    }, ensure_ascii=False) + "\n")
    sounds = sum(1 for w in raw if is_event(w))
    return {
        "word_times": MEASURED, "seconds": round(seconds, 1),
        "words_moved": _moved(raw, aligned), "saved_to": str(aligned_path(project)),
        "sounds": sounds, "sounds_that_were_words": sounds - sum(1 for w in saved if is_event(w)),
        "words_given_room": given_room(saved), **soft_room.as_dict(),
    }
