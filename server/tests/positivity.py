"""Twelve made-up words where a pause cut took most of a word.

"...itself recipes, warm or sunshine, positivity at certain times of the
season," from 18 to 27 s. The words are invented. The times are measured, on
a 21 minute take, and they are the data. The aligner ended "positivity" at
23.521 s, where the soft "s" and the closed mouth before its "t" read as
0.172 s of silence. "-tivity" sounds from 23.692 to 24.202 s, under no word,
and the saved edit cut 23.521 to 24.89 as one pause. The transcript gave the
word 23.28 to 24.88.

Each row is ``(word, transcript start, transcript end, aligned start,
aligned end, saved start, saved end)``: the saved times are the ones the
plugin wrote with room for each word (``word_times`` version 2). "itself"
is the first word here, so nothing gives it room before its start. The
silences are the ones measured, to a tenth of a millisecond, at -25 dB
(``QUIET``) and at the soft level, -40 dB (``SOFT_QUIET``). The take is cut
off in the pause after "season,".
"""

from __future__ import annotations

import json
from pathlib import Path

from lumr_studio.engine.audio_boundaries import Silence

from lumr_studio import word_times
from lumr_studio.project import Project, open_project

DURATION = 28.0
ROWS = [
    ("itself", 18.56, 19.44, 18.369, 19.032, 18.369, 19.032),
    ("recipes,", 20.08, 21.36, 20.21, 20.284, 20.183, 20.752),
    ("warm", 21.36, 21.92, 21.49, 21.745, 21.466, 21.745),
    ("or", 21.92, 22.16, 21.97, 22.016, 21.83, 22.016),
    ("sunshine,", 22.16, 23.28, 22.131, 22.866, 22.126, 22.866),
    ("positivity", 23.28, 24.88, 23.251, 23.521, 23.251, 23.521),
    ("at", 24.88, 25.2, 25.092, 25.172, 25.07, 25.177),
    ("certain", 25.2, 25.44, 25.232, 25.309, 25.231, 25.465),
    ("times", 25.44, 25.76, 25.512, 25.694, 25.512, 25.694),
    ("of", 25.76, 26.0, 25.772, 25.832, 25.756, 25.852),
    ("the", 26.0, 26.32, 25.852, 25.934, 25.852, 25.934),
    ("season,", 26.32, 27.68, 25.992, 26.199, 25.992, 26.659),
]
QUIET = [
    (18.2228, 18.3296), (19.0323, 20.1835), (20.2842, 20.3845), (20.6791, 20.7351), (20.7524, 21.4661),
    (21.7451, 21.8297), (22.0161, 22.126), (22.8659, 22.9488), (22.9488, 23.2732), (23.5209, 23.6925),
    (23.7327, 23.8123), (24.2019, 25.0704), (25.1771, 25.2308), (25.3089, 25.3733), (25.4648, 25.5688),
    (25.6938, 25.7557), (25.9337, 26.0045), (26.1991, 26.257), (26.6591, DURATION),
]
SOFT_QUIET = [
    (19.1964, 20.1834), (21.0014, 21.4549), (22.9885, 23.2557), (23.6376, 23.6924), (24.2563, 24.5037),
    (24.5038, 24.6087), (24.6087, 25.0693), (26.7119, 27.0807), (27.0815, DURATION),
]
# "positivity" as the aligner gave it, with the sound of "-tivity" in it, and with its soft end too.
AS_ALIGNED = (23.251, 23.521)
WHOLE = (23.251, 24.202)
WITH_ITS_SOFT_END = (23.251, 24.256)
# The pause after it, before "at".
PAUSE_AFTER = (24.202, 25.07)
# What the saved edit cut there, at the Standard pace.
CUT_AS_SAVED = {"start": 23.521, "end": 24.89, "reason": "auto: pause: 1.5s (+1 more)", "source": "auto", "kind": "pauses"}


def transcript() -> list[dict]:
    """The words with the times the transcript gave them."""
    return [{"word": text, "start": start, "end": end} for text, start, end, *_ in ROWS]


def aligned() -> list[dict]:
    """The words as the aligner gave them, each with the transcript's times beside."""
    return [
        {"word": text, "start": a, "end": b, word_times.RAW_START: start, word_times.RAW_END: end}
        for text, start, end, a, b, _s, _e in ROWS
    ]


def as_saved_with_room_before() -> list[dict]:
    """The words as the plugin saved them with room, before sound heard inside a word was the word's."""
    out = []
    for text, start, end, a, b, s, e in ROWS:
        entry = {"word": text, "start": s, "end": e, word_times.RAW_START: start, word_times.RAW_END: end}
        if (s, e) != (a, b):
            entry[word_times.ALIGNED_START], entry[word_times.ALIGNED_END] = a, b
        out.append(entry)
    return out


def silences(_video: Path | None = None) -> list[Silence]:
    """The take's measured silences. A ``SilenceProvider``."""
    return [Silence(start=a, end=b) for a, b in QUIET]


def soft(_video: Path | None = None) -> list[Silence]:
    """The take's silences at the soft level. A ``SilenceProvider``."""
    return [Silence(start=a, end=b) for a, b in SOFT_QUIET]


class Aligner:
    """An aligner that gives the take's words the aligned times of ``ROWS``."""

    def lacks(self) -> str:
        return ""

    def align(self, video, words, silences):
        return [{**w, "start": a, "end": b} for w, (_t, _s, _e, a, b, *_saved) in zip(words, ROWS)]


def on_video_as_saved_before(video: Path) -> Project:
    """The project of ``video`` with these words aligned and given room as now, and the Standard edit the plugin saved before."""
    video.with_suffix(".words.json").write_text(json.dumps(transcript()))
    project = open_project(str(video))
    word_times.align_project(project, aligner=Aligner(), silences=silences, soft=soft)
    standard = {"pace": "standard", "take_out": {"pauses": True, "fillers": True, "repeats": True, "likes": True},
                "gap_length": None, "fine": {"gap_length": 0.6, "rhythm": 3.0}}
    project.edit_path.write_text(json.dumps({
        "version": 1, "video": str(video), "duration": DURATION, "auto_tighten": True, "gap_length": None,
        "pace": "standard", "cuts": [CUT_AS_SAVED], "requested": [], "rejected": [], "keep": [],
        "treatment": standard, "set_by_claude": standard, "word_times": word_times.MEASURED, "word_times_made_as": 2,
    }))
    return project


def times_of(words: list[dict], text: str) -> tuple[float, float]:
    return next((w["start"], w["end"]) for w in words if w["word"] == text)
