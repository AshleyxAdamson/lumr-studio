"""A made-up take with a word the aligner cut short, for every test of word room.

"You can just take one step at a time." then "And um, it works." The
aligner gave "take" 1 ms, inside the pause after "just": the closed mouth
before its "t" is a measured silence, and ClipForge ends a word at the first
one after its start. The rest of "take" sounds from 2.033 to 2.162 s, under
no word. "um," sits alone between two pauses; a test can put another word
there (``on_video(video, alone="like,")``).

Each row is ``(word, transcript start, transcript end, aligned start,
aligned end)``. The transcript's words touch, as Hammy's do.
"""

from __future__ import annotations

import json
from pathlib import Path

from lumr_studio.engine.audio_boundaries import Silence

from lumr_studio import word_times
from lumr_studio.project import Project, open_project

ROWS = [
    ("You", 1.00, 1.28, 0.98, 1.08),
    ("can", 1.28, 1.60, 1.12, 1.26),
    ("just", 1.60, 1.92, 1.32, 1.507),
    ("take", 1.92, 2.24, 1.941, 1.942),
    ("one", 2.24, 2.56, 2.402, 2.518),
    ("step", 2.56, 2.88, 2.582, 2.764),
    ("at", 2.88, 3.04, 2.862, 2.942),
    ("a", 3.04, 3.28, 2.962, 2.981),
    ("time.", 3.28, 3.80, 3.042, 3.377),
    ("And", 5.00, 5.40, 5.00, 5.12),
    ("um,", 5.40, 6.40, 5.60, 5.72),
    ("it", 6.40, 6.70, 6.50, 6.62),
    ("works.", 6.70, 7.30, 6.66, 7.00),
]
# What the silence detector would measure. The rest is sound.
QUIET = [
    (0.0, 0.97), (1.507, 1.932), (1.9415, 2.033), (2.162, 2.2231), (2.2232, 2.333), (2.518, 2.58),
    (2.764, 2.856), (2.981, 3.04), (3.377, 4.99), (5.13, 5.59), (5.73, 6.49), (7.01, 20.0),
]
# "take" as the aligner gave it, and with the room for its sound.
TAKE_AS_ALIGNED = (1.941, 1.942)
TAKE = (1.932, 2.162)
# The part of "take" that sounds after the closed mouth: what a pause removal took on the real take.
TAKE_SOUNDS = (2.033, 2.162)
# The word that sits alone between two pauses, as the transcript has it, and with its room.
ALONE = "um,"
ALONE_AS_ALIGNED = (5.6, 5.72)
ALONE_ROOM = (5.59, 5.73)


def _rows(alone: str) -> list[tuple]:
    return [(alone if text == ALONE else text, *rest) for text, *rest in ROWS]


def transcript(alone: str = ALONE) -> list[dict]:
    """The words with the times the transcript gave them. ``alone`` is the word between the two pauses."""
    return [{"word": text, "start": start, "end": end} for text, start, end, _a, _b in _rows(alone)]


def aligned(alone: str = ALONE) -> list[dict]:
    """The words as the aligner gave them, each with the transcript's times beside."""
    return [
        {"word": text, "start": a, "end": b, word_times.RAW_START: start, word_times.RAW_END: end}
        for text, start, end, a, b in _rows(alone)
    ]


def silences(_video: Path | None = None) -> list[Silence]:
    """The take's measured silences. A ``SilenceProvider``."""
    return [Silence(start=a, end=b) for a, b in QUIET]


class ShortWordAligner:
    """An aligner that gives the take's words the aligned times of ``ROWS``."""

    def lacks(self) -> str:
        return ""

    def align(self, video, words, silences):
        times = {start: (a, b) for _text, start, _end, a, b in ROWS}
        return [{**w, "start": times[w["start"]][0], "end": times[w["start"]][1]} for w in words]


def on_video(video: Path, alone: str = ALONE) -> Project:
    """The project of ``video`` with this take as its transcript, aligned, each word with its room."""
    video.with_suffix(".words.json").write_text(json.dumps(transcript(alone)))
    project = open_project(str(video))
    word_times.align_project(project, aligner=ShortWordAligner(), silences=silences)
    return project


def as_saved_by_an_earlier_build(project: Project, version: int = 1) -> None:
    """Put the project's aligned file back to what an earlier build saved: the aligner's times as they came."""
    path = word_times.aligned_path(project)
    data = json.loads(path.read_text())
    data["version"] = version
    data["words"] = aligned(next(w["word"] for w in data["words"] if w["start"] > 5.5))
    path.write_text(json.dumps(data))


def times_of(words: list[dict], text: str) -> tuple[float, float]:
    return next((w["start"], w["end"]) for w in words if w["word"] == text)
