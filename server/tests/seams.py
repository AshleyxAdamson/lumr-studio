"""Two made-up lines where the aligner ran one word over the next, at a seam of its 40 s windows.

"so on the road, pass him now if this happens" from 359.2 s, where "him"
(360.039-360.261) runs over all of "now" (360.08-360.2), and "a big plan we
will make" from 639.32 s, where "plan" (639.979-640.433) runs over "we"
(640.24-640.38). The later word is right: "we" sounds from 640.21. Before
words were kept apart, a double-click on "now" took "him" with it, and one
on "we" left most of it playing.

The words are invented. The times are measured on a take, and they are the
data. The lines are moved to the start of the 20 s test video: every time is
the take's less ``SHIFT``. Each row is ``(line, word, transcript start,
transcript end, aligned start, aligned end)``. The silences are the ones
measured on the take, and a pause joins the two lines.
"""

from __future__ import annotations

import json
from pathlib import Path

from lumr_studio.engine.audio_boundaries import Silence

from lumr_studio import word_times
from lumr_studio.project import Project, open_project

SHIFT = {1: 358.0, 2: 634.0}
ROWS = [
    (1, "so", 359.2, 359.36, 359.379, 359.479), (1, "on", 359.36, 359.52, 359.519, 359.659),
    (1, "the", 359.52, 359.6, 359.699, 359.759), (1, "road,", 359.6, 359.76, 359.779, 359.879),
    (1, "pass", 359.76, 359.84, 359.919, 360.019), (1, "him", 359.84, 360.12, 360.039, 360.261),
    (1, "now", 360.12, 360.28, 360.08, 360.2), (1, "if", 360.28, 360.36, 360.24, 360.261),
    (1, "this", 360.36, 360.52, 360.32, 360.46), (1, "happens", 360.52, 360.92, 360.52, 360.858),
    (2, "a", 639.32, 639.64, 639.439, 639.519), (2, "big", 639.64, 639.96, 639.659, 639.919),
    (2, "plan", 639.96, 640.28, 639.979, 640.433), (2, "we", 640.28, 640.52, 640.24, 640.38),
    (2, "will", 640.52, 640.84, 640.48, 640.78), (2, "make", 640.84, 641.16, 640.84, 640.96),
]
# (line, start, end) as measured on the take. The first line's last pause runs into the second line's first.
QUIET = [
    (1, 359.2177, 359.3347), (1, 360.2611, 360.3417), (1, 360.4737, 360.527), (1, 360.8579, 363.4381),
    (2, 640.4327, 640.4868), (2, 640.9729, 654.0),
]
# The two words the aligner ran over, and the words it ran them over with, as the aligner gave them.
RUN_OVER = {"now": ("him", 360.261), "we": ("plan", 640.433)}


def _at(line: int, t: float) -> float:
    return round(t - SHIFT[line], 4)


def transcript() -> list[dict]:
    return [{"word": text, "start": _at(line, a), "end": _at(line, b)} for line, text, a, b, _s, _e in ROWS]


def silences(_video: Path | None = None) -> list[Silence]:
    """The measured silences, moved. A ``SilenceProvider``."""
    return [Silence(start=_at(line, a), end=_at(line, b)) for line, a, b in QUIET]


class SeamAligner:
    """An aligner that gives the words the aligned times of ``ROWS``."""

    def lacks(self) -> str:
        return ""

    def align(self, video, words, silences):
        times = {_at(line, a): (_at(line, s), _at(line, e)) for line, _text, a, _b, s, e in ROWS}
        return [{**w, "start": times[w["start"]][0], "end": times[w["start"]][1]} for w in words]


def on_video(video: Path) -> Project:
    """The project of ``video`` with these words as its transcript, aligned and given room."""
    video.with_suffix(".words.json").write_text(json.dumps(transcript()))
    project = open_project(str(video))
    word_times.align_project(project, aligner=SeamAligner(), silences=silences)
    return project
