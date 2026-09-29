"""The compact take report ``analyze_take`` returns."""

from __future__ import annotations

from typing import Any

from lumr_studio.autocuts import plan_auto_cuts
from lumr_studio.engine.audio_boundaries import Silence
from lumr_studio.engine.filler import detect_fillers
from lumr_studio.engine.render import _merge_spans
from lumr_studio.retakes import find_retakes
from lumr_studio.transcript import spoken_words, timing_report

LONGEST_PAUSES_SHOWN = 5
FILLERS_SHOWN = 20
RETAKES_SHOWN = 10
# detect_fillers level 1: the um/uh family only, never real words like "so".
FILLER_LEVEL = 1


def longest_pauses(words: list[dict[str, Any]], limit: int = LONGEST_PAUSES_SHOWN) -> list[dict[str, float]]:
    """The longest silent gaps between consecutive entries, longest first.

    Vocalization events count as sound: a laugh is not a pause.
    """
    gaps = [
        (float(b["start"]) - float(a["end"]), float(a["end"]), float(b["start"]))
        for a, b in zip(words, words[1:])
    ]
    gaps.sort(reverse=True)
    return [
        {"start": round(s, 2), "end": round(e, 2), "seconds": round(g, 2)}
        for g, s, e in gaps[:limit] if g > 0
    ]


def analyze(
    words: list[dict[str, Any]],
    duration: float,
    silences: list[Silence],
) -> dict[str, Any]:
    """Duration, pace, pauses, fillers, retakes, and the auto micro-cut saving."""
    spoken = spoken_words(words)
    minutes = duration / 60 if duration > 0 else 0
    fillers = detect_fillers(spoken, level=FILLER_LEVEL)
    retakes = find_retakes(words)
    auto = plan_auto_cuts(words, duration=duration, silences=silences)
    auto_saved = sum(e - s for s, e in _merge_spans([(c["start"], c["end"]) for c in auto]))
    return {
        "duration": round(duration, 2),
        "word_count": len(spoken),
        "event_count": len(words) - len(spoken),
        "words_per_minute": round(len(spoken) / minutes, 1) if minutes else 0.0,
        "longest_pauses": longest_pauses(words),
        "filler_count": len(fillers),
        "fillers": [
            {"start": round(f.start, 2), "end": round(f.end, 2), "word": f.reason.removeprefix("filler: ")}
            for f in fillers[:FILLERS_SHOWN]
        ],
        "retake_count": len(retakes),
        "retakes": [r.to_dict() for r in retakes[:RETAKES_SHOWN]],
        "auto_microcuts": {"count": len(auto), "seconds_saved": round(auto_saved, 2)},
        **timing_report(words),
    }
