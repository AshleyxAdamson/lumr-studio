"""Find repeated phrases that look like retakes.

ClipForge's ``longform`` repeat detector (``EditLevel.cut_repeats``) only
catches one word said twice in a row, and ``detect_edit_cuts`` then filters
those for cutting (minimum cut length, minimum kept segment), so it cannot
report retakes. A retake is bigger: the speaker restarts a sentence, so the
same run of words shows up again a little later. This finds those candidates.
It only reports; it never cuts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from lumr_studio.transcript import spoken_words

MIN_PHRASE_WORDS = 4
MAX_RETAKE_GAP_SECONDS = 60.0


def _norm(text: str) -> str:
    return re.sub(r"[^\w']", "", text.lower())


@dataclass
class Retake:
    """A phrase said at ``first_*`` and said again at ``repeat_*``."""

    phrase: str
    first_start: float
    first_end: float
    repeat_start: float
    repeat_end: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "phrase": self.phrase,
            "first": [round(self.first_start, 2), round(self.first_end, 2)],
            "repeat": [round(self.repeat_start, 2), round(self.repeat_end, 2)],
        }


def find_retakes(
    words: list[dict[str, Any]],
    min_words: int = MIN_PHRASE_WORDS,
    max_gap: float = MAX_RETAKE_GAP_SECONDS,
) -> list[Retake]:
    """Runs of at least ``min_words`` words that recur within ``max_gap`` seconds.

    Each match is extended as far as the two runs keep agreeing, and a run
    already reported is not reported again as a shorter overlap.
    """
    spoken = [w for w in spoken_words(words) if _norm(w["word"])]
    tokens = [_norm(w["word"]) for w in spoken]
    n = len(tokens)
    seen: dict[tuple[str, ...], list[int]] = {}
    covered_until = -1
    found: list[Retake] = []

    for j in range(n - min_words + 1):
        gram = tuple(tokens[j:j + min_words])
        earlier = seen.setdefault(gram, [])
        if j > covered_until:
            for i in reversed(earlier):
                if i + min_words > j:
                    continue  # overlapping occurrence, not a restart
                gap = float(spoken[j]["start"]) - float(spoken[i + min_words - 1]["end"])
                if gap > max_gap:
                    break
                length = min_words
                while j + length < n and i + length < j and tokens[i + length] == tokens[j + length]:
                    length += 1
                found.append(Retake(
                    phrase=" ".join(w["word"].strip() for w in spoken[j:j + length]),
                    first_start=float(spoken[i]["start"]),
                    first_end=float(spoken[i + length - 1]["end"]),
                    repeat_start=float(spoken[j]["start"]),
                    repeat_end=float(spoken[j + length - 1]["end"]),
                ))
                covered_until = j + length - 1
                break
        earlier.append(j)
    return found

