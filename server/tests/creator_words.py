"""What the creator must never read: one list, used by every test of creator-facing words.

The creator has never seen an editing timeline. Editor slang and marketing
shorthand mean nothing to her, and a time like ``412.30`` is a number, not a
moment in her video. The page's own test and the engine's test both check
against these, so a word banned once is banned everywhere.

The rule on decimals, as it is meant: a time in the video is never a decimal.
A moment reads as a clock time (``6:52``) and a length as a clock length
(``0:04 out``). One thing may be a decimal: a length on a Fine tune slider
(``0.35 s``), because the slider moves in steps of 0.05 s and a clock can't
show that. ``is_slider_length`` and ``decimals_off_the_slider`` are that one
exception. Nothing else may use it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# Editor slang the page must not show, and the shorthand Claude's reasons
# drifted into on the first live test. "user" because the page says "creator"
# in its code and "you" on screen. "trim" because the creator asked never to
# read it: the page says what comes out ("0:04 out", "Long pauses"). Text
# written for Claude (TOOLS.md, tool descriptions, field names such as
# ``paces[].trims``) is not checked against this list and may say trim.
BANNED_WORDS = (
    "seam", "ripple", "veto", "audition", "bypass", "scrub",
    "cta", "pitch tail", "plug", "back-catalog", "b-roll", "user", "trim",
)
# Endings a banned word can carry: plurals, and -ed and -ing forms with the
# last letter doubled (plugging, scrubbed, trimmed).
ENDINGS = "s|es|ing|ed|ging|ged|bing|bed|ming|med"

# A decimal number such as 412.30 or 9.7. A clock time with tenths (0:09.7)
# is allowed, so a digit or colon right before it doesn't count.
DECIMAL_TIME = re.compile(r"(?<![\d:.])\d+\.\d+")
# A length on a Fine tune slider, as the page shows it ("0.35 s") and as a
# screen reader says it ("0.35 seconds"). The whole text, nothing around it.
# One digit before the point and two after at most: no slider reaches 10 s,
# and a time in the video (412.30) can never pass as one.
SLIDER_LENGTH = re.compile(r"\d(?:\.\d{1,2})? (?:s|seconds?)")


def banned_in(text: str) -> list[str]:
    """The banned words in ``text``, with their plurals, -ed and -ing forms, ignoring case."""
    return [
        word for word in BANNED_WORDS
        if re.search(rf"\b{re.escape(word)}(?:{ENDINGS})?\b", text, re.I)
    ]


def decimal_times_in(text: str) -> list[str]:
    """Every decimal number in ``text``, such as ``412.30``."""
    return DECIMAL_TIME.findall(text)


def is_slider_length(text: str) -> bool:
    """Whether ``text`` is a length on a slider and nothing more: ``0.35 s``, ``2 s``, ``0.35 seconds``."""
    return SLIDER_LENGTH.fullmatch(text) is not None


def decimals_off_the_slider(text: str, slider_numbers: Iterable[float]) -> list[str]:
    """The decimal numbers in ``text`` that are not one of ``slider_numbers``.

    For the sentence that refuses a value the slider does not have. It names
    the slider's own ends and step ("from 0.15 to 1.5 seconds in steps of
    0.05"), so those may be decimals. Any other decimal is returned.
    """
    allowed = {float(n) for n in slider_numbers}
    return [found for found in decimal_times_in(text) if float(found) not in allowed]
