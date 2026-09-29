"""The pace dial: how hard automatic trims tighten a take.

One named level sets three things:

* which pauses get trimmed (``gap_length``, the shortest pause that qualifies),
* how far apart cuts must be (``rhythm``, ClipForge's 1 to 5 spacing dial:
  lower keeps more speech between two cuts, so trims can't fire back to back),
* how much pause stays at the join, by the kind of boundary. A pause between
  sentences is the speaker's full stop and gets the most room. A pause inside
  a clause is often a thought forming and gets the least, though never none.

ClipForge plans the trims and the edit builder moves their edges to word
boundaries. ``leave_pauses`` runs last, on the finished trims: it shortens any
that would leave less pause than the level allows, and drops one when nothing
worth cutting remains. Shortening only ever gives time back, so a finished
trim's edges stay clear of words.

A pause is "air": time that is measured silence or lies between two
transcript entries. Neither source is enough alone. Estimated word times butt
most words together (on the test footage 94% of neighbouring words have no
gap; with measured times, 6%), and measured silence misses a pause filled by
a breath.

The six levels were measured on a 21 minute take with measured word times
(see ``word_times.py``). The last one, ``max``, reaches as far as the Mac
app's hardest setting (every pause from 0.15 s up is cut, with no spacing
between cuts) and still leaves a small pause at each join, which the Mac app
does not: the creator prefers how these joins read.

Under the stops sit two sliders, the two the Mac app has: the shortest pause
cut and the rhythm. A setting made with them is the pace ``custom``. It is
no seventh stop: ``level_for`` is the one place that turns a pace name and
the sliders' values into the level an edit is built with, and
``custom_level`` gives her setting the pause floors of the stops around it,
so her own setting still leaves a pause at each join.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Any

from lumr_studio.autocuts import SHORTEST_PIECE_SECONDS
from lumr_studio.engine.audio_boundaries import Silence
from lumr_studio.engine.microcut_pacing import (
    GAP_LENGTH_MAX,
    GAP_LENGTH_MIN,
    RHYTHM_MAX,
    RHYTHM_MIN,
    _spacing_for_rhythm,
)
from lumr_studio.errors import StudioError
from lumr_studio.transcript import _is_clause_end, _is_sentence_end, is_event

# Cuts of single words: the creator's by hand, and the filler words Claude picked.
WORD_CUTS = ("you", "pick")
# A trim shorter than this after shortening removes nothing worth removing.
MIN_TRIM_SECONDS = SHORTEST_PIECE_SECONDS
# A trim this close to leaving enough pause is left alone.
ROUNDING = 0.005


@dataclass(frozen=True)
class Pace:
    """One level of the dial. All values in seconds except ``rhythm``."""

    name: str
    summary: str
    gap_length: float
    rhythm: float
    sentence_pause: float
    clause_pause: float
    inner_pause: float

    def pause_after(self, word: dict[str, Any]) -> float:
        """The pause this level leaves after ``word``, by the boundary it ends."""
        if _is_sentence_end(word):
            return self.sentence_pause
        if _is_clause_end(word):
            return self.clause_pause
        return self.inner_pause


# The pauses left follow what editors and editing tools report: 0.3 to 0.5s
# between sentences for a standard talking-head cut, about half that inside a
# sentence. The test footage's own pause between sentences is 0.43s (median),
# so "standard" leaves a sentence break close to how the creator talks.
# Spacing follows ClipForge's rhythm curve: 2 keeps 2.0s of speech between
# cuts, 3 keeps 1.2s, 4 keeps 0.5s, 4.5 keeps 0.25s, 5 keeps none.
#
# The summaries are read by the creator on the page. From "hard" on, each
# says what she gives up.
PACES: dict[str, Pace] = {
    p.name: p
    for p in (
        Pace(
            name="natural",
            summary="Only long pauses go. Reads as one unbroken take.",
            gap_length=1.0, rhythm=2, sentence_pause=0.6, clause_pause=0.4, inner_pause=0.3,
        ),
        Pace(
            name="standard",
            summary="The usual for talking to camera: tight, with room to breathe between sentences.",
            gap_length=0.6, rhythm=3, sentence_pause=0.4, clause_pause=0.25, inner_pause=0.18,
        ),
        Pace(
            name="fast",
            summary="Quick and punchy. Short breaks between sentences, and more of the small pauses go.",
            gap_length=0.4, rhythm=4, sentence_pause=0.25, clause_pause=0.15, inner_pause=0.1,
        ),
        Pace(
            name="tight",
            summary="Small pauses inside sentences go too, and only a breath is left between sentences.",
            gap_length=0.3, rhythm=4.5, sentence_pause=0.18, clause_pause=0.1, inner_pause=0.06,
        ),
        Pace(
            name="hard",
            summary="Almost no pause is left between sentences, and cuts can land a few words apart.",
            gap_length=0.2, rhythm=5, sentence_pause=0.12, clause_pause=0.06, inner_pause=0.03,
        ),
        Pace(
            name="max",
            summary="Every pause that can go does, with only a very short one left at each join, so some joins will sound choppy.",
            gap_length=0.15, rhythm=5, sentence_pause=0.1, clause_pause=0.05, inner_pause=0.03,
        ),
    )
}
# The starting pace when the creator has no usual saved. The first live test
# on real footage went well at standard, so that is where a video starts.
DEFAULT_PACE = "standard"


def get_pace(name: str | None) -> Pace:
    """The stop called ``name`` (default ``DEFAULT_PACE``). Raises StudioError for an unknown name.

    ``custom`` is not a stop and is refused here: it is made by the sliders
    on the page, never picked by name. See ``level_for``.
    """
    key = (name or DEFAULT_PACE).strip().lower()
    if key == CUSTOM:
        raise StudioError(
            f"pace {CUSTOM!r} is the creator's own setting, made with the sliders on the page, and can't be "
            f"picked by name. Leave pace out to keep it, or pass one of: {', '.join(PACES)}."
        )
    if key not in PACES:
        raise StudioError(f"pace {name!r} is not a level. Pass one of: {', '.join(PACES)}.")
    return PACES[key]


# ── Her own setting ───────────────────────────────────────────────────────────

# The pace a setting made with the sliders reads as.
CUSTOM = "custom"
CUSTOM_LABEL = "Custom"
# The sliders. The ranges are ClipForge's own; the steps are the Mac app's
# for the pause length, and half a rhythm for the rhythm, which gives nine
# positions with the engine's own in-between spacings.
GAP_STEP = 0.05
RHYTHM_STEP = 0.5
GAP_LABEL = "Cut pauses longer than"
RHYTHM_LABEL = "Speech kept between cuts"
# A value this close to a step is on it: the page sends 0.35 as 0.35000000000000003.
ON_STEP = 1e-6


def _steps(low: float, high: float, step: float) -> list[float]:
    return [round(low + step * i, 2) for i in range(int(round((high - low) / step)) + 1)]


GAP_POSITIONS = _steps(GAP_LENGTH_MIN, GAP_LENGTH_MAX, GAP_STEP)
RHYTHM_POSITIONS = _steps(RHYTHM_MIN, RHYTHM_MAX, RHYTHM_STEP)


def _on_slider(value: float, positions: list[float]) -> float | None:
    """The slider position ``value`` sits on, or None when it sits on none."""
    return next((p for p in positions if abs(p - value) <= ON_STEP), None)


def check_fine(gap_length: float, rhythm: float) -> tuple[float, float]:
    """The two slider values, each moved onto its exact step. Raises StudioError for one that is on no step.

    The messages are read by the creator.
    """
    gap, beat = _on_slider(gap_length, GAP_POSITIONS), _on_slider(rhythm, RHYTHM_POSITIONS)
    if gap is None:
        raise StudioError(
            f"That pause length is not on the slider. It goes from {GAP_LENGTH_MIN:g} to {GAP_LENGTH_MAX:g} "
            f"seconds in steps of {GAP_STEP:g}."
        )
    if beat is None:
        raise StudioError(
            f"That rhythm is not on the slider. It goes from {RHYTHM_MIN:g} to {RHYTHM_MAX:g} in steps of {RHYTHM_STEP:g}."
        )
    return gap, beat


def speech_kept(rhythm: float) -> float:
    """Seconds of speech ClipForge keeps between two cuts at ``rhythm``. Its own curve, halves included."""
    return round(_spacing_for_rhythm(rhythm), 2)


def _seconds(value: float) -> tuple[str, str]:
    """A length as shown beside a slider and as said by a screen reader: ``("0.35 s", "0.35 seconds")``."""
    return f"{value:g} s", f"{value:g} second" + ("" if value == 1 else "s")


def fine_ranges() -> dict[str, Any]:
    """The two sliders for the page: range, step, label, and the text to show at every position.

    The rhythm slider shows the speech kept between cuts, since a rhythm of
    3.5 means nothing to the creator. ``harder`` names the end that cuts
    harder, so the page can draw both sliders cutting harder to the right.
    """
    def slider(positions: list[float], step: float, label: str, harder: str, shown: Any) -> dict[str, Any]:
        return {
            "min": positions[0], "max": positions[-1], "step": step, "label": label, "harder": harder,
            "positions": [dict(zip(("value", "shown", "said"), (p, *_seconds(shown(p))))) for p in positions],
        }

    return {
        "gap_length": slider(GAP_POSITIONS, GAP_STEP, GAP_LABEL, "min", lambda p: p),
        "rhythm": slider(RHYTHM_POSITIONS, RHYTHM_STEP, RHYTHM_LABEL, "max", speech_kept),
    }


def _stops_by_gap() -> list[Pace]:
    return sorted(PACES.values(), key=lambda p: p.gap_length)


def stops_around(gap_length: float) -> tuple[Pace, Pace]:
    """The two stops whose pause lengths sit either side of ``gap_length``: ``(harder, gentler)``.

    On a stop's own pause length, or past either end, both are that stop.
    """
    stops = _stops_by_gap()
    if gap_length <= stops[0].gap_length:
        return stops[0], stops[0]
    for harder, gentler in zip(stops, stops[1:]):
        if gap_length < gentler.gap_length:
            return (harder, harder) if gap_length == harder.gap_length else (harder, gentler)
    return stops[-1], stops[-1]


def custom_summary(gap_length: float) -> str:
    """One plain sentence on where her setting sits among the stops. No numbers: the sliders show those."""
    harder, gentler = stops_around(gap_length)
    if harder is gentler:
        return f"Your own setting, close to {harder.name.capitalize()}."
    return f"Your own setting, between {gentler.name.capitalize()} and {harder.name.capitalize()}."


def custom_level(gap_length: float, rhythm: float) -> Pace:
    """The level for a setting made with the sliders.

    The pause left at each join is drawn on a straight line between the two
    stops whose pause lengths sit either side of hers. At or past either end
    it is that end's. So her own setting always leaves a pause at each join,
    the way the stops do.
    """
    harder, gentler = stops_around(gap_length)
    share = 0.0 if harder is gentler else (gap_length - harder.gap_length) / (gentler.gap_length - harder.gap_length)

    def between(a: float, b: float) -> float:
        return round(a + (b - a) * share, 3)

    return Pace(
        name=CUSTOM, summary=custom_summary(gap_length), gap_length=gap_length, rhythm=rhythm,
        sentence_pause=between(harder.sentence_pause, gentler.sentence_pause),
        clause_pause=between(harder.clause_pause, gentler.clause_pause),
        inner_pause=between(harder.inner_pause, gentler.inner_pause),
    )


def level_for(pace: str | None, fine: tuple[float, float] | None = None) -> Pace:
    """The level an edit is built with: the stop called ``pace``, or her own setting when ``pace`` is ``custom``.

    The one place that knows ``custom`` is not a stop. ``fine`` is
    ``(gap_length, rhythm)`` and is read only for ``custom``. Raises
    StudioError for an unknown name, and for ``custom`` without values.
    """
    if (pace or "").strip().lower() != CUSTOM:
        return get_pace(pace)
    if fine is None:
        raise StudioError("The saved pace is the creator's own setting but its two values are missing. Pick a pace on the page.")
    return custom_level(*check_fine(*fine))


def describe_paces() -> list[dict[str, Any]]:
    """Every level as plain data, gentlest first, for reports and the skill."""
    return [
        {
            "pace": p.name, "summary": p.summary, "trims_pauses_longer_than": p.gap_length,
            "speech_kept_between_cuts": speech_kept(p.rhythm),
            "pause_left": {"between_sentences": p.sentence_pause, "at_a_comma": p.clause_pause,
                           "inside_a_clause": p.inner_pause},
        }
        for p in PACES.values()
    ]


# ── Measuring quiet ───────────────────────────────────────────────────────────

Span = tuple[float, float]


def _mid(word: dict[str, Any]) -> float:
    return (float(word["start"]) + float(word["end"])) / 2


def _overlap(a: float, b: float, c: float, d: float) -> float:
    return max(0.0, min(b, d) - max(a, c))


def _merge(spans: list[Span]) -> list[Span]:
    merged: list[Span] = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        elif b > a:
            merged.append((a, b))
    return merged


@dataclass(frozen=True)
class _Quiet:
    """A take's words and silences sorted once, so the air near a trim is found by binary search.

    A take at the hardest pace has about 700 trims, 3400 words and 1500
    silences; reading every word and silence for every trim took seconds.
    """

    words: list[Span]
    word_starts: list[float]
    longest_word: float
    silences: list[Span]
    silence_starts: list[float]
    longest_silence: float

    @classmethod
    def of(cls, words: list[dict[str, Any]], silences: list[Silence]) -> _Quiet:
        spans = sorted((float(w["start"]), float(w["end"])) for w in words)
        quiet = sorted((float(s.start), float(s.end)) for s in silences)
        return cls(
            words=spans, word_starts=[a for a, _ in spans],
            longest_word=max((b - a for a, b in spans), default=0.0),
            silences=quiet, silence_starts=[a for a, _ in quiet],
            longest_silence=max((b - a for a, b in quiet), default=0.0),
        )

    @staticmethod
    def _touching(spans: list[Span], starts: list[float], longest: float, lo: float, hi: float) -> list[Span]:
        """The spans that overlap ``[lo, hi]``, in start order."""
        first = bisect.bisect_left(starts, lo - longest)
        last = bisect.bisect_left(starts, hi)
        return [(a, b) for a, b in spans[first:last] if b > lo]

    def air(self, lo: float, hi: float) -> list[Span]:
        """The stretches of air inside ``[lo, hi]``, sorted and merged."""
        found = [(max(lo, a), min(hi, b))
                 for a, b in self._touching(self.silences, self.silence_starts, self.longest_silence, lo, hi)]
        at = lo
        for start, end in self._touching(self.words, self.word_starts, self.longest_word, lo, hi):
            if start > at:
                found.append((at, min(start, hi)))
            at = max(at, end)
        if at < hi:
            found.append((at, hi))
        return _merge(found)


def air_spans(words: list[dict[str, Any]], silences: list[Silence], lo: float, hi: float) -> list[Span]:
    """The stretches of air inside ``[lo, hi]``, sorted and merged.

    Air is measured silence, plus every gap between transcript entries (words
    and sounds both count as not air).
    """
    return _Quiet.of(words, silences).air(lo, hi)


def _air_inside(spans: list[Span], lo: float, hi: float) -> float:
    return sum(_overlap(a, b, lo, hi) for a, b in spans)


# ── Leaving the pause ─────────────────────────────────────────────────────────


@dataclass
class PauseReport:
    """What ``leave_pauses`` did to the planned trims."""

    shortened: int = 0
    dropped: int = 0


def _neighbours(
    spoken: list[dict[str, Any]], mids: list[float], start: float, end: float
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """The last kept word before the trim and the first kept word after it.

    ``spoken`` is sorted by midpoint and ``mids`` holds those midpoints, so
    both are found by binary search.
    """
    before = bisect.bisect_right(mids, start) - 1
    after = bisect.bisect_left(mids, end)
    if before < 0 or after >= len(spoken):
        return None
    return spoken[before], spoken[after]


def _give_back(start: float, end: float, air: list[Span], need: float) -> Span:
    """Shorten the trim ``[start, end]`` so ``need`` more seconds of air survive.

    Air comes back from the end first, where it becomes the lead-in to the
    next word, then from the start. Only air is given back, never a removed
    word, so a filler the trim takes stays taken.
    """
    tail = next((end - max(a, start) for a, b in reversed(air) if a < end <= b), 0.0)
    from_end = min(need, max(0.0, tail))
    end -= from_end
    need -= from_end
    if need > 0:
        head = next((min(b, end) - start for a, b in air if a <= start < b), 0.0)
        start += min(need, max(0.0, head))
    return start, end


def _held_inside(start: float, end: float, held: list[Span]) -> Span | None:
    """The stretch of ``[start, end]`` that must stay removed: from the first held span inside it to the last."""
    inside = [(a, b) for a, b in held if a < end and start < b]
    if not inside:
        return None
    return max(start, min(a for a, _ in inside)), min(end, max(b for _, b in inside))


def leave_pauses(
    trims: list[dict[str, Any]],
    words: list[dict[str, Any]],
    silences: list[Silence],
    pace: Pace,
    held: list[Span] | None = None,
) -> tuple[list[dict[str, Any]], PauseReport]:
    """The trims, each shortened so the join keeps the pause ``pace`` allows.

    Returns the trims that remain and a count of those shortened and dropped.
    A trim at the very start or end of the take has one neighbour only and is
    left as it is. ``held`` are spans that must stay removed whatever the
    pause (words the creator cut by hand): a trim holding one is shortened
    around it and never dropped. Any other trim shorter than
    ``MIN_TRIM_SECONDS``, shortened here or not, is dropped: between two words
    with soft ends it removes nothing you hear, and the picture still jumps.
    """
    report = PauseReport()
    if not any((pace.sentence_pause, pace.clause_pause, pace.inner_pause)):
        return list(trims), report  # this level leaves no pause, so there is nothing to give back
    spoken = sorted((w for w in words if not is_event(w)), key=_mid)
    mids = [_mid(w) for w in spoken]
    quiet = _Quiet.of(words, silences)
    kept: list[dict[str, Any]] = []
    for trim in trims:
        start, end = float(trim["start"]), float(trim["end"])
        core = _held_inside(start, end, held or [])
        if core is None and end - start < MIN_TRIM_SECONDS:
            report.dropped += 1
            continue
        pair = _neighbours(spoken, mids, start, end)
        if pair is None:
            kept.append(trim)
            continue
        before, after = pair
        air = quiet.air(_mid(before), _mid(after))
        left = _air_inside(air, _mid(before), _mid(after)) - _air_inside(air, start, end)
        need = pace.pause_after(before) - left
        if need <= ROUNDING:
            kept.append(trim)
            continue
        start, end = _give_back(start, end, air, need)
        if core is not None:
            start, end = min(start, core[0]), max(end, core[1])
        elif end - start < MIN_TRIM_SECONDS:
            report.dropped += 1
            continue
        report.shortened += 1
        kept.append({**trim, "start": round(start, 3), "end": round(end, 3)})
    return kept, report


def settle_trims(
    cuts: list[dict[str, Any]],
    words: list[dict[str, Any]],
    silences: list[Silence],
    pace: Pace,
    held: list[Span] | None = None,
) -> tuple[list[dict[str, Any]], PauseReport]:
    """A finished cut list with ``leave_pauses`` applied to its automatic trims.

    Cuts the model chose pass through untouched, in place. A word cut by the
    creator's hand, or picked by Claude as filler, passes through too, unless
    an automatic trim joined it: a word knocked out beside a pause would
    otherwise take the whole pause with it and jam the words either side
    together. Such a cut is shortened like a trim, around ``held``, the
    words as placed.
    """
    def settles(cut: dict[str, Any]) -> bool:
        return cut.get("source") == "auto" or (cut.get("source") in WORD_CUTS and bool(cut.get("auto_trims")))

    settled, report = leave_pauses([c for c in cuts if settles(c)], words, silences, pace, held)
    passed = [c for c in cuts if not settles(c)]
    return sorted(passed + settled, key=lambda c: float(c["start"])), report
