"""Find every place a word or a short phrase is said, for Claude to pick the fillers among them.

No rule can tell "I like jazz" from "divide this into like three sections".
A reader can. ``find_words`` gives Claude each place with the words around
it, the quiet either side, and whether a cut there would be clean; Claude
reads them, picks the ones that are filler, and hands their ids to
``set_edit`` as ``picks``.

The answer is packed text, one place a line, the way ``read_transcript``
packs the transcript: 112 places take about 12,000 characters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lumr_studio.edit import SaidOrder, SilenceIndex, clock, pick_id, why_not_clean
from lumr_studio.errors import StudioError
from lumr_studio.transcript import is_event, plain_text

# Claude asks for a few words at a time, so one answer stays readable.
MAX_PHRASES = 5
# A filler is a word or two: "like", "you know", "sort of".
MAX_PHRASE_WORDS = 4
# Words shown either side of each place. Six is enough to tell a verb from a
# filler ("I like jazz" / "into like three sections").
CONTEXT_WORDS = 6
# The packed text stays under this many characters. A word said 112 times
# takes about 12,000; the rest is room for a word said twice as often.
PACK_BUDGET_CHARS = 24000

# Who removes a place already, as the packed text names them.
OUT_BY = {1: "pace", 2: "claude", 3: "creator", 4: "pick"}


@dataclass(frozen=True)
class Place:
    """One place a phrase is said."""

    phrase: str
    id: str
    start: float
    end: float
    before: str
    said: str
    after: str
    quiet_before: float
    quiet_after: float
    out_by: str        # "" when it plays, else one of OUT_BY's values
    why_not_clean: str  # "" when a cut here is clean

    @property
    def clean(self) -> bool:
        return not self.why_not_clean


def check_phrases(phrases: Any) -> list[list[str]]:
    """The phrases as lists of plain words. Raises StudioError for a list Claude should fix."""
    example = '["like"] or ["you know", "so"]'
    if not isinstance(phrases, list) or not phrases or not all(isinstance(p, str) for p in phrases):
        raise StudioError(f"words must be a list of one to {MAX_PHRASES} words or short phrases, like {example}.")
    if len(phrases) > MAX_PHRASES:
        raise StudioError(f"words holds {len(phrases)} entries. Ask for at most {MAX_PHRASES} at a time, like {example}.")
    out = []
    for phrase in phrases:
        parts = [plain_text(t) for t in phrase.split()]
        if not parts or not all(parts):
            raise StudioError(f"{phrase!r} holds no word to look for. Pass words or short phrases, like {example}.")
        if len(parts) > MAX_PHRASE_WORDS:
            raise StudioError(f"{phrase!r} is {len(parts)} words long. A phrase is at most {MAX_PHRASE_WORDS} words.")
        if parts not in out:
            out.append(parts)
    return out


def _quote(tokens: list[dict[str, Any]]) -> str:
    return " ".join("(sound)" if is_event(w) else str(w["word"]).strip() for w in tokens)


def find_places(
    words: list[dict[str, Any]], phrases: list[list[str]], silences: list[Any], duration: float,
    codes: dict[tuple[float, float], int] | None = None,
) -> list[Place]:
    """Every place each phrase is said, in time order.

    ``codes`` says why a word is removed already, by its ``(start, end)``
    rounded to the millisecond, with the values the page uses in ``words[]``.
    """
    said = SaidOrder.of(words)
    quiet = SilenceIndex(silences)
    plain = [None if is_event(w) else plain_text(str(w["word"])) for w in said.tokens]
    codes = codes or {}
    places = []
    for parts in phrases:
        n = len(parts)
        for i in range(len(plain) - n + 1):
            if plain[i:i + n] != parts:
                continue
            picked = range(i, i + n)
            chosen = said.tokens[i:i + n]
            start = round(min(float(w["start"]) for w in chosen), 3)
            end = round(max(float(w["end"]) for w in chosen), 3)
            floor, ceiling = said.neighbours(picked, duration)
            owners = [codes.get((round(float(w["start"]), 3), round(float(w["end"]), 3)), 0) for w in chosen]
            places.append(Place(
                phrase=" ".join(parts), id=pick_id(start, end), start=start, end=end,
                before=_quote(said.tokens[max(0, i - CONTEXT_WORDS):i]), said=_quote(chosen),
                after=_quote(said.tokens[i + n:i + n + CONTEXT_WORDS]),
                quiet_before=round(max(0.0, start - floor), 2), quiet_after=round(max(0.0, ceiling - end), 2),
                out_by=OUT_BY.get(owners[0], "") if all(owners) else "",
                why_not_clean=why_not_clean(picked, said, quiet, duration),
            ))
    return sorted(places, key=lambda p: (p.start, p.end))


def _short(seconds: float) -> str:
    """A length to two decimals without the leading zero: ``.04``, ``1.20``."""
    return f"{seconds:.2f}".removeprefix("0")


def _line(place: Place) -> str:
    notes = (f" OUT:{place.out_by}" if place.out_by else "") + ("" if place.clean else f" NOT CLEAN: {place.why_not_clean}")
    return (
        f"{place.id} {clock(place.start)} {place.before} [{place.said}] {place.after} "
        f"q{_short(place.quiet_before)}/{_short(place.quiet_after)}{notes}"
    )


HOW_TO_READ = (
    "Each line: id, clock, the words before, [the word], the words after, q<quiet before>/<quiet after> in seconds. "
    "OUT:<who> means the saved edit removes it already (pace, claude, creator, or pick for one you picked before). "
    "NOT CLEAN means a cut there would clip the word beside it; set_edit leaves such a pick in. "
    "Hand the ids of the fillers to set_edit as picks, each with a short reason."
)


def pack_places(places: list[Place], phrases: list[list[str]], *, start: float | None = None,
                budget_chars: int = PACK_BUDGET_CHARS) -> dict[str, Any]:
    """The places as packed text, with a count for each phrase.

    Returns ``{text, words: [{word, said, clean, already_out}], listed,
    next_start}``. The counts are for the whole video. The lines start at
    ``start`` and stop when the budget is used up; the text then ends with
    ``NEXT <time>``, the ``start`` to pass to go on, else with ``END``.
    """
    counts = []
    for parts in phrases:
        phrase = " ".join(parts)
        mine = [p for p in places if p.phrase == phrase]
        counts.append({"word": phrase, "said": len(mine), "clean": sum(p.clean for p in mine),
                       "already_out": sum(bool(p.out_by) for p in mine)})
    lines = [
        f"{c['word']}: said {c['said']} times, {c['clean']} clean to cut, {c['said'] - c['clean']} not, "
        f"{c['already_out']} already out."
        for c in counts
    ] + [HOW_TO_READ]
    used = sum(len(line) + 1 for line in lines)
    listed, next_start = 0, None
    for place in places:
        if start is not None and place.start < start:
            continue
        line = _line(place)
        if listed and used + len(line) + 1 > budget_chars:
            next_start = place.start
            break
        lines.append(line)
        used += len(line) + 1
        listed += 1
    lines.append(f"NEXT {next_start!r}" if next_start is not None else "END")
    return {"text": "\n".join(lines), "words": counts, "listed": listed, "next_start": next_start}
