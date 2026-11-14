"""Find every place a word or a short phrase is said, for Claude to pick the fillers among them.

No rule can tell "I like jazz" from "divide this into like three sections".
A reader can. ``find_words`` gives Claude each place with the words around
it, the quiet either side, and whether a cut there would be clean; Claude
reads them, picks the ones that are filler, and hands their ids to
``set_edit`` as ``picks``.

An um is found the same way, and it is one thing however it got into the
transcript: a word the speech model wrote ("Um,", "uhh") or a sound heard
between two words that it wrote nothing for (``sounds.UM``). Asking for "um"
in any spelling lists both kinds under the one word. Each line says first,
in plain words, the two facts a reader decides by: whether the um opens a
sentence or sits inside one, and how much quiet lies before and after it.
An um with a clear pause beside it can go; one tucked between words holds
the sentence together. Which is which is for the reader: nothing here
removes an um or says that one should go.

The answer is packed text, one place a line, the way ``read_transcript``
packs the transcript: 112 places take about 12,000 characters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lumr_studio.edit import UM_TEXT, SaidOrder, SilenceIndex, clock, pick_id, why_not_clean
from lumr_studio.errors import StudioError
from lumr_studio.sounds import LAUGH, UM_WORDS, is_um_word
from lumr_studio.transcript import PHRASE_PAUSE_SECONDS, UM, _is_sentence_end, is_event, label_of, label_text, plain_text

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

# Where an um sits, as its line says it. It opens a sentence when it is the
# first thing said after a sentence end or after a pause the transcript
# prints (``PHRASE_PAUSE_SECONDS``), or the first thing said at all. Every
# other um has words running on either side of it.
OPENS, INSIDE = "opens", "inside"
POSITION_TEXT = {OPENS: "OPENS A SENTENCE", INSIDE: "INSIDE A SENTENCE"}


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
    position: str = ""  # for an um: OPENS or INSIDE; "" for every other word
    sound: bool = False  # an um the speech model wrote no word for

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
        if len(parts) == 1 and parts[0] in UM_WORDS:
            parts = [UM]  # every spelling of an um asks for every um
        if parts not in out:
            out.append(parts)
    return out


def _quote(tokens: list[dict[str, Any]], labels: list[dict[str, Any]] | None = None, lengths: bool = False) -> str:
    """Words and sounds as one line of text. A sound reads ``(um)``, ``(laugh)`` or ``(sound)``; with ``lengths``, as the transcript prints it."""
    out = []
    for w in tokens:
        if not is_event(w):
            out.append(str(w["word"]).strip())
            continue
        kind = (label_of(w, labels) or {}).get("kind")
        if lengths and kind == UM:
            out.append(label_text(label_of(w, labels)))
        else:
            out.append(UM_TEXT if kind == UM else "(laugh)" if kind == LAUGH else "(sound)")
    return " ".join(out)


def _is_um(token: dict[str, Any], labels: list[dict[str, Any]] | None) -> bool:
    """Whether a word or sound of the transcript is an um: a hesitation as the speech model spelled it, or an um sound."""
    return is_um_word(token) or (label_of(token, labels) or {}).get("kind") == UM


def _position(tokens: list[dict[str, Any]], ums: list[bool], i: int) -> str:
    """Where the um at ``tokens[i]`` sits: ``OPENS`` a sentence or ``INSIDE`` one. ``ums`` says which tokens are ums.

    Ums said one after another are one hesitation, so the first one's place
    is the place of all of them. It opens a sentence when nothing is said
    before it, when a pause of ``PHRASE_PAUSE_SECONDS`` or more lies between
    it and the word before, or when that word ends a sentence. A laugh or
    another sound between the two is not a word and ends no sentence.
    """
    first = i
    while first > 0 and ums[first - 1]:
        first -= 1
    if first == 0:
        return OPENS
    run = tokens[first - 1:i + 1]
    if any(float(b["start"]) - float(a["end"]) >= PHRASE_PAUSE_SECONDS for a, b in zip(run, run[1:])):
        return OPENS
    before = next((w for w in reversed(tokens[:first]) if not is_event(w)), None)
    return OPENS if before is None or _is_sentence_end(before) else INSIDE


def find_places(
    words: list[dict[str, Any]], phrases: list[list[str]], silences: list[Any], duration: float,
    codes: dict[tuple[float, float], int] | None = None, labels: list[dict[str, Any]] | None = None,
) -> list[Place]:
    """Every place each phrase is said, in time order.

    ``codes`` says why a word is removed already, by its ``(start, end)``
    rounded to the millisecond, with the values the page uses in ``words[]``.
    ``labels`` are the sound labels (``sounds.classify_events``). The phrase
    ``um`` finds every hesitation the transcript holds as a word, in any
    spelling, and with ``labels`` every um sound too; each of those places
    says where it sits (``_position``). No other sound is ever a place.
    """
    said = SaidOrder.of(words)
    quiet = SilenceIndex(silences)
    plain = [None if is_event(w) else plain_text(str(w["word"])) for w in said.tokens]
    ums = [_is_um(w, labels) for w in said.tokens]
    codes = codes or {}
    places = []
    for parts in phrases:
        n = len(parts)
        for i in range(len(plain) - n + 1):
            um = parts == [UM] and ums[i]
            if plain[i:i + n] != parts and not um:
                continue
            picked = range(i, i + n)
            chosen = said.tokens[i:i + n]
            start = round(min(float(w["start"]) for w in chosen), 3)
            end = round(max(float(w["end"]) for w in chosen), 3)
            floor, ceiling = said.neighbours(picked, duration)
            owners = [codes.get((round(float(w["start"]), 3), round(float(w["end"]), 3)), 0) for w in chosen]
            places.append(Place(
                phrase=" ".join(parts), id=pick_id(start, end), start=start, end=end,
                before=_quote(said.tokens[max(0, i - CONTEXT_WORDS):i], labels), said=_quote(chosen, labels, lengths=True),
                after=_quote(said.tokens[i + n:i + n + CONTEXT_WORDS], labels),
                quiet_before=round(max(0.0, start - floor), 2), quiet_after=round(max(0.0, ceiling - end), 2),
                out_by=OUT_BY.get(owners[0], "") if all(owners) else "",
                why_not_clean=why_not_clean(picked, said, quiet, duration),
                position=_position(said.tokens, ums, i) if um else "",
                sound=um and is_event(chosen[0]),
            ))
    return sorted(places, key=lambda p: (p.start, p.end))


def _short(seconds: float) -> str:
    """A length to two decimals without the leading zero: ``.04``, ``1.20``."""
    return f"{seconds:.2f}".removeprefix("0")


def _line(place: Place) -> str:
    notes = (f" OUT:{place.out_by}" if place.out_by else "") + ("" if place.clean else f" NOT CLEAN: {place.why_not_clean}")
    if place.position:
        # An um: where it sits and the quiet either side come first, in words, since they decide it.
        return (
            f"{place.id} {clock(place.start)} {POSITION_TEXT[place.position]}, "
            f"QUIET {place.quiet_before:.2f}s BEFORE, {place.quiet_after:.2f}s AFTER: "
            f"{place.before} [{place.said}] {place.after}{notes}"
        )
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
# Added when "um" was asked for.
HOW_TO_READ_UMS = (
    "An um line says, right after the clock, where the um sits, then QUIET with the seconds of quiet before it "
    f"and after it, in place of q. {POSITION_TEXT[OPENS]}: it is the first thing said after a sentence end, or "
    f"after a pause long enough for read_transcript to print one. {POSITION_TEXT[INSIDE]}: words run on either "
    "side of it. "
    "[(um 0.5s)] is an um heard between two words that the transcript has no word for, with its length; it is "
    "picked by its id like a word. No rule takes an um out: only your pick does, so read each one."
)


def _count_line(count: dict[str, Any]) -> str:
    """The first line for one phrase: how often it is said and how many of those places are clean to cut."""
    rest = f"{count['clean']} clean to cut, {count['said'] - count['clean']} not, {count['already_out']} already out."
    if count["word"] != UM:
        return f"{count['word']}: said {count['said']} times, {rest}"
    return (
        f"{UM}: {count['said']} in this video, {count['opens']} open a sentence, {count['inside']} inside one. "
        f"{count['sounds']} of them are sounds with no word in the transcript. {rest}"
    )


def pack_places(places: list[Place], phrases: list[list[str]], *, start: float | None = None,
                budget_chars: int = PACK_BUDGET_CHARS) -> dict[str, Any]:
    """The places as packed text, with a count for each phrase.

    Returns ``{text, words: [{word, said, clean, already_out}], listed,
    next_start}``. The entry for ``um`` also carries ``opens``, ``inside``
    and ``sounds``: how many open a sentence, how many sit inside one, and
    how many are sounds the transcript has no word for. The counts are for
    the whole video. The lines start at ``start`` and stop when the budget
    is used up; the text then ends with ``NEXT <time>``, the ``start`` to
    pass to go on, else with ``END``.
    """
    counts = []
    for parts in phrases:
        phrase = " ".join(parts)
        mine = [p for p in places if p.phrase == phrase]
        count = {"word": phrase, "said": len(mine), "clean": sum(p.clean for p in mine),
                 "already_out": sum(bool(p.out_by) for p in mine)}
        if parts == [UM]:
            count.update(opens=sum(p.position == OPENS for p in mine), inside=sum(p.position == INSIDE for p in mine),
                         sounds=sum(p.sound for p in mine))
        counts.append(count)
    lines = [_count_line(c) for c in counts] + [HOW_TO_READ]
    if [UM] in phrases:
        lines.append(HOW_TO_READ_UMS)
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
