"""Read every join in an edit the way a viewer hears it, and flag the ones likely to sound wrong.

A join is where playback skips a cut: the last kept word before the cut now
sits right next to the first kept word after it. Claude places cuts from a
transcript and never hears the result, so a cut can leave a sentence with no
ending, open one with no beginning, take out a joke, or jam two words
together. This module reads each join from the words alone and says, with
source times, what to change. It never edits anything.

Each flagged join is written for two readers:

* ``note`` is for Claude, who can move a cut's start or end. It names the fix
  with source seconds Claude can pass straight back to ``set_edit``: a word's
  start to cut from that word on, a word's end to keep that word.
* ``why`` is for the creator on the treatment page, who can only put a cut
  back or leave it out. It says in plain words what the viewer will hear,
  with no times and no advice the creator can't act on.

A word counts as removed by the midpoint rule, the reading the render and the
packed transcript use (``transcript.word_is_cut``).

The creator's own cuts (``source`` ``you``) are read the way Claude's are. She
often takes one word out of a sentence, so a join inside one sentence says
that in its ``why`` and not that two sentences were joined.

A filler word Claude picked (``source`` ``pick``) is judged as lightly as an
automatic trim: it takes one filler word on purpose, and it is no row the
creator could act on.

Automatic trims are judged more lightly than Claude's cuts. They take pauses
and filler words on purpose, so a filler dropped mid-sentence or a pause
removed whole is the trim doing its job; only a trim that touches a joke, or
leaves a stutter, is flagged.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from typing import Any

from lumr_studio.edit import clock, clock_span
from lumr_studio.errors import StudioError
from lumr_studio.transcript import _is_clause_end, _is_sentence_end, is_event, word_is_cut

Word = dict[str, Any]
Label = dict[str, Any]

# ── Flags ─────────────────────────────────────────────────────────────────────

MID_SENTENCE_OUT = "mid_sentence_out"
MID_SENTENCE_IN = "mid_sentence_in"
SPLICE = "splice"
REMOVES_LAUGH = "removes_laugh"
CLIPS_BEAT = "clips_beat"
RE_ENTRY = "re_entry"
LONG_JUMP = "long_jump"
FRAGMENT = "fragment"
TIGHT = "tight"

# How much each flag costs the viewer. Rows are listed worst first so the few
# Claude reads first are the ones most worth fixing. A lost joke is what the
# creator objected to most; a splice is often a correct retake fix.
SEVERITY = {
    REMOVES_LAUGH: 6,
    MID_SENTENCE_OUT: 5,
    MID_SENTENCE_IN: 5,
    RE_ENTRY: 4,
    CLIPS_BEAT: 4,
    FRAGMENT: 3,
    LONG_JUMP: 3,
    TIGHT: 2,
    SPLICE: 1,
}
FLAGS = tuple(SEVERITY)

# Words that open a sentence by pointing back at what was just said. After a
# long cut the thing they point at is often gone.
BACK_POINTERS = frozenset({"that", "it", "this", "they", "which", "so", "and", "but", "because", "then"})
# Answer words point back at the line they answer. That line is usually the one
# just cut, however short the cut, so they need no minimum length.
ANSWER_WORDS = frozenset({"yes", "no", "yeah"})
# A back-pointer after a short cut usually still has its referent in view.
RE_ENTRY_MIN_SECONDS = 5.0
# Openers that don't start a sentence of their own: a sentence whose only
# removed words are these still begins cleanly at the next word. "So, before
# you..." with a comma is the speaker announcing a new part, not pointing back.
LEAD_INS = frozenset({"so", "and", "but", "um", "uh", "like", "well", "okay", "ok", "now", "oh"})
# Removing this much at once risks the viewer losing the thread.
LONG_JUMP_SECONDS = 20.0
# Kept speech shorter than either of these between two cuts sounds like a stutter.
FRAGMENT_SECONDS = 1.0
FRAGMENT_WORDS = 3
# Less silence than this between two joined words runs them together.
TIGHT_SECONDS = 0.10
# A pause at least this long counts as a beat the speaker gave. Tight only
# fires when a cut took one of these away: this transcript format gives most
# neighbouring words no gap at all (94% on the real footage), so zero silence
# between words that were already touching is ordinary speech.
REAL_PAUSE_SECONDS = 0.20
# A retake fix may leave one stray word (a filler) before the repeated take.
RETAKE_LEAD_WORDS = 1
# How far back or forward, in words, a splice fix looks for a clause edge.
SPLICE_REACH_WORDS = 6
# An automatic trim overlapping a beat by less than this is a rounding touch, not a trim.
BEAT_BITE_SECONDS = 0.05
# A search for a clean sentence to resume on looks this far past the cut.
RESUME_SEARCH_SECONDS = 20.0
# A transcript word shorter than this has no audible span of its own (the
# real transcript has words stamped with zero length), so removing it does not
# make a cut remove speech.
MIN_WORD_SECONDS = 0.02
# A suggested edge this close to the current one is no change at all.
SAME_EDGE_SECONDS = 0.05

# What the creator reads for each flag: what the viewer hears, in plain words,
# with no times and no fix (the creator can only put a cut back or leave it
# out). ``{who}`` is ``WHO_CUT`` or ``WHO_AUTO``; ``{word}`` the opener as
# said; ``{length}`` a length like "25 seconds". The creator never reads the
# word "trim", so an automatic one is "an automatic cut" here.
WHO_CUT = "This cut"
WHO_AUTO = "An automatic cut"
WHYS = {
    "mid_sentence_out": "The sentence before this cut never finishes.",
    "mid_sentence_in": "After this cut, the talk picks up in the middle of a sentence.",
    "splice": "This joins the start of one sentence to the end of another.",
    "splice_retake": "This takes out a restart and joins the sentence back up. Hear it once to be sure it sounds whole.",
    "splice_retake_and_more": "This takes out a restart, and a few words before it too.",
    "splice_inside": "This takes words out of the middle of a sentence. Hear it once to be sure it still sounds whole.",
    "removes_laugh": "{who} takes out a laugh.",
    "removes_punchline": "{who} takes out the punchline before a laugh.",
    "removes_pause_before_laugh": "{who} takes out the pause before a laugh, which can spoil the timing.",
    "removes_setup": "This cut takes out the setup of a joke.",
    "clips_beat": "An automatic cut shortens the pause around a laugh.",
    "re_entry": "After this cut, the next line starts with \"{word}\", which may point back at words that were cut.",
    "long_jump": "This cut takes out {length} at once, so a viewer may lose the thread.",
    "fragment": "Only a few words are left between this cut and the next one, which can sound choppy.",
    "tight": "The words either side of this cut run together, with no pause between them.",
}

# Kept words shown either side of a join, and removed words shown from each end.
CONTEXT_WORDS = 4
REMOVED_EDGE_WORDS = 4
MAX_LISTED = 25


# ── Reading the edit ──────────────────────────────────────────────────────────


@dataclass
class _Cut:
    start: float
    end: float
    source: str
    reason: str

    @property
    def auto(self) -> bool:
        """Judged lightly: an automatic trim, or a filler word Claude picked."""
        return self.source in ("auto", "pick")

    @property
    def by_creator(self) -> bool:
        return self.source == "you"


@dataclass
class _Edit:
    """The transcript indexed against the cuts, for fast join lookups."""

    cuts: list[_Cut]
    tokens: list[Word]              # every entry, words and events, sorted by start
    token_cut: list[int]            # per token: index of the cut removing it, or -1
    spoken: list[Word]              # the speech entries only
    spoken_cut: list[int]           # per spoken word: index of the cut removing it, or -1
    label_at: dict[float, Label]
    laughs: list[_Laugh] = field(default_factory=list)
    kept_spoken: list[int] = field(default_factory=list)
    kept_spoken_mids: list[float] = field(default_factory=list)
    kept_tokens: list[int] = field(default_factory=list)
    kept_token_mids: list[float] = field(default_factory=list)
    removed_by: dict[int, list[int]] = field(default_factory=dict)  # cut -> audible spoken indices


def _mid(w: Word) -> float:
    return (float(w["start"]) + float(w["end"])) / 2


def _norm(word: str) -> str:
    return re.sub(r"[^\w']", "", word.lower())


def _parse_cuts(cuts: list[dict[str, Any]], duration: float) -> list[_Cut]:
    """Cuts as ``_Cut``, in time order. Raises StudioError when they overlap or are malformed."""
    parsed = []
    for i, c in enumerate(cuts):
        try:
            start, end = float(c["start"]), float(c["end"])
        except (KeyError, TypeError, ValueError):
            raise StudioError(f"Cut {i} has no numeric start and end. Call set_edit to rebuild the edit.") from None
        if end <= start:
            raise StudioError(f"Cut {i} ends at {end} before it starts at {start}. Call set_edit to rebuild the edit.")
        parsed.append(_Cut(start, end, str(c.get("source", "claude")), str(c.get("reason", ""))))
    parsed.sort(key=lambda c: c.start)
    for a, b in zip(parsed, parsed[1:]):
        if b.start < a.end:
            raise StudioError(
                f"Cuts {a.start:.2f}-{a.end:.2f} and {b.start:.2f}-{b.end:.2f} overlap. "
                "Call set_edit to rebuild the edit; it merges overlapping cuts."
            )
    if parsed and parsed[-1].end > duration + SAME_EDGE_SECONDS:
        raise StudioError(
            f"A cut ends at {parsed[-1].end:.2f}, past the video's end ({duration:.2f}). "
            "The edit was made for a different video; call set_edit to rebuild it."
        )
    return parsed


def _cut_of(w: Word, cuts: list[_Cut], starts: list[float]) -> int:
    """Index of the cut that removes ``w`` by the midpoint rule, or -1."""
    i = bisect.bisect_right(starts, _mid(w)) - 1
    if i >= 0 and word_is_cut(w, [(cuts[i].start, cuts[i].end)]):
        return i
    return -1


def _read(cuts: list[dict[str, Any]], words: list[Word], duration: float, labels: list[Label] | None) -> _Edit:
    parsed = _parse_cuts(cuts, duration)
    starts = [c.start for c in parsed]
    tokens = sorted(words, key=lambda w: float(w["start"]))
    spoken = [w for w in tokens if not is_event(w)]
    e = _Edit(
        cuts=parsed,
        tokens=tokens,
        token_cut=[_cut_of(w, parsed, starts) for w in tokens],
        spoken=spoken,
        spoken_cut=[_cut_of(w, parsed, starts) for w in spoken],
        label_at={round(float(lb["start"]), 2): lb for lb in labels or []},
    )
    e.kept_spoken = [k for k, c in enumerate(e.spoken_cut) if c < 0]
    e.kept_spoken_mids = [_mid(spoken[k]) for k in e.kept_spoken]
    e.kept_tokens = [k for k, c in enumerate(e.token_cut) if c < 0]
    e.kept_token_mids = [_mid(tokens[k]) for k in e.kept_tokens]
    for k, c in enumerate(e.spoken_cut):
        if c >= 0 and _end(e, k) - _start(e, k) >= MIN_WORD_SECONDS:
            e.removed_by.setdefault(c, []).append(k)
    e.laughs = _laughs(e, labels or [])
    return e


def _kept_before(index: list[int], mids: list[float], t: float, n: int = 1) -> list[int]:
    """The last ``n`` kept indices whose midpoint is before ``t``."""
    k = bisect.bisect_left(mids, t)
    return index[max(0, k - n):k]


def _kept_from(index: list[int], mids: list[float], t: float, n: int = 1) -> list[int]:
    """The first ``n`` kept indices whose midpoint is at or after ``t``."""
    k = bisect.bisect_left(mids, t)
    return index[k:k + n]


def _t(x: float) -> str:
    return f"{x:.2f}"


def _cap(text: str) -> str:
    """``text`` with its first letter capitalised, so each part of a note reads as a sentence."""
    return text[:1].upper() + text[1:]


def _how_long(seconds: float) -> str:
    """A length for the creator: "25 seconds", or "1:05" from a minute up."""
    whole = round(seconds)
    return f"{whole} seconds" if whole < 60 else clock(whole)


def _start(e: _Edit, k: int) -> float:
    return float(e.spoken[k]["start"])


def _end(e: _Edit, k: int) -> float:
    return float(e.spoken[k]["end"])


def _text(e: _Edit, k: int) -> str:
    return e.spoken[k]["word"].strip()


# ── Sentences ─────────────────────────────────────────────────────────────────


def _sentence_start(e: _Edit, k: int) -> int:
    while k > 0 and not _is_sentence_end(e.spoken[k - 1]):
        k -= 1
    return k


def _sentence_end(e: _Edit, k: int) -> int:
    while k < len(e.spoken) - 1 and not _is_sentence_end(e.spoken[k]):
        k += 1
    return k


def _starts_sentence(e: _Edit, k: int) -> bool:
    """True when ``k`` opens a sentence, or only lead-in words ("So", "um") come before it."""
    return all(_norm(_text(e, j)) in LEAD_INS for j in range(_sentence_start(e, k), k))


def _points_back(e: _Edit, k: int) -> bool:
    word = _text(e, k)
    opener = _norm(word)
    if opener == "so" and word.rstrip("\"')").endswith(","):
        return False  # "So," announces a new part
    return opener in BACK_POINTERS or opener in ANSWER_WORDS


def _quote(e: _Edit, first: int, last: int, limit: int = 3) -> str:
    """Words ``first..last`` as a quote, only the last ``limit`` of them."""
    return '"' + " ".join(_text(e, j) for j in range(max(first, last - limit + 1), last + 1)) + '"'


def _opening(e: _Edit, k: int, limit: int = 3) -> str:
    return '"' + " ".join(_text(e, j) for j in range(k, min(len(e.spoken), k + limit))) + '"'


# ── One join ──────────────────────────────────────────────────────────────────


@dataclass
class _Join:
    """One cut seen from both sides, with the flags it collects (flag -> note)."""

    i: int
    cut: _Cut
    removed: list[int]
    p: int | None      # spoken index of the last kept word before the cut
    q: int | None      # spoken index of the first kept word after it
    flags: dict[str, str] = field(default_factory=dict)   # flag -> note, for Claude
    whys: dict[str, str] = field(default_factory=dict)    # flag -> why, for the creator

    def flag(self, name: str, note: str, why: str) -> None:
        """Raise ``name`` with Claude's ``note`` and the creator's ``why``."""
        self.flags[name] = note
        self.whys[name] = why

    @property
    def seconds(self) -> float:
        return self.cut.end - self.cut.start


def _repeat_at(e: _Edit, j: _Join) -> int | None:
    """Where in the removed words the first words after the cut are said already, or None.

    A restart: the speaker says a phrase, stops, and says it again. The
    removed words then contain the kept opening after the cut.
    """
    if j.q is None or not j.removed:
        return None
    cut = [_norm(_text(e, k)) for k in j.removed]
    after = [_norm(_text(e, k)) for k in range(j.q, min(len(e.spoken), j.q + 2))]
    if len(cut) <= 2 and cut[0] == after[0]:
        return 0
    n = len(after)
    if n < 2:
        return None
    return next((i for i in range(len(cut) - n + 1) if cut[i:i + n] == after), None)


def _drop_or(options: list[str]) -> str:
    return _cap(", or ".join(options)) if options else "Drop the cut"


def _first_edge(e: _Edit, ks: list[int]) -> int | None:
    """The first of ``ks`` that ends a clause or a sentence."""
    return next((k for k in ks if _is_clause_end(e.spoken[k]) or _is_sentence_end(e.spoken[k])), None)


def _note_out(e: _Edit, j: _Join) -> str:
    s0 = _sentence_start(e, j.p)
    options = []
    if _start(e, s0) < j.cut.start - SAME_EDGE_SECONDS:
        options.append(f"start at {_t(_start(e, s0))} to drop it")
    edge = _first_edge(e, j.removed[:SPLICE_REACH_WORDS])
    if edge is not None and _end(e, edge) < j.cut.end - SAME_EDGE_SECONDS:
        options.append(f"start at {_t(_end(e, edge))} to end on {_quote(e, j.removed[0], edge, 2)}")
    return f"Sentence left unfinished. {_drop_or(options)}."


def _note_in(e: _Edit, j: _Join) -> str:
    s0 = _sentence_start(e, j.q)
    nxt = _sentence_end(e, j.q) + 1
    options = []
    if nxt < len(e.spoken):
        options.append(f"end at {_t(_start(e, nxt))} to skip to the next sentence")
    if s0 in j.removed and _start(e, s0) > j.cut.start + SAME_EDGE_SECONDS:
        options.append(f"end at {_t(_start(e, s0))} to keep it whole")
    return f"Opens mid-sentence. {_drop_or(options)}."


def _note_splice(e: _Edit, j: _Join) -> str:
    at = _repeat_at(e, j)
    if at is not None and at <= RETAKE_LEAD_WORDS:
        return f"Retake fix: the cut words repeat {_opening(e, j.q, 2)}. Fine as is."
    if at is not None:
        first = j.removed[at]
        return (f"Also cuts {_quote(e, j.removed[0], j.removed[at - 1])} before the retake. "
                f"Start at {_t(_start(e, first))} so only the repeat goes.")
    options = []
    for k in range(j.p, max(-1, j.p - SPLICE_REACH_WORDS), -1):
        if _is_clause_end(e.spoken[k]) or _is_sentence_end(e.spoken[k]):
            if k < j.p:
                options.append(f"start at {_t(_start(e, k + 1))} (after \"{_text(e, k)}\")")
            break
    for k in range(j.q, min(len(e.spoken) - 1, j.q + SPLICE_REACH_WORDS)):
        if _is_clause_end(e.spoken[k]) or _is_sentence_end(e.spoken[k]):
            options.append(f"end at {_t(_start(e, k + 1))} (after \"{_text(e, k)}\")")
            break
    if not options:
        return "Joins two half sentences. Unless it's a false start, drop the cut."
    return f"Joins two half sentences. Unless it's a false start, {' or '.join(options)}."


def _why_splice(e: _Edit, j: _Join) -> str:
    at = _repeat_at(e, j)
    if at is not None and at <= RETAKE_LEAD_WORDS:
        return WHYS["splice_retake"]
    if at is not None:
        return WHYS["splice_retake_and_more"]
    if j.cut.by_creator and j.p is not None and j.q is not None \
            and _sentence_start(e, j.p) == _sentence_start(e, j.q):
        return WHYS["splice_inside"]
    return WHYS["splice"]


def _sentence_flags(e: _Edit, j: _Join) -> None:
    """mid_sentence_out, mid_sentence_in, splice.

    A cut that only takes lead-in and filler words gets none. A join inside
    one sentence is a splice. A retake's kept take stands where the first
    take started, so that is where the sentence position is read.
    """
    if all(_norm(_text(e, k)) in LEAD_INS for k in j.removed):
        return
    opens_at = j.q
    at = _repeat_at(e, j)
    if at is not None and at <= RETAKE_LEAD_WORDS:
        opens_at = j.removed[at]
    open_out = j.p is not None and not _is_sentence_end(e.spoken[j.p])
    open_in = opens_at is not None and not _starts_sentence(e, opens_at)
    if j.p is not None and j.q is not None and _sentence_start(e, j.p) == _sentence_start(e, j.q):
        open_out = open_in = True  # no sentence end anywhere between the joined words
    if open_out and not open_in:
        j.flag(MID_SENTENCE_OUT, _note_out(e, j), WHYS[MID_SENTENCE_OUT])
    elif open_in and not open_out:
        j.flag(MID_SENTENCE_IN, _note_in(e, j), WHYS[MID_SENTENCE_IN])
    elif open_out and open_in:
        j.flag(SPLICE, _note_splice(e, j), _why_splice(e, j))


def _resume_point(e: _Edit, q: int) -> int | None:
    """The first sentence start after ``q``'s sentence that does not point back, within reach."""
    k = _sentence_end(e, q) + 1
    limit = _start(e, q) + RESUME_SEARCH_SECONDS
    while k < len(e.spoken) and _start(e, k) <= limit:
        if not _points_back(e, k):
            return k
        k = _sentence_end(e, k) + 1
    return None


def _last_sentence_start(e: _Edit, j: _Join) -> float | None:
    """Where the last removed sentence starts, when the cut holds more than that one."""
    s0 = _sentence_start(e, j.removed[-1])
    return _start(e, s0) if s0 in j.removed and s0 != j.removed[0] else None


def _keep_last_sentence(e: _Edit, j: _Join) -> str:
    at = _last_sentence_start(e, j)
    return f"end at {_t(at)} to keep the last cut sentence" if at is not None else "drop the cut"


def _plain_word(e: _Edit, k: int) -> str:
    """Word ``k`` as said, without the punctuation around it."""
    return re.sub(r"^[^\w']+|[^\w']+$", "", _text(e, k))


def _re_entry(e: _Edit, j: _Join) -> None:
    if j.q is None or not _starts_sentence(e, j.q) or not _points_back(e, j.q):
        return
    if _norm(_text(e, j.q)) not in ANSWER_WORDS and j.seconds < RE_ENTRY_MIN_SECONDS:
        return
    options = [_keep_last_sentence(e, j)]
    resume = _resume_point(e, j.q)
    if resume is not None:
        options.append(f"end at {_t(_start(e, resume))} to resume on {_opening(e, resume)}")
    j.flag(
        RE_ENTRY,
        f"The next line opens with {_opening(e, j.q, 1)}, which points back at cut words. "
        f"{_cap(', or '.join(options))}.",
        WHYS[RE_ENTRY].format(word=_plain_word(e, j.q)),
    )


def _long_jump(e: _Edit, j: _Join) -> None:
    if j.seconds < LONG_JUMP_SECONDS:
        return
    at = _last_sentence_start(e, j)
    fix = f"End at {_t(at)} to keep the last cut sentence as a bridge" if at is not None \
        else "Split it so a line of the thread stays"
    j.flag(LONG_JUMP, f"Removes {j.seconds:.0f}s at once. {fix}.",
           WHYS[LONG_JUMP].format(length=_how_long(j.seconds)))


# ── Laughs ────────────────────────────────────────────────────────────────────


@dataclass
class _Laugh:
    """One laugh label with the spans around it that carry the joke's timing."""

    start: float
    end: float
    punch_start: float
    punch_end: float
    has_punchline: bool
    name: str                       # "laugh at 12.30" or "possible laugh at 12.30"
    setup_word: int | None          # spoken index of the last word before the punchline
    beats: list[tuple[float, float]]  # pause before the punchline, before the laugh, after it


def _laughs(e: _Edit, labels: list[Label]) -> list[_Laugh]:
    ends = [float(w["end"]) for w in e.spoken]
    starts = [float(w["start"]) for w in e.tokens]
    out = []
    for lb in labels:
        if lb.get("kind") != "laugh":
            continue
        ls, le = float(lb["start"]), float(lb["end"])
        punch = lb.get("punchline")
        ps, pe = (float(punch[0]), float(punch[1])) if punch else (ls, ls)
        k = max((i for i, t in enumerate(ends) if t <= ps - 1e-3), default=None)
        n = bisect.bisect_left(starts, le - 1e-6)
        after = starts[n] if n < len(starts) else le
        out.append(_Laugh(
            start=ls, end=le, punch_start=ps, punch_end=pe, has_punchline=punch is not None,
            name=f"{'laugh' if lb.get('confidence') == 'likely' else 'possible laugh'} at {_t(ls)}",
            setup_word=k,
            beats=[((ends[k] if k is not None else ps), ps), (pe, ls), (le, after)],
        ))
    return out


def _overlap(a: float, b: float, c: float, d: float) -> float:
    return max(0.0, min(b, d) - max(a, c))


def _keep_joke(a: float, b: float, keep_from: float, keep_to: float) -> str:
    options = []
    if keep_from > a + SAME_EDGE_SECONDS:
        options.append(f"end the cut at {_t(keep_from)}")
    if keep_to < b - SAME_EDGE_SECONDS:
        options.append(f"start it at {_t(keep_to)}")
    return _cap(" or ".join(options)) if options else "Drop the cut"


def _ask_to_keep(lf: _Laugh) -> str:
    """The fix for an automatic trim at a joke: only the creator's keep drops a trim there."""
    return (f"Ask the creator to keep {clock_span(lf.punch_start, lf.end)} on the page, "
            "which drops the trims there")


def _laugh_flags(e: _Edit, j: _Join) -> None:
    """removes_laugh and clips_beat, from the sound labels."""
    a, b = j.cut.start, j.cut.end
    removed = set(j.removed)
    for lf in e.laughs:
        keep = f"{_t(lf.punch_start)}-{_t(lf.end)}"
        takes_laugh = _overlap(a, b, lf.start, lf.end) >= (lf.end - lf.start) / 2
        takes_punch = lf.has_punchline and any(
            lf.punch_start <= _mid(e.spoken[k]) <= lf.punch_end for k in removed)
        between = _overlap(a, b, lf.punch_end, lf.start)
        takes_between = between > BEAT_BITE_SECONDS and (
            not j.cut.auto or between >= (lf.start - lf.punch_end) * 0.9)
        if takes_laugh or (takes_punch and not j.cut.auto) or takes_between:
            what = "the" if takes_laugh else "the punchline of the" if takes_punch else "the pause before the"
            fix = (_ask_to_keep(lf) if j.cut.auto
                   else f"{_keep_joke(a, b, lf.punch_start, lf.end)} to keep {keep}")
            why = WHYS["removes_laugh" if takes_laugh else "removes_punchline" if takes_punch
                       else "removes_pause_before_laugh"]
            j.flag(REMOVES_LAUGH, f"Takes {what} {lf.name}. {fix}.",
                   why.format(who=WHO_AUTO if j.cut.auto else WHO_CUT))
            return

        touches = takes_punch or any(_overlap(a, b, x, y) > BEAT_BITE_SECONDS for x, y in lf.beats)
        if j.cut.auto and touches:
            j.flag(CLIPS_BEAT, f"Trim shifts the timing of the {lf.name}. {_ask_to_keep(lf)}.",
                   WHYS[CLIPS_BEAT])
            return

        # A Claude cut that runs up to the punchline takes its setup. A retake
        # fix there is fine: the kept take is the setup.
        if (lf.has_punchline and not j.cut.auto and lf.setup_word in removed
                and _repeat_at(e, j) is None):
            j.flag(REMOVES_LAUGH,
                   f"Cuts the setup of the joke at {_t(lf.punch_start)} ({lf.name}). {_cap(_keep_last_sentence(e, j))}.",
                   WHYS["removes_setup"])
            return


# ── Fragments and tight joins ─────────────────────────────────────────────────


def _fragment(e: _Edit, j: _Join) -> None:
    """Short kept speech between this cut and the next, when both remove speech."""
    if j.i + 1 >= len(e.cuts):
        return
    nxt = e.cuts[j.i + 1]
    if not j.removed or not e.removed_by.get(j.i + 1) or (j.cut.auto and nxt.auto):
        return  # a pause trim beside kept words lets them flow on: no stutter
    kept = [k for k in e.kept_spoken if j.cut.end <= _mid(e.spoken[k]) < nxt.start]
    gap = nxt.start - j.cut.end
    if kept and (gap < FRAGMENT_SECONDS or len(kept) < FRAGMENT_WORDS):
        j.flag(
            FRAGMENT,
            f"Keeps only {_quote(e, kept[0], kept[-1], FRAGMENT_WORDS + 1)} ({gap:.1f}s) before cut #{j.i + 1}. "
            f"End at {_t(nxt.end)} to cut it too, or keep more around it.",
            WHYS[FRAGMENT],
        )


def _tight(e: _Edit, j: _Join) -> None:
    """Under TIGHT_SECONDS of silence left where the source had a real pause."""
    if j.cut.auto:
        return  # the pacing layer removes whole pauses by design; see check_joins' summary line
    lo = e.cuts[j.i - 1].end if j.i > 0 else 0.0
    hi = e.cuts[j.i + 1].start if j.i + 1 < len(e.cuts) else float("inf")
    left = [k for k in _kept_before(e.kept_tokens, e.kept_token_mids, j.cut.start) if _mid(e.tokens[k]) >= lo]
    right = [k for k in _kept_from(e.kept_tokens, e.kept_token_mids, j.cut.end) if _mid(e.tokens[k]) < hi]
    if not left or not right:
        return
    pw, nw = e.tokens[left[0]], e.tokens[right[0]]
    silence = max(0.0, j.cut.start - float(pw["end"])) + max(0.0, float(nw["start"]) - j.cut.end)
    if silence >= TIGHT_SECONDS:
        return
    # The pauses the speaker gave on each side, before the cut took them.
    gone = [t for t in e.tokens[left[0] + 1:right[0]]]
    pauses = ([float(gone[0]["start"]) - float(pw["end"]), float(nw["start"]) - float(gone[-1]["end"])]
              if gone else [float(nw["start"]) - float(pw["end"])])
    if max(pauses) < REAL_PAUSE_SECONDS:
        return
    fix = _tight_fix(e, j)
    if fix is None:
        return  # nothing in reach would help, and a flag without a fix only costs attention
    j.flag(TIGHT,
           f"Only {silence:.2f}s left between \"{_token_text(e, pw)}\" and \"{_token_text(e, nw)}\" "
           f"where the speaker paused {max(pauses):.1f}s. {_cap(fix)}.",
           WHYS[TIGHT])


def _tight_fix(e: _Edit, j: _Join) -> str | None:
    """Move an edge to a removed-word boundary that has a real pause, or None when none is near."""
    for k in j.removed[:SPLICE_REACH_WORDS]:
        if k + 1 < len(e.spoken) and _start(e, k + 1) - _end(e, k) >= REAL_PAUSE_SECONDS:
            return f"start at {_t(_end(e, k))} to keep the pause after \"{_text(e, k)}\""
    for k in reversed(j.removed[-SPLICE_REACH_WORDS:]):
        if k > 0 and _start(e, k) - _end(e, k - 1) >= REAL_PAUSE_SECONDS:
            return f"end at {_t(_start(e, k))} to keep the pause before \"{_text(e, k)}\""
    return None


# ── Rows ──────────────────────────────────────────────────────────────────────


def _token_text(e: _Edit, w: Word) -> str:
    if not is_event(w):
        return w["word"].strip()
    label = e.label_at.get(round(float(w["start"]), 2))
    if label and label.get("kind") == "laugh":
        return "(laugh)" if label.get("confidence") == "likely" else "(laugh?)"
    return "(sound)"


def _context(e: _Edit, ks: list[int]) -> str:
    return " ".join(_token_text(e, e.tokens[k]) for k in ks)


def _removed_text(e: _Edit, removed: list[int]) -> str:
    words = [_text(e, k) for k in removed]
    if len(words) <= 2 * REMOVED_EDGE_WORDS:
        return " ".join(words)
    return " ".join(words[:REMOVED_EDGE_WORDS]) + " ... " + " ".join(words[-REMOVED_EDGE_WORDS:])


def _analyse(e: _Edit, i: int) -> _Join:
    cut = e.cuts[i]
    before = _kept_before(e.kept_spoken, e.kept_spoken_mids, cut.start)
    after = _kept_from(e.kept_spoken, e.kept_spoken_mids, cut.end)
    j = _Join(i=i, cut=cut, removed=e.removed_by.get(i, []),
              p=before[0] if before else None, q=after[0] if after else None)
    if e.laughs:
        _laugh_flags(e, j)
    if j.removed and not cut.auto:
        _sentence_flags(e, j)
        _re_entry(e, j)
        _long_jump(e, j)
    _fragment(e, j)
    _tight(e, j)
    return j


def _to_row(e: _Edit, j: _Join) -> dict[str, Any]:
    flags = sorted(j.flags, key=lambda f: -SEVERITY[f])
    return {
        "cut": j.i,
        "start": round(j.cut.start, 2),
        "end": round(j.cut.end, 2),
        "seconds": round(j.seconds, 2),
        "source": j.cut.source,
        "reason": j.cut.reason,
        "removed_words": len(j.removed),
        "before": _context(e, _kept_before(e.kept_tokens, e.kept_token_mids, j.cut.start, CONTEXT_WORDS)),
        "removed": _removed_text(e, j.removed),
        "after": _context(e, _kept_from(e.kept_tokens, e.kept_token_mids, j.cut.end, CONTEXT_WORDS)),
        "flags": flags,
        "note": j.flags[flags[0]] if flags else "",
        "why": j.whys[flags[0]] if flags else "",
    }


def _rows(e: _Edit) -> list[dict[str, Any]]:
    return [_to_row(e, _analyse(e, i)) for i in range(len(e.cuts))]


def join_rows(
    cuts: list[dict[str, Any]],
    words: list[Word],
    duration: float,
    *,
    labels: list[Label] | None = None,
) -> list[dict[str, Any]]:
    """One join report row for EVERY cut, flagged or not, in time order. The treatment page uses this.

    ``cuts`` are the saved edit's cuts (``{start, end, reason, source}``),
    ``words`` the whole transcript, ``labels`` the sound labels or None.
    Without labels the laugh flags are never raised. ``cut`` in each row is
    the cut's position in time order. Raises StudioError when the cuts overlap
    or run past ``duration``.
    """
    return _rows(_read(cuts, words, duration, labels))


# ── The summary Claude reads ──────────────────────────────────────────────────


def _rank(row: dict[str, Any]) -> tuple[Any, ...]:
    """Worst first: the most serious flag, then the flag total, retakes last, then time."""
    sev = [SEVERITY[f] for f in row["flags"]]
    return (-max(sev), -sum(sev), row["note"].startswith("Retake"), row["start"])


def _row_text(row: dict[str, Any]) -> str:
    before = f"...{row['before']}" if row["before"] else "(no words before)"
    after = f"{row['after']}..." if row["after"] else "(no words after)"
    return (
        f"#{row['cut']} {row['start']:.2f}-{row['end']:.2f} -{row['seconds']:.1f}s "
        f"{row['source']} [{', '.join(row['flags'])}]\n"
        f"   {before} | {after}\n"
        f"   fix: {row['note']}"
    )


def check_joins(
    cuts: list[dict[str, Any]],
    words: list[Word],
    duration: float,
    *,
    labels: list[Label] | None = None,
    max_listed: int = MAX_LISTED,
) -> dict[str, Any]:
    """Read every join the way a viewer hears it and flag the ones likely to sound wrong.

    Returns {"joins": n, "flagged": k, "clean": n - k,
             "by_flag": {flag: count},
             "rows": [join report row, flagged only, worst first, at most max_listed],
             "not_listed": m,
             "text": the same rows as compact plain text for the model}
    """
    if max_listed < 0:
        raise StudioError(f"max_listed {max_listed} is negative. Pass 0 or more.")
    e = _read(cuts, words, duration, labels)
    rows = _rows(e)
    flagged = sorted((r for r in rows if r["flags"]), key=_rank)
    listed = flagged[:max_listed]
    by_flag = {f: n for f in FLAGS if (n := sum(f in r["flags"] for r in flagged))}

    head = f"{len(rows)} joins, {len(flagged)} flagged, {len(rows) - len(flagged)} clean."
    if by_flag:
        head += " " + ", ".join(f"{f} {n}" for f, n in by_flag.items()) + "."
    lines = [head]
    if not labels:
        lines.append("No sound labels, so laughs were not checked.")
    lines += [_row_text(r) for r in listed]
    if len(flagged) > len(listed):
        lines.append(f"+{len(flagged) - len(listed)} more flagged, not listed.")
    return {
        "joins": len(rows),
        "flagged": len(flagged),
        "clean": len(rows) - len(flagged),
        "by_flag": by_flag,
        "rows": listed,
        "not_listed": len(flagged) - len(listed),
        "text": "\n".join(lines),
    }
