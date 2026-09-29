"""Validate Claude's proposed cuts and save the edit.

Claude's cuts are CONTENT cuts: they remove whole words and sentences on
purpose. Each one is snapped to word boundaries with the midpoint rule (the
convention the editor uses, ``isWordRemoved`` in mobile/src/logic/inserts.js):
a word whose midpoint lies inside the cut is removed, every other word is kept.
Each edge then sits in the gap beside the nearest kept word, leaving the
natural tail and lead-in ClipForge gives kept material
(``longform.pad_keep_ranges_into_gaps``), nudged into measured silence when the
gap has some.

The placed cuts go through ``render.finalize_removed_ranges`` as verbatim
(user) spans, and ``auto_tighten`` micro-cuts go through it as auto spans with
full word safety, exactly the split the editor makes. What is saved is the
finalized removed set, so the render plays what was saved. The edit recipe
checks the finished set last (``keep_words_whole``): no automatic removal
starts or ends inside a word or a sound of the transcript.

Two kinds of span are off limits. ``keeps`` are spans the creator restored on
the review page: a cut of Claude's that overlaps one is rejected, and an
automatic trim that overlaps one is dropped. ``protected`` are punchlines and
laughs: automatic trims that overlap one are dropped, since a trim there
changes the timing of a joke. Claude may still cut inside a protected span on
purpose; the join check flags it.

Each of Claude's cuts carries a ``kind``, one of ``CUT_KINDS``, so the creator
sees them grouped the way Claude meant them. The kind comes from Claude, never
from reading the reason: a guess from keywords is how a joke setup gets
called a repeat. A cut sent without one is ``other``.

The creator cuts words by hand on the page. Those cuts (``creator_cuts``) are
hers, not Claude's: each removes exactly the words she picked, is never
widened to a sentence, and is never stopped by a kept span. Words she brought
back one by one (``exact_keeps``) are the other side of that: a cut of
Claude's splits around them and an automatic trim leaves them in.

Claude also picks single filler words, by reading (``picks``). No rule can
tell "I like jazz" from "into like three sections"; a reader can. A pick is
no row of Claude's cuts: the creator has one switch for all of them and a
double-click for each. Each is placed the way her own word cuts are, by
``place_creator_cut``, the one way a word is cut. A pick that would clip the
word beside it is left in (``why_not_clean``).
"""

from __future__ import annotations

import bisect
import json
import logging
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from lumr_studio.engine.audio_boundaries import Silence
from lumr_studio.engine.longform import pad_keep_ranges_into_gaps
from lumr_studio.engine.render import finalize_removed_ranges, kept_segments

from lumr_studio import word_times as times
from lumr_studio.autocuts import SHORTEST_PIECE_SECONDS, TRIM_KINDS, Said, around, span_without, take_mark
from lumr_studio.errors import StudioError
from lumr_studio.project import Project, now_iso, write_json_atomic
from lumr_studio.timeline import edited_duration

log = logging.getLogger(__name__)

EDIT_VERSION = 1
# An end this close past the video end is treated as the end, not rejected.
END_TOLERANCE = 0.05
# Edge moves smaller than this are rounding, not adjustments worth reporting.
ADJUST_EPS = 0.001
# A requested cut shorter than this after placement removes nothing worth removing.
MIN_CUT_SECONDS = SHORTEST_PIECE_SECONDS
# get_edit lists at most this many automatic cuts when asked for them.
AUTO_CUTS_SHOWN = 50

# The kinds of cut Claude chooses, as the creator reads them, in the order shown.
CUT_KINDS: dict[str, str] = {
    "repeat": "Said twice",
    "false_start": "False starts",
    "off_topic": "Off topic",
    "other": "Other cuts",
}
DEFAULT_CUT_KIND = "other"
# Who made a cut, as saved in each cut's ``source``. ``pick`` is a filler
# word Claude picked by reading.
BY_CLAUDE, BY_CREATOR, BY_AUTO, BY_PICK = "claude", "you", "auto", "pick"
# When several made one removed span, the first of these that did is its source.
SOURCE_ORDER = (BY_CLAUDE, BY_CREATOR, BY_PICK, BY_AUTO)
# The filler words Claude picks have one switch on the page.
PICK_KIND = "likes"
PICK_LABEL = "Filler likes"
PICK_REASON = "A filler word Claude picked"
# Every switch under Take out, in the order shown: one for each kind of
# automatic trim, then the one over Claude's picks. This is the one list of
# what the creator can switch. ``TRIM_KINDS`` is the part of it the pace
# plans; ``CUT_KINDS`` below are the groups of Claude's cuts, which are rows
# she puts back one by one and never a switch.
SWITCHES: dict[str, str] = {**TRIM_KINDS, PICK_KIND: PICK_LABEL}
# Ids of the creator's own cuts and of the spans she kept start with these.
CREATOR_ROW_PREFIX, KEEP_PREFIX = "y", "k"
# Where a saved edit says how the measured word times it was placed on were made.
MADE_AS = "word_times_made_as"
# Set on a loaded edit whose spans were moved onto the word times the project
# has now: the times they were placed on before. Its ``cuts`` are still the
# ones saved, to be made again by the recipe. Never saved.
MOVED_FROM = "moved_from_word_times"
# The creator's own cuts have one group on the page and need no reason from her.
CREATOR_KIND = "yours"
CREATOR_KIND_LABEL = "Your cuts"
CREATOR_REASON = "Cut by you"
# A requested cut saved with this flag may cut into a kept span. set_edit
# sets it on every cut when Claude passes override_keeps.
OVERRIDE_FLAG = "override_keeps"


# ── Word-safe placement of one content cut ────────────────────────────────────


class CutRejected(Exception):
    """A single cut that cannot be made word-safe. The message says why."""


def _mid(w: dict[str, Any]) -> float:
    return (float(w["start"]) + float(w["end"])) / 2


def _label(w: dict[str, Any]) -> str:
    return f"'{w['word'].strip()}' ({float(w['start']):.2f}-{float(w['end']):.2f})"


def _in_silence(t: float, silences: list[Silence]) -> bool:
    return any(s.start <= t <= s.end for s in silences)


def _toward_silence(t: float, lo: float, hi: float, silences: list[Silence]) -> float:
    """Move ``t`` to the nearest measured-silence point inside ``[lo, hi]``.

    Leaves ``t`` alone when it is already in silence or no silence overlaps the gap.
    """
    if hi <= lo or _in_silence(t, silences):
        return t
    best: float | None = None
    for sil in silences:
        a, b = max(lo, sil.start), min(hi, sil.end)
        if b < a:
            continue
        point = min(max(t, a), b)
        if best is None or abs(point - t) < abs(best - t):
            best = point
    return t if best is None else best


def _reject_no_whole_word(a: float, b: float, touched: list[dict[str, Any]]) -> CutRejected:
    words = ", ".join(_label(w) for w in touched)
    first, last = touched[0], touched[-1]
    return CutRejected(
        f"removes no whole word: it only clips {words}. "
        f"Nearest word boundaries are {float(first['start']):.2f} and {float(last['end']):.2f}. "
        f"To remove those words cut {float(first['start']):.2f}-{float(last['end']):.2f}; "
        "to keep them, drop this cut."
    )


def place_cut(
    a: float,
    b: float,
    tokens: list[dict[str, Any]],
    duration: float,
    silences: list[Silence] | None = None,
) -> tuple[float, float]:
    """Word-safe ``(start, end)`` for the requested cut ``[a, b]``.

    ``tokens`` are all transcript entries (words and events) sorted by start.
    A cut that lies wholly in a pause is returned unchanged. Raises CutRejected
    for a cut that clips words without removing any whole one, and for a cut
    shorter than ``MIN_CUT_SECONDS`` once placed.
    """
    start, end = _placed_edges(a, b, tokens, duration, silences or [])
    if end - start < MIN_CUT_SECONDS:
        raise CutRejected(
            f"is too short to matter, under {MIN_CUT_SECONDS}s ({end - start:.3f}s once placed). "
            "Widen it or drop it."
        )
    return start, end


def _placed_edges(
    a: float, b: float, tokens: list[dict[str, Any]], duration: float, silences: list[Silence]
) -> tuple[float, float]:
    removed = [w for w in tokens if a < _mid(w) < b]
    if not removed:
        touched = [w for w in tokens if float(w["start"]) < b and float(w["end"]) > a]
        if touched:
            raise _reject_no_whole_word(a, b, touched)
        return a, b

    removed_ids = {id(w) for w in removed}
    first_start = min(float(w["start"]) for w in removed)
    last_end = max(float(w["end"]) for w in removed)
    kept_before = [w for w in tokens if id(w) not in removed_ids and _mid(w) <= a]
    kept_after = [w for w in tokens if id(w) not in removed_ids and _mid(w) >= b]

    if kept_before:
        prev_end = max(float(w["end"]) for w in kept_before)
        # The kept word's tail runs into the gap: the cut starts where it stops.
        start = pad_keep_ranges_into_gaps([(prev_end, prev_end)], tokens)[0][1]
        start = max(prev_end, min(start, first_start))
        start = _toward_silence(start, prev_end, first_start, silences)
    else:
        start = max(0.0, min(a, first_start))

    if kept_after:
        next_start = min(float(w["start"]) for w in kept_after)
        # The next kept word gets its lead-in: the cut ends before it starts.
        end = pad_keep_ranges_into_gaps([(next_start, next_start)], tokens)[0][0]
        end = min(next_start, max(end, last_end))
        end = _toward_silence(end, last_end, next_start, silences)
    else:
        end = min(duration, max(b, last_end))

    if end <= start:
        raise _reject_no_whole_word(a, b, removed)
    return start, end


def _place_piece(
    a: float, b: float, tokens: list[dict[str, Any]], duration: float, silences: list[Silence]
) -> tuple[float, float] | None:
    """Edges for one piece of a cut that was split around words the creator brought back.

    A piece is kept when it still removes a whole word, however short: the
    cut it came from was long enough. A piece that removes none is dropped.
    """
    if not any(a < _mid(w) < b for w in tokens):
        return None
    try:
        return _placed_edges(a, b, tokens, duration, silences)
    except CutRejected:
        return None


# Words either side of a creator's cut that are read to find where its
# neighbours end and start. Words are in the order they are said, so the
# nearest few hold the latest end before the cut and the earliest start after.
NEIGHBOURS_READ = 8


@dataclass(frozen=True)
class SaidOrder:
    """A transcript's words and sounds in the order they are said (by their middle), for binary search."""

    tokens: list[dict[str, Any]]
    mids: list[float]

    @classmethod
    def of(cls, words: list[dict[str, Any]]) -> SaidOrder:
        tokens = sorted(words, key=_mid)
        return cls(tokens=tokens, mids=[_mid(w) for w in tokens])

    def picked(self, a: float, b: float) -> range:
        """Positions of the words and sounds whose middle falls inside ``[a, b]``, ends included.

        The ends count so a word the transcript gives no length can still be
        picked by its own time.
        """
        return range(bisect.bisect_left(self.mids, a), bisect.bisect_right(self.mids, b))

    def neighbours(self, picked: range, duration: float) -> tuple[float, float]:
        """Where the words before ``picked`` end and the words after it start: ``(floor, ceiling)``."""
        before = self.tokens[max(0, picked.start - NEIGHBOURS_READ):picked.start]
        after = self.tokens[picked.stop:picked.stop + NEIGHBOURS_READ]
        return (max((float(w["end"]) for w in before), default=0.0),
                min((float(w["start"]) for w in after), default=duration))


def place_creator_cut(a: float, b: float, said: SaidOrder, duration: float) -> tuple[float, float, int]:
    """``(start, end, words picked)`` for a cut the creator made by hand over ``[a, b]``.

    The cut removes exactly the picked words: it runs from the first one's
    start to the last one's end and stops at the neighbouring words, so it
    never takes a piece of a word she left in. A cut shorter than
    ``MIN_CUT_SECONDS`` widens into the quiet beside it, as far as that quiet
    goes, and is made even when it stays shorter. Raises CutRejected when no
    word's middle is inside, or when the words have no length to cut.
    """
    picked = said.picked(a, b)
    if not picked:
        raise CutRejected("holds no word. Pick at least one word.")
    chosen = said.tokens[picked.start:picked.stop]
    first = min(float(w["start"]) for w in chosen)
    last = max(float(w["end"]) for w in chosen)
    floor, ceiling = said.neighbours(picked, duration)
    start, end = max(first, min(floor, last)), min(last, max(ceiling, first))
    if end <= start:
        start, end = first, last  # the neighbours overlap these words whole; cut what she picked
    short = MIN_CUT_SECONDS - (end - start)
    if short > 0:
        room_before, room_after = max(0.0, start - floor), max(0.0, ceiling - end)
        take_after = min(room_after, short / 2)
        take_before = min(room_before, short - take_after)
        take_after = min(room_after, short - take_before)
        start, end = start - take_before, end + take_after
    if end <= start:
        raise CutRejected("has no length in the transcript, so there is nothing to cut. Pick it with the word beside it.")
    return start, end, len(picked)


# ── Whether a word can be cut without clipping the one beside it ─────────────

# The aligner works in frames of 0.02 s and puts one empty frame between two
# words it tells apart, so a gap of one frame says only that the words are
# two. From 0.03 s up there is quiet between them. On the test footage 17 of
# 112 "like"s sit closer than that to a neighbour, such as "a bit like a",
# and have no measured silence at that edge: the words run into each other,
# and a cut there takes the end or the start of the neighbour with it.
CLEAR_OF_NEIGHBOUR_SECONDS = 0.03
# A measured silence this near a word's edge counts as at the edge.
SILENCE_AT_EDGE_SECONDS = 0.01
# No single word lasts this long. Aligning sometimes stretches one over a
# pause (a 1.9 s "like" on the test footage); its edges can't be trusted.
LONGEST_WORD_SECONDS = 1.0

CLIPS_THE_WORD_BEFORE = "cutting it would clip the word before"
CLIPS_THE_NEXT_WORD = "cutting it would clip the next word"
TIME_NOT_TRUSTED = "its time in the transcript is too long to be right"


class SilenceIndex:
    """Measured silences sorted once, to ask whether one sits at a given time."""

    def __init__(self, silences: list[Silence] | None) -> None:
        self.spans = sorted((float(s.start), float(s.end)) for s in silences or [])
        self.starts = [a for a, _ in self.spans]

    def at(self, t: float, reach: float = SILENCE_AT_EDGE_SECONDS) -> bool:
        """Whether a measured silence holds ``t``, or comes within ``reach`` of it."""
        i = bisect.bisect_right(self.starts, t + reach) - 1
        return i >= 0 and self.spans[i][1] >= t - reach


def _as_aligned(word: dict[str, Any]) -> Span:
    """A word's edges as the aligner gave them: where it heard the word start and end."""
    return (float(word.get(times.ALIGNED_START, word["start"])), float(word.get(times.ALIGNED_END, word["end"])))


def why_not_clean(picked: range, said: SaidOrder, quiet: SilenceIndex, duration: float) -> str:
    """Why cutting the picked words would not be clean, or "" when it would.

    A cut is clean when each side has quiet between the picked words and
    their neighbour: ``CLEAR_OF_NEIGHBOUR_SECONDS`` between the two as the
    aligner placed them, or a measured silence at that edge. The research
    for this round found the harm in cuts that remove sound straight after
    or straight before a word; this is that test, asked before the cut is
    made.

    The test reads the edges the aligner gave, not the room each word has
    for its sound (``word_times.with_room``). A word's room runs up to its
    neighbour wherever the sound does, so by their rooms nearly every two
    words touch, and that says nothing about whether the aligner could tell
    them apart. The cut itself takes the word's whole room and stops at the
    neighbours' rooms, so it leaves no piece of the word behind.
    """
    chosen = said.tokens[picked.start:picked.stop]
    first = min(_as_aligned(w)[0] for w in chosen)
    last = max(_as_aligned(w)[1] for w in chosen)
    room = max(float(w["end"]) for w in chosen) - min(float(w["start"]) for w in chosen)
    if max(last - first, room) > LONGEST_WORD_SECONDS * len(chosen):
        return TIME_NOT_TRUSTED
    before = said.tokens[max(0, picked.start - NEIGHBOURS_READ):picked.start]
    after = said.tokens[picked.stop:picked.stop + NEIGHBOURS_READ]
    floor = max((_as_aligned(w)[1] for w in before), default=0.0)
    ceiling = min((_as_aligned(w)[0] for w in after), default=duration)
    if picked.start > 0 and first - floor < CLEAR_OF_NEIGHBOUR_SECONDS and not quiet.at(first):
        return CLIPS_THE_WORD_BEFORE
    if picked.stop < len(said.tokens) and ceiling - last < CLEAR_OF_NEIGHBOUR_SECONDS and not quiet.at(last):
        return CLIPS_THE_NEXT_WORD
    return ""


# ── The words Claude picks ────────────────────────────────────────────────────

PICK_ID_PREFIX = "w"
MAX_PICK_REASON_CHARS = 200
# What became of a pick when the edit was built.
PICK_OUT, PICK_LEFT_IN, PICK_KEPT, PICK_LOST = "out", "left_in", "kept_by_you", "lost"
# A pick's id holds its words' times to the millisecond; times this close are the same.
SAME_TIME_SECONDS = 0.0015


def pick_id(start: float, end: float) -> str:
    """The id of one place a word is said: ``w312.395-312.541``, from the word's own times."""
    return f"{PICK_ID_PREFIX}{start:.3f}-{end:.3f}"


def span_of_pick_id(value: Any) -> Span:
    """The times an id from ``find_words`` holds. Raises CutRejected when it is not such an id."""
    try:
        if not isinstance(value, str) or not value.startswith(PICK_ID_PREFIX):
            raise ValueError
        start, end = (float(part) for part in value[len(PICK_ID_PREFIX):].split("-"))
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end):
            raise ValueError
    except ValueError:
        raise CutRejected(f"has id {value!r}, which is not an id from find_words. Ids look like w312.395-312.541.") from None
    return start, end


def parse_pick(index: int, raw: Any, said: SaidOrder) -> dict[str, Any]:
    """Check one ``{id, reason}`` against the transcript: ``{id, start, end, text, reason}``. Raises CutRejected.

    The id must name words as they are timed now, so a pick made before the
    word times changed is refused and not guessed at.
    """
    if not isinstance(raw, dict):
        raise CutRejected("is not an object. Pass {id, reason}.")
    start, end = span_of_pick_id(raw.get("id"))
    reason = raw.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise CutRejected("has no reason. Every pick needs a few words on why the word is filler.")
    if len(reason) > MAX_PICK_REASON_CHARS:
        raise CutRejected(f"has a reason over {MAX_PICK_REASON_CHARS} characters. A few words are enough.")
    picked = said.picked(start, end)
    chosen = said.tokens[picked.start:picked.stop]
    same = bool(chosen) and abs(min(float(w["start"]) for w in chosen) - start) <= SAME_TIME_SECONDS \
        and abs(max(float(w["end"]) for w in chosen) - end) <= SAME_TIME_SECONDS
    if not same:
        raise CutRejected(
            f"has id {raw['id']!r}, which names no word as the transcript is timed now. "
            "Call find_words again and use its ids."
        )
    return {
        "id": pick_id(start, end), "start": round(start, 3), "end": round(end, 3),
        "text": " ".join(str(w["word"]).strip() for w in chosen), "reason": reason.strip(),
    }


def creator_cut_id(cut: dict[str, Any]) -> str:
    """The stable id of one of the creator's cuts: its saved ``id``, else ``y<start>-<end>`` of its words."""
    return str(cut.get("id") or f"{CREATOR_ROW_PREFIX}{float(cut['start']):.2f}-{float(cut['end']):.2f}")


# ── Input checks ──────────────────────────────────────────────────────────────


@dataclass
class CutRequest:
    index: int
    start: float
    end: float
    reason: str
    kind: str = DEFAULT_CUT_KIND
    override_keeps: bool = False


@dataclass
class EditOutcome:
    """What ``build_edit`` decided, ready to save and to report."""

    cuts: list[dict[str, Any]]
    applied: list[dict[str, Any]] = field(default_factory=list)
    adjusted: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    auto_cut_count: int = 0
    auto_unsafe: int = 0
    # Automatic trims dropped for touching a joke or a span the creator restored.
    auto_skipped_protected: int = 0
    auto_skipped_kept: int = 0
    # Every requested cut that placed word-safe, as {index, start, end, reason,
    # kind, put_back}. put_back is true when a kept span stopped it. A cut
    # split around words the creator brought back has one entry per piece.
    placed: list[dict[str, Any]] = field(default_factory=list)
    # Each of the creator's own cuts as placed: {id, start, end, word_start,
    # word_end, words}. ``creator_lost`` counts those whose words are gone
    # from the transcript.
    creator_placed: list[dict[str, Any]] = field(default_factory=list)
    creator_lost: int = 0
    # Each filler word Claude picked and what became of it: {id, start, end,
    # word_start, word_end, status, why}. ``status`` is ``PICK_OUT`` (cut;
    # start and end are the cut as placed), ``PICK_LEFT_IN`` (not clean;
    # ``why`` says what it would clip), ``PICK_KEPT`` (the creator brought it
    # back or kept the part it sits in) or ``PICK_LOST`` (its words are gone).
    picks: list[dict[str, Any]] = field(default_factory=list)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def parse_cut(index: int, raw: Any, duration: float) -> CutRequest:
    """Check one ``{start, end, reason, kind?}`` against the video. Raises CutRejected.

    A missing or null ``kind`` is ``DEFAULT_CUT_KIND``; any other value must be
    one of ``CUT_KINDS``.
    """
    if not isinstance(raw, dict):
        raise CutRejected("is not an object. Pass {start, end, reason, kind}.")
    start, end, reason = raw.get("start"), raw.get("end"), raw.get("reason")
    if not _finite(start) or not _finite(end):
        raise CutRejected("start and end must be numbers of seconds in the source video.")
    if not isinstance(reason, str) or not reason.strip():
        raise CutRejected("has no reason. Every cut needs a short reason saying why it goes.")
    kind = raw.get("kind") or DEFAULT_CUT_KIND
    if kind not in CUT_KINDS:
        raise CutRejected(f"has kind {kind!r}, which is not one of: {', '.join(CUT_KINDS)}.")
    if start < 0:
        raise CutRejected(f"start {start} is before 0. The video starts at 0.")
    if end > duration + END_TOLERANCE:
        raise CutRejected(f"end {end} is past the end of the video ({duration:.2f}).")
    end = min(float(end), duration)
    if end <= start:
        raise CutRejected(f"start {start} is not before end {end}.")
    return CutRequest(
        index=index, start=float(start), end=end, reason=reason.strip(), kind=kind,
        override_keeps=raw.get(OVERRIDE_FLAG) is True,
    )


# ── Building the edit ─────────────────────────────────────────────────────────


def _overlaps(a: float, b: float, c: float, d: float) -> bool:
    return a < d and c < b


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


Span = tuple[float, float]


def keep_spans(edit: dict[str, Any]) -> list[Span]:
    """The spans the creator restored on the review page, as ``(start, end)`` pairs."""
    return [(float(k["start"]), float(k["end"])) for k in edit.get("keep", [])]


def _first_overlap(start: float, end: float, spans: list[Span]) -> Span | None:
    return next((s for s in spans if _overlaps(start, end, s[0], s[1])), None)


def _allowed_auto_cuts(
    auto: list[dict[str, Any]], keeps: list[Span], protected: list[Span], outcome: EditOutcome,
    exact: list[Span] | None = None, said: Said | None = None,
) -> list[dict[str, Any]]:
    """The automatic trims that touch neither a kept span nor a protected one.

    ``exact`` are words the creator brought back, and ``said`` the spoken
    words they are found in. A trim over such a word is shortened around it
    (``autocuts.around``), so the word plays and the pause beside it still
    goes. Counts what it drops on ``outcome``.
    """
    allowed = []
    for cut in auto:
        start, end = float(cut["start"]), float(cut["end"])
        holes = [k for k in exact or [] if _overlaps(start, end, k[0], k[1])]
        if _first_overlap(start, end, keeps):
            outcome.auto_skipped_kept += 1
        elif _first_overlap(start, end, protected):
            outcome.auto_skipped_protected += 1
        elif holes and said is not None:
            pieces = around(cut, holes, said)
            allowed += pieces
            outcome.auto_skipped_kept += not pieces
        else:
            allowed.append(cut)
    return allowed


# ── Finalizing near each trim ─────────────────────────────────────────────────

# ClipForge lifts a trim's edge clear of the word it sits in and then onto
# the nearest word edge within 0.12 s, so only the words around a trim can
# move it. Reading the whole transcript for every trim cost 0.6 s a pace at
# the hardest stop; the words within this reach give the same answer.
CLAMP_REACH_SECONDS = 1.0


@dataclass(frozen=True)
class _Nearby:
    """Words and silences sorted once, so the ones around a span are found by binary search."""

    words: list[dict[str, Any]]
    word_starts: list[float]
    longest_word: float
    silences: list[Silence]
    silence_starts: list[float]
    longest_silence: float

    @classmethod
    def of(cls, words: list[dict[str, Any]], silences: list[Silence]) -> _Nearby:
        by_start = sorted(words, key=lambda w: float(w["start"]))
        quiet = sorted(silences, key=lambda s: s.start)
        return cls(
            words=by_start, word_starts=[float(w["start"]) for w in by_start],
            longest_word=max((float(w["end"]) - float(w["start"]) for w in by_start), default=0.0),
            silences=quiet, silence_starts=[s.start for s in quiet],
            longest_silence=max((s.end - s.start for s in quiet), default=0.0),
        )

    def around(self, start: float, end: float) -> tuple[list[dict[str, Any]], list[Silence]]:
        """The words and silences that overlap ``[start, end]`` widened by ``CLAMP_REACH_SECONDS``."""
        lo, hi = start - CLAMP_REACH_SECONDS, end + CLAMP_REACH_SECONDS
        first = bisect.bisect_left(self.word_starts, lo - self.longest_word)
        words = [w for w in self.words[first:bisect.bisect_right(self.word_starts, hi)] if float(w["end"]) >= lo]
        first = bisect.bisect_left(self.silence_starts, lo - self.longest_silence)
        quiet = [s for s in self.silences[first:bisect.bisect_right(self.silence_starts, hi)] if s.end >= lo]
        return words, quiet


def finalize_near(
    auto_spans: list[Span], verbatim_spans: list[Span], words: list[dict[str, Any]],
    silences: list[Silence] | None,
) -> tuple[list[list[float]], int]:
    """``render.finalize_removed_ranges`` with each trim read against the words near it: ``(ranges, unsafe)``.

    Gives the ranges ClipForge gives for the whole transcript (a test holds
    the two equal), in a fraction of the time.
    """
    near = _Nearby.of(words, silences or [])
    placed: list[Span] = []
    unsafe = 0
    for start, end in auto_spans:
        local_words, local_quiet = near.around(start, end)
        one = finalize_removed_ranges([(start, end)], [], local_words, silences=local_quiet)
        placed += [(float(a), float(b)) for a, b in one.ranges]
        unsafe += one.unsafe
    merged = finalize_removed_ranges([], placed + list(verbatim_spans), [], silences=[])
    return merged.ranges, unsafe


# ── No automatic removal inside a word ────────────────────────────────────────


def _has_an_automatic_part(cut: dict[str, Any]) -> bool:
    """Whether some of ``cut`` was planned by rule: it is an automatic trim, or one joined it."""
    return cut.get("source") == BY_AUTO or bool(cut.get("auto_trims"))


def keep_words_whole(
    cuts: list[dict[str, Any]], words: list[dict[str, Any]], held: list[Span],
) -> tuple[list[dict[str, Any]], int]:
    """``cuts`` with no automatic removal starting or ending inside a spoken word or a sound of the transcript, and how many edges moved.

    The last check on a finished edit. A word is its span on the word times
    the edit was built on, its room included (``word_times.with_room``).
    ClipForge lands a trim clear of words, but it counts an edge in
    measured silence as clean even inside a word. So the parts of a word
    quieter than the silences are measured at are left to this check: a
    soft start or end, or the quiet inside "positivity". On the test
    footage it moves up to 6 edges a stop, the soft starts of "and" and
    "And" among them.

    A sound of the transcript (a laugh, a breath, an "uhh") is kept whole
    the same way. ClipForge reads the quiet inside one as a pause, and the
    page strikes a sound that a removal takes any part of: on the test
    footage trims at Standard took 4% to 96% of 25 sounds, which showed
    struck while the rest played. A trim may still take a sound whole.

    An edge of a cut with an automatic part that sits inside a word or a
    sound moves to its far side, toward the pause (``word_times.Spoken.clear``).
    ``held`` are the parts Claude and the creator chose, as placed: an edge
    of theirs never moves and no move passes their part, so a cut that
    holds one is always saved with it. A cut with no part of theirs goes
    when less than ``MIN_CUT_SECONDS`` is left. Each change is logged.
    """
    spoken = times.Spoken.of(words, sounds=True)
    theirs = _ByStart([(a, b, (a, b)) for a, b in held])
    out: list[dict[str, Any]] = []
    moved = 0
    for cut in cuts:
        was = (float(cut["start"]), float(cut["end"]))
        if not _has_an_automatic_part(cut):
            out.append(cut)
            continue
        own = theirs.overlapping(*was)
        keep = (min(a for a, _ in own), max(b for _, b in own)) if own else None
        start, end = spoken.clear(*was, keep)
        if (start, end) == was:
            out.append(cut)
            continue
        moved += (start != was[0]) + (end != was[1])
        if keep is None and end - start < MIN_CUT_SECONDS:
            log.warning("an automatic cut at %.3f-%.3f sat inside a word or a sound, and what is left goes", *was)
            continue
        log.warning("an automatic cut at %.3f-%.3f sat inside a word or a sound; now %.3f-%.3f", *was, start, end)
        out.append({**cut, "start": round(start, 3), "end": round(end, 3)})
    return out, moved


# ── Remembered placements ─────────────────────────────────────────────────────

# Where a cut of Claude's lands depends on the words, not on the pace. The
# page builds the edit at all six paces for every action, so each placement
# is worked out once and remembered.
PLACEMENTS_KEPT = 512
_placements: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
_placements_lock = threading.Lock()


def forget_placements() -> None:
    """Empty the remembered placements, for tests."""
    with _placements_lock:
        _placements.clear()


class _ByStart:
    """Spans with something attached, sorted by start, so the ones overlapping a range are found by binary search."""

    def __init__(self, items: list[tuple[float, float, Any]]) -> None:
        self.items = sorted(items, key=lambda item: (item[0], item[1]))
        self.starts = [item[0] for item in self.items]
        self.longest = max((item[1] - item[0] for item in self.items), default=0.0)

    def overlapping(self, start: float, end: float) -> list[Any]:
        """What is attached to the spans that overlap ``[start, end]``, in start order."""
        first = bisect.bisect_left(self.starts, start - self.longest)
        last = bisect.bisect_left(self.starts, end)
        return [what for a, b, what in self.items[first:last] if _overlaps(a, b, start, end)]


def _restored_message(span: Span) -> str:
    return (
        f"overlaps {span[0]:.2f}-{span[1]:.2f}, which the creator restored "
        "on the review page. Leave it in. If the creator has now asked for this "
        "cut, call set_edit again with override_keeps=true."
    )


def _place_request(
    req: CutRequest, tokens: list[dict[str, Any]], duration: float, silences: list[Silence] | None,
    holes: list[Span], take: str,
) -> tuple[list[Span], Span | None]:
    """Where one of Claude's cuts lands: ``(pieces, whole)``, remembered for this ``take``.

    With no ``holes`` (words the creator brought back inside it) there is one
    piece, the cut as ``place_cut`` puts it. With holes the cut splits around
    them. ``whole`` is the unsplit placement when every piece fell away, so
    the cut can read as put back. ``take`` names the words and silences the
    cut is placed on. Raises CutRejected as ``place_cut`` does.
    """
    key = (take, req.start, req.end, tuple(holes))
    with _placements_lock:
        hit = _placements.get(key)
        if hit is not None:
            _placements.move_to_end(key)
    if hit is None:
        try:
            hit = _work_out_placement(req, tokens, duration, silences, holes)
        except CutRejected as why:
            hit = why
        with _placements_lock:
            _placements[key] = hit
            while len(_placements) > PLACEMENTS_KEPT:
                _placements.popitem(last=False)
    if isinstance(hit, CutRejected):
        raise hit
    pieces, whole = hit
    return list(pieces), whole


def _work_out_placement(
    req: CutRequest, tokens: list[dict[str, Any]], duration: float, silences: list[Silence] | None,
    holes: list[Span],
) -> tuple[list[Span], Span | None]:
    whole = place_cut(req.start, req.end, tokens, duration, silences)
    if not holes:
        return [whole], None
    pieces = [
        placed for a, b in span_without(req.start, req.end, holes)
        if (placed := _place_piece(a, b, tokens, duration, silences or [])) is not None
    ]
    return pieces, None if pieces else whole


def _placed_picks(
    picks: list[dict[str, Any]], said: SaidOrder | None, quiet: SilenceIndex, kept: list[Span], duration: float
) -> list[dict[str, Any]]:
    """What becomes of each filler word Claude picked. See ``EditOutcome.picks``.

    A pick in a span the creator kept, or on a word she brought back, stays
    in: what she did by hand wins. A pick that is not clean stays in. The
    rest are placed the way her own word cuts are.
    """
    out = []
    for pick in picks:
        start, end = float(pick["start"]), float(pick["end"])
        found = said.picked(start, end) if said is not None else range(0)
        entry = {"id": pick.get("id") or pick_id(start, end), "start": start, "end": end,
                 "word_start": start, "word_end": end, "status": PICK_LOST, "why": ""}
        if found:
            why = why_not_clean(found, said, quiet, duration)
            if _first_overlap(start, end, kept) or any(a <= start and end <= b for a, b in kept):
                entry["status"] = PICK_KEPT
            elif why:
                entry["status"], entry["why"] = PICK_LEFT_IN, why
            else:
                try:
                    a, b, _ = place_creator_cut(start, end, said, duration)
                    entry.update(start=round(a, 3), end=round(b, 3), status=PICK_OUT)
                except CutRejected as rejected:
                    entry["status"], entry["why"] = PICK_LEFT_IN, str(rejected)
        out.append(entry)
    return out


def build_edit(
    raw_cuts: list[Any],
    words: list[dict[str, Any]],
    duration: float,
    *,
    auto_cuts: list[dict[str, Any]] | None = None,
    silences: list[Silence] | None = None,
    keeps: list[Span] | None = None,
    protected: list[Span] | None = None,
    override_keeps: bool = False,
    creator_cuts: list[dict[str, Any]] | None = None,
    exact_keeps: list[Span] | None = None,
    picks: list[dict[str, Any]] | None = None,
) -> EditOutcome:
    """Validate, place, merge and finalize Claude's cuts, the creator's own cuts, Claude's picks and any auto cuts.

    ``keeps`` are spans the creator restored; ``protected`` are punchlines and
    laughs (see the module docstring). ``override_keeps`` lets Claude's cuts
    into kept spans, for when the creator asks for the cut after all; a raw
    cut carrying ``OVERRIDE_FLAG`` gets the same for itself alone.

    ``creator_cuts`` are the cuts the creator made by hand, each ``{start,
    end}`` over the words she picked. ``exact_keeps`` are the words she
    brought back one by one: a cut of Claude's splits around them.
    ``picks`` are the filler words Claude picked, each ``{id, start, end}``
    over a word's own times; pass none when their switch is off. A pick is
    cut only when it is clean and no kept span holds it.

    Returns the finalized cut list (merged, sorted, each with its reasons and a
    ``source`` of ``claude``, ``you``, ``pick`` or ``auto``) and the applied /
    adjusted / rejected report. Raises StudioError when the result would remove the
    whole video.
    """
    tokens = sorted(words, key=lambda w: float(w["start"]))
    placed: list[tuple[CutRequest, float, float]] = []
    outcome = EditOutcome(cuts=[])
    kept_by_creator = keeps or []
    brought_back = exact_keeps or []
    take = take_mark(tokens, silences or [], duration) if raw_cuts else ""

    for i, raw in enumerate(raw_cuts):
        try:
            req = parse_cut(i, raw, duration)
            forced = override_keeps or req.override_keeps
            holes = [] if forced else [k for k in brought_back if _overlaps(req.start, req.end, k[0], k[1])]
            pieces, whole = _place_request(req, tokens, duration, silences, holes, take)
        except CutRejected as why:
            outcome.rejected.append({"index": i, "cut": raw, "why": f"Cut {i} {why}"})
            continue
        for start, end in pieces or [whole]:
            restored = None if forced else _first_overlap(start, end, kept_by_creator)
            if not pieces:
                restored = holes[0]  # she brought back every word this cut removes
            outcome.placed.append({
                "index": i, "start": round(start, 3), "end": round(end, 3),
                "reason": req.reason, "kind": req.kind, "put_back": bool(restored),
            })
            if restored:
                outcome.rejected.append({"index": i, "cut": raw, "why": f"Cut {i} {_restored_message(restored)}"})
                continue
            placed.append((req, start, end))
            if abs(start - req.start) > ADJUST_EPS or abs(end - req.end) > ADJUST_EPS:
                outcome.adjusted.append({
                    "index": i,
                    "requested": {"start": round(req.start, 3), "end": round(req.end, 3)},
                    "final": {"start": round(start, 3), "end": round(end, 3)},
                    "why": "split around words the creator brought back" if holes else
                           "edges moved to word boundaries (midpoint rule) with a natural tail and lead-in kept",
                })

    hers: list[Span] = []
    said = SaidOrder.of(tokens) if creator_cuts or brought_back or picks else None
    for cut in creator_cuts or []:
        try:
            start, end, picked = place_creator_cut(float(cut["start"]), float(cut["end"]), said, duration)
        except CutRejected:
            outcome.creator_lost += 1
            continue
        hers.append((start, end))
        outcome.creator_placed.append({
            "id": creator_cut_id(cut), "start": round(start, 3), "end": round(end, 3),
            "word_start": float(cut["start"]), "word_end": float(cut["end"]), "words": picked,
        })

    outcome.picks = _placed_picks(picks or [], said, SilenceIndex(silences), kept_by_creator + brought_back, duration)
    picked_out = [(p["start"], p["end"]) for p in outcome.picks if p["status"] == PICK_OUT]

    auto = _allowed_auto_cuts(
        auto_cuts or [], kept_by_creator, protected or [], outcome, brought_back,
        Said.of(tokens) if brought_back else None,
    )
    ranges, outcome.auto_unsafe = finalize_near(
        [(float(c["start"]), float(c["end"])) for c in auto],
        [(s, e) for _, s, e in placed] + hers + picked_out,
        words, silences,
    )
    outcome.auto_cut_count = len(auto)

    claudes = _ByStart([(s, e, (req, s, e)) for req, s, e in placed])
    creators = _ByStart([(h[0], h[1], h) for h in hers])
    pickeds = _ByStart([(a, b, (a, b)) for a, b in picked_out])
    trims = _ByStart([(float(c["start"]), float(c["end"]), c) for c in auto])
    for rs, re_ in ranges:
        mine = claudes.overlapping(rs, re_)
        yours = creators.overlapping(rs, re_)
        chosen = pickeds.overlapping(rs, re_)
        theirs = trims.overlapping(rs, re_)
        cut: dict[str, Any] = {"start": round(rs, 3), "end": round(re_, 3)}
        if mine or yours or chosen:
            if mine:
                cut["reason"] = "; ".join(_unique([req.reason for req, _, _ in mine]))
                cut["source"] = BY_CLAUDE
                cut["kind"] = mine[0][0].kind
            elif yours:
                cut["reason"] = CREATOR_REASON
                cut["source"] = BY_CREATOR
                cut["kind"] = CREATOR_KIND
            else:
                cut["reason"] = PICK_REASON
                cut["source"] = BY_PICK
                cut["kind"] = PICK_KIND
            if yours and mine:
                cut["creator_cuts"] = len(yours)
            if chosen and (mine or yours):
                cut["picks"] = len(chosen)
            if theirs:
                cut["auto_trims"] = len(theirs)
        else:
            cut["reason"] = "; ".join(_unique([f"auto: {c['reason']}" for c in theirs])) or "auto"
            cut["source"] = BY_AUTO
            kinds = [c["kind"] for c in theirs if c.get("kind")]
            if kinds:
                # A trim merged from several is filed under its longest part.
                longest = max(theirs, key=lambda c: float(c["end"]) - float(c["start"]))
                cut["kind"] = longest.get("kind", kinds[0])
        outcome.cuts.append(cut)
        if mine:
            outcome.applied.append({**display_cut(cut), "from_cuts": [req.index for req, _, _ in mine]})

    if not kept_segments([(c["start"], c["end"]) for c in outcome.cuts], duration):
        raise StudioError("These cuts remove the whole video. Keep at least some of it and try again.")
    return outcome


# ── The saved edit ────────────────────────────────────────────────────────────


def removed_spans(edit: dict[str, Any]) -> list[tuple[float, float]]:
    """The saved edit's removed ranges as ``(start, end)`` pairs."""
    return [(float(c["start"]), float(c["end"])) for c in edit.get("cuts", [])]


def clock(seconds: float) -> str:
    """``m:ss`` for a time in seconds, rounded down to the whole second.

    240.4 is ``4:00``. The model gets this ready-made: asked to format a time
    by hand it has written ``3:60``.
    """
    whole = int(seconds)
    return f"{whole // 60}:{whole % 60:02d}"


def clock_span(start: float, end: float) -> str:
    """``m:ss-m:ss`` for a span, each end as ``clock`` writes it."""
    return f"{clock(start)}-{clock(end)}"


def display_cut(cut: dict[str, Any]) -> dict[str, Any]:
    """A saved cut as the reader sees it: ``{start, end, clock, seconds, reason[, kind]}``.

    ``clock`` is the source times as ``m:ss-m:ss``, ready to show a creator.
    Automatic trims merged into a Claude cut show as a short count, not their reasons.
    """
    reason = cut.get("reason", "")
    trims = int(cut.get("auto_trims", 0))
    if trims:
        reason += f" (+{trims} automatic trim{'s' if trims > 1 else ''})"
    shown = {
        "start": cut["start"],
        "end": cut["end"],
        "clock": clock_span(float(cut["start"]), float(cut["end"])),
        "seconds": round(float(cut["end"]) - float(cut["start"]), 3),
        "reason": reason,
    }
    if cut.get("source") == BY_CLAUDE and cut.get("kind"):
        shown["kind"] = cut["kind"]
    return shown


def edit_summary(edit: dict[str, Any], duration: float, include_auto: bool = False) -> dict[str, Any]:
    """Claude's cuts with reasons, a summary of the automatic ones, and the totals.

    Automatic cuts are listed one by one only with ``include_auto``, and then
    only the first ``AUTO_CUTS_SHOWN``. The creator's own cuts are counted in
    the totals and listed apart, by ``treatment.creator_changes``.
    """
    cuts = edit.get("cuts", [])
    auto = [c for c in cuts if c.get("source") == BY_AUTO]
    kept = kept_segments(removed_spans(edit), duration)
    new_duration = edited_duration(kept)
    auto_summary: dict[str, Any] = {
        "count": len(auto),
        "seconds_removed": round(sum(float(c["end"]) - float(c["start"]) for c in auto), 3),
    }
    if include_auto:
        auto_summary["cuts"] = [display_cut(c) for c in auto[:AUTO_CUTS_SHOWN]]
        auto_summary["not_shown"] = max(0, len(auto) - AUTO_CUTS_SHOWN)
    summary: dict[str, Any] = {
        "cuts": [display_cut(c) for c in cuts if c.get("source") == BY_CLAUDE],
        "auto": auto_summary,
        "auto_tighten": bool(edit.get("auto_tighten", False)),
        "gap_length": edit.get("gap_length"),
        "pace": edit.get("pace"),
        "cut_count": len(cuts),
        "removed_seconds": round(duration - new_duration, 3),
        "duration": round(duration, 3),
        "new_duration": round(new_duration, 3),
    }
    if edit.get("keep"):
        summary["creator_keeps"] = edit["keep"]
    return summary


def empty_edit() -> dict[str, Any]:
    """The edit of a video nobody has cut yet."""
    return {"version": EDIT_VERSION, "cuts": [], "auto_tighten": False, "gap_length": None, "pace": None}


def validate_saved_edit(data: Any) -> dict[str, Any]:
    """Check an edit.json read from disk. Raises StudioError when it is unusable."""
    fix = "Call set_edit to replace it."
    if not isinstance(data, dict) or not isinstance(data.get("cuts"), list):
        raise StudioError(f"The saved edit.json is not a valid edit. {fix}")
    for i, c in enumerate(data["cuts"]):
        if not (isinstance(c, dict) and _finite(c.get("start")) and _finite(c.get("end"))
                and c["end"] > c["start"]):
            raise StudioError(f"The saved edit.json has a bad cut at index {i}. {fix}")
    return data


# ── Persistence ───────────────────────────────────────────────────────────────

# A saved edit whose recorded duration differs from the video's by more than this
# was made against a different file.
DURATION_DRIFT = 0.5


def placed_on(edit: dict[str, Any]) -> str:
    """Which word times a saved edit was placed on. One saved before aligning existed was placed on estimates."""
    saved = edit.get("word_times")
    return saved if saved in times.SOURCES else times.ESTIMATED


def placed_on_these(edit: dict[str, Any], source: str) -> bool:
    """Whether a saved edit was placed on word times of ``source`` as they are made now.

    Measured times are made one way at a time (``word_times.made_as``). An
    edit placed on measured times made another way, such as before words
    had room for their sound, was placed on other word edges.
    """
    if placed_on(edit) != source:
        return False
    return source != times.MEASURED or edit.get(MADE_AS, times.WITHOUT_ROOM_VERSION) == times.made_as(source)


def _moved_with_its_words(span: Any, words: list[dict[str, Any]], was_on: str, prefix: str = "") -> Any:
    """A saved span (a cut, a kept part, a picked word) moved onto the measured times of the words it holds."""
    if not (isinstance(span, dict) and _finite(span.get("start")) and _finite(span.get("end"))):
        return span
    start, end = times.on_measured_times(float(span["start"]), float(span["end"]), words, was_on)
    moved = {**span, "start": start, "end": end}
    if prefix == PICK_ID_PREFIX:
        moved["id"] = pick_id(start, end)
    elif prefix:
        moved["id"] = f"{prefix}{start:.2f}-{end:.2f}"
    return moved


def on_the_word_times_of(project: Project, edit: dict[str, Any]) -> dict[str, Any]:
    """``edit`` with every span somebody chose moved onto the word times the project has now.

    Claude's cuts, the creator's cuts, the spans she kept and the words
    Claude picked are saved in source seconds. When the word times become
    measured, or measured ones are made another way, each span moves with
    its words, so it holds the words it held before. This is the one place
    they move, so no reader of a saved edit gets a span that lost its words.
    A moved edit says so under ``MOVED_FROM``; its ``cuts`` are as saved.
    """
    hers = ("requested", "keep", "creator_cuts", "picks")
    if not any(edit.get(key) for key in hers):
        return edit
    now = times.source_of(project)
    if now != times.MEASURED or placed_on_these(edit, now):
        return edit
    was_on = placed_on(edit)
    words = times.load_words(project)
    prefixes = {"keep": KEEP_PREFIX, "creator_cuts": CREATOR_ROW_PREFIX, "picks": PICK_ID_PREFIX}
    moved = {**edit, MOVED_FROM: was_on}
    for key in hers:
        if isinstance(edit.get(key), list):
            moved[key] = [_moved_with_its_words(span, words, was_on, prefixes.get(key, "")) for span in edit[key]]
    return moved


def load_edit(project: Project, duration: float) -> dict[str, Any]:
    """The saved edit, or an empty one. Raises StudioError when it is unusable.

    The spans in it are on the word times the project has now (see
    ``on_the_word_times_of``).
    """
    if not project.edit_path.exists():
        return empty_edit()
    try:
        data = json.loads(project.edit_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise StudioError("The saved edit.json is not valid JSON. Call set_edit to replace it.") from None
    edit = validate_saved_edit(data)
    saved = edit.get("duration")
    if _finite(saved) and abs(saved - duration) > DURATION_DRIFT:
        raise StudioError(
            f"The saved edit was made for a {saved:.2f}s video but this file is {duration:.2f}s. "
            "The video changed; call set_edit to make a new edit."
        )
    return on_the_word_times_of(project, edit)


def save_edit(
    project: Project,
    outcome: EditOutcome,
    *,
    duration: float,
    requested: list[Any],
    auto_tighten: bool,
    gap_length: float | None,
    keep: list[dict[str, Any]] | None = None,
    pace: str | None = None,
    treatment: dict[str, Any] | None = None,
    set_by_claude: dict[str, Any] | None = None,
    creator_cuts: list[dict[str, Any]] | None = None,
    word_times: str | None = None,
    undo: dict[str, Any] | None = None,
    picks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write edit.json atomically and return what was written.

    ``keep`` is the list of spans the creator restored and ``creator_cuts``
    the cuts she made by hand. Both outlive every ``set_edit``: pass the
    previous edit's lists so a new edit can't forget them.
    ``treatment`` is the pace and take-out switches the edit was built with,
    and ``set_by_claude`` the ones Claude last asked for; get_edit compares
    the two to tell Claude what the creator changed. ``word_times`` says
    which word times the cuts were placed on (see ``word_times.py``); the
    way measured times were made is saved beside it. ``undo`` holds what
    the page needs to take back her last change.
    ``picks`` are the filler words Claude picked; None means Claude has not
    picked, which an empty list does not.
    """
    edit = {
        "version": EDIT_VERSION,
        "video": str(project.video),
        "duration": round(duration, 3),
        "updated_at": now_iso(),
        "auto_tighten": auto_tighten,
        "gap_length": gap_length,
        "pace": pace,
        "cuts": outcome.cuts,
        "requested": requested,
        "rejected": outcome.rejected,
        "keep": keep or [],
    }
    if treatment is not None:
        edit["treatment"] = treatment
    if set_by_claude is not None:
        edit["set_by_claude"] = set_by_claude
    if creator_cuts:
        edit["creator_cuts"] = creator_cuts
    if word_times is not None:
        edit["word_times"] = word_times
        if times.made_as(word_times) is not None:
            edit[MADE_AS] = times.made_as(word_times)
    if undo is not None:
        edit["undo"] = undo
    if picks is not None:
        edit["picks"] = picks
    write_json_atomic(project.edit_path, edit)
    return edit
