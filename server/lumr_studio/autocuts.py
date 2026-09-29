"""Automatic micro-cuts: the pause and filler trims ClipForge's pacing layer plans.

One function, shared by ``analyze_take`` (which reports the candidates) and
the edit recipe in ``treatment.py`` (which applies them), so the count Claude
sees is the count it gets.

Every trim carries a ``kind``, the creator's name for it: ``pauses``,
``fillers`` (named for what it removes: "like", "so", "you know" and the
like, far more often than an "um") or ``repeats`` (repeated words, which
ClipForge calls stutters).
The kind comes from the label ClipForge puts at the head of the trim's
reason, the same one its own ``classify`` reads. The treatment page turns
each kind on or off.

One correction to that label. ClipForge lands a trim's edges on measured
silence after it has named it, so a trim it calls a filler can end up over
the pause beside the word and take no word at all: on the test footage 9 of
17 filler trims at a middle pace. Such a trim is a pause and is filed as one
(``file_by_what_it_takes``), so switching filler words off leaves the pauses
to their own switch.

What a trim may take (``trims_taken``). A trim is planned for a pause, but a
word can sit inside the span it lands on: a word said quietly counts as
silence to the detector. So every trim is read against the words it holds.

* A word that no switch owns is never taken. The trim is shortened around
  it. On the test footage 34 such words went at the hardest pace before
  this rule, "take" in "You can just take one step at a time" among them.
* A filler belongs to the Filler words switch, whatever its trim was named
  for: on, the trim may take it; off, the trim is shortened around it and
  the pause beside it still goes. A filler is a word on the filler list of
  ClipForge's level for the pace (``engine_level``), the list its planner
  reads. The plugin keeps no list of its own.
* The first of a word said twice belongs to Stutters the same way, at the
  paces where ClipForge cuts them.
* A word that is judged by reading (``JUDGED_BY_READING``, today "like") is
  never taken, though ClipForge lists it as a filler. No rule can tell "not
  feeling like I know the way" from "into like three
  sections". Claude reads each one and picks the fillers, and the creator
  has the Filler likes switch for those. A rule and a judgment never share
  a word.

Planning depends only on its inputs, so results are kept: in a small
in-process cache, and in ``plans.json`` in the project folder when the caller
says where that is. On a 21 minute take with measured word times one plan
takes 0.1 s at the gentlest pace and 0.6 s at the hardest, 2 s for all six,
and the page shows all six. Saved plans make the page's first load after a
restart as quick as the ones after it.

The sliders on the page can ask for 252 settings. The plans of the six stops
are always kept; of the others the file keeps the most recent
``CUSTOM_PLANS_KEPT``, and drops the oldest first.
"""

from __future__ import annotations

import bisect
import copy
import hashlib
import json
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lumr_studio.engine.audio_boundaries import Silence
from lumr_studio.engine.longform import EDIT_LEVELS, EditLevel
from lumr_studio.engine.microcut_pacing import (
    DEFAULT_GAP_LENGTH,
    DEFAULT_RHYTHM,
    GAP_LENGTH_MAX,
    GAP_LENGTH_MIN,
    _level_for_gap_length,
    plan_microcuts,
)
from lumr_studio.errors import StudioError
from lumr_studio.project import write_json_atomic
from lumr_studio.sounds import is_filler
from lumr_studio.transcript import is_event, plain_text

log = logging.getLogger(__name__)

# The kinds of automatic trim, as the creator reads them, in the order shown.
TRIM_KINDS: dict[str, str] = {
    "pauses": "Long pauses",
    "fillers": "Filler words",
    "repeats": "Stutters",
}
# ClipForge's reason labels ("pause: 1.2s", "filler: um", "stutter: the the")
# mapped to the kinds above. An unknown label counts as a pause, as in
# ClipForge's own ``classify``.
_LABEL_TO_KIND = {"pause": "pauses", "filler": "fillers", "stutter": "repeats", "repeat": "repeats"}
DEFAULT_TRIM_KIND = "pauses"
FILLER_TRIM_KIND = "fillers"
REPEAT_TRIM_KIND = "repeats"
# A removal shorter than this removes nothing worth removing. The one place
# that says so: what is left of a trim shortened around a word, a trim
# shortened to leave a pause (``pace``) and a cut of Claude's once placed
# (``edit``) are all held to it.
SHORTEST_PIECE_SECONDS = 0.1

# The words that are judged by reading and so are never taken by the pace,
# each as ``plain_text`` gives it. Claude reads every place one is said and
# picks the fillers (``word_finder``, ``edit.PICK_KIND``). On the test
# footage the pace took 10 "like"s at its usual stop, and three of the ten
# carried the sentence: "ask what it was like." Said twice in a row it is
# still judged by reading: one of the two pairs on the test footage is the
# end of one sentence and the start of the next ("like. Like,").
JUDGED_BY_READING = frozenset({"like"})

# Every word and two-word filler ClipForge counts as a filler at some level,
# less the words judged by reading. The plugin keeps no list of its own. A
# word on none of these is never named under the Filler words switch.
FILLER_WORDS = frozenset().union(*(level.filler_words for level in EDIT_LEVELS.values())) - JUDGED_BY_READING
FILLER_PAIRS = frozenset().union(*(level.bigram_fillers for level in EDIT_LEVELS.values()))

# How many plans the cache holds: six paces for a few videos, and the
# slider positions the creator is trying.
PLAN_CACHE_SIZE = 48

# Plans saved with a project, so a restart does not plan again.
PLANS_FILE = "plans.json"
# Bump when the planner or the shape of a plan changes, so saved plans are made again.
# 4: a trim named for a filler that takes no filler is filed as a pause.
# 5: no trim takes a word that no switch owns, and a filler is one by the pace's own list.
# 6: no trim takes a word that is judged by reading ("like").
PLANS_VERSION = 6
# Plans kept in the file beside the six stops': the settings the creator
# tried last with the sliders. One plan is about 60 KB at the hardest
# settings, so twelve keep the file near 1 MB.
CUSTOM_PLANS_KEPT = 12

_cache: OrderedDict[tuple[Any, ...], list[dict[str, Any]]] = OrderedDict()
_cache_lock = threading.Lock()
_store_lock = threading.Lock()


def forget_plans() -> None:
    """Empty the plan cache, for tests that swap the planner."""
    with _cache_lock:
        _cache.clear()


def check_gap_length(gap_length: float | None) -> float:
    """Return a usable gap length, or raise StudioError when it is out of range."""
    if gap_length is None:
        return DEFAULT_GAP_LENGTH
    if not GAP_LENGTH_MIN <= gap_length <= GAP_LENGTH_MAX:
        raise StudioError(
            f"gap_length {gap_length} is out of range. Pass a pause length between "
            f"{GAP_LENGTH_MIN} and {GAP_LENGTH_MAX} seconds, or omit it for {DEFAULT_GAP_LENGTH}."
        )
    return gap_length


def trim_kind(reason: str) -> str:
    """The take-out kind of a trim from its reason: ``pauses``, ``fillers`` or ``repeats``.

    Reads the label at the head of the reason, after an optional ``auto:``
    prefix, the way ClipForge's ``classify`` does.
    """
    head = str(reason).split(";")[0].strip().removeprefix("auto:").strip()
    return _LABEL_TO_KIND.get(head.split(":")[0].strip().lower(), DEFAULT_TRIM_KIND)


def fillers_in(removed: list[dict[str, Any]], level: EditLevel | None = None) -> list[str]:
    """The fillers among ``removed``, the spoken words of one trim in the order said.

    Each reads as the creator would name it: lower case, and a two-word
    filler as one ("you know"). A word on no filler list is left out, and
    so is a word judged by reading. ``level`` is the ClipForge level whose
    lists count; left out, a filler at any level counts.
    """
    singles, pairs = filler_lists(level)
    out: list[str] = []
    texts = [plain_text(str(w.get("word", ""))) for w in removed]
    i = 0
    while i < len(texts):
        pair = " ".join(texts[i:i + 2])
        if pair in pairs:
            out.append(pair)
            i += 2
            continue
        if texts[i] in singles:
            out.append(texts[i])
        i += 1
    return out


def filler_lists(level: EditLevel | None) -> tuple[frozenset[str], frozenset[str]]:
    """The fillers the pace may take at ``level``, one word and two: ClipForge's lists less the words judged by reading.

    Without a ``level``, the fillers of any level.
    """
    if level is None:
        return FILLER_WORDS, FILLER_PAIRS
    return frozenset(level.filler_words) - JUDGED_BY_READING, frozenset(level.bigram_fillers)


def engine_level(gap_length: float | None) -> EditLevel:
    """ClipForge's level for a shortest pause: the one whose filler lists its planner reads at that pace."""
    return EDIT_LEVELS[_level_for_gap_length(check_gap_length(gap_length))]


@dataclass(frozen=True)
class Said:
    """A take's spoken words in the order said, to find the ones a span holds."""

    words: list[dict[str, Any]]
    mids: list[float]

    @classmethod
    def of(cls, words: list[dict[str, Any]]) -> Said:
        spoken = sorted(
            (w for w in words if not is_event(w)), key=lambda w: float(w["start"]) + float(w["end"])
        )
        return cls(words=spoken, mids=[(float(w["start"]) + float(w["end"])) / 2 for w in spoken])

    def held(self, start: float, end: float) -> range:
        """Positions of the words ``[start, end]`` takes: those whose middle lies inside, the rule every cut goes by."""
        return range(bisect.bisect_right(self.mids, start), bisect.bisect_left(self.mids, end))

    def owners(self, held: range, level: EditLevel | None) -> list[str | None]:
        """Which switch owns each of the ``held`` words: a kind of ``TRIM_KINDS``, or None for a word no switch owns.

        A filler belongs to ``FILLER_TRIM_KIND``. The first of a word said
        twice belongs to ``REPEAT_TRIM_KIND`` at a level that cuts repeats,
        by ClipForge's own test. A word judged by reading belongs to none,
        said once or twice. Without a ``level`` a filler at any level
        counts and so does any word said twice.
        """
        singles, pairs = filler_lists(level)
        repeats = level is None or level.cut_repeats
        texts = [plain_text(str(self.words[i].get("word", ""))) for i in held]
        out: list[str | None] = [None] * len(texts)
        i = 0
        while i < len(texts):
            if i + 1 < len(texts) and f"{texts[i]} {texts[i + 1]}" in pairs:
                out[i] = out[i + 1] = FILLER_TRIM_KIND
                i += 2
                continue
            after = held.start + i + 1
            if texts[i] in JUDGED_BY_READING:
                pass
            elif texts[i] in singles:
                out[i] = FILLER_TRIM_KIND
            elif repeats and len(texts[i]) > 1 and after < len(self.words) \
                    and plain_text(str(self.words[after].get("word", ""))) == texts[i]:
                out[i] = REPEAT_TRIM_KIND
            i += 1
        return out


def span_without(start: float, end: float, holes: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """``[start, end]`` with every hole taken out: the pieces left, in order."""
    pieces = []
    at = start
    for a, b in sorted(holes):
        if b <= at or a >= end:
            continue
        if a > at:
            pieces.append((at, a))
        at = max(at, b)
    if at < end:
        pieces.append((at, end))
    return pieces


def _filed(kind: str, start: float, end: float, said: Said, level: EditLevel | None,
           on: frozenset[str] | None = None) -> str:
    """The kind a trim over ``[start, end]`` named ``kind`` is filed under, by what it takes.

    A trim named for a word (a filler, a repeat) that takes no such word is
    a pause. With ``on``, the switches that are on, a word whose switch is
    off is not taken.
    """
    if kind == DEFAULT_TRIM_KIND:
        return kind
    held = said.held(start, end)
    owners = said.owners(held, level)
    if kind == REPEAT_TRIM_KIND and level is None:
        return kind if len(held) else DEFAULT_TRIM_KIND  # any word it takes was named by ClipForge
    takes = kind in owners and (on is None or kind in on)
    return kind if takes else DEFAULT_TRIM_KIND


def file_by_what_it_takes(trims: list[dict[str, Any]], words: list[dict[str, Any]],
                          level: EditLevel | None = None) -> None:
    """Give each trim its ``kind``, in place: from its reason, corrected by the words it takes.

    A trim that takes no spoken word is a pause. So is a trim named for a
    filler that takes no filler (``fillers_in``). A word is taken when its
    middle lies inside the trim, the rule every cut goes by. ``level`` is
    the ClipForge level whose filler lists count; left out, any level's do.
    """
    said = Said.of(words)
    for trim in trims:
        trim["kind"] = _filed(trim_kind(trim["reason"]), float(trim["start"]), float(trim["end"]), said, level)


def around(trim: dict[str, Any], holes: list[tuple[float, float]], said: Said,
           level: EditLevel | None = None, on: frozenset[str] | None = None) -> list[dict[str, Any]]:
    """``trim`` shortened around ``holes``, the words it must leave in: the pieces worth cutting.

    A piece under ``SHORTEST_PIECE_SECONDS`` goes. A piece of a filler or
    stutter trim that no longer holds such a word is a pause, and says so.
    With no hole inside it the trim comes back as it is.
    """
    start, end = float(trim["start"]), float(trim["end"])
    inside = [(a, b) for a, b in holes if a < end and start < b]
    if not inside:
        return [trim]
    pieces = []
    for a, b in span_without(start, end, inside):
        if b - a < SHORTEST_PIECE_SECONDS:
            continue
        piece = {**trim, "start": round(a, 3), "end": round(b, 3)}
        kind = str(trim.get("kind") or trim_kind(str(trim.get("reason", ""))))
        if _filed(kind, a, b, said, level, on) != kind:
            piece["kind"], piece["reason"] = DEFAULT_TRIM_KIND, f"pause: {b - a:.1f}s"
        pieces.append(piece)
    return pieces


def trims_taken(
    planned: list[dict[str, Any]],
    words: list[dict[str, Any]],
    *,
    gap_length: float | None = None,
    on: frozenset[str] = frozenset(TRIM_KINDS),
) -> list[dict[str, Any]]:
    """The planned trims as the switches ``on`` leave them. The one place that says what a trim may take.

    A trim of a kind that is off is dropped. Each other trim is shortened
    around the words it holds and may not take: a word no switch owns,
    always, and a word whose switch is off (see the module docstring).
    ``gap_length`` is the shortest pause the plan cuts; it names the
    ClipForge level whose filler lists count. What is left of a trim named
    for a word, once that word stays in, is a pause, and goes only when
    pauses are on.
    """
    level = engine_level(gap_length)
    said = Said.of(words)
    taken = []
    for trim in planned:
        kind = str(trim.get("kind") or trim_kind(str(trim.get("reason", ""))))
        if kind not in on:
            continue
        held = said.held(float(trim["start"]), float(trim["end"]))
        stay = [
            (float(said.words[i]["start"]), float(said.words[i]["end"]))
            for i, owner in zip(held, said.owners(held, level)) if owner not in on
        ]
        taken += [piece for piece in around({**trim, "kind": kind}, stay, said, level, on) if piece["kind"] in on]
    return taken


def words_key(words: list[dict[str, Any]]) -> tuple[Any, ...]:
    """A hashable fingerprint of a transcript, for caches keyed on it."""
    return tuple((str(w.get("word", "")), float(w["start"]), float(w["end"]), w.get("type")) for w in words)


def silences_key(silences: list[Silence]) -> tuple[tuple[float, float], ...]:
    """A hashable fingerprint of measured silences."""
    return tuple((float(s.start), float(s.end)) for s in silences)


def take_mark(words: list[dict[str, Any]], silences: list[Silence], duration: float) -> str:
    """A short mark of the words, silences and length of a take. The same take gives the same mark."""
    said = [(str(w.get("word", "")), float(w["start"]), float(w["end"]), w.get("type")) for w in words]
    quiet = [(float(q.start), float(q.end)) for q in silences]
    return hashlib.sha1(json.dumps([said, quiet, round(float(duration), 3)]).encode("utf-8")).hexdigest()


def _saved_plans(path: Path, mark: str) -> dict[str, dict[str, Any]]:
    """The plans saved at ``path`` for the take ``mark``: ``{name: {stop, trims}}``, oldest first.

    None when the file is for another take or can't be read.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict) or data.get("version") != PLANS_VERSION or data.get("take") != mark:
        return {}
    plans = data.get("plans")
    if not isinstance(plans, dict):
        return {}
    return {k: v for k, v in plans.items() if isinstance(v, dict) and isinstance(v.get("trims"), list)}


def _within_the_cap(plans: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """``plans`` with every stop's plan, and of the others the ``CUSTOM_PLANS_KEPT`` saved last."""
    others = [name for name, plan in plans.items() if not plan.get("stop")]
    dropped = set(others[:max(0, len(others) - CUSTOM_PLANS_KEPT)])
    return {name: plan for name, plan in plans.items() if name not in dropped}


def _save_plan(path: Path, mark: str, name: str, planned: list[dict[str, Any]], stop: bool) -> None:
    """Add one plan to the file at ``path``. A plan that can't be saved is only planned again later."""
    with _store_lock:
        plans = _saved_plans(path, mark)
        plans.pop(name, None)  # saved again, it is the newest
        plans[name] = {"stop": stop, "trims": planned}
        try:
            write_json_atomic(path, {"version": PLANS_VERSION, "take": mark, "plans": _within_the_cap(plans)})
        except OSError as why:
            log.warning("could not save the plan at %s: %s", path, why)


def plan_auto_cuts(
    words: list[dict[str, Any]],
    *,
    duration: float,
    silences: list[Silence],
    gap_length: float | None = None,
    rhythm: float = DEFAULT_RHYTHM,
    saved_in: Path | None = None,
    stop: bool = True,
) -> list[dict[str, Any]]:
    """``[{start, end, reason, kind}]`` automatic trims on pauses, fillers and stutters.

    No trim takes a word that no switch owns (``trims_taken``).
    ``gap_length`` is the shortest pause that gets trimmed (default
    ``DEFAULT_GAP_LENGTH``). ``rhythm`` is ClipForge's 1 to 5 spacing dial:
    lower keeps more speech between two cuts. At its default every qualifying
    cut fires. ``saved_in`` is a folder to keep the plan in (the project's);
    a plan saved there for this very take is read and not made again.
    ``stop`` says the setting is one of the pace's stops, whose plans are
    always kept; the others fall under ``CUSTOM_PLANS_KEPT``.
    Returns a fresh copy each call, so callers may change it.
    """
    shortest = check_gap_length(gap_length)
    key = (words_key(words), round(float(duration), 3), silences_key(silences), shortest, float(rhythm))
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return copy.deepcopy(hit)
    planned = None
    if saved_in is not None:
        mark, name = take_mark(words, silences, duration), f"{shortest}/{float(rhythm)}"
        planned = _saved_plans(saved_in / PLANS_FILE, mark).get(name, {}).get("trims")
    if planned is None:
        planned = plan_microcuts(words, gap_length=shortest, rhythm=rhythm, silences=silences, duration=duration)
        file_by_what_it_takes(planned, words, engine_level(shortest))
        planned = trims_taken(planned, words, gap_length=shortest)
        if saved_in is not None:
            _save_plan(saved_in / PLANS_FILE, mark, name, planned, stop)
    with _cache_lock:
        _cache[key] = copy.deepcopy(planned)
        while len(_cache) > PLAN_CACHE_SIZE:
            _cache.popitem(last=False)
    return copy.deepcopy(planned)


# A sound with no gap at all from the word right before it, and this short
# or shorter, is that word's own voice running on, not a fresh sound: on the
# real take "in" (aligned end 299.44) ran on 0.119 s into a "[vocalization]"
# logged right there, voiced 0.5, the edge of held. Longer than this a
# run-on is its own filler, held or not: the measured "uhh" after "And" ran
# on 0.169 s and is one. Matches ``GAP_LENGTH_MIN``, the shortest pause any
# pace cuts and the same gap ``word_times.RUNS_ON_SECONDS`` calls no gap.
WORD_TAIL_MAX_SECONDS = GAP_LENGTH_MIN
# How close a sound's start must sit to the word before it to call it no gap
# at all. Times are saved to the millisecond; this gives a little past that.
WORD_TAIL_TOUCH_SECONDS = 0.002


def _is_word_tail(sound: dict[str, Any], spoken_ends: list[float]) -> bool:
    """Whether ``sound`` is the trailing voice of the spoken word right before it, not a sound of its own.

    True when some word ends within ``WORD_TAIL_TOUCH_SECONDS`` of where
    ``sound`` starts, no gap at all, and ``sound`` is
    ``WORD_TAIL_MAX_SECONDS`` or shorter.
    """
    if float(sound["end"]) - float(sound["start"]) > WORD_TAIL_MAX_SECONDS:
        return False
    start = float(sound["start"])
    return any(abs(end - start) <= WORD_TAIL_TOUCH_SECONDS for end in spoken_ends)


def vocalization_filler_trims(sounds: list[dict[str, Any]], words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``[{start, end, reason, kind}]``: one whole-removal trim for every filler vocalization among ``sounds``.

    ClipForge's own planner never proposes a cut over a "[vocalization]" (a
    sound of the transcript is protected end to end, see ``word_times.py``),
    so a held "umm" or "uhh" needs a trim of its own. A sound label already
    holds nothing but its audible span (``word_times._sounds_clear_of_words``
    trims it to where it sounds, past any silent lead-in or tail the
    transcript's loose timing left on it), so taking it whole takes exactly
    what is heard and nothing of a neighbouring word; the pause it sat in is
    left to ``pace.leave_pauses`` like any other trim's, held so that shortening
    never eats into it (``treatment.make_edit`` adds these spans to ``hers``).
    Filed under ``FILLER_TRIM_KIND``, the switch "um" and "so" answer to, so
    one switch covers every filler, said or made with the voice alone.

    Left alone: a sound too short to be worth a cut once trimmed
    (``SHORTEST_PIECE_SECONDS``); a breath or a laugh train
    (``sounds.is_filler``); and a short run-on with no gap from the word
    before it, which is that word's own tail, not a sound of its own
    (``_is_word_tail``, ``words``).
    """
    spoken_ends = [float(w["end"]) for w in words if not is_event(w)]
    return [
        {"start": float(s["start"]), "end": float(s["end"]), "reason": "filler: sound", "kind": FILLER_TRIM_KIND}
        for s in sounds
        if is_filler(s) and not _is_word_tail(s, spoken_ends)
        and float(s["end"]) - float(s["start"]) >= SHORTEST_PIECE_SECONDS
    ]
