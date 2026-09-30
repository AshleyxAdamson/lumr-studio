"""The treatment: how hard an edit trims, what it takes out, and what the creator cuts and keeps.

The treatment page (``treatment_page/index.html``, served by
``review_server.py``) is where the creator tunes what Claude did. Two kinds of
decision never share a control there:

* Rules: a pace dial and a switch per kind of automatic trim (long pauses,
  filler words, stutters), each with a count.
* Judgment: Claude's own cuts, one row each with Claude's reason, grouped by
  the kind Claude gave them. Each can be put back or kept out.

On top of both the creator works on the words themselves. She can flag any
stretch "keep no matter what", cut any words by hand, and bring any removed
word back. Her cuts and keeps are hers: no pace, switch or new edit from
Claude undoes them, and when two of her own actions meet, the newer one wins.

Two more controls sit under those. Fine tune: two sliders that set the
shortest pause cut and the rhythm by hand; the pace then reads ``custom``
(see ``pace.level_for``). Filler likes: the single filler words Claude picked
by reading, with one switch for all of them (see ``edit.SWITCHES``).

This module owns:

* ``make_edit``, the one recipe that builds an edit from Claude's cuts, the
  creator's cuts and a treatment. ``tools.set_edit`` and every page action
  call it.
* The page's state (``page_state``) and its actions (``change_treatment``,
  ``set_cut_state``, ``add_cut``, ``remove_cut``, ``add_keep``,
  ``remove_keep``, ``undo``, ``new_samples``, ``save_as_usual``). Each action
  rebuilds the edit with the recipe, saves it, and answers the new state
  with what changed.
* The creator's usual settings, one small file for every video.

The state also shows two things this module only reads: the creator's photos
and clips (saved by the overlay tools, view only here) and the latest export
(a render job; the server passes in how to read it).

Kept spans (cuts put back, stretches flagged) live in ``edit["keep"]`` and
outlive every later edit, by Claude or by a change of pace.
"""

from __future__ import annotations

import bisect
import copy
import json
import logging
import math
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lumr_studio import edit as edits
from lumr_studio import taste, word_times
from lumr_studio.autocuts import (
    FILLER_PAIRS,
    FILLER_TRIM_KIND,
    TRIM_KINDS,
    Said,
    check_gap_length,
    engine_level,
    fillers_in,
    forget_plans,
    plan_auto_cuts,
    silences_key,
    trim_kind,
    trims_taken,
    vocalization_filler_trims,
    words_key,
)
from lumr_studio.engine.audio_boundaries import Silence
from lumr_studio.engine.render import kept_segments
from lumr_studio.errors import StudioError
from lumr_studio.overlays import HIDDEN, load_overlays, page_row, place_on_edit
from lumr_studio.pace import (
    CUSTOM,
    CUSTOM_LABEL,
    DEFAULT_PACE,
    PACES,
    Pace,
    PauseReport,
    check_fine,
    fine_ranges,
    get_pace,
    level_for,
    settle_trims,
    speech_kept,
)
from lumr_studio.project import Project, now_iso, projects_root, write_json_atomic
from lumr_studio.render_jobs import nothing_exported
from lumr_studio.review import FLAG_LABELS, page_words
from lumr_studio.samples import (
    SAMPLE_LABELS,
    EditedClock,
    Edits,
    busy_sections,
    choose_samples,
    clusters,
    plain_why,
    sentence_spans,
)
from lumr_studio.silences import SilenceProvider
from lumr_studio.sounds import LAUGH, LIKELY, protected_spans
from lumr_studio.transcript import is_event, plain_text

log = logging.getLogger(__name__)

Words = list[dict[str, Any]]
Span = tuple[float, float]
LabelsFor = Callable[[Project, Words], list[dict[str, Any]]]
JoinRows = Callable[..., list[dict[str, Any]]]
# The latest export of a project, given its saved edit. See render_jobs.export_status.
ExportFor = Callable[[dict[str, Any]], dict[str, Any]]

USUAL_FILE = "usual.json"
TREATMENT_FILE = "treatment.json"
TREATMENT_VERSION = 1

# Row states on the page.
KEPT_OUT, PUT_BACK = "kept_out", "put_back"
CUT_STATES = (KEPT_OUT, PUT_BACK)
# Where a kept span came from: a cut the creator put back, a stretch they
# flagged, or words they brought back one by one. A keep saved before origins
# existed has none and reads as put back.
ORIGIN_PUT_BACK, ORIGIN_FLAGGED, ORIGIN_EXACT = "put_back", "flagged", "exact"
KEPT_BY_YOU = "kept by you"
BROUGHT_BACK_BY_YOU = "brought back by you"

# Why a word on the page is removed, the fifth value of each ``words`` entry.
PLAYING, REMOVED_AUTO, REMOVED_BY_CLAUDE, REMOVED_BY_YOU, REMOVED_AS_PICKED = 0, 1, 2, 3, 4
# The word the Filler likes switch covers. ``find_words`` takes any word;
# this round the switch and the skill cover this one.
LIKES_WORD = "like"
# The creator's cuts are rows with ids of their own, so a route for Claude's
# cuts can never be handed one of hers.
CLAUDE_ROW_PREFIX, CREATOR_ROW_PREFIX, KEEP_PREFIX = "c", edits.CREATOR_ROW_PREFIX, edits.KEEP_PREFIX
# A creator knocks out words by the dozen; this many means something is wrong.
MAX_CREATOR_CUTS = 2000
# Only the filler switch says which words it takes out. The others carry an
# empty list, so every entry has the same shape.
KINDS_THAT_LIST_WORDS = frozenset({FILLER_TRIM_KIND})
# What a filler vocalization (a held "umm" or "uhh", no word inside it) is
# named in the filler words list, beside "um" and "so" (``taken_by_kind``).
SOUND_FILLER_LABEL = "sound"

# A note on a kept part longer than this is not from the page.
MAX_NOTE_CHARS = 200
# The words a kept part covers are shown up to about this many characters.
KEEP_TEXT_CHARS = 80
# Words either side of a cut, and the most removed words shown before "...".
CONTEXT_WORDS = 8
REMOVED_WORDS_SHOWN = 12
# A creator flags a few parts; this many means something is wrong.
MAX_KEEPS = 500
# Recipes kept in memory: six paces, a few switch settings, a few videos.
RECIPE_CACHE_SIZE = 64

# A sample's ``detail``: what the edit does there, in words the page shows as
# they are. None says "trim"; time out reads as a clock, never as seconds.
DETAIL_NOTHING_OUT = "nothing out"
# Under this a sample reads "nothing out": it would round to 0:00.
SHORTEST_TIME_OUT = 0.5
# A cluster always removes something, so its time out never reads 0:00.
SHORTEST_TIME_OUT_SHOWN = 1.0
DETAIL_PUT_BACK = "put back"
DETAIL_LAUGH_WHOLE = "laugh whole"
DETAIL_LAUGH_CUT = "laugh cut"


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _overlaps(a: float, b: float, c: float, d: float) -> bool:
    return a < d and c < b


# ── Settings ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Treatment:
    """How an edit cuts by rule: a pace, the switches that are on, and the values behind the pace.

    ``pace`` is one of the six stops, or ``custom`` with ``fine`` holding the
    creator's own ``(gap_length, rhythm)`` from the sliders. ``gap_length``
    is a shortest-pause override on a stop, set only by Claude through
    set_edit; the page never sets it, and a new pace clears it. ``take_out``
    holds the switches that are on, of ``edit.SWITCHES``.
    """

    pace: str = DEFAULT_PACE
    take_out: frozenset[str] = frozenset(edits.SWITCHES)
    gap_length: float | None = None
    fine: tuple[float, float] | None = None

    @property
    def trims(self) -> frozenset[str]:
        """The kinds of automatic trim that are on."""
        return self.take_out & frozenset(TRIM_KINDS)

    @property
    def picks_on(self) -> bool:
        """Whether the filler words Claude picked are taken out."""
        return edits.PICK_KIND in self.take_out

    @property
    def custom(self) -> bool:
        return self.pace == CUSTOM

    def level(self) -> Pace:
        """The level this treatment cuts at. See ``pace.level_for``."""
        return level_for(self.pace, self.fine)

    def shortest_pause(self) -> float:
        """The shortest pause that gets cut: Claude's override when there is one, else the level's."""
        return check_gap_length(self.gap_length) if self.gap_length is not None else self.level().gap_length

    def fine_values(self) -> dict[str, float]:
        """Where the two sliders stand: ``{gap_length, rhythm}``."""
        return {"gap_length": self.shortest_pause(), "rhythm": float(self.level().rhythm)}

    def as_dict(self) -> dict[str, Any]:
        return {
            "pace": self.pace,
            "take_out": {k: k in self.take_out for k in edits.SWITCHES},
            "gap_length": self.gap_length,
            "fine": self.fine_values(),
        }

    @classmethod
    def from_dict(cls, data: Any) -> Treatment:
        """A treatment read from a saved file. Raises StudioError when it can't be used.

        A file saved before a switch existed has that switch on.
        """
        if not isinstance(data, dict):
            raise StudioError("The saved settings are not an object. Save them again from the page.")
        switches = data.get("take_out", {})
        if not isinstance(switches, dict):
            raise StudioError("The saved take-out switches are not an object. Save them again from the page.")
        take_out = frozenset(k for k in edits.SWITCHES if switches.get(k, True) is True)
        if str(data.get("pace") or "").strip().lower() == CUSTOM:
            fine = data.get("fine")
            if not isinstance(fine, dict) or not _finite(fine.get("gap_length")) or not _finite(fine.get("rhythm")):
                raise StudioError("The saved setting of your own has lost its two values. Pick a pace on the page.")
            return cls(pace=CUSTOM, take_out=take_out, fine=check_fine(float(fine["gap_length"]), float(fine["rhythm"])))
        gap = data.get("gap_length")
        return cls(
            pace=get_pace(data.get("pace")).name, take_out=take_out,
            gap_length=float(gap) if _finite(gap) else None,
        )


def usual_path():
    """The creator's usual settings: one file in the projects folder, for every video."""
    return projects_root() / USUAL_FILE


def load_usual() -> Treatment | None:
    """The saved usual, or None when the creator hasn't saved one.

    A usual that can't be read is treated as none, since it only picks a
    starting point.
    """
    path = usual_path()
    if not path.exists():
        return None
    try:
        return Treatment.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError, StudioError):
        return None


def starting_treatment() -> Treatment:
    """Where a video with no saved treatment starts: the usual, else ``DEFAULT_PACE`` with every kind on."""
    return load_usual() or Treatment()


def saved_treatment(edit: dict[str, Any]) -> Treatment:
    """The treatment the saved edit was built with.

    An edit saved before treatments existed reads as its pace with every kind
    on when it tightened automatically, else with none.
    """
    if isinstance(edit.get("treatment"), dict):
        return Treatment.from_dict(edit["treatment"])
    tightened = bool(edit.get("auto_tighten"))
    gap = edit.get("gap_length")
    start = starting_treatment()
    return Treatment(
        pace=get_pace(edit.get("pace") or (DEFAULT_PACE if start.custom else start.pace)).name,
        take_out=frozenset(edits.SWITCHES) if tightened else frozenset({edits.PICK_KIND}),
        gap_length=float(gap) if _finite(gap) else None,
    )


# ── The recipe ────────────────────────────────────────────────────────────────


@dataclass
class MadeEdit:
    """What ``make_edit`` built, ready to save and to report."""

    outcome: edits.EditOutcome
    treatment: Treatment
    level: Pace
    shortest: float | None
    planned: int = 0
    pauses: PauseReport = field(default_factory=PauseReport)
    # Edges of automatic removals the last check moved out of a word or a
    # sound of the transcript, where a removal met a part of it quieter than
    # the measured silences (``edit.keep_words_whole``).
    edges_moved_out_of_words: int = 0


_recipes: OrderedDict[tuple[Any, ...], MadeEdit] = OrderedDict()
_recipes_lock = threading.Lock()


def forget_recipes() -> None:
    """Empty the recipe, plan and placement caches, for tests that swap the planner."""
    with _recipes_lock:
        _recipes.clear()
    forget_plans()
    edits.forget_placements()


def _recipe_key(cuts: list[Any], words: Words, duration: float, silences: list[Silence],
                protected: list[Span], keeps: list[Span], treatment: Treatment, override_keeps: bool,
                creator_cuts: list[dict[str, Any]], exact_keeps: list[Span],
                picks: list[dict[str, Any]], filler_trims: list[dict[str, Any]]) -> tuple[Any, ...]:
    return (
        json.dumps(cuts, sort_keys=True, default=str), words_key(words), round(float(duration), 3),
        silences_key(silences), tuple(protected), tuple(keeps), treatment, override_keeps,
        tuple((float(c["start"]), float(c["end"])) for c in creator_cuts), tuple(exact_keeps),
        tuple((float(c["start"]), float(c["end"])) for c in picks),
        tuple((float(t["start"]), float(t["end"])) for t in filler_trims),
    )


def make_edit(
    cuts: list[Any],
    words: Words,
    duration: float,
    *,
    treatment: Treatment,
    silences: list[Silence],
    sounds: list[dict[str, Any]],
    keeps: list[Span],
    override_keeps: bool = False,
    creator_cuts: list[dict[str, Any]] | None = None,
    exact_keeps: list[Span] | None = None,
    plans_in: Path | None = None,
    picks: list[dict[str, Any]] | None = None,
) -> MadeEdit:
    """Build an edit from Claude's ``cuts``, the creator's own and a ``treatment``. Pure; results are cached.

    The recipe: plan the automatic trims for the pace, keep what the
    switches that are on may take (``autocuts.trims_taken``: no trim takes a
    word that no switch owns), build the edit with Claude's cuts and the
    creator's (trims near a joke or inside a kept span are dropped there),
    then shorten each trim so its join leaves the pause the pace allows.
    Last, no automatic removal keeps an edge inside a word or a sound of the
    transcript (``edit.keep_words_whole``).
    ``sounds`` are the
    sound labels; their laughs become the protected spans, and a filler
    vocalization (``sounds.is_filler``: a held "umm" or "uhh", not the tail
    of the word before it) becomes a trim of its own while Filler words is
    on (``autocuts.vocalization_filler_trims``), whatever the pace. Its span
    is held the way her own cuts are: shortening a trim for the pause it
    joined never eats into it, and the last check never widens or drops it,
    since what it removes is already exactly the sound. ``keeps``
    are the kept spans that stop a cut whole, ``exact_keeps`` the words
    brought back one by one (see ``edit.build_edit``). ``plans_in`` is the
    project folder, where the plan of each pace is kept from one run to the
    next. ``picks`` are the filler words Claude picked; they are taken out
    while their switch is on. Raises StudioError when the result would
    remove the whole video.
    """
    level = treatment.level()
    protected = protected_spans(sounds, words)
    creator_cuts, exact_keeps = list(creator_cuts or []), list(exact_keeps or [])
    picks = list(picks or []) if treatment.picks_on else []
    filler_trims = vocalization_filler_trims(sounds, words)
    key = _recipe_key(cuts, words, duration, silences, protected, keeps, treatment, override_keeps,
                      creator_cuts, exact_keeps, picks, filler_trims)
    with _recipes_lock:
        hit = _recipes.get(key)
        if hit is not None:
            _recipes.move_to_end(key)
            return copy.deepcopy(hit)

    auto: list[dict[str, Any]] = []
    shortest = None
    if treatment.trims:
        shortest = treatment.shortest_pause()
        on_a_stop = not treatment.custom and treatment.gap_length is None
        planned = plan_auto_cuts(words, duration=duration, silences=silences, gap_length=shortest,
                                 rhythm=level.rhythm, saved_in=plans_in, stop=on_a_stop)
        auto = trims_taken(planned, words, gap_length=shortest, on=treatment.trims)
    if FILLER_TRIM_KIND in treatment.trims:
        auto = auto + filler_trims
    outcome = edits.build_edit(
        cuts, words, duration, auto_cuts=auto, silences=silences,
        keeps=keeps, protected=protected, override_keeps=override_keeps,
        creator_cuts=creator_cuts, exact_keeps=exact_keeps, picks=picks,
    )
    made = MadeEdit(outcome=outcome, treatment=treatment, level=level, shortest=shortest, planned=len(auto))
    hers = [(c["start"], c["end"]) for c in outcome.creator_placed]
    hers += [(c["start"], c["end"]) for c in outcome.picks if c["status"] == edits.PICK_OUT]
    if FILLER_TRIM_KIND in treatment.trims:
        hers += [(t["start"], t["end"]) for t in filler_trims]
    if treatment.trims:
        outcome.cuts, made.pauses = settle_trims(outcome.cuts, words, silences, level, hers)
    chosen = hers + [(p["start"], p["end"]) for p in outcome.placed if not p["put_back"]]
    outcome.cuts, made.edges_moved_out_of_words = edits.keep_words_whole(outcome.cuts, words, chosen)
    with _recipes_lock:
        _recipes[key] = copy.deepcopy(made)
        while len(_recipes) > RECIPE_CACHE_SIZE:
            _recipes.popitem(last=False)
    return made


def save_made(
    project: Project,
    made: MadeEdit,
    *,
    duration: float,
    requested: list[Any],
    keep: list[dict[str, Any]],
    set_by_claude: dict[str, Any] | None,
    creator_cuts: list[dict[str, Any]] | None = None,
    times: str | None = None,
    undo: dict[str, Any] | None = None,
    picks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write what ``make_edit`` built to edit.json and return the saved edit.

    ``times`` is where the word times the edit was placed on came from
    (``word_times.MEASURED`` or ``ESTIMATED``). ``picks`` is every filler
    word Claude picked, whether its switch is on or off; None when Claude
    has not picked.
    """
    t = made.treatment
    return edits.save_edit(
        project, made.outcome, duration=duration, requested=requested,
        auto_tighten=bool(t.trims), gap_length=t.gap_length if t.trims else None,
        keep=keep, pace=t.pace if t.trims else None,
        treatment=t.as_dict(), set_by_claude=set_by_claude,
        creator_cuts=creator_cuts, word_times=times, undo=undo, picks=picks,
    )


# ── Kept spans ────────────────────────────────────────────────────────────────


def keep_id(keep: dict[str, Any]) -> str:
    """The stable id of a kept span: its saved ``id``, else ``k<start>-<end>``."""
    return str(keep.get("id") or f"k{float(keep['start']):.2f}-{float(keep['end']):.2f}")


def _origin(keep: dict[str, Any]) -> str:
    return str(keep.get("origin") or ORIGIN_PUT_BACK)


def keep_spans_of(keep: list[dict[str, Any]]) -> list[Span]:
    """The kept spans that stop a cut whole: cuts put back and parts flagged to keep."""
    return [(float(k["start"]), float(k["end"])) for k in keep if _origin(k) != ORIGIN_EXACT]


def exact_spans_of(keep: list[dict[str, Any]]) -> list[Span]:
    """The words the creator brought back one by one. A cut splits around these."""
    return [(float(k["start"]), float(k["end"])) for k in keep if _origin(k) == ORIGIN_EXACT]


def widen_to_sentences(start: float, end: float, words: Words) -> Span:
    """``[start, end]`` widened to the whole sentences it touches, a laugh after the last one included.

    A span that touches no sentence (it sits in a pause) is returned as is.
    """
    touched = [s for s in sentence_spans(words) if _overlaps(s[0], s[1], start, end)]
    if not touched:
        return start, end
    return min(start, touched[0][0]), max(end, touched[-1][1])


def _text_between(words: Words, start: float, end: float) -> list[str]:
    return [
        str(w["word"]).strip() for w in words
        if not is_event(w) and start <= (float(w["start"]) + float(w["end"])) / 2 <= end
    ]


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


# ── Loading one project ───────────────────────────────────────────────────────


@dataclass
class Context:
    """Everything an action needs about one project, loaded once per request."""

    project: Project
    duration: float
    words: Words
    silences: list[Silence]
    sounds: list[dict[str, Any]]
    edit: dict[str, Any]
    join_rows: JoinRows
    export_for: ExportFor | None = None
    # Where the word times came from, and the note for the creator when they are estimates.
    times: str = word_times.ESTIMATED
    times_note: str = ""


def _no_labels(_project: Project, _words: Words) -> list[dict[str, Any]]:
    return []


def _real_join_rows(cuts: list[dict[str, Any]], words: Words, duration: float, *, labels: Any = None) -> list[dict[str, Any]]:
    from lumr_studio.joins import join_rows

    return join_rows(cuts, words, duration, labels=labels)


def load_context(
    project: Project,
    *,
    duration: float | None = None,
    silences: SilenceProvider,
    labels_for: LabelsFor | None = None,
    join_rows: JoinRows | None = None,
    export_for: ExportFor | None = None,
) -> Context:
    """The saved edit, words, measured silences and sound labels of ``project``.

    The words come from ``word_times.read``, the one source. A saved edit
    placed on other word times than those is made again on these first (see
    ``bring_up_to_date``). ``export_for`` reads the latest export for the
    state; without it the state says nothing was exported.
    """
    duration = project.duration() if duration is None else duration
    times = word_times.read(project)
    ctx = Context(
        project=project, duration=duration, words=times.words,
        silences=silences(project.video), sounds=(labels_for or _no_labels)(project, times.words),
        edit=edits.load_edit(project, duration), join_rows=join_rows or _real_join_rows,
        export_for=export_for, times=times.source, times_note=times.note,
    )
    bring_up_to_date(ctx)
    return ctx


def saved_picks(edit: dict[str, Any]) -> list[dict[str, Any]] | None:
    """The filler words Claude picked, as saved, or None when Claude has not picked."""
    picks = edit.get("picks")
    return [dict(k) for k in picks] if isinstance(picks, list) else None


def bring_up_to_date(ctx: Context) -> bool:
    """Make the saved edit again when the word times changed under it. True when it did.

    Claude's cuts, the creator's cuts and every kept span are saved in source
    seconds. When the times go from estimated to measured, or measured ones
    are made another way, each one moves with its words as the edit is
    loaded (``edit.on_the_word_times_of``), and here the edit is placed
    again on them. Row ids come from placed edges, so they change. An edit
    with nothing in it is left alone.

    An edit the recipe made is also made again when the recipe would now
    remove something else from the same cuts and settings (the plugin was
    updated, or the sound labels were measured since). Without this the
    creator's next click would carry that difference: one word cut, and the
    page reporting two seconds gone.
    """
    edit = ctx.edit
    if not (edit.get("cuts") or edit.get("requested") or edit.get("creator_cuts")):
        return False
    if edits.MOVED_FROM not in edit and edits.placed_on_these(edit, ctx.times):
        return _remade_if_the_recipe_changed(ctx)
    try:
        _rebuild(ctx, saved_treatment(edit))
    except StudioError as why:
        log.warning("the saved edit could not be made again on the new word times: %s", why)
        return False
    return True


def _removes(cuts: list[dict[str, Any]]) -> list[tuple[float, float, str]]:
    return [(round(float(c["start"]), 3), round(float(c["end"]), 3), str(c.get("source"))) for c in cuts]


def _made_by_the_recipe(edit: dict[str, Any]) -> bool:
    """Whether the recipe made this edit's cuts. One saved by hand, or changed on the older review page, is not."""
    return isinstance(edit.get("treatment"), dict) and not any(c.get("edited_in_review") for c in edit.get("cuts", []))


def _remade_if_the_recipe_changed(ctx: Context) -> bool:
    if not _made_by_the_recipe(ctx.edit):
        return False
    try:
        treatment = saved_treatment(ctx.edit)
        if _removes(_remake(ctx, treatment).outcome.cuts) == _removes(ctx.edit.get("cuts", [])):
            return False
        _rebuild(ctx, treatment, undo=ctx.edit.get("undo"))
    except StudioError as why:
        log.warning("the saved edit could not be checked against the recipe: %s", why)
        return False
    return True


def _remake(ctx: Context, treatment: Treatment, *, requested: list[Any] | None = None,
            keep: list[dict[str, Any]] | None = None,
            creator_cuts: list[dict[str, Any]] | None = None,
            picks: list[dict[str, Any]] | None = None) -> MadeEdit:
    keep = ctx.edit.get("keep", []) if keep is None else keep
    picks = saved_picks(ctx.edit) if picks is None else picks
    return make_edit(
        list(ctx.edit.get("requested", [])) if requested is None else requested,
        ctx.words, ctx.duration, treatment=treatment, silences=ctx.silences, sounds=ctx.sounds,
        keeps=keep_spans_of(keep), exact_keeps=exact_spans_of(keep),
        creator_cuts=ctx.edit.get("creator_cuts", []) if creator_cuts is None else creator_cuts,
        plans_in=ctx.project.root, picks=picks,
    )


def _rebuild(ctx: Context, treatment: Treatment, *, requested: list[Any] | None = None,
             keep: list[dict[str, Any]] | None = None,
             creator_cuts: list[dict[str, Any]] | None = None,
             undo: dict[str, Any] | None = None,
             picks: list[dict[str, Any]] | None = None) -> None:
    """Make the edit again with the recipe and save it. Updates ``ctx.edit``.

    ``undo`` is what the next ``undo`` action restores; without it the saved
    edit has nothing to undo. Claude's picks carry over as saved unless
    ``picks`` replaces them.
    """
    requested = list(ctx.edit.get("requested", [])) if requested is None else requested
    keep = list(ctx.edit.get("keep", [])) if keep is None else keep
    creator_cuts = list(ctx.edit.get("creator_cuts", [])) if creator_cuts is None else creator_cuts
    picks = saved_picks(ctx.edit) if picks is None else picks
    made = _remake(ctx, treatment, requested=requested, keep=keep, creator_cuts=creator_cuts, picks=picks or [])
    ctx.edit = save_made(
        ctx.project, made, duration=ctx.duration, requested=requested, keep=keep,
        set_by_claude=ctx.edit.get("set_by_claude"), creator_cuts=creator_cuts, times=ctx.times, undo=undo,
        picks=picks,
    )


# ── Samples on disk ───────────────────────────────────────────────────────────


def _treatment_path(project: Project):
    return project.root / TREATMENT_FILE


def load_samples_store(project: Project) -> dict[str, Any]:
    """The saved samples and every stretch used so far: ``{samples, used}``, empty when none."""
    path = _treatment_path(project)
    if not path.exists():
        return {"samples": [], "used": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"samples": [], "used": []}
    if not isinstance(data, dict):
        return {"samples": [], "used": []}
    return {"samples": list(data.get("samples", [])), "used": list(data.get("used", []))}


def _save_samples_store(project: Project, samples: list[dict[str, Any]], used: list[list[float]]) -> None:
    write_json_atomic(_treatment_path(project), {
        "version": TREATMENT_VERSION, "updated_at": now_iso(), "samples": samples, "used": used,
    })


def _listen_inputs(ctx: Context, made: MadeEdit) -> tuple[list[Span], EditedClock, Edits]:
    """Sentences, the edited clock, and what the saved edit does, for samples and sections."""
    cuts = ctx.edit.get("cuts", [])
    clock = EditedClock(kept_segments(edits.removed_spans(ctx.edit), ctx.duration))
    trims = [(float(c["start"]), float(c["end"]), _cut_trim_kind(c)) for c in cuts if c.get("source") == "auto"]
    claude = sorted(
        ((float(p["start"]), float(p["end"])) for p in made.outcome.placed if not p["put_back"]),
        key=lambda s: -(s[1] - s[0]),
    )
    laughs = sorted(
        (s for s in ctx.sounds if s.get("kind") == LAUGH),
        key=lambda s: (s.get("confidence") != LIKELY, -(float(s["end"]) - float(s["start"])), float(s["start"])),
    )
    return sentence_spans(ctx.words), clock, Edits(trims=trims, cuts=claude, laughs=laughs)


def _pick(ctx: Context, made: MadeEdit, avoid: list[Span]) -> tuple[list[dict[str, Any]], bool]:
    """Three samples avoiding ``avoid``; the flag says the history had to be forgotten."""
    sentences, clock, what = _listen_inputs(ctx, made)
    picked = choose_samples(sentences, clock, what, avoid=avoid)
    if picked is not None:
        return picked, False
    current = [(float(s["start"]), float(s["end"])) for s in load_samples_store(ctx.project)["samples"]]
    picked = choose_samples(sentences, clock, what, avoid=current)
    if picked is None:
        picked = choose_samples(sentences, clock, what, allow_overlap=True) or []
    return picked, True


def _ensure_samples(ctx: Context, made: MadeEdit) -> list[dict[str, Any]]:
    """The saved samples, choosing and saving the first three when there are none."""
    store = load_samples_store(ctx.project)
    if store["samples"]:
        return store["samples"]
    picked, _ = _pick(ctx, made, [])
    if picked:
        _save_samples_store(ctx.project, picked, [[s["start"], s["end"]] for s in picked])
    return picked


# ── The page's state ──────────────────────────────────────────────────────────


def _cut_trim_kind(cut: dict[str, Any]) -> str:
    return str(cut.get("kind") or trim_kind(str(cut.get("reason", ""))))


def _automatic(made: MadeEdit) -> list[dict[str, Any]]:
    """The removals of an edit that the pace made and nobody else had a hand in."""
    return [c for c in made.outcome.cuts if c.get("source") == edits.BY_AUTO]


def _seconds_out(cuts: list[dict[str, Any]]) -> float:
    return sum(float(c["end"]) - float(c["start"]) for c in cuts)


@dataclass
class Taken:
    """What one kind of automatic trim takes out of an edit."""

    count: int = 0
    seconds: float = 0.0
    # The words it takes, by how the creator would name each, with how often.
    words: dict[str, int] = field(default_factory=dict)

    def listed(self) -> list[list[Any]]:
        """The words with counts, most first. Ties read in the order the words are first said."""
        return [[text, n] for text, n in sorted(self.words.items(), key=lambda item: -item[1])]


def taken_by_kind(made: MadeEdit, words: Words) -> dict[str, Taken]:
    """What each kind of automatic trim takes out of an edit: ``{kind: Taken}``. The one count of it.

    A removal counts, with its length, for the kind it is filed under. A
    word inside it that another switch owns goes to that switch, with the
    word's own length: "like" inside a long pause is a filler word, and the
    pause around it a long pause. So no second is counted twice, and the
    three add up to what the pace removes.

    Filler words are counted by the word, since a filler goes with whatever
    removal holds it. The other kinds are counted by the removal. A filler
    vocalization holds no word at all (``autocuts.vocalization_filler_trims``
    takes a "[vocalization]" whole); it counts once under Filler words, named
    ``SOUND_FILLER_LABEL``, the way a word would be.
    """
    said = Said.of(words)
    level = engine_level(made.shortest)
    taken = {kind: Taken() for kind in TRIM_KINDS}
    for cut in _automatic(made):
        start, end = float(cut["start"]), float(cut["end"])
        filed = _cut_trim_kind(cut)
        held = said.held(start, end)
        inside = said.words[held.start:held.stop]
        owners = [owner or filed for owner in said.owners(held, level)]
        taken[filed].seconds += end - start
        for word, owner in zip(inside, owners):
            if owner != filed:
                length = min(end, float(word["end"])) - max(start, float(word["start"]))
                taken[owner].seconds += length
                taken[filed].seconds -= length
        if filed != FILLER_TRIM_KIND:
            taken[filed].count += 1
        elif not inside:
            taken[filed].count += 1
            taken[filed].words[SOUND_FILLER_LABEL] = taken[filed].words.get(SOUND_FILLER_LABEL, 0) + 1
        for kind in TRIM_KINDS:
            mine = [w for w, owner in zip(inside, owners) if owner == kind]
            named = fillers_in(mine, level) if kind == FILLER_TRIM_KIND else _said(mine)
            for text in named:
                taken[kind].words[text] = taken[kind].words.get(text, 0) + 1
            if kind == FILLER_TRIM_KIND:
                taken[kind].count += len(named)
    return taken


def _with_switches(current: Treatment, pace: str, take_out: frozenset[str]) -> Treatment:
    """``current`` at ``pace`` with the switches ``take_out``. Claude's shortest-pause override stays with its own pace."""
    if pace == CUSTOM:
        return Treatment(pace=CUSTOM, take_out=take_out, fine=current.fine)
    return Treatment(pace=pace, take_out=take_out, gap_length=current.gap_length if pace == current.pace else None)


def _every_trim_on(current: Treatment) -> frozenset[str]:
    """Every kind of automatic trim on, with Claude's picks as their switch stands now.

    What a pace would find is read from an edit built this way. The picks
    stay as they are because a picked word beside a pause joins that
    pause's trim: with the picks off the trim counts by itself again.
    """
    return frozenset(TRIM_KINDS) | (current.take_out & {edits.PICK_KIND})


def _paces(ctx: Context, current: Treatment) -> tuple[
    list[dict[str, Any]], dict[str, dict[str, int]], dict[str, Any] | None,
]:
    """Each stop's removals with the current switches, what each kind finds at every stop, and her own setting's.

    Two edits are built for a stop, both kept by the recipe cache: one with
    the switches as they are, for what the stop removes, and one with every
    kind on, for what each kind would find there. They are the same edit
    while every switch is on. The third value is the ``custom`` entry, None
    when the pace is a stop.
    """
    every = _every_trim_on(current)

    def removes(pace: str) -> dict[str, Any]:
        cuts = _automatic(_remake(ctx, _with_switches(current, pace, current.take_out)))
        return {"trims": len(cuts), "seconds_saved": round(_seconds_out(cuts), 1)}

    def finds(pace: str) -> dict[str, int]:
        made = _remake(ctx, _with_switches(current, pace, every))
        return {kind: taken.count for kind, taken in taken_by_kind(made, ctx.words).items()}

    paces = []
    by_pace: dict[str, dict[str, int]] = {}
    for name, level in PACES.items():
        by_pace[name] = finds(name)
        paces.append({"pace": name, "label": pace_label(name), "summary": level.summary, **removes(name)})
    custom = None
    if current.custom:
        custom = {"label": CUSTOM_LABEL, "summary": current.level().summary, **removes(CUSTOM)}
    return paces, by_pace, custom


def _switches(ctx: Context, current: Treatment, by_pace: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    """The ``take_out`` entries of the kinds of automatic trim: what each switch takes at the current pace.

    Shown whether the switch is on or off, so each is read from the edit
    with that switch on and the others as they stand. While every switch is
    on that is one edit, the saved one.
    """
    out = []
    for kind, label in TRIM_KINDS.items():
        on = _remake(ctx, _with_switches(current, current.pace, current.take_out | {kind}))
        taken = taken_by_kind(on, ctx.words)[kind]
        out.append({
            "key": kind, "label": label, "count": taken.count, "seconds": round(max(0.0, taken.seconds), 1),
            "by_pace": {name: found[kind] for name, found in by_pace.items()},
            "words": taken.listed() if kind in KINDS_THAT_LIST_WORDS else [],
        })
    return out


def _likes(ctx: Context, current: Treatment, stops: list[str]) -> dict[str, Any]:
    """The Filler likes entry of ``take_out``: what Claude picked and what became of each.

    Counted with the switch on, whether it is on or off, as the other
    entries are.
    """
    picks = saved_picks(ctx.edit)
    on = _remake(ctx, Treatment(pace=current.pace, take_out=current.take_out | {edits.PICK_KIND},
                                gap_length=current.gap_length, fine=current.fine))
    became = [p["status"] for p in on.outcome.picks]
    out = [p for p in on.outcome.picks if p["status"] == edits.PICK_OUT]
    said = sum(1 for w in ctx.words if not is_event(w) and _plain(w) == LIKES_WORD)
    tally: dict[str, int] = {}
    for p in out:
        for text in _said([w for w in ctx.words if not is_event(w) and p["word_start"] <= _mid(w) <= p["word_end"]]):
            tally[text] = tally.get(text, 0) + 1
    return {
        "key": edits.PICK_KIND, "label": edits.PICK_LABEL, "count": len(out),
        "seconds": round(sum(p["end"] - p["start"] for p in out), 1),
        "by_pace": {name: len(out) for name in stops},
        "words": [[t, n] for t, n in sorted(tally.items(), key=lambda item: -item[1])],
        "picked": len(became), "said": said,
        "left_in": became.count(edits.PICK_LEFT_IN) + became.count(edits.PICK_LOST),
        "kept_by_you": became.count(edits.PICK_KEPT),
        "ready": picks is not None, "word": LIKES_WORD,
    }


def pace_label(name: str) -> str:
    """A pace's name as the creator reads it: ``Standard``."""
    return name.capitalize()


def _plain(word: dict[str, Any]) -> str:
    """A word as said, lower case, without the punctuation around it."""
    return plain_text(str(word.get("word", "")))


def _said(removed: Words) -> list[str]:
    """The removed words of one trim as the creator would name them: two-word fillers read as one.

    Every word is named, on a filler list or not.
    """
    out: list[str] = []
    texts = [t for t in (_plain(w) for w in removed) if t]
    i = 0
    while i < len(texts):
        pair = " ".join(texts[i:i + 2])
        if pair in FILLER_PAIRS:
            out.append(pair)
            i += 2
        else:
            out.append(texts[i])
            i += 1
    return out


def words_taken_out(made: MadeEdit, words: Words) -> dict[str, list[list[Any]]]:
    """The words the automatic removals take, with counts, most first: ``{kind: [[text, count], ...]}``.

    A filler counts under filler words whatever its removal was named for,
    so the list holds every filler the pace takes (``taken_by_kind``).
    """
    return {kind: taken.listed() for kind, taken in taken_by_kind(made, words).items()}


def removal_codes(ctx: Context) -> dict[tuple[float, float], int]:
    """Why each removed word of the saved edit is removed, by the word's ``(start, end)`` to the millisecond.

    The values are the ones ``words[]`` carries on the page. A word that
    plays is not in it.
    """
    made = _remake(ctx, saved_treatment(ctx.edit))
    listed = _page_words(ctx, _merged(edits.removed_spans(ctx.edit)), made)
    return {(round(s, 3), round(e, 3)): why for _text, s, e, _removed, why in listed if why}


def plan_every_pace(ctx: Context) -> None:
    """Build the edit at every pace and keep the results, so the page's first state is quick. Saves nothing."""
    _paces(ctx, saved_treatment(ctx.edit))


def _mid(word: dict[str, Any]) -> float:
    return (float(word["start"]) + float(word["end"])) / 2


def _words_either_side(words: Words, start: float, end: float) -> tuple[str, str, str]:
    spoken = [w for w in words if not is_event(w)]
    mid = [(float(w["start"]) + float(w["end"])) / 2 for w in spoken]
    before = [str(w["word"]).strip() for w, m in zip(spoken, mid) if m <= start][-CONTEXT_WORDS:]
    after = [str(w["word"]).strip() for w, m in zip(spoken, mid) if m >= end][:CONTEXT_WORDS]
    removed = [str(w["word"]).strip() for w, m in zip(spoken, mid) if start < m < end]
    if len(removed) > REMOVED_WORDS_SHOWN:
        half = REMOVED_WORDS_SHOWN // 2
        removed = removed[:half] + ["..."] + removed[-half:]
    return " ".join(before), " ".join(removed), " ".join(after)


def claude_row_id(placed: dict[str, Any]) -> str:
    """The id of one of Claude's cuts on the page, from its placed edges."""
    return f"{CLAUDE_ROW_PREFIX}{placed['start']:.2f}-{placed['end']:.2f}"


def _row(ctx: Context, joined: list[dict[str, Any]], rid: str, start: float, end: float, *,
         by: str, group: str, reason: str, state: str) -> dict[str, Any]:
    """One row of the cuts list: a cut, who made it, and what the join check says of it."""
    join = None if state == PUT_BACK else next(
        (j for j in joined if j.get("source") != edits.BY_AUTO
         and _overlaps(float(j["start"]), float(j["end"]), start, end)), None,
    )
    flags = [f for f in (join or {}).get("flags", []) if isinstance(f, str)]
    before, removed, after = _words_either_side(ctx.words, start, end)
    return {
        "id": rid, "start": start, "end": end, "clock": edits.clock_span(start, end),
        "seconds": round(end - start, 1), "group": group, "reason": reason, "by": by,
        "flags": flags, "flag_labels": [FLAG_LABELS.get(f, f.replace("_", " ")) for f in flags],
        "note": str((join or {}).get("note", "") or "") if flags else "",
        "why": str((join or {}).get("why", "") or "") if flags else "",
        "state": state, "before": before, "removed": removed, "after": after,
    }


def _rows(ctx: Context, made: MadeEdit, joined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per cut Claude or the creator chose, in time order, with the join check's flags.

    The creator's rows carry ``by: "you"`` and sit in the group ``yours``;
    they are always kept out, since taking one back removes the row.
    """
    rows = []
    seen: set[str] = set()
    for p in made.outcome.placed:
        rid = claude_row_id(p)
        if rid in seen:
            continue
        seen.add(rid)
        rows.append(_row(
            ctx, joined, rid, p["start"], p["end"], by=edits.BY_CLAUDE, group=p["kind"],
            reason=p["reason"], state=PUT_BACK if p["put_back"] else KEPT_OUT,
        ))
    for c in made.outcome.creator_placed:
        rows.append(_row(
            ctx, joined, c["id"], c["start"], c["end"], by=edits.BY_CREATOR, group=edits.CREATOR_KIND,
            reason=edits.CREATOR_REASON, state=KEPT_OUT,
        ))
    return sorted(rows, key=lambda r: (r["start"], r["end"]))


def _groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The creator's cuts first, then Claude's by kind. Every group is listed, empty or not."""
    out = []
    for key, label in {edits.CREATOR_KIND: edits.CREATOR_KIND_LABEL, **edits.CUT_KINDS}.items():
        mine = [r for r in rows if r["group"] == key]
        out.append({
            "key": key, "label": label, "count": len(mine),
            "seconds": round(sum(r["seconds"] for r in mine if r["state"] == KEPT_OUT), 1),
            "need_a_look": sum(1 for r in mine if r["flags"]),
            "rows": [r["id"] for r in mine],
        })
    return out


def _merged(spans: list[Span]) -> list[Span]:
    out: list[Span] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _inside_any(mid: float, spans: list[Span], starts: list[float]) -> bool:
    """Whether ``mid`` is inside one of ``spans``, which are sorted, apart, and start at ``starts``."""
    i = bisect.bisect_right(starts, mid) - 1
    return i >= 0 and mid < spans[i][1]


def _page_words(ctx: Context, removed: list[Span], made: MadeEdit) -> list[list[Any]]:
    """The words as ``[text, start, end, removed, why]``.

    ``removed`` is 1 when the word's middle is cut. ``why`` says who removed
    it: ``REMOVED_BY_YOU`` when one of the creator's cuts holds it, else
    ``REMOVED_BY_CLAUDE`` when one of Claude's cuts does, else
    ``REMOVED_AS_PICKED`` when it is a filler word Claude picked, else
    ``REMOVED_AUTO``, and ``PLAYING`` for a word that plays. ``removed`` must
    be sorted and merged.
    """
    owners = [
        (REMOVED_BY_YOU, [(c["start"], c["end"]) for c in made.outcome.creator_placed]),
        (REMOVED_BY_CLAUDE, [(p["start"], p["end"]) for p in made.outcome.placed if not p["put_back"]]),
        (REMOVED_AS_PICKED, [(p["start"], p["end"]) for p in made.outcome.picks if p["status"] == edits.PICK_OUT]),
    ]
    owners = [(why, spans, [a for a, _ in spans]) for why, spans in ((w, _merged(sp)) for w, sp in owners)]
    starts = [a for a, _ in removed]
    out = []
    for s, e, text, _sound in page_words(ctx.words, ctx.sounds):
        mid = (s + e) / 2
        why = PLAYING
        if _inside_any(mid, removed, starts):
            why = next((w for w, spans, at in owners if _inside_any(mid, spans, at)), REMOVED_AUTO)
        out.append([text, s, e, 1 if why else 0, why])
    return out


def _keeps(ctx: Context) -> list[dict[str, Any]]:
    """The parts the creator flagged and the words she brought back, for the page.

    Put-back cuts show as rows instead. ``exact`` is true for words brought
    back one by one.
    """
    out = []
    for k in sorted(ctx.edit.get("keep", []), key=lambda k: float(k["start"])):
        if _origin(k) == ORIGIN_PUT_BACK and k.get("origin"):
            continue
        if not k.get("origin") and _matches_a_cut(k, ctx):
            continue
        start, end = float(k["start"]), float(k["end"])
        exact = _origin(k) == ORIGIN_EXACT
        out.append({
            "id": keep_id(k), "start": start, "end": end, "clock": edits.clock_span(start, end),
            "text": k.get("text") or _shorten(" ".join(_text_between(ctx.words, start, end)), KEEP_TEXT_CHARS),
            "note": str(k.get("note") or (BROUGHT_BACK_BY_YOU if exact else KEPT_BY_YOU)),
            "exact": exact,
        })
    return out


def _matches_a_cut(keep: dict[str, Any], ctx: Context) -> bool:
    """Whether a keep with no origin (restored on the old review page) covers one of Claude's cuts."""
    start, end = float(keep["start"]), float(keep["end"])
    return any(
        start <= float(r.get("start", -1)) + 0.05 and float(r.get("end", -1)) - 0.05 <= end
        for r in ctx.edit.get("requested", []) if isinstance(r, dict) and _finite(r.get("start")) and _finite(r.get("end"))
    )


def time_out(seconds: float) -> str:
    """Time removed, ready to show: ``0:14 out``, to the nearest whole second, a half going up."""
    return f"{edits.clock(int(seconds + 0.5))} out"


def _sample_detail(sample: dict[str, Any], made: MadeEdit, removed: list[Span], clock: EditedClock) -> str:
    """What the current edit does in one sample, in a few plain words.

    A big cut says ``put back`` when the creator put it back. A joke says
    whether its laugh plays whole. Every other case is the time the edit
    takes out of the sample. ``removed`` must be sorted and merged.
    """
    start, end = float(sample["start"]), float(sample["end"])
    anchor = None if sample.get("fallback") else sample.get("anchor")
    if sample["key"] == "big_cut" and anchor:
        row = next((p for p in made.outcome.placed if _overlaps(p["start"], p["end"], anchor[0], anchor[1])), None)
        if row is not None and row["put_back"]:
            return DETAIL_PUT_BACK
    if sample["key"] == "joke" and anchor:
        cut = any(_overlaps(a, b, float(anchor[0]), float(anchor[1])) for a, b in removed)
        return DETAIL_LAUGH_CUT if cut else DETAIL_LAUGH_WHOLE
    out = (end - start) - clock.length(start, end)
    return time_out(out) if out >= SHORTEST_TIME_OUT else DETAIL_NOTHING_OUT


def _sample_rows(made: MadeEdit, samples: list[dict[str, Any]], removed: list[Span],
                 clock: EditedClock) -> list[dict[str, Any]]:
    """The saved samples with their live detail: what the current edit does in each."""
    out = []
    for s in samples:
        start, end = float(s["start"]), float(s["end"])
        out.append({
            "key": s["key"], "label": SAMPLE_LABELS[s["key"]], "start": start, "end": end,
            "clock": edits.clock(start), "why": plain_why(str(s["why"])),
            "detail": _sample_detail(s, made, removed, clock),
            "edited_seconds": round(clock.length(start, end), 1),
        })
    return out


def _clusters(sentences: list[Span], cuts: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    """The clusters of the saved edit, each with its clocks and its time out ready to show."""
    sections = busy_sections(
        sentences, [(float(c["start"]), float(c["end"]), str(c.get("source"))) for c in cuts], duration,
    )
    return [
        {
            "number": c["number"], "start": round(c["start"], 3), "end": round(c["end"], 3),
            "clock": edits.clock(c["start"]),
            "clock_range": f"{edits.clock(c['start'])} to {edits.clock(c['end'])}",
            "edits": c["edits"], "seconds_removed": c["seconds_removed"],
            "out": time_out(max(c["seconds_removed"], SHORTEST_TIME_OUT_SHOWN)), "level": c["level"],
        }
        for c in clusters(sections)
    ]


def _overlays(ctx: Context, removed: list[Span]) -> list[dict[str, Any]]:
    """The creator's saved photos and clips, in time order, each with its status under this edit.

    The page only shows them, so a saved list that can't be read shows as
    none here and is logged. A render still stops on it, with how to fix it.
    """
    try:
        saved = load_overlays(ctx.project)["overlays"]
        placed = place_on_edit(saved, kept_segments(removed, ctx.duration), ctx.words)
        rows = [page_row(p) for p in placed]
    except (StudioError, KeyError, TypeError, ValueError) as why:
        log.warning("the saved photos and clips can't be read, so the page shows none: %s", why)
        return []
    return sorted(rows, key=lambda r: (r["start"], r["id"]))


def page_state(ctx: Context, *, changed: dict[str, Any] | None = None) -> dict[str, Any]:
    """Everything the treatment page shows, from the saved edit. The round's API notes list each field.

    ``busy`` is ``clusters`` under its earlier name with ``cuts`` beside
    ``edits``, kept so an earlier page still draws. ``taste`` is
    ``{videos}``, how many other videos Lumr is learning from, or None.
    """
    treatment = saved_treatment(ctx.edit)
    made = _remake(ctx, treatment)
    cuts = ctx.edit.get("cuts", [])
    joined = ctx.join_rows(cuts, ctx.words, ctx.duration, labels=ctx.sounds or None) if cuts else []
    rows = _rows(ctx, made, joined)
    groups = _groups(rows)
    paces, by_pace, custom = _paces(ctx, treatment)
    removed = _merged(edits.removed_spans(ctx.edit))
    trims = [(float(c["start"]), float(c["end"]), _cut_trim_kind(c)) for c in cuts if c.get("source") == "auto"]
    sentences, clock, _ = _listen_inputs(ctx, made)
    samples = _ensure_samples(ctx, made)
    found = _clusters(sentences, cuts, ctx.duration)
    edited = sum(e - s for s, e in kept_segments(removed, ctx.duration))
    usual = load_usual()
    return {
        "video": {"duration": round(ctx.duration, 3)},
        "word_times": ctx.times,
        "word_times_note": ctx.times_note,
        "settings": treatment.as_dict(),
        "usual": None if usual is None else {k: v for k, v in usual.as_dict().items() if k != "gap_length"},
        "taste": taste.summary_for_page(ctx.project.root),
        "paces": paces,
        "custom": custom,
        "fine_ranges": fine_ranges(),
        "take_out": _switches(ctx, treatment, by_pace) + [_likes(ctx, treatment, list(PACES))],
        "need_a_look": sum(g["need_a_look"] for g in groups),
        "cut_counts": {"yours": sum(1 for r in rows if r["by"] == edits.BY_CREATOR),
                       "claude": sum(1 for r in rows if r["by"] == edits.BY_CLAUDE)},
        "groups": groups,
        "rows": rows,
        "removed": [[round(s, 3), round(e, 3)] for s, e in removed],
        "trims": [[round(s, 3), round(e, 3), k] for s, e, k in trims],
        "keeps": _keeps(ctx),
        "samples": _sample_rows(made, samples, removed, clock),
        "clusters": found,
        "busy": [{**c, "cuts": c["edits"]} for c in found],
        "overlays": _overlays(ctx, removed),
        "export": ctx.export_for(ctx.edit) if ctx.export_for else nothing_exported(),
        "durations": {"full": round(ctx.duration, 1), "edited": round(edited, 1),
                      "saved": round(ctx.duration - edited, 1)},
        "changed": changed,
        "can_undo": isinstance(ctx.edit.get("undo"), dict),
        "words": _page_words(ctx, removed, made),
        "sounds": [{"start": float(x["start"]), "end": float(x["end"]), "kind": x["kind"],
                    "confidence": x["confidence"]} for x in ctx.sounds if x.get("kind") == LAUGH],
        "updated_at": ctx.edit.get("updated_at"),
    }


# ── What changed ──────────────────────────────────────────────────────────────


@dataclass
class Snapshot:
    """The parts of an edit that "what changed" compares."""

    trims: list[Span]
    edited: float
    flagged: list[Span]
    # Ids of the photos and clips the edit hides.
    hidden: list[str] = field(default_factory=list)
    # How many cuts of her own the creator has.
    your_cuts: int = 0
    # How many filler words Claude picked are out.
    likes: int = 0


def _holds_a_trim(cut: dict[str, Any]) -> bool:
    """An automatic trim, or a cut of the creator's that one joined.

    A word knocked out beside a trimmed pause joins that trim into one cut.
    The pause still goes, so "what changed" must not say a trim was dropped.
    """
    by_a_word_cut = cut.get("source") in (edits.BY_CREATOR, edits.BY_PICK)
    return cut.get("source") == edits.BY_AUTO or (by_a_word_cut and bool(cut.get("auto_trims")))


def _picks_out(ctx: Context) -> int:
    """How many of Claude's picked filler words the saved edit takes out."""
    if not ctx.edit.get("picks"):
        return 0
    try:
        made = _remake(ctx, saved_treatment(ctx.edit))
    except StudioError:
        return 0
    return sum(1 for p in made.outcome.picks if p["status"] == edits.PICK_OUT)


def snapshot(ctx: Context) -> Snapshot:
    cuts = ctx.edit.get("cuts", [])
    joined = ctx.join_rows(cuts, ctx.words, ctx.duration, labels=ctx.sounds or None) if cuts else []
    removed = edits.removed_spans(ctx.edit)
    return Snapshot(
        trims=[(float(c["start"]), float(c["end"])) for c in cuts if _holds_a_trim(c)],
        edited=sum(e - s for s, e in kept_segments(removed, ctx.duration)),
        flagged=[(float(j["start"]), float(j["end"])) for j in joined if j.get("flags")],
        hidden=[o["id"] for o in _overlays(ctx, _merged(removed)) if o["status"] == HIDDEN],
        your_cuts=len(ctx.edit.get("creator_cuts", [])),
        likes=_picks_out(ctx),
    )


def what_changed(before: Snapshot, after: Snapshot) -> dict[str, Any]:
    """The difference between two snapshots, for the page's "what changed" line.

    A trim or a flagged join counts as the same one when it overlaps one from
    before: a new pace moves edges without making it a different pause.
    ``overlays_hidden`` counts the photos and clips this change newly hid.
    ``words_cut``, ``words_back`` and ``keeps_changed`` are zero here; the
    action that cut or brought back words fills them in.
    """
    def unmatched(spans: list[Span], others: list[Span]) -> int:
        return sum(1 for a in spans if not any(_overlaps(*a, *b) for b in others))

    added = unmatched(after.trims, before.trims)
    dropped = unmatched(before.trims, after.trims)
    return {
        "trims": len(after.trims) - len(before.trims),
        "trims_added": added,
        "trims_dropped": dropped,
        "seconds": round(after.edited - before.edited, 1),
        "need_a_look": len(after.flagged),
        "new_flags": unmatched(after.flagged, before.flagged),
        "overlays_hidden": len(set(after.hidden) - set(before.hidden)),
        "your_cuts": after.your_cuts - before.your_cuts,
        "words_cut": 0,
        "words_back": 0,
        "keeps_changed": 0,
        "likes": after.likes - before.likes,
    }


def _answer(ctx: Context, before: Snapshot, **words: int) -> dict[str, Any]:
    """The new state with what changed. ``words`` are the counts only the action knows."""
    return page_state(ctx, changed={**what_changed(before, snapshot(ctx)), **words})


# ── Actions ───────────────────────────────────────────────────────────────────


def _need_object(body: Any, allowed: set[str], example: str) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise StudioError(f"Send a JSON object like {example}.")
    extra = sorted(set(body) - allowed)
    if extra:
        raise StudioError(f"Unknown field {extra[0]!r}. Send only {', '.join(sorted(allowed))}, like {example}.")
    return body


def _fine_asked(body: dict[str, Any], current: Treatment) -> tuple[float, float]:
    """The slider values ``body["fine"]`` asks for; the one left out stays where the slider stands."""
    asked = body["fine"]
    example = '{"fine": {"gap_length": 0.35, "rhythm": 3.5}}'
    if not isinstance(asked, dict) or not asked:
        raise StudioError(f"fine must hold gap_length, rhythm, or both, like {example}.")
    extra = sorted(set(asked) - {"gap_length", "rhythm"})
    if extra:
        raise StudioError(f"fine has {extra[0]!r}, which is not a slider. Send gap_length and rhythm, like {example}.")
    stands = current.fine_values()
    for key in asked:
        if not _finite(asked[key]):
            raise StudioError(f"fine.{key} must be a number, like {example}.")
    return check_fine(float(asked.get("gap_length", stands["gap_length"])), float(asked.get("rhythm", stands["rhythm"])))


def parse_treatment_change(body: Any, current: Treatment) -> Treatment:
    """The treatment ``body`` asks for, starting from ``current``. Raises StudioError when it is bad.

    ``pace`` picks a stop, which sets the sliders back to that stop's values.
    ``fine`` sets the sliders, and the pace then reads ``custom``.
    """
    example = '{"pace": "standard", "take_out": {"fillers": false}}'
    body = _need_object(body, {"pace", "take_out", "fine"}, example)
    if not body:
        raise StudioError(f"Send a pace, take_out switches, or both, like {example}.")
    if "pace" in body and "fine" in body:
        raise StudioError("Send a pace or fine, not both: a pace sets the sliders to its own values.")
    pace, gap, fine = current.pace, current.gap_length, current.fine
    if "pace" in body:
        if not isinstance(body["pace"], str):
            raise StudioError(f"pace must be one of: {', '.join(PACES)}.")
        new = get_pace(body["pace"]).name
        if new != pace:
            pace, gap = new, None
        fine = None
    if "fine" in body:
        pace, gap, fine = CUSTOM, None, _fine_asked(body, current)
    take_out = set(current.take_out)
    if "take_out" in body:
        switches = body["take_out"]
        if not isinstance(switches, dict):
            raise StudioError(f"take_out must be an object like {{\"fillers\": false}}, with keys {', '.join(edits.SWITCHES)}.")
        for key, on in switches.items():
            if key not in edits.SWITCHES:
                raise StudioError(f"take_out has {key!r}, which is not a kind. Use {', '.join(edits.SWITCHES)}.")
            if not isinstance(on, bool):
                raise StudioError(f"take_out.{key} must be true or false.")
            (take_out.add if on else take_out.discard)(key)
    return Treatment(pace=pace, take_out=frozenset(take_out), gap_length=gap, fine=fine)


def change_treatment(ctx: Context, body: Any) -> dict[str, Any]:
    """Set the pace and switches, rebuild the edit, and answer the new state."""
    treatment = parse_treatment_change(body, saved_treatment(ctx.edit))
    before = snapshot(ctx)
    _rebuild(ctx, treatment, undo=ctx.edit.get("undo"))
    return _answer(ctx, before)


def _need_id(body: dict[str, Any], prefix: str) -> str:
    value = body.get("id")
    if not isinstance(value, str) or not value.startswith(prefix) or len(value) > 64:
        raise StudioError(f"Send the id as a string starting with {prefix!r}, as the page listed it.")
    return value


def set_cut_state(ctx: Context, body: Any) -> dict[str, Any]:
    """Put one of Claude's cuts back, or keep it out again. Answers the new state.

    Putting back saves the cut's span as a kept span, so no later edit cuts
    there. Keeping out removes the kept span that putting it back made.
    Raises StudioError when the cut sits inside a part the creator flagged.
    """
    body = _need_object(body, {"id", "state"}, '{"id": "c99.61-107.37", "state": "put_back"}')
    if isinstance(body.get("id"), str) and body["id"].startswith(CREATOR_ROW_PREFIX):
        raise StudioError("That cut is one of yours. Take it back instead.")
    rid = _need_id(body, CLAUDE_ROW_PREFIX)
    state = body.get("state")
    if state not in CUT_STATES:
        raise StudioError(f"state must be {KEPT_OUT!r} or {PUT_BACK!r}.")
    treatment = saved_treatment(ctx.edit)
    made = _remake(ctx, treatment)
    placed = next((p for p in made.outcome.placed if claude_row_id(p) == rid), None)
    if placed is None:
        raise StudioError("That cut is not one of Claude's cuts in this edit any more. The edit may have changed; reload the page.")
    others = {claude_row_id(p) for p in made.outcome.placed} - {rid}
    before = snapshot(ctx)
    requested = [dict(r) if isinstance(r, dict) else r for r in ctx.edit.get("requested", [])]
    keep = [dict(k) for k in ctx.edit.get("keep", [])]
    start, end = placed["start"], placed["end"]
    if state == PUT_BACK:
        if isinstance(requested[placed["index"]], dict):
            requested[placed["index"]].pop(edits.OVERRIDE_FLAG, None)
        if not placed["put_back"]:
            keep.append({
                "id": f"k{start:.2f}-{end:.2f}", "start": start, "end": end, "origin": ORIGIN_PUT_BACK,
                "cut": rid, "note": f"Put back by the creator. Claude's reason was: {placed['reason']}",
            })
    else:
        # The kept span this cut's putting back made. Row ids move when the
        # word times change, so a span naming a row that is gone is this one's.
        mine = [k for k in keep if _origin(k) == ORIGIN_PUT_BACK
                and _overlaps(float(k["start"]), float(k["end"]), start, end)
                and (k.get("cut") == rid or k.get("cut") not in others or not k.get("origin"))]
        keep = [k for k in keep if k not in mine]
        # A cut reads as put back when she brought back every word of it.
        # Keeping it out again lets those words go.
        keep = [k for k in keep if not (_origin(k) == ORIGIN_EXACT
                                        and _overlaps(float(k["start"]), float(k["end"]), start, end))]
        blocking = [k for k in keep if _origin(k) != ORIGIN_EXACT
                    and _overlaps(float(k["start"]), float(k["end"]), start, end)]
        if blocking:
            k = blocking[0]
            where = edits.clock_span(float(k["start"]), float(k["end"]))
            if _origin(k) == ORIGIN_FLAGGED:
                raise StudioError(f"This cut sits inside a part you flagged to keep ({where}). Remove that keep first.")
            raise StudioError(f"This cut overlaps another cut you put back ({where}). Keep that one out first.")
    _rebuild(ctx, treatment, requested=requested, keep=keep)
    return _answer(ctx, before)


def _need_seconds(body: dict[str, Any], key: str, duration: float) -> float:
    value = body.get(key)
    if not _finite(value):
        raise StudioError(f"{key} must be a number of seconds in the video.")
    if not 0 <= value <= duration:
        raise StudioError(f"The {key} is outside the video, which runs from 0:00 to {edits.clock(duration)}.")
    return float(value)


# ── The creator's hand on the words ───────────────────────────────────────────


class _Yard(edits.SaidOrder):
    """The words and sounds in the order they are said, for finding what a span picks."""

    def span(self, picked: range) -> Span:
        """From the first picked word's start to the last one's end."""
        chosen = self.tokens[picked.start:picked.stop]
        return (round(min(float(w["start"]) for w in chosen), 3), round(max(float(w["end"]) for w in chosen), 3))

    def zone(self, picked: range, duration: float) -> Span:
        """``span`` with the quiet either side, up to the neighbouring words: all a cut of these words can take."""
        start, end = self.span(picked)
        floor, ceiling = self.neighbours(picked, duration)
        return min(start, floor), max(end, ceiling)

    def text(self, picked: range) -> str:
        chosen = self.tokens[picked.start:picked.stop]
        return _shorten(" ".join(str(w["word"]).strip() for w in chosen if not is_event(w)), KEEP_TEXT_CHARS)

    def runs_without(self, whole: range, gone: range) -> list[range]:
        """``whole`` with ``gone`` taken out: the runs of words left, in order."""
        runs = [range(whole.start, min(whole.stop, gone.start)), range(max(whole.start, gone.stop), whole.stop)]
        return [r for r in runs if len(r)]


def _touch(a: range, b: range) -> bool:
    """Whether two runs of words overlap or sit side by side with no word between."""
    return a.start <= b.stop and b.start <= a.stop


def _need_span(ctx: Context, body: dict[str, Any]) -> Span:
    start = _need_seconds(body, "start", ctx.duration)
    end = _need_seconds(body, "end", ctx.duration)
    if end < start:
        raise StudioError(f"The start must come before the end ({edits.clock(start)} to {edits.clock(end)}).")
    return start, end


def _need_words(yard: _Yard, start: float, end: float) -> range:
    picked = yard.picked(start, end)
    if not picked:
        raise StudioError("No word sits between those times. Pick at least one word.")
    return picked


def _creator_cut(yard: _Yard, picked: range) -> dict[str, Any]:
    start, end = yard.span(picked)
    return {"id": f"{CREATOR_ROW_PREFIX}{start:.2f}-{end:.2f}", "start": start, "end": end,
            "text": yard.text(picked), "words": len(picked)}


def _copies(edit: dict[str, Any]) -> tuple[list[Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Copies of the saved ``requested``, ``keep`` and ``creator_cuts``, safe to change."""
    return (
        [dict(r) if isinstance(r, dict) else r for r in edit.get("requested", [])],
        [dict(k) for k in edit.get("keep", [])],
        [dict(c) for c in edit.get("creator_cuts", [])],
    )


def _undo_of(edit: dict[str, Any]) -> dict[str, Any]:
    """What ``undo`` restores: the three lists as they are before an action changes them."""
    requested, keep, cuts = _copies(edit)
    return {"requested": requested, "keep": keep, "creator_cuts": cuts}


def _keeps_around(keep: list[dict[str, Any]], yard: _Yard, picked: range,
                  duration: float) -> tuple[list[dict[str, Any]], int]:
    """``keep`` with the picked words taken out of every kept span: ``(new keep list, spans changed)``.

    A span that holds the words shrinks or splits around them; a piece left
    holding no word goes. The pieces keep the span's origin and note.
    """
    start, end = yard.span(picked)
    zone = yard.zone(picked, duration)
    out: list[dict[str, Any]] = []
    changed = 0
    for k in keep:
        ks, ke = float(k["start"]), float(k["end"])
        if not _overlaps(ks, ke, start, end) and not (start == end and ks <= start <= ke):
            out.append(k)
            continue
        changed += 1
        for a, b in edits.span_without(ks, ke, [zone]):
            inside = yard.picked(a, b)
            if not inside:
                continue
            piece = {**k, "id": f"{KEEP_PREFIX}{a:.2f}-{b:.2f}", "start": round(a, 3), "end": round(b, 3)}
            if "text" in k:
                piece["text"] = yard.text(inside)
            out.append(piece)
    return sorted(out, key=lambda k: float(k["start"])), changed


def _cuts_without(cuts: list[dict[str, Any]], yard: _Yard, gone: range) -> tuple[list[dict[str, Any]], int]:
    """The creator's ``cuts`` with the words ``gone`` taken back: ``(new cuts, words taken back)``."""
    out: list[dict[str, Any]] = []
    back = 0
    for c in cuts:
        whole = yard.picked(float(c["start"]), float(c["end"]))
        shared = range(max(whole.start, gone.start), min(whole.stop, gone.stop))
        if not shared:
            out.append(c)
            continue
        back += len(shared)
        out += [_creator_cut(yard, run) for run in yard.runs_without(whole, shared)]
    return out, back


def add_cut(ctx: Context, body: Any) -> dict[str, Any]:
    """Cut exactly the words whose middle falls inside ``{start, end}``. Answers the new state.

    The cut is the creator's own. It joins any cut of hers it touches or
    overlaps. Inside a part she kept, the newer action wins: the kept span
    shrinks or splits so these words are cut, and ``changed.keeps_changed``
    says how many spans that touched.
    """
    body = _need_object(body, {"start", "end"}, '{"start": 412.3, "end": 412.6}')
    start, end = _need_span(ctx, body)
    yard = _Yard.of(ctx.words)
    picked = _need_words(yard, start, end)
    requested, keep, cuts = _copies(ctx.edit)
    if len(cuts) >= MAX_CREATOR_CUTS:
        raise StudioError(f"This video already has {len(cuts)} cuts of yours. Take some back before adding more.")
    try:
        edits.place_creator_cut(*yard.span(picked), yard, ctx.duration)
    except edits.CutRejected as why:
        raise StudioError(f"That cut {why}") from None
    new = len(picked)
    apart = []
    for c in cuts:
        theirs = yard.picked(float(c["start"]), float(c["end"]))
        if _touch(theirs, picked):
            new -= len(range(max(theirs.start, picked.start), min(theirs.stop, picked.stop)))
            picked = range(min(theirs.start, picked.start), max(theirs.stop, picked.stop))
        else:
            apart.append(c)
    cuts = sorted([*apart, _creator_cut(yard, picked)], key=lambda c: c["start"])
    keep, changed = _keeps_around(keep, yard, picked, ctx.duration)
    before = snapshot(ctx)
    _rebuild(ctx, saved_treatment(ctx.edit), requested=requested, keep=keep, creator_cuts=cuts,
             undo=_undo_of(ctx.edit))
    return _answer(ctx, before, words_cut=new, keeps_changed=changed)


def remove_cut(ctx: Context, body: Any) -> dict[str, Any]:
    """Take back one of the creator's cuts, or the words of it inside ``{start, end}``. Answers the new state.

    Nothing is kept by this: a word Claude or the pace also removes stays
    removed, and shows as theirs.
    """
    body = _need_object(body, {"id", "start", "end"}, '{"id": "y412.30-412.60"}')
    if isinstance(body.get("id"), str) and body["id"].startswith(CLAUDE_ROW_PREFIX):
        raise StudioError("That cut is one of Claude's. Put it back instead.")
    rid = _need_id(body, CREATOR_ROW_PREFIX)
    requested, keep, cuts = _copies(ctx.edit)
    mine = next((c for c in cuts if edits.creator_cut_id(c) == rid), None)
    if mine is None:
        raise StudioError("That cut is not in this video any more. Reload the page to see your cuts.")
    yard = _Yard.of(ctx.words)
    gone = yard.picked(float(mine["start"]), float(mine["end"]))
    if "start" in body or "end" in body:
        gone = _need_words(yard, *_need_span(ctx, body))
    left, back = _cuts_without([mine], yard, gone)
    if not back:
        raise StudioError("Those words are not part of that cut. Reload the page to see your cuts.")
    cuts = sorted([c for c in cuts if c is not mine] + left, key=lambda c: c["start"])
    before = snapshot(ctx)
    _rebuild(ctx, saved_treatment(ctx.edit), requested=requested, keep=keep, creator_cuts=cuts,
             undo=_undo_of(ctx.edit))
    return _answer(ctx, before, words_back=back)


def _removed_now(ctx: Context, yard: _Yard, picked: range) -> int:
    """How many of the picked words the saved edit removes."""
    removed = _merged(edits.removed_spans(ctx.edit))
    starts = [a for a, _ in removed]
    return sum(1 for m in yard.mids[picked.start:picked.stop] if _inside_any(m, removed, starts))


def _note_of(body: dict[str, Any]) -> str:
    note = body.get("note")
    if note is not None and (not isinstance(note, str) or len(note) > MAX_NOTE_CHARS):
        raise StudioError(f"note must be text of at most {MAX_NOTE_CHARS} characters.")
    return note.strip() if note else ""


def _let_go_of_overrides(requested: list[Any], start: float, end: float) -> None:
    """The creator's newer word wins over an earlier override of Claude's for this stretch."""
    for r in requested:
        if isinstance(r, dict) and _finite(r.get("start")) and _finite(r.get("end")) \
                and _overlaps(float(r["start"]), float(r["end"]), start, end):
            r.pop(edits.OVERRIDE_FLAG, None)


def _keep_exactly(ctx: Context, yard: _Yard, picked: range, note: str,
                  keep: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``keep`` with the picked words kept exactly, joined to any words brought back beside them."""
    joined = []
    apart = []
    for k in keep:
        theirs = yard.picked(float(k["start"]), float(k["end"]))
        if _origin(k) == ORIGIN_EXACT and theirs and _touch(theirs, picked):
            picked = range(min(theirs.start, picked.start), max(theirs.stop, picked.stop))
            joined.append(k)
        else:
            apart.append(k)
    start, end = yard.span(picked)
    notes = [str(k["note"]) for k in joined if k.get("note") and k["note"] != BROUGHT_BACK_BY_YOU]
    apart.append({
        "id": f"{KEEP_PREFIX}{start:.2f}-{end:.2f}", "start": start, "end": end, "origin": ORIGIN_EXACT,
        "text": yard.text(picked),
        "note": "; ".join(dict.fromkeys([*notes, *([note] if note else [])])) or BROUGHT_BACK_BY_YOU,
    })
    return apart


def _keep_whole(ctx: Context, yard: _Yard, picked: range, span: Span, note: str, keep: list[dict[str, Any]],
                cuts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``keep`` with ``span`` kept, widened to whole sentences and joined to flagged parts it overlaps.

    Where the widened part holds a cut of the creator's that she did not pick
    now, her cut stays and the kept part splits around it.
    """
    start, end = widen_to_sentences(span[0], span[1], ctx.words)
    merged = [k for k in keep if _origin(k) == ORIGIN_FLAGGED and _overlaps(float(k["start"]), float(k["end"]), start, end)]
    notes = [str(k.get("note")) for k in merged if k.get("note") and k.get("note") != KEPT_BY_YOU]
    for k in merged:
        start, end = min(start, float(k["start"])), max(end, float(k["end"]))
    holes = [yard.zone(yard.picked(float(c["start"]), float(c["end"])), ctx.duration) for c in cuts
             if _overlaps(float(c["start"]), float(c["end"]), start, end)]
    keep = [k for k in keep if k not in merged]
    text = "; ".join(dict.fromkeys([*notes, *([note] if note else [])])) or KEPT_BY_YOU
    for a, b in edits.span_without(start, end, holes):
        a, b = round(a, 3), round(b, 3)
        if holes and not yard.picked(a, b):
            continue
        keep.append({
            "id": f"{KEEP_PREFIX}{a:.2f}-{b:.2f}", "start": a, "end": b, "origin": ORIGIN_FLAGGED,
            "text": _shorten(" ".join(_text_between(ctx.words, a, b)), KEEP_TEXT_CHARS), "note": text,
        })
    return keep


def add_keep(ctx: Context, body: Any) -> dict[str, Any]:
    """Keep a stretch no matter what. Answers the new state.

    Without ``exact`` the stretch widens to whole sentences. Claude's cuts
    that overlap it are put back and automatic trims inside it are dropped.
    A keep that overlaps an earlier flagged one merges with it.

    With ``exact`` true it keeps exactly the words whose middle falls inside,
    which is how one removed word comes back. A cut of Claude's splits around
    them, and an automatic trim leaves them in at every pace.

    Either way the newer action wins over her own cuts: the picked words
    come out of any cut of hers.
    """
    example = '{"start": 480.1, "end": 484.0, "note": "the story about the trip"}'
    body = _need_object(body, {"start", "end", "note", "exact"}, example)
    start, end = _need_span(ctx, body)
    exact = body.get("exact", False)
    if not isinstance(exact, bool):
        raise StudioError("exact must be true or false.")
    if end <= start and not exact:
        raise StudioError(f"The start must come before the end ({edits.clock(start)} to {edits.clock(end)}).")
    note = _note_of(body)
    requested, keep, cuts = _copies(ctx.edit)
    if len(keep) >= MAX_KEEPS:
        raise StudioError(f"This video already has {len(keep)} kept parts. Remove some before adding more.")
    yard = _Yard.of(ctx.words)
    picked = _need_words(yard, start, end) if exact else yard.picked(start, end)
    back = _removed_now(ctx, yard, picked)
    cuts, _ = _cuts_without(cuts, yard, picked)
    if exact:
        keep = _keep_exactly(ctx, yard, picked, note, keep)
        start, end = yard.span(picked)
    else:
        keep = _keep_whole(ctx, yard, picked, (start, end), note, keep, cuts)
        start, end = widen_to_sentences(start, end, ctx.words)
    _let_go_of_overrides(requested, start, end)
    before = snapshot(ctx)
    _rebuild(ctx, saved_treatment(ctx.edit), requested=requested, creator_cuts=cuts,
             keep=sorted(keep, key=lambda k: float(k["start"])), undo=_undo_of(ctx.edit))
    return _answer(ctx, before, words_back=back if exact else 0)


def remove_keep(ctx: Context, body: Any) -> dict[str, Any]:
    """Remove one kept part and rebuild the edit. Answers the new state."""
    body = _need_object(body, {"id"}, '{"id": "k479.28-486.20"}')
    kid = _need_id(body, KEEP_PREFIX)
    keep = [dict(k) for k in ctx.edit.get("keep", [])]
    left = [k for k in keep if keep_id(k) != kid]
    if len(left) == len(keep):
        raise StudioError("That kept part is not in this video any more. Reload the page to see the current list.")
    before = snapshot(ctx)
    _rebuild(ctx, saved_treatment(ctx.edit), keep=left, undo=_undo_of(ctx.edit))
    return _answer(ctx, before)


def undo(ctx: Context, body: Any) -> dict[str, Any]:
    """Take back the last thing the creator did to the words. One step. Answers the new state.

    That is the last cut, take-back, keep or bring-back. The pace and the
    switches stay as they are now.
    """
    _need_object(body, set(), "{}")
    saved = ctx.edit.get("undo")
    if not isinstance(saved, dict):
        raise StudioError("There is nothing to undo.")
    before = snapshot(ctx)
    _rebuild(
        ctx, saved_treatment(ctx.edit), requested=list(saved.get("requested", [])),
        keep=list(saved.get("keep", [])), creator_cuts=list(saved.get("creator_cuts", [])),
    )
    return _answer(ctx, before)


def new_samples(ctx: Context, body: Any) -> dict[str, Any]:
    """Pick three samples never used before, until every choice has been used. Answers the new state."""
    _need_object(body, set(), "{}")
    made = _remake(ctx, saved_treatment(ctx.edit))
    store = load_samples_store(ctx.project)
    used = [(float(a), float(b)) for a, b in store["used"]]
    current = [(float(s["start"]), float(s["end"])) for s in store["samples"]]
    picked, forgot = _pick(ctx, made, used + current)
    history = [] if forgot else [list(u) for u in used]
    history += [[s["start"], s["end"]] for s in picked]
    _save_samples_store(ctx.project, picked, history)
    return page_state(ctx, changed=None)


def save_as_usual(ctx: Context, body: Any) -> dict[str, Any]:
    """Save this video's pace and switches as the creator's usual. Answers the new state."""
    _need_object(body, set(), "{}")
    treatment = saved_treatment(ctx.edit)
    write_json_atomic(usual_path(), {
        "version": TREATMENT_VERSION, "updated_at": now_iso(),
        "pace": treatment.pace, "take_out": treatment.as_dict()["take_out"], "fine": treatment.fine_values(),
    })
    return page_state(ctx, changed=None)


# ── What the creator changed, for Claude ──────────────────────────────────────


# What get_edit tells Claude beside the creator's slider values.
FINE_IS_HERS = (
    "The creator set these two with the sliders on the page; the pace reads custom. They are hers. "
    "Leave pace and gap_length out of set_edit to keep them."
)


def _shown(span: dict[str, Any], **more: Any) -> dict[str, Any]:
    start, end = float(span["start"]), float(span["end"])
    return {"clock": edits.clock_span(start, end), "start": span["start"], "end": span["end"], **more}


def cut_word_counts(cuts: list[dict[str, Any]]) -> list[list[Any]]:
    """The words the creator cut by hand, with counts, most first: ``[["like", 11], ["so", 3]]``.

    A cut of one or two words counts as what it says ("you know"); a longer
    cut counts each of its words.
    """
    tally: dict[str, int] = {}
    for c in cuts:
        said = [w for w in (plain_text(t) for t in str(c.get("text", "")).split()) if w]
        for text in ([" ".join(said)] if 0 < len(said) <= 2 else said):
            tally[text] = tally.get(text, 0) + 1
    return [[t, n] for t, n in sorted(tally.items(), key=lambda item: -item[1])]


def creator_changes(edit: dict[str, Any]) -> dict[str, Any] | None:
    """What the creator changed on the treatment page, or None.

    ``pace`` and ``take_out`` appear only when they differ from what Claude
    asked for. ``fine`` holds the two slider values when the pace is the
    creator's own. ``put_back`` lists Claude's cuts the creator put back,
    ``kept`` the parts they flagged to keep, ``brought_back`` the words they
    brought back one by one, ``cuts`` the cuts they made by hand, and
    ``cut_words`` what those cuts say, counted.
    """
    out: dict[str, Any] = {}
    now = saved_treatment(edit)
    asked = edit.get("set_by_claude")
    if isinstance(asked, dict):
        claude = Treatment.from_dict(asked)
        if claude.pace != now.pace and now.trims:
            out["pace"] = {"claude": claude.pace, "creator": now.pace}
        switched = {k: k in now.take_out for k in edits.SWITCHES if (k in now.take_out) != (k in claude.take_out)}
        if switched:
            out["take_out"] = switched
    if now.custom:
        out["fine"] = {
            **now.fine_values(), "speech_kept_between_cuts": speech_kept(now.level().rhythm), "note": FINE_IS_HERS,
        }
    keep = edit.get("keep", [])
    put_back = [k for k in keep if k.get("origin") == ORIGIN_PUT_BACK]
    if put_back:
        out["put_back"] = [
            _shown(k, reason=str(k.get("note", "")).removeprefix("Put back by the creator. Claude's reason was: "))
            for k in put_back
        ]
    flagged = [k for k in keep if k.get("origin") == ORIGIN_FLAGGED]
    if flagged:
        out["kept"] = [_shown(k, text=k.get("text", ""), note=k.get("note", KEPT_BY_YOU)) for k in flagged]
    exact = [k for k in keep if k.get("origin") == ORIGIN_EXACT]
    if exact:
        out["brought_back"] = [_shown(k, text=k.get("text", "")) for k in exact]
    cuts = edit.get("creator_cuts", [])
    if cuts:
        out["cuts"] = [_shown(c, text=c.get("text", "")) for c in cuts]
        out["cut_words"] = cut_word_counts(cuts)
    return out or None
