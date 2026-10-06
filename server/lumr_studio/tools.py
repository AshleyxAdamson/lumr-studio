"""One plain function per tool in TOOLS.md.

Each takes the tool's inputs, does the work through the domain modules, and
returns a short JSON-able dict. None of them know about MCP: server.py wraps
them, and tests call them directly. Collaborators that touch the outside world
(the speech model, the aligner, measured silences, measured sounds, the job registry) are
keyword parameters, so tests can pass fakes.

Every tool that reads words gets them from ``word_times``, the one source.
"""

from __future__ import annotations

import logging
import threading
import webbrowser
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from lumr_studio import edit as edits
from lumr_studio import frames as screenshots
from lumr_studio import models, taste
from lumr_studio import treatment as treatments
from lumr_studio import word_times
from lumr_studio.aligner import Wav2Vec2Aligner
from lumr_studio.analysis import analyze
from lumr_studio.autocuts import TRIM_KINDS, check_gap_length
from lumr_studio.engine.render import kept_segments
from lumr_studio.errors import StudioError
from lumr_studio.jobs import JOBS, JobHandle, JobRegistry
from lumr_studio.joins import check_joins, join_rows
from lumr_studio.look import render_join_picture
from lumr_studio.pace import CUSTOM, describe_paces, get_pace
from lumr_studio.project import Project, append_receipt, open_project, reserve_unique_path
from lumr_studio.publish_kit import write_publish_kit
from lumr_studio.render_jobs import (
    RENDER_KIND,
    holds_earlier_edit,
    preview_window,
    render_preview,
    start_full_render,
)
from lumr_studio.review import ACCEPTED, EDITED, RESTORED, load_review
from lumr_studio.review_server import review_url
from lumr_studio.silences import SilenceProvider, measured_silences
from lumr_studio.sounds import classify_events, load_or_measure_labels
from lumr_studio.timeline import edited_duration, to_edited_time
from lumr_studio.transcript import pack_transcript, timing_report
from lumr_studio.transcription import Transcriber, run_transcription, speech_transcriber, transcript_summary
from lumr_studio.treatment import Treatment
from lumr_studio.word_finder import check_phrases, find_places, pack_places
from lumr_studio.word_times import Aligner


logger = logging.getLogger(__name__)

Words = list[dict[str, Any]]
# Labels for the transcript's sounds: which are laughs. See sounds.py.
LabelProvider = Callable[[Project, Words], list[dict[str, Any]]]

# job_status holds a call open at most this long. MCP clients give up on a
# tool call after about a minute.
MAX_WAIT_SECONDS = 50.0


def measured_labels(project: Project, words: Words) -> list[dict[str, Any]]:
    """The production provider: labels from the measured audio, cached per project.

    Falls back to labels from length and position alone when the audio can't
    be measured, so a decoding problem never blocks an edit.
    """
    try:
        return load_or_measure_labels(project, words)
    except StudioError as exc:
        logger.warning("could not measure sounds, labelling from the transcript alone: %s", exc)
        return classify_events(words)


def unmeasured_labels(_project: Project, words: Words) -> list[dict[str, Any]]:
    """A provider that decodes nothing: labels from length and position alone."""
    return classify_events(words)


def _joins_report(cuts: list[dict[str, Any]], words: Words, duration: float, labels: list[dict[str, Any]]) -> dict[str, Any]:
    """The join self-check without its row dicts: the text form says the same in less."""
    report = check_joins(cuts, words, duration, labels=labels)
    report.pop("rows")
    return report


# The kind of job that fetches models, and what transcribe says when the word times stay estimates.
DOWNLOAD_KIND = "download_models"
RUNS_ON_ESTIMATES = "The edit runs on estimated word times, so the harder paces find less to cut."
EDIT_PLACED_AGAIN = (
    "The saved edit was placed again on the measured word times. Pauses the estimates hid are "
    "now cut, so it removes more. Ids of cuts changed; read get_edit before you change one."
)


def _edit_totals(project: Project, duration: float) -> dict[str, Any] | None:
    """Cut count and time removed of the saved edit, or None when there is none to read."""
    try:
        edit = edits.load_edit(project, duration)
    except StudioError:
        return None
    if not edit.get("cuts"):
        return None
    summary = edits.edit_summary(edit, duration)
    return {"cut_count": summary["cut_count"], "removed_seconds": summary["removed_seconds"]}


def _measure_word_times(
    project: Project, aligner: Aligner, silences: SilenceProvider, labels: LabelProvider
) -> dict[str, Any]:
    """Make measured word times, and place the saved edit again on them.

    Never fails the job: a machine that can't align answers
    ``word_times: "estimated"`` with a note that says why.
    """
    try:
        made = word_times.align_project(project, aligner=aligner, silences=silences)
    except StudioError as why:
        return {"word_times": word_times.ESTIMATED, "word_times_note": f"{why} {RUNS_ON_ESTIMATES}"}
    except Exception as exc:
        logger.exception("measuring word times failed for %s", project.video)
        return {
            "word_times": word_times.ESTIMATED,
            "word_times_note": f"Measuring word times failed ({type(exc).__name__}). {RUNS_ON_ESTIMATES}",
        }
    result: dict[str, Any] = {"word_times": word_times.MEASURED, "aligning": made}
    if made.get("warning"):
        result["word_times_warning"] = made["warning"]
    duration = project.duration()
    before = _edit_totals(project, duration)
    ctx = treatments.load_context(project, duration=duration, silences=silences, labels_for=labels)
    if before is not None:
        result["edit"] = {"before": before, "now": _edit_totals(project, duration), "note": EDIT_PLACED_AGAIN}
    # Plan every pace now, inside the wait Claude already has, so the page opens quickly later.
    treatments.plan_every_pace(ctx)
    return result


# What transcribe says when a model is missing: for Claude, to act on.
ASK_BEFORE_DOWNLOADING = (
    "Nothing has downloaded. Tell the creator what would download, how big it is, where from and under "
    "which license, and ask before you go on. Only when they say yes, call transcribe again with "
    "download_models set to true, and wait on job_status."
)
IF_THEY_SAY_NO = " If they say no, go on with the transcript as it is, on estimated word times."
MODELS_DOWNLOADED_NEXT = "Call transcribe again with the same video_path. It goes on now."


def _needs_models(missing: list[models.Model], summary: dict[str, Any] | None) -> dict[str, Any]:
    """The ``needs_models`` answer. With a ``summary``, the transcript is there and only word times are missing."""
    described = [m.describe() for m in missing]
    answer: dict[str, Any] = {
        "status": "needs_models",
        "models": described,
        "total_mb": sum(m["size_mb"] for m in described),  # the sizes as shown, so the sum adds up
        "note": ASK_BEFORE_DOWNLOADING + (IF_THEY_SAY_NO if summary else ""),
    }
    if summary:
        answer.update(summary, word_times_note=RUNS_ON_ESTIMATES)
    return answer


def _start_download(
    project: Project, missing: list[models.Model], opener: models.Opener, jobs: JobRegistry
) -> dict[str, Any]:
    """Start the download of ``missing``, or join the one already running for this video."""

    def work(handle: JobHandle) -> dict[str, Any]:
        fetched = models.download(missing, opener, handle.set_progress)
        return {"models_downloaded": [m.title for m in fetched], "next": MODELS_DOWNLOADED_NEXT}

    job_id, _started = jobs.start_one(DOWNLOAD_KIND, project, work)
    return {"status": "downloading", "job_id": job_id, "models": [m.describe() for m in missing]}


def transcribe(
    video_path: str,
    force: bool = False,
    download_models: bool = False,
    *,
    transcriber: Transcriber = speech_transcriber,
    aligner: Aligner | None = None,
    silences: SilenceProvider = measured_silences,
    labels: LabelProvider | None = None,
    jobs: JobRegistry = JOBS,
    model_set: models.ModelSet | None = None,
    opener: models.Opener = models.open_url,
) -> dict[str, Any]:
    """Start transcription and the measuring of word times, or report what already exists.

    One job does both, so Claude waits once. With a transcript but no
    measured word times, the job only measures them (``status: "aligning"``).
    Before any of that, a model this call needs may be missing:
    ``status: "needs_models"`` names them and starts nothing, and with
    ``download_models`` true a ``download_models`` job fetches them
    (``status: "downloading"``). On a machine that can't measure them for
    another reason nothing starts and the answer says why under
    ``word_times_note``.
    """
    project = open_project(video_path)
    aligner = aligner or Wav2Vec2Aligner()
    labels = labels or measured_labels
    model_set = models.PRODUCTION if model_set is None else model_set
    have_transcript = project.words_path.exists() and not force
    summary = transcript_summary(project) if have_transcript else None
    measured = summary is not None and summary["word_times"] == word_times.MEASURED
    wanted = [] if have_transcript else [model_set.speech]
    if not measured:
        wanted.append(model_set.aligner)
    missing = [m for m in wanted if m is not None and not m.present()]
    if missing:
        if download_models:
            return _start_download(project, missing, opener, jobs)
        return _needs_models(missing, summary)
    if summary is None:
        job_id = jobs.start("transcribe", project, lambda _h: {
            **run_transcription(project, force, transcriber),
            **_measure_word_times(project, aligner, silences, labels),
        })
        return {"job_id": job_id}
    if measured:
        return {"status": "exists", **summary}
    lacks = aligner.lacks()
    if lacks:
        return {"status": "exists", **summary, "word_times_note": f"{lacks} {RUNS_ON_ESTIMATES}"}
    job_id = jobs.start("transcribe", project, lambda _h: {
        **transcript_summary(project), **_measure_word_times(project, aligner, silences, labels),
    })
    return {"status": "aligning", "job_id": job_id, **summary}


def read_transcript(
    video_path: str,
    start: float | None = None,
    end: float | None = None,
    show_cuts: bool = False,
    *,
    labels: LabelProvider = measured_labels,
) -> dict[str, Any]:
    """Packed transcript text for a window, ending in ``NEXT <time>`` or ``END``."""
    project = open_project(video_path)
    words = word_times.load_words(project)
    removed = None
    if show_cuts:
        removed = edits.removed_spans(edits.load_edit(project, project.duration()))
    return pack_transcript(words, start=start, end=end, removed=removed, labels=labels(project, words))


def analyze_take(video_path: str, *, silences: SilenceProvider = measured_silences) -> dict[str, Any]:
    """The compact take report, with what each pace level would trim."""
    project = open_project(video_path)
    times = word_times.read(project)
    words = times.words
    duration = project.duration()
    measured = silences(project.video)
    report = analyze(words, duration, measured)
    report["word_times"] = times.source
    if times.note:
        report["word_times_note"] = times.note
    report["paces"] = [
        {**level, **_pace_estimate(project, words, duration, measured, level["pace"])} for level in describe_paces()
    ]
    return report


def _pace_estimate(project: Project, words: Words, duration: float, measured: list[Any], name: str) -> dict[str, Any]:
    """How many trims ``name`` would make on this take and the time they'd save."""
    made = treatments.make_edit(
        [], words, duration, treatment=Treatment(pace=name), silences=measured, sounds=[], keeps=[],
        plans_in=project.root,
    )
    trims = made.outcome.cuts
    return {
        "trims": len(trims),
        "seconds_saved": round(sum(float(t["end"]) - float(t["start"]) for t in trims), 1),
    }


def get_edit(
    video_path: str,
    include_auto: bool = False,
    check: bool = False,
    *,
    labels: LabelProvider = measured_labels,
    jobs: JobRegistry = JOBS,
) -> dict[str, Any]:
    """Claude's saved cuts with reasons, an automatic-cut summary, and the totals.

    With ``check`` the result also carries ``joins``, the self-check of every join.
    When the creator changed something on the treatment page, ``creator``
    says what: the pace, the switches, the cuts the creator put back, the
    parts they flagged to keep, the words they brought back, and the cuts
    they made by hand. ``taste`` sums up what the creator changed on their
    other videos, with ``lessons`` in plain sentences; it stays on this Mac,
    and the creator can have Lumr forget it through Claude (``forget_taste``). When a full
    render of this video has run since the server started, ``export`` names
    its job, so Claude can follow an export the creator started from the
    page. ``word_times`` says whether the word times are measured or
    estimated.
    """
    project = open_project(video_path)
    duration = project.duration()
    edit = edits.load_edit(project, duration)
    summary = edits.edit_summary(edit, duration, include_auto=include_auto)
    if project.words_path.exists():
        summary["word_times"] = word_times.source_of(project)
    picked = _picks_summary(project, edit, duration)
    if picked:
        summary["picks"] = picked
    reviewed = _review_summary(project, edit)
    if reviewed:
        summary["review"] = reviewed
    changes = treatments.creator_changes(edit)
    if changes:
        summary["creator"] = changes
    learned = taste.learn(exclude=project.root)
    if learned:
        summary["taste"] = learned
    exported = _latest_export(project, edit, jobs)
    if exported:
        summary["export"] = exported
    if check:
        words = word_times.load_words(project)
        summary["joins"] = _joins_report(edit.get("cuts", []), words, duration, labels(project, words))
    return summary


def _picks_summary(project: Project, edit: dict[str, Any], duration: float) -> dict[str, Any] | None:
    """The filler words Claude picked and whether their switch is on, or None when Claude has not picked.

    ``{switch_on, picked, picks: [{id, clock, text, reason}]}``. What became
    of each pick when the edit was built is in set_edit's answer.
    """
    picks = treatments.saved_picks(edit)
    if picks is None:
        return None
    return {
        "switch_on": treatments.saved_treatment(edit).picks_on,
        "picked": len(picks),
        "picks": [
            {"id": k["id"], "clock": edits.clock(float(k["start"])), "text": k.get("text", ""), "reason": k.get("reason", "")}
            for k in picks
        ],
    }


def _latest_export(project: Project, edit: dict[str, Any], jobs: JobRegistry) -> dict[str, Any] | None:
    """The latest full render of this video, from the page or from render, or None when there is none.

    ``{job_id, status, edit_changed_since}``: pass ``job_id`` to job_status
    for the progress, the file or the error. ``edit_changed_since`` is true
    when the saved edit now removes something else than the render holds.
    """
    job = jobs.latest(RENDER_KIND, project)
    if job is None:
        return None
    return {"job_id": job.job_id, "status": job.status, "edit_changed_since": holds_earlier_edit(job, edit)}


def _review_summary(project: Project, edit: dict[str, Any]) -> dict[str, Any] | None:
    """What the creator decided on the review page, or None when they haven't used it.

    ``restored`` are cuts they put back (now spans to keep), ``changed`` are
    cuts whose edges they moved. ``waiting_for_apply`` counts decisions made on
    the page that Apply hasn't written into the edit yet.
    """
    decisions = load_review(project).get("decisions", {})
    # Keeps made on the treatment page carry an origin and are reported under
    # ``creator``; the older page's restores have none.
    restored = [k for k in edit.get("keep", []) if not k.get("origin")]
    changed = [edits.display_cut(c) for c in edit.get("cuts", []) if c.get("edited_in_review")]
    if not decisions and not restored and not changed:
        return None
    states = [d.get("state") for d in decisions.values()]
    waiting = states.count(RESTORED) + sum(
        1 for d in decisions.values() if d.get("state") == EDITED and not _is_applied(d, edit)
    )
    summary: dict[str, Any] = {
        "accepted": states.count(ACCEPTED),
        "restored": restored,
        "changed": changed,
        "waiting_for_apply": waiting,
    }
    if waiting:
        summary["note"] = (
            f"{waiting} decisions on the review page are not in the edit yet. "
            "Ask the creator to press Apply on the page."
        )
    return summary


def _is_applied(decision: dict[str, Any], edit: dict[str, Any]) -> bool:
    """Whether an edited decision's edges already match a cut in the edit."""
    return any(
        abs(float(c["start"]) - float(decision.get("start", -1))) < 0.005
        and abs(float(c["end"]) - float(decision.get("end", -1))) < 0.005
        for c in edit.get("cuts", [])
    )


def _saved_edit_or_empty(project: Project, duration: float) -> dict[str, Any]:
    """The saved edit, or an empty one when edit.json can't be read: set_edit is how it gets replaced."""
    try:
        return edits.load_edit(project, duration)
    except StudioError:
        return edits.empty_edit()


def _treatment_for_set_edit(
    previous: dict[str, Any], auto_tighten: bool, pace: str | None, gap_length: float | None
) -> Treatment:
    """The treatment a set_edit call asks for.

    A video with a saved treatment keeps its switches and, when ``pace`` is
    left out, its pace; the creator may have set them on the page. A video
    without one starts from the creator's usual. ``auto_tighten`` false turns
    every kind of automatic trim off and leaves the pace where it was.

    A pace the creator set with the sliders (``custom``) stays hers when
    ``pace`` is left out. The switch over Claude's picked filler words is
    the creator's too, and no call changes it.
    """
    start = treatments.saved_treatment(previous) if "treatment" in previous else treatments.starting_treatment()
    picks_switch = start.take_out & {edits.PICK_KIND}
    trims = frozenset()
    if auto_tighten:
        trims = start.trims or treatments.starting_treatment().trims or frozenset(TRIM_KINDS)
    if pace is None and start.custom:
        if gap_length is not None:
            raise StudioError(
                "The creator set the pace with the sliders on the page, and gap_length would change it. "
                "Leave gap_length out to keep her setting, or pass a pace to replace it."
            )
        return Treatment(pace=CUSTOM, take_out=trims | picks_switch, fine=start.fine)
    gap = check_gap_length(gap_length) if gap_length is not None and auto_tighten else None
    return Treatment(pace=get_pace(pace or start.pace).name, take_out=trims | picks_switch, gap_length=gap)


def _picks_for_set_edit(
    picks: list[Any] | None, previous: dict[str, Any], words: Words
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]]]:
    """The filler words to save as picked, and the picks that were refused: ``(picks, rejected)``.

    With ``picks`` left out the list saved before carries over. A list
    replaces it. Raises StudioError when picks were given and every one is
    refused, so a stale list of ids can never wipe a good one.
    """
    if picks is None:
        return treatments.saved_picks(previous), []
    said = edits.SaidOrder.of(words)
    kept: dict[str, dict[str, Any]] = {}
    rejected = []
    for i, raw in enumerate(picks):
        try:
            pick = edits.parse_pick(i, raw, said)
        except edits.CutRejected as why:
            rejected.append({"index": i, "pick": raw, "why": f"Pick {i} {why}"})
            continue
        kept.setdefault(pick["id"], pick)
    if picks and not kept:
        whys = " ".join(r["why"] for r in rejected)
        raise StudioError(
            f"Every pick was refused, so the saved edit was left unchanged. {whys} "
            "Call find_words again and pass its ids. To clear the picks on purpose, pass an empty list."
        )
    return sorted(kept.values(), key=lambda k: k["start"]), rejected


def _picks_report(made: Any, picks: list[dict[str, Any]], rejected: list[dict[str, Any]], switch_on: bool) -> dict[str, Any]:
    """What became of the picked filler words, for set_edit's answer."""
    became = {p["id"]: p for p in made.outcome.picks}
    text = {k["id"]: k.get("text", "") for k in picks}

    def listed(status: str) -> list[dict[str, Any]]:
        return [
            {"id": pid, "clock": edits.clock(p["word_start"]), "text": text.get(pid, ""), **({"why": p["why"]} if p["why"] else {})}
            for pid, p in became.items() if p["status"] == status
        ]

    report: dict[str, Any] = {
        "picked": len(picks), "switch_on": switch_on,
        "out": sum(1 for p in became.values() if p["status"] == edits.PICK_OUT),
        "left_in": listed(edits.PICK_LEFT_IN) + listed(edits.PICK_LOST),
        "kept_by_the_creator": listed(edits.PICK_KEPT),
        "rejected": rejected,
    }
    if not switch_on:
        report["note"] = "The creator switched Filler likes off on the page, so no pick is taken out. The list is saved for when it goes on."
    return report


def set_edit(
    video_path: str,
    cuts: list[dict[str, Any]],
    auto_tighten: bool = False,
    gap_length: float | None = None,
    override_keeps: bool = False,
    pace: str | None = None,
    picks: list[dict[str, Any]] | None = None,
    *,
    silences: SilenceProvider = measured_silences,
    labels: LabelProvider = measured_labels,
) -> dict[str, Any]:
    """Validate the proposed cuts, replace the saved edit, and report the outcome.

    ``pace`` names how hard ``auto_tighten`` trims (see pace.py); left out, it
    is the pace already saved for this video, else the creator's usual.
    ``gap_length`` overrides only the pace's shortest-pause setting. Each cut
    may carry a ``kind`` (see ``edit.CUT_KINDS``); without one it is "other".

    Spans the creator kept outlive every call: a cut inside one is rejected
    unless ``override_keeps`` is true, and a cut over words the creator
    brought back splits around them. The cuts the creator made by hand
    outlive every call too. Automatic trims stay clear of punchlines and
    laughs. The result carries ``joins``, the self-check of every join in
    the saved edit.

    ``picks`` are single filler words Claude picked by reading, each ``{id,
    reason}`` with an id from ``find_words``. They are no part of ``cuts``:
    the creator has one switch for all of them. Left out, the picks saved
    before stay; a list replaces them. A pick that would clip the word
    beside it is left in, and the result says which.

    An empty ``cuts`` list without ``auto_tighten`` clears the edit, and the
    result says ``cleared: true``. When cuts were given and every one is
    rejected, nothing is saved: a StudioError lists the rejections, so a typo
    can never wipe a good edit.
    """
    if gap_length is not None and not auto_tighten:
        raise StudioError("gap_length only applies with auto_tighten=true. Turn it on or drop gap_length.")
    if pace is not None and not auto_tighten:
        raise StudioError("pace only applies with auto_tighten=true. Turn it on or drop pace.")
    get_pace(pace)
    project = open_project(video_path)
    times = word_times.read(project)
    words = times.words
    duration = project.duration()
    measured = silences(project.video)
    sounds = labels(project, words)
    previous = _saved_edit_or_empty(project, duration)
    keep = list(previous.get("keep", []))
    creator_cuts = list(previous.get("creator_cuts", []))
    ratings = treatments.saved_ratings(previous)
    treatment = _treatment_for_set_edit(previous, auto_tighten, pace, gap_length)
    picked, refused = _picks_for_set_edit(picks, previous, words)
    requested = [
        {**c, edits.OVERRIDE_FLAG: True} if override_keeps and isinstance(c, dict) else c for c in cuts
    ]
    made = treatments.make_edit(
        requested, words, duration, treatment=treatment, silences=measured, sounds=sounds,
        keeps=treatments.keep_spans_of(keep), exact_keeps=treatments.exact_spans_of(keep),
        creator_cuts=creator_cuts, plans_in=project.root, picks=picked,
    )
    outcome = made.outcome
    if cuts and len(outcome.rejected) == len(cuts):
        whys = " ".join(r["why"] for r in outcome.rejected)
        raise StudioError(
            f"Every cut was rejected, so the saved edit was left unchanged. {whys} "
            "Fix the cuts and call set_edit again. To clear the edit on purpose, pass an empty cuts list."
        )
    saved = treatments.save_made(
        project, made, duration=duration, requested=requested, keep=keep, set_by_claude=treatment.as_dict(),
        creator_cuts=creator_cuts, times=times.source, picks=picked, ratings=ratings,
    )
    summary = edits.edit_summary(saved, duration)
    append_receipt(
        project, "set_edit", cut_count=summary["cut_count"], rejected=len(outcome.rejected),
        removed_seconds=summary["removed_seconds"], new_duration=summary["new_duration"],
    )
    result: dict[str, Any] = {
        "applied": outcome.applied,
        "adjusted": outcome.adjusted,
        "rejected": outcome.rejected,
        "removed_seconds": summary["removed_seconds"],
        "new_duration": summary["new_duration"],
    }
    if not cuts and not auto_tighten and not picked:
        result["cleared"] = True
    if picked is not None:
        # Counted with the switch on, so Claude reads what its picks would do whether it is on or off.
        on = made if treatment.picks_on else treatments.make_edit(
            requested, words, duration, silences=measured, sounds=sounds, keeps=treatments.keep_spans_of(keep),
            exact_keeps=treatments.exact_spans_of(keep), creator_cuts=creator_cuts, plans_in=project.root, picks=picked,
            treatment=Treatment(pace=treatment.pace, take_out=treatment.take_out | {edits.PICK_KIND},
                                gap_length=treatment.gap_length, fine=treatment.fine),
        )
        result["picks"] = _picks_report(on, picked, refused, treatment.picks_on)
    warning = timing_report(words).get("warning")
    if warning:
        result["warning"] = warning
    result["word_times"] = times.source
    if creator_cuts:
        result["creator_cuts_kept"] = len(creator_cuts)
    if auto_tighten:
        result["auto"] = {
            "pace": treatment.pace,
            "fine": treatment.fine_values(),
            "take_out": sorted(treatment.trims, key=list(TRIM_KINDS).index),
            "trims_pauses_longer_than": made.shortest,
            "planned": made.planned,
            "shortened_to_leave_a_pause": made.pauses.shortened,
            "dropped_to_leave_a_pause": made.pauses.dropped,
            "cuts_after_merge": sum(1 for c in outcome.cuts if c["source"] == "auto"),
            "unsafe_kept_at_original_edges": outcome.auto_unsafe,
            "edges_moved_out_of_words": made.edges_moved_out_of_words,
            "skipped_to_protect_jokes": outcome.auto_skipped_protected,
            "skipped_in_spans_the_creator_restored": outcome.auto_skipped_kept,
        }
    result["joins"] = _joins_report(outcome.cuts, words, duration, sounds)
    return result


def find_words(
    video_path: str,
    words: list[str],
    start: float | None = None,
    *,
    silences: SilenceProvider = measured_silences,
    labels: LabelProvider = measured_labels,
) -> dict[str, Any]:
    """Every place each of ``words`` is said, as packed text, for Claude to pick the fillers among them.

    ``words`` holds one to five words or short phrases. Each place has an
    id, its clock time, the words around it, the quiet either side, whether
    the saved edit removes it already and who does, and whether a cut there
    would be clean. ``start`` goes on from a ``NEXT <time>`` line.
    """
    phrases = check_phrases(words)
    project = open_project(video_path)
    duration = project.duration()
    if start is not None and not 0 <= start <= duration:
        raise StudioError(f"start {start} is outside the video (0 to {duration:.2f}).")
    ctx = treatments.load_context(project, duration=duration, silences=silences, labels_for=labels)
    codes = treatments.removal_codes(ctx) if ctx.edit.get("cuts") else {}
    places = find_places(ctx.words, phrases, ctx.silences, duration, codes)
    return {**pack_places(places, phrases, start=start), "word_times": ctx.times}


def _saved_removed(video_path: str) -> tuple[Project, float, list[tuple[float, float]]]:
    """The project, the video duration, and the saved edit's removed ranges."""
    project = open_project(video_path)
    duration = project.duration()
    return project, duration, edits.removed_spans(edits.load_edit(project, duration))


def preview(video_path: str, at: float, pad: float = 6.0, *, jobs: JobRegistry = JOBS) -> dict[str, Any]:
    """Start rendering a short clip of the edited result around source time ``at``."""
    project, duration, removed = _saved_removed(video_path)
    preview_window(removed, duration, at, pad)  # fail now, not inside the job
    job_id = jobs.start("preview", project, lambda h: render_preview(project, removed, duration, at, pad, h))
    return {"job_id": job_id}


# What render says when it starts nothing because one is already going.
RENDER_ALREADY_RUNNING = (
    "A render of this video is already running, so no second one was started. It may be an "
    "export the creator started from the page. It holds the edit as saved when it started. "
    "Wait for it with job_status; call render again after it finishes if the edit changed since."
)


def render(video_path: str, *, jobs: JobRegistry = JOBS) -> dict[str, Any]:
    """Start rendering the full edited video into exports/.

    One full render runs per video at a time. The creator can start one from
    the page (Export video); when one is running, this answers its job id
    with ``already_running: true`` and starts nothing.
    """
    project = open_project(video_path)
    job_id, started = start_full_render(project, jobs=jobs)
    if started:
        return {"job_id": job_id}
    return {"job_id": job_id, "already_running": True, "note": RENDER_ALREADY_RUNNING}


def look(
    video_path: str,
    at: float,
    span: float = 3.0,
    *,
    labels: LabelProvider = measured_labels,
) -> dict[str, Any]:
    """Draw the join nearest source time ``at`` into ``looks/`` and describe the picture."""
    project, duration, removed = _saved_removed(video_path)
    words = word_times.load_words(project)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = reserve_unique_path(project.looks_dir, f"join-{at:.1f}s-{stamp}", ".png")
    try:
        picture = render_join_picture(
            project.video, removed, words, duration, at, out, span=span, labels=labels(project, words)
        )
    except BaseException:
        out.unlink(missing_ok=True)
        raise
    append_receipt(project, "look", at=at, path=picture["path"])
    return picture


def frames(
    video_path: str,
    times: list[float],
    region: list[float] | None = None,
    *,
    footage: screenshots.ScreenFootage | None = None,
) -> dict[str, Any]:
    """Save a screenshot of the source video at each of ``times`` into ``frames/`` and describe them.

    Needs no transcript and no saved edit. Each picture is the frame on screen
    at that time, cropped to ``region`` when there is one. Nothing is written
    unless every frame was read.
    """
    project = open_project(video_path)
    shots, info, box = screenshots.capture(project.video, times, region, footage=footage)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    described: list[dict[str, Any]] = []
    reserved: list[Path] = []
    try:
        for shot in shots:
            cs = round(shot.at * 100)
            base = f"frame-{cs // 6000}m{(cs % 6000) / 100:05.2f}s-{stamp}"
            out = reserve_unique_path(project.frames_dir, base, shot.suffix)
            reserved.append(out)
            out.write_bytes(shot.data)
            described.append(shot.describe(out))
    except BaseException:
        for out in reserved:
            out.unlink(missing_ok=True)
        raise
    append_receipt(project, "frames", times=[s.at for s in shots], paths=[d["path"] for d in described])
    return {
        "frames": described,
        "source_size": [info.width, info.height],
        "region": list(box) if box else None,
    }


BrowserOpener = Callable[[str], Any]


def review(
    video_path: str,
    open_browser: bool = True,
    *,
    address_for: Callable[[Project], str] | None = None,
    opener: BrowserOpener = webbrowser.open,
    labels: LabelProvider = measured_labels,
    silences: SilenceProvider = measured_silences,
) -> dict[str, Any]:
    """Serve the treatment page for this video and return its address.

    On the page the creator sets the pace and what to take out, puts back or
    keeps out each of Claude's cuts, cuts words by hand and brings words
    back, flags parts to keep, and listens to three samples. ``address_for``
    gives the page address for a project; the default starts the shared
    local server with the production collaborators. An edit placed on other
    word times than the project has now is placed again first, so the counts
    are the ones the page will show.
    """
    project = open_project(video_path)
    duration = project.duration()
    edit = edits.load_edit(project, duration)
    if not edit.get("cuts") and not edit.get("requested"):
        raise StudioError("There is no edit to review yet. Save an edit with set_edit first.")
    ctx = treatments.load_context(project, duration=duration, silences=silences, labels_for=labels)
    edit, cuts = ctx.edit, ctx.edit.get("cuts", [])
    rows = join_rows(cuts, ctx.words, duration, labels=ctx.sounds) if cuts else []
    url = (address_for or _review_address)(project)
    opened = bool(open_browser and opener(url))
    claude = sum(1 for c in cuts if c.get("source") == edits.BY_CLAUDE)
    append_receipt(project, "review", cuts=claude)
    return {
        "url": url,
        "opened_in_browser": opened,
        "claude_cuts": claude,
        "creator_cuts": len(edit.get("creator_cuts", [])),
        "trims": sum(1 for c in cuts if c.get("source") == edits.BY_AUTO),
        "need_a_look": sum(1 for row in rows if row.get("flags")),
        "word_times": ctx.times,
    }


def _review_address(project: Project) -> str:
    """The page's address, with the paces being planned already so the page's first load is quick.

    Planning the automatic trims at all six paces takes about two seconds on
    a 20 minute take. The page asks for the state the moment it opens; a
    thread started here has the plans ready, or nearly, by then.
    """
    url = review_url(project, labels_for=measured_labels)
    threading.Thread(target=_plan_ahead, args=(project,), name="lumr-plan-ahead", daemon=True).start()
    return url


def _plan_ahead(project: Project) -> None:
    """Work out the edit at every pace, which fills the caches the page reads. Saves nothing new."""
    try:
        ctx = treatments.load_context(project, silences=measured_silences, labels_for=measured_labels)
        treatments.plan_every_pace(ctx)
    except Exception as why:  # the page will say what is wrong when it asks
        logger.info("could not plan the paces ahead of the page: %s", why)


def job_status(job_id: str, wait: float = 0.0, *, jobs: JobRegistry = JOBS) -> dict[str, Any]:
    """Status, progress, result or error of a job.

    With ``wait`` the call holds until the job finishes or ``wait`` seconds pass.
    """
    if not 0 <= wait <= MAX_WAIT_SECONDS:
        raise StudioError(f"wait {wait} must be between 0 and {MAX_WAIT_SECONDS:.0f} seconds.")
    if wait:
        return jobs.wait(job_id, timeout=wait)
    return jobs.status(job_id)


def forget_taste(everything: bool | None = None, word: str | None = None, kind: str | None = None) -> dict[str, Any]:
    """Stop learning from some of what the creator changed, across all their videos. Exactly one input.

    ``everything`` forgets every change made so far; changes made after this
    count again. ``word`` forgets one word, and ``kind`` one kind of cut
    (``edit.CUT_KINDS`` or ``likes``). No video is named: the taste profile
    spans them all. Nothing of the creator's is deleted or changed; only
    ``taste.json`` in the projects folder is written. Answers ``forgot``, what
    was forgotten, and ``taste``, what Lumr still learns from (None when
    nothing).
    """
    asked = [name for name, value in (("everything", everything), ("word", word), ("kind", kind)) if value is not None]
    if len(asked) != 1:
        raise StudioError("Send exactly one of everything (true), word or kind, like {\"word\": \"so\"}.")
    if everything is not None:
        if everything is not True:
            raise StudioError("everything must be true. To forget one word or one kind of cut, send word or kind instead.")
        taste.forget_all()
        forgot = "everything"
    elif word is not None:
        forgot = f'the word "{taste.forget_word(word)}"'
    else:
        forgot = f'the kind "{taste.forget_kind(kind)}"'
    return {"forgot": forgot, "taste": taste.learn(exclude=None)}


def chapter_times(video_path: str, source_times: list[float]) -> dict[str, Any]:
    """Convert source times to edited times through the saved edit."""
    project, duration, removed = _saved_removed(video_path)
    kept = kept_segments(removed, duration)
    out = []
    for t in source_times:
        if not 0 <= t <= duration:
            raise StudioError(f"Source time {t} is outside the video (0 to {duration:.2f}).")
        edited, in_cut = to_edited_time(t, kept)
        out.append({"source": t, "edited": round(edited, 3), "inside_cut": in_cut})
    return {"times": out, "edited_duration": round(edited_duration(kept), 3)}


def save_publish_kit(
    video_path: str,
    titles: list[str],
    description: str,
    chapters: list[dict[str, Any]],
    tags: list[str],
) -> dict[str, Any]:
    """Validate the kit against YouTube's chapter rules and write publish-kit/."""
    project, duration, removed = _saved_removed(video_path)
    files = write_publish_kit(
        project.publish_kit_dir,
        titles=titles, description=description, chapters=chapters, tags=tags,
        edited_duration=edited_duration(kept_segments(removed, duration)),
    )
    append_receipt(project, "save_publish_kit", files=files)
    return {"files": files}
