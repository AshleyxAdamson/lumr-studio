"""The plain functions behind ``set_overlays`` and ``get_overlays``.

Like tools.py, none of these know about MCP: overlay_mcp.py wraps them and
tests call them directly. Collaborators that touch the outside world (the file
probe, the still drawer) are keyword parameters, so tests can pass fakes.

Overlays are anchored to SOURCE times (see overlays.py). Every answer also
says where each one lands in the edited video under the edit saved right
now, since the creator may have changed the cuts since the overlays were set.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from lumr_studio import edit as edits
from lumr_studio.engine.render import kept_segments
from lumr_studio.errors import StudioError
from lumr_studio.overlay_render import draw_stills
from lumr_studio.overlays import (
    HIDDEN,
    Placement,
    Prober,
    build_overlays,
    describe,
    load_overlays,
    place_on_edit,
    probe_media,
    save_overlays,
)
from lumr_studio.project import Project, append_receipt, open_project, reserve_unique_path
from lumr_studio.timeline import edited_duration, source_spans_for_window
from lumr_studio.word_times import words_if_any

Words = list[dict[str, Any]]
StillDrawer = Callable[..., dict[str, Any]]


def _words(project: Project) -> Words:
    """The transcript when there is one. Overlays work without it, with less to say."""
    return words_if_any(project)


def _saved_edit(project: Project, duration: float) -> tuple[list[tuple[float, float]], str | None]:
    """The saved edit's removed spans, and a note when it can't be read.

    An unreadable edit places overlays on the uncut video rather than failing:
    the overlays themselves are fine, and set_edit is how the edit gets fixed.
    """
    try:
        return edits.removed_spans(edits.load_edit(project, duration)), None
    except StudioError as exc:
        return [], f"The saved edit can't be read ({exc}), so times below are for the uncut video."


def _previous_next_id(project: Project) -> int:
    """The first free id number, so an id is never handed out twice for one video."""
    try:
        return int(load_overlays(project).get("next_id", 1))
    except (StudioError, TypeError, ValueError):
        return 1


def _source_at(kept: list[tuple[float, float]]) -> Callable[[float], float]:
    def at(edited: float) -> float:
        spans = source_spans_for_window(kept, edited, edited + 1e-3)
        return spans[0][0] if spans else kept[-1][1]
    return at


def _short_row(p: Placement) -> dict[str, Any]:
    o = p.overlay
    return {
        "id": o["id"],
        "kind": o["kind"],
        "file": o["file"],
        "clock": f"{edits.clock(o['start'])}-{edits.clock(o['end'])}",
        "place": o["place"],
        "layer": o["layer"],
        "on_screen": {"at": round(p.at, 3), "seconds": round(p.seconds, 3)},
        "status": p.status,
    }


def set_overlays(
    video_path: str,
    overlays: list[dict[str, Any]],
    picture: bool = True,
    *,
    probe: Prober = probe_media,
    stills: StillDrawer = draw_stills,
) -> dict[str, Any]:
    """Check the creator's photos and clips, replace the saved overlays, and report.

    Each overlay that fails a check is rejected with the reason and the rest
    are saved. When overlays were given and every one is rejected, nothing is
    saved: a StudioError lists why. An empty list clears the overlays.

    With ``picture`` the result carries ``picture.path``: stills of the
    overlays as the viewer will see them, for Claude to look at.
    """
    project = open_project(video_path)
    duration = project.duration()
    source = probe(project.video)
    words = _words(project)
    outcome = build_overlays(
        overlays, duration=duration, frame=(source.width, source.height), words=words,
        next_id=_previous_next_id(project), probe=probe,
    )
    if overlays and not outcome.overlays:
        whys = " ".join(r["why"] for r in outcome.rejected)
        raise StudioError(
            f"Every overlay was rejected, so the saved overlays were left unchanged. {whys} "
            "Fix them and call set_overlays again. To clear the overlays on purpose, pass an empty list."
        )
    save_overlays(project, outcome, duration=duration, requested=overlays)
    removed, edit_note = _saved_edit(project, duration)
    kept = kept_segments(removed, duration)
    placements = place_on_edit(outcome.overlays, kept, words)
    append_receipt(project, "set_overlays", saved=len(outcome.overlays), rejected=len(outcome.rejected))

    result: dict[str, Any] = {
        "saved": [_short_row(p) for p in placements],
        "rejected": outcome.rejected,
        "notes": [n for n in [edit_note, *(n for p in placements for n in p.notes)] if n],
    }
    if not overlays:
        result["cleared"] = True
    if picture and any(p.shown for p in placements):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = reserve_unique_path(project.looks_dir, f"overlays-{stamp}", ".jpg")
        try:
            drawn = stills(project.video, placements, _source_at(kept), out)
        except BaseException:
            out.unlink(missing_ok=True)
            raise
        if not drawn.get("path"):
            out.unlink(missing_ok=True)
        result["picture"] = drawn
    return result


def get_overlays(video_path: str) -> dict[str, Any]:
    """The saved overlays, each with where it lands in the edited video right now."""
    project = open_project(video_path)
    duration = project.duration()
    saved = load_overlays(project)
    removed, edit_note = _saved_edit(project, duration)
    kept = kept_segments(removed, duration)
    words = _words(project)
    placements = place_on_edit(saved["overlays"], kept, words)
    rows = [describe(p, words, removed) for p in placements]
    notes = [n for n in [edit_note, *(n for p in placements for n in p.notes)] if n]
    return {
        "overlays": rows,
        "shown": sum(1 for p in placements if p.status != HIDDEN),
        "not_shown": [p.overlay["id"] for p in placements if p.status == HIDDEN],
        "notes": notes,
        "edited_duration": round(edited_duration(kept), 3),
    }
