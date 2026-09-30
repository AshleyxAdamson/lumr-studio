"""What the creator's corrections teach, across every video they have edited.

Each video's ``edit.json`` already says what the creator changed on the page
(``treatment.creator_changes``). This module reads that from every other
project in the projects folder and adds it up: the kinds of cut they put back,
the words they cut by hand or bring back, the pace and switches they choose.
``get_edit`` hands the sum to Claude so the next cut starts from it.

Nothing is stored but a reset marker. The profile is worked out again from the
edits each time, so it can never drift from them, and it never leaves this
Mac. ``forget`` writes ``taste.json`` in the projects folder; a project whose
edit was last saved before that no longer counts, and one changed after it
counts again.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lumr_studio import edit as edits
from lumr_studio.pace import PACES
from lumr_studio.project import projects_root, write_json_atomic
from lumr_studio.transcript import plain_text

log = logging.getLogger(__name__)

TASTE_FILE = "taste.json"
EDIT_FILE = "edit.json"
TASTE_VERSION = 1
# The longest lists the profile carries.
TOP = 10
# A lesson needs this many videos, or this many events in fewer.
LESSON_VIDEOS, LESSON_EVENTS = 2, 3
# Most lessons of one sort, so a long history doesn't bury the ones that matter.
LESSONS_PER_SORT = 3
# A put-back cut whose kind can't be found from Claude's own cuts.
UNKNOWN_KIND = "unknown"
# How each kind of Claude's cuts reads in a lesson.
KIND_WORDS = {"repeat": "restated-point", "false_start": "false-start", "off_topic": "off-topic", "other": "other"}


def taste_path() -> Path:
    """The reset marker: one file in the projects folder, for every video."""
    return projects_root() / TASTE_FILE


def _forgotten_at() -> float | None:
    """When the creator last asked to forget, in epoch seconds, or None.

    A marker that can't be read forgets nothing, since a profile only picks a
    starting point.
    """
    try:
        data = json.loads(taste_path().read_text(encoding="utf-8"))
        return datetime.fromisoformat(str(data["forgotten_at"])).timestamp()
    except (OSError, ValueError, KeyError, TypeError):
        return None


def forget() -> dict[str, str]:
    """Forget what earlier videos taught. Their edits are not touched; only a marker is written.

    The time keeps its microseconds: a project saved a moment before this
    call must read as earlier, and whole seconds could not tell.
    """
    at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    write_json_atomic(taste_path(), {"version": TASTE_VERSION, "forgotten_at": at})
    return {"forgotten_at": at}


def _changed_projects(exclude: Path | None) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """``(name, edit, creator)`` for each project with something the creator changed, skipping ``exclude``.

    A project that can't be read, or holds an edit that can't be summed up,
    is skipped: one broken folder must never cost the creator the rest.
    """
    # The page and Claude both ask; treatment.page_state asks this module in turn.
    from lumr_studio.treatment import creator_changes

    root = projects_root()
    if not root.is_dir():
        return []
    forgotten = _forgotten_at()
    skip = exclude.resolve() if exclude is not None else None
    found = []
    for folder in sorted(root.iterdir()):
        path = folder / EDIT_FILE
        try:
            if not path.is_file() or (skip is not None and folder.resolve() == skip):
                continue
            if forgotten is not None and path.stat().st_mtime <= forgotten:
                continue
            edit = json.loads(path.read_text(encoding="utf-8"))
            changes = creator_changes(edit) if isinstance(edit, dict) else None
        except Exception as err:  # noqa: BLE001 - any one project may fail without costing the rest
            log.debug("Skipping %s for the taste profile: %s", folder.name, err)
            continue
        if changes:
            found.append((folder.name, edit, changes))
    return found


def _span_overlap(a: dict[str, Any], b: dict[str, Any]) -> float:
    return min(float(a["end"]), float(b["end"])) - max(float(a["start"]), float(b["start"]))


def _kind_of(kept: dict[str, Any], requested: list[dict[str, Any]]) -> str:
    """The kind of Claude's cut that ``kept`` put back: the one it overlaps most, else ``unknown``."""
    best, kind = 0.0, UNKNOWN_KIND
    for cut in requested:
        try:
            overlap = _span_overlap(kept, cut)
        except (KeyError, TypeError, ValueError):
            continue
        if overlap > best:
            best = overlap
            kind = cut.get("kind") if cut.get("kind") in edits.CUT_KINDS else edits.DEFAULT_CUT_KIND
    return kind


def _said(text: Any) -> str:
    """The words of ``text`` as said, lower case and without punctuation: ``"Actually,"`` reads ``actually``."""
    return " ".join(w for w in (plain_text(t) for t in str(text).split()) if w)


def _top(tally: dict[str, int]) -> list[list[Any]]:
    """The ``TOP`` most counted entries, most first, ties in the order first seen."""
    return [[k, n] for k, n in sorted(tally.items(), key=lambda item: -item[1])[:TOP]]


def _times(n: int) -> str:
    return "once" if n == 1 else f"{n} times"


def _videos(n: int) -> str:
    return f"{n} video" if n == 1 else f"{n} videos"


def _backed(events: int, videos: int) -> bool:
    return videos >= LESSON_VIDEOS or events >= LESSON_EVENTS


def learn(exclude: Path | None = None) -> dict[str, Any] | None:
    """What every other video's changes teach, or None when none of them changed anything.

    ``exclude`` is the project folder of the video being edited: its own
    changes reach Claude as ``creator`` and are not counted twice. Videos
    whose edit was saved before the last ``forget`` don't count.

    ``videos`` counts the projects with at least one change. ``put_back`` maps
    each kind of Claude's cuts to how many were put back and how many Claude
    proposed in those projects. ``cut_by_hand`` and ``brought_back`` are
    ``[word, count]`` lists, most first. ``lessons`` are plain sentences
    written from the numbers, and only for what at least two videos, or three
    events, back.
    """
    projects = _changed_projects(exclude)
    if not projects:
        return None
    n = len(projects)
    put_back: dict[str, dict[str, int]] = {}
    put_back_in: dict[str, set[str]] = {}
    cut_words: dict[str, int] = {}
    cut_in: dict[str, set[str]] = {}
    brought: dict[str, int] = {}
    brought_in: dict[str, set[str]] = {}
    pace: dict[str, dict[str, int]] = {"claude": {}, "creator": {}}
    took: dict[str, dict[str, int]] = {}
    slower_faster: dict[str, dict[str, int]] = {"faster": {}, "slower": {}}
    kept_parts = 0
    stops = list(PACES)
    for name, edit, changes in projects:
        requested = [r for r in edit.get("requested", []) if isinstance(r, dict)]
        for r in requested:
            kind = r.get("kind") if r.get("kind") in edits.CUT_KINDS else edits.DEFAULT_CUT_KIND
            put_back.setdefault(kind, {"put_back": 0, "proposed": 0})["proposed"] += 1
        for k in changes.get("put_back", []):
            kind = _kind_of(k, requested)
            put_back.setdefault(kind, {"put_back": 0, "proposed": 0})["put_back"] += 1
            put_back_in.setdefault(kind, set()).add(name)
        for text, count in changes.get("cut_words", []):
            cut_words[text] = cut_words.get(text, 0) + count
            cut_in.setdefault(text, set()).add(name)
        for entry in changes.get("brought_back", []):
            said = _said(entry.get("text", ""))
            if 0 < len(said.split()) <= 2:
                brought[said] = brought.get(said, 0) + 1
                brought_in.setdefault(said, set()).add(name)
        kept_parts += len(changes.get("kept", []))
        if "pace" in changes:
            claude, creator = changes["pace"]["claude"], changes["pace"]["creator"]
            pace["claude"][claude] = pace["claude"].get(claude, 0) + 1
            pace["creator"][creator] = pace["creator"].get(creator, 0) + 1
            if claude in stops and creator in stops and creator != claude:
                side = "faster" if stops.index(creator) > stops.index(claude) else "slower"
                slower_faster[side][creator] = slower_faster[side].get(creator, 0) + 1
        for switch, on in changes.get("take_out", {}).items():
            took.setdefault(switch, {"off": 0, "on": 0})["on" if on else "off"] += 1
    profile: dict[str, Any] = {
        "videos": n,
        "put_back": {k: v for k, v in put_back.items() if v["put_back"]},
        "cut_by_hand": _top(cut_words),
        "brought_back": _top(brought),
        "kept_parts": kept_parts,
        "pace": {k: v for k, v in pace.items() if v},
        "take_out": took,
    }
    lessons = _lessons_put_back(put_back, put_back_in)
    lessons += _lessons_cut_by_hand(cut_words, cut_in)
    lessons += _lessons_brought_back(brought, brought_in)
    lessons += _lessons_take_out(took, n)
    lessons += _lessons_pace(slower_faster)
    profile["lessons"] = lessons
    return profile


# ── The lessons: plain sentences for Claude, written from the numbers ─────────


def _lessons_put_back(tally: dict[str, dict[str, int]], where: dict[str, set[str]]) -> list[str]:
    out = []
    for kind in edits.CUT_KINDS:
        got = tally.get(kind)
        if not got or not got["put_back"] or not _backed(got["put_back"], len(where[kind])):
            continue
        back, of = got["put_back"], max(got["put_back"], got["proposed"])
        out.append(f"Put back {back} of {of} {KIND_WORDS[kind]} cuts ({kind}) across {_videos(len(where[kind]))}.")
    return out


def _lessons_cut_by_hand(tally: dict[str, int], where: dict[str, set[str]]) -> list[str]:
    backed = [[w, c] for w, c in _top(tally) if _backed(c, len(where[w]))]
    return [f'Cut "{w}" by hand {_times(c)} across {_videos(len(where[w]))}.' for w, c in backed[:LESSONS_PER_SORT]]


def _lessons_brought_back(tally: dict[str, int], where: dict[str, set[str]]) -> list[str]:
    backed = [[w, c] for w, c in _top(tally) if _backed(c, len(where[w]))]
    return [f'Brought back "{w}" {_times(c)}.' for w, c in backed[:LESSONS_PER_SORT]]


def _lessons_take_out(tally: dict[str, dict[str, int]], videos: int) -> list[str]:
    out = []
    for switch, label in edits.SWITCHES.items():
        for side in ("off", "on"):
            count = tally.get(switch, {}).get(side, 0)
            if count >= LESSON_VIDEOS:
                out.append(f"Switched {label.lower()} {side} in {count} of {videos} videos.")
    return out


def _lessons_pace(tally: dict[str, dict[str, int]]) -> list[str]:
    out = []
    for side, chosen in tally.items():
        count = sum(chosen.values())
        if count >= LESSON_VIDEOS:
            usual = max(chosen, key=lambda p: chosen[p])
            out.append(f"Chose a {side} pace than Claude in {_videos(count)}: {usual}.")
    return out


def summary_for_page(exclude: Path | None = None) -> dict[str, int] | None:
    """``{"videos": n}`` for the line on the page, or None when nothing has been learned."""
    found = _changed_projects(exclude)
    return {"videos": len(found)} if found else None
