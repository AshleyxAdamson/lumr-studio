"""What the creator's corrections teach, across every video they have edited.

Each video's ``edit.json`` already says what the creator changed on the page
(``treatment.creator_changes``). This module reads that from every other
project in the projects folder and adds it up: the kinds of cut they put back,
the words they cut by hand or bring back, the pace and switches they choose.
``get_edit`` hands the sum to Claude so the next cut starts from it.

Nothing is stored but what the creator asked to forget. The profile is worked
out again from the edits each time, so it can never drift from them, and it
never leaves this Mac. ``taste.json`` in the projects folder lists what to
leave out: the changes that existed when the creator forgot everything (by
id, so a change made later counts again), and any words and kinds of cut they
asked to forget. Forgetting never touches an edit.

``taste.json``::

    {"version": 2,
     "forgotten": {"<project folder>": {"keep": [ids], "creator_cuts": [ids], "ratings": [ids],
                                        "treatment": <treatment dict or null>}},
     "ignored_words": ["so"],
     "ignored_kinds": ["repeat"]}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from lumr_studio import edit as edits
from lumr_studio.pace import PACES
from lumr_studio.errors import StudioError
from lumr_studio.project import projects_root, write_json_atomic
from lumr_studio.transcript import plain_text

log = logging.getLogger(__name__)

TASTE_FILE = "taste.json"
EDIT_FILE = "edit.json"
TASTE_VERSION = 2
# The longest lists the profile carries.
TOP = 10
# A lesson needs this many videos, or this many events in fewer.
LESSON_VIDEOS, LESSON_EVENTS = 2, 3
# Most lessons of one sort, so a long history doesn't bury the ones that matter.
LESSONS_PER_SORT = 3
# A put-back cut whose kind can't be found from Claude's own cuts.
UNKNOWN_KIND = "unknown"
# The kinds a rating can carry: Claude's cuts, and the filler words it picks.
RATED_KINDS = (*edits.CUT_KINDS, edits.PICK_KIND)
# How each kind of Claude's cuts reads in a lesson.
KIND_WORDS = {"repeat": "restated-point", "false_start": "false-start", "off_topic": "off-topic", "other": "other"}


def taste_path() -> Path:
    """What the creator asked to forget: one file in the projects folder, for every video."""
    return projects_root() / TASTE_FILE


def _strings(value: Any) -> list[str]:
    return [v for v in value if isinstance(v, str)] if isinstance(value, list) else []


def _marker() -> dict[str, Any]:
    """What to leave out, read from ``taste.json``. A file that is missing or can't be read forgets nothing.

    The profile only picks a starting point, so a damaged marker must never
    stop it. Whatever part of the file is readable is used.
    """
    try:
        data = json.loads(taste_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = None
    data = data if isinstance(data, dict) else {}
    forgotten = data.get("forgotten")
    return {
        "forgotten": {k: v for k, v in forgotten.items() if isinstance(v, dict)} if isinstance(forgotten, dict) else {},
        "ignored_words": _strings(data.get("ignored_words")),
        "ignored_kinds": _strings(data.get("ignored_kinds")),
    }


def _write_marker(marker: dict[str, Any]) -> None:
    write_json_atomic(taste_path(), {"version": TASTE_VERSION, **marker})


def _ids(entries: Any) -> list[str]:
    """The saved ids of a list of entries, in order."""
    return [e["id"] for e in entries if isinstance(e, dict) and isinstance(e.get("id"), str)] if isinstance(entries, list) else []


def _snapshot(edit: dict[str, Any]) -> dict[str, Any]:
    """The ids of the creator's changes in ``edit`` right now, and its treatment."""
    from lumr_studio.treatment import keep_id

    keeps = [keep_id(k) for k in edit.get("keep", []) if isinstance(k, dict) and "start" in k and "end" in k] \
        if isinstance(edit.get("keep"), list) else []
    cuts = [edits.creator_cut_id(c) for c in edit.get("creator_cuts", []) if isinstance(c, dict) and "start" in c and "end" in c] \
        if isinstance(edit.get("creator_cuts"), list) else []
    treatment = edit.get("treatment")
    return {"keep": keeps, "creator_cuts": cuts, "ratings": _ids(edit.get("ratings")),
            "treatment": treatment if isinstance(treatment, dict) else None}


def _union(old: list[str], new: list[str]) -> list[str]:
    return list(dict.fromkeys([*old, *new]))


def forget_all() -> dict[str, int]:
    """Forget everything the creator has changed so far, on every video. Their edits are not touched.

    Each project's current changes are written down by id: kept parts, cuts of
    their own, ratings, and the treatment as it stands. ``learn`` leaves those
    out, and the pace and switches while the treatment is still that one. A
    change made after this counts again. An earlier snapshot is kept, with the
    new ids added. Answers ``{"videos": n}``, how many projects were noted.
    """
    marker = _marker()
    root = projects_root()
    noted = 0
    for folder in sorted(root.iterdir()) if root.is_dir() else []:
        try:
            path = folder / EDIT_FILE
            edit = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
            if not isinstance(edit, dict):
                continue
            now = _snapshot(edit)
        except Exception as err:  # noqa: BLE001 - one broken folder must not stop the rest
            log.debug("Skipping %s when forgetting: %s", folder.name, err)
            continue
        before = marker["forgotten"].get(folder.name, {})
        marker["forgotten"][folder.name] = {
            **{key: _union(_strings(before.get(key)), now[key]) for key in ("keep", "creator_cuts", "ratings")},
            "treatment": now["treatment"] if now["treatment"] is not None else before.get("treatment"),
        }
        noted += 1
    _write_marker(marker)
    return {"videos": noted}


def forget_word(word: str) -> str:
    """Stop learning from one word, on every video. Answers the word as it is kept: lower case, no punctuation."""
    said = _said(word) if isinstance(word, str) else ""
    if not said:
        raise StudioError('Send the word to forget, like "so".')
    marker = _marker()
    marker["ignored_words"] = _union(marker["ignored_words"], [said])
    _write_marker(marker)
    return said


def forgettable_kinds() -> list[str]:
    """The kinds of cut the creator can forget: Claude's ``CUT_KINDS`` and the filler likes Claude picks."""
    return [*edits.CUT_KINDS, edits.PICK_KIND]


def forget_kind(kind: str) -> str:
    """Stop learning from one kind of cut, on every video. Raises StudioError for a kind that is not one."""
    if kind not in forgettable_kinds():
        raise StudioError(f"{kind!r} is not a kind of cut. The kinds are {', '.join(forgettable_kinds())}.")
    marker = _marker()
    marker["ignored_kinds"] = _union(marker["ignored_kinds"], [kind])
    _write_marker(marker)
    return kind


def _left_out(edit: dict[str, Any], gone: dict[str, Any], kinds: list[str]) -> tuple[dict[str, Any], bool]:
    """``edit`` without the changes that were forgotten, and whether its treatment is still the forgotten one."""
    from lumr_studio.treatment import keep_id

    keep, cuts, rated = set(gone.get("keep", [])), set(gone.get("creator_cuts", [])), set(gone.get("ratings", []))
    out = dict(edit)
    if isinstance(edit.get("keep"), list):
        out["keep"] = [k for k in edit["keep"] if not (isinstance(k, dict) and "start" in k and "end" in k and keep_id(k) in keep)]
    if isinstance(edit.get("creator_cuts"), list):
        out["creator_cuts"] = [c for c in edit["creator_cuts"]
                               if not (isinstance(c, dict) and "start" in c and "end" in c and edits.creator_cut_id(c) in cuts)]
    if isinstance(edit.get("ratings"), list):
        out["ratings"] = [r for r in edit["ratings"]
                          if isinstance(r, dict) and r.get("id") not in rated and r.get("kind") not in kinds]
    return out, bool(gone) and edit.get("treatment") == gone.get("treatment")


def _ignoring(changes: dict[str, Any], requested: list[dict[str, Any]], words: list[str], kinds: list[str]) -> dict[str, Any]:
    """``changes`` without the words and kinds the creator asked to forget."""
    out = dict(changes)
    if words and "cut_words" in out:
        out["cut_words"] = [[t, n] for t, n in out["cut_words"] if t not in words]
        if not out["cut_words"]:
            out.pop("cut_words")
            out.pop("cuts", None)
    if words and "brought_back" in out:
        out["brought_back"] = [b for b in out["brought_back"] if _said(b.get("text", "")) not in words]
        if not out["brought_back"]:
            out.pop("brought_back")
    if kinds and "put_back" in out:
        out["put_back"] = [k for k in out["put_back"] if _kind_of(k, requested) not in kinds]
        if not out["put_back"]:
            out.pop("put_back")
    return out


def _changed_projects(exclude: Path | None) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """``(name, edit, creator)`` for each project with something the creator changed, skipping ``exclude``.

    Each edit comes without what the creator asked to forget (see
    ``forget_all``, ``forget_word`` and ``forget_kind``). A project that can't
    be read, or holds an edit that can't be summed up, is skipped: one broken
    folder must never cost the creator the rest.
    """
    # The page and Claude both ask; treatment.page_state asks this module in turn.
    from lumr_studio.treatment import creator_changes

    root = projects_root()
    if not root.is_dir():
        return []
    marker = _marker()
    skip = exclude.resolve() if exclude is not None else None
    found = []
    try:
        folders = sorted(root.iterdir())
    except OSError:
        return []  # an unreadable projects folder costs the profile line, never the page or get_edit
    for folder in folders:
        path = folder / EDIT_FILE
        try:
            if not path.is_file() or (skip is not None and folder.resolve() == skip):
                continue
            edit = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(edit, dict):
                continue
            edit, same = _left_out(edit, marker["forgotten"].get(folder.name, {}), marker["ignored_kinds"])
            changes = creator_changes(edit)
            if changes and same:
                for key in ("pace", "take_out", "fine"):
                    changes.pop(key, None)
            if changes:
                requested = [r for r in edit.get("requested", []) if isinstance(r, dict)]
                changes = _ignoring(changes, requested, marker["ignored_words"], marker["ignored_kinds"])
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


def _marked_wrong(kept: dict[str, Any], wrong: list[dict[str, Any]]) -> bool:
    """Whether the creator rated the cut ``kept`` put back as wrong: a bad rating whose span overlaps it."""
    for rating in wrong:
        try:
            if _span_overlap(kept, rating) > 0:
                return True
        except (KeyError, TypeError, ValueError):
            continue
    return False


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
    changes reach Claude as ``creator`` and are not counted twice. What the
    creator asked to forget is left out (see ``forget_all``, ``forget_word``
    and ``forget_kind``).

    ``videos`` counts the projects with at least one change. ``put_back`` maps
    each kind of Claude's cuts to how many were put back and how many Claude
    proposed in those projects. A cut the creator rated wrong was put back
    too, and counts only as a bad rating here. ``cut_by_hand`` and ``brought_back`` are
    ``[word, count]`` lists, most first. ``ratings`` maps each kind (Claude's
    cut kinds, or ``likes`` for its picked filler words) to how many the
    creator marked ``good`` and ``bad``. ``lessons`` are plain sentences
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
    rated: dict[str, dict[str, int]] = {}
    rated_in: dict[tuple[str, str], set[str]] = {}
    kept_parts = 0
    stops = list(PACES)
    for name, edit, changes in projects:
        requested = [r for r in edit.get("requested", []) if isinstance(r, dict)]
        for r in requested:
            kind = r.get("kind") if r.get("kind") in edits.CUT_KINDS else edits.DEFAULT_CUT_KIND
            put_back.setdefault(kind, {"put_back": 0, "proposed": 0})["proposed"] += 1
        marked_wrong = changes.get("rated", {}).get("bad", [])
        for k in changes.get("put_back", []):
            if _marked_wrong(k, marked_wrong):
                continue  # a "Wrong cut" also puts the cut back; it counts once, as a bad rating
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
        for side, entries in changes.get("rated", {}).items():
            for entry in entries:
                kind = entry.get("kind") if entry.get("kind") in RATED_KINDS else edits.DEFAULT_CUT_KIND
                rated.setdefault(kind, {"good": 0, "bad": 0})[side] += 1
                rated_in.setdefault((kind, side), set()).add(name)
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
        "ratings": rated,
        "pace": {k: v for k, v in pace.items() if v},
        "take_out": took,
    }
    lessons = _lessons_put_back(put_back, put_back_in)
    lessons += _lessons_cut_by_hand(cut_words, cut_in)
    lessons += _lessons_brought_back(brought, brought_in)
    lessons += _lessons_rated(rated, rated_in)
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


def _lessons_rated(tally: dict[str, dict[str, int]], where: dict[tuple[str, str], set[str]]) -> list[str]:
    """What thumbs up and down teach, kind by kind, the most marked first."""
    said = {"good": "good", "bad": "wrong"}
    found = []
    for kind in RATED_KINDS:
        for side in ("good", "bad"):
            count = tally.get(kind, {}).get(side, 0)
            if count and _backed(count, len(where[(kind, side)])):
                what = "filler-word picks" if kind == edits.PICK_KIND else f"{KIND_WORDS[kind]} cuts ({kind})"
                found.append((count, f"Marked {count} {what} {said[side]} across {_videos(len(where[(kind, side)]))}."))
    return [line for _count, line in sorted(found, key=lambda item: -item[0])[:LESSONS_PER_SORT]]


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
