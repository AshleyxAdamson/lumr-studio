"""The creator's decision queue: one row per saved cut, and her decisions on them.

The round 2 review page showed every cut in the saved edit as a row the
creator could accept, restore, or re-edge. The treatment page replaced it and
the page file is gone, but its routes (``api/queue``, ``api/decisions``,
``api/apply``) still answer, ``get_edit`` still reports decisions saved from
it, and the treatment page borrows ``FLAG_LABELS`` and ``page_words``. This
module owns the data behind them:

* ``build_queue`` turns the saved edit, the transcript and the join self-check
  rows into the page's rows, sections and counts. Pure.
* ``load_review`` / ``save_decisions`` keep her decisions in
  ``<project.root>/review.json`` so a reload brings back the same states.
* ``apply_review`` folds the decisions into the edit: restored cuts go, edited
  cuts get their new edges placed word-safe, and every restored span is
  recorded in ``edit["keep"]`` so later edits know she wants it kept. Pure.
* ``queue_for_project`` and ``apply_to_project`` are the two project-level
  entry points the server and the tools call.

Row ids come from the cut's edges rounded to hundredths (``c412.30-431.80``),
so they survive a reload and a re-apply for every cut whose edges did not move.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lumr_studio.edit import (
    CutRejected,
    edit_summary,
    load_edit,
    place_cut,
    removed_spans,
)
from lumr_studio.engine.audio_boundaries import Silence
from lumr_studio.errors import StudioError
from lumr_studio.project import Project, append_receipt, now_iso, write_json_atomic
from lumr_studio.transcript import is_event
from lumr_studio.word_times import load_words

REVIEW_VERSION = 1
REVIEW_FILE = "review.json"

# Decision states. "edited" means the creator moved an edge and has not yet
# accepted the new edges, so it still counts as open (design fix #4: nudging an
# accepted cut reopens it and the done count drops).
OPEN, ACCEPTED, RESTORED, EDITED = "open", "accepted", "restored", "edited"
STATES = (OPEN, ACCEPTED, RESTORED, EDITED)
DECIDED_STATES = (ACCEPTED, RESTORED)
DECISION_KEYS = {"state", "start", "end"}

# Sections, in the order the page shows them.
SECTION_FLAGGED, SECTION_CLAUDE, SECTION_AUTO = "flagged", "claude", "auto"
SECTION_LABELS = {
    SECTION_FLAGGED: "Needs a look",
    SECTION_CLAUDE: "Claude's cuts",
    SECTION_AUTO: "Automatic trims",
}

# An audition plays this much kept material either side of the cut: the
# design's "seek 2s before, jump the cut, play 2s more".
AUDITION_LEAD = 2.0
AUDITION_TAIL = 2.0

# Plain words for the self-check flags (Lane B's names are part of its
# contract; the creator never sees them raw).
FLAG_LABELS = {
    "mid_sentence_out": "sentence left unfinished",
    "mid_sentence_in": "picks up mid-sentence",
    "splice": "joins two half sentences",
    "removes_laugh": "removes a laugh",
    "clips_beat": "shortens the pause by a laugh",
    "re_entry": "next line points back at cut words",
    "long_jump": "takes out 20 seconds or more",
    "fragment": "leaves a tiny piece",
    "tight": "no breath at the join",
}

# The automatic-trim reason classes a scoped bulk accept may close, named the
# way the creator reads them. Built by build_edit as "auto: pause: 0.8s".
AUTO_CLASS_LABELS = {"pause": "pause trims", "filler": "filler trims", "stutter": "stutter trims"}
CONTENT_CLASS = "content"

# A POSTed decisions map larger than this many rows is not from the page.
MAX_DECISIONS = 5000

JoinRows = Callable[..., list[dict[str, Any]]]
LabelsFor = Callable[[Project, list[dict[str, Any]]], list[dict[str, Any]]]


# ── Row ids and small helpers ────────────────────────────────────────────────


def row_id(start: float, end: float) -> str:
    """The stable id of the cut ``[start, end]``: its edges rounded to hundredths."""
    return f"c{float(start):.2f}-{float(end):.2f}"


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _union(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sorted, merged copy of ``spans``."""
    out: list[tuple[float, float]] = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def reason_class(cut: dict[str, Any]) -> str:
    """``pause``, ``filler`` or ``stutter`` for an automatic trim; ``content`` otherwise."""
    if cut.get("source") != "auto":
        return CONTENT_CLASS
    head = str(cut.get("reason", "")).split(";")[0]
    head = head.removeprefix("auto:").strip()
    label = head.split(":")[0].strip().lower()
    return label if label in AUTO_CLASS_LABELS else "other"


def flag_label(flag: str) -> str:
    """The plain words for one self-check flag."""
    return FLAG_LABELS.get(flag, flag.replace("_", " "))


def fmt_time(t: float) -> str:
    """``m:ss.s``, the one time format the creator sees."""
    tenths = round(max(0.0, float(t)) * 10)
    minutes, rest = divmod(tenths, 600)
    return f"{minutes}:{rest / 10:04.1f}"


# ── The queue ────────────────────────────────────────────────────────────────


def _match_join_rows(cuts: list[dict[str, Any]], join_rows: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
    """The join row for each cut: matched on rounded edges, else on cut index."""
    by_id = {row_id(r["start"], r["end"]): r for r in join_rows if _finite(r.get("start")) and _finite(r.get("end"))}
    by_index = {r["cut"]: r for r in join_rows if isinstance(r.get("cut"), int)}
    out: list[dict[str, Any] | None] = []
    for i, cut in enumerate(cuts):
        out.append(by_id.get(row_id(cut["start"], cut["end"])) or by_index.get(i))
    return out


def _removed_word_count(start: float, end: float, words: list[dict[str, Any]]) -> int:
    """Spoken words whose midpoint lies inside ``[start, end]`` (the edit's midpoint rule)."""
    return sum(
        1 for w in words
        if not is_event(w) and start < (float(w["start"]) + float(w["end"])) / 2 < end
    )


def page_words(words: list[dict[str, Any]], labels: list[dict[str, Any]] | None = None) -> list[list[Any]]:
    """The transcript in the page's compact form: ``[start, end, text, is_sound]``.

    Sounds read as plain words: "(laugh)", "(laugh?)" for a possible laugh, or
    "(sound)", using the sound labels when given.
    """
    label_at = {round(float(lb["start"]), 2): lb for lb in (labels or [])}
    out: list[list[Any]] = []
    for w in words:
        s, e = round(float(w["start"]), 3), round(float(w["end"]), 3)
        if is_event(w):
            lb = label_at.get(round(s, 2))
            text = "(sound)"
            if lb and lb.get("kind") == "laugh":
                text = "(laugh)" if lb.get("confidence") == "likely" else "(laugh?)"
            out.append([s, e, text, 1])
        else:
            out.append([s, e, str(w["word"]).strip(), 0])
    return out


def _edges(cut: dict[str, Any], decision: dict[str, Any] | None) -> tuple[float, float]:
    if decision and _finite(decision.get("start")) and _finite(decision.get("end")):
        return float(decision["start"]), float(decision["end"])
    return float(cut["start"]), float(cut["end"])


def audition_seconds(start: float, end: float, duration: float) -> float:
    """How long one audition of the cut ``[start, end]`` plays."""
    return min(AUDITION_LEAD, max(0.0, start)) + min(AUDITION_TAIL, max(0.0, duration - end))


def edited_seconds(cuts: list[dict[str, Any]], decisions: dict[str, dict[str, Any]], duration: float) -> float:
    """Length of the edit if the decisions were applied now (restored cuts gone, edited edges used)."""
    spans = []
    for cut in cuts:
        d = decisions.get(row_id(cut["start"], cut["end"]))
        if d and d.get("state") == RESTORED:
            continue
        s, e = _edges(cut, d)
        spans.append((max(0.0, s), min(duration, e)))
    return duration - sum(e - s for s, e in _union(spans))


def build_queue(
    edit: dict[str, Any],
    words: list[dict[str, Any]],
    duration: float,
    *,
    join_rows: list[dict[str, Any]],
    labels: list[dict[str, Any]] | None = None,
    review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything the review page needs, from the saved edit and the creator's decisions.

    ``join_rows`` is the join self-check's output for this edit (one row per
    cut); its flags decide which rows go in the "Needs a look" section.
    ``review`` is ``load_review`` output. Rows are grouped flagged, then
    Claude's cuts, then automatic trims, each in time order.

    Returns ``{video, rows, sections, bulk, counts, durations, audition, words, keep}``.
    """
    cuts = sorted(edit.get("cuts", []), key=lambda c: float(c["start"]))
    decisions = (review or {}).get("decisions", {})
    matched = _match_join_rows(cuts, join_rows)

    rows: list[dict[str, Any]] = []
    for i, (cut, jr) in enumerate(zip(cuts, matched)):
        rid = row_id(cut["start"], cut["end"])
        d = decisions.get(rid)
        flags = [f for f in (jr or {}).get("flags", []) if isinstance(f, str)]
        source = "claude" if cut.get("source") == "claude" else "auto"
        section = SECTION_FLAGGED if flags else (SECTION_CLAUDE if source == "claude" else SECTION_AUTO)
        start, end = float(cut["start"]), float(cut["end"])
        edges = _edges(cut, d)
        rows.append({
            "id": rid,
            "index": i,
            "start": start,
            "end": end,
            "seconds": round(end - start, 3),
            "source": source,
            "reason": str(cut.get("reason", "")).strip(),
            "auto_trims": int(cut.get("auto_trims", 0) or 0),
            "cls": reason_class(cut),
            "section": section,
            "flags": flags,
            "flag_labels": [flag_label(f) for f in flags],
            "note": str((jr or {}).get("note", "") or ""),
            "removed_words": _removed_word_count(start, end, words),
            "state": d["state"] if d else OPEN,
            "edges": list(edges) if d and "start" in d else None,
        })

    order = {SECTION_FLAGGED: 0, SECTION_CLAUDE: 1, SECTION_AUTO: 2}
    rows.sort(key=lambda r: (order[r["section"]], r["start"]))

    open_rows = [r for r in rows if r["state"] not in DECIDED_STATES]
    listen_left = sum(audition_seconds(*(r["edges"] or (r["start"], r["end"])), duration) for r in open_rows)
    return {
        "video": {"name": Path(str(edit.get("video", ""))).stem, "duration": round(duration, 3)},
        "rows": rows,
        "sections": [
            {"key": key, "label": SECTION_LABELS[key], "count": sum(1 for r in rows if r["section"] == key)}
            for key in (SECTION_FLAGGED, SECTION_CLAUDE, SECTION_AUTO)
        ],
        "bulk": bulk_sets(rows),
        "counts": {
            "rows": len(rows),
            "decided": len(rows) - len(open_rows),
            "by_state": {s: sum(1 for r in rows if r["state"] == s) for s in STATES},
        },
        "durations": {
            "full": round(duration, 3),
            "saved_edit": round(duration - sum(e - s for s, e in _union(removed_spans(edit))), 3),
            "with_decisions": round(edited_seconds(cuts, decisions, duration), 3),
            "listen_left": round(listen_left, 1),
        },
        "audition": {"lead": AUDITION_LEAD, "tail": AUDITION_TAIL},
        "words": page_words(words, labels),
        "keep": edit.get("keep", []),
        "updated_at": edit.get("updated_at"),
    }


def bulk_sets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The scoped bulk accepts: one per automatic-trim class with open, unflagged rows.

    Each set states its class, label and exact row ids. Claude's cuts and
    flagged rows never join a bulk set, and there is no "accept everything".
    """
    sets: dict[str, list[str]] = {}
    for r in rows:
        if r["section"] == SECTION_AUTO and r["state"] == OPEN and r["cls"] in AUTO_CLASS_LABELS:
            sets.setdefault(r["cls"], []).append(r["id"])
    return [
        {"cls": cls, "label": AUTO_CLASS_LABELS[cls], "ids": ids}
        for cls, ids in sorted(sets.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    ]


# ── Decisions on disk ────────────────────────────────────────────────────────


def review_path(project: Project) -> Path:
    return project.root / REVIEW_FILE


def load_review(project: Project) -> dict[str, Any]:
    """The saved review, or ``{}`` when the creator has not decided anything yet.

    Raises StudioError when review.json exists but cannot be read.
    """
    path = review_path(project)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise StudioError(
            f"The saved review at {path} is not valid JSON. Delete it to start the review over."
        ) from None
    if not isinstance(data, dict) or not isinstance(data.get("decisions"), dict):
        raise StudioError(f"The saved review at {path} has no decisions map. Delete it to start the review over.")
    return data


def validate_decisions(
    raw: Any,
    cuts_by_id: dict[str, dict[str, Any]],
    duration: float,
    words: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Check a decisions map from the page and return it normalized.

    Every key must be the id of a cut in the saved edit, every state one of
    ``STATES``, and any edges finite, inside the video, start before end. With
    ``words``, new edges must also place word-safe. Open decisions without
    edges are dropped (open is the default). Raises StudioError naming the
    first bad row and how to fix it.
    """
    if not isinstance(raw, dict):
        raise StudioError("decisions must be an object mapping row ids to {state, start, end}.")
    if len(raw) > MAX_DECISIONS:
        raise StudioError(f"decisions has {len(raw)} rows; the edit has {len(cuts_by_id)}. Send one per row.")
    tokens = sorted(words, key=lambda w: float(w["start"])) if words is not None else None
    out: dict[str, dict[str, Any]] = {}
    for rid, d in raw.items():
        if rid not in cuts_by_id:
            raise StudioError(
                f"Row {rid!r} is not a cut in the saved edit. The edit may have changed since the page "
                "loaded; reload the page."
            )
        if not isinstance(d, dict):
            raise StudioError(f"Row {rid}: the decision must be an object like {{\"state\": \"accepted\"}}.")
        extra = set(d) - DECISION_KEYS
        if extra:
            raise StudioError(f"Row {rid}: unknown field {sorted(extra)[0]!r}. Send only state, start and end.")
        state = d.get("state")
        if state not in STATES:
            raise StudioError(f"Row {rid}: state {state!r} is not one of {', '.join(STATES)}.")
        entry: dict[str, Any] = {"state": state}
        has_start, has_end = "start" in d, "end" in d
        if has_start != has_end:
            raise StudioError(f"Row {rid}: send both start and end, or neither.")
        if state == EDITED and not has_start:
            raise StudioError(f"Row {rid}: an edited cut needs its new start and end.")
        if has_start:
            s, e = d["start"], d["end"]
            if not _finite(s) or not _finite(e):
                raise StudioError(f"Row {rid}: start and end must be numbers of seconds.")
            if s < 0 or e > duration:
                raise StudioError(f"Row {rid}: {s}-{e} runs outside the video (0 to {duration:.2f}).")
            if e <= s:
                raise StudioError(f"Row {rid}: start {s} is not before end {e}.")
            if tokens is not None and state != RESTORED:
                try:
                    place_cut(float(s), float(e), tokens, duration)
                except CutRejected as why:
                    raise StudioError(f"Row {rid}: the new edges {s:.2f}-{e:.2f} {why}") from None
            entry["start"], entry["end"] = round(float(s), 3), round(float(e), 3)
        if state == OPEN and not has_start:
            continue
        out[rid] = entry
    return out


def save_decisions(
    project: Project,
    decisions: Any,
    *,
    edit: dict[str, Any] | None = None,
    words: list[dict[str, Any]] | None = None,
    duration: float | None = None,
) -> dict[str, Any]:
    """Validate the page's full decisions map and write it to review.json.

    The map replaces what was saved. ``edit``, ``words`` and ``duration`` are
    loaded from the project when not given. Returns ``{saved, decided, rows,
    with_decisions}``. Raises StudioError for a bad decision; nothing is
    written then.
    """
    duration = project.duration() if duration is None else duration
    edit = load_edit(project, duration) if edit is None else edit
    words = load_words(project) if words is None else words
    cuts_by_id = {row_id(c["start"], c["end"]): c for c in edit.get("cuts", [])}
    clean = validate_decisions(decisions, cuts_by_id, duration, words)
    write_json_atomic(review_path(project), {"version": REVIEW_VERSION, "updated_at": now_iso(), "decisions": clean})
    return {
        "saved": len(clean),
        "decided": sum(1 for d in clean.values() if d["state"] in DECIDED_STATES),
        "rows": len(cuts_by_id),
        "with_decisions": round(edited_seconds(list(cuts_by_id.values()), clean, duration), 3),
    }


# ── Applying the review to the edit ──────────────────────────────────────────


def _keep_note(cut: dict[str, Any]) -> str:
    reason = str(cut.get("reason", "")).strip()
    return f"Restored by the creator in review. Claude's reason was: {reason}" if reason else "Restored by the creator in review."


def _merge_cuts(cuts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort and merge overlapping cuts; a merged cut is Claude's when any part was.

    Cuts that only touch stay apart, so their row ids (and the creator's
    decisions on them) survive.
    """
    out: list[dict[str, Any]] = []
    for cut in sorted(cuts, key=lambda c: c["start"]):
        if out and cut["start"] < out[-1]["end"]:
            last = out[-1]
            last["end"] = max(last["end"], cut["end"])
            reasons = [r for r in (last.get("reason"), cut.get("reason")) if r]
            last["reason"] = "; ".join(dict.fromkeys(reasons))
            if cut.get("source") == "claude":
                last["source"] = "claude"
            if cut.get("edited_in_review"):
                last["edited_in_review"] = True
        else:
            out.append(dict(cut))
    return out


def _apply(
    edit: dict[str, Any],
    review: dict[str, Any],
    words: list[dict[str, Any]],
    duration: float,
    silences: list[Silence] | None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, int]]:
    """``apply_review`` plus the decisions carried over to the new row ids, and counts."""
    tokens = sorted(words, key=lambda w: float(w["start"]))
    decisions = review.get("decisions", {}) if review else {}
    kept_cuts: list[dict[str, Any]] = []
    carried_from: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    keep = [dict(k) for k in edit.get("keep", [])]
    counts = {"restored": 0, "edited": 0}

    for cut in edit.get("cuts", []):
        d = decisions.get(row_id(cut["start"], cut["end"]))
        state = d["state"] if d else OPEN
        if state == RESTORED:
            counts["restored"] += 1
            span = {"start": round(float(cut["start"]), 3), "end": round(float(cut["end"]), 3), "note": _keep_note(cut)}
            if not any(k.get("start") == span["start"] and k.get("end") == span["end"] for k in keep):
                keep.append(span)
            continue
        new = dict(cut)
        if d and "start" in d:
            try:
                s, e = place_cut(float(d["start"]), float(d["end"]), tokens, duration, silences)
            except CutRejected as why:
                raise StudioError(
                    f"The edited cut {row_id(cut['start'], cut['end'])} can't be placed: its new edges "
                    f"{d['start']:.2f}-{d['end']:.2f} {why}"
                ) from None
            new["start"], new["end"] = round(s, 3), round(e, 3)
            new["edited_in_review"] = True
            counts["edited"] += 1
        kept_cuts.append(new)
        carried_from.append((new, d))

    merged = _merge_cuts(kept_cuts)
    new_edit = {**edit, "cuts": merged, "keep": sorted(keep, key=lambda k: k["start"]), "updated_at": now_iso()}

    # Carry each surviving decision to the id of the cut it ended up in. A
    # merged cut stays open unless every part of it was decided.
    carried: dict[str, dict[str, Any]] = {}
    for m in merged:
        parts = [d for c, d in carried_from if m["start"] <= c["start"] and c["end"] <= m["end"]]
        states = [(d or {}).get("state", OPEN) for d in parts]
        if parts and all(s == ACCEPTED for s in states):
            carried[row_id(m["start"], m["end"])] = {"state": ACCEPTED}
        elif any(s == EDITED for s in states):
            carried[row_id(m["start"], m["end"])] = {"state": EDITED, "start": m["start"], "end": m["end"]}
    return new_edit, carried, counts


def apply_review(
    edit: dict[str, Any],
    review: dict[str, Any],
    words: list[dict[str, Any]],
    duration: float,
    *,
    silences: list[Silence] | None = None,
) -> dict[str, Any]:
    """The edit with the creator's decisions applied.

    Restored cuts are dropped. Edited cuts get their new edges, placed
    word-safe with ``edit.place_cut``. Accepted and open cuts stay.
    Every restored span is added to ``edit["keep"]`` (a list of
    {start, end, note}) so later edits know the creator wants it kept.
    Returns the new edit dict; the caller saves it. Raises StudioError when
    an edited cut can no longer be placed word-safe.
    """
    return _apply(edit, review, words, duration, silences)[0]


# ── Project-level entry points ───────────────────────────────────────────────


def queue_for_project(
    project: Project,
    *,
    join_rows: JoinRows,
    labels_for: LabelsFor | None = None,
    duration: float | None = None,
) -> dict[str, Any]:
    """The live queue for ``project``: saved edit, transcript, sound labels and decisions.

    ``join_rows`` is the join self-check function, called as
    ``join_rows(cuts, words, duration, labels=labels)``. ``labels_for`` gives
    the sound labels for the project, or None to go without.
    """
    duration = project.duration() if duration is None else duration
    edit = load_edit(project, duration)
    words = load_words(project)
    labels = labels_for(project, words) if labels_for else None
    rows = join_rows(edit.get("cuts", []), words, duration, labels=labels)
    queue = build_queue(edit, words, duration, join_rows=rows, labels=labels, review=load_review(project))
    if not queue["video"]["name"]:
        queue["video"]["name"] = project.video.stem
    return queue


def apply_to_project(
    project: Project,
    *,
    silences: list[Silence] | None = None,
    duration: float | None = None,
) -> dict[str, Any]:
    """Apply the saved decisions to the saved edit and write both back.

    edit.json gets the new cuts and keep list; review.json keeps the
    decisions of the cuts that survived, under their new ids; a receipt
    records the step. Returns ``{restored, edited, summary}`` where
    ``summary`` is ``edit.edit_summary`` of the new edit.
    """
    duration = project.duration() if duration is None else duration
    edit = load_edit(project, duration)
    review = load_review(project)
    words = load_words(project)
    before = edit_summary(edit, duration)
    new_edit, carried, counts = _apply(edit, review, words, duration, silences)
    write_json_atomic(project.edit_path, new_edit)
    write_json_atomic(review_path(project), {"version": REVIEW_VERSION, "updated_at": now_iso(), "decisions": carried})
    summary = edit_summary(new_edit, duration)
    append_receipt(
        project, "review_apply", restored=counts["restored"], edited=counts["edited"],
        cut_count=summary["cut_count"], new_duration=summary["new_duration"],
    )
    return {
        "restored": counts["restored"],
        "edited": counts["edited"],
        "previous_duration": before["new_duration"],
        "new_duration": summary["new_duration"],
        "summary": summary,
    }
