"""The shape of what the creator changed, sent to the team only when they press Send.

A send is a list of records, one for each thing the creator did to Claude's
edit: put a cut back, rate one, bring words back, cut some by hand, keep a
part, or leave a cut in place. A record holds numbers and names from a fixed
list (how long the cut was, how many words, what kind, what the words either
side of it sound like, whether a laugh was near). It never holds a word from
the video, and never the video. ``validate_send`` checks that: it accepts
exactly what the team's server accepts, and ``send`` runs it before every
request. The record format is written down once, in the schema the server and
this module share (``SCHEMA``, and the sets below, which are copied from it).

The page never calls the internet. It asks this module for a preview
(``build_send`` and ``describe``), and when the creator presses Send, this
module makes the one request (``send``): 15 seconds at most, no retry, and a
line in ``<project>/shares.jsonl`` with the send's ID so they can ask for it
to be deleted.
"""

from __future__ import annotations

import http.client
import bisect
import json
import logging
import math
import re
import secrets
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lumr_studio import edit as edits
from lumr_studio import feedback, share, taste, treatment, word_times
from lumr_studio.errors import StudioError
from lumr_studio.pace import PACES
from lumr_studio.project import Project
from lumr_studio.sounds import LAUGH, SOUNDS_FILE
from lumr_studio.transcript import _is_sentence_end, is_event, plain_text

log = logging.getLogger(__name__)

SCHEMA = 1
SENDS_PATH = "/v1/sends"
SHARES_FILE = "shares.jsonl"
TIMEOUT_SECONDS = 15
UNKNOWN_VERSION = "0.0.0"

# ── The schema: nothing else is allowed ───────────────────────────────────────

PACE = frozenset({"natural", "standard", "fast", "tight", "hard", "max", "custom"})
SOURCE = frozenset({"claude", "pick", "auto", "creator"})
KIND = frozenset({"pause", "filler", "stutter", "repeat", "false_start", "off_topic", "likes", "other"})
POS = frozenset({"noun", "verb", "adj", "adv", "pron", "det", "prep", "conj", "num", "interj", "filler", "other"})
PITCH = frozenset({"rising", "falling", "flat", "unknown"})
SENTENCE_POSITION = frozenset({"start", "middle", "end", "whole"})
ACTION = frozenset({"kept", "put_back", "rated_good", "rated_bad", "brought_back_words", "cut_by_hand", "kept_part"})
FLAG = frozenset({"mid_sentence_out", "mid_sentence_in", "splice", "removes_laugh", "clips_beat", "re_entry",
                  "long_jump", "fragment", "tight"})
FILLER = frozenset({"um", "uh", "er", "ah", "hmm", "like", "so", "you know", "i mean", "basically", "literally",
                    "right", "actually", "well"})

MAX_RECORDS = 500
MAX_BODY_BYTES = 256 * 1024
MAX_CONTEXT_WORDS = 3
MAX_WORD_SECONDS, MAX_GAP_SECONDS = 10, 60
MAX_LENGTH_SECONDS, MAX_WORDS, MAX_LAUGH_SECONDS = 3600, 10000, 600
SEND_ID = re.compile(r"[A-Za-z0-9_-]{16,32}")
VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")

# ── What the creator's changes are called in a record ─────────────────────────

# Inside Lumr, a kind of cut or trim, and the name it has in a record.
SENDABLE_KIND = {
    "pauses": "pause", "fillers": "filler", "repeats": "stutter",
    **{k: k for k in ("repeat", "false_start", "off_topic", "other", edits.PICK_KIND)},
}
# A cut this close to a laugh, or closer, is "near a laugh" in the preview.
NEAR_A_LAUGH_SECONDS = 3
# The words the record says a cut of each kind is, for the preview.
KIND_NOUN = {
    "pause": "long-pause cut", "filler": "filler-word cut", "stutter": "stutter cut", "repeat": "restated-point cut",
    "false_start": "false-start cut", "off_topic": "off-topic cut", "likes": "filler-word pick", "other": "cut",
}

# ── Parts of speech, from a small closed list. Nothing here reads a sentence. ─

_PRONOUNS = frozenset(
    "i me my mine myself you your yours yourself he him his himself she her hers herself it its itself we us our ours "
    "ourselves they them their theirs themselves who whom whose what which someone anyone everyone nobody "
    "something anything everything nothing".split())
_DETERMINERS = frozenset("a an the this that these those some any each every another both all many few several most".split())
_PREPOSITIONS = frozenset(
    "in on at to of for with from by about into over under after before between through during without within around "
    "against among across toward towards upon onto off out up down near behind beside beyond above below along "
    "inside outside until per via".split())
_CONJUNCTIONS = frozenset(
    "and but or nor yet because although though while if unless since than whereas whether once".split())
_INTERJECTIONS = frozenset(
    "oh wow yeah yep yes nope no hey hi hello okay ok oops ouch whoa huh aha yay ugh bye alright".split())
_NUMBER_WORDS = frozenset(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
    "eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion first "
    "second third".split())
_TWO_WORD_FILLERS = frozenset(f for f in FILLER if " " in f)


def part_of_speech(said: str) -> str:
    """The part of speech of one word as said (``plain_text``), from the closed lists above, else ``other``.

    Fillers come first, so a "like" or "so" reads as ``filler`` wherever it stands.
    """
    if said in FILLER:
        return "filler"
    if said in _PRONOUNS:
        return "pron"
    if said in _DETERMINERS:
        return "det"
    if said in _PREPOSITIONS:
        return "prep"
    if said in _CONJUNCTIONS:
        return "conj"
    if said in _NUMBER_WORDS or any(c.isdigit() for c in said):
        return "num"
    if said in _INTERJECTIONS:
        return "interj"
    return "other"


# ── The check: a mirror of the team's server ──────────────────────────────────


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _two_decimals(value: float) -> bool:
    return abs(value - math.floor(value * 100 + 0.5) / 100) < 1e-9


def _choice(value: Any, allowed: frozenset[str], label: str) -> str | None:
    if not isinstance(value, str):
        return f"{label} must be string"
    if value not in allowed:
        return f"{label} invalid: {value}"
    return None


def _number(value: Any, low: float, high: float, label: str, *, whole: bool = False) -> str | None:
    if not _is_number(value):
        return f"{label} must be number"
    if value < low or value > high:
        return f"{label} out of range"
    if whole:
        return None if value == math.floor(value) else f"{label} must be integer"
    return None if _two_decimals(value) else f"{label} must be rounded to 2 decimals"


def _only(value: dict[str, Any], keys: set[str], label: str) -> str | None:
    for key in value:
        if key not in keys:
            return f"{label}unknown key: {key}"
    return None


def _word_shape(shape: Any) -> str | None:
    if not isinstance(shape, dict):
        return "must be object"
    return (
        _only(shape, {"pos", "dur", "gap_after", "pitch", "filler"}, "")
        or _choice(shape.get("pos"), POS, "pos")
        or _number(shape.get("dur"), 0, MAX_WORD_SECONDS, "dur")
        or _number(shape.get("gap_after"), 0, MAX_GAP_SECONDS, "gap_after")
        or _choice(shape.get("pitch"), PITCH, "pitch")
        or ("filler must be string or null" if "filler" not in shape
            else None if shape["filler"] is None else _choice(shape["filler"], FILLER, "filler"))
    )


def _words_of(value: Any, label: str) -> str | None:
    if not isinstance(value, list):
        return f"{label} must be array"
    if len(value) > MAX_CONTEXT_WORDS:
        return f"{label} exceeds {MAX_CONTEXT_WORDS}"
    for i, shape in enumerate(value):
        if (why := _word_shape(shape)):
            return f"{label}[{i}]: {why}"
    return None


def _proposed(p: Any) -> str | None:
    if not isinstance(p, dict):
        return "proposed must be object"
    return (
        _only(p, {"source", "kind", "length_s", "words"}, "")
        or _choice(p.get("source"), SOURCE, "source")
        or _choice(p.get("kind"), KIND, "kind")
        or _number(p.get("length_s"), 0, MAX_LENGTH_SECONDS, "length_s")
        or _number(p.get("words"), 0, MAX_WORDS, "words", whole=True)
    )


def _context(c: Any) -> str | None:
    if not isinstance(c, dict):
        return "context must be object"
    if (why := _only(c, {"before", "after", "sentence_position", "laugh_within_s"}, "")):
        return why
    if (why := _words_of(c.get("before"), "before") or _words_of(c.get("after"), "after")):
        return why
    if (why := _choice(c.get("sentence_position"), SENTENCE_POSITION, "sentence_position")):
        return why
    if "laugh_within_s" not in c:
        return "laugh_within_s must be number or null"
    laugh = c["laugh_within_s"]
    return None if laugh is None else _number(laugh, 0, MAX_LAUGH_SECONDS, "laugh_within_s")


def _join(j: Any) -> str | None:
    if not isinstance(j, dict):
        return "join must be object"
    if (why := _only(j, {"gap_left_s", "flags"}, "")):
        return why
    if "gap_left_s" not in j:
        return "gap_left_s must be number or null"
    gap = j["gap_left_s"]
    if gap is not None and (why := _number(gap, 0, MAX_GAP_SECONDS, "gap_left_s")):
        return why
    flags = j.get("flags")
    if not isinstance(flags, list):
        return "flags must be array"
    for i, flag in enumerate(flags):
        if (why := _choice(flag, FLAG, f"flags[{i}]")):
            return why
    return None


def _record(rec: Any) -> str | None:
    if not isinstance(rec, dict):
        return "Record must be object"
    if (why := _only(rec, {"pace", "proposed", "context", "creator", "join"}, "")):
        return why
    if (why := _choice(rec.get("pace"), PACE, "pace")):
        return why
    if (why := _proposed(rec.get("proposed"))):
        return f"proposed: {why}"
    if (why := _context(rec.get("context"))):
        return f"context: {why}"
    creator = rec.get("creator")
    if not isinstance(creator, dict):
        return "creator must be object"
    if (why := _choice(creator.get("action"), ACTION, "creator.action")):
        return why
    if (why := _only(creator, {"action"}, "creator ")):
        return why
    if (why := _join(rec.get("join"))):
        return f"join: {why}"
    return None


def validate_send(send: Any) -> str | None:
    """Why ``send`` would be turned away, in a short phrase, or None when it is a good send.

    The rules are the ones the team's server applies: the keys, the lists of
    allowed names, the number ranges, two decimals at most, up to 500
    records, and a body of at most 256 KiB. Anything else, a word from the
    video included, has no place to go.
    """
    if not isinstance(send, dict):
        return "Expected an object"
    if not _is_number(send.get("schema")) or send.get("schema") != SCHEMA:
        return f"schema must be {SCHEMA}"
    send_id = send.get("send_id")
    if not isinstance(send_id, str):
        return "send_id must be string"
    if not SEND_ID.fullmatch(send_id):
        return "send_id invalid format"
    version = send.get("plugin_version")
    if not isinstance(version, str):
        return "plugin_version must be string"
    if not VERSION.fullmatch(version):
        return "plugin_version invalid format"
    records = send.get("records")
    if not isinstance(records, list):
        return "records must be array"
    if len(records) > MAX_RECORDS:
        return f"records exceeds {MAX_RECORDS}"
    for key in send:
        if key not in ("schema", "send_id", "plugin_version", "records"):
            return f"Unknown key at root: {key}"
    for i, rec in enumerate(records):
        if (why := _record(rec)):
            return f"Record {i}: {why}"
    try:
        size = len(body_of(send))
    except (TypeError, ValueError):
        return "Body is not JSON"
    if size > MAX_BODY_BYTES:
        return "Body too large"
    return None


def body_of(send: dict[str, Any]) -> bytes:
    """The bytes of ``send`` as they are posted: JSON with sorted keys."""
    return json.dumps(send, sort_keys=True).encode("utf-8")


# ── Building the records ──────────────────────────────────────────────────────


def _two(value: float, high: float) -> float:
    """``value`` between 0 and ``high``, to two decimals."""
    return round(min(max(float(value), 0.0), float(high)), 2)


class _Talk:
    """The spoken words of one video, and which of them the saved edit still plays."""

    def __init__(self, words: list[dict[str, Any]], removed: list[tuple[float, float]]) -> None:
        self.spoken = sorted((w for w in words if not is_event(w)), key=lambda w: (float(w["start"]), float(w["end"])))
        self.mids = [(float(w["start"]) + float(w["end"])) / 2 for w in self.spoken]
        self.removed = sorted(removed)
        self._removed_starts = [s for s, _ in self.removed]

    def is_kept(self, i: int) -> bool:
        mid = self.mids[i]
        j = bisect.bisect_right(self._removed_starts, mid) - 1
        return not (j >= 0 and self.removed[j][0] <= mid <= self.removed[j][1])

    def inside(self, start: float, end: float) -> range:
        """The indexes of the spoken words whose middle falls inside ``start`` to ``end``."""
        return range(bisect.bisect_left(self.mids, start), bisect.bisect_right(self.mids, end))

    def said(self, i: int) -> str:
        return plain_text(str(self.spoken[i]["word"]))

    def filler_at(self, i: int) -> str | None:
        """The filler word at ``i`` as said, or the two-word filler it is half of, else None."""
        here = self.said(i)
        if here in FILLER:
            return here
        for j in (i - 1, i + 1):
            if 0 <= j < len(self.spoken):
                pair = f"{self.said(j)} {here}" if j < i else f"{here} {self.said(j)}"
                if pair in _TWO_WORD_FILLERS:
                    return pair
        return None

    def shape(self, i: int) -> dict[str, Any]:
        """The shape of the word at ``i``: how it sounds and where it stands, never what it says."""
        word = self.spoken[i]
        end = float(word["end"])
        following = float(self.spoken[i + 1]["start"]) - end if i + 1 < len(self.spoken) else 0.0
        filler = self.filler_at(i)
        return {
            "pos": "filler" if filler else part_of_speech(self.said(i)),
            "dur": _two(end - float(word["start"]), MAX_WORD_SECONDS),
            "gap_after": _two(following, MAX_GAP_SECONDS),
            "pitch": "unknown",
            "filler": filler,
        }

    def around(self, start: float, end: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Up to three played words before ``start`` and after ``end``, each side in speaking order."""
        inside = self.inside(start, end)
        before, after = [], []
        i = inside.start - 1
        while i >= 0 and len(before) < MAX_CONTEXT_WORDS:
            if self.is_kept(i):
                before.insert(0, self.shape(i))
            i -= 1
        i = inside.stop
        while i < len(self.spoken) and len(after) < MAX_CONTEXT_WORDS:
            if self.is_kept(i):
                after.append(self.shape(i))
            i += 1
        return before, after

    def sentence_position(self, start: float, end: float) -> str:
        inside = self.inside(start, end)
        if not len(inside):
            previous = inside.start - 1
            return "end" if previous >= 0 and _is_sentence_end(self.spoken[previous]) else "middle"
        first, last = inside.start, inside.stop - 1
        begins = first == 0 or _is_sentence_end(self.spoken[first - 1])
        ends = _is_sentence_end(self.spoken[last])
        return "whole" if begins and ends else "start" if begins else "end" if ends else "middle"

    def silence_left(self, start: float, end: float) -> float | None:
        """The silence a played cut leaves at its join: the pause before it plus the pause after it, or None."""
        inside = self.inside(start, end)
        i = inside.start - 1
        while i >= 0 and not self.is_kept(i):
            i -= 1
        j = inside.stop
        while j < len(self.spoken) and not self.is_kept(j):
            j += 1
        if i < 0 or j >= len(self.spoken):
            return None
        left = (start - float(self.spoken[i]["end"])) + (float(self.spoken[j]["start"]) - end)
        return _two(left, MAX_GAP_SECONDS)


def _overlap(a: dict[str, Any], b: dict[str, Any]) -> float:
    return min(float(a["end"]), float(b["end"])) - max(float(a["start"]), float(b["start"]))


def _cached_labels(project: Project, sounds: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """The sound labels: the ones given, else the ones cached beside the project. Never measures."""
    if sounds is None:
        try:
            sounds = json.loads((project.root / SOUNDS_FILE).read_text(encoding="utf-8")).get("labels")
        except (OSError, ValueError, AttributeError):
            sounds = None
    return [x for x in sounds if isinstance(x, dict)] if isinstance(sounds, list) else []


def _laugh_spans(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The laughs among the sound labels, as spans."""
    found = []
    for label in labels:
        try:
            if label.get("kind") == LAUGH:
                found.append({"start": float(label["start"]), "end": float(label["end"])})
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
    return found


def _laugh_within(laughs: list[dict[str, Any]], span: dict[str, Any]) -> float | None:
    """The distance in seconds from ``span`` to the nearest laugh, 0 when they touch, or None."""
    if not laughs:
        return None
    nearest = min(max(0.0, -_overlap(span, laugh)) for laugh in laughs)
    return round(nearest, 2) if nearest <= MAX_LAUGH_SECONDS else None


def _kind(internal: Any) -> str:
    return SENDABLE_KIND.get(internal, "other") if isinstance(internal, str) else "other"


def _signals(edit: dict[str, Any], changes: dict[str, Any]) -> list[dict[str, Any]]:
    """What the creator did, each as ``{action, start, end, source, kind}``, then the cuts they left be."""
    requested = [r for r in edit.get("requested", []) if isinstance(r, dict)]
    rated = changes.get("rated", {})
    wrong = rated.get("bad", [])
    found: list[dict[str, Any]] = []

    def add(action: str, span: dict[str, Any], source: str, kind: Any) -> None:
        found.append({"action": action, "start": float(span["start"]), "end": float(span["end"]),
                      "source": source, "kind": _kind(kind)})

    for k in changes.get("put_back", []):
        if not taste._marked_wrong(k, wrong):  # a Wrong cut is one record: the rating
            add("put_back", k, "claude", taste._kind_of(k, requested))
    for side in edits.RATINGS:
        for r in rated.get(side, []):
            add(f"rated_{side}", r, "pick" if r.get("kind") == edits.PICK_KIND else "claude", r.get("kind"))
    for b in changes.get("brought_back", []):
        ours = taste._kind_of(b, requested)
        if ours != taste.UNKNOWN_KIND:
            add("brought_back_words", b, "claude", ours)
        else:
            add("brought_back_words", b, "auto", "fillers" if taste._said(b.get("text", "")) in FILLER else "other")
    for c in changes.get("cuts", []):
        add("cut_by_hand", c, "creator", "other")
    for k in changes.get("kept", []):
        add("kept_part", k, "creator", "other")
    marked = [r for side in edits.RATINGS for r in rated.get(side, [])]
    for cut in edit.get("cuts", []):
        if cut.get("source") not in (edits.BY_CLAUDE, edits.BY_PICK):
            continue
        if any(_overlap(cut, r) > 0 for r in marked):
            continue
        add("kept", cut, "pick" if cut.get("source") == edits.BY_PICK else "claude", cut.get("kind"))
    return found


def build_send(
    project: Project,
    *,
    duration: float | None = None,
    sounds: list[dict[str, Any]] | None = None,
    join_rows: Any = None,
) -> dict[str, Any]:
    """The send for one video, built from its saved edit and its words. Nothing is sent or saved.

    One record for each thing the creator changed (see ``_signals``), then one
    ``kept`` record for each of Claude's cuts they left in place, up to 500
    records and 256 KiB. ``duration``, the laugh labels ``sounds`` and the
    ``join_rows`` check are looked up when left out; the labels are only read
    from the cache, never measured. Raises StudioError when the video has no
    transcript, and checks what it built with ``validate_send``.
    """
    duration = project.duration() if duration is None else duration
    edit = edits.load_edit(project, duration)
    words = word_times.read(project).words
    talk = _Talk(words, edits.removed_spans(edit))
    changes = treatment.creator_changes(edit) or {}
    pace = treatment.saved_treatment(edit).pace
    pace = pace if pace in PACES else "custom"
    labels = _cached_labels(project, sounds)
    laughs = _laugh_spans(labels)
    cuts = [c for c in edit.get("cuts", []) if isinstance(c, dict) and "start" in c and "end" in c]
    rows = _join_rows(cuts, words, duration, labels, join_rows) if cuts else []

    records = []
    for sig in _signals(edit, changes):
        start, end = sig["start"], sig["end"]
        before, after = talk.around(start, end)
        cut = max((c for c in cuts if _overlap(c, sig) > 0), key=lambda c: _overlap(c, sig), default=None)
        row = max((r for r in rows if _overlap(r, sig) > 0), key=lambda r: _overlap(r, sig), default=None)
        records.append({
            "pace": pace,
            "proposed": {"source": sig["source"], "kind": sig["kind"], "length_s": _two(end - start, MAX_LENGTH_SECONDS),
                         "words": min(len(talk.inside(start, end)), MAX_WORDS)},
            "context": {"before": before, "after": after, "sentence_position": talk.sentence_position(start, end),
                        "laugh_within_s": _laugh_within(laughs, sig)},
            "creator": {"action": sig["action"]},
            "join": {"gap_left_s": talk.silence_left(float(cut["start"]), float(cut["end"])) if cut else None,
                     "flags": [f for f in (row or {}).get("flags", []) if f in FLAG]},
        })
    send = {"schema": SCHEMA, "send_id": secrets.token_urlsafe(16)[:22], "plugin_version": _version(),
            "records": _fitted(records)}
    if (why := validate_send(send)):
        raise StudioError(f"The changes could not be made into something safe to send ({why}). Nothing was sent.")
    return send


def _join_rows(cuts: list[dict[str, Any]], words: list[dict[str, Any]], duration: float,
               labels: list[dict[str, Any]], join_rows: Any) -> list[dict[str, Any]]:
    """The join check's rows for the video's cuts, or none when it can't say."""
    try:
        if join_rows is None:
            from lumr_studio.joins import join_rows
        return join_rows(cuts, words, duration, labels=labels or None)
    except Exception as err:  # noqa: BLE001 - flags are a bonus; a send goes without them
        log.debug("no join flags for the send: %s", err)
        return []


def _version() -> str:
    version = feedback.plugin_version()
    return version if version and VERSION.fullmatch(version) else UNKNOWN_VERSION


def _fitted(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The first records that fit in 500 records and 256 KiB. What the creator did comes before what they left be."""
    used = len(body_of({"schema": SCHEMA, "send_id": "x" * 22, "plugin_version": "0.0.0", "records": []}))
    kept = []
    for rec in records[:MAX_RECORDS]:
        used += len(json.dumps(rec, sort_keys=True).encode("utf-8")) + 1
        if used > MAX_BODY_BYTES:
            break
        kept.append(rec)
    return kept


# ── The preview ───────────────────────────────────────────────────────────────


def _how_long(seconds: float, words: int) -> str:
    whole = round(seconds)
    length = f"{whole} s" if whole else "under 1 s"
    return f"({length}, {words} word{'' if words == 1 else 's'})" if words else f"({length})"


def describe(send: dict[str, Any]) -> list[str]:
    """One plain line for each record, for the creator to read before anything is sent.

    Lengths are whole seconds. No word from the video appears; only a filler
    word the record names would, and the lines never name one.
    """
    lines = []
    for rec in send.get("records", []):
        proposed, action = rec["proposed"], rec["creator"]["action"]
        name = KIND_NOUN.get(proposed["kind"], "cut")
        noun = f"{'an' if name[0] in 'aeiou' else 'a'} {name}"
        size = _how_long(proposed["length_s"], proposed["words"])
        line = {
            "put_back": f"You put back {noun} {size}",
            "rated_good": f"You marked {noun} as good {size}",
            "rated_bad": f"You marked {noun} as wrong {size}",
            "brought_back_words": f"You brought back words that were cut {size}",
            "cut_by_hand": f"You cut words by hand {size}",
            "kept_part": f"You kept a part no matter what {size}",
            "kept": f"You left {noun} in place {size}",
        }[action]
        laugh = rec["context"]["laugh_within_s"]
        lines.append(line + (" near a laugh." if laugh is not None and laugh <= NEAR_A_LAUGH_SECONDS else "."))
    return lines


# ── Sending ───────────────────────────────────────────────────────────────────


def _need_indexes(removed: Any, count: int) -> set[int]:
    if removed is None:
        return set()
    if not isinstance(removed, (list, tuple, set)) or any(
            isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < count for i in removed):
        raise StudioError("Those lines are not in the list. Open the preview again.")
    return set(removed)


def send(project: Project, payload: dict[str, Any], removed_indexes: Any = None) -> dict[str, Any]:
    """Send ``payload`` without the records at ``removed_indexes``. Answers ``{"send_id", "records"}``.

    The records left are checked with ``validate_send``, then posted once to
    the team's server: 15 seconds at most, no retry, nothing that names this
    machine. On success a line goes into ``<project>/shares.jsonl`` with the
    ID, the date and the record count. Raises StudioError, in plain words,
    when sharing is off, nothing is left to send, the check fails, the server
    can't be reached or it answers with an error.
    """
    url = share.share_url()
    if url is None:
        raise StudioError("Sharing isn't set up on this machine yet, so nothing can be sent.")
    records = payload.get("records", []) if isinstance(payload, dict) else []
    if not isinstance(records, list):
        raise StudioError("There is nothing to send. Open the preview again.")
    gone = _need_indexes(removed_indexes, len(records))
    body = {**payload, "records": [r for i, r in enumerate(records) if i not in gone]}
    if not body["records"]:
        raise StudioError("There is nothing left to send.")
    if (why := validate_send(body)):
        raise StudioError(f"That can't be sent ({why}). Nothing was sent.")
    version = body["plugin_version"]
    sorry = "The Lumr Studio server answered with an error{code}. Nothing was saved. Try again later."
    try:
        request = urllib.request.Request(
            url + SENDS_PATH, data=body_of(body),
            headers={"Content-Type": "application/json", "User-Agent": f"lumr-studio/{version}"}, method="POST",
        )
        response = urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS)  # noqa: S310 - the address is the creator's own setting
        try:
            status = getattr(response, "status", None) or response.getcode()
        finally:
            close = getattr(response, "close", None)
            if close:
                close()
    except urllib.error.HTTPError as err:
        log.info("send refused: HTTP %s", err.code)
        raise StudioError(sorry.format(code=f" ({err.code})")) from None
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as err:
        log.info("send not made: %s", type(err).__name__)
        raise StudioError("Couldn't reach the Lumr Studio server. Check your connection and try again.") from None
    if not isinstance(status, int) or not 200 <= status < 300:
        log.info("send refused: HTTP %s", status)
        raise StudioError(sorry.format(code=f" ({status})" if isinstance(status, int) else ""))
    _remember(project.root, body["send_id"], len(body["records"]))
    return {"send_id": body["send_id"], "records": len(body["records"])}


def _remember(root: Path, send_id: str, records: int) -> None:
    """Add the send to ``shares.jsonl``, so the creator can ask for it to be deleted. A failure is only logged."""
    line = {"send_id": send_id, "sent_at": datetime.now(timezone.utc).date().isoformat(), "records": records}
    try:
        with open(root / SHARES_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, sort_keys=True) + "\n")
    except OSError as err:
        log.warning("could not note the send in %s: %s", SHARES_FILE, err)
