"""Carry a saved edit from one video onto another made from the same footage.

A creator edits a reel, then wants a clip added, taken out or moved.
``assemble`` builds the new sequence as a new reel, which is a new video with
no transcript and no edit. ``carry_edit`` takes the edit across.

The two videos are compared by the footage they hold. A reel says which clip
seconds play where in its sequence file; a plain video is one clip, itself.
A stretch of a clip that plays in both videos is a ``Stretch``: it maps old
seconds to new seconds. ``time_map`` finds the stretches and ``move_span``
takes one span through them. A span's start moves as a start and its end as
an end, so a cut that ended where a piece ends still ends there, and does not
reach into whatever plays next in the new video.

Three things are carried, in this order:

1. The words. The new transcript is the old one's raw words (sounds too)
   moved through the map, then the words of any clip of the new video that
   has a transcript of its own, and the speech model only for what is left.
   Hearing the same sound again can give other words, which would move the
   cuts made on them, so nothing that was heard once is heard again.
2. The word times, measured as ``transcribe`` measures them (``tools``).
3. The edit: Claude's cuts with their reasons and kinds, the filler words
   Claude picked, the creator's own cuts, the parts she kept, her ratings,
   the pace and switches, and her photos and clips. The automatic cuts are
   not carried: the recipe makes them again at the same pace.

A span is ``whole`` when all of it lands in one place, ``split`` when other
footage now plays inside it or its footage plays more than once,
``shortened`` when part of its footage is gone, and ``dropped`` when none of
it is in the new video. The report names every one that is not whole.

Nothing of the old video's project is written. In the new one, a transcript
or a list of overlays that is already there is left alone, and a saved edit
stops the carry before it starts.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from lumr_studio import edit as edits
from lumr_studio import overlays as overlay_files
from lumr_studio import reels, word_times
from lumr_studio import treatment as treatments
from lumr_studio.errors import StudioError
from lumr_studio.project import (
    Project,
    now_iso,
    open_project,
    project_dir_name,
    projects_root,
    resolve_video,
    words_path_for,
    write_json_atomic,
)
from lumr_studio.silences import SilenceProvider
from lumr_studio.taste import CARRIED_FILE
from lumr_studio.transcript import is_event, load_transcript, plain_text, spoken_words, validate_words
from lumr_studio.transcription import Transcriber, run_transcription, speech_transcriber
from lumr_studio.treatment import Treatment

Words = list[dict[str, Any]]
Span = tuple[float, float]
Pieces = list[dict[str, Any]]
LabelsFor = Callable[[Project, Words], list[dict[str, Any]]]
# (video, start, end) -> the words said in that stretch, timed in the video's own seconds.
Hear = Callable[[Path, float, float], Words]

# The kind of job that carries an edit.
KIND = "carry_edit"
# The new project records which video its edit came from in ``taste.CARRIED_FILE``:
# the old project then no longer teaches, so a change the creator made counts once.
CARRIED_VERSION = 1

# Less footage than this is no stretch, and a part of a span this short that a
# piece's edge clipped off is no part: no word fits in it. Reel pieces are a
# whole number of frames long, so edges that should meet can miss by a frame.
SLIVER_SECONDS = 0.05
# Two parts of a span this close in the new video are one part.
TOUCH_SECONDS = 0.05
# A span that lost less than this lost nothing.
SAME_SECONDS = 0.05
# A stretch without words shorter than this is not given to the speech model.
SHORTEST_HEARD_SECONDS = 0.3
# The sound of a stretch as the speech model gets it: what its own conversion makes of a video.
HEARD_SAMPLE_RATE = 16000

WHOLE, SPLIT, SHORTENED, DROPPED = "whole", "split", "shortened", "dropped"

WHY_DROPPED = "Its footage is not in the new video."
WHY_SHORTENED = "Part of its footage is not in the new video."
WHY_TWICE = "Its footage plays more than once in the new video, so it is there each time."
WHY_SPLIT = "Other footage now plays in the middle of it, so it is in {n} parts, one each side."
WHY_NO_WORD = "No whole word of it is in the new video."
WHY_OTHER_WORD = "The word at that place reads differently in the new video's transcript."
WHY_LONGEST_PART = "Other footage now plays in the middle of it, so it covers its longest part only."
WHY_TOO_SHORT = "Less than a second of its footage is in the new video."
WHY_NOT_READ = "It could not be read."

TRANSCRIPT_WAS_THERE = (
    "The new video already had a transcript, so it was left as it is and no words were carried. "
    "The edit was placed on those words by time; check the cuts and the picked words against them."
)
OVERLAYS_WERE_THERE = "The new video already had photos or clips placed, so they were left as they are and none were carried."
AFTER_CARRYING = (
    "The automatic cuts were made again at the same pace, so their count can differ from the old video. "
    "Read joins and fix what it flags. Tell the creator what carried, and name everything that was split, "
    "shortened, dropped or not placed. The old video and its edit are as they were."
)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_span(value: Any) -> bool:
    return isinstance(value, dict) and _finite(value.get("start")) and _finite(value.get("end")) \
        and value["end"] >= value["start"]


# ── The map ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Stretch:
    """Footage that plays in both videos: old seconds ``old_start`` to ``old_end`` play from ``new_start`` in the new one."""

    old_start: float
    old_end: float
    new_start: float

    @property
    def length(self) -> float:
        return self.old_end - self.old_start

    @property
    def new_end(self) -> float:
        return self.new_start + self.length

    @property
    def shift(self) -> float:
        """What to add to an old time inside this stretch to get the new one."""
        return self.new_start - self.old_start


def time_map(old: Pieces, new: Pieces) -> list[Stretch]:
    """Every stretch of footage the two videos share, in the old video's order.

    ``old`` and ``new`` are pieces as a sequence file lists them: ``{clip,
    start, end, reel_start, reel_end}``, the clip seconds used and where
    they play. Two pieces of one clip share the clip seconds both use. A clip
    that plays twice in the new video gives two stretches for the same old
    seconds; one that played twice in the old video gives two that land on
    the same new seconds.
    """
    found = []
    for o in old:
        for n in new:
            if o["clip"] != n["clip"]:
                continue
            lo, hi = max(float(o["start"]), float(n["start"])), min(float(o["end"]), float(n["end"]))
            old_at = float(o["reel_start"]) + lo - float(o["start"])
            new_at = float(n["reel_start"]) + lo - float(n["start"])
            # A piece of a reel is a whole number of frames, so it can end a little before its clip seconds do.
            length = min(hi - lo, float(o["reel_end"]) - old_at, float(n["reel_end"]) - new_at)
            if length >= SLIVER_SECONDS:
                found.append(Stretch(old_start=old_at, old_end=old_at + length, new_start=new_at))
    return sorted(found, key=lambda s: (s.old_start, s.new_start))


@dataclass(frozen=True)
class Part:
    """One place a moved span lands: ``start`` to ``end`` in the new video. ``was`` is where it started in the old one."""

    start: float
    end: float
    was: float


@dataclass(frozen=True)
class Moved:
    """A span taken through the map: its parts in the new video's play order, and the seconds of it that are gone."""

    parts: list[Part]
    missing: float
    length: float
    # Whether its footage plays more than once in the new video.
    twice: bool = False

    @property
    def how(self) -> str:
        if not self.parts:
            return DROPPED
        if len(self.parts) > 1:
            return SPLIT
        return SHORTENED if self.missing > SAME_SECONDS else WHOLE

    def why(self) -> str:
        """Why it did not arrive whole, in words a creator can read. Empty when it did."""
        if self.how == DROPPED:
            return WHY_DROPPED
        if self.how == SPLIT:
            return WHY_TWICE if self.twice else WHY_SPLIT.format(n=len(self.parts))
        return WHY_SHORTENED if self.how == SHORTENED else ""


def _merged(spans: list[Span]) -> list[Span]:
    out: list[Span] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def move_span(start: float, end: float, stretches: list[Stretch]) -> Moved:
    """Where the old video's ``[start, end]`` plays in the new one.

    Each stretch the span overlaps gives one part: the span's start moves
    with the stretch it starts in and its end with the stretch it ends in.
    Parts that meet in the new video are one part, so a span over two pieces
    that still play back to back arrives whole. A part shorter than
    ``SLIVER_SECONDS`` that an edge clipped off is left out. A span with no
    length is a point and lands in the first stretch that holds it.
    """
    length = end - start
    found: list[Part] = []
    old: list[Span] = []
    for s in stretches:
        a, b = max(start, s.old_start), min(end, s.old_end)
        if b < a or (b == a and length > 0):
            continue
        if b - a < min(SLIVER_SECONDS, length):
            continue
        found.append(Part(start=a + s.shift, end=b + s.shift, was=a))
        old.append((a, b))
        if length <= 0:
            break
    parts: list[Part] = []
    for part in sorted(found, key=lambda p: (p.start, p.end)):
        if parts and part.start - parts[-1].end <= TOUCH_SECONDS:
            parts[-1] = replace(parts[-1], end=max(parts[-1].end, part.end))
        else:
            parts.append(part)
    held = sum(b - a for a, b in _merged(old))
    twice = sum(b - a for a, b in old) > held + SAME_SECONDS
    return Moved(parts=parts, missing=max(0.0, length - held), length=length, twice=twice)


# ── The words ─────────────────────────────────────────────────────────────────


def _mid(word: dict[str, Any]) -> float:
    return (float(word["start"]) + float(word["end"])) / 2


def _without(spans: list[Span], holes: list[Span]) -> list[Span]:
    return [piece for a, b in spans for piece in edits.span_without(a, b, holes)]


def fill_from(words: Words, stretches: list[Stretch], free: list[Span]) -> tuple[Words, list[Span]]:
    """The words of one transcript moved into the parts of the new video that have none yet.

    ``words`` are on the times ``stretches`` map from; ``free`` are the new
    video's seconds still without words. A word goes with the stretch its
    middle sits in, keeps every other key it has (a sound stays a sound) and
    never reaches past the footage it came with. Returns the moved words in
    time order and what of ``free`` is still without words. Footage that
    played twice in the old video fills its place once.
    """
    out: Words = []
    for s in stretches:
        holes = [(max(a, s.new_start), min(b, s.new_end)) for a, b in free]
        holes = [(a, b) for a, b in holes if b > a]
        if not holes:
            continue
        for word in words:
            mid = _mid(word)
            if not s.old_start <= mid < s.old_end:
                continue
            hole = next((h for h in holes if h[0] <= mid + s.shift < h[1]), None)
            if hole is None:
                continue
            a = min(max(float(word["start"]) + s.shift, hole[0]), hole[1])
            b = min(max(float(word["end"]) + s.shift, hole[0]), hole[1])
            out.append({**word, "start": round(a, 3), "end": round(b, 3)})
        free = _without(free, holes)
    return sorted(out, key=lambda w: (float(w["start"]), float(w["end"]))), free


def cut_sound(video: Path, start: float, end: float, out: Path) -> None:
    """Write the sound of ``video`` from ``start`` to ``end`` to ``out``, mono, as the speech model takes it."""
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-y",
        "-ss", f"{start:.3f}", "-t", f"{max(end - start, 0.0):.3f}", "-i", str(video),
        "-vn", "-ac", "1", "-ar", str(HEARD_SAMPLE_RATE), str(out),
    ]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        raise StudioError("ffmpeg is not installed or not on the PATH. Install ffmpeg and try again.") from None
    if done.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        raise StudioError(
            f"ffmpeg could not read the sound of {video.name} from {edits.clock(start)} to {edits.clock(end)}: "
            f"{(done.stderr or '').strip()[-300:]} Check that the video plays through."
        )


def hear(video: Path, start: float, end: float, *, transcriber: Transcriber = speech_transcriber) -> Words:
    """The words said in ``video`` between ``start`` and ``end``, from the speech model, timed in the video's seconds.

    The stretch's sound is written to a folder of its own, transcribed there
    the way ``transcribe`` transcribes a video, and the folder is removed.
    Nothing is written beside the video.
    """
    with tempfile.TemporaryDirectory(prefix="lumr-carry-") as folder:
        piece = Path(folder) / "stretch.wav"
        cut_sound(video, start, end, piece)
        run_transcription(Project(video=piece, root=Path(folder) / "project"), True, transcriber)
        heard = load_transcript(words_path_for(piece))
    out = []
    for word in heard:
        a = min(max(float(word["start"]) + start, start), end)
        b = min(max(float(word["end"]) + start, start), end)
        out.append({**word, "start": round(a, 3), "end": round(b, 3)})
    return out


# ── The plan ──────────────────────────────────────────────────────────────────


def footage_of(video: Path, duration: float) -> Pieces:
    """The footage a video holds, as pieces: itself as one whole clip, and, for a reel, the clips it was joined from."""
    whole = {"clip": str(video), "start": 0.0, "end": duration, "reel_start": 0.0, "reel_end": duration}
    path = reels.sequence_path_for(video)
    if not path.is_file():
        return [whole]
    try:
        return [whole, *reels.read_sequence(path)["pieces"]]
    except StudioError:
        return [whole]


def _clip_transcripts(pieces: Pieces) -> list[tuple[Words, list[Stretch]]]:
    """The transcript of each clip of the new video that has one beside it, with where its seconds play."""
    found = []
    for clip in dict.fromkeys(str(p["clip"]) for p in pieces):
        path = words_path_for(Path(clip))
        if not path.is_file():
            continue
        try:
            words = load_transcript(path)
        except StudioError:
            continue
        whole = {"clip": clip, "start": 0.0, "end": math.inf, "reel_start": 0.0, "reel_end": math.inf}
        found.append((words, time_map([whole], pieces)))
    return found


def _has_an_edit(edit: dict[str, Any]) -> bool:
    return any(edit.get(key) for key in ("cuts", "requested", "creator_cuts", "keep", "picks"))


@dataclass(frozen=True)
class CarryPlan:
    """Everything a carry needs, read and checked before anything is written."""

    source: Project
    target: Project
    stretches: list[Stretch]
    edit: dict[str, Any]
    # Whether the old edit sits on measured word times.
    measured: bool
    duration: float
    # The new transcript as far as it can be made without the speech model, or None when the new video has one.
    words: Words | None = None
    carried_words: int = 0
    # The new video's seconds the speech model has to hear.
    to_hear: tuple[Span, ...] = ()


def plan_carry(from_video: str, to_video: str) -> CarryPlan:
    """Check both videos and work out what carries. Writes nothing but the new video's empty project folder.

    Raises StudioError, saying what to do, when the old video has no saved
    edit, the new one already has one, or the two share no footage.
    """
    old_video = resolve_video(from_video)
    source = Project(video=old_video, root=projects_root() / project_dir_name(old_video))
    if resolve_video(to_video) == old_video:
        raise StudioError("from_video and to_video are the same video. Pass the video that has the edit and the new one.")
    old_duration = source.duration()
    edit = edits.load_edit(source, old_duration) if source.edit_path.exists() else edits.empty_edit()
    if not _has_an_edit(edit):
        raise StudioError(
            f"{old_video.name} has no saved edit, so there is nothing to carry. "
            "Check that from_video is the video that was edited, or make the edit on to_video with set_edit."
        )
    target = open_project(to_video)
    if target.edit_path.exists():
        raise StudioError(
            f"{target.video.name} already has a saved edit, and carry_edit never replaces one. "
            "To carry onto a fresh video, call assemble with the same pieces under another name and pass that reel "
            "as to_video. To keep the edit that is there, change it with set_edit."
        )
    duration = target.duration()
    old_pieces, new_pieces = footage_of(old_video, old_duration), footage_of(target.video, duration)
    stretches = time_map(old_pieces, new_pieces)
    if not stretches:
        raise StudioError(
            f"{old_video.name} and {target.video.name} share no footage, so no cut has a place to go. "
            "carry_edit works when to_video was joined by assemble from clips of from_video, or holds from_video as a clip."
        )
    measured = source.words_path.exists() and word_times.source_of(source) == word_times.MEASURED
    plan = CarryPlan(source=source, target=target, stretches=stretches, edit=edit, measured=measured, duration=duration)
    if target.words_path.exists():
        return plan
    free: list[Span] = [(0.0, duration)]
    words: Words = []
    if source.words_path.exists():
        words, free = fill_from(load_transcript(source.words_path), stretches, free)
    carried = len(words)
    for theirs, where in _clip_transcripts(new_pieces):
        more, free = fill_from(theirs, where, free)
        words += more
    to_hear = tuple((a, b) for a, b in _merged(free) if b - a >= SHORTEST_HEARD_SECONDS)
    return replace(plan, words=words, carried_words=carried, to_hear=to_hear)


# ── Writing the transcript ────────────────────────────────────────────────────


def _write_new(path: Path, words: Words) -> bool:
    """Write ``words`` to ``path`` unless a file is there. True when it was written."""
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(words, fh, ensure_ascii=False, indent=2)
        reels.put_in_place(tmp, path)
    except FileExistsError:
        tmp.unlink(missing_ok=True)
        return False
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return True


def write_transcript(
    plan: CarryPlan, *, hear: Hear = hear, on_progress: Callable[[float], None] = lambda _f: None
) -> dict[str, Any]:
    """Write the new video's transcript from the carried words and what the speech model hears in the rest.

    Answers ``{status: "written", words_path, word_count, carried_words,
    words_from_clips, heard: [{clock, start, end, words}]}``, or ``{status:
    "exists", words_path, note}`` when the new video already had a
    transcript, which is never replaced.
    """
    path = plan.target.words_path
    if plan.words is None:
        return {"status": "exists", "words_path": str(path), "note": TRANSCRIPT_WAS_THERE}
    words = list(plan.words)
    heard = []
    for i, (start, end) in enumerate(plan.to_hear):
        said = hear(plan.target.video, start, end)
        words += said
        heard.append({
            "clock": edits.clock_span(start, end), "start": round(start, 3), "end": round(end, 3),
            "words": len(spoken_words(said)),
        })
        on_progress((i + 1) / len(plan.to_hear))
    words = sorted(validate_words(words), key=lambda w: (float(w["start"]), float(w["end"])))
    if not _write_new(path, words):
        return {"status": "exists", "words_path": str(path), "note": TRANSCRIPT_WAS_THERE}
    return {
        "status": "written", "words_path": str(path), "word_count": len(spoken_words(words)),
        "carried_words": plan.carried_words, "words_from_clips": len(plan.words) - plan.carried_words,
        "heard": heard,
    }


# ── Placing the edit ──────────────────────────────────────────────────────────


def _shown(start: float, end: float) -> dict[str, Any]:
    return {"clock": edits.clock_span(start, end), "start": round(start, 3), "end": round(end, 3)}


def _entry(span: dict[str, Any], parts: list[Span], why: str, **more: Any) -> dict[str, Any]:
    """One line of the report: where a span was, where it is now, and why it is not as it was."""
    return {
        "was": _shown(float(span["start"]), float(span["end"])), "now": [_shown(a, b) for a, b in parts],
        **{k: v for k, v in more.items() if v}, "why": why,
    }


class _Tally:
    """What became of one sort of span: a count of those that arrived whole and a line for each that did not."""

    def __init__(self, total: int) -> None:
        self.report: dict[str, Any] = {"total": total, "carried": 0, SPLIT: [], SHORTENED: [], DROPPED: []}

    def add(self, span: dict[str, Any], how: str, parts: list[Span], why: str, **more: Any) -> None:
        if how == WHOLE:
            self.report["carried"] += 1
        else:
            self.report[how].append(_entry(span, parts, why, **more))

    def dropped(self, span: Any, why: str, **more: Any) -> None:
        shown = span if _is_span(span) else {"start": 0.0, "end": 0.0}
        self.report[DROPPED].append(_entry(shown, [], why, **more))


def _spans(moved: Moved) -> list[Span]:
    return [(round(p.start, 3), round(p.end, 3)) for p in moved.parts]


def _said(words: Words) -> str:
    """What a run of words says, shortened the way the page shortens a kept part."""
    text = " ".join(str(w["word"]).strip() for w in words if not is_event(w))
    limit = treatments.KEEP_TEXT_CHARS
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "..."


def _on_words(a: float, b: float, said: edits.SaidOrder) -> tuple[float, float, Words] | None:
    """``[a, b]`` drawn onto the words whose middle it holds: from the first one's start to the last one's end."""
    picked = said.picked(a, b)
    if not picked:
        return None
    chosen = said.tokens[picked.start:picked.stop]
    return (round(min(float(w["start"]) for w in chosen), 3), round(max(float(w["end"]) for w in chosen), 3), chosen)


def _carried_requests(old: list[Any], stretches: list[Stretch]) -> tuple[list[dict[str, Any]], list[int], list[Moved], _Tally]:
    """Claude's cuts moved: ``(requested, which old cut each came from, how each old cut moved, tally)``."""
    tally = _Tally(len(old))
    requested: list[dict[str, Any]] = []
    origin: list[int] = []
    moves: list[Moved] = []
    for i, raw in enumerate(old):
        if not (_is_span(raw) and raw["end"] > raw["start"]):
            moves.append(Moved(parts=[], missing=0.0, length=0.0))
            tally.dropped(raw, WHY_NOT_READ)
            continue
        moved = move_span(float(raw["start"]), float(raw["end"]), stretches)
        moves.append(moved)
        for a, b in _spans(moved):
            requested.append({**raw, "start": a, "end": b})
            origin.append(i)
    return requested, origin, moves, tally


def _carried_word_spans(old: list[Any], stretches: list[Stretch], said: edits.SaidOrder, prefix: str,
                        tally: _Tally) -> list[dict[str, Any]]:
    """Spans that hold exactly the words picked (the creator's cuts, the words she brought back), moved onto the same words."""
    out = []
    for raw in old:
        if not _is_span(raw):
            tally.dropped(raw, WHY_NOT_READ)
            continue
        moved = move_span(float(raw["start"]), float(raw["end"]), stretches)
        parts: list[Span] = []
        for a, b in _spans(moved):
            found = _on_words(a, b, said)
            if found is None:
                continue
            start, end, chosen = found
            parts.append((start, end))
            piece = {**raw, "id": f"{prefix}{start:.2f}-{end:.2f}", "start": start, "end": end, "text": _said(chosen)}
            if "words" in raw:
                piece["words"] = len(chosen)
            out.append(piece)
        text = str(raw.get("text", ""))
        if not parts:
            tally.dropped(raw, WHY_NO_WORD if moved.parts else WHY_DROPPED, text=text)
        else:
            tally.add(raw, SPLIT if len(parts) > 1 else moved.how, parts, moved.why(), text=text)
    return sorted(out, key=lambda c: float(c["start"]))


def _carried_keeps(old: list[Any], stretches: list[Stretch], words: Words, said: edits.SaidOrder) -> tuple[list[dict[str, Any]], _Tally]:
    """The parts the creator kept, moved. Words brought back one by one stay on those words."""
    tally = _Tally(len(old))
    exact = [k for k in old if isinstance(k, dict) and k.get("origin") == treatments.ORIGIN_EXACT]
    out = _carried_word_spans(exact, stretches, said, edits.KEEP_PREFIX, tally)
    for raw in old:
        if raw in exact:
            continue
        if not _is_span(raw):
            tally.dropped(raw, WHY_NOT_READ)
            continue
        moved = move_span(float(raw["start"]), float(raw["end"]), stretches)
        parts = _spans(moved)
        for a, b in parts:
            piece = {**raw, "id": f"{edits.KEEP_PREFIX}{a:.2f}-{b:.2f}", "start": a, "end": b}
            if "text" in raw and moved.how != WHOLE:
                piece["text"] = _said([w for w in words if a <= _mid(w) <= b])
            out.append(piece)
        tally.add(raw, moved.how, parts, moved.why(), text=str(raw.get("text", "")), note=str(raw.get("note", "")))
    return sorted(out, key=lambda k: float(k["start"])), tally


def _carried_picks(old: list[dict[str, Any]], stretches: list[Stretch], said: edits.SaidOrder) -> tuple[list[dict[str, Any]], _Tally]:
    """The filler words Claude picked, found again by where they play now. An id names a word's own times."""
    tally = _Tally(len(old))
    out: dict[str, dict[str, Any]] = {}
    for raw in old:
        if not _is_span(raw):
            tally.dropped(raw, WHY_NOT_READ)
            continue
        text = str(raw.get("text", ""))
        moved = move_span(float(raw["start"]), float(raw["end"]), stretches)
        spots: list[Span] = []
        reads_differently = False
        for a, b in _spans(moved):
            found = _on_words(a, b, said)
            if found is None:
                continue
            start, end, chosen = found
            if text and plain_text(" ".join(str(w["word"]).strip() for w in chosen)) != plain_text(text):
                reads_differently = True
                continue
            spots.append((start, end))
        if not spots:
            why = WHY_OTHER_WORD if reads_differently else WHY_NO_WORD if moved.parts else WHY_DROPPED
            tally.dropped(raw, why, text=text)
            continue
        for start, end in spots:
            pick = {**raw, "id": edits.pick_id(start, end), "start": start, "end": end}
            out.setdefault(pick["id"], pick)
        tally.add(raw, SPLIT if len(spots) > 1 else WHOLE, spots, WHY_TWICE if len(spots) > 1 else "", text=text)
    return sorted(out.values(), key=lambda k: float(k["start"])), tally


def _carried_ratings(old: list[dict[str, Any]], stretches: list[Stretch]) -> tuple[list[dict[str, Any]], _Tally]:
    """The creator's ratings of Claude's cuts, each on the first place its cut plays now."""
    tally = _Tally(len(old))
    out = []
    for raw in old:
        if not _is_span(raw):
            tally.dropped(raw, WHY_NOT_READ)
            continue
        moved = move_span(float(raw["start"]), float(raw["end"]), stretches)
        parts = _spans(moved)
        if not parts:
            tally.dropped(raw, WHY_DROPPED, rating=str(raw.get("rating", "")), reason=str(raw.get("reason", "")))
            continue
        a, b = parts[0]
        out.append({**raw, "id": f"{edits.RATING_PREFIX}{a:.2f}-{b:.2f}", "start": a, "end": b})
        tally.report["carried"] += 1
    return out, tally


def _name_the_rows(made: treatments.MadeEdit, keep: list[dict[str, Any]], ratings: list[dict[str, Any]]) -> None:
    """Point each put-back part and each rating at the row its cut has on the new video's page.

    A row's id comes from where the cut is placed, so the old video's ids
    name nothing here. One that finds no row loses the name and is found by
    its span.
    """
    placed = made.outcome.placed

    def row_of(span: dict[str, Any], put_back: bool) -> str | None:
        return next((
            treatments.claude_row_id(p) for p in placed
            if p["start"] < float(span["end"]) and float(span["start"]) < p["end"] and (p["put_back"] or not put_back)
        ), None)

    for k in keep:
        if "cut" in k:
            row = row_of(k, put_back=True)
            k.pop("cut")
            if row:
                k["cut"] = row
    for r in ratings:
        row = row_of(r, put_back=False)
        r.pop("row", None)
        if row:
            r["row"] = row


def _old_requests(edit: dict[str, Any]) -> list[Any]:
    """Claude's cuts as asked for. An edit saved without them gives the cuts it saved as Claude's."""
    requested = edit.get("requested")
    if isinstance(requested, list) and requested:
        return list(requested)
    return [
        {"start": c["start"], "end": c["end"], "reason": c.get("reason") or "Carried from the old video",
         **({"kind": c["kind"]} if c.get("kind") in edits.CUT_KINDS else {})}
        for c in edit.get("cuts", []) if c.get("source") == edits.BY_CLAUDE
    ]


def _old_treatment(edit: dict[str, Any]) -> Treatment:
    try:
        return treatments.saved_treatment(edit)
    except StudioError:
        return treatments.starting_treatment()


def _not_placed(made: treatments.MadeEdit, requested: list[dict[str, Any]], origin: list[int],
                old: list[Any]) -> tuple[list[dict[str, Any]], set[int]]:
    """The moved cuts the new video's words would not take, with why: ``(lines, which old cuts)``.

    A cut the creator put back is refused by her kept part in every edit; it
    carried, and reads as put back on the page.
    """
    put_back = {p["index"] for p in made.outcome.placed if p["put_back"]}
    lines, which = [], set()
    for r in made.outcome.rejected:
        i = r["index"]
        if i in put_back:
            continue
        which.add(origin[i])
        was = old[origin[i]]
        lines.append(_entry(
            was, [(requested[i]["start"], requested[i]["end"])],
            str(r["why"]).removeprefix(f"Cut {i} ").strip().capitalize(), reason=str(was.get("reason", "")),
        ))
    return lines, which


def _carried_overlays(plan: CarryPlan, words: Words) -> dict[str, Any] | None:
    """Move the creator's photos and clips into the new project. None when the old video has none.

    An overlay keeps its id, its file and its look. One that other footage
    now plays inside covers its longest part, and a clip then starts as far
    in as that part started. One with less than a second left is dropped.
    """
    try:
        saved = overlay_files.load_overlays(plan.source)
    except StudioError:
        return None
    old = saved["overlays"]
    if not old:
        return None
    path = overlay_files.overlays_path(plan.target)
    if path.exists():
        return {"total": len(old), "carried": 0, SPLIT: [], SHORTENED: [], DROPPED: [], "note": OVERLAYS_WERE_THERE}
    tally = _Tally(len(old))
    out = []
    for raw in old:
        if not _is_span(raw):
            tally.dropped(raw, WHY_NOT_READ, id=str(raw.get("id", "")))
            continue
        named = {"id": str(raw.get("id", "")), "file": Path(str(raw.get("file", ""))).name}
        moved = move_span(float(raw["start"]), float(raw["end"]), plan.stretches)
        longest = max(moved.parts, key=lambda p: p.end - p.start, default=None)
        if longest is None or longest.end - longest.start < overlay_files.MIN_SPAN_SECONDS:
            tally.dropped(raw, WHY_TOO_SHORT if moved.parts else WHY_DROPPED, **named)
            continue
        start, end = round(longest.start, 3), round(min(longest.end, plan.duration), 3)
        overlay = {**raw, "start": start, "end": end, **overlay_files.anchor_words(start, end, words)}
        skipped = longest.was - float(raw["start"])
        if raw.get("kind") == overlay_files.VIDEO and skipped > SAME_SECONDS and _finite(raw.get("clip_in")) \
                and _finite(raw.get("clip_out")) and raw["clip_in"] + skipped <= raw["clip_out"] - overlay_files.MIN_SPAN_SECONDS:
            overlay["clip_in"] = round(raw["clip_in"] + skipped, 3)
        out.append(overlay)
        tally.add(raw, moved.how, [(start, end)], WHY_LONGEST_PART if moved.how == SPLIT else moved.why(), **named)
    outcome = overlay_files.OverlayOutcome(
        overlays=sorted(out, key=lambda o: float(o["start"])), next_id=int(saved.get("next_id") or 1),
    )
    requested = [{k: o[k] for k in overlay_files.INPUT_FIELDS if o.get(k) is not None} for o in outcome.overlays]
    overlay_files.save_overlays(plan.target, outcome, duration=plan.duration, requested=requested)
    return tally.report


@dataclass
class Placed:
    """The carried edit as saved, with what the join check needs to read it."""

    report: dict[str, Any]
    cuts: list[dict[str, Any]]
    words: Words
    duration: float
    sounds: list[dict[str, Any]]


def place_edit(plan: CarryPlan, *, silences: SilenceProvider, labels: LabelsFor) -> Placed:
    """Move every span of the old edit through the map, build the edit on the new video's words, and save it.

    The edit is built by the one recipe (``treatment.make_edit``) with the
    old edit's pace and switches, so the automatic cuts are made again and
    every rule that holds for ``set_edit`` holds here. Raises StudioError
    when the new video got a saved edit in the meantime.
    """
    project, old = plan.target, plan.edit
    times = word_times.read(project)
    words, duration = times.words, plan.duration
    said = edits.SaidOrder.of(words)
    measured = silences(project.video)
    sounds = labels(project, words)

    asked = _old_requests(old)
    requested, origin, moves, cuts = _carried_requests(asked, plan.stretches)
    hers = _Tally(len(old.get("creator_cuts", [])))
    creator_cuts = _carried_word_spans(old.get("creator_cuts", []), plan.stretches, said, edits.CREATOR_ROW_PREFIX, hers)
    keep, kept = _carried_keeps(old.get("keep", []), plan.stretches, words, said)
    old_picks = treatments.saved_picks(old)
    picks, picked = _carried_picks(old_picks, plan.stretches, said) if old_picks is not None else (None, None)
    ratings, rated = _carried_ratings(treatments.saved_ratings(old), plan.stretches)
    treatment = _old_treatment(old)

    made = treatments.make_edit(
        requested, words, duration, treatment=treatment, silences=measured, sounds=sounds,
        keeps=treatments.keep_spans_of(keep), exact_keeps=treatments.exact_spans_of(keep),
        creator_cuts=creator_cuts, plans_in=project.root, picks=picks,
    )
    _name_the_rows(made, keep, ratings)
    if project.edit_path.exists():
        raise StudioError(
            f"{project.video.name} got a saved edit while this one was being carried, and carry_edit never replaces one. "
            "Read it with get_edit."
        )
    overlays = _carried_overlays(plan, words)
    saved = treatments.save_made(
        project, made, duration=duration, requested=requested, keep=keep,
        set_by_claude=old.get("set_by_claude") if isinstance(old.get("set_by_claude"), dict) else None,
        creator_cuts=creator_cuts, times=times.source, picks=picks, ratings=ratings,
    )
    write_json_atomic(project.root / CARRIED_FILE, {
        "version": CARRIED_VERSION, "at": now_iso(), "from_video": str(plan.source.video),
        "from_project": plan.source.root.name,
    })

    refused, which = _not_placed(made, requested, origin, asked)
    for i, moved in enumerate(moves):
        if moved.parts and not (moved.how == WHOLE and i in which):
            cuts.add(asked[i], moved.how, _spans(moved), moved.why(), reason=str(asked[i].get("reason", "")))
        elif not moved.parts and _is_span(asked[i]) and asked[i]["end"] > asked[i]["start"]:
            cuts.dropped(asked[i], WHY_DROPPED, reason=str(asked[i].get("reason", "")))
    summary = edits.edit_summary(saved, duration)
    report: dict[str, Any] = {
        "cuts": {**cuts.report, "not_placed": refused},
        "creator_cuts": hers.report,
        "kept_parts": kept.report,
    }
    if picked is not None:
        report["picks"] = {**picked.report, "switch_on": treatment.picks_on}
    if rated.report["total"]:
        report["ratings"] = rated.report
    if overlays is not None:
        report["overlays"] = overlays
    report.update({
        "treatment": treatment.as_dict(),
        "automatic_cuts": summary["auto"]["count"],
        "cut_count": summary["cut_count"],
        "duration": summary["duration"],
        "removed_seconds": summary["removed_seconds"],
        "new_duration": summary["new_duration"],
    })
    return Placed(report=report, cuts=made.outcome.cuts, words=words, duration=duration, sounds=sounds)
