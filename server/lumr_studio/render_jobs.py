"""The work behind ``render`` and ``preview``: ffmpeg encodes of the saved edit.

Both run as background jobs (see jobs.py) and write only into the project's
``exports/`` folder, always under a new file name.

The edit is cut here, in one ffmpeg pass that reads the source once (see "The
cut"). ClipForge still plans it: which source to read, what is kept, how it is
encoded. ClipForge's own encode runs only where ffmpeg is too old for that pass.

When the creator has overlays saved (overlays.py), each runs in two passes:
the cut into the project's ``work/`` folder, then overlay_render.py draws the
overlays over that file into ``exports/``.

A full render has two ways in: Claude's ``render`` tool and the Export video
button on the treatment page. Both call ``start_full_render``, so both land in
the same job registry and only one runs per video at a time.
``export_status`` is the page's reading of that job.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

from lumr_studio import edit as edits
from lumr_studio.engine.render import (
    AUDIO_JOIN_FADE,
    RenderPlan,
    RenderProgress,
    build_render_plan,
    kept_segments,
    probe_duration,
    run_render,
    scale_filter_for,
)
from lumr_studio.errors import StudioError
from lumr_studio.jobs import Job, JobHandle, JobRegistry
from lumr_studio.overlay_render import (
    AUDIO_ARGS,
    FALLBACK_FPS,
    X264_ARGS,
    items_for_render,
    items_for_window,
    lay_over,
    run_ffmpeg,
)
from lumr_studio.overlays import (
    Placement,
    load_overlays,
    place_on_edit,
    probe_media,
    recheck_files,
    work_dir,
)
from lumr_studio.project import Project, reserve_unique_path
from lumr_studio.timeline import edited_duration, source_spans_for_window, to_edited_time
from lumr_studio.word_times import words_if_any

PREVIEW_CRF = 23
MAX_PREVIEW_PAD = 30.0

Span = tuple[float, float]


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


# ── The cut: every kept piece in one pass ─────────────────────────────────────
#
# ClipForge's encode gives each kept piece two trim filters of its own, and
# ffmpeg hands every frame to every one of them. Its time grows with the square
# of the number of pieces: on a 22 minute take 259 pieces took six and a half
# minutes and 423 took 16. It also stretches each piece to the longer of its
# picture and its sound, so the file ran about 6 ms over for every piece.
#
# Here one filter picks the picture's frames and one splits the sound, so the
# time is the same for 100 pieces as for 700. The sound is cut on the sample
# the edit names. Each frame recorded inside a kept piece shows at the moment
# nearest to where its sound plays, so picture and sound are never more than
# half a frame apart, no frame of what was removed is shown, and the file is
# as long as the edit says.

# When the edit opens late, ffmpeg starts reading this long before the first
# kept piece, and always reads this long past the last one.
READ_MARGIN = 1.0
# A frame rate outside 1 to this is a misread: FALLBACK_FPS stands in.
HIGHEST_FPS = 240.0
# 29.97 is 30000/1001: no frame rate in use needs a larger denominator.
FPS_DENOMINATOR = 1001
# A graph longer than this goes to ffmpeg in a file. One argument holds 128 KB on Linux.
GRAPH_INLINE_LIMIT = 100_000
# ffmpeg reads a graph file with "-/filter_complex" from version 7, and with
# "-filter_complex_script" before it.
GRAPH_FILE_FLAG_SINCE = 7
PROBE_FFMPEG_SECONDS = 20.0
# What a failed cut says it was doing, and what to check.
ENCODING = "encode"
CHECK_SOURCE = "Check that the source video plays and that ffmpeg is installed."
OLD_FFMPEG_NOTE = (
    "This ffmpeg has no asegment filter (it came with version 5), so the render ran the slow way "
    "and its time grows fast with the number of cuts. A newer ffmpeg makes it quick."
)


def frame_rate(fps: float | None) -> Fraction:
    """A probed frame rate as an exact fraction, so 29.97 stays 30000/1001."""
    if fps is None or not 1 <= fps <= HIGHEST_FPS:
        fps = FALLBACK_FPS
    return Fraction(fps).limit_denominator(FPS_DENOMINATOR)


def _exact(seconds: float) -> Fraction:
    """``seconds`` with the six decimals ffmpeg is given."""
    return Fraction(f"{seconds:.6f}")


def _clock(seconds: Fraction) -> str:
    return f"{float(seconds):.6f}"


@dataclass(frozen=True)
class Shown:
    """A run of source frames, and the frame of the result that the first of them becomes."""

    first: int
    count: int
    lands_on: int

    @property
    def moved(self) -> int:
        """How many frames earlier the run plays in the result than in the source."""
        return self.first - self.lands_on


@dataclass(frozen=True)
class Picture:
    """Which source frames the result shows, and how many frames long it is."""

    shown: list[Shown]
    frames: int


def plan_picture(kept: list[Span], fps: Fraction) -> Picture:
    """Place every frame recorded inside a kept piece where its sound plays.

    The sound is cut exactly, so the picture follows it. A kept piece moves
    earlier by what was removed before it, rounded to a whole frame: each of
    its frames then shows within half a frame of its sound. Only frames
    recorded inside a kept piece are shown. Where that rounding leaves a
    frame of the result empty, the frame before it is held. Where it sends
    two frames to one place, the earlier piece's frame stays.
    """
    shown: list[Shown] = []
    sound = Fraction(0)
    free = 0  # the first frame of the result not yet filled
    for start, end in kept:
        s, e = _exact(start), _exact(end)
        moved = round((s - sound) * fps)
        sound += e - s
        first, last = math.ceil(s * fps), math.ceil(e * fps) - 1
        first = max(first, free + moved)
        if first <= last:
            shown.append(Shown(first, last - first + 1, first - moved))
            free = last - moved + 1
    frames = round(sound * fps)
    inside = [
        Shown(run.first, min(run.count, frames - run.lands_on), run.lands_on)
        for run in shown if run.lands_on < frames
    ]
    return Picture(shown=inside, frames=frames)


def read_from(kept: list[Span], fps: Fraction) -> int:
    """The source frame ffmpeg starts reading at: ``READ_MARGIN`` before the first kept piece."""
    lead = _exact(kept[0][0]) - _exact(READ_MARGIN)
    return math.floor(lead * fps) if lead > 0 else 0


def picture_chain(picture: Picture, fps: Fraction, origin: int = 0, scale: str | None = None) -> str:
    """The filter chain that shows ``picture``. Pure.

    ``origin`` is the source frame ffmpeg starts reading at: it counts frames
    from there. After the first ``fps`` a frame's time stamp is its number, so
    every step below works in whole frames. ``select`` keeps the shown runs,
    ``setpts`` moves each run to its place, the second ``fps`` holds a frame
    over each empty place, and the last two steps end the picture on its
    last frame, held if need be.
    """
    rate = f"{fps.numerator}/{fps.denominator}"
    picks = "+".join(
        f"between(pts,{run.first - origin},{run.first + run.count - 1 - origin})" for run in picture.shown
    )
    moves = [str(picture.shown[0].moved - origin)]
    for before, run in zip(picture.shown, picture.shown[1:]):
        if run.moved != before.moved:
            moves.append(f"gte(PTS,{run.first - origin})*{run.moved - before.moved}")
    steps = [
        f"fps={rate}",
        f"select='{picks}'",
        f"setpts='PTS-({'+'.join(moves)})'",
        f"fps={rate}:start_time=0",
        "tpad=stop_mode=clone:stop=-1",
        f"trim=end_frame={picture.frames}",
    ]
    if scale:
        steps.append(scale)
    return f"[0:v]{','.join(steps)}[outv]"


def _join_fades(index: int, pieces: int, seconds: float, fade: float) -> str:
    """The fades on one kept piece of sound, by ClipForge's rule (render.build_filter_complex).

    In at every join, out at every join, none at the file's own start and end,
    and none on a piece too short to hold them.
    """
    fade_in, fade_out = index > 0, index < pieces - 1
    if fade <= 0 or seconds <= (2 if fade_in and fade_out else 1) * fade:
        return ""
    steps = [f",afade=t=in:st=0:d={fade:.3f}"] if fade_in else []
    if fade_out:
        steps.append(f",afade=t=out:st={seconds - fade:.3f}:d={fade:.3f}")
    return "".join(steps)


def sound_chain(kept: list[Span], origin: Fraction, fade: float = AUDIO_JOIN_FADE) -> str:
    """The filter chains that join the kept pieces of sound, cut on the sample. Pure.

    One asegment filter splits the sound at every edge and hands each stretch
    to an output of its own. The kept ones are faded and joined; the rest are
    dropped. The sound that comes out is ClipForge's, sample for sample.
    """
    edges: list[str] = []
    stretches: list[int | None] = []  # a kept piece's number, or None for a stretch that is dropped
    at = Fraction(0)
    for n, (start, end) in enumerate(kept):
        if _exact(start) - origin > at:
            edges.append(_clock(_exact(start) - origin))
            stretches.append(None)
        edges.append(_clock(_exact(end) - origin))
        stretches.append(n)
        at = _exact(end) - origin
    stretches.append(None)  # what follows the last kept piece
    chains = [f"[0:a]asegment=timestamps={'|'.join(edges)}" + "".join(f"[s{j}]" for j in range(len(stretches)))]
    for j, n in enumerate(stretches):
        if n is None:
            chains.append(f"[s{j}]anullsink")
        else:
            start, end = kept[n]
            chains.append(f"[s{j}]asetpts=PTS-STARTPTS{_join_fades(n, len(kept), end - start, fade)}[a{n}]")
    chains.append("".join(f"[a{n}]" for n in range(len(kept))) + f"concat=n={len(kept)}:v=0:a=1[outa]")
    return ";".join(chains)


@dataclass(frozen=True)
class Cut:
    """One cut, ready to run: what to read, the graph, and how long the result is."""

    source: Path
    out: Path
    graph: str
    read_at: float  # where in the source ffmpeg starts reading, in seconds
    read_seconds: float  # how much of the source it reads
    seconds: float  # the picture's length, which is the file's
    has_sound: bool
    crf: int


def plan_cut(plan: RenderPlan, *, fps: float | None, has_sound: bool) -> Cut:
    """The one-pass cut of ``plan``'s kept pieces. Pure.

    Raises StudioError when the kept pieces hold less than one frame.
    """
    rate = frame_rate(fps)
    picture = plan_picture(plan.kept, rate)
    if not picture.shown:
        raise StudioError("The edit keeps less than one frame of the video. Put something back, then render.")
    origin = read_from(plan.kept, rate)
    read_at = Fraction(origin) / rate
    chains = [picture_chain(picture, rate, origin, scale_filter_for(plan.source_height))]
    if has_sound:
        chains.append(sound_chain(plan.kept, read_at))
    return Cut(
        source=plan.source, out=plan.out_path, graph=";".join(chains), read_at=float(read_at),
        read_seconds=float(_exact(plan.kept[-1][1]) - read_at) + READ_MARGIN,
        seconds=float(Fraction(picture.frames) / rate), has_sound=has_sound, crf=plan.crf,
    )


def cut_command(cut: Cut, graph_args: list[str]) -> list[str]:
    """The ffmpeg argv of ``cut``. The encode is ClipForge's (render.build_ffmpeg_cmd), setting for setting."""
    seek = ["-ss", f"{cut.read_at:.6f}"] if cut.read_at else []
    sound = ["-map", "[outa]", *AUDIO_ARGS] if cut.has_sound else []
    return [
        "ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-nostats", "-progress", "pipe:1",
        *seek, "-t", f"{cut.read_seconds:.6f}", "-i", str(cut.source),
        *graph_args,
        "-map", "[outv]", *X264_ARGS, "-crf", str(cut.crf),
        *sound, "-movflags", "+faststart", str(cut.out),
    ]


@dataclass(frozen=True)
class Ffmpeg:
    """What the ffmpeg on this machine can do. ``version`` is None when it doesn't say."""

    version: int | None
    splits_sound: bool  # has the asegment filter, which the one pass needs

    @property
    def graph_file_flag(self) -> str:
        old = self.version is not None and self.version < GRAPH_FILE_FLAG_SINCE
        return "-filter_complex_script" if old else "-/filter_complex"


_ffmpeg: list[Ffmpeg] = []


def this_ffmpeg(run: Callable[..., Any] = subprocess.run) -> Ffmpeg:
    """Ask ffmpeg its version and filters, once. With no ffmpeg to ask, the answer is not kept."""
    if _ffmpeg:
        return _ffmpeg[0]
    try:
        asked = [
            run(["ffmpeg", "-hide_banner", flag], capture_output=True, text=True, timeout=PROBE_FFMPEG_SECONDS)
            for flag in ("-version", "-filters")
        ]
    except (OSError, subprocess.SubprocessError):
        return Ffmpeg(version=None, splits_sound=False)
    version = re.match(r"ffmpeg version n?(\d+)\.", asked[0].stdout or "")
    found = Ffmpeg(
        version=int(version.group(1)) if version else None,
        splits_sound=bool(re.search(r"^\s*\S+\s+asegment\s", asked[1].stdout or "", re.MULTILINE)),
    )
    _ffmpeg.append(found)
    return found


def _cut_in_one_pass(cut: Cut, handle: JobHandle, share: float, ffmpeg: Ffmpeg) -> None:
    """Run ``cut``. A long graph goes to ffmpeg in a file, which is gone when this returns."""
    done = [0.0]
    handle.track(lambda: share * done[0])
    with tempfile.TemporaryDirectory(prefix="lumr-cut-") as folder:
        graph_args = ["-filter_complex", cut.graph]
        if len(cut.graph) > GRAPH_INLINE_LIMIT:
            graph_file = Path(folder) / "graph.txt"
            graph_file.write_text(cut.graph, encoding="utf-8")
            graph_args = [ffmpeg.graph_file_flag, str(graph_file)]
        cut.out.parent.mkdir(parents=True, exist_ok=True)
        run_ffmpeg(
            cut_command(cut, graph_args), cut.out, total=cut.seconds,
            on_progress=lambda f: done.__setitem__(0, f), doing=ENCODING, check=CHECK_SOURCE,
        )


def _cut_the_slow_way(plan: RenderPlan, handle: JobHandle, job_label: str, share: float) -> None:
    """ClipForge's own encode of ``plan``, for an ffmpeg too old for the one pass.

    It reads from the first kept piece on, so a preview late in the video
    doesn't decode everything before it.
    """
    seek = plan.kept[0][0]
    moved = replace(plan, kept=[(s - seek, e - seek) for s, e in plan.kept])

    def seeking_runner(cmd: list[str], **kwargs: Any) -> subprocess.Popen:
        return subprocess.Popen(with_input_seek(cmd, seek) if seek > 0 else cmd, **kwargs)

    progress = RenderProgress(job_id=job_label)
    handle.track(lambda: share * progress.progress)
    run_render(moved, progress, runner=seeking_runner)
    if progress.status != "done":
        raise StudioError(
            f"ffmpeg could not {ENCODING} {plan.out_path.name}: {(progress.error or '')[-600:]} {CHECK_SOURCE}"
        )


def cut_edit(plan: RenderPlan, handle: JobHandle, job_label: str, share: float = 1.0) -> str | None:
    """Cut ``plan``'s kept pieces out of its source into ``plan.out_path``.

    ``share`` is the part of the job's progress the cut fills, from 0.
    Returns a note for Claude when it ran the slow way, else None.
    Raises StudioError when ffmpeg fails.
    """
    ffmpeg = this_ffmpeg()
    if not ffmpeg.splits_sound:
        _cut_the_slow_way(plan, handle, job_label, share)
        return OLD_FFMPEG_NOTE
    source = probe_media(plan.source)
    _cut_in_one_pass(plan_cut(plan, fps=source.fps, has_sound=source.has_sound), handle, share, ffmpeg)
    return None


# ── Overlays: the optional second pass ────────────────────────────────────────

# With overlays the job has two passes, and each fills the share of the
# progress that it takes of the time. Measured on a 22 minute take at 1280x720,
# against the cost of encoding one second in the cut: reading a second of the
# source costs READ_COST, and drawing over a second and encoding it again
# costs DRAW_COST.
READ_COST = 0.12
DRAW_COST = 1.05


def first_pass_share(kept: list[Span]) -> float:
    """The share of a two-pass job's time that the cut takes, from what it reads and what it keeps."""
    keeps = sum(end - start for start, end in kept)
    cut = READ_COST * (kept[-1][1] - kept[0][0]) + keeps
    return cut / (cut + DRAW_COST * keeps)


def _saved_overlays(project: Project) -> list[dict[str, Any]]:
    """The creator's saved overlays, each file checked again. Empty when there are none."""
    overlays = load_overlays(project)["overlays"]
    if overlays:
        recheck_files(overlays)
    return overlays


def _first_pass_target(project: Project, out: Path, overlays: list[dict[str, Any]]) -> Path:
    """Where the edit's encode writes: straight to ``out``, or to ``work/`` when overlays follow."""
    if not overlays:
        return out
    return reserve_unique_path(work_dir(project), f"{out.stem}-before-overlays", ".mp4")


def _draw_overlays(
    base: Path, out: Path, items: list[Any], crf: int, handle: JobHandle, first_share: float
) -> dict[str, Any]:
    """The second pass: ``items`` drawn over ``base`` into ``out``, filling the progress after ``first_share``."""
    if not items:
        base.replace(out)  # nothing lands in this stretch: the first pass is the result
        return {"drawn": [], "clip_sound_mixed": False}
    done = [0.0]
    handle.track(lambda: first_share + (1 - first_share) * done[0])
    return lay_over(base, out, items, crf=crf, on_progress=lambda f: done.__setitem__(0, f))


def _overlay_report(
    placements: list[Placement], drawn: dict[str, Any], passes: list[float]
) -> dict[str, Any]:
    """What happened to the overlays, for the job result."""
    return {
        **drawn,
        "not_shown": [p.overlay["id"] for p in placements if not p.shown],
        "notes": [note for p in placements for note in p.notes],
        "edit_pass_seconds": round(passes[0], 1),
        "overlay_pass_seconds": round(passes[1], 1),
    }


# ── Full render and preview ───────────────────────────────────────────────────


def render_full(
    project: Project,
    removed: list[tuple[float, float]],
    words: list[dict[str, Any]],
    handle: JobHandle,
) -> dict[str, Any]:
    """Render the whole edited video into ``exports/`` and describe the result.

    ``removed`` is the saved, already finalized edit, so it renders verbatim.
    Saved overlays are drawn in a second pass; the result then carries ``overlays``.
    """
    overlays = _saved_overlays(project)
    out = reserve_unique_path(project.exports_dir, f"{project.video.stem}-edit-{_stamp()}", ".mp4")
    target = _first_pass_target(project, out, overlays)
    try:
        plan = build_render_plan(
            video_path=project.video,
            words_path=project.words_path if project.words_path.exists() else None,
            removed=[],
            user_removed=removed,
            words=words,
            out_dir=target.parent,
            out_name=target.name,
        )
        placements = place_on_edit(overlays, plan.kept, words) if overlays else []
        items = items_for_render(placements)
        share = first_pass_share(plan.kept) if items else 1.0
        started = time.monotonic()
        note = cut_edit(plan, handle, out.stem, share)
        if overlays:
            passes = [time.monotonic() - started]
            drawn = _draw_overlays(target, out, items, plan.crf, handle, share)
            passes.append(time.monotonic() - started - passes[0])
    except BaseException:
        out.unlink(missing_ok=True)
        raise
    finally:
        if target != out:
            target.unlink(missing_ok=True)
    result = {
        "output_path": str(out),
        "duration": round(probe_duration(out), 3),
        "expected_duration": round(plan.kept_duration, 3),
        "removed_seconds": round(plan.removed_duration, 3),
        "kept_segments": len(plan.kept),
        "resolution": plan.out_resolution,
        "source_note": plan.source_note,
    }
    if note:
        result["note"] = note
    if overlays:
        result["overlays"] = _overlay_report(placements, drawn, passes)
    return result


# ── Starting a full render, and what the page reads of it ────────────────────

# The job kind of a full render, whoever started it.
RENDER_KIND = "render"

# What ``export_status`` answers in ``state``.
IDLE, RUNNING, DONE, FAILED = "idle", "running", "done", "failed"

NOTHING_EXPORTED = "Nothing exported yet."
# The job's own error is written for Claude and can hold ffmpeg's output. The
# creator gets what to do next; Claude reads the rest with job_status.
EXPORT_FAILED = (
    "The export did not finish. Press Export video to try again. "
    "If it stops again, ask Claude what went wrong."
)

# The work a full render does: ``render_full``, or a stand-in in tests.
RenderWork = Callable[[Project, list[tuple[float, float]], list[dict[str, Any]], JobHandle], dict[str, Any]]


def start_full_render(
    project: Project,
    *,
    jobs: JobRegistry,
    duration: float | None = None,
    render: RenderWork | None = None,
) -> tuple[str, bool]:
    """Start the full render of the saved edit, unless one is already running for this video.

    Returns ``(job_id, started)``. When a render of this video is running,
    whoever started it, nothing new starts and its id comes back with False.
    The job remembers what its edit removes (``about.edit_mark``).
    Raises StudioError when the saved edit can't be read.
    """
    work = render or render_full
    edit = edits.load_edit(project, project.duration() if duration is None else duration)
    removed = edits.removed_spans(edit)
    words = words_if_any(project)
    return jobs.start_one(
        RENDER_KIND, project, lambda handle: work(project, removed, words, handle),
        about={"edit_mark": edit_mark(edit)},
    )


def edit_mark(edit: dict[str, Any]) -> str:
    """A short mark of what ``edit`` removes. Two edits with the same mark render the same cuts.

    Read from the removed spans and not from the save time: the save time
    counts whole seconds, so two saves in one second would look the same, and
    a change the creator made and then undid would look like a new edit.
    """
    spans = [[round(start, 3), round(end, 3)] for start, end in edits.removed_spans(edit)]
    return hashlib.sha1(json.dumps(spans).encode("utf-8")).hexdigest()[:16]


def holds_earlier_edit(job: Job, edit: dict[str, Any]) -> bool:
    """Whether ``edit`` removes something else than the edit ``job`` renders.

    A render started with no mark is never called behind: nothing says it is.
    """
    now = edit_mark(edit)
    return job.about.get("edit_mark", now) != now


def nothing_exported() -> dict[str, Any]:
    """The export status of a video with no render to show."""
    return {
        "state": IDLE, "progress": None, "file": None, "folder": None,
        "message": NOTHING_EXPORTED, "holds_earlier_edit": False,
    }


def export_status(project: Project, jobs: JobRegistry, *, edit: dict[str, Any]) -> dict[str, Any]:
    """The latest full render of ``project`` as the treatment page shows it.

    ``{state, progress, file, folder, message, holds_earlier_edit}``. ``state``
    is ``idle`` when no render has run since the server started. ``file`` and
    ``folder`` are set once it is done. ``message`` is a sentence the creator
    can read; a failed one says what to do next. ``holds_earlier_edit`` is
    true when ``edit``, the edit as saved now, removes something else than the
    edit the render started with.
    """
    job = jobs.latest(RENDER_KIND, project)
    status = nothing_exported()
    if job is None:
        return status
    seen = job.to_dict()
    status["holds_earlier_edit"] = holds_earlier_edit(job, edit)
    if seen["status"] == DONE and isinstance(seen["result"], dict) and seen["result"].get("output_path"):
        out = Path(str(seen["result"]["output_path"]))
        status.update(state=DONE, progress=1.0, file=out.name, folder=str(out.parent), message=f"Exported {out.name}.")
    elif seen["status"] == RUNNING:
        progress = seen["progress"]
        done = "" if progress is None else f" {round(progress * 100)}% done."
        status.update(state=RUNNING, progress=progress, message=f"Exporting.{done}")
    else:
        status.update(state=FAILED, message=EXPORT_FAILED)
    return status


def with_input_seek(cmd: list[str], seek: float) -> list[str]:
    """Insert ``-ss <seek>`` before ffmpeg's ``-i`` so decoding starts near the window.

    Input seeking resets timestamps to 0 at ``seek``, so segment times in the
    filter graph must be relative to ``seek``.
    """
    i = cmd.index("-i")
    return cmd[:i] + ["-ss", f"{seek:.3f}"] + cmd[i:]


def preview_window(
    removed: list[tuple[float, float]], duration: float, at: float, pad: float
) -> dict[str, Any]:
    """Which source spans a preview around source time ``at`` plays.

    The window is ``pad`` seconds of EDITED time either side of where ``at``
    lands in the edited video. Raises StudioError for a bad ``at`` or ``pad``.
    """
    if not 0 <= at <= duration:
        raise StudioError(f"at {at} is outside the video (0 to {duration:.2f}). Pass a source time inside it.")
    if not 0 < pad <= MAX_PREVIEW_PAD:
        raise StudioError(f"pad {pad} must be above 0 and at most {MAX_PREVIEW_PAD} seconds.")
    kept = kept_segments(removed, duration)
    edited_at, in_cut = to_edited_time(at, kept)
    total = edited_duration(kept)
    w0, w1 = max(0.0, edited_at - pad), min(total, edited_at + pad)
    spans = source_spans_for_window(kept, w0, w1)
    if not spans:
        raise StudioError("Nothing plays around that time in the edited video. Pick another time.")
    return {"edited_at": edited_at, "in_cut": in_cut, "window": (w0, w1), "spans": spans}


def render_preview(
    project: Project,
    removed: list[tuple[float, float]],
    duration: float,
    at: float,
    pad: float,
    handle: JobHandle,
) -> dict[str, Any]:
    """Render a short clip of the edited result around source time ``at``.

    Saved overlays that show in the clip are drawn in a second pass; the
    result then carries ``overlays``.
    """
    window = preview_window(removed, duration, at, pad)
    spans = window["spans"]
    overlays = _saved_overlays(project)
    out = reserve_unique_path(
        project.exports_dir, f"{project.video.stem}-preview-{at:.1f}s-{_stamp()}", ".mp4"
    )
    target = _first_pass_target(project, out, overlays)
    plan = RenderPlan(
        source=project.video,
        out_path=target,
        kept=spans,
        removed=[],
        duration=duration,
        out_height=0,
        source_width=0,
        source_height=0,
        source_note="preview",
        crf=PREVIEW_CRF,
    )

    w0, w1 = window["window"]
    try:
        placements = (
            place_on_edit(overlays, kept_segments(removed, duration), words_if_any(project)) if overlays else []
        )
        items = items_for_window(placements, w0, w1)
        share = first_pass_share(spans) if items else 1.0
        started = time.monotonic()
        note = cut_edit(plan, handle, out.stem, share)
        if overlays:
            passes = [time.monotonic() - started]
            drawn = _draw_overlays(target, out, items, PREVIEW_CRF, handle, share)
            passes.append(time.monotonic() - started - passes[0])
    except BaseException:
        out.unlink(missing_ok=True)
        raise
    finally:
        if target != out:
            target.unlink(missing_ok=True)
    result = {
        "output_path": str(out),
        "duration": round(probe_duration(out), 3),
        "at": at,
        "edited_at": round(window["edited_at"], 3),
        "at_is_inside_a_cut": window["in_cut"],
        "edited_window": [round(w0, 3), round(w1, 3)],
        "source_spans": [[round(s, 3), round(e, 3)] for s, e in spans],
    }
    if note:
        result["note"] = note
    if overlays:
        in_window = {i.overlay["id"] for i in items}
        report = _overlay_report([p for p in placements if p.overlay["id"] in in_window], drawn, passes)
        result["overlays"] = report
    return result
