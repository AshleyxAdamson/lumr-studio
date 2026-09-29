"""Lay the creator's photos and clips over a rendered video with one ffmpeg pass.

The existing render makes the edited video. This pass reads that file and
draws every overlay on it with one filter graph, so ClipForge's render stays
exactly as it is. The cost is a second encode; the render result reports the
time each pass took.

Times here are in the BASE file's own seconds: the edited video for a full
render, the preview clip for a preview. ``overlays.place_on_edit`` has already
turned source anchors into edited time; this module only draws.

The same graph draws the stills that set_overlays returns, one frame at a time
from the source video, so the picture Claude checks is drawn by the code that
renders.

ffmpeg's warnings go to a file, never a pipe: a full stderr pipe froze a
render once, and the real footage prints "Late SEI" warnings thousands of times.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lumr_studio.edit import clock
from lumr_studio.errors import StudioError
from lumr_studio.overlays import FULL, IMAGE, VIDEO, Media, Placement, probe_media

# Fade in and out, seconds. Longer reads as a transition, which is out of scope.
FADE_SECONDS = 0.3
# A fade takes at most this share of an overlay's time on screen.
FADE_MAX_SHARE = 0.25
# How far a drifting photo zooms over its whole time on screen: 8% is felt,
# not noticed.
ZOOM_AMOUNT = 0.08
# The photo is scaled up this much before zooming, so each frame's crop moves
# by less than a pixel of the output and the drift doesn't shake.
ZOOM_SUPERSAMPLE = 2
# Gap between a box and the frame's edge, as a share of the frame's short side.
BOX_MARGIN_SHARE = 0.04
# A thin white edge around a box, as a share of the frame's short side (4
# pixels at 720p), so a clip of the same room still reads as a separate picture.
BOX_BORDER_SHARE = 0.005
BOX_BORDER_COLOR = "white"
# The blurred backdrop behind a photo shown whole: blur radius as a share of
# the frame's short side, and how much darker than the photo it is.
BACKDROP_BLUR_SHARE = 0.03
BACKDROP_DARKEN = -0.12
# Same video settings as ClipForge's export (render.build_ffmpeg_cmd), so a
# render with overlays looks like one without.
X264_ARGS = ("-c:v", "libx264", "-preset", "medium", "-profile:v", "high", "-pix_fmt", "yuv420p")
AUDIO_ARGS = ("-c:a", "aac", "-b:a", "192k")
# Frame rate used when a file doesn't say its own.
FALLBACK_FPS = 25.0
# Stills in one picture from set_overlays, and the size of each.
MAX_STILLS = 6
STILL_WIDTH = 640
STILL_LABEL_HEIGHT = 30
STILL_COLUMNS = 2
STILL_JPEG_QUALITY = 85
STILL_TIMEOUT_SECONDS = 60.0
# Characters of ffmpeg's error output quoted when it fails.
ERROR_TAIL_CHARS = 600


@dataclass(frozen=True)
class Item:
    """One overlay to draw into one output file.

    ``at``: where it appears, in the base file's seconds. ``seconds``: how long
    it shows in this file. ``skip``: how much of it already played before this
    file starts (a preview that opens mid-overlay). ``total``: its whole time on
    screen in the edited video, which paces the zoom and the fade out.
    """

    overlay: dict[str, Any]
    at: float
    seconds: float
    skip: float = 0.0
    total: float = 0.0

    @property
    def total_seconds(self) -> float:
        return self.total or self.seconds


def items_for_render(placements: list[Placement]) -> list[Item]:
    """The overlays a full render draws, in drawing order."""
    return [Item(p.overlay, p.at, p.seconds, 0.0, p.seconds) for p in placements if p.shown]


def items_for_window(placements: list[Placement], w0: float, w1: float) -> list[Item]:
    """The overlays that show in the edited window ``[w0, w1]``, timed from ``w0``."""
    items = []
    for p in placements:
        if not p.shown:
            continue
        lo, hi = max(p.at, w0), min(p.at + p.seconds, w1)
        if hi - lo > 1e-3:
            items.append(Item(p.overlay, lo - w0, hi - lo, lo - p.at, p.seconds))
    return items


# ── Sizes and positions ───────────────────────────────────────────────────────


def _even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def fit_inside(width: int, height: int, box_w: int, box_h: int) -> tuple[int, int]:
    """The largest even size with the shape of ``width x height`` that fits the box."""
    scale = min(box_w / width, box_h / height)
    return min(_even(width * scale), box_w), min(_even(height * scale), box_h)


def box_border(frame: tuple[int, int]) -> int:
    """Width of the white edge around a box in this frame, in pixels. Even, for yuv420p."""
    return _even(BOX_BORDER_SHARE * min(frame))


def picture_size(overlay: dict[str, Any], frame: tuple[int, int]) -> tuple[int, int]:
    """The size of the picture itself: the whole frame, or fitted inside its box's edge."""
    w, h = frame
    if overlay["place"] == FULL:
        return w, h
    size, edge = float(overlay["size"]), 2 * box_border(frame)
    media = overlay["media"]
    return fit_inside(int(media["width"]), int(media["height"]), _even(w * size) - edge, _even(h * size) - edge)


def overlay_size(overlay: dict[str, Any], frame: tuple[int, int]) -> tuple[int, int]:
    """The drawn size of an overlay in a frame: the whole frame, or its box picture and edge."""
    pw, ph = picture_size(overlay, frame)
    if overlay["place"] == FULL:
        return pw, ph
    edge = 2 * box_border(frame)
    return pw + edge, ph + edge


def overlay_position(overlay: dict[str, Any], frame: tuple[int, int]) -> tuple[int, int]:
    """Top-left corner of the overlay in the frame, from its named place."""
    w, h = frame
    ow, oh = overlay_size(overlay, frame)
    margin = round(BOX_MARGIN_SHARE * min(w, h))
    place = overlay["place"]
    if place == FULL:
        return 0, 0
    if place == "center":
        return (w - ow) // 2, (h - oh) // 2
    x = margin if place.endswith("left") else w - margin - ow
    y = margin if place.startswith("top") else h - margin - oh
    return x, y


# ── The filter graph ──────────────────────────────────────────────────────────


@dataclass
class Graph:
    """ffmpeg inputs after the base, the filter graph, and its output labels."""

    inputs: list[str]
    filter: str
    video: str
    audio: str | None  # None: the base audio passes through untouched


def _compose(overlay: dict[str, Any], frame: tuple[int, int], src: str, k: int) -> tuple[list[str], str]:
    """Chains that shape one overlay's picture to its drawn size. Returns (chains, label)."""
    w, h = frame
    pw, ph = picture_size(overlay, frame)
    out = f"shaped{k}"
    if overlay["place"] != FULL:
        return [f"[{src}]scale={pw}:{ph},setsar=1[{out}]"], out
    cover = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
    if overlay["fit"] == "fill":
        return [f"[{src}]{cover},setsar=1[{out}]"], out
    blur = max(2, round(BACKDROP_BLUR_SHARE * min(w, h)))
    return [
        f"[{src}]split[bgsrc{k}][fgsrc{k}]",
        f"[bgsrc{k}]{cover},boxblur={blur}:2,eq=brightness={BACKDROP_DARKEN}[bg{k}]",
        f"[fgsrc{k}]scale={w}:{h}:force_original_aspect_ratio=decrease[fg{k}]",
        f"[bg{k}][fg{k}]overlay=x=(W-w)/2:y=(H-h)/2,setsar=1[{out}]",
    ], out


def _fades(item: Item, audio: bool = False) -> list[str]:
    """Fade in at the overlay's first moment and out at its last, if they fall in this file."""
    if not item.overlay.get("fade", True):
        return []
    fade = min(FADE_SECONDS, FADE_MAX_SHARE * item.total_seconds)
    name, extra = ("afade", "") if audio else ("fade", ":alpha=1")
    parts = []
    if item.skip < 1e-3:
        parts.append(f"{name}=t=in:st=0:d={fade:.3f}{extra}")
    out_at = item.total_seconds - fade - item.skip
    if out_at < item.seconds:
        parts.append(f"{name}=t=out:st={max(0.0, out_at):.3f}:d={fade:.3f}{extra}")
    return parts


def _zoom(item: Item, size: tuple[int, int], fps: float) -> str:
    """zoompan that drifts the photo in or out across its whole time on screen."""
    ow, oh = size
    frames = max(1, math.ceil(item.seconds * fps))
    total_frames = max(2, math.ceil(item.total_seconds * fps))
    first = round(item.skip * fps)
    progress = f"(on+{first})/{total_frames - 1}"
    z = (
        f"1+{ZOOM_AMOUNT}*{progress}" if item.overlay["motion"] == "zoom_in"
        else f"1+{ZOOM_AMOUNT}*(1-{progress})"
    )
    big = f"scale={ow * ZOOM_SUPERSAMPLE}:{oh * ZOOM_SUPERSAMPLE}"
    return (
        f"{big},zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        f":d={frames}:s={ow}x{oh}:fps={fps:g}"
    )


def _edge(overlay: dict[str, Any], frame: tuple[int, int]) -> list[str]:
    """The white edge around a box, added after any zoom so the edge holds still."""
    if overlay["place"] == FULL:
        return []
    b = box_border(frame)
    ow, oh = overlay_size(overlay, frame)
    return [f"pad={ow}:{oh}:{b}:{b}:color={BOX_BORDER_COLOR}"]


def _finish(item: Item, frame: tuple[int, int]) -> list[str]:
    """Steps every overlay ends with: its edge, see-through for fades, the fades, its start time."""
    return [
        *_edge(item.overlay, frame), "format=yuva420p", *_fades(item),
        f"setpts=PTS-STARTPTS+{item.at:.3f}/TB",
    ]


def _image_chain(item: Item, k: int, frame: tuple[int, int], fps: float) -> list[str]:
    chains, shaped = _compose(item.overlay, frame, f"{k}:v", k)
    frames = max(1, math.ceil(item.seconds * fps))
    if item.overlay["motion"] == "still":
        motion = f"loop=loop={frames - 1}:size=1:start=0,setpts=N/{fps:g}/TB"
    else:
        motion = _zoom(item, picture_size(item.overlay, frame), fps)
    chains.append(f"[{shaped}]" + ",".join([motion, *_finish(item, frame)]) + f"[ov{k}]")
    return chains


def _video_chain(item: Item, k: int, frame: tuple[int, int], fps: float) -> list[str]:
    chains = [f"[{k}:v]fps={fps:g},setpts=PTS-STARTPTS[raw{k}]"]
    more, shaped = _compose(item.overlay, frame, f"raw{k}", k)
    chains += more
    chains.append(f"[{shaped}]" + ",".join(_finish(item, frame)) + f"[ov{k}]")
    return chains


def _audio_chain(item: Item, k: int) -> str:
    delay = round(item.at * 1000)
    steps = [
        "asetpts=PTS-STARTPTS",
        *_fades(item, audio=True),
        f"volume={float(item.overlay['volume']):g}",
        f"adelay={delay}:all=1",
    ]
    return f"[{k}:a]" + ",".join(steps) + f"[oa{k}]"


def _input_args(item: Item) -> list[str]:
    o = item.overlay
    if o["kind"] == IMAGE:
        return ["-i", o["file"]]
    seek = float(o["clip_in"]) + item.skip
    return ["-ss", f"{seek:.3f}", "-t", f"{item.seconds:.3f}", "-i", o["file"]]


def has_sound(item: Item) -> bool:
    return item.overlay["kind"] == VIDEO and float(item.overlay.get("volume") or 0) > 0


def build_graph(
    items: list[Item],
    frame: tuple[int, int],
    fps: float,
    *,
    first_input: int = 1,
    base_video: str = "0:v",
    base_audio: str | None = "0:a",
) -> Graph:
    """The inputs and filter graph that draw ``items`` over the base video. Pure.

    ``items`` are drawn in order, so a later one covers an earlier one: pass
    them bottom layer first. Clip sound is mixed under the base audio at the
    clip's volume; with no clip sound the graph leaves audio alone. Pass
    ``base_audio=None`` for a picture with no sound at all.
    """
    inputs: list[str] = []
    chains: list[str] = []
    mixes: list[str] = []
    current = base_video
    for n, item in enumerate(items):
        k = first_input + n
        inputs += _input_args(item)
        chain = _image_chain if item.overlay["kind"] == IMAGE else _video_chain
        chains += chain(item, k, frame, fps)
        x, y = overlay_position(item.overlay, frame)
        end = item.at + item.seconds
        label = f"layered{n}"
        chains.append(
            f"[{current}][ov{k}]overlay=x={x}:y={y}:eof_action=pass"
            f":enable='between(t,{item.at:.3f},{end:.3f})'[{label}]"
        )
        current = label
        if base_audio and has_sound(item):
            chains.append(_audio_chain(item, k))
            mixes.append(f"[oa{k}]")
    chains.append(f"[{current}]format=yuv420p[vout]")
    audio = None
    if mixes:
        chains.append(
            f"[{base_audio}]{''.join(mixes)}amix=inputs={len(mixes) + 1}:duration=first"
            ":normalize=0:dropout_transition=0[aout]"
        )
        audio = "aout"
    return Graph(inputs=inputs, filter=";".join(chains), video="vout", audio=audio)


def overlay_command(base: Path, out: Path, graph: Graph, *, crf: int) -> list[str]:
    """The ffmpeg argv for the pass. Audio is copied as is unless a clip's sound is mixed in."""
    audio = ["-map", f"[{graph.audio}]", *AUDIO_ARGS] if graph.audio else ["-map", "0:a?", "-c:a", "copy"]
    return [
        "ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-nostats", "-progress", "pipe:1",
        "-i", str(base), *graph.inputs,
        "-filter_complex", graph.filter,
        "-map", f"[{graph.video}]", *X264_ARGS, "-crf", str(crf),
        *audio, "-movflags", "+faststart", str(out),
    ]


# ── Running ffmpeg ────────────────────────────────────────────────────────────


def _progress_seconds(line: str) -> float | None:
    key, _, value = line.strip().partition("=")
    if key in ("out_time_us", "out_time_ms"):
        try:
            return float(value) / 1e6
        except ValueError:
            return None
    return None


# What a failed pass says it was doing, and what to check. The defaults are this pass's.
DRAWING = "draw the overlays into"
CHECK_OVERLAY_FILES = "Check that every overlay file opens in a video player."


def run_ffmpeg(
    cmd: list[str],
    out: Path,
    *,
    total: float,
    on_progress: Callable[[float], None] = lambda _f: None,
    runner: Callable[..., Any] = subprocess.Popen,
    doing: str = DRAWING,
    check: str = CHECK_OVERLAY_FILES,
) -> None:
    """Run one ffmpeg command that writes ``out`` and reports ``-progress`` on stdout.

    stderr goes to a temporary file, so it can never fill a pipe. Raises
    StudioError with the end of ffmpeg's error output when it fails, and
    removes a half-written ``out``. ``doing`` and ``check`` word that error
    for the pass that ran: the render's cut runs through here too.
    """
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as errors:
        try:
            proc = runner(cmd, stdout=subprocess.PIPE, stderr=errors, stdin=subprocess.DEVNULL, text=True)
        except FileNotFoundError:
            raise StudioError("ffmpeg is not installed or not on the PATH. Install ffmpeg and try again.") from None
        for line in proc.stdout:
            seconds = _progress_seconds(line)
            if seconds is not None and total > 0:
                on_progress(max(0.0, min(1.0, seconds / total)))
        proc.wait()
        errors.seek(0)
        tail = errors.read()[-ERROR_TAIL_CHARS:].strip()
    if proc.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        out.unlink(missing_ok=True)
        raise StudioError(f"ffmpeg could not {doing} {out.name} (exit {proc.returncode}): {tail} {check}")
    on_progress(1.0)


def frame_facts(media: Media) -> tuple[tuple[int, int], float]:
    """A video's frame size and frame rate, for sizing the graph."""
    return (media.width, media.height), media.fps or FALLBACK_FPS


def lay_over(
    base: Path,
    out: Path,
    items: list[Item],
    *,
    crf: int,
    on_progress: Callable[[float], None] = lambda _f: None,
    runner: Callable[..., Any] = subprocess.Popen,
    probe: Callable[[Path], Media] = probe_media,
) -> dict[str, Any]:
    """Draw ``items`` over ``base`` into ``out``. Raises StudioError when ffmpeg fails."""
    media = probe(base)
    frame, fps = frame_facts(media)
    graph = build_graph(items, frame, fps, base_audio="0:a" if media.has_sound else None)
    run_ffmpeg(
        overlay_command(base, out, graph, crf=crf), out,
        total=float(media.duration or 0), on_progress=on_progress, runner=runner,
    )
    return {"drawn": [i.overlay["id"] for i in items], "clip_sound_mixed": graph.audio is not None}


# ── Stills for Claude to check ────────────────────────────────────────────────


def _on_screen_at(moment: float, placements: list[Placement], fps: float) -> list[Item]:
    """Every overlay showing at edited time ``moment``, as items for a two-frame file."""
    return [
        Item(p.overlay, 0.0, min(p.at + p.seconds - moment, 2 / fps), moment - p.at, p.seconds)
        for p in placements
        if p.shown and p.at <= moment < p.at + p.seconds
    ]


def still_command(
    video: Path, source_at: float, items: list[Item], frame: tuple[int, int], fps: float, out: Path
) -> list[str]:
    """ffmpeg argv for one frame of the source at ``source_at`` with ``items`` drawn on it."""
    graph = build_graph(items, frame, fps, base_audio=None)
    return [
        "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
        "-ss", f"{source_at:.3f}", "-i", str(video), *graph.inputs,
        "-filter_complex", graph.filter, "-map", f"[{graph.video}]",
        "-frames:v", "1", "-fps_mode", "passthrough", str(out),
    ]


def _label(p: Placement) -> str:
    o = p.overlay
    name = Path(o["file"]).name
    text = f"{o['id']}  {clock(o['start'])}  {o['place']}  {name}"
    return text if p.status == "shown" else f"{text}  ({p.status})"


def draw_stills(
    video: Path,
    placements: list[Placement],
    source_at: Callable[[float], float],
    out: Path,
    *,
    run: Callable[..., Any] = subprocess.run,
    probe: Callable[[Path], Media] = probe_media,
) -> dict[str, Any]:
    """One picture: a still of each shown overlay at the middle of its time on screen.

    ``source_at`` maps an edited time to the source time playing then. Each
    still is the source frame with every overlay on screen at that moment
    drawn on it, in drawing order, so a box over a photo shows as the viewer
    sees it. At most ``MAX_STILLS``; the result says how many were left out.
    """
    from PIL import Image, ImageDraw, ImageFont

    shown = [p for p in placements if p.shown]
    chosen = sorted(shown, key=lambda p: p.at)[:MAX_STILLS]
    frame, fps = frame_facts(probe(video))
    tiles: list[tuple[Placement, Image.Image]] = []
    with tempfile.TemporaryDirectory(prefix="lumr-stills-") as tmp:
        for n, p in enumerate(chosen):
            moment = p.at + p.seconds / 2
            items = _on_screen_at(moment, shown, fps)
            png = Path(tmp) / f"still{n}.png"
            done = run(
                still_command(video, source_at(moment), items, frame, fps, png),
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                timeout=STILL_TIMEOUT_SECONDS,
            )
            if done.returncode != 0 or not png.exists():
                raise StudioError(
                    f"ffmpeg could not draw a still of {p.overlay['id']}: {(done.stderr or '')[-ERROR_TAIL_CHARS:]} "
                    "Check that its file opens in a video player."
                )
            with Image.open(png) as img:
                scale = STILL_WIDTH / img.width
                tiles.append((p, img.convert("RGB").resize((STILL_WIDTH, round(img.height * scale)))))
    if not tiles:
        return {"path": None, "shows": [], "left_out": 0}
    tile_h = max(t.height for _, t in tiles) + STILL_LABEL_HEIGHT
    cols = min(STILL_COLUMNS, len(tiles))
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * STILL_WIDTH, rows * tile_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)
    for n, (p, tile) in enumerate(tiles):
        x, y = (n % cols) * STILL_WIDTH, (n // cols) * tile_h
        sheet.paste(tile, (x, y))
        draw.text((x + 8, y + tile.height + 6), _label(p), fill="black", font=font)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, "JPEG", quality=STILL_JPEG_QUALITY)
    return {
        "path": str(out),
        "shows": [p.overlay["id"] for p, _ in tiles],
        "left_out": len(shown) - len(tiles),
    }
