"""A picture of one join, so Claude can check a cut before the creator hears it.

Claude edits video it never sees. This module draws one PNG of the join
nearest a source time, laid out the way the edited video plays it, so a single
look answers three questions: does the picture jump at the join, does the sound
run through it, and which words now sit next to each other.

Top to bottom the picture holds five bands:

1. Seam pair: the last kept frame before the cut and the first kept frame after.
2. Frame strip: small stills across the window, in edited order.
3. Sound wave: the edited audio for the same window, on a loudness scale.
4. Words: kept words placed on the same time axis, sounds and laughs as blocks.
5. Removed: what the cut took out, and how the join now reads.

Bands 2 to 4 share one time axis with the join at its centre, marked by one
vertical line at the same x in every band.

Frames and audio come from ffmpeg with input seeking (``-ss`` before ``-i``),
one short decode per kept span, run in parallel. ffmpeg writes into a temp
folder that is removed afterwards, and its stderr goes to a log file there, so
a chatty stream (the "Late SEI" warnings some cameras cause) can never fill a
pipe and stall the call.
"""

from __future__ import annotations

import io
import math
import subprocess
import tempfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lumr_studio.engine.render import kept_segments
from lumr_studio.errors import StudioError
from lumr_studio.timeline import edited_duration, source_spans_for_window, to_edited_time
from lumr_studio.transcript import is_event, word_is_cut

# ── Limits ────────────────────────────────────────────────────────────────────

# More than this either side and fast speech stacks into staggered rows that
# no longer read in order (checked on real footage at 8 and 10 seconds).
MAX_SPAN_SECONDS = 6.0
# Image limits from the tool contract: the long edge the model reads without
# downscaling, and a file size that keeps the tool result cheap.
MAX_LONG_EDGE = 1568
MAX_FILE_BYTES = 1_000_000
# A decode of a few seconds takes a fraction of a second; a minute means ffmpeg hung.
FFMPEG_TIMEOUT_SECONDS = 60.0
# ffmpeg's JPEG quality for decoded stills (2 is near lossless); JPEG keeps a
# ten second span to a few MB of temp files where raw frames would take 100MB.
FRAME_JPEG_QUALITY = 2
# Mono at 8 kHz is plenty to draw speech and keeps the decode small.
AUDIO_RATE = 8000
# How much audio either side of the join the level readout averages.
JOIN_LEVEL_SECONDS = 0.08
# The loudness scale spans this many dB below the window's loud peaks, so
# quiet trailing sounds still show and true silence sits on the baseline.
WAVE_RANGE_DB = 40.0
# A window quieter than this is drawn against this level, so silence stays flat.
WAVE_QUIETEST_REF_DB = -45.0
# Two cut edges closer than this are the same point in time.
TIME_EPS = 1e-6

# ── Layout ────────────────────────────────────────────────────────────────────

CANVAS_WIDTH = 1400
MARGIN = 16
# The time axis runs from AXIS_LEFT to AXIS_RIGHT; the join sits at its centre.
AXIS_LEFT = MARGIN
AXIS_RIGHT = CANVAS_WIDTH - MARGIN
BAND_GAP = 12
SEAM_HEIGHT = 288  # frames are decoded at this height
SEAM_BOX_WIDTH = 512  # a 16:9 frame at SEAM_HEIGHT
SEAM_GAP = 120
STRIP_FRAMES = 8  # even, so the join falls on a cell boundary
STRIP_HEIGHT = 96
WAVE_HEIGHT = 110
LEVEL_ROW_HEIGHT = 22  # the dB readout row above the wave
# The join line stays clear of labels by this many pixels.
JOIN_CLEAR = 4
JOIN_LINE_WIDTH = 3
# (font size, rows) tried in order for the word labels. One row reads in
# order, so the font shrinks before a second row is allowed.
WORD_LAYOUTS = ((20, 1), (18, 1), (16, 1), (20, 2), (18, 2), (16, 2), (16, 3), (15, 4))
WORD_GAP = 8  # pixels between labels in one row
# A label may slide this far right of its word's time to stay in its row.
WORD_MAX_SHIFT = 60
WORD_BAR_GAP = 3  # pixels left open at the end of each word's time bar
LABEL_SIZE = 16
TICK_SIZE = 15
REMOVED_SIZE = 18
# Tick spacing grows through these steps until ticks sit this far apart.
TICK_STEPS = (0.5, 1.0, 2.0, 5.0)
TICK_MIN_PIXELS = 110

# ── Colours: light ground, dark ink, one accent for the join, one for removed ──

BACKGROUND = (255, 255, 255)
INK = (17, 17, 17)
MUTED = (100, 100, 100)
JOIN = (214, 0, 28)
REMOVED = (0, 84, 196)
OUTSIDE = (228, 228, 228)  # time before the start or after the end of the video
WAVE_INK = (35, 35, 35)
WORD_BAR = (150, 150, 150)
SOUND_FILL = (222, 222, 222)


# ── Footage access (the only part that touches ffmpeg) ────────────────────────


class Footage:
    """Reads stills and audio from a video with ffmpeg.

    Tests pass a subclass or any object with the same three methods, so no
    test needs a real decode unless it wants one.
    """

    def has_audio(self, video: Path) -> bool:
        """Whether ``video`` has an audio stream. Raises StudioError when ffprobe fails."""
        out = _run(
            ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
             "-of", "csv=p=0", str(video)],
            what="ffprobe could not read the streams of",
            video=video,
            capture_stdout=True,
        )
        return bool(out.strip())

    def frames(self, video: Path, start: float, end: float, times: Sequence[float]) -> list[Image.Image | None]:
        """One still per source time in ``times`` (each inside ``[start, end]``).

        Decodes the span once at ``SEAM_HEIGHT`` into JPEG files (every frame,
        none duplicated or dropped) and picks the frame on screen at each time.
        A time at ``end`` gets the span's last frame. An entry is None when the
        span is too short to hold a frame.
        """
        with tempfile.TemporaryDirectory(prefix="lumr-look-") as tmp:
            folder = Path(tmp)
            _run(
                ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
                 "-i", str(video), "-map", "0:v:0", "-vf", f"scale=-2:{SEAM_HEIGHT}",
                 "-fps_mode", "passthrough", "-c:v", "mjpeg", "-q:v", str(FRAME_JPEG_QUALITY),
                 "-f", "image2", str(folder / "f%05d.jpg")],
                what="ffmpeg could not read frames from",
                video=video,
                log=folder / "ffmpeg.log",
            )
            files = sorted(folder.glob("f*.jpg"))
            picked: list[Image.Image | None] = []
            for t in times:
                if not files:
                    picked.append(None)
                    continue
                index = int((t - start) / max(end - start, TIME_EPS) * len(files))
                with Image.open(files[min(len(files) - 1, max(0, index))]) as im:
                    picked.append(im.convert("RGB"))
            return picked

    def audio(self, video: Path, start: float, end: float) -> np.ndarray:
        """Mono samples in [-1, 1] at ``AUDIO_RATE`` for source ``[start, end]``."""
        with tempfile.TemporaryDirectory(prefix="lumr-look-") as tmp:
            folder = Path(tmp)
            raw = folder / "audio.raw"
            _run(
                ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
                 "-i", str(video), "-map", "0:a:0", "-ac", "1", "-ar", str(AUDIO_RATE),
                 "-f", "s16le", str(raw)],
                what="ffmpeg could not read the sound of",
                video=video,
                log=folder / "ffmpeg.log",
            )
            return np.fromfile(raw, dtype="<i2").astype(np.float32) / 32768.0


def _run(cmd: list[str], *, what: str, video: Path, log: Path | None = None,
         capture_stdout: bool = False) -> str:
    """Run one ffmpeg or ffprobe command to completion and return its stdout.

    stderr goes to ``log`` (a file, never a pipe that could fill) or is
    captured by ``communicate`` for ffprobe's short output. Raises StudioError
    with the tail of stderr when the command fails, hangs, or is missing.
    """
    try:
        if log is not None:
            with open(log, "wb") as err:
                done = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                      stderr=err, timeout=FFMPEG_TIMEOUT_SECONDS, check=False)
            stderr = log.read_bytes()
            stdout = b""
        else:
            done = subprocess.run(cmd, stdin=subprocess.DEVNULL,
                                  stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
                                  stderr=subprocess.PIPE, timeout=FFMPEG_TIMEOUT_SECONDS, check=False)
            stderr = done.stderr or b""
            stdout = done.stdout or b""
    except FileNotFoundError:
        raise StudioError(f"{cmd[0]} is not installed or not on PATH. Install ffmpeg and try again.") from None
    except subprocess.TimeoutExpired:
        raise StudioError(
            f"{what} {video.name}: it ran past {FFMPEG_TIMEOUT_SECONDS:.0f}s. "
            "Check that the video plays and is on a local disk."
        ) from None
    if done.returncode != 0:
        tail = stderr.decode("utf-8", "replace").strip()[-400:]
        raise StudioError(f"{what} {video.name}: {tail} Check that the file is a playable video.")
    return stdout.decode("utf-8", "replace")


# ── Where the window sits in time ─────────────────────────────────────────────


@dataclass(frozen=True)
class Piece:
    """One stretch of source that plays inside the window, and where on the axis.

    ``at`` is the axis time (seconds from the window centre) where it begins.
    """

    src_start: float
    src_end: float
    at: float

    @property
    def end_at(self) -> float:
        return self.at + (self.src_end - self.src_start)

    def to_axis(self, t: float) -> float:
        return self.at + (t - self.src_start)

    def to_source(self, u: float) -> float:
        return self.src_start + (u - self.at)


@dataclass(frozen=True)
class Window:
    """What the picture shows: the chosen cut (or None) and the pieces that play."""

    join: tuple[float, float] | None
    edited_at: float
    pieces: list[Piece]
    cuts: list[tuple[float, float]]


def _cuts_from_kept(kept: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    """The removed spans as the render sees them: every gap around the kept segments."""
    cuts: list[tuple[float, float]] = []
    prev = 0.0
    for s, e in kept:
        if s > prev + TIME_EPS:
            cuts.append((prev, s))
        prev = e
    if duration > prev + TIME_EPS:
        cuts.append((prev, duration))
    return cuts


def _nearest_cut(cuts: list[tuple[float, float]], at: float, span: float) -> tuple[float, float] | None:
    """The cut closest to ``at`` (0 when ``at`` is inside it), or None beyond ``span``."""
    best: tuple[float, float] | None = None
    best_gap = math.inf
    for a, b in cuts:
        gap = 0.0 if a <= at <= b else min(abs(at - a), abs(at - b))
        if gap < best_gap:
            best, best_gap = (a, b), gap
    return best if best_gap <= span else None


def plan_window(removed: list[tuple[float, float]], duration: float, at: float, span: float) -> Window:
    """Choose the join nearest ``at`` and the source pieces within ``span`` of it.

    With a join, the window is ``span`` seconds of EDITED time either side of
    it. Without one, it is the plain source around ``at``. Raises StudioError
    when the edit keeps nothing.
    """
    kept = kept_segments(removed, duration)
    if not kept:
        raise StudioError("The edit removes the whole video, so there is no join to draw. Restore some material.")
    cuts = _cuts_from_kept(kept, duration)
    join = _nearest_cut(cuts, at, span)
    if join is None:
        lo, hi = max(0.0, at - span), min(duration, at + span)
        return Window(None, to_edited_time(at, kept)[0], [Piece(lo, hi, lo - at)], cuts)
    centre = to_edited_time(join[1], kept)[0]
    w0, w1 = max(0.0, centre - span), min(edited_duration(kept), centre + span)
    pieces = [Piece(s, e, to_edited_time(s, kept)[0] - centre) for s, e in source_spans_for_window(kept, w0, w1)]
    return Window(join, centre, pieces, cuts)


def _piece_at(pieces: list[Piece], u: float) -> Piece | None:
    for p in pieces:
        if p.at - TIME_EPS <= u < p.end_at - TIME_EPS:
            return p
    return None


# ── Frame and audio requests ──────────────────────────────────────────────────


@dataclass
class Media:
    """Everything decoded for one picture."""

    strip: list[Image.Image | None]  # one per strip cell, None outside the video
    before: Image.Image | None
    after: Image.Image | None
    audio: np.ndarray  # axis-aligned samples, NaN outside the video


def strip_times(span: float) -> list[float]:
    """Axis time at the centre of each strip cell. Even count, so 0 is a cell edge."""
    step = 2 * span / STRIP_FRAMES
    return [-span + (i + 0.5) * step for i in range(STRIP_FRAMES)]


def _decode(video: Path, window: Window, span: float, footage: Footage) -> Media:
    """Decode every still and all audio the picture needs, one ffmpeg run per piece and kind."""
    requests: dict[int, list[tuple[str, float]]] = {i: [] for i in range(len(window.pieces))}
    for cell, u in enumerate(strip_times(span)):
        piece = _piece_at(window.pieces, u)
        if piece is not None:
            requests[window.pieces.index(piece)].append((f"strip{cell}", piece.to_source(u)))
    if window.join is not None:
        for i, p in enumerate(window.pieces):
            if abs(p.end_at) < TIME_EPS:
                requests[i].append(("before", p.src_end))
            if abs(p.at) < TIME_EPS:
                requests[i].append(("after", p.src_start))

    audio = np.full(int(round(2 * span * AUDIO_RATE)), np.nan, dtype=np.float32)
    with ThreadPoolExecutor(max_workers=2 * len(window.pieces) + 1) as pool:
        stills = {
            i: pool.submit(footage.frames, video, p.src_start, p.src_end, [t for _, t in requests[i]])
            for i, p in enumerate(window.pieces) if requests[i]
        }
        if footage.has_audio(video):
            sounds = [pool.submit(footage.audio, video, p.src_start, p.src_end) for p in window.pieces]
            for p, sound in zip(window.pieces, sounds):
                _place(audio, sound.result(), p, span)
        else:
            for p in window.pieces:
                _place(audio, np.zeros(0, dtype=np.float32), p, span)
        found: dict[str, Image.Image | None] = {}
        for i, future in stills.items():
            for (name, _), image in zip(requests[i], future.result()):
                found[name] = image

    return Media(
        strip=[found.get(f"strip{c}") for c in range(STRIP_FRAMES)],
        before=found.get("before"),
        after=found.get("after"),
        audio=audio,
    )


def _place(axis_audio: np.ndarray, samples: np.ndarray, piece: Piece, span: float) -> None:
    """Copy one piece's samples into the axis-aligned buffer at its position."""
    lo = int(round((piece.at + span) * AUDIO_RATE))
    want = int(round((piece.src_end - piece.src_start) * AUDIO_RATE))
    chunk = samples[:want]
    hi = min(len(axis_audio), lo + len(chunk))
    if hi > lo:
        axis_audio[lo:hi] = chunk[: hi - lo]
    # A decode a few samples short of the span is silence, not "outside the video".
    short_end = min(len(axis_audio), lo + want)
    if short_end > hi:
        axis_audio[hi:short_end] = 0.0


# ── Text on the axis ──────────────────────────────────────────────────────────


@dataclass
class Label:
    """One word or sound block to place on the axis."""

    text: str
    x0: float  # where its time starts, in pixels
    x1: float  # where its time ends, in pixels
    kind: str  # "word", "sound" or "laugh"
    left_of_join: bool
    row: int = 0
    left: float = 0.0  # where the text box starts after placement
    width: float = 0.0


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


def _x(u: float, span: float) -> float:
    """Pixel x for axis time ``u``."""
    return AXIS_LEFT + (u + span) / (2 * span) * (AXIS_RIGHT - AXIS_LEFT)


def _sound_name(label: dict[str, Any] | None) -> tuple[str, str]:
    """``(name, kind)`` for one sound: ``laugh``, ``laugh?`` (possible) or ``sound``."""
    if label is None or label.get("kind") != "laugh":
        return "sound", "sound"
    return ("laugh" if label.get("confidence") == "likely" else "laugh?"), "laugh"


def _label_for(entry: dict[str, Any], labels: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    """The sound label covering a transcript event's midpoint, if any."""
    mid = _mid(entry)
    for label in labels or []:
        if float(label["start"]) <= mid <= float(label["end"]):
            return label
    return None


def _clip_to_pieces(start: float, end: float, pieces: list[Piece]) -> list[tuple[Piece, float, float]]:
    """The parts of source ``[start, end]`` that play, per piece."""
    parts = []
    for p in pieces:
        lo, hi = max(start, p.src_start), min(end, p.src_end)
        if hi > lo:
            parts.append((p, lo, hi))
    return parts


def axis_labels(words: list[dict[str, Any]], window: Window, span: float,
                labels: list[dict[str, Any]] | None) -> list[Label]:
    """Kept words and sound blocks that fall in the window, in pixel coordinates."""
    out: list[Label] = []
    for w in words:
        if is_event(w) or word_is_cut(w, window.cuts):
            continue
        mid = _mid(w)
        for p, lo, hi in _clip_to_pieces(float(w["start"]), float(w["end"]), window.pieces):
            if p.src_start <= mid <= p.src_end:
                u0, u1 = p.to_axis(lo), p.to_axis(hi)
                out.append(Label(w["word"].strip(), _x(u0, span), _x(u1, span), "word", u1 <= TIME_EPS))

    if labels is not None:
        sounds = [(float(s["start"]), float(s["end"]), s) for s in labels]
    else:
        sounds = [(float(w["start"]), float(w["end"]), None) for w in words
                  if is_event(w) and not word_is_cut(w, window.cuts)]
    for start, end, label in sounds:
        name, kind = _sound_name(label)
        text = f"{name} {end - start:.1f}s"
        for p, lo, hi in _clip_to_pieces(start, end, window.pieces):
            u0, u1 = p.to_axis(lo), p.to_axis(hi)
            out.append(Label(text, _x(u0, span), _x(u1, span), kind, u1 <= TIME_EPS))
    return sorted(out, key=lambda lab: lab.x0)


def _anchor(item: Label, width: float, join_x: float | None) -> float:
    """Where a label would like to start: at its time, kept to its side of the join."""
    left = item.x0
    if join_x is not None and item.left_of_join:
        left = min(left, join_x - JOIN_CLEAR - width)
    elif join_x is not None:
        left = max(left, join_x + JOIN_CLEAR)
    return max(AXIS_LEFT, min(left, AXIS_RIGHT - width))


def _fits(left: float, width: float, item: Label, join_x: float | None) -> bool:
    """A label placed at ``left`` stays on the canvas and on its side of the join."""
    if left + width > AXIS_RIGHT:
        return False
    return not (join_x is not None and item.left_of_join and left + width > join_x - JOIN_CLEAR)


def _try_layout(items: list[Label], size: int, rows: int, join_x: float | None) -> int:
    """Place labels in time order with ``rows`` rows at ``size``; return how many had no room.

    Each label takes the first row where it can start within
    ``WORD_MAX_SHIFT`` pixels of its anchor, so a single row keeps reading
    order even when a label has to slide a little right of its time.
    """
    font = _font(size)
    ends = [-math.inf] * rows
    dropped = 0
    for item in items:
        pad = 10 if item.kind != "word" else 0
        item.width = max(font.getlength(item.text) + pad, item.x1 - item.x0 if pad else 0)
        anchor = _anchor(item, item.width, join_x)
        item.row = -1
        for row in range(rows):
            left = max(anchor, ends[row] + WORD_GAP)
            if left - anchor <= WORD_MAX_SHIFT and _fits(left, item.width, item, join_x):
                item.row, item.left, ends[row] = row, left, left + item.width
                break
        dropped += item.row < 0
    return dropped


def layout_labels(items: list[Label], join_x: float | None) -> tuple[int, int, int]:
    """Give every label a row and a left edge so no two labels overlap.

    Tries each ``(size, rows)`` in ``WORD_LAYOUTS`` in turn and keeps the first
    that places every label, so one row at a smaller size wins over a second
    row: one row reads in order. Labels left of the join end before its line,
    labels right of it start after. When no layout fits, the last one is used
    and the labels without room are left out.

    Returns ``(font size, rows, labels left out)``; left-out labels get ``row = -1``.
    """
    items.sort(key=lambda lab: lab.x0)
    for size, rows in WORD_LAYOUTS:
        dropped = _try_layout(items, size, rows, join_x)
        if dropped == 0:
            break
    used = max((item.row for item in items), default=0) + 1
    return size, max(1, used), dropped


# ── Drawing ───────────────────────────────────────────────────────────────────


def _paste_fit(canvas: Image.Image, image: Image.Image | None, box: tuple[int, int, int, int],
               empty_text: str) -> None:
    """Draw ``image`` scaled to fit ``box`` and centred, or a grey box with ``empty_text``."""
    x0, y0, x1, y1 = box
    if image is None:
        draw = ImageDraw.Draw(canvas)
        draw.rectangle(box, fill=OUTSIDE)
        font = _font(LABEL_SIZE)
        w = font.getlength(empty_text)
        draw.text(((x0 + x1 - w) / 2, (y0 + y1) / 2 - LABEL_SIZE / 2), empty_text, fill=MUTED, font=font)
        return
    scale = min((x1 - x0) / image.width, (y1 - y0) / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    fitted = image.resize(size, Image.Resampling.LANCZOS)
    canvas.paste(fitted, (x0 + (x1 - x0 - size[0]) // 2, y0 + (y1 - y0 - size[1]) // 2))


def _draw_seam(canvas: Image.Image, y: int, window: Window, media: Media) -> int:
    """Band 1: the two frames either side of the join, with their source times."""
    draw = ImageDraw.Draw(canvas)
    font = _font(LABEL_SIZE)
    a, b = window.join  # type: ignore[misc]
    centre = CANVAS_WIDTH // 2
    left_box = (centre - SEAM_GAP // 2 - SEAM_BOX_WIDTH, y + LABEL_SIZE + 8, centre - SEAM_GAP // 2,
                y + LABEL_SIZE + 8 + SEAM_HEIGHT)
    right_box = (centre + SEAM_GAP // 2, left_box[1], centre + SEAM_GAP // 2 + SEAM_BOX_WIDTH, left_box[3])
    left_text = "nothing before: start of video"
    if media.before is not None:
        left_text = f"last frame before the cut ({a:.2f})"
    draw.text((left_box[0], y), left_text, fill=INK, font=font)
    right_text = "nothing after: end of video"
    if media.after is not None:
        right_text = f"first frame after the cut ({b:.2f})"
    draw.text((right_box[2] - font.getlength(right_text), y), right_text, fill=INK, font=font)
    cut_text = f"cut {b - a:.2f}s"
    draw.text((centre - font.getlength(cut_text) / 2, y), cut_text, fill=REMOVED, font=font)
    _paste_fit(canvas, media.before, left_box, "start of video")
    _paste_fit(canvas, media.after, right_box, "end of video")
    return left_box[3]


def _draw_strip(canvas: Image.Image, y: int, span: float, media: Media) -> int:
    """Band 2: evenly spaced stills across the window, in edited order."""
    cell = (AXIS_RIGHT - AXIS_LEFT) / STRIP_FRAMES
    for i, image in enumerate(media.strip):
        x0 = round(AXIS_LEFT + i * cell) + 1
        x1 = round(AXIS_LEFT + (i + 1) * cell) - 1
        _paste_fit(canvas, image, (x0, y, x1, y + STRIP_HEIGHT), "outside video")
    return y + STRIP_HEIGHT


def _db(x: np.ndarray | float) -> np.ndarray | float:
    return 20 * np.log10(np.maximum(x, 1e-6))


def join_levels(audio: np.ndarray) -> tuple[float | None, float | None]:
    """RMS level in dB of the sound just before and just after the axis centre."""
    mid = len(audio) // 2
    n = int(JOIN_LEVEL_SECONDS * AUDIO_RATE)
    out: list[float | None] = []
    for chunk in (audio[max(0, mid - n):mid], audio[mid:mid + n]):
        chunk = chunk[~np.isnan(chunk)]
        out.append(float(_db(np.sqrt(np.mean(chunk ** 2)))) if len(chunk) else None)
    return out[0], out[1]


def _draw_levels(canvas: Image.Image, y: int, media: Media) -> int:
    """The sound level just before and just after the join, either side of its line."""
    draw = ImageDraw.Draw(canvas)
    font = _font(LABEL_SIZE)
    before, after = join_levels(media.audio)
    centre = CANVAS_WIDTH // 2
    if before is not None:
        text = f"level {before:.0f} dB"
        draw.text((centre - JOIN_CLEAR - 4 - font.getlength(text), y), text, fill=JOIN, font=font)
    if after is not None:
        draw.text((centre + JOIN_CLEAR + 4, y), f"{after:.0f} dB", fill=JOIN, font=font)
    return y + LEVEL_ROW_HEIGHT


def _draw_wave(canvas: Image.Image, y: int, media: Media) -> int:
    """Band 3: per-column peak loudness on a dB scale, mirrored about a centre line."""
    draw = ImageDraw.Draw(canvas)
    audio = media.audio
    columns = AXIS_RIGHT - AXIS_LEFT
    edges = np.linspace(0, len(audio), columns + 1).astype(int)[:-1]
    valid = ~np.isnan(audio)
    magnitude = np.where(valid, np.abs(np.nan_to_num(audio)), 0.0)
    peaks = np.maximum.reduceat(magnitude, edges)
    inside = np.add.reduceat(valid.astype(np.int32), edges) > 0
    mid_y = y + WAVE_HEIGHT / 2
    if inside.any():
        ref = max(float(_db(np.percentile(peaks[inside], 99))), WAVE_QUIETEST_REF_DB)
        heights = np.clip((_db(peaks) - (ref - WAVE_RANGE_DB)) / WAVE_RANGE_DB, 0.0, 1.0)
    else:
        heights = np.zeros(columns)
    for c in range(columns):
        x = AXIS_LEFT + c
        if not inside[c]:
            draw.line([(x, y), (x, y + WAVE_HEIGHT)], fill=OUTSIDE)
            continue
        half = heights[c] * (WAVE_HEIGHT / 2 - 2)
        draw.line([(x, mid_y - half), (x, mid_y + half)], fill=WAVE_INK)
    draw.line([(AXIS_LEFT, mid_y), (AXIS_RIGHT, mid_y)], fill=WAVE_INK)
    return y + WAVE_HEIGHT


def _row_height(size: int) -> int:
    return size + 12


def _draw_words(canvas: Image.Image, y: int, items: list[Label], size: int, rows: int) -> int:
    """Band 4: each label under a thin bar showing its word's real time span."""
    draw = ImageDraw.Draw(canvas)
    font = _font(size)
    for item in items:
        if item.row < 0:
            continue
        top = y + item.row * _row_height(size)
        if item.kind == "word":
            # Hammy ends a word where the next begins; the gap keeps neighbours apart.
            draw.rectangle([item.x0, top, max(item.x0 + 2, item.x1 - WORD_BAR_GAP), top + 2], fill=WORD_BAR)
            draw.text((item.left, top + 5), item.text, fill=INK, font=font)
            continue
        box = [item.left, top, item.left + item.width, top + size + 8]
        outline = INK if item.kind == "laugh" else WORD_BAR
        draw.rectangle(box, fill=SOUND_FILL, outline=outline, width=2 if item.kind == "laugh" else 1)
        draw.text((item.left + 5, top + 3), item.text, fill=INK, font=font)
    return y + rows * _row_height(size)


def _tick_step(span: float) -> float:
    per_second = (AXIS_RIGHT - AXIS_LEFT) / (2 * span)
    for step in TICK_STEPS:
        if step * per_second >= TICK_MIN_PIXELS:
            return step
    return TICK_STEPS[-1]


def _draw_axis(canvas: Image.Image, y: int, window: Window, span: float, at: float) -> int:
    """Source times under the axis: one tick per step, the join's two edges at the centre."""
    draw = ImageDraw.Draw(canvas)
    font = _font(TICK_SIZE)
    draw.line([(AXIS_LEFT, y), (AXIS_RIGHT, y)], fill=MUTED)
    step = _tick_step(span)
    k = 1
    while k * step < span:
        for u in (-k * step, k * step):
            piece = _piece_at(window.pieces, u)
            if piece is None:
                continue
            x = _x(u, span)
            text = f"{piece.to_source(u):.2f}"
            draw.line([(x, y), (x, y + 5)], fill=MUTED)
            draw.text((x - font.getlength(text) / 2, y + 6), text, fill=MUTED, font=font)
        k += 1
    centre = CANVAS_WIDTH // 2
    if window.join is not None:
        text, colour = f"{window.join[0]:.2f} | {window.join[1]:.2f}", JOIN
    else:
        text, colour = f"{at:.2f}", INK
    draw.text((centre - font.getlength(text) / 2, y + 6), text, fill=colour, font=font)
    return y + 6 + TICK_SIZE + 4


def _fit_words(tokens: list[str], font: ImageFont.FreeTypeFont, width: float, prefix: str) -> str:
    """``prefix`` plus the tokens, eliding the middle until the line fits ``width``."""
    full = prefix + " ".join(tokens)
    if font.getlength(full) <= width:
        return full
    for keep in range(len(tokens) // 2, 0, -1):
        text = prefix + " ".join(tokens[:keep]) + " ... " + " ".join(tokens[-keep:])
        if font.getlength(text) <= width:
            return text
    return prefix + "..."


def _mid(entry: dict[str, Any]) -> float:
    return (float(entry["start"]) + float(entry["end"])) / 2


def _token(entry: dict[str, Any], labels: list[dict[str, Any]] | None) -> str:
    """A transcript entry as it reads in a line of text; an event reads ``(laugh)`` or ``(sound)``."""
    if is_event(entry):
        return f"({_sound_name(_label_for(entry, labels))[0]})"
    return entry["word"].strip()


def _draw_removed(canvas: Image.Image, y: int, window: Window, words: list[dict[str, Any]],
                  labels: list[dict[str, Any]] | None, span: float, at: float) -> int:
    """Band 5: what the cut took out, then how the join reads now."""
    draw = ImageDraw.Draw(canvas)
    font = _font(REMOVED_SIZE)
    width = AXIS_RIGHT - AXIS_LEFT
    line = REMOVED_SIZE + 10
    if window.join is None:
        text = f"No cut within {span:.1f}s of {at:.2f}. Plain source shown."
        draw.text((AXIS_LEFT, y), text, fill=INK, font=font)
        return y + line
    a, b = window.join
    gone = [w for w in words if a <= _mid(w) < b]
    spoken = sum(1 for w in gone if not is_event(w))
    prefix = f"REMOVED {a:.2f}-{b:.2f} ({b - a:.2f}s, {spoken} words): "
    body = [_token(w, labels) for w in gone] or ["no words, pause only"]
    draw.text((AXIS_LEFT, y), _fit_words(body, font, width, prefix), fill=REMOVED, font=font)

    kept = [w for w in words if not word_is_cut(w, window.cuts)]
    before = [_token(w, labels) for w in kept if _mid(w) < a][-6:]
    after = [_token(w, labels) for w in kept if _mid(w) > b][:6]
    left = "NOW READS: " + ("..." + " ".join(before) if before else "(start of video)") + " "
    draw.text((AXIS_LEFT, y + line), left, fill=INK, font=font)
    x = AXIS_LEFT + font.getlength(left)
    draw.text((x, y + line), "|", fill=JOIN, font=font)
    right = " ".join(after) + "..." if after else "(end of video)"
    draw.text((x + font.getlength("| "), y + line), right, fill=INK, font=font)
    return y + 2 * line


def _draw_join_lines(canvas: Image.Image, seam_top: int, axis_top: int, bottom: int,
                     window: Window, span: float) -> None:
    """The join in one heavy line from the seam pair down; other joins thin and dashed.

    The seam frames sit either side of the canvas centre, which is also the
    axis centre, so the heavy line runs through the gap between them and on
    through the strip, wave and words at the same x.
    """
    draw = ImageDraw.Draw(canvas)
    if window.join is not None:
        centre = CANVAS_WIDTH // 2
        draw.line([(centre, seam_top), (centre, bottom)], fill=JOIN, width=JOIN_LINE_WIDTH)
    for p in window.pieces[1:]:
        if abs(p.at) < TIME_EPS:
            continue
        x = _x(p.at, span)
        for yy in range(axis_top, bottom, 10):
            draw.line([(x, yy), (x, min(bottom, yy + 5))], fill=JOIN, width=1)


def _encode_png(canvas: Image.Image) -> bytes:
    """PNG bytes under ``MAX_FILE_BYTES``: full colour when it fits, else a 256-colour palette."""
    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=True)
    if buf.tell() <= MAX_FILE_BYTES:
        return buf.getvalue()
    buf = io.BytesIO()
    canvas.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(
        buf, format="PNG", optimize=True)
    if buf.tell() > MAX_FILE_BYTES:
        raise StudioError("The join picture came out over 1MB. Pass a smaller span and try again.")
    return buf.getvalue()


# ── The tool ──────────────────────────────────────────────────────────────────


def render_join_picture(
    video: Path,
    removed: list[tuple[float, float]],
    words: list[dict[str, Any]],
    duration: float,
    at: float,
    out_path: Path,
    *,
    span: float = 3.0,
    labels: list[dict[str, Any]] | None = None,
    footage: Footage | None = None,
) -> dict[str, Any]:
    """One PNG showing the join nearest source time ``at`` as the viewer gets it.

    ``removed`` is the saved edit's removed spans in source seconds. The window
    is ``span`` seconds of edited time either side of the join. When no cut
    lies within ``span`` of ``at``, the plain source around ``at`` is drawn
    with no seam pair. ``labels`` are sound labels (laugh or sound) drawn as
    blocks; without them every transcript event is drawn as a sound.
    ``out_path`` is written (overwritten if present); the caller picks a new
    name, for example with ``project.reserve_unique_path``.

    Returns ``{path, width, height, join, edited_at, frames, seconds_shown}``.
    Raises StudioError for bad arguments or when ffmpeg cannot read the video.
    """
    if not duration > 0:
        raise StudioError(f"duration {duration} must be above 0. Pass the source video's length.")
    if not 0 <= at <= duration:
        raise StudioError(f"at {at} is outside the video (0 to {duration:.2f}). Pass a source time inside it.")
    if not 0 < span <= MAX_SPAN_SECONDS:
        raise StudioError(f"span {span} must be above 0 and at most {MAX_SPAN_SECONDS:.0f} seconds.")
    if Path(out_path).resolve().parent == Path(video).resolve().parent:
        raise StudioError("out_path is beside the source video. Write the picture into the project folder.")

    window = plan_window(removed, duration, at, span)
    media = _decode(Path(video), window, span, footage or Footage())
    has_join = window.join is not None
    join_x = CANVAS_WIDTH / 2 if has_join else None
    items = axis_labels(words, window, span, labels)
    size, rows, dropped = layout_labels(items, join_x)

    # Draw top down on a canvas taller than any layout needs, then crop.
    canvas = Image.new("RGB", (CANVAS_WIDTH, MAX_LONG_EDGE), BACKGROUND)
    y = MARGIN
    seam_top = y + LABEL_SIZE + 8
    if has_join:
        y = _draw_seam(canvas, y, window, media) + BAND_GAP
    strip_top = y
    y = _draw_strip(canvas, y, span, media) + BAND_GAP
    if has_join:
        y = _draw_levels(canvas, y, media)
    y = _draw_wave(canvas, y, media) + BAND_GAP
    y = _draw_words(canvas, y, items, size, rows)
    _draw_join_lines(canvas, seam_top, strip_top, y, window, span)
    y = _draw_axis(canvas, y, window, span, at) + BAND_GAP
    y = _draw_removed(canvas, y, window, words, labels, span, at)
    if dropped:
        note = f"{dropped} word labels had no room and are not shown. Pass a smaller span to see them."
        ImageDraw.Draw(canvas).text((AXIS_LEFT, y), note, fill=MUTED, font=_font(LABEL_SIZE))
        y += LABEL_SIZE + 8
    y += MARGIN
    canvas = canvas.crop((0, 0, CANVAS_WIDTH, y))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(_encode_png(canvas))

    frames = sum(im is not None for im in media.strip) + (media.before is not None) + (media.after is not None)
    shown = sum(p.src_end - p.src_start for p in window.pieces)
    return {
        "path": str(out),
        "width": canvas.width,
        "height": canvas.height,
        "join": [round(window.join[0], 3), round(window.join[1], 3)] if has_join else None,
        "edited_at": round(window.edited_at, 3),
        "frames": frames,
        "seconds_shown": round(shown, 3),
    }
