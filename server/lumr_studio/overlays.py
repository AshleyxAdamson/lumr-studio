"""The creator's photos and clips laid over the talk: checking, saving, placing.

An overlay covers the picture for a stretch while the creator's voice keeps
playing. It fills the frame or sits in a box at a named corner. The creator
points at files they already have; nothing is fetched or generated, and no
file is copied. The saved list keeps the absolute path, and every render
checks the file again.

Overlays live in their own file, ``overlays.json``, beside ``edit.json``. A new
edit never touches them and they never touch the cuts, so the creator can
move the pace dial as often as they like without losing a placement.

Anchoring. An overlay's ``start`` and ``end`` are SOURCE seconds, and the words
under them are recorded as ``starts_on`` and ``ends_on``. The source video
never changes, so the anchor holds through any change to the cuts. Only at
render time is it mapped to EDITED time through the saved edit. The rules for
the hard cases live in ``place_on_edit``:

- The start falls inside a cut: the overlay appears when the talk resumes.
- A cut falls inside the span: the overlay covers the same words for less
  time, and hides the jump in the picture there.
- So much is cut that under ``MIN_SCREEN_SECONDS`` is left: it is not shown.
- A clip runs out before its span ends: the talk shows again for the rest.
- An end past the video's end is rejected when saved.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lumr_studio.edit import clock
from lumr_studio.errors import StudioError
from lumr_studio.project import Project, now_iso, write_json_atomic
from lumr_studio.timeline import source_spans_for_window, to_edited_time
from lumr_studio.transcript import spoken_words, word_is_cut

OVERLAYS_VERSION = 1
OVERLAYS_FILE = "overlays.json"
# Intermediate renders live here while a render with overlays runs, out of
# sight of the exports folder, and are removed when it finishes.
WORK_FOLDER = "work"

IMAGE = "image"
VIDEO = "video"

FULL = "full"
# Boxes sit at a named spot. Names, never pixels, so a placement works at any
# video size.
BOX_PLACES = ("top_left", "top_right", "bottom_left", "bottom_right", "center")
PLACES = (FULL, *BOX_PLACES)
# fill: covers the frame, edges may be cropped. fit: the whole picture shows,
# over a blurred copy of itself.
FITS = ("fill", "fit")
MOTIONS = ("still", "zoom_in", "zoom_out")

# A box takes this share of the frame's width and height; the picture fits
# inside it and keeps its shape. Under 0.2 a phone viewer can't read it; over
# 0.6 it covers the creator's face in most framings.
BOX_SIZE_DEFAULT = 0.4
BOX_SIZE_MIN = 0.2
BOX_SIZE_MAX = 0.6
# A photo drifts by default: a still picture held for six seconds looks dead.
MOTION_DEFAULT = "zoom_in"
# A picture whose shape is within this share of the frame's shape fills the
# frame by default; a very different shape (a phone screenshot on a wide
# video) is shown whole, since filling would crop most of it away.
FILL_ASPECT_TOLERANCE = 0.2
# Clip sound mixes UNDER the voice: at most half as loud, never level with it.
# 0.2 sits quietly under speech.
MAX_CLIP_VOLUME = 0.5
MAX_LAYER = 9
# Each overlay is one ffmpeg input and one step in one filter graph. Past
# this the render time stops being predictable, and a video with more than
# thirty is better cut from a different recording.
MAX_OVERLAYS = 30
# A span shorter than this reads as a flash, not a picture.
MIN_SPAN_SECONDS = 1.0
# After cuts, an overlay left with less than this on screen is not shown.
MIN_SCREEN_SECONDS = 1.0
# An end this close past the video end is the end, not an error.
END_TOLERANCE = 0.05
# Two overlays on one layer may touch; overlapping by more than this clashes.
OVERLAP_EPS = 0.01
# A screen length differing from the span by less than this is rounding.
SHORTENED_EPS = 0.05
MIN_MEDIA_SIDE = 16
# 100 megapixels is past any phone photo; bigger is a scan or a mistake, and
# decoding it for every frame is slow.
MAX_IMAGE_PIXELS = 100_000_000
NOTE_MAX_CHARS = 200
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
ID_PREFIX = "ov"
# ffprobe answers fast; this only stops a stuck call.
PROBE_TIMEOUT_SECONDS = 30.0
# Words shown in ``over``, the quote of what the creator says under an overlay.
QUOTE_WORDS = 12
# The recorded start word should still sit this close to its time in the transcript.
WORD_DRIFT_SECONDS = 0.5

# ffprobe container names that mean a still picture.
IMAGE_FORMATS = ("image2", "png_pipe", "jpeg_pipe", "webp_pipe", "bmp_pipe", "tiff_pipe")

INPUT_FIELDS = (
    "id", "file", "start", "end", "place", "size", "fit", "motion",
    "clip_in", "clip_out", "volume", "layer", "fade", "note",
)
# Facts the server adds. Accepted in input so a list read back with
# get_overlays can be passed straight to set_overlays, and then ignored.
DERIVED_FIELDS = ("kind", "media", "starts_on", "ends_on", "clock", "on_screen", "status", "notes", "over")


class OverlayRejected(Exception):
    """One overlay that can't be saved. The message says why and how to fix it."""


# ── Paths ─────────────────────────────────────────────────────────────────────


def overlays_path(project: Project) -> Path:
    return project.root / OVERLAYS_FILE


def work_dir(project: Project) -> Path:
    return project.root / WORK_FOLDER


# ── Probing the creator's files ───────────────────────────────────────────────


@dataclass(frozen=True)
class Media:
    """What ffprobe says about one file."""

    kind: str  # IMAGE or VIDEO
    width: int
    height: int
    duration: float | None  # None for a still picture
    has_sound: bool
    fps: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width, "height": self.height,
            "duration": None if self.duration is None else round(self.duration, 3),
            "has_sound": self.has_sound,
        }


Prober = Callable[[Path], Media]


def _rate(text: str | None) -> float | None:
    try:
        num, den = (text or "").split("/")
        return float(num) / float(den) if float(den) else None
    except ValueError:
        return None


# EXIF orientations that turn a photo on its side (a phone held upright).
EXIF_ORIENTATION_TAG = 0x0112
SIDEWAYS_EXIF = (5, 6, 7, 8)


def _photo_turned(path: Path) -> bool:
    """Whether a photo's EXIF says it displays turned a quarter.

    ffmpeg applies that turn when it decodes the photo, but ffprobe reports the
    stored size, so the width and height must be swapped to match.
    """
    from PIL import Image

    try:
        with Image.open(path) as img:
            return img.getexif().get(EXIF_ORIENTATION_TAG) in SIDEWAYS_EXIF
    except (OSError, ValueError):
        return False


def _video_turned(stream: dict[str, Any]) -> bool:
    """Whether a clip's display rotation turns it a quarter, as phone video often does."""
    for side in stream.get("side_data_list") or []:
        rotation = side.get("rotation")
        if _finite(rotation) and round(abs(rotation)) % 180 == 90:
            return True
    return False


def probe_media(path: Path, *, run: Callable[..., Any] = subprocess.run) -> Media:
    """Read a file's kind, size, length and sound with ffprobe. Never trusts the extension.

    The size is the size as displayed, after any turn a phone recorded.
    Raises StudioError when ffprobe can't read it as a picture or a video.
    """
    fix = "Pass a photo (JPEG, PNG, WebP) or a video clip (MP4, MOV)."
    try:
        out = run(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=PROBE_TIMEOUT_SECONDS,
        )
        data = json.loads(out.stdout or "{}")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        raise StudioError(f"Could not read {path.name} with ffprobe. {fix}") from None
    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    pictures = [
        s for s in streams
        if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")
    ]
    if not pictures:
        raise StudioError(f"{path.name} has no picture in it. {fix}")
    picture = pictures[0]
    width, height = int(picture.get("width") or 0), int(picture.get("height") or 0)
    if width <= 0 or height <= 0:  # ffprobe guessed from the name but found no picture data
        raise StudioError(f"ffprobe could not read a picture from {path.name}. {fix}")
    has_sound = any(s.get("codec_type") == "audio" for s in streams)
    format_names = str(fmt.get("format_name", "")).split(",")
    if any(name in IMAGE_FORMATS for name in format_names):
        if _photo_turned(path):
            width, height = height, width
        return Media(IMAGE, width, height, None, False)
    if _video_turned(picture):
        width, height = height, width
    try:
        duration = float(fmt.get("duration") or picture.get("duration") or 0)
    except ValueError:
        duration = 0.0
    if not math.isfinite(duration) or duration <= 0:
        raise StudioError(f"ffprobe found no length in {path.name}. {fix}")
    return Media(VIDEO, width, height, duration, has_sound, _rate(picture.get("avg_frame_rate")))


def resolve_media_path(raw: Any) -> Path:
    """Check ``raw`` is an absolute path to an existing file. Raises OverlayRejected."""
    if not isinstance(raw, str) or not raw.strip():
        raise OverlayRejected("has no file. Pass the absolute path to the creator's photo or clip.")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise OverlayRejected(f"file {raw!r} is relative. Pass the absolute path to the file.")
    if not path.exists():
        raise OverlayRejected(f"has no file at {path}. Check the path with the creator.")
    if not path.is_file():
        raise OverlayRejected(f"file {path} is a folder. Pass the path to one photo or clip.")
    return path.resolve()


def _check_media(media: Media, name: str) -> None:
    if min(media.width, media.height) < MIN_MEDIA_SIDE:
        raise OverlayRejected(
            f"file {name} is {media.width}x{media.height}, too small to show. Use a bigger picture."
        )
    if media.kind == IMAGE and media.width * media.height > MAX_IMAGE_PIXELS:
        raise OverlayRejected(
            f"file {name} is {media.width}x{media.height}, over {MAX_IMAGE_PIXELS // 1_000_000} "
            "megapixels. Export a smaller copy, 4000 pixels on the long side is plenty."
        )


# ── Checking one overlay ──────────────────────────────────────────────────────


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _given(raw: dict[str, Any], key: str) -> bool:
    return raw.get(key) is not None


def _choice(raw: dict[str, Any], key: str, options: tuple[str, ...], default: str) -> str:
    value = raw.get(key)
    if value is None:
        return default
    if value not in options:
        raise OverlayRejected(f"{key} {value!r} is not one of {', '.join(options)}.")
    return value


def _number(raw: dict[str, Any], key: str, lo: float, hi: float, default: float) -> float:
    value = raw.get(key)
    if value is None:
        return default
    if not _finite(value) or not lo <= value <= hi:
        raise OverlayRejected(f"{key} {value!r} must be a number from {lo:g} to {hi:g}.")
    return float(value)


def _only_for(raw: dict[str, Any], keys: tuple[str, ...], applies_to: str) -> None:
    for key in keys:
        if _given(raw, key):
            raise OverlayRejected(f"{key} only applies to {applies_to}. Drop it.")


def _check_fields(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise OverlayRejected("is not an object. Pass {file, start, end} and any options.")
    unknown = sorted(set(raw) - set(INPUT_FIELDS) - set(DERIVED_FIELDS))
    if unknown:
        raise OverlayRejected(
            f"has unknown field {', '.join(map(repr, unknown))}. Fields are {', '.join(INPUT_FIELDS)}."
        )
    return raw


def _check_span(raw: dict[str, Any], duration: float) -> tuple[float, float]:
    start, end = raw.get("start"), raw.get("end")
    if not _finite(start) or not _finite(end):
        raise OverlayRejected("start and end must be numbers of seconds in the source video.")
    if start < 0:
        raise OverlayRejected(f"start {start} is before 0. The video starts at 0.")
    if end > duration + END_TOLERANCE:
        raise OverlayRejected(
            f"end {end} is past the end of the video ({duration:.2f}, {clock(duration)}). "
            "End it at or before the last word."
        )
    end = min(float(end), duration)
    if end - start < MIN_SPAN_SECONDS:
        raise OverlayRejected(
            f"runs {max(0.0, end - start):.2f}s ({start}-{end}). Give it at least "
            f"{MIN_SPAN_SECONDS:g}s: shorter reads as a flash."
        )
    return float(start), end


def _default_fit(media: Media, frame: tuple[int, int]) -> str:
    frame_aspect = frame[0] / frame[1]
    aspect = media.width / media.height
    return "fill" if abs(aspect / frame_aspect - 1) <= FILL_ASPECT_TOLERANCE else "fit"


def _clip_range(raw: dict[str, Any], media: Media, name: str) -> tuple[float, float]:
    length = float(media.duration or 0)
    clip_in = _number(raw, "clip_in", 0.0, length, 0.0)
    clip_out = raw.get("clip_out")
    if clip_out is None:
        clip_out = length
    elif not _finite(clip_out) or clip_out > length + END_TOLERANCE:
        raise OverlayRejected(
            f"clip_out {clip_out!r} is past the end of {name}, which runs {length:.2f}s."
        )
    clip_out = min(float(clip_out), length)
    if clip_out - clip_in < MIN_SPAN_SECONDS:
        raise OverlayRejected(
            f"plays {clip_out - clip_in:.2f}s of {name} (clip_in {clip_in:g} to clip_out {clip_out:g}). "
            f"Play at least {MIN_SPAN_SECONDS:g}s."
        )
    return clip_in, clip_out


def parse_overlay(
    raw: Any,
    *,
    duration: float,
    frame: tuple[int, int],
    probe: Prober = probe_media,
) -> dict[str, Any]:
    """Check one overlay against the video and its file; return it with defaults filled in.

    ``frame`` is the source video's ``(width, height)``. The id is checked but
    not assigned here. Raises OverlayRejected with a message that says how to fix it.
    """
    raw = _check_fields(raw)
    path = resolve_media_path(raw.get("file"))
    try:
        media = probe(path)
    except StudioError as exc:
        raise OverlayRejected(f"file {exc}") from None
    _check_media(media, path.name)
    start, end = _check_span(raw, duration)
    oid = raw.get("id")
    if oid is not None and (not isinstance(oid, str) or not ID_PATTERN.match(oid)):
        raise OverlayRejected(
            f"id {oid!r} must be 1 to 32 lowercase letters, digits, - or _. Leave it out for a new one."
        )
    place = _choice(raw, "place", PLACES, FULL)
    note = raw.get("note")
    if note is not None and (not isinstance(note, str) or len(note) > NOTE_MAX_CHARS):
        raise OverlayRejected(f"note must be text of at most {NOTE_MAX_CHARS} characters.")
    fade = True if raw.get("fade") is None else raw["fade"]
    if not isinstance(fade, bool):
        raise OverlayRejected("fade must be true or false.")
    layer = 1 if raw.get("layer") is None else raw["layer"]
    if not isinstance(layer, int) or isinstance(layer, bool) or not 1 <= layer <= MAX_LAYER:
        raise OverlayRejected(f"layer {layer!r} must be a whole number from 1 to {MAX_LAYER}. Higher shows on top.")

    if place == FULL:
        _only_for(raw, ("size",), "a box (a place other than full)")
        size, fit = None, _choice(raw, "fit", FITS, _default_fit(media, frame))
    else:
        _only_for(raw, ("fit",), "full-frame overlays; a box always shows the whole picture")
        size, fit = _number(raw, "size", BOX_SIZE_MIN, BOX_SIZE_MAX, BOX_SIZE_DEFAULT), None

    if media.kind == IMAGE:
        _only_for(raw, ("clip_in", "clip_out", "volume"), "video clips")
        motion = _choice(raw, "motion", MOTIONS, MOTION_DEFAULT)
        clip_in = clip_out = volume = None
    else:
        _only_for(raw, ("motion",), "photos; a clip moves on its own")
        motion = None
        clip_in, clip_out = _clip_range(raw, media, path.name)
        volume = _number(raw, "volume", 0.0, MAX_CLIP_VOLUME, 0.0)
        if volume > 0 and not media.has_sound:
            raise OverlayRejected(f"volume is {volume:g} but {path.name} has no sound. Set volume to 0.")

    return {
        "id": oid,
        "file": str(path),
        "kind": media.kind,
        "start": round(start, 3),
        "end": round(end, 3),
        "place": place,
        "size": size,
        "fit": fit,
        "motion": motion,
        "clip_in": None if clip_in is None else round(clip_in, 3),
        "clip_out": None if clip_out is None else round(clip_out, 3),
        "volume": volume,
        "layer": layer,
        "fade": fade,
        "note": note,
        "media": media.to_dict(),
    }


# ── Checking the whole list ───────────────────────────────────────────────────


@dataclass
class OverlayOutcome:
    """What ``build_overlays`` decided, ready to save and to report."""

    overlays: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    next_id: int = 1


def _id_number(oid: str) -> int:
    match = re.fullmatch(rf"{ID_PREFIX}(\d+)", oid)
    return int(match.group(1)) if match else 0


def _clash(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        a["layer"] == b["layer"]
        and a["start"] < b["end"] - OVERLAP_EPS
        and b["start"] < a["end"] - OVERLAP_EPS
    )


def _span_label(o: dict[str, Any]) -> str:
    return f"{o['id'] or 'the new one'} ({clock(o['start'])}-{clock(o['end'])})"


def build_overlays(
    raw_list: list[Any],
    *,
    duration: float,
    frame: tuple[int, int],
    words: list[dict[str, Any]] | None = None,
    next_id: int = 1,
    probe: Prober = probe_media,
) -> OverlayOutcome:
    """Check every overlay, give each a stable id, and reject clashes on one layer.

    ``next_id`` is the first number free for a new ``ov<n>`` id; ids are never
    reused, so the creator's page can hold on to them. An overlay that fails a
    check is rejected on its own and the rest are kept. Of two overlays that
    overlap on one layer, the later in the list is rejected.
    """
    if len(raw_list) > MAX_OVERLAYS:
        raise StudioError(
            f"{len(raw_list)} overlays is more than {MAX_OVERLAYS}. Keep the ones that matter most."
        )
    outcome = OverlayOutcome(next_id=next_id)
    ids = [r.get("id") for r in raw_list if isinstance(r, dict) and isinstance(r.get("id"), str)]
    outcome.next_id = max([next_id, *(_id_number(i) + 1 for i in ids)])
    seen: set[str] = set()
    for i, raw in enumerate(raw_list):
        try:
            overlay = parse_overlay(raw, duration=duration, frame=frame, probe=probe)
            if overlay["id"] in seen:
                raise OverlayRejected(f"repeats id {overlay['id']!r}. Every overlay needs its own id.")
            clash = next((o for o in outcome.overlays if _clash(o, overlay)), None)
            if clash:
                raise OverlayRejected(
                    f"overlaps {_span_label(clash)} and both are on layer {overlay['layer']}. "
                    "Put the one that should show on top on a higher layer, or move one so they "
                    "don't overlap."
                )
        except OverlayRejected as why:
            outcome.rejected.append({"index": i, "overlay": raw, "why": f"Overlay {i} {why}"})
            continue
        if overlay["id"] is None:
            overlay["id"] = f"{ID_PREFIX}{outcome.next_id}"
            outcome.next_id += 1
        seen.add(overlay["id"])
        overlay.update(anchor_words(overlay["start"], overlay["end"], words or []))
        outcome.overlays.append(overlay)
    return outcome


# ── Words under an overlay ────────────────────────────────────────────────────


def _mid(w: dict[str, Any]) -> float:
    return (float(w["start"]) + float(w["end"])) / 2


def _word_ref(w: dict[str, Any] | None) -> dict[str, Any] | None:
    if w is None:
        return None
    return {"word": str(w["word"]).strip(), "start": round(float(w["start"]), 3)}


def anchor_words(start: float, end: float, words: list[dict[str, Any]]) -> dict[str, Any]:
    """The first and last spoken words under ``[start, end]``, by the midpoint rule.

    Recorded so a reader sees which words an overlay covers, and so a changed
    transcript can be spotted later.
    """
    under = [w for w in spoken_words(words) if start <= _mid(w) <= end]
    return {
        "starts_on": _word_ref(under[0] if under else None),
        "ends_on": _word_ref(under[-1] if under else None),
    }


def quote(start: float, end: float, words: list[dict[str, Any]], removed: list[tuple[float, float]]) -> str:
    """The kept words under ``[start, end]``, shortened to ``QUOTE_WORDS`` with an ellipsis."""
    kept = [
        str(w["word"]).strip() for w in spoken_words(words)
        if start <= _mid(w) <= end and not word_is_cut(w, removed)
    ]
    if len(kept) <= QUOTE_WORDS:
        return " ".join(kept)
    half = QUOTE_WORDS // 2
    return " ".join(kept[:half]) + " ... " + " ".join(kept[-half:])


def _word_near(t: float, words: list[dict[str, Any]]) -> dict[str, Any] | None:
    after = [w for w in spoken_words(words) if _mid(w) >= t]
    return after[0] if after else None


def _transcript_moved(overlay: dict[str, Any], words: list[dict[str, Any]]) -> bool:
    ref = overlay.get("starts_on")
    if not ref or not words:
        return False
    return not any(
        str(w["word"]).strip() == ref["word"] and abs(float(w["start"]) - ref["start"]) <= WORD_DRIFT_SECONDS
        for w in spoken_words(words)
    )


# ── Placing on the edited video ───────────────────────────────────────────────

Segment = tuple[float, float]

SHOWN = "shown"
SHORTENED = "shortened"
HIDDEN = "hidden"


@dataclass(frozen=True)
class Placement:
    """Where one overlay lands in the EDITED video, and what to tell Claude about it.

    ``at`` and ``seconds`` are edited time. ``status`` is ``shown`` (as saved),
    ``shortened`` (cuts under it or a short clip, still shown) or ``hidden``
    (not rendered). ``notes`` say what happened in words Claude can act on.
    """

    overlay: dict[str, Any]
    at: float
    seconds: float
    status: str
    notes: tuple[str, ...]

    @property
    def shown(self) -> bool:
        return self.status != HIDDEN


def _source_at(edited: float, kept: list[Segment]) -> float:
    """The source time that plays at edited time ``edited``."""
    spans = source_spans_for_window(kept, edited, edited + 1e-3)
    return spans[0][0] if spans else (kept[-1][1] if kept else 0.0)


def _cuts_inside(start: float, end: float, kept: list[Segment]) -> int:
    """How many joins of the edit fall strictly inside ``(start, end)``."""
    return sum(1 for (_, e), (s2, _) in zip(kept, kept[1:]) if start < e and s2 < end)


def place_overlay(overlay: dict[str, Any], kept: list[Segment], words: list[dict[str, Any]]) -> Placement:
    """Map one saved overlay onto the edited video. Pure.

    The rules, in order:

    - Start inside a cut: it appears when the talk resumes, on the next kept word.
    - Cuts inside the span: it covers the same words for less time, and hides
      the jump in the picture at each join.
    - Under ``MIN_SCREEN_SECONDS`` left on screen: hidden, not rendered.
    - A clip shorter than its time on screen: it ends with the clip and the talk
      shows again for the rest.
    """
    oid = overlay["id"]
    start, end = float(overlay["start"]), float(overlay["end"])
    at, start_cut = to_edited_time(start, kept)
    until, _ = to_edited_time(end, kept)
    seconds = max(0.0, until - at)
    notes: list[str] = []
    status = SHOWN

    if seconds < MIN_SCREEN_SECONDS:
        notes.append(
            f"{oid}: only {seconds:.1f}s of {clock(start)}-{clock(end)} is left after the cuts, so it "
            "is not shown. Move it to a stretch the edit keeps, or ask the creator whether to keep "
            "that part of the talk."
        )
        return Placement(overlay, at, 0.0, HIDDEN, tuple(notes))

    if start_cut:
        resumes = _source_at(at, kept)
        word = _word_near(resumes, words)
        on = f" on '{str(word['word']).strip()}'" if word else ""
        notes.append(
            f"{oid}: its start {clock(start)} is inside a cut, so it appears when the talk resumes at "
            f"{clock(resumes)}{on}. To start on a different word, move start."
        )
    joins = _cuts_inside(start, end, kept)
    if seconds < (end - start) - SHORTENED_EPS:
        status = SHORTENED
        if joins:
            fall = "cut falls" if joins == 1 else "cuts fall"
            notes.append(
                f"{oid}: {joins} {fall} under it. It covers the same words "
                f"in {seconds:.1f}s instead of {end - start:.1f}s and hides the jump in the picture there."
            )

    if overlay["kind"] == VIDEO:
        length = float(overlay["clip_out"]) - float(overlay["clip_in"])
        if length < seconds - SHORTENED_EPS:
            status = SHORTENED
            ends = _source_at(at + length, kept)
            notes.append(
                f"{oid}: the clip plays {length:.1f}s but the span is {seconds:.1f}s on screen, so the "
                f"talk shows again from {clock(ends)}. Play more of the clip (clip_out), or end the "
                f"overlay at {ends:.2f}."
            )
            seconds = length

    if _transcript_moved(overlay, words):
        notes.append(
            f"{oid}: the transcript changed since it was placed and '{overlay['starts_on']['word']}' "
            "is no longer at its start. Read that stretch again and check start and end."
        )
    return Placement(overlay, at, seconds, status, tuple(notes))


def place_on_edit(
    overlays: list[dict[str, Any]], kept: list[Segment], words: list[dict[str, Any]]
) -> list[Placement]:
    """Every overlay placed on the edited video, bottom layer first, then by time.

    That order is the order they are drawn: a later one covers an earlier one.
    """
    placed = [place_overlay(o, kept, words) for o in overlays]
    return sorted(placed, key=lambda p: (p.overlay["layer"], p.at, p.overlay["id"]))


def describe(placement: Placement, words: list[dict[str, Any]], removed: list[Segment]) -> dict[str, Any]:
    """One overlay as get_overlays shows it: the saved fields plus where it lands now."""
    o = placement.overlay
    return {
        **o,
        "clock": f"{clock(o['start'])}-{clock(o['end'])}",
        "on_screen": {"at": round(placement.at, 3), "seconds": round(placement.seconds, 3)},
        "status": placement.status,
        "notes": list(placement.notes),
        "over": quote(o["start"], o["end"], words, removed),
    }


# What the treatment page calls each kind: the creator's words for her own
# files, where the saved list uses the kind of media.
PAGE_KINDS = {IMAGE: "photo", VIDEO: "clip"}


def page_row(placement: Placement) -> dict[str, Any]:
    """One overlay as the treatment page shows it: ``{id, name, kind, start, end, clock, status, tag}``.

    The page only looks: it gets the file's name and never its folder.
    ``start`` and ``end`` are source seconds, ``clock`` is ``m:ss`` of the
    start, and ``tag`` is the label shown on the video, such as
    ``Photo: berlin.jpg``.
    """
    o = placement.overlay
    kind = PAGE_KINDS.get(o["kind"], PAGE_KINDS[IMAGE])
    name = Path(str(o["file"])).name
    return {
        "id": o["id"], "name": name, "kind": kind,
        "start": round(float(o["start"]), 3), "end": round(float(o["end"]), 3),
        "clock": clock(float(o["start"])), "status": placement.status,
        "tag": f"{kind.capitalize()}: {name}",
    }


# ── Persistence ───────────────────────────────────────────────────────────────


def empty_overlays() -> dict[str, Any]:
    return {"version": OVERLAYS_VERSION, "overlays": [], "next_id": 1}


def load_overlays(project: Project) -> dict[str, Any]:
    """The saved overlays, or an empty list. Raises StudioError when the file is unusable."""
    path = overlays_path(project)
    if not path.exists():
        return empty_overlays()
    fix = "Call set_overlays to replace it."
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise StudioError(f"The saved {OVERLAYS_FILE} is not valid JSON. {fix}") from None
    if not isinstance(data, dict) or not isinstance(data.get("overlays"), list):
        raise StudioError(f"The saved {OVERLAYS_FILE} holds no overlay list. {fix}")
    for i, o in enumerate(data["overlays"]):
        ok = isinstance(o, dict) and all(k in o for k in ("id", "file", "kind", "start", "end", "layer"))
        if not ok:
            raise StudioError(f"The saved {OVERLAYS_FILE} has a bad overlay at index {i}. {fix}")
    return data


def save_overlays(
    project: Project, outcome: OverlayOutcome, *, duration: float, requested: list[Any]
) -> dict[str, Any]:
    """Write overlays.json atomically and return what was written."""
    data = {
        "version": OVERLAYS_VERSION,
        "video": str(project.video),
        "duration": round(duration, 3),
        "updated_at": now_iso(),
        "next_id": outcome.next_id,
        "overlays": outcome.overlays,
        "requested": requested,
        "rejected": outcome.rejected,
    }
    write_json_atomic(overlays_path(project), data)
    return data


def recheck_files(overlays: list[dict[str, Any]], *, probe: Prober = probe_media) -> None:
    """Before a render: every file still there and still what it was when saved.

    Raises StudioError naming the overlay and the fix.
    """
    for o in overlays:
        try:
            path = resolve_media_path(o["file"])
            media = probe(path)
        except (OverlayRejected, StudioError) as exc:
            raise StudioError(
                f"Overlay {o['id']} {exc} Put the file back, or call set_overlays without {o['id']}."
            ) from None
        saved = o.get("media") or {}
        if media.kind != o["kind"] or (media.width, media.height) != (saved.get("width"), saved.get("height")):
            raise StudioError(
                f"Overlay {o['id']}'s file {path.name} changed since it was placed. "
                "Call set_overlays again so it is checked and sized afresh."
            )
        if o["kind"] == VIDEO and float(o["clip_out"]) > float(media.duration or 0) + END_TOLERANCE:
            raise StudioError(
                f"Overlay {o['id']}'s clip {path.name} is now {media.duration:.2f}s long, shorter than "
                f"its clip_out {o['clip_out']}. Call set_overlays again with a clip_out inside it."
            )
