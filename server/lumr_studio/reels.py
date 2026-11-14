"""Reels: several clips joined into one video that every other tool edits as it is.

A creator records one video as several clips, with retakes inside them.
``assemble`` joins the pieces they name, in order, into one file, the reel, and
writes a small sequence file beside it that records which reel seconds came
from which clip. The reel is an ordinary video to transcribe, read, cut,
check, preview, review and render. Only ``get_edit`` and ``read_transcript``
look for the sequence file, to show where each clip starts.

    <lumr home>/studio/reels/<slug>/
        <slug>-<id>.mp4          the reel
        <slug>-<id>.reel.json    the sequence file, written after the reel

``id`` is the first 8 hex of a sha1 over the name and each piece (the clip's
resolved path, byte size, mtime, and the start and end). The same sequence
always lands on the same file, so asking twice builds once; a different
sequence is a different file. Nothing is ever overwritten.

Whole clips that agree in every detail a copy depends on are joined by
copying: no quality is lost and it takes seconds. Otherwise, because a piece
has a start or an end or the clips differ, one ffmpeg run trims, matches size,
frame rate and sound, and encodes once, at a quality high enough for the
export that follows to encode it again. Each encoded piece is a whole number
of frames long, and that length is what the sequence file records.

A clip recorded in HDR (HLG or PQ) or in wide color (BT.2020) is turned into
standard color (bt709) inside that same ffmpeg run, by macOS's own converter
(the ``scale_vt`` filter), so it matches the other clips and what QuickTime
shows. Such a sequence is never copied, every piece of it ends up bt709, and
the reel is tagged bt709. The clip is not changed and no working copy is made.

Each build writes its own ``<slug>-<id>.<random>.partial.mp4`` and moves it to
the reel's name only if nothing is there. A partial left by a build that died
is never touched: Studio did not make it in this run, so it does not remove it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

from lumr_studio.engine.paths import lumr_home
from lumr_studio.errors import StudioError
from lumr_studio.jobs import JobHandle, Work
from lumr_studio.overlay_render import FALLBACK_FPS, X264_ARGS, run_ffmpeg
from lumr_studio.overlays import PROBE_TIMEOUT_SECONDS, VIDEO, probe_media
from lumr_studio.project import Project, now_iso, project_dir_name, projects_root, write_json_atomic

# The kind of job that builds a reel.
KIND = "assemble"
REELS_FOLDER = "reels"
SEQUENCE_SUFFIX = ".reel.json"
SEQUENCE_VERSION = 1
MAX_NAME_CHARS = 60
MAX_PIECES = 200
NAME_PATTERN = re.compile(r"[A-Za-z0-9 _-]+")

COPY = "copy"
ENCODE = "encode"
# A joined file this much longer or shorter than its pieces add up to is wrong.
LENGTH_TOLERANCE = 0.5
# An end this far past the clip's last frame is the clip's end: transcript times round.
END_TOLERANCE = 0.05
# The reel gets encoded again at export, so it is made at a quality that keeps the first encode out of sight.
REEL_CRF = "16"
REEL_AUDIO_ARGS = ("-c:a", "aac", "-b:a", "256k")
REEL_SAMPLE_RATE = 48000

# How a clip's picture can differ from standard color (bt709), by what ffprobe calls it.
HDR = "HDR"
WIDE_COLOR = "wide color"
HDR_TRANSFERS = ("arib-std-b67", "smpte2084")  # HLG and PQ
WIDE_PRIMARIES = ("bt2020",)
WIDE_MATRICES = ("bt2020nc", "bt2020c")
UNNAMED = ("", "unknown", "unspecified")  # a color tag the clip leaves out
# The ffmpeg filter that hands frames to macOS to convert. Homebrew's ffmpeg has it from 6.1 on.
CONVERTER = "scale_vt"
# Part of the id of a sequence with a converted clip, so a reel joined before the conversion existed is not taken for this one.
CONVERTED_MARK = "standard color 1"
# The names ffmpeg's scale filter takes for a clip's YCbCr matrix, by what ffprobe calls it.
SCALE_MATRICES = {
    "bt709": "bt709", "smpte170m": "smpte170m", "bt470bg": "bt470", "fcc": "fcc", "smpte240m": "smpte240m",
}

# What to run when Studio can't convert a clip itself: every Mac has avconvert.
COPY_BY_HAND = "avconvert --preset Preset1920x1080 --source CLIP --output COPY.mov"

USE_THE_REEL = (
    "Use reel as the video_path for every other tool: transcribe it, read it, cut it, review it. "
    "read_transcript marks where each clip starts."
)


# ── Pieces ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Streams:
    """The stream details that decide whether clips can be joined without encoding."""

    video_codec: str
    audio_codec: str
    width: int
    height: int
    rotation: int
    fps: str  # a fraction as ffmpeg writes it, such as 30000/1001
    pix_fmt: str
    sample_rate: int
    channels: int
    # The next five can differ between clips that agree on all of the above and
    # still break a copy: the concat demuxer reads every file in the first
    # file's time base, so a clip counted in another unit lands in the wrong place.
    video_time_base: str = ""
    audio_time_base: str = ""
    audio_profile: str = ""
    sample_aspect_ratio: str = ""
    field_order: str = ""
    # How the picture's colors are written. They never decide whether clips agree
    # (two clips equal in all of the above are equal with these left out); they
    # decide whether a clip has to be turned into standard color.
    color_transfer: str = field(default="", compare=False)
    color_primaries: str = field(default="", compare=False)
    color_space: str = field(default="", compare=False)


@dataclass(frozen=True)
class Piece:
    """One stretch of one clip, checked: ``start`` and ``end`` are seconds into the clip."""

    clip: Path  # resolved
    size: int
    mtime_ns: int
    duration: float  # of the whole clip
    start: float
    end: float
    width: int  # as displayed, after any turn a phone recorded
    height: int
    streams: Streams

    @property
    def length(self) -> float:
        return self.end - self.start

    @property
    def whole(self) -> bool:
        return self.start == 0.0 and self.end == self.duration

    @property
    def converted_from(self) -> str | None:
        """``HDR`` or ``wide color`` when the clip's picture has to be turned into standard color, else None.

        HDR is a clip whose brightness is written as HLG or PQ. Wide color is
        any other clip in BT.2020 colors: read as bt709 its colors come out
        pale, the same fault in a milder form, so it takes the same road.
        """
        streams = self.streams
        if streams.color_transfer in HDR_TRANSFERS:
            return HDR
        if streams.color_primaries in WIDE_PRIMARIES or streams.color_space in WIDE_MATRICES:
            return WIDE_COLOR
        return None


def _finite_seconds(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _rotation(stream: dict[str, Any]) -> int:
    for side in stream.get("side_data_list") or []:
        if _finite_seconds(side.get("rotation")):
            return round(side["rotation"]) % 360
    try:
        return round(float((stream.get("tags") or {}).get("rotate", 0))) % 360
    except ValueError:
        return 0


def _frame_rate(stream: dict[str, Any]) -> str:
    """The nominal frame rate as a fraction, 25 when the file doesn't say."""
    for key in ("r_frame_rate", "avg_frame_rate"):
        try:
            rate = Fraction(str(stream.get(key) or ""))
        except (ValueError, ZeroDivisionError):
            continue
        if rate > 0:
            return f"{rate.numerator}/{rate.denominator}"
    return f"{Fraction(FALLBACK_FPS).numerator}/{Fraction(FALLBACK_FPS).denominator}"


def probe_streams(path: Path, *, run: Callable[..., Any] = subprocess.run) -> Streams:
    """Codec names, size, frame rate, pixel format and sound details of a clip, from ffprobe.

    ``probe_media`` has the length and the displayed size; this has what it leaves out.
    """
    try:
        out = run(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=PROBE_TIMEOUT_SECONDS,
        )
        data = json.loads(out.stdout or "{}")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        raise StudioError(f"Could not read the streams of {path.name} with ffprobe. Check that it plays in a video player.") from None
    streams = data.get("streams") or []
    video = next(
        (s for s in streams if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")), {}
    )
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
    return Streams(
        video_codec=str(video.get("codec_name") or ""),
        audio_codec=str(audio.get("codec_name") or ""),
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        rotation=_rotation(video),
        fps=_frame_rate(video),
        pix_fmt=str(video.get("pix_fmt") or ""),
        sample_rate=int(audio.get("sample_rate") or 0),
        channels=int(audio.get("channels") or 0),
        video_time_base=str(video.get("time_base") or ""),
        audio_time_base=str(audio.get("time_base") or ""),
        audio_profile=str(audio.get("profile") or ""),
        sample_aspect_ratio=str(video.get("sample_aspect_ratio") or ""),
        field_order=str(video.get("field_order") or ""),
        color_transfer=str(video.get("color_transfer") or ""),
        color_primaries=str(video.get("color_primaries") or ""),
        color_space=str(video.get("color_space") or ""),
    )


def can_convert_color(*, run: Callable[..., Any] = subprocess.run) -> bool:
    """Whether this ffmpeg has the filter that turns HDR into standard color."""
    try:
        out = run(
            ["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True, timeout=PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return any(line.split()[1:2] == [CONVERTER] for line in (out.stdout or "").splitlines())


def check_name(name: Any) -> str:
    """The name, trimmed. Raises StudioError when it is not 1 to 60 plain letters (A to Z), digits, spaces, ``-`` or ``_``."""
    text = name.strip() if isinstance(name, str) else ""
    if not text or len(text) > MAX_NAME_CHARS or not NAME_PATTERN.fullmatch(text) or not slug_for(text):
        raise StudioError(
            f"The name {name!r} won't do. Use 1 to {MAX_NAME_CHARS} plain letters (A to Z), digits, spaces, - or _, "
            "such as 'bread week 5'."
        )
    return text


def slug_for(name: str) -> str:
    """The name lower case, with each run of anything but letters and digits turned into one ``-``."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _check_seconds(value: Any, label: str, where: str) -> float | None:
    if value is None:
        return None
    if not _finite_seconds(value):
        raise StudioError(f"{where}: {label} must be a number of seconds, got {value!r}.")
    return round(float(value), 3)


def check_pieces(
    raw: Any,
    *,
    probe: Callable[[Path], Any] = probe_media,
    streams_of: Callable[[Path], Streams] = probe_streams,
) -> list[Piece]:
    """Check every piece of a sequence against its clip and return them in order.

    Each piece is ``{clip, start?, end?}``. Raises StudioError naming the
    first piece that can't be used and what to do about it. A clip is probed
    once however many pieces use it.
    """
    if not isinstance(raw, list) or not raw:
        raise StudioError("assemble needs at least one piece. Pass the clips in play order, each as {clip, start, end}.")
    if len(raw) > MAX_PIECES:
        raise StudioError(f"assemble takes at most {MAX_PIECES} pieces and got {len(raw)}. Join the video in two steps.")
    seen: dict[Path, tuple[Any, Streams, int, int]] = {}
    pieces: list[Piece] = []
    for number, item in enumerate(raw, start=1):
        where = f"Piece {number}"
        if not isinstance(item, dict):
            raise StudioError(f"{where} is not an object. Pass {{clip, start, end}} with an absolute clip path.")
        given = item.get("clip")
        if not isinstance(given, str) or not given.strip():
            raise StudioError(f"{where} has no clip. Pass the absolute path to the clip.")
        path = Path(given).expanduser()
        if not path.is_absolute():
            raise StudioError(f"{where}: clip {given!r} is relative. Pass the absolute path to the clip.")
        if not path.exists():
            raise StudioError(f"{where}: no file at {path}. Check the path with the creator.")
        if not path.is_file():
            raise StudioError(f"{where}: {path} is a folder, not a clip. Pass the path to the video file itself.")
        clip = path.resolve()
        if clip not in seen:
            try:
                media = probe(clip)
            except StudioError:
                raise StudioError(
                    f"{where}: {clip.name} is not a video Lumr Studio can read. Pass a video file such as MP4 or MOV."
                ) from None
            if media.kind != VIDEO:
                raise StudioError(f"{where}: {clip.name} is a picture, not a video. Leave it out of the sequence.")
            if not media.has_sound:
                raise StudioError(
                    f"{where}: {clip.name} has no sound. Lumr Studio reads the words, so every clip needs its sound. "
                    "Leave this clip out."
                )
            info = clip.stat()
            seen[clip] = (media, streams_of(clip), info.st_size, info.st_mtime_ns)
        media, streams, size, mtime_ns = seen[clip]
        length = round(media.duration, 3)
        start = _check_seconds(item.get("start"), "start", where)
        end = _check_seconds(item.get("end"), "end", where)
        start = 0.0 if start is None else start
        if end is not None and length < end <= length + END_TOLERANCE:
            end = length
        end = length if end is None else end
        if not 0 <= start < length:
            raise StudioError(f"{where}: start {start:g} is outside {clip.name}, which runs 0 to {length:g} seconds.")
        if end > length:
            raise StudioError(f"{where}: end {end:g} is past the end of {clip.name}, which runs {length:g} seconds.")
        if end <= start:
            raise StudioError(f"{where}: end {end:g} must come after start {start:g} in {clip.name}.")
        pieces.append(Piece(
            clip=clip, size=size, mtime_ns=mtime_ns, duration=length, start=start, end=end,
            width=media.width, height=media.height, streams=streams,
        ))
    return pieces


def sequence_id(name: str, pieces: list[Piece]) -> str:
    """Eight hex characters that change when the name, a clip, or a piece's start or end changes.

    A sequence with a clip that is turned into standard color carries a mark
    of that too, so a reel joined before Studio converted such clips, with
    the HDR picture written as it was, is not answered as this reel.
    """
    parts: list[Any] = [name, [[str(p.clip), p.size, p.mtime_ns, p.start, p.end] for p in pieces]]
    if any(p.converted_from for p in pieces):
        parts.append(CONVERTED_MARK)
    return hashlib.sha1(json.dumps(parts).encode("utf-8")).hexdigest()[:8]


def how_to_join(pieces: list[Piece]) -> str:
    """``copy`` when every piece is a whole clip and the clips agree on every detail a copy depends on, else ``encode``.

    A piece with a start or an end is never copied: it needs a cut, and a cut needs an encode.
    Nor is a sequence with a clip in HDR or wide color, two such clips alone
    included: a copy would keep the picture as it was recorded.
    """
    if any(p.converted_from for p in pieces):
        return ENCODE
    first = pieces[0].streams
    agree = all(p.whole and p.streams == first for p in pieces)
    return COPY if agree and first.video_codec == "h264" and first.audio_codec == "aac" else ENCODE


# ── Where a reel goes ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReelPlan:
    """A checked sequence and the files it will land in."""

    name: str
    pieces: list[Piece]
    how: str
    reel: Path
    sequence: Path

    @property
    def lengths(self) -> list[float]:
        """What each piece adds to the reel. A copy adds the piece's own length; an encode adds whole frames."""
        if self.how == COPY:
            return [p.length for p in self.pieces]
        return [frames_in(p.length, self.pieces[0].streams.fps) / float(Fraction(self.pieces[0].streams.fps)) for p in self.pieces]

    @property
    def total(self) -> float:
        return sum(self.lengths)

    @property
    def project(self) -> Project:
        """The reel's project, which exists before the reel does: the job needs one for its receipt."""
        return Project(video=self.reel, root=projects_root() / project_dir_name(self.reel))


def sequence_path_for(video: Path) -> Path:
    """Where the sequence file of a reel sits: beside it, same stem."""
    return video.with_suffix(SEQUENCE_SUFFIX)


def frames_in(seconds: float, fps: str) -> int:
    """How many frames of ``fps`` a stretch of ``seconds`` rounds to; at least one."""
    return max(1, round(seconds * float(Fraction(fps))))


def plan_reel(name: Any, raw_pieces: Any, *, can_convert: Callable[[], bool] = can_convert_color, **probes: Any) -> ReelPlan:
    """Check the name and the pieces and work out the files. Writes nothing.

    Raises StudioError when a clip has to be turned into standard color and this ffmpeg can't do it.
    """
    name = check_name(name)
    pieces = check_pieces(raw_pieces, **probes)
    odd = next((p for p in pieces if p.converted_from), None)
    if odd is not None and not can_convert():
        raise StudioError(
            f"{odd.clip.name} was recorded in {odd.converted_from}, and the ffmpeg on this Mac can't turn that into "
            f"standard color: it has no {CONVERTER} filter. Install the current ffmpeg with Homebrew (brew install ffmpeg), "
            f"or make a standard color copy of the clip and join that ({COPY_BY_HAND})."
        )
    folder = (lumr_home() / "studio" / REELS_FOLDER / slug_for(name)).resolve()
    reel = folder / f"{slug_for(name)}-{sequence_id(name, pieces)}.mp4"
    return ReelPlan(name, pieces, how_to_join(pieces), reel, sequence_path_for(reel))


# ── The sequence file ─────────────────────────────────────────────────────────


def _sequence_record(plan: ReelPlan) -> dict[str, Any]:
    at = 0.0
    rows = []
    for p, length in zip(plan.pieces, plan.lengths):
        row: dict[str, Any] = {
            "clip": str(p.clip), "clip_duration": p.duration, "start": p.start, "end": p.end,
            "reel_start": round(at, 3), "reel_end": round(at + length, 3),
        }
        if p.converted_from:
            row["converted_from"] = p.converted_from
        rows.append(row)
        at += length
    return {
        "version": SEQUENCE_VERSION, "name": plan.name, "made_at": now_iso(), "how": plan.how,
        "duration": round(at, 3), "pieces": rows,
    }


def read_sequence(path: Path) -> dict[str, Any]:
    """Read and check a sequence file. Raises StudioError naming it when it can't be used."""
    broken = (
        f"The sequence file {path} can't be read. Move it and the reel beside it somewhere else, "
        "then call assemble again."
    )
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise StudioError(broken) from None
    pieces = record.get("pieces") if isinstance(record, dict) else None
    keys = ("start", "end", "reel_start", "reel_end")
    listed = isinstance(pieces, list) and bool(pieces) and all(
        isinstance(p, dict) and isinstance(p.get("clip"), str) and all(_finite_seconds(p.get(k)) for k in keys)
        for p in pieces
    )
    if not (
        listed and record.get("version") == SEQUENCE_VERSION and isinstance(record.get("name"), str)
        and isinstance(record.get("how"), str) and _finite_seconds(record.get("duration"))
    ):
        raise StudioError(broken)
    return record


def color_note(rows: list[dict[str, Any]]) -> str:
    """What to tell the creator about the clips that were turned into standard color; empty when none was."""
    kinds: dict[str, str] = {}  # clip name -> HDR or wide color, in play order, each clip once
    for row in rows:
        if isinstance(row.get("converted_from"), str):
            kinds.setdefault(Path(row["clip"]).name, row["converted_from"])
    if not kinds:
        return ""
    how = " or ".join(sorted(set(kinds.values())))
    if len(kinds) == 1:
        told = (
            f"{next(iter(kinds))} was recorded in {how}; the joined video shows it in standard color. "
            "The clip itself was not changed."
        )
    else:
        told = (
            f"{', '.join(kinds)} were recorded in {how}; the joined video shows them in standard color. "
            "The clips themselves were not changed."
        )
    return f"{told} Tell the creator in one sentence."


def answer_for(status: str, plan: ReelPlan, record: dict[str, Any]) -> dict[str, Any]:
    """What assemble tells the caller about a reel: ``exists`` when it was there, ``built`` when the job made it.

    A piece whose clip was turned into standard color carries ``converted_from``, and ``note`` says so in plain words.
    """
    keys = ("clip", "start", "end", "reel_start", "reel_end", "converted_from")
    return {
        "status": status,
        "reel": str(plan.reel),
        "name": record["name"],
        "duration": record["duration"],
        "how": record["how"],
        "pieces": [{k: p[k] for k in keys if k in p} for p in record["pieces"]],
        "sequence_file": str(plan.sequence),
        "note": " ".join(filter(None, [USE_THE_REEL, color_note(record["pieces"])])),
    }


def existing_answer(plan: ReelPlan) -> dict[str, Any] | None:
    """The ``exists`` answer when this sequence was built before, or None when it has to be built.

    Raises StudioError when a reel file is there without its sequence file,
    which means a build was cut short: the file is never overwritten.
    """
    if not plan.reel.exists():
        return None
    if not plan.sequence.exists():
        raise StudioError(
            f"{plan.reel} is already there without {plan.sequence.name} beside it, so it may be half made. "
            "Lumr Studio never overwrites a file. Move the reel somewhere else, then call assemble again."
        )
    return answer_for("exists", plan, read_sequence(plan.sequence))


def reel_for(video: Path) -> dict[str, Any] | None:
    """Which clips a video is made of, or None when no sequence file sits beside it.

    ``{name, pieces: [{clip, reel_start, reel_end}]}``. A sequence file that
    can't be read counts as none: it must never stop an edit.
    """
    path = sequence_path_for(video)
    if not path.is_file():
        return None
    try:
        record = read_sequence(path)
    except StudioError:
        return None
    return {
        "name": record["name"],
        "pieces": [{k: p[k] for k in ("clip", "reel_start", "reel_end")} for p in record["pieces"]],
    }


def clip_markers(reel: dict[str, Any]) -> list[tuple[float, str]]:
    """The lines read_transcript prints where each clip of a reel starts, with their times."""
    pieces = reel["pieces"]
    return [
        (float(p["reel_start"]), f"== clip {i} of {len(pieces)}: {Path(p['clip']).name} starts {float(p['reel_start']):.2f} ==")
        for i, p in enumerate(pieces, start=1)
    ]


# ── ffmpeg ────────────────────────────────────────────────────────────────────

QUIET = ("-y", "-nostdin", "-loglevel", "error", "-nostats", "-progress", "pipe:1")


def listing_line(path: Path) -> str:
    """One ``file`` line for ffmpeg's concat list. A single quote in the path is closed, escaped and reopened."""
    return "file '" + str(path).replace("'", "'\\''") + "'\n"


def copy_command(listing: Path, out: Path) -> list[str]:
    """Join the clips in ``listing`` as they are: the first picture and the first sound of each."""
    return [
        "ffmpeg", *QUIET, "-f", "concat", "-safe", "0", "-i", str(listing),
        "-map", "0:v:0", "-map", "0:a:0", "-c", "copy", "-movflags", "+faststart", str(out),
    ]


# Hand the frames to macOS, which turns HDR and wide color into bt709 the way QuickTime does, and take them back.
# The range is set first because the upload reads every frame as video range. scale_vt converts the matrix too
# but leaves the old name on the frame, so setparams puts the right one on.
TO_STANDARD_COLOR = (
    "scale=out_range=tv,format=p010le,hwupload,"
    f"{CONVERTER}=color_matrix=bt709:color_primaries=bt709:color_transfer=bt709,"
    "hwdownload,format=p010le,setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709"
)
CONVERTER_DEVICE = ("-init_hw_device", "videotoolbox=vt", "-filter_hw_device", "vt")
BT709_TAGS = "setparams=color_primaries=bt709:color_trc=bt709"


def to_standard_color(piece: Piece) -> str:
    """The filters that turn this piece's picture into bt709.

    macOS converts by the frame's own tags and leaves a frame alone when one
    is missing. HLG and PQ are only ever written in BT.2020 colors, so an HDR
    clip that leaves its primaries or its matrix out is given them first.
    """
    missing = []
    if piece.converted_from == HDR:
        if piece.streams.color_primaries in UNNAMED:
            missing.append("color_primaries=bt2020")
        if piece.streams.color_space in UNNAMED:
            missing.append("colorspace=bt2020nc")
    return (f"setparams={':'.join(missing)}," if missing else "") + TO_STANDARD_COLOR


def matrix_of(piece: Piece) -> str:
    """The YCbCr matrix a plain clip is written in, as the scale filter names it.

    A clip that doesn't say is read the way players read it: bt709 when it is HD, the older smpte170m when smaller.
    """
    named = SCALE_MATRICES.get(piece.streams.color_space)
    if named:
        return named
    return "bt709" if piece.streams.width >= 1280 or piece.streams.height > 576 else "smpte170m"


def encode_command(pieces: list[Piece], out: Path) -> list[str]:
    """Trim each piece, fit it to the first clip's size and frame rate and to 48 kHz stereo, join, and encode once.

    Each piece is its own input, seeked to its start, so a piece late in a
    long clip doesn't decode all that comes before it. Each piece then ends
    on a whole frame count (``frames_in``), with its sound padded or cut to
    the same length, so the reel's pieces are exactly as long as the
    sequence file says. Without that, ffmpeg rounds each piece's picture up
    to whole frames and the error adds up over the reel.

    When a piece is in HDR or wide color (``Piece.converted_from``) the whole
    reel is made bt709: that piece goes through ``to_standard_color`` first,
    every other piece is brought to the bt709 matrix by the scale it already
    has, and each is tagged bt709, so the reel's own tags are true of every
    frame. A sequence with no such piece is built exactly as before.
    """
    first = pieces[0]
    fps = first.streams.fps
    width, height = first.width - first.width % 2, first.height - first.height % 2
    converting = any(p.converted_from for p in pieces)
    inputs: list[str] = []
    graph: list[str] = []
    for i, p in enumerate(pieces):
        frames = frames_in(p.length, fps)
        samples = round(frames / float(Fraction(fps)) * REEL_SAMPLE_RATE)
        if p.start > 0:
            inputs += ["-ss", f"{p.start:.3f}"]
        inputs += ["-t", f"{p.length:.3f}", "-i", str(p.clip)]
        before = matrix = tags = ""
        if converting:
            before = f"{to_standard_color(p)}," if p.converted_from else ""
            matrix = f":in_color_matrix={'bt709' if p.converted_from else matrix_of(p)}:out_color_matrix=bt709"
            tags = f",{BT709_TAGS}"
        graph.append(
            f"[{i}:v:0]setpts=PTS-STARTPTS,{before}scale={width}:{height}:force_original_aspect_ratio=decrease{matrix},"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},format=yuv420p{tags},"
            f"tpad=stop_mode=clone:stop=-1,trim=end_frame={frames}[v{i}]"
        )
        graph.append(
            f"[{i}:a:0]asetpts=PTS-STARTPTS,aresample={REEL_SAMPLE_RATE},aformat=channel_layouts=stereo,"
            f"apad,atrim=end_sample={samples}[a{i}]"
        )
    joined = "".join(f"[v{i}][a{i}]" for i in range(len(pieces))) + f"concat=n={len(pieces)}:v=1:a=1[v][a]"
    return [
        "ffmpeg", *QUIET, *(CONVERTER_DEVICE if converting else ()), *inputs,
        "-filter_complex_threads", "1", "-filter_complex", ";".join([*graph, joined]),
        "-map", "[v]", "-map", "[a]", *X264_ARGS, "-crf", REEL_CRF, *REEL_AUDIO_ARGS,
        "-movflags", "+faststart", str(out),
    ]


# ── The job ───────────────────────────────────────────────────────────────────


class WrongJoin(StudioError):
    """The joined file is not the length of its pieces, picture or sound."""


def stream_seconds(path: Path, *, run: Callable[..., Any] = subprocess.run) -> dict[str, float]:
    """The length of the picture (``video``) and of the sound (``audio``) in a file, each on its own.

    A stream whose length the file doesn't say is left out.
    """
    try:
        out = run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=PROBE_TIMEOUT_SECONDS,
        )
        streams = json.loads(out.stdout or "{}").get("streams") or []
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return {}
    seconds: dict[str, float] = {}
    for stream in streams:
        try:
            length = float(stream.get("duration"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(length) and stream.get("codec_type") in ("video", "audio"):
            seconds.setdefault(str(stream["codec_type"]), length)
    return seconds


def new_partial(reel: Path) -> Path:
    """A file this build alone owns, beside the reel: ``<stem>.<random>.partial.mp4``, made empty and new.

    ffmpeg writes into it and it is moved to the reel's name at the end. It
    is never a name another build could be writing, so a leftover ffmpeg from
    a server that died can never end up renamed into place by a later build.
    """
    while True:
        path = reel.with_name(f"{reel.stem}.{secrets.token_hex(4)}.partial.mp4")
        try:
            os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666))
        except FileExistsError:
            continue
        return path


def put_in_place(partial: Path, reel: Path) -> None:
    """Move ``partial`` to ``reel``, which must not exist. Raises FileExistsError when it does.

    A hard link fails if the name is taken, so there is no gap between
    looking and moving. A disk that can't link (some external drives) gets
    the next best thing: the name is claimed first with an exclusive create,
    which also fails if the name is taken, and the partial is then renamed
    over that empty file of our own. Only a file that lands on the name in
    the instant between the claim and the rename could be replaced, and only
    by a rename; a claim made by another build is never taken over.
    """
    try:
        os.link(partial, reel)
    except FileExistsError:
        raise
    except OSError:
        os.close(os.open(reel, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666))  # FileExistsError if it is taken
        try:
            partial.rename(reel)
        except BaseException:
            reel.unlink(missing_ok=True)  # the empty file just made above
            raise
        return
    partial.unlink()


def builder(
    plan: ReelPlan,
    *,
    run: Callable[..., None] = run_ffmpeg,
    probe: Callable[[Path], Any] = probe_media,
    seconds_of: Callable[[Path], dict[str, float]] = stream_seconds,
) -> Work:
    """The job's work: build the reel beside its final name, check its length, move it into place, write the sequence file.

    The only files this removes are its own: the partial it made and the
    concat list. A reel that is already there is never replaced. A copy that
    comes out the wrong length is built again as an encode, once.
    """

    def make(current: ReelPlan, partial: Path, handle: JobHandle) -> None:
        listing: Path | None = None
        try:
            if current.how == COPY:
                fd, name = tempfile.mkstemp(prefix=f"{current.reel.stem}.", suffix=".list.txt", dir=current.reel.parent)
                listing = Path(name)
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.writelines(listing_line(p.clip) for p in current.pieces)
                command = copy_command(listing, partial)
            else:
                command = encode_command(current.pieces, partial)
            check = "Check that every clip plays through in a video player."
            odd = next((p for p in current.pieces if p.converted_from), None)
            if odd is not None:
                check += (
                    f" If they all do, turning {odd.clip.name} from {odd.converted_from} into standard color is what failed: "
                    f"make a standard color copy of it and join that ({COPY_BY_HAND})."
                )
            run(command, partial, total=current.total, on_progress=handle.set_progress, doing="join the clips into", check=check)
            check_length(current, partial)
        finally:
            if listing is not None:
                listing.unlink(missing_ok=True)

    def check_length(current: ReelPlan, partial: Path) -> None:
        total = current.total
        made = probe(partial).duration
        advice = (
            "Your clips were not changed. If the clips come from different cameras or apps, join them in two steps; "
            "if a piece has a start or an end, try another one."
        )
        if abs(made - total) > LENGTH_TOLERANCE:
            raise WrongJoin(
                f"The joined video came out {made:.1f} seconds long but the clips add up to {total:.1f}, "
                f"so it was thrown away. {advice}"
            )
        seconds = seconds_of(partial)
        # A copy keeps the last clip as it is, so a stream that ends before the clip does ends the reel early too.
        # That is the clip's own shortfall, not a bad join: forgive that much and hold the rest to the length.
        last = seconds_of(current.pieces[-1].clip) if current.how == COPY else {}
        for key, which in (("video", "picture"), ("audio", "sound")):
            expected = total - max(0.0, current.pieces[-1].duration - last.get(key, current.pieces[-1].duration))
            if key in seconds and abs(seconds[key] - expected) > LENGTH_TOLERANCE:
                raise WrongJoin(
                    f"The joined video's {which} is {seconds[key]:.1f} seconds long but the clips add up to {expected:.1f}, "
                    f"so it was thrown away. {advice}"
                )

    def work(handle: JobHandle) -> dict[str, Any]:
        plan.reel.parent.mkdir(parents=True, exist_ok=True)
        current = plan
        partial = new_partial(plan.reel)
        try:
            try:
                make(current, partial, handle)
            except WrongJoin:
                if current.how != COPY:
                    raise
                current = replace(current, how=ENCODE)
                make(current, partial, handle)
            try:
                put_in_place(partial, current.reel)
            except FileExistsError:
                partial.unlink(missing_ok=True)
                if current.sequence.exists():
                    return answer_for("exists", current, read_sequence(current.sequence))
                raise StudioError(
                    f"{current.reel} appeared while the clips were being joined, and {current.sequence.name} is not beside it yet. "
                    "Lumr Studio never overwrites a file. If another Lumr Studio is joining the same clips, "
                    "wait a minute and call assemble again."
                ) from None
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        record = _sequence_record(current)
        write_json_atomic(current.sequence, record)
        return answer_for("built", current, record)

    return work
