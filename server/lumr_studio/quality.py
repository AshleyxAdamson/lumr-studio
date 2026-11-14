"""The two qualities a finished video, or an encoded reel, can be made at.

``standard`` is the everyday export and what every file was before there was
a choice. ``master`` is for uploading: as close to the footage as an export
gets, and a much bigger file. ``render`` and ``assemble`` take the name;
``preview`` has no choice.

Both are libx264 High at preset medium, in yuv420p. What differs is how much
of the picture and the sound each keeps (measured on made-up 1080p footage at
25 frames a second, written the way a camera writes it, 30 megabits of
picture a second; TOOLS.md, "Quality", has the tables):

- CRF, x264's dial for how much of the picture is kept: lower keeps more.
  Standard is 19 for an export and 16 for a reel. Master is 12 for both: the
  first step where every test clip scored 99.3 or more on VMAF and 50 dB or
  more on PSNR against the footage. A step further (10) adds 14 to 46 percent
  to the file for 0.3 VMAF at most.
- Preset medium for both. Preset slow gave the same score and the same size
  in twice the time.
- Sound: AAC at 192k (256k for a reel) for standard, 320k for master, which
  stayed 16 dB closer to the sound it read than 192k did.

A master export with photos or clips drawn over it would be two encodes. Its
first pass writes a working file that loses nothing instead (x264 at qp 0,
preset ultrafast, with ALAC sound), so the pass that draws is the one encode
the picture and the sound get. That file is big, up to four times the
footage, and is removed when the export is done.

Neither quality changes the picture's range or its color tags: ffmpeg carries
them from the source to the export.
"""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from lumr_studio.errors import StudioError

STANDARD = "standard"
MASTER = "master"
PROBE_SECONDS = 20.0


@dataclass(frozen=True)
class Quality:
    """How much of the picture and the sound one quality keeps."""

    name: str
    crf: int  # for a finished video
    reel_crf: int  # for a reel that has to be encoded
    sound: tuple[str, ...]  # ffmpeg's arguments for a finished video's sound
    reel_sound: tuple[str, ...]
    # Whether an export in two passes writes its first one to a working file that loses nothing.
    working_file: bool = False

    @property
    def is_master(self) -> bool:
        return self.name == MASTER


QUALITIES = {
    STANDARD: Quality(
        STANDARD, crf=19, reel_crf=16,
        sound=("-c:a", "aac", "-b:a", "192k"), reel_sound=("-c:a", "aac", "-b:a", "256k"),
    ),
    MASTER: Quality(
        MASTER, crf=12, reel_crf=12,
        sound=("-c:a", "aac", "-b:a", "320k"), reel_sound=("-c:a", "aac", "-b:a", "320k"),
        working_file=True,
    ),
}

# A working file: every frame and every sample as the first pass made them, written as fast as x264 goes.
# It is read once by the pass that follows and then removed, so it has no +faststart to pay for.
WORKING_VIDEO_ARGS = ("-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-pix_fmt", "yuv420p")
WORKING_SOUND_ARGS = ("-c:a", "alac")

# What each quality is, in words a creator can be told.
IN_WORDS = {
    STANDARD: "the everyday export: looks good and stays small",
    MASTER: "for uploading: as close to your footage as an export gets, and a much bigger file",
}


def check_quality(value: Any) -> Quality:
    """The quality named ``value``. Raises StudioError, in plain words, for anything else."""
    if isinstance(value, Quality):
        return value
    if isinstance(value, str) and value.strip().lower() in QUALITIES:
        return QUALITIES[value.strip().lower()]
    raise StudioError(
        f"quality {value!r} is not one Lumr Studio has. Pass \"standard\" ({IN_WORDS[STANDARD]}) "
        f"or \"master\" ({IN_WORDS[MASTER]})."
    )


def megabytes(path: Path) -> float | None:
    """The size of the file in megabytes, to one decimal; None when it can't be read."""
    try:
        return round(path.stat().st_size / 1e6, 1)
    except OSError:
        return None


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _turned(stream: dict[str, Any]) -> bool:
    """Whether the picture is shown on its side, as a phone held upright records it."""
    turns = [side.get("rotation") for side in stream.get("side_data_list") or [] if isinstance(side, dict)]
    turns.append((stream.get("tags") or {}).get("rotate"))
    for turn in turns:
        try:
            if round(float(turn)) % 180 == 90:
                return True
        except (TypeError, ValueError, OverflowError):
            continue
    return False


def file_facts(path: Path, *, run: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    """What a video file is, for telling the creator what they have or what they got.

    ``{resolution, fps, video_bitrate_mbps, size_mb}``: the picture's size as
    shown, its frames a second, the megabits of picture the file holds for
    each second, and the file's size in megabytes. A fact ffprobe can't read
    is left out, so a file that is no video gives ``{}`` or its size alone.
    """
    facts: dict[str, Any] = {}
    try:
        out = run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=PROBE_SECONDS,
        )
        streams = json.loads(out.stdout or "{}").get("streams") or []
    except (OSError, subprocess.SubprocessError, ValueError):
        streams = []
    picture = streams[0] if streams and isinstance(streams[0], dict) else {}
    width, height = _number(picture.get("width")), _number(picture.get("height"))
    if width and height:
        if _turned(picture):
            width, height = height, width
        facts["resolution"] = f"{int(width)}x{int(height)}"
    try:
        rate = _number(Fraction(str(picture.get("avg_frame_rate") or "")))
    except (ValueError, ZeroDivisionError):
        rate = None
    if rate:
        facts["fps"] = round(rate, 2)
    bitrate = _number(picture.get("bit_rate"))
    if bitrate:
        facts["video_bitrate_mbps"] = round(bitrate / 1e6, 2)
    size = megabytes(path)
    if size is not None and facts:
        facts["size_mb"] = size
    return facts
