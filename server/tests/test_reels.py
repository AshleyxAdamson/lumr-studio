"""assemble: several clips joined into one reel, and how the other tools see its clips."""

import dataclasses
import json
import os
import random
import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np
import pytest

from lumr_studio import reels, tools, word_times
from lumr_studio.errors import StudioError
from lumr_studio.jobs import JobHandle
from lumr_studio.overlays import probe_media
from lumr_studio.project import open_project
from lumr_studio.transcript import pack_transcript
from lumr_studio.tools import unmeasured_labels

from conftest import make_words


def make_clip(path: Path, *, size: str = "160x120", rate: int = 30, seconds: float = 3.0, sound: bool = True) -> Path:
    """A small test-pattern clip, nothing like the 320x240 25 fps mono master."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}:duration={seconds}"]
    if sound:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=330:sample_rate=44100:duration={seconds}", "-ac", "2", "-c:a", "aac"]
    else:
        cmd += ["-an"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-shortest", str(path)]
    subprocess.run(cmd, check=True)
    return path


def make_tone_clip(path: Path, *, color: str, frequency: int, seconds: float = 12.0, extra: tuple[str, ...] = ()) -> Path:
    """A solid-colour 320x240 25 fps clip with one steady tone, to tell where each piece of a reel really starts.

    Every clip made here matches the others in size, frame rate and sound, so
    only ``extra`` (output options such as a time base) tells them apart.
    """
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c={color}:size=320x240:rate=25:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency={frequency}:sample_rate=48000:duration={seconds}",
            "-ac", "2", "-c:a", "aac", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            *extra, str(path),
        ],
        check=True,
    )
    return path


def make_short_picture_clip(path: Path, *, picture: float, sound: float) -> Path:
    """Like ``make_tone_clip`` (red, 300 Hz) but with picture and sound of different lengths, such as 9.2 s and 10 s."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c=red:size=320x240:rate=25:duration={picture}",
            "-f", "lavfi", "-i", f"sine=frequency=300:sample_rate=48000:duration={sound}",
            "-ac", "2", "-c:a", "aac", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )
    return path


def picture_changes(reel: Path) -> list[float]:
    """Seconds at which the picture of a red-and-blue reel changes colour, read from the decoded frames."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(reel), "-an", "-vf", "scale=1:1,format=rgb24", "-fps_mode", "passthrough",
         "-f", "rawvideo", "-"],
        capture_output=True, check=True,
    ).stdout
    pixels = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
    blue = pixels[:, 2] > pixels[:, 0]
    return [round(float(i) / 25, 3) for i in range(1, len(blue)) if blue[i] != blue[i - 1]]


def tone_changes(reel: Path) -> list[float]:
    """Seconds at which a reel's sound changes between the low tone (300 Hz) and the high one (3 kHz)."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(reel), "-vn", "-ac", "1", "-ar", "48000", "-f", "s16le", "-"],
        capture_output=True, check=True,
    ).stdout
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    step = 480  # 10 ms
    changes, last = [], None
    for i in range(len(samples) // step):
        window = samples[i * step:(i + 1) * step]
        if np.abs(window).max() < 500:
            continue
        crossings = int(np.count_nonzero(np.signbit(window[1:]) != np.signbit(window[:-1])))
        high = crossings / 2 / (step / 48000) > 1200
        if last is not None and high != last:
            changes.append(round(i * step / 48000, 3))
        last = high
    return changes


def stream_seconds(reel: Path) -> dict[str, float]:
    """The length of the picture and of the sound in a file, each on its own."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration", "-of", "json", str(reel)],
        capture_output=True, text=True, check=True,
    ).stdout
    return {s["codec_type"]: float(s["duration"]) for s in json.loads(out)["streams"]}


def second_copy(video: Path, name: str = "talk two.mp4") -> Path:
    path = video.parent / name
    shutil.copy(video, path)
    return path


def run_assemble(jobs, name, pieces) -> dict:
    """Call assemble, wait for its job, and return the job's result."""
    started = tools.assemble(name, pieces, jobs=jobs)
    assert set(started) == {"job_id"}, started
    job = jobs.wait(started["job_id"], timeout=120)
    assert job["status"] == "done", job["error"]
    return job["result"]


def whole(*clips: Path) -> list[dict]:
    return [{"clip": str(c)} for c in clips]


# ── names and ids ─────────────────────────────────────────────────────────────


def test_the_slug_is_the_name_in_lower_case_with_runs_of_anything_else_as_one_dash():
    assert reels.slug_for("Bread  Week_5 -- Take") == "bread-week-5-take"
    assert reels.slug_for("  Hello  ") == "hello"


@pytest.mark.parametrize("name", ["", "   ", "a/b", "x" * 61, "---", "café", "what?"])
def test_a_bad_name_is_refused_in_plain_words(name):
    with pytest.raises(StudioError, match=r"1 to 60 plain letters \(A to Z\), digits, spaces"):
        reels.check_name(name)


def test_a_good_name_is_kept_trimmed():
    assert reels.check_name("  Bread_5-weeks ") == "Bread_5-weeks"
    assert reels.check_name("x" * 60) == "x" * 60


def test_the_id_is_stable_and_changes_with_the_name_a_piece_or_a_clip(video):
    other = second_copy(video)
    pieces = reels.check_pieces([{"clip": str(video)}, {"clip": str(other), "start": 1, "end": 5}])
    first = reels.sequence_id("demo", pieces)
    assert first == reels.sequence_id("demo", reels.check_pieces([{"clip": str(video)}, {"clip": str(other), "start": 1, "end": 5}]))
    assert len(first) == 8 and int(first, 16) >= 0

    assert reels.sequence_id("other name", pieces) != first
    moved = reels.check_pieces([{"clip": str(video)}, {"clip": str(other), "start": 1, "end": 6}])
    assert reels.sequence_id("demo", moved) != first
    reordered = reels.check_pieces([{"clip": str(other), "start": 1, "end": 5}, {"clip": str(video)}])
    assert reels.sequence_id("demo", reordered) != first

    stat = other.stat()
    os.utime(other, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    touched = reels.check_pieces([{"clip": str(video)}, {"clip": str(other), "start": 1, "end": 5}])
    assert reels.sequence_id("demo", touched) != first


def test_an_end_a_hair_past_the_clip_is_the_clip_end(video):
    piece = reels.check_pieces([{"clip": str(video), "start": 2, "end": 20.03}])[0]
    assert piece.end == piece.duration and not piece.whole
    assert reels.check_pieces([{"clip": str(video), "start": 0, "end": 20.03}])[0].whole


# ── the copy path ─────────────────────────────────────────────────────────────


def test_clips_that_match_are_copied_into_a_reel_with_a_sequence_file(video, jobs, tmp_path):
    other = second_copy(video)
    result = run_assemble(jobs, "Demo Reel", whole(video, other))

    reel = Path(result["reel"])
    assert result["status"] == "built" and result["how"] == "copy" and result["name"] == "Demo Reel"
    assert reel.is_file() and reel.parent == (tmp_path / "lumr-home" / "studio" / "reels" / "demo-reel").resolve()
    assert reel.name.startswith("demo-reel-") and reel.suffix == ".mp4" and reel.name.endswith(f"{reel.stem[-8:]}.mp4")
    assert probe_media(reel).duration == pytest.approx(40.0, abs=0.5)
    assert result["duration"] == pytest.approx(40.0, abs=0.5)
    assert [(p["clip"], p["start"]) for p in result["pieces"]] == [(str(video), 0.0), (str(other), 0.0)]
    assert [p["reel_start"] for p in result["pieces"]] == [0.0, result["pieces"][0]["reel_end"]]
    assert result["pieces"][1]["reel_end"] == pytest.approx(result["duration"])
    assert "video_path" in result["note"]

    sequence = Path(result["sequence_file"])
    assert sequence == reel.with_suffix(".reel.json")
    record = json.loads(sequence.read_text())
    assert record["version"] == 1 and record["name"] == "Demo Reel" and record["how"] == "copy"
    assert record["made_at"] and record["duration"] == result["duration"]
    assert set(record["pieces"][0]) == {"clip", "clip_duration", "start", "end", "reel_start", "reel_end"}
    assert record["pieces"][0]["clip_duration"] == pytest.approx(20.0, abs=0.1)
    assert reels.read_sequence(sequence) == record


def test_a_finished_reel_leaves_only_the_reel_and_its_sequence_file(video, jobs):
    result = run_assemble(jobs, "tidy", whole(video, second_copy(video)))
    names = sorted(p.name for p in Path(result["reel"]).parent.iterdir())
    assert names == sorted([Path(result["reel"]).name, Path(result["sequence_file"]).name])
    assert not any("proxy" in n or n.endswith(".source") for n in names)


def test_paths_with_spaces_and_apostrophes_are_joined(video, jobs, tmp_path):
    folder = tmp_path / "Ashley's takes" / "day one"
    folder.mkdir(parents=True)
    first, second = folder / "it's the first.mp4", folder / "second take.mp4"
    shutil.copy(video, first)
    shutil.copy(video, second)
    result = run_assemble(jobs, "apostrophe", whole(first, second))
    assert result["how"] == "copy"
    assert probe_media(Path(result["reel"])).duration == pytest.approx(40.0, abs=0.5)


def test_a_quote_in_a_path_is_closed_escaped_and_reopened_in_the_concat_list():
    assert reels.listing_line(Path("/a/it's here/x.mp4")) == "file '/a/it'\\''s here/x.mp4'\n"
    assert reels.listing_line(Path("/plain/x.mp4")) == "file '/plain/x.mp4'\n"


def test_the_same_clip_may_come_twice(video, jobs):
    result = run_assemble(jobs, "twice over", whole(video, video))
    assert result["how"] == "copy" and result["duration"] == pytest.approx(40.0, abs=0.5)


# ── the encode path ───────────────────────────────────────────────────────────


def test_clips_that_differ_are_encoded_once_to_the_first_clips_size(video, jobs, tmp_path):
    small = make_clip(tmp_path / "small.mp4")
    result = run_assemble(jobs, "mixed", [{"clip": str(video), "start": 2, "end": 7}, {"clip": str(small)}])

    reel = Path(result["reel"])
    assert result["how"] == "encode"
    assert probe_media(reel).duration == pytest.approx(8.0, abs=0.5)
    assert [(p["start"], p["end"]) for p in result["pieces"]] == [(2.0, 7.0), (0.0, 3.0)]
    assert [(p["reel_start"], p["reel_end"]) for p in result["pieces"]] == [(0.0, 5.0), (5.0, 8.0)]
    assert result["duration"] == 8.0

    media, streams = probe_media(reel), reels.probe_streams(reel)
    assert (media.width, media.height) == (320, 240)
    assert (streams.video_codec, streams.audio_codec, streams.pix_fmt) == ("h264", "aac", "yuv420p")
    assert (streams.sample_rate, streams.channels, streams.fps) == (48000, 2, "25/1")
    assert reels.read_sequence(Path(result["sequence_file"]))["how"] == "encode"


def test_one_trimmed_piece_is_enough_to_encode(video, jobs):
    result = run_assemble(jobs, "trimmed", [{"clip": str(video), "end": 10}, {"clip": str(video), "start": 10}])
    assert result["how"] == "encode"
    assert probe_media(Path(result["reel"])).duration == pytest.approx(20.0, abs=0.5)


def test_how_to_join_asks_for_a_whole_clip_and_the_same_details(video, tmp_path):
    small = make_clip(tmp_path / "small.mp4")
    assert reels.how_to_join(reels.check_pieces(whole(video, video))) == "copy"
    assert reels.how_to_join(reels.check_pieces(whole(video, small))) == "encode"
    assert reels.how_to_join(reels.check_pieces([{"clip": str(video), "start": 1}])) == "encode"


def red_and_blue(tmp_path: Path, **extra: tuple[str, ...]) -> tuple[Path, Path]:
    """A red clip with a 300 Hz tone and a blue clip with a 3 kHz tone, 12 s each."""
    red = make_tone_clip(tmp_path / "red.mp4", color="red", frequency=300, extra=extra.get("red", ()))
    blue = make_tone_clip(tmp_path / "blue.mp4", color="blue", frequency=3000, extra=extra.get("blue", ()))
    return red, blue


def alternating(red: Path, blue: Path, count: int, seed: int) -> list[dict]:
    """``count`` trimmed pieces, red then blue then red, with starts and ends that fall between frames."""
    rng = random.Random(seed)
    pieces = []
    for i in range(count):
        start = round(rng.uniform(0.3, 8.0), 3)
        pieces.append({
            "clip": str(red if i % 2 == 0 else blue), "start": start, "end": round(start + rng.uniform(0.5, 3.0), 3),
        })
    return pieces


def test_trimmed_pieces_off_the_frame_grid_start_where_the_sequence_file_says(jobs, tmp_path):
    red, blue = red_and_blue(tmp_path)
    result = run_assemble(jobs, "off the grid", alternating(red, blue, 12, seed=7))
    reel = Path(result["reel"])
    recorded = [p["reel_start"] for p in result["pieces"]][1:]

    assert result["how"] == "encode"
    assert picture_changes(reel) == pytest.approx(recorded, abs=0.001)
    assert tone_changes(reel) == pytest.approx(recorded, abs=0.03)
    assert probe_media(reel).duration == pytest.approx(result["duration"], abs=0.03)
    assert stream_seconds(reel)["video"] == pytest.approx(result["duration"], abs=0.03)
    assert stream_seconds(reel)["audio"] == pytest.approx(result["duration"], abs=0.05)


def test_forty_trimmed_pieces_still_pass_the_length_check(jobs, tmp_path):
    red, blue = red_and_blue(tmp_path)
    result = run_assemble(jobs, "forty pieces", alternating(red, blue, 40, seed=9))
    assert result["how"] == "encode" and len(result["pieces"]) == 40
    assert probe_media(Path(result["reel"])).duration == pytest.approx(result["duration"], abs=0.03)


def test_a_last_clip_whose_picture_ends_a_second_before_its_sound_is_joined(jobs, tmp_path):
    first = make_tone_clip(tmp_path / "first.mp4", color="blue", frequency=3000, seconds=10)
    short = make_short_picture_clip(tmp_path / "short.mp4", picture=9.2, sound=10.0)
    assert stream_seconds(short)["video"] < stream_seconds(short)["audio"] - 0.7

    sequences = {
        "short last": whole(first, short),
        "short alone": whole(short),
        "short trimmed to its end": [{"clip": str(first)}, {"clip": str(short), "start": 1.0, "end": 10.0}],
    }
    for name, pieces in sequences.items():
        result = run_assemble(jobs, name, pieces)
        lengths = stream_seconds(Path(result["reel"]))
        assert lengths["audio"] == pytest.approx(result["duration"], abs=0.1)  # the reel ends with the sound
        if result["how"] == "encode":  # the last picture is held to fill the piece
            assert lengths["video"] == pytest.approx(result["duration"], abs=0.1)
        else:  # a copy keeps the clip as it is, picture ending early
            assert lengths["video"] == pytest.approx(result["duration"] - 0.8, abs=0.1)


def test_a_last_clip_whose_sound_ends_early_is_still_copied(jobs, tmp_path):
    first = make_tone_clip(tmp_path / "first.mp4", color="blue", frequency=3000, seconds=10)
    quiet_end = make_short_picture_clip(tmp_path / "quiet.mp4", picture=10.0, sound=9.2)
    assert stream_seconds(quiet_end)["audio"] < stream_seconds(quiet_end)["video"] - 0.7

    result = run_assemble(jobs, "quiet end", whole(first, quiet_end))
    assert result["how"] == "copy"
    assert stream_seconds(Path(result["reel"]))["audio"] == pytest.approx(19.2, abs=0.1)


def test_a_copy_whose_sound_stops_early_for_any_other_reason_is_still_built_again(video, tmp_path):
    """Only the last clip's own shortfall is forgiven, and only as much as the clip itself falls short."""
    plan = reels.plan_reel("still held", whole(video, second_copy(video)))
    assert plan.how == "copy"
    forty = tmp_path / "forty.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-stream_loop", "1", "-i", str(video), "-c", "copy", str(forty)], check=True)
    seen: list[str] = []

    def sound_stops_at_half(cmd, out, **_kw):
        seen.append("concat" if "concat" in cmd else "filter")
        if "concat" in cmd:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(forty), "-af", "atrim=end=20", "-c:v", "copy", str(out)], check=True
            )
        else:
            shutil.copy(forty, out)

    reels.builder(plan, run=sound_stops_at_half)(JobHandle())
    assert seen == ["concat", "filter"]


def test_the_encode_may_hold_the_last_picture_as_long_as_the_piece_needs_and_no_longer(video):
    plan = reels.plan_reel("hold", [{"clip": str(video), "start": 0.0, "end": 5.0}])
    command = " ".join(reels.encode_command(plan.pieces, Path("out.mp4")))
    assert "tpad=stop_mode=clone:stop=-1" in command and "trim=end_frame=" in command


def test_the_most_trimmed_pieces_allowed_still_build_with_every_frame(jobs, tmp_path):
    """Each piece is its own input, and ffmpeg gives each input its own threads. With the default it runs out
    of threads near 195 pieces on a 10 core Mac, drops the picture of the last pieces and still exits 0."""
    red, blue = red_and_blue(tmp_path)
    pieces = alternating(red, blue, reels.MAX_PIECES, seed=3)
    for piece in pieces:
        piece["end"] = round(piece["start"] + 0.3, 3)
    result = run_assemble(jobs, "two hundred", pieces)
    frames = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames",
         "-of", "csv=p=0", result["reel"]],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert len(result["pieces"]) == reels.MAX_PIECES
    assert int(frames) == round(result["duration"] * 25)


def test_the_encode_command_keeps_each_filter_graph_to_one_thread(video):
    plan = reels.plan_reel("threads", [{"clip": str(video), "start": 0.0, "end": 5.0}])
    command = reels.encode_command(plan.pieces, Path("out.mp4"))
    at = command.index("-filter_complex_threads")
    assert command[at + 1] == "1" and at < command.index("-filter_complex")


def test_every_recorded_piece_is_a_whole_number_of_frames(video, tmp_path):
    small = make_clip(tmp_path / "small.mp4")  # 30 fps; the reel runs at the first clip's 25
    plan = reels.plan_reel("frames", [{"clip": str(video), "start": 0.013, "end": 1.371}, {"clip": str(small)}])
    assert plan.how == "encode"
    for length in plan.lengths:
        assert length * 25 == pytest.approx(round(length * 25), abs=1e-9)
    assert plan.lengths[0] == pytest.approx(1.36, abs=1e-9)  # 1.358 s is 33.95 frames, so 34
    assert plan.total == pytest.approx(sum(plan.lengths))


def test_a_piece_shorter_than_a_frame_still_lasts_one_frame(video):
    plan = reels.plan_reel("sliver", [{"clip": str(video), "start": 1.0, "end": 1.004}])
    assert plan.lengths == [pytest.approx(0.04)]


# ── clips that look alike but cannot be copied together ───────────────────────


def streams_with(piece: reels.Piece, **changes) -> reels.Piece:
    return dataclasses.replace(piece, streams=dataclasses.replace(piece.streams, **changes))


@pytest.mark.parametrize("change", [
    {"video_time_base": "1/600"},
    {"audio_time_base": "1/44100"},
    {"audio_profile": "HE-AACv2"},
    {"sample_aspect_ratio": "4:3"},
    {"field_order": "tt"},
    {"rotation": 90},
])
def test_clips_that_differ_in_a_detail_that_breaks_a_copy_are_encoded(video, change):
    first = reels.check_pieces(whole(video))[0]
    assert reels.how_to_join([first, first]) == "copy"
    assert reels.how_to_join([first, streams_with(first, **change)]) == "encode"


@pytest.mark.parametrize("order", ["12800 then 600", "600 then 12800"])
def test_clips_with_different_time_bases_are_encoded_and_land_where_recorded(jobs, tmp_path, order):
    scales = ("12800", "600") if order.startswith("12800") else ("600", "12800")
    red = make_tone_clip(tmp_path / "red.mp4", color="red", frequency=300, extra=("-video_track_timescale", scales[0]))
    blue = make_tone_clip(tmp_path / "blue.mp4", color="blue", frequency=3000, extra=("-video_track_timescale", scales[1]))
    assert {reels.probe_streams(red).video_time_base, reels.probe_streams(blue).video_time_base} == {"1/12800", "1/600"}

    result = run_assemble(jobs, "time bases " + order, whole(red, blue))
    reel = Path(result["reel"])
    assert result["how"] == "encode"
    assert picture_changes(reel) == pytest.approx([12.0], abs=0.001)
    assert tone_changes(reel) == pytest.approx([12.0], abs=0.03)
    assert stream_seconds(reel)["video"] == pytest.approx(24.0, abs=0.05)


def test_clips_with_different_pixel_shapes_are_encoded(jobs, tmp_path):
    red = make_tone_clip(tmp_path / "red.mp4", color="red", frequency=300, seconds=3)
    blue = make_tone_clip(tmp_path / "blue.mp4", color="blue", frequency=3000, seconds=3, extra=("-vf", "setsar=4/3"))
    assert run_assemble(jobs, "shapes", whole(red, blue))["how"] == "encode"


# ── clips in HDR or wide color come out in standard color ─────────────────────

HLG, PQ = "arib-std-b67", "smpte2084"


def make_hdr_clip(
    path: Path, *, color: str, transfer: str = HLG, primaries: str = "bt2020", frequency: int = 300, seconds: float = 3.0,
) -> Path:
    """A solid 320x240 25 fps HEVC clip, 10-bit, tagged BT.2020 with an HDR transfer, as a phone records HDR.

    ``color`` is the signal itself, such as ``0xC04040``: three quarters red
    and one quarter green and blue, in HLG or PQ, in BT.2020 colors.
    ``primaries="unknown"`` leaves that tag out, as a careless re-encode does.
    Skips the test where this ffmpeg can't write such a clip or can't convert one.
    """
    encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if " libx265 " in encoders:
        encoder = ["-c:v", "libx265", "-x265-params", "log-level=error"]
    elif " hevc_videotoolbox " in encoders:
        encoder = ["-c:v", "hevc_videotoolbox", "-profile:v", "main10"]
    else:
        pytest.skip("this ffmpeg has no HEVC encoder to write an HDR clip with")
    if not reels.can_convert_color():
        pytest.skip("this ffmpeg has no scale_vt filter")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", (
                f"color=c={color}:size=320x240:rate=25:duration={seconds},format=gbrp,"
                "scale=out_color_matrix=bt2020nc:out_range=tv,format=yuv420p10le,"
                f"setparams=color_primaries={primaries}:color_trc={transfer}:colorspace=bt2020nc:range=tv"
            ),
            "-f", "lavfi", "-i", f"sine=frequency={frequency}:sample_rate=48000:duration={seconds}",
            "-ac", "2", "-c:a", "aac", *encoder, "-tag:v", "hvc1", str(path),
        ],
        check=True,
    )
    return path


def color_at(video: Path, seconds: float) -> tuple[int, int, int]:
    """The red, green and blue of the frame at ``seconds``, averaged over the picture."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{seconds:.3f}", "-i", str(video), "-frames:v", "1",
         "-vf", "scale=1:1:flags=area,format=rgb24", "-f", "rawvideo", "-"],
        capture_output=True, check=True,
    ).stdout
    return tuple(raw[:3])


def color_tags(video: Path) -> tuple[str, str, str]:
    streams = reels.probe_streams(video)
    return streams.color_space, streams.color_transfer, streams.color_primaries


BT709 = ("bt709", "bt709", "bt709")
# What macOS (QuickTime, avconvert) shows for the HLG signal 0xC04040. Written out as bt709 unchanged it reads (188, 61, 61).
HLG_PATCH, HLG_PATCH_SHOWN = "0xC04040", (198, 0, 41)


@pytest.mark.parametrize("tags, kind", [
    ({"color_transfer": HLG}, "HDR"),
    ({"color_transfer": PQ}, "HDR"),
    ({"color_transfer": "bt709", "color_primaries": "bt2020"}, "wide color"),
    ({"color_space": "bt2020nc"}, "wide color"),
    ({"color_transfer": "bt709", "color_primaries": "bt709", "color_space": "bt709"}, None),
    ({}, None),
])
def test_a_clip_is_hdr_or_wide_color_by_its_color_tags(video, tags, kind):
    assert streams_with(reels.check_pieces(whole(video))[0], **tags).converted_from == kind


def test_a_sequence_with_an_hdr_clip_is_never_copied_and_color_tags_alone_do_not_stop_a_copy(video):
    plain = reels.check_pieces(whole(video))[0]
    hdr = streams_with(plain, color_transfer=HLG, color_primaries="bt2020", color_space="bt2020nc")
    assert reels.how_to_join([plain, plain]) == "copy"
    assert reels.how_to_join([hdr, hdr]) == "encode"
    assert reels.how_to_join([plain, hdr]) == "encode"
    assert reels.how_to_join([plain, streams_with(plain, color_space="bt709")]) == "copy"


def test_an_hdr_sequence_has_its_own_id(video):
    plain = reels.check_pieces(whole(video))
    hdr = [streams_with(plain[0], color_transfer=PQ)]
    assert reels.sequence_id("demo", hdr) != reels.sequence_id("demo", plain)
    assert reels.sequence_id("demo", hdr) == reels.sequence_id("demo", [streams_with(plain[0], color_transfer=HLG)])


def test_only_a_sequence_with_an_hdr_clip_is_sent_through_the_converter(video):
    plain = reels.check_pieces(whole(video, video))
    command = reels.encode_command(plain, Path("out.mp4"))
    assert not any(word in " ".join(command) for word in ("scale_vt", "hwupload", "setparams", "color_matrix", "-init_hw_device"))

    command = reels.encode_command([plain[0], streams_with(plain[1], color_transfer=HLG)], Path("out.mp4"))
    graph = command[command.index("-filter_complex") + 1].split(";")
    assert command.index("-init_hw_device") < command.index("-i")
    assert "scale_vt" not in graph[0] and "in_color_matrix=smpte170m:out_color_matrix=bt709" in graph[0]
    assert "scale_vt=color_matrix=bt709:color_primaries=bt709:color_transfer=bt709" in graph[2]
    assert all("setparams=color_primaries=bt709:color_trc=bt709" in graph[i] for i in (0, 2))


def test_an_hdr_clip_that_leaves_a_color_tag_out_is_read_as_bt2020(video):
    plain = reels.check_pieces(whole(video))[0]
    whole_tags = streams_with(plain, color_transfer=HLG, color_primaries="bt2020", color_space="bt2020nc")
    assert reels.to_standard_color(whole_tags) == reels.TO_STANDARD_COLOR
    bare = streams_with(plain, color_transfer=PQ, color_primaries="unknown")
    assert reels.to_standard_color(bare) == "setparams=color_primaries=bt2020:colorspace=bt2020nc," + reels.TO_STANDARD_COLOR
    wide = streams_with(plain, color_space="bt2020nc")  # not HDR: nothing says what its other tags should be
    assert reels.to_standard_color(wide) == reels.TO_STANDARD_COLOR


def test_an_hdr_clip_without_its_primaries_still_comes_out_in_standard_color(jobs, tmp_path):
    bare = make_hdr_clip(tmp_path / "bare.mov", color=HLG_PATCH, primaries="unknown", seconds=2)
    assert color_tags(bare) == ("bt2020nc", HLG, "")
    result = run_assemble(jobs, "bare", whole(bare))
    assert color_at(Path(result["reel"]), 1.0) == pytest.approx(HLG_PATCH_SHOWN, abs=8)
    assert color_tags(Path(result["reel"])) == BT709


@pytest.mark.parametrize("tagged, size, read_as", [
    ("bt709", (320, 240), "bt709"), ("bt470bg", (1920, 1080), "bt470"), ("", (1920, 1080), "bt709"),
    ("unknown", (1280, 720), "bt709"), ("", (720, 576), "smpte170m"),
])
def test_a_plain_clip_that_does_not_name_its_matrix_is_read_as_players_read_it(video, tagged, size, read_as):
    piece = streams_with(reels.check_pieces(whole(video))[0], color_space=tagged, width=size[0], height=size[1])
    assert reels.matrix_of(piece) == read_as


def test_whether_this_ffmpeg_can_convert_is_read_from_its_filter_list():
    def listing(*names):
        lines = "Filters:\n" + "".join(f" .. {name}            V->V       Does something.\n" for name in names)
        return lambda *_a, **_kw: subprocess.CompletedProcess([], 0, stdout=lines, stderr="")

    def no_ffmpeg(*_a, **_kw):
        raise FileNotFoundError("ffmpeg")

    assert reels.can_convert_color(run=listing("scale", "scale_vt", "tonemap")) is True
    assert reels.can_convert_color(run=listing("scale", "scale_vtb", "tonemap")) is False
    assert reels.can_convert_color(run=no_ffmpeg) is False


def test_an_hdr_clip_is_refused_in_plain_words_when_this_ffmpeg_cannot_convert_it(video):
    def hdr(path):
        return dataclasses.replace(reels.probe_streams(path), color_transfer=HLG)

    with pytest.raises(StudioError, match=r"talk\.mp4 was recorded in HDR.*no scale_vt filter.*brew install ffmpeg.*avconvert"):
        reels.plan_reel("no converter", whole(video), streams_of=hdr, can_convert=lambda: False)
    assert reels.plan_reel("no converter", whole(video), can_convert=lambda: False).how == "copy"  # a plain clip never asks


def test_a_failed_join_with_an_hdr_clip_says_what_to_try_and_leaves_nothing_behind(video):
    def hdr(path):
        return dataclasses.replace(reels.probe_streams(path), color_transfer=HLG)

    plan = reels.plan_reel("fails in hdr", whole(video), streams_of=hdr, can_convert=lambda: True)
    said = {}

    def broken(cmd, out, *, check, **_kw):
        said["check"] = check
        out.write_bytes(b"half")
        raise StudioError("ffmpeg could not join the clips")

    with pytest.raises(StudioError, match="could not join"):
        reels.builder(plan, run=broken)(JobHandle())
    assert "turning talk.mp4 from HDR into standard color is what failed" in said["check"] and "avconvert" in said["check"]
    assert list(plan.reel.parent.iterdir()) == []


def test_an_hdr_clip_joined_with_a_plain_one_comes_out_in_standard_color(jobs, tmp_path):
    blue = make_tone_clip(tmp_path / "blue.mp4", color="blue", frequency=3000, seconds=4)
    hdr = make_hdr_clip(tmp_path / "phone.mov", color=HLG_PATCH)
    before = (hdr.read_bytes(), hdr.stat().st_mtime_ns)
    assert color_tags(hdr) == ("bt2020nc", HLG, "bt2020")
    pieces = [{"clip": str(blue), "start": 0.5, "end": 3.3}, {"clip": str(hdr), "start": 0.52, "end": 2.6}, {"clip": str(blue)}]

    result = run_assemble(jobs, "phone and camera", pieces)
    reel = Path(result["reel"])

    assert result["how"] == "encode"
    assert color_tags(reel) == BT709 and reels.probe_streams(reel).pix_fmt == "yuv420p"
    assert color_at(reel, result["pieces"][1]["reel_start"] + 1) == pytest.approx(HLG_PATCH_SHOWN, abs=8)
    # The plain clip looks as it did: both read by the same reader, the clip by its size and the reel by its bt709 tag.
    assert color_at(reel, 1.0) == pytest.approx(color_at(blue, 1.0), abs=6)
    # The pieces still start where the sequence file says, to the frame.
    assert [(p["reel_start"], p["reel_end"]) for p in result["pieces"]] == [(0.0, 2.8), (2.8, 4.88), (4.88, 8.88)]
    assert picture_changes(reel) == pytest.approx([2.8, 4.88], abs=0.001)
    assert stream_seconds(reel)["video"] == pytest.approx(8.88, abs=0.03)

    assert [p.get("converted_from") for p in result["pieces"]] == [None, "HDR", None]
    assert "converted_from" not in result["pieces"][0]
    assert result["note"].startswith(reels.USE_THE_REEL)
    assert "phone.mov was recorded in HDR; the joined video shows it in standard color." in result["note"]
    record = reels.read_sequence(Path(result["sequence_file"]))
    assert [p.get("converted_from") for p in record["pieces"]] == [None, "HDR", None]
    assert set(reels.reel_for(reel)["pieces"][1]) == {"clip", "reel_start", "reel_end"}

    # The clip is as it was, and the reel's folder holds the reel and its sequence file and no working copy.
    assert (hdr.read_bytes(), hdr.stat().st_mtime_ns) == before
    assert sorted(p.name for p in reel.parent.iterdir()) == sorted([reel.name, Path(result["sequence_file"]).name])
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == ["blue.mp4", "phone.mov"]

    again = tools.assemble("phone and camera", pieces, jobs=jobs)
    assert again == {**result, "status": "exists"}


@pytest.mark.parametrize("transfer, shown", [(HLG, 98), (PQ, 185)])
def test_two_hdr_clips_alone_are_encoded_into_standard_color(jobs, tmp_path, transfer, shown):
    """Mid grey, 0x808080: left as it was it reads about 126 in both; macOS shows 98 for HLG and 185 for PQ."""
    one = make_hdr_clip(tmp_path / "one.mov", color="0x808080", transfer=transfer, seconds=2)
    two = make_hdr_clip(tmp_path / "two.mov", color="0x808080", transfer=transfer, frequency=3000, seconds=2)

    result = run_assemble(jobs, "phone only", whole(one, two))
    reel = Path(result["reel"])

    assert result["how"] == "encode"
    assert color_tags(reel) == BT709 and reels.probe_streams(reel).pix_fmt == "yuv420p"
    for piece in result["pieces"]:
        assert color_at(reel, piece["reel_start"] + 1) == pytest.approx((shown, shown, shown), abs=6)
        assert piece["converted_from"] == "HDR"
    assert "one.mov, two.mov were recorded in HDR; the joined video shows them in standard color." in result["note"]
    assert tone_changes(reel) == pytest.approx([2.0], abs=0.03)


def test_a_reel_with_no_hdr_clip_says_nothing_about_color(video, jobs):
    result = run_assemble(jobs, "plain", [{"clip": str(video), "end": 4}])
    assert result["note"] == reels.USE_THE_REEL
    assert set(result["pieces"][0]) == {"clip", "start", "end", "reel_start", "reel_end"}
    assert "converted_from" not in reels.read_sequence(Path(result["sequence_file"]))["pieces"][0]


def test_a_copied_reel_whose_pieces_come_out_wrong_is_built_again_as_an_encode(video, tmp_path):
    plan = reels.plan_reel("second try", whole(video, second_copy(video)))
    assert plan.how == "copy"
    forty = tmp_path / "forty.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-stream_loop", "1", "-i", str(video), "-c", "copy", str(forty)], check=True)
    commands: list[str] = []

    def copy_goes_wrong(cmd, out, **_kw):
        commands.append("concat" if "concat" in cmd else "filter")
        shutil.copy(video if "concat" in cmd else forty, out)  # ffmpeg was happy, but the copy is half as long as it should be

    answer = reels.builder(plan, run=copy_goes_wrong)(JobHandle())
    assert commands == ["concat", "filter"]
    assert answer["status"] == "built" and answer["how"] == "encode"
    assert reels.read_sequence(plan.sequence)["how"] == "encode"
    assert sorted(p.name for p in plan.reel.parent.iterdir()) == sorted([plan.reel.name, plan.sequence.name])


def test_the_picture_and_the_sound_are_each_held_to_the_length(video, tmp_path):
    plan = reels.plan_reel("lopsided", whole(video))  # 20 s

    def silent_half(cmd, out, **_kw):  # the container says 20 s, the sound stops at 10
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-af", "atrim=end=10", "-c:v", "copy", str(out)], check=True
        )

    with pytest.raises(StudioError, match="sound is 10.0 seconds long"):
        reels.builder(plan, run=silent_half)(JobHandle())
    assert list(plan.reel.parent.iterdir()) == []



# ── asking twice, and never overwriting ───────────────────────────────────────


def test_asking_again_for_the_same_sequence_answers_at_once_and_builds_nothing(video, jobs):
    pieces = whole(video, second_copy(video))
    built = run_assemble(jobs, "again", pieces)
    again = tools.assemble("again", pieces, jobs=jobs)

    assert again["status"] == "exists" and "job_id" not in again
    assert {k: again[k] for k in built if k != "status"} == {k: v for k, v in built.items() if k != "status"}
    assert len(list(Path(built["reel"]).parent.glob("*.mp4"))) == 1


def test_another_sequence_is_another_file_and_the_first_is_untouched(video, jobs):
    first = run_assemble(jobs, "keep", whole(video))
    before = Path(first["reel"]).read_bytes()
    second = run_assemble(jobs, "keep", whole(video, video))

    assert second["reel"] != first["reel"] and Path(second["reel"]).parent == Path(first["reel"]).parent
    assert Path(first["reel"]).read_bytes() == before
    assert Path(first["sequence_file"]).is_file() and Path(second["sequence_file"]).is_file()


def test_a_reel_file_with_no_sequence_file_is_never_replaced(video, jobs):
    plan = reels.plan_reel("stray", whole(video))
    plan.reel.parent.mkdir(parents=True)
    plan.reel.write_bytes(b"somebody's file")

    with pytest.raises(StudioError, match=plan.reel.name) as caught:
        tools.assemble("stray", whole(video), jobs=jobs)
    assert "never overwrites" in str(caught.value)
    assert plan.reel.read_bytes() == b"somebody's file" and not plan.sequence.exists()


def test_a_sequence_file_whose_reel_is_gone_is_built_again(video, jobs):
    built = run_assemble(jobs, "gone", whole(video))
    Path(built["reel"]).unlink()
    again = run_assemble(jobs, "gone", whole(video))
    assert again["status"] == "built" and Path(again["reel"]).is_file()


def test_a_sequence_file_that_cannot_be_read_is_named(video, jobs):
    built = run_assemble(jobs, "broken", whole(video))
    Path(built["sequence_file"]).write_text("{not json")
    with pytest.raises(StudioError, match=Path(built["sequence_file"]).name):
        tools.assemble("broken", whole(video), jobs=jobs)


def test_a_job_that_fails_removes_its_own_partial_and_its_list_file(video, tmp_path):
    plan = reels.plan_reel("fails", whole(video))

    def broken(cmd, out, **_kw):
        out.write_bytes(b"half")
        raise StudioError("ffmpeg could not join the clips")

    with pytest.raises(StudioError, match="could not join"):
        reels.builder(plan, run=broken)(JobHandle())
    assert list(plan.reel.parent.iterdir()) == []


def test_a_joined_file_of_the_wrong_length_is_thrown_away(video, tmp_path):
    small = make_clip(tmp_path / "small.mp4")
    plan = reels.plan_reel("short", whole(video))  # the clips add up to 20 s

    def too_short(cmd, out, **_kw):
        shutil.copy(small, out)  # 3 s

    with pytest.raises(StudioError, match="came out 3.0 seconds long but the clips add up to 20"):
        reels.builder(plan, run=too_short)(JobHandle())
    assert list(plan.reel.parent.iterdir()) == []


def test_the_job_reports_progress_from_ffmpeg(video, tmp_path):
    plan = reels.plan_reel("progress", whole(video))
    seen = []

    def run(cmd, out, *, total, on_progress, **_kw):
        assert total == plan.total
        on_progress(0.5)
        shutil.copy(video, out)

    handle = JobHandle()
    handle.set_progress = seen.append  # type: ignore[method-assign]
    reels.builder(plan, run=run)(handle)
    assert seen == [0.5]


# ── one partial file per build ────────────────────────────────────────────────


def test_each_build_writes_its_own_partial_file(video):
    plan = reels.plan_reel("own partial", whole(video))
    seen: list[Path] = []

    def keep_track(cmd, out, **_kw):
        seen.append(out)
        raise StudioError("stop here")

    for _ in range(2):
        with pytest.raises(StudioError, match="stop here"):
            reels.builder(plan, run=keep_track)(JobHandle())
    assert seen[0] != seen[1]
    for path in seen:
        assert path.parent == plan.reel.parent and path.name.startswith(plan.reel.stem) and path.name.endswith(".partial.mp4")
    assert list(plan.reel.parent.iterdir()) == []


@pytest.mark.parametrize("stale_name", ["{stem}.partial.mp4", "{stem}.k3j9x2ab.partial.mp4"])
def test_a_partial_left_by_a_dead_build_is_left_alone_and_never_becomes_the_reel(video, jobs, stale_name):
    plan = reels.plan_reel("leftover", whole(video))
    plan.reel.parent.mkdir(parents=True)
    stale = plan.reel.parent / stale_name.format(stem=plan.reel.stem)
    stale.write_bytes(b"a half-written reel from a build that died")

    result = run_assemble(jobs, "leftover", whole(video))
    assert Path(result["reel"]) == plan.reel
    assert probe_media(plan.reel).duration == pytest.approx(20.0, abs=0.5)
    assert stale.read_bytes() == b"a half-written reel from a build that died"


def test_a_build_that_lost_the_race_leaves_the_winners_reel_alone_and_answers_from_it(video):
    plan = reels.plan_reel("race", whole(video))

    def other_server_finishes_first(cmd, out, **_kw):
        shutil.copy(video, out)
        reels.builder(plan, run=lambda c, o, **_k: shutil.copy(video, o))(JobHandle())

    answer = reels.builder(plan, run=other_server_finishes_first)(JobHandle())

    assert answer["status"] == "exists" and answer["reel"] == str(plan.reel)
    assert plan.reel.read_bytes() == video.read_bytes()
    assert sorted(p.name for p in plan.reel.parent.iterdir()) == sorted([plan.reel.name, plan.sequence.name])


def test_a_reel_that_appears_without_its_sequence_file_is_not_replaced(video):
    plan = reels.plan_reel("half", whole(video))

    def somebody_else_wrote_the_reel(cmd, out, **_kw):
        shutil.copy(video, out)
        plan.reel.write_bytes(b"somebody's file")

    with pytest.raises(StudioError, match="never overwrites a file"):
        reels.builder(plan, run=somebody_else_wrote_the_reel)(JobHandle())
    assert plan.reel.read_bytes() == b"somebody's file"
    assert [p.name for p in plan.reel.parent.iterdir()] == [plan.reel.name]


def test_on_a_disk_without_hard_links_a_reel_that_is_already_there_is_not_replaced(video, monkeypatch):
    """Covers a file already at the name, and a file that lands there after the look and before the move.

    What is not covered, because it can't be: a file that lands between the exclusive create of the name and the
    rename. Without hard links the platform has no move-if-absent, so that instant stays open.
    """
    def no_links(*_a, **_k):
        raise PermissionError("operation not permitted")

    monkeypatch.setattr(os, "link", no_links)
    plan = reels.plan_reel("no links", whole(video))
    answer = reels.builder(plan, run=lambda c, o, **_k: shutil.copy(video, o))(JobHandle())
    assert answer["status"] == "built" and plan.reel.is_file()

    again = reels.plan_reel("no links", whole(video))
    plan.sequence.unlink()
    plan.reel.write_bytes(b"somebody's file")
    with pytest.raises(StudioError, match="never overwrites a file"):
        reels.builder(again, run=lambda c, o, **_k: shutil.copy(video, o))(JobHandle())
    assert plan.reel.read_bytes() == b"somebody's file"
    assert [p.name for p in plan.reel.parent.iterdir()] == [plan.reel.name]


def test_on_a_disk_without_hard_links_a_file_that_lands_after_the_look_is_not_replaced(tmp_path, monkeypatch):
    def no_links(*_a, **_k):
        raise PermissionError("operation not permitted")

    monkeypatch.setattr(os, "link", no_links)
    partial, reel = tmp_path / "x.partial.mp4", tmp_path / "x.mp4"
    partial.write_bytes(b"ours")
    real_exists = Path.exists
    # The look says "free"; a rival file appears right after it, before the move.
    monkeypatch.setattr(Path, "exists", lambda self: False if self == reel else real_exists(self))
    reel.write_bytes(b"theirs")

    with pytest.raises(FileExistsError):
        reels.put_in_place(partial, reel)
    assert reel.read_bytes() == b"theirs" and partial.read_bytes() == b"ours"


def test_on_a_disk_without_hard_links_a_failed_move_removes_only_the_empty_file_it_made(tmp_path, monkeypatch):
    def no_links(*_a, **_k):
        raise PermissionError("operation not permitted")

    monkeypatch.setattr(os, "link", no_links)
    partial, reel = tmp_path / "x.partial.mp4", tmp_path / "x.mp4"  # no partial file, so the rename fails
    with pytest.raises(FileNotFoundError):
        reels.put_in_place(partial, reel)
    assert list(tmp_path.iterdir()) == []


def test_a_second_call_while_the_first_job_runs_gets_the_same_job(video, jobs, monkeypatch):
    release = threading.Event()
    real = reels.builder

    def slow_builder(plan, **kw):
        work = real(plan, run=lambda c, o, **_k: (release.wait(30), shutil.copy(video, o))[1], **kw)
        return work

    monkeypatch.setattr(reels, "builder", slow_builder)
    first = tools.assemble("busy", whole(video), jobs=jobs)
    second = tools.assemble("busy", whole(video), jobs=jobs)
    assert set(first) == {"job_id"} and second == first
    release.set()
    assert jobs.wait(first["job_id"], timeout=60)["status"] == "done"
    assert tools.assemble("busy", whole(video), jobs=jobs)["status"] == "exists"



# ── what is refused ───────────────────────────────────────────────────────────


def test_a_missing_clip_is_named(video, jobs, tmp_path):
    with pytest.raises(StudioError, match=r"Piece 2: no file at .*nope\.mp4"):
        tools.assemble("x", whole(video, tmp_path / "nope.mp4"), jobs=jobs)


def test_a_relative_path_and_a_folder_are_refused(video, jobs):
    with pytest.raises(StudioError, match="relative"):
        tools.assemble("x", [{"clip": "talk.mp4"}], jobs=jobs)
    with pytest.raises(StudioError, match="folder, not a clip"):
        tools.assemble("x", whole(video.parent), jobs=jobs)


def test_a_file_that_is_not_a_video_is_named(video, jobs, tmp_path):
    notes = tmp_path / "notes.mp4"
    notes.write_text("not a video")
    with pytest.raises(StudioError, match=r"notes\.mp4 is not a video"):
        tools.assemble("x", whole(notes), jobs=jobs)


def test_a_clip_without_sound_is_named(video, jobs, tmp_path):
    silent = make_clip(tmp_path / "silent.mp4", sound=False)
    with pytest.raises(StudioError, match=r"silent\.mp4 has no sound"):
        tools.assemble("x", whole(video, silent), jobs=jobs)


@pytest.mark.parametrize("start, end, said", [
    (25, None, "start 25 is outside talk.mp4"),
    (-1, 5, "start -1 is outside talk.mp4"),
    (0, 25, "end 25 is past the end of talk.mp4"),
    (5, 5, "end 5 must come after start 5"),
    (8, 3, "end 3 must come after start 8"),
    (float("nan"), 5, "start must be a number"),
    ("soon", 5, "start must be a number"),
])
def test_a_start_or_end_outside_the_clip_is_refused(video, jobs, start, end, said):
    with pytest.raises(StudioError, match=said):
        tools.assemble("x", [{"clip": str(video), "start": start, "end": end}], jobs=jobs)


def test_a_bad_name_stops_assemble_before_anything_is_written(video, jobs, tmp_path):
    with pytest.raises(StudioError, match="won't do"):
        tools.assemble("a/b", whole(video), jobs=jobs)
    assert not (tmp_path / "lumr-home" / "studio" / "reels").exists()


def test_no_pieces_and_too_many_pieces_are_refused(video, jobs):
    with pytest.raises(StudioError, match="at least one piece"):
        tools.assemble("x", [], jobs=jobs)
    with pytest.raises(StudioError, match="at most 200 pieces and got 201"):
        tools.assemble("x", whole(*[video] * 201), jobs=jobs)
    with pytest.raises(StudioError, match="Piece 1 is not an object"):
        tools.assemble("x", ["talk.mp4"], jobs=jobs)  # type: ignore[list-item]


# ── seeing the clips from inside the reel ─────────────────────────────────────


def build_reel_with_words(video, jobs) -> tuple[Path, dict]:
    """A 40 s reel of two copies of the test video, with a words file beside it: the talk twice."""
    other = second_copy(video)
    result = run_assemble(jobs, "seen", whole(video, other))
    reel = Path(result["reel"])
    second = [{**w, "start": w["start"] + 20.0, "end": w["end"] + 20.0} for w in make_words()]
    reel.with_suffix(".words.json").write_text(json.dumps(make_words() + second))
    return reel, result


def test_reel_for_finds_the_sequence_file_and_gives_none_for_a_plain_video(video, jobs):
    assert reels.reel_for(video) is None
    reel, result = build_reel_with_words(video, jobs)
    found = reels.reel_for(reel)
    assert found["name"] == "seen"
    assert [set(p) for p in found["pieces"]] == [{"clip", "reel_start", "reel_end"}] * 2
    assert [Path(p["clip"]).name for p in found["pieces"]] == ["talk.mp4", "talk two.mp4"]
    assert found["pieces"][1]["reel_start"] == result["pieces"][1]["reel_start"]


def test_a_sequence_file_that_cannot_be_read_is_not_a_reel(video):
    reels.sequence_path_for(video).write_text("{not json")
    assert reels.reel_for(video) is None
    reels.sequence_path_for(video).write_text(json.dumps({"version": 1, "name": "x", "pieces": []}))
    assert reels.reel_for(video) is None


def test_get_edit_shows_the_clips_of_a_reel_and_nothing_for_a_plain_video(video, jobs):
    assert "reel" not in tools.get_edit(str(video), labels=unmeasured_labels)
    reel, result = build_reel_with_words(video, jobs)
    shown = tools.get_edit(str(reel), labels=unmeasured_labels)["reel"]
    assert shown == {
        "name": "seen",
        "pieces": [{k: p[k] for k in ("clip", "reel_start", "reel_end")} for p in result["pieces"]],
    }


def test_read_transcript_marks_where_each_clip_starts(video, jobs):
    reel, result = build_reel_with_words(video, jobs)
    start = result["pieces"][1]["reel_start"]
    text = tools.read_transcript(str(reel), labels=unmeasured_labels)["text"]
    lines = text.splitlines()

    assert lines[0] == "== clip 1 of 2: talk.mp4 starts 0.00 =="
    marker = f"== clip 2 of 2: talk two.mp4 starts {start:.2f} =="
    assert lines.count(marker) == 1
    after = lines[lines.index(marker) + 1:]
    assert after[0].startswith("  (pause") and "Hello" in after[1]
    assert all(not ln.startswith("==") for ln in lines if ln != marker and ln != lines[0])


def test_a_window_inside_one_clip_has_no_marker_for_the_other(video, jobs):
    reel, result = build_reel_with_words(video, jobs)
    start = result["pieces"][1]["reel_start"]
    text = tools.read_transcript(str(reel), start=2.0, end=start - 1, labels=unmeasured_labels)["text"]
    assert "==" not in text


def test_a_plain_video_reads_exactly_as_it_did(video):
    project = open_project(str(video))
    words = word_times.load_words(project)
    plain = pack_transcript(words, labels=unmeasured_labels(project, words))
    assert tools.read_transcript(str(video), labels=unmeasured_labels) == plain
    assert "==" not in plain["text"]
    # Pinned to what the packer printed before reels existed, so a change to plain output shows here.
    assert plain["text"].splitlines()[:6] == [
        "[0.50-2.40] Hello everyone and welcome.",
        "  (pause 1.0)",
        "[3.40-6.00] um today we talk about editing",
        "[6.00-6.60] (sound 0.6s)",
        "  (pause 1.4)",
        "[8.00-10.70] today we talk about editing video.",
    ]


# ── markers in the packed text ────────────────────────────────────────────────

MARKER = (20.0, "== clip 2 of 2: b.mp4 starts 20.00 ==")


def spoken(*rows: tuple[str, float, float]) -> list[dict]:
    return [{"word": w, "start": s, "end": e} for w, s, e in rows]


def test_no_line_runs_across_the_start_of_a_clip():
    words = spoken(("hello", 19.5, 19.9), ("again", 20.1, 20.5), ("there.", 20.6, 21.0))
    assert pack_transcript(words)["text"].splitlines()[0] == "[19.50-21.00] hello again there."
    lines = pack_transcript(words, markers=[MARKER])["text"].splitlines()
    assert lines == ["[19.50-19.90] hello", MARKER[1], "[20.10-21.00] again there.", "END"]


def test_a_marker_goes_before_the_pause_that_opens_its_clip():
    words = spoken(("one.", 18.0, 18.4), ("two.", 21.0, 21.4))
    lines = pack_transcript(words, markers=[MARKER])["text"].splitlines()
    assert lines == ["[18.00-18.40] one.", MARKER[1], "  (pause 2.6)", "[21.00-21.40] two.", "END"]


def test_a_marker_outside_the_window_or_past_the_last_word_is_not_printed():
    words = spoken(("one.", 18.0, 18.4), ("two.", 21.0, 21.4))
    assert "==" not in pack_transcript(words, start=20.5, markers=[MARKER])["text"]
    assert "==" not in pack_transcript(words, end=19.0, markers=[MARKER])["text"]
    assert "==" not in pack_transcript(words[:1], markers=[MARKER])["text"]


def test_reading_on_with_next_prints_every_marker_and_word_once():
    rows = [(f"w{i}.", i * 1.0, i * 1.0 + 0.4) for i in range(40)]
    words = spoken(*rows)
    markers = [(0.0, "== clip 1 of 3: a.mp4 starts 0.00 =="), (13.0, "== clip 2 of 3: b.mp4 starts 13.00 =="),
               (27.0, "== clip 3 of 3: c.mp4 starts 27.00 ==")]
    lines: list[str] = []
    start = None
    for _ in range(100):
        packed = pack_transcript(words, start=start, budget_chars=90, markers=markers)
        lines += [ln for ln in packed["text"].splitlines() if not ln.startswith(("NEXT", "END"))]
        if packed["next_start"] is None:
            break
        assert start is None or packed["next_start"] > start
        start = packed["next_start"]
    else:
        pytest.fail("reading on never reached END")
    assert [ln for ln in lines if ln.startswith("==")] == [m[1] for m in markers]
    said = [w for ln in lines if ln.startswith("[") for w in ln.split("] ", 1)[1].split()]
    assert said == [f"w{i}." for i in range(40)]


def test_a_window_that_starts_on_a_clip_start_prints_its_marker_and_one_that_ends_there_does_not():
    words = spoken(("one.", 18.0, 18.4), ("two.", 20.0, 20.4), ("three.", 21.0, 21.4))
    starting = pack_transcript(words, start=20.0, markers=[MARKER])["text"].splitlines()
    assert starting[0] == MARKER[1] and starting[1].startswith("[20.00-")
    ending = pack_transcript(words, end=20.0, markers=[MARKER])
    assert "==" not in ending["text"]
