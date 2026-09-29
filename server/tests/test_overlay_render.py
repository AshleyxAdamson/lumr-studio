"""Drawing overlays: sizes and places, the ffmpeg graph and command, and real renders."""

import subprocess
import sys
from pathlib import Path

import pytest
from lumr_studio.engine.render import probe_duration

from lumr_studio import overlay_tools, tools
from lumr_studio.errors import StudioError
from lumr_studio.overlay_render import (
    FADE_SECONDS,
    Item,
    build_graph,
    fit_inside,
    items_for_render,
    items_for_window,
    overlay_command,
    overlay_position,
    overlay_size,
    picture_size,
    run_ffmpeg,
)
from lumr_studio.overlays import HIDDEN, SHOWN, Placement, overlays_path
from lumr_studio.project import open_project
from lumr_studio.silences import no_silences
from test_overlays import make_clip, make_image

FRAME = (1280, 720)
FPS = 25.0
DURATION_TOLERANCE = 0.2
CUTS = [{"start": 7.9, "end": 10.8, "reason": "retake of the intro line"}]


def photo(**fields):
    base = {
        "id": "ov1", "kind": "image", "file": "/media/photo.jpg", "start": 2.0, "end": 6.0,
        "place": "full", "fit": "fill", "size": None, "motion": "still", "layer": 1, "fade": True,
        "media": {"width": 400, "height": 300, "duration": None, "has_sound": False},
    }
    return {**base, **fields}


def clip(**fields):
    base = {
        "id": "ov2", "kind": "video", "file": "/media/clip.mp4", "start": 3.0, "end": 7.0,
        "place": "full", "fit": "fill", "size": None, "motion": None, "layer": 1, "fade": True,
        "clip_in": 1.5, "clip_out": 5.5, "volume": 0.0,
        "media": {"width": 160, "height": 120, "duration": 6.0, "has_sound": True},
    }
    return {**base, **fields}


# ── Sizes and places ──────────────────────────────────────────────────────────


def test_fit_inside_keeps_the_shape_and_even_sizes():
    assert fit_inside(400, 300, 512, 288) == (384, 288)
    assert fit_inside(300, 600, 512, 288) == (144, 288)
    w, h = fit_inside(333, 333, 101, 101)
    assert w % 2 == 0 and h % 2 == 0 and w <= 101


def test_a_box_is_a_share_of_the_frame_at_any_size():
    box = photo(place="top_right", size=0.4, fit=None)
    # 40% of 720 is 288 tall; the picture sits inside a 4 pixel white edge.
    assert picture_size(box, FRAME) == (374, 280) and overlay_size(box, FRAME) == (382, 288)
    assert overlay_size(box, (640, 360)) == (190, 144)
    assert overlay_size(photo(), FRAME) == picture_size(photo(), FRAME) == FRAME


@pytest.mark.parametrize(
    "place, expected",
    [
        ("full", (0, 0)),
        ("top_left", (29, 29)),
        ("top_right", (1280 - 29 - 382, 29)),
        ("bottom_left", (29, 720 - 29 - 288)),
        ("bottom_right", (1280 - 29 - 382, 720 - 29 - 288)),
        ("center", ((1280 - 382) // 2, (720 - 288) // 2)),
    ],
)
def test_named_places(place, expected):
    o = photo(place=place, size=None if place == "full" else 0.4, fit="fill" if place == "full" else None)
    assert overlay_position(o, FRAME) == expected


# ── Timing items ──────────────────────────────────────────────────────────────


def placement(overlay, at, seconds, status=SHOWN):
    return Placement(overlay, at, seconds, status, ())


def test_render_items_skip_hidden_overlays():
    items = items_for_render([placement(photo(), 2.0, 4.0), placement(clip(), 0.0, 0.0, HIDDEN)])
    assert [(i.overlay["id"], i.at, i.seconds, i.skip) for i in items] == [("ov1", 2.0, 4.0, 0.0)]


def test_window_items_are_timed_from_the_window_and_know_what_already_played():
    p = placement(photo(), 10.0, 6.0)
    (item,) = items_for_window([p], 12.0, 20.0)
    assert (item.at, item.seconds, item.skip, item.total) == (0.0, 4.0, 2.0, 6.0)
    (late,) = items_for_window([p], 4.0, 14.0)
    assert (late.at, late.seconds, late.skip) == (6.0, 4.0, 0.0)
    assert items_for_window([p], 17.0, 25.0) == []


# ── The graph and the command ─────────────────────────────────────────────────


def test_graph_for_one_still_photo():
    g = build_graph([Item(photo(), 2.0, 4.0, 0.0, 4.0)], FRAME, FPS)
    assert g.inputs == ["-i", "/media/photo.jpg"]
    assert "[1:v]scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720" in g.filter
    assert "loop=loop=99:size=1:start=0,setpts=N/25/TB" in g.filter
    assert f"fade=t=in:st=0:d={FADE_SECONDS:.3f}:alpha=1" in g.filter
    assert f"fade=t=out:st={4.0 - FADE_SECONDS:.3f}" in g.filter
    assert "setpts=PTS-STARTPTS+2.000/TB[ov1]" in g.filter
    assert "[0:v][ov1]overlay=x=0:y=0:eof_action=pass:enable='between(t,2.000,6.000)'" in g.filter
    assert g.filter.endswith("format=yuv420p[vout]") and g.audio is None


def test_graph_for_a_drifting_photo_shown_whole():
    g = build_graph([Item(photo(motion="zoom_in", fit="fit"), 0.0, 5.0, 0.0, 5.0)], FRAME, FPS)
    assert "split[bgsrc1][fgsrc1]" in g.filter and "boxblur" in g.filter
    assert "scale=2560:1440,zoompan=z='1+0.08*(on+0)/124'" in g.filter
    assert ":d=125:s=1280x720:fps=25" in g.filter
    out = build_graph([Item(photo(motion="zoom_out"), 0.0, 5.0, 2.0, 7.0)], FRAME, FPS).filter
    assert "z='1+0.08*(1-(on+50)/174)'" in out  # a preview mid-overlay picks up the drift where it is


def test_graph_for_one_muted_clip():
    g = build_graph([Item(clip(), 3.0, 4.0, 0.0, 4.0)], FRAME, FPS)
    assert g.inputs == ["-ss", "1.500", "-t", "4.000", "-i", "/media/clip.mp4"]
    assert "[1:v]fps=25,setpts=PTS-STARTPTS[raw1]" in g.filter
    assert "amix" not in g.filter and g.audio is None


def test_graph_for_a_clip_with_its_sound_under_the_voice():
    g = build_graph([Item(clip(volume=0.2), 3.0, 4.0, 0.0, 4.0)], FRAME, FPS)
    assert "[1:a]asetpts=PTS-STARTPTS,afade=t=in" in g.filter
    assert "volume=0.2,adelay=3000:all=1[oa1]" in g.filter
    assert "[0:a][oa1]amix=inputs=2:duration=first:normalize=0" in g.filter
    assert g.audio == "aout"


def test_graph_for_picture_in_picture():
    box = clip(place="bottom_left", size=0.3, fit=None)
    g = build_graph([Item(box, 1.0, 3.0, 0.0, 3.0)], FRAME, FPS)
    x, y = overlay_position(box, FRAME)
    assert picture_size(box, FRAME) == (278, 208) and overlay_size(box, FRAME) == (286, 216)
    assert "[raw1]scale=278:208,setsar=1[shaped1]" in g.filter
    assert "[shaped1]pad=286:216:4:4:color=white,format=yuva420p" in g.filter
    assert f"overlay=x={x}:y={y}:" in g.filter


def test_graph_for_two_overlapping_overlays_draws_the_top_layer_last():
    under = Item(photo(), 2.0, 4.0, 0.0, 4.0)
    over = Item(clip(place="top_right", size=0.4, fit=None, layer=2), 3.0, 4.0, 0.0, 4.0)
    g = build_graph([under, over], FRAME, FPS)
    assert g.inputs[1] == "/media/photo.jpg" and g.inputs[-1] == "/media/clip.mp4"
    assert "[0:v][ov1]overlay" in g.filter and "[layered0][ov2]overlay" in g.filter
    assert g.filter.index("[0:v][ov1]") < g.filter.index("[layered0][ov2]")


def test_fades_respect_skip_short_overlays_and_off():
    mid = build_graph([Item(photo(), 0.0, 2.0, 1.0, 6.0)], FRAME, FPS).filter
    assert "t=in" not in mid and "t=out" not in mid  # neither edge of the overlay is in this file
    short = build_graph([Item(photo(), 0.0, 1.0, 0.0, 1.0)], FRAME, FPS).filter
    assert "d=0.250" in short
    off = build_graph([Item(photo(fade=False), 0.0, 3.0, 0.0, 3.0)], FRAME, FPS).filter
    assert "fade" not in off


def test_command_copies_the_voice_unless_clip_sound_is_mixed(tmp_path):
    quiet = build_graph([Item(clip(), 3.0, 4.0)], FRAME, FPS)
    cmd = overlay_command(Path("/w/base.mp4"), tmp_path / "out.mp4", quiet, crf=19)
    assert cmd[cmd.index("-map", cmd.index("-filter_complex")) + 1] == "[vout]"
    assert ["-map", "0:a?", "-c:a", "copy"] == cmd[cmd.index("0:a?") - 1: cmd.index("0:a?") + 3]
    assert "-progress" in cmd and "pipe:1" in cmd and cmd[-1] == str(tmp_path / "out.mp4")
    mixed = build_graph([Item(clip(volume=0.2), 3.0, 4.0)], FRAME, FPS)
    cmd = overlay_command(Path("/w/base.mp4"), tmp_path / "out.mp4", mixed, crf=19)
    assert ["-map", "[aout]", "-c:a", "aac"] == cmd[cmd.index("[aout]") - 1: cmd.index("[aout]") + 3]


# ── Running ffmpeg ────────────────────────────────────────────────────────────

FAKE_FFMPEG = """
import sys
code, out = int(sys.argv[1]), sys.argv[2]
print("out_time_us=1000000", flush=True)
for _ in range(20000):
    sys.stderr.write("Late SEI is not implemented\\n")
sys.stderr.write("THE-LAST-LINE\\n")
if code == 0:
    open(out, "wb").write(b"x" * 16)
print("progress=end", flush=True)
sys.exit(code)
"""


def fake_runner(tmp_path, code):
    script = tmp_path / "fake_ffmpeg.py"
    script.write_text(FAKE_FFMPEG)

    def runner(cmd, **kwargs):
        return subprocess.Popen([sys.executable, str(script), str(code), cmd[-1]], **kwargs)
    return runner


def test_a_flood_of_warnings_does_not_stall_the_pass(tmp_path):
    out = tmp_path / "out.mp4"
    seen = []
    run_ffmpeg(["ffmpeg", str(out)], out, total=2.0, on_progress=seen.append, runner=fake_runner(tmp_path, 0))
    assert out.exists() and seen[0] == 0.5 and seen[-1] == 1.0


def test_a_failed_pass_reports_the_end_of_the_error_and_leaves_nothing(tmp_path):
    out = tmp_path / "out.mp4"
    out.write_bytes(b"half")
    with pytest.raises(StudioError, match="THE-LAST-LINE"):
        run_ffmpeg(["ffmpeg", str(out)], out, total=2.0, runner=fake_runner(tmp_path, 1))
    assert not out.exists()


def test_a_failed_pass_says_which_pass_it_was(tmp_path):
    out = tmp_path / "out.mp4"
    with pytest.raises(StudioError) as drawing:
        run_ffmpeg(["ffmpeg", str(out)], out, total=2.0, runner=fake_runner(tmp_path, 1))
    assert "could not draw the overlays into out.mp4 (exit 1)" in str(drawing.value)
    assert str(drawing.value).endswith("Check that every overlay file opens in a video player.")
    with pytest.raises(StudioError) as cutting:
        run_ffmpeg(
            ["ffmpeg", str(out)], out, total=2.0, runner=fake_runner(tmp_path, 1),
            doing="encode", check="Check that the source video plays.",
        )
    assert "could not encode out.mp4 (exit 1)" in str(cutting.value) and "overlay" not in str(cutting.value)
    assert "THE-LAST-LINE" in str(cutting.value)


# ── Real renders on the synthetic video ───────────────────────────────────────


def _frame(path: Path, t: float) -> bytes:
    return subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
         "-vf", "scale=32:24,format=gray", "-f", "rawvideo", "-"],
        capture_output=True, check=True,
    ).stdout


def _diff(a: bytes, b: bytes) -> float:
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def _mean(frame: bytes) -> float:
    return sum(frame) / len(frame)


def _finish(jobs, job):
    status = jobs.wait(job["job_id"], timeout=120)
    assert status["status"] == "done", status["error"]
    return status["result"]


@pytest.fixture
def overlay_media(tmp_path):
    folder = tmp_path / "creator-media"
    folder.mkdir()
    return {
        "photo": make_image(folder / "photo.jpg", (400, 300), "white"),
        "clip": make_clip(folder / "broll.mp4", 3.0),
    }


def test_render_draws_overlays_in_a_second_pass(video, jobs, overlay_media):
    edit = tools.set_edit(str(video), CUTS, silences=no_silences)
    plain = Path(_finish(jobs, tools.render(str(video), jobs=jobs))["output_path"])

    saved = overlay_tools.set_overlays(str(video), [
        {"file": str(overlay_media["photo"]), "start": 1.0, "end": 5.0, "motion": "still"},
        {"file": str(overlay_media["clip"]), "start": 12.0, "end": 14.5, "place": "top_right"},
    ], picture=False)
    assert not saved["rejected"]
    result = _finish(jobs, tools.render(str(video), jobs=jobs))
    out = Path(result["output_path"])

    assert probe_duration(out) == pytest.approx(edit["new_duration"], abs=DURATION_TOLERANCE)
    assert result["overlays"]["drawn"] == ["ov1", "ov2"] and not result["overlays"]["not_shown"]
    assert result["overlays"]["clip_sound_mixed"] is False
    photo_at = saved["saved"][0]["on_screen"]["at"] + 2.0
    clip_at = saved["saved"][1]["on_screen"]["at"] + 1.0
    assert _diff(_frame(out, photo_at), _frame(plain, photo_at)) > 20  # a white photo covers the pattern
    assert _mean(_frame(out, photo_at)) > 200
    assert _diff(_frame(out, clip_at), _frame(plain, clip_at)) > 1  # a box covers part of it
    assert _diff(_frame(out, 6.0), _frame(plain, 6.0)) < 1  # nothing drawn between them
    project = open_project(str(video))
    assert not any((project.root / "work").iterdir())  # the first pass's file is gone


def test_preview_draws_overlays_in_its_window(video, jobs, overlay_media):
    tools.set_edit(str(video), CUTS, silences=no_silences)
    overlay_tools.set_overlays(
        str(video), [{"file": str(overlay_media["photo"]), "start": 11.0, "end": 16.0}], picture=False
    )
    result = _finish(jobs, tools.preview(str(video), at=12.0, pad=2.0, jobs=jobs))
    w0, w1 = result["edited_window"]
    assert result["duration"] == pytest.approx(w1 - w0, abs=DURATION_TOLERANCE)
    assert result["overlays"]["drawn"] == ["ov1"]
    clip = Path(result["output_path"])
    # The photo starts inside the cut, so it appears where the talk resumes: 1.85s into this clip.
    before, during = _frame(clip, 0.5), _frame(clip, 3.0)
    assert _mean(during) > 200 and _diff(before, during) > 20


def test_hidden_overlays_are_not_drawn_and_say_why(video, jobs, overlay_media):
    tools.set_edit(str(video), CUTS, silences=no_silences)
    overlay_tools.set_overlays(
        str(video), [{"file": str(overlay_media["photo"]), "start": 8.2, "end": 10.2}], picture=False
    )
    result = _finish(jobs, tools.render(str(video), jobs=jobs))
    assert result["overlays"]["drawn"] == [] and result["overlays"]["not_shown"] == ["ov1"]
    assert "is not shown" in result["overlays"]["notes"][0]
    assert Path(result["output_path"]).stat().st_size > 0


def test_a_missing_overlay_file_fails_the_render_before_encoding(video, jobs, overlay_media):
    overlay_tools.set_overlays(
        str(video), [{"file": str(overlay_media["photo"]), "start": 1.0, "end": 3.0}], picture=False
    )
    overlay_media["photo"].unlink()
    status = jobs.wait(tools.render(str(video), jobs=jobs)["job_id"], timeout=60)
    assert status["status"] == "failed" and "call set_overlays without ov1" in status["error"]
    project = open_project(str(video))
    assert list(project.exports_dir.iterdir()) == []


def test_no_overlays_file_means_the_single_pass_of_before(video, jobs):
    tools.set_edit(str(video), CUTS, silences=no_silences)
    result = _finish(jobs, tools.render(str(video), jobs=jobs))
    assert "overlays" not in result
    project = open_project(str(video))
    assert not overlays_path(project).exists() and not (project.root / "work").exists()
