"""End to end on a real (synthetic, 20 s) video: set_edit, preview, render, and a cut with many pieces."""

import array
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest
from lumr_studio.engine.render import probe_duration

from lumr_studio import tools
from lumr_studio.errors import StudioError
from lumr_studio.jobs import JobHandle
from lumr_studio.project import open_project
from lumr_studio.render_jobs import plan_picture, render_full, sound_chain, with_input_seek
from lumr_studio.silences import no_silences

DURATION_TOLERANCE = 0.2
FPS = Fraction(25)
FRAME = 1 / 25
SOUND_RATE = 48000
# AAC packs sound in blocks of 1024 samples, so a sound track can run one block over.
AAC_BLOCK = 1024 / SOUND_RATE
TINY = (32, 24)
CUTS = [
    {"start": 7.9, "end": 10.8, "reason": "retake of the intro line"},
    {"start": 3.35, "end": 3.75, "reason": "um"},
]


def _frame(path: Path, t: float) -> bytes:
    """One tiny grayscale frame at time ``t``, for comparing pictures."""
    return subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
         "-vf", "scale=32:24,format=gray", "-f", "rawvideo", "-"],
        capture_output=True, check=True,
    ).stdout


def _diff(a: bytes, b: bytes) -> float:
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def _finish(jobs, job):
    status = jobs.wait(job["job_id"], timeout=120)
    assert status["status"] == "done", status["error"]
    return status["result"]


def test_set_edit_preview_render(video, jobs):
    footage_before = sorted(p.name for p in video.parent.iterdir())
    edit = tools.set_edit(str(video), CUTS, silences=no_silences)
    assert not edit["rejected"]
    new_duration = edit["new_duration"]

    preview = _finish(jobs, tools.preview(str(video), at=9.0, pad=2.0, jobs=jobs))
    assert Path(preview["output_path"]).exists()
    assert preview["at_is_inside_a_cut"] is True
    w0, w1 = preview["edited_window"]
    assert preview["duration"] == pytest.approx(w1 - w0, abs=DURATION_TOLERANCE)
    # The clip opens on the source frame at its first span, not at a wrong seek.
    first_span_start = preview["source_spans"][0][0]
    clip_open = _frame(Path(preview["output_path"]), 0.0)
    assert _diff(clip_open, _frame(video, first_span_start)) < _diff(clip_open, _frame(video, 0.0)) / 4

    first = _finish(jobs, tools.render(str(video), jobs=jobs))
    out = Path(first["output_path"])
    assert out.exists() and out.parent.name == "exports"
    assert probe_duration(out) == pytest.approx(new_duration, abs=DURATION_TOLERANCE)

    second = _finish(jobs, tools.render(str(video), jobs=jobs))
    assert second["output_path"] != first["output_path"]  # never overwrite an export
    assert out.exists()

    # Nothing new beside the source video.
    assert sorted(p.name for p in video.parent.iterdir()) == footage_before


def test_preview_rejects_bad_inputs(video, jobs):
    with pytest.raises(StudioError, match="outside the video"):
        tools.preview(str(video), at=50.0, jobs=jobs)
    with pytest.raises(StudioError, match="pad"):
        tools.preview(str(video), at=5.0, pad=0, jobs=jobs)


def test_input_seek_goes_before_the_input():
    cmd = ["ffmpeg", "-nostats", "-i", "in.mp4", "out.mp4"]
    assert with_input_seek(cmd, 12.5) == ["ffmpeg", "-nostats", "-ss", "12.500", "-i", "in.mp4", "out.mp4"]


# ── a cut with many pieces ────────────────────────────────────────────────────

# 38 removals in 20 s, none on a frame's edge: 39 kept pieces of about a third of a second.
REMOVED = [(round(0.137 + i * 0.5, 3), round(0.137 + i * 0.5 + 0.171, 3)) for i in range(38)]
KEPT = [(a, b) for a, b in zip([0.0] + [e for _s, e in REMOVED], [s for s, _e in REMOVED] + [20.0])]


def _every_frame(path: Path) -> list[bytes]:
    """Each frame of ``path``, tiny and gray, in order."""
    w, h = TINY
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf", f"scale={w}:{h},format=gray",
         "-fps_mode", "passthrough", "-f", "rawvideo", "-"],
        capture_output=True, check=True,
    ).stdout
    return [raw[i:i + w * h] for i in range(0, len(raw), w * h)]


def _is_source_frame(made: bytes, source: list[bytes], n: int) -> bool:
    """Whether ``made`` is frame ``n`` of the source and not the frame before or after it."""
    beside = [source[i] for i in (n - 1, n + 1) if 0 <= i < len(source)]
    return all(_diff(made, source[n]) < _diff(made, other) for other in beside)


def _planned_frames(kept) -> list[int]:
    """The source frame that each frame of the result should be, held frames included."""
    picture = plan_picture(kept, FPS)
    planned = [picture.shown[0].first] * picture.frames
    for run in picture.shown:
        for k in range(run.count):
            planned[run.lands_on + k] = run.first + k
        for held in range(run.lands_on + run.count, picture.frames):
            planned[held] = run.first + run.count - 1
    return planned


def _streams(path: Path) -> dict[str, float]:
    """The length of the picture and of the sound in ``path``, in seconds."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return {s["codec_type"]: float(s["duration"]) for s in json.loads(out)["streams"]}


def test_many_pieces_come_out_as_long_as_the_edit_and_in_step(video):
    project = open_project(str(video))
    result = render_full(project, REMOVED, [], JobHandle())
    out = Path(result["output_path"])
    expected = sum(e - s for s, e in KEPT)

    assert result["kept_segments"] == len(KEPT) == 39 and "note" not in result
    assert result["expected_duration"] == pytest.approx(expected, abs=1e-3)
    lengths = _streams(out)
    assert lengths["video"] == pytest.approx(expected, abs=FRAME / 2 + 1e-6)
    assert expected - 1e-3 <= lengths["audio"] <= expected + AAC_BLOCK + 1e-3
    assert result["duration"] == pytest.approx(expected, abs=FRAME)

    # Every frame of the result is the source frame the plan names, held frames included.
    source, made, planned = _every_frame(video), _every_frame(out), _planned_frames(KEPT)
    assert len(made) == len(planned)
    wrong = [n for n, frame in enumerate(made) if not _is_source_frame(frame, source, planned[n])]
    assert wrong == []
    for frame in planned:
        assert any(start <= frame * FRAME < end for start, end in KEPT)  # none of what was removed


def test_the_sound_is_cut_on_the_sample_at_every_join(tmp_path):
    """The source's sound here is a slow ramp, so each sample's value says when it was recorded."""
    seconds = 20
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi",
         "-i", f"aevalsrc='t/{seconds}':s={SOUND_RATE}:d={seconds}",
         "-filter_complex", sound_chain(KEPT, Fraction(0)), "-map", "[outa]", "-f", "f32le", "-"],
        capture_output=True, check=True,
    ).stdout
    sound = array.array("f")
    sound.frombytes(raw)

    lengths = [round(e * SOUND_RATE) - round(s * SOUND_RATE) for s, e in KEPT]
    assert len(sound) == sum(lengths)
    one_sample = 1 / SOUND_RATE / seconds
    at = 0
    for (start, _end), length in zip(KEPT, lengths):
        middle = length // 2  # clear of the fades at the joins
        recorded_at = (round(start * SOUND_RATE) + middle) / SOUND_RATE
        assert abs(sound[at + middle] - recorded_at / seconds) < one_sample / 2
        at += length


def test_a_preview_late_in_the_video_is_cut_to_the_frame(video, jobs):
    tools.set_edit(str(video), CUTS, silences=no_silences)
    preview = _finish(jobs, tools.preview(str(video), at=15.0, pad=1.5, jobs=jobs))
    spans = [tuple(span) for span in preview["source_spans"]]
    assert spans[0][0] > 10  # so the read starts late in the source
    source, made = _every_frame(video), _every_frame(Path(preview["output_path"]))
    planned = _planned_frames(spans)
    assert len(made) == len(planned)
    assert [n for n, frame in enumerate(made) if not _is_source_frame(frame, source, planned[n])] == []
