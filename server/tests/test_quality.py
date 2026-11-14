"""The quality of a finished video and of an encoded reel: standard as it always was, and master.

Made-up footage only. The commands are pinned as lists; the renders and the
joins that run ffmpeg skip cleanly without it, through the ``video`` fixture.
"""

import hashlib
import json
import re
import shutil
import subprocess
import threading
from pathlib import Path

import pytest
from lumr_studio.engine.render import RenderPlan, build_ffmpeg_cmd

from lumr_studio import overlay_render, overlay_tools, reels, render_jobs, tools
from lumr_studio.errors import StudioError
from lumr_studio.jobs import JobHandle
from lumr_studio.overlay_render import AUDIO_ARGS, Item, build_graph, overlay_command
from lumr_studio.project import open_project
from lumr_studio.quality import IN_WORDS, MASTER, QUALITIES, STANDARD, check_quality, file_facts
from lumr_studio.render_jobs import (
    MADE,
    REEL_HOLDS_IT_BACK,
    RENDER_KIND,
    Ffmpeg,
    cut_command,
    cut_edit,
    plan_cut,
    reel_holds_back,
    render_full,
    start_full_render,
    written_as,
)
from lumr_studio.silences import no_silences

from conftest import FakeAligner, make_words
from creator_words import banned_in
from test_overlays import make_image

QUIET = ("ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-nostats", "-progress", "pipe:1")
X264 = ("-c:v", "libx264", "-preset", "medium", "-profile:v", "high", "-pix_fmt", "yuv420p")
WORKING = ("-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-pix_fmt", "yuv420p")
STANDARD_SOUND = ("-c:a", "aac", "-b:a", "192k")
MASTER_SOUND = ("-c:a", "aac", "-b:a", "320k")
FINISH = ("-movflags", "+faststart")
CUTS = [{"start": 7.9, "end": 10.8, "reason": "the opening line said a second time"}]
# How much closer to its source a master export has to score than a standard one, in dB of PSNR.
# Measured on the footage made here: 5.2 dB, and 6.4 with a photo drawn over the talk (43.4 against 48.6,
# and 41.4 against 47.8). The encodes are the same every run, so three leaves room for another ffmpeg.
CLEARLY_CLOSER = 3.0


def plan_of(kept, **fields) -> RenderPlan:
    base = dict(
        source=Path("/footage/bread.mp4"), out_path=Path("/exports/bread-edit.mp4"), kept=kept, removed=[],
        duration=60.0, out_height=720, source_width=1280, source_height=720, source_note="test",
    )
    return RenderPlan(**{**base, **fields})


def photo_item() -> Item:
    overlay = {
        "id": "ov1", "kind": "image", "file": "/media/loaf.jpg", "start": 2.0, "end": 6.0,
        "place": "full", "fit": "fill", "size": None, "motion": "still", "layer": 1, "fade": True,
        "media": {"width": 400, "height": 300, "duration": None, "has_sound": False},
    }
    return Item(overlay, 2.0, 4.0)


def clip_item(volume: float) -> Item:
    overlay = {
        "id": "ov2", "kind": "video", "file": "/media/kneading.mp4", "start": 3.0, "end": 7.0,
        "place": "full", "fit": "fill", "size": None, "motion": None, "layer": 1, "fade": True,
        "clip_in": 1.5, "clip_out": 5.5, "volume": volume,
        "media": {"width": 160, "height": 120, "duration": 6.0, "has_sound": True},
    }
    return Item(overlay, 3.0, 4.0)


# ── The two qualities ─────────────────────────────────────────────────────────


def test_there_are_two_qualities_and_standard_is_what_every_file_was():
    assert set(QUALITIES) == {"standard", "master"}
    standard, master = QUALITIES[STANDARD], QUALITIES[MASTER]
    assert (standard.crf, standard.reel_crf) == (19, int(reels.REEL_CRF)) == (19, 16)
    assert standard.sound == AUDIO_ARGS == STANDARD_SOUND and standard.reel_sound == reels.REEL_AUDIO_ARGS
    assert not standard.working_file and not standard.is_master
    assert (master.crf, master.reel_crf) == (12, 12)
    assert master.sound == master.reel_sound == MASTER_SOUND
    assert master.working_file and master.is_master
    assert check_quality("master") is master and check_quality(" Standard ") is standard


@pytest.mark.parametrize("bad", ["best", "4k", "", "high", None, 12, ["master"]])
def test_a_quality_that_is_not_one_is_refused_in_plain_words(bad):
    with pytest.raises(StudioError) as refused:
        check_quality(bad)
    said = str(refused.value)
    assert '"standard"' in said and '"master"' in said, "it names the two there are"
    assert IN_WORDS[STANDARD] in said and IN_WORDS[MASTER] in said, "and says what each is for"
    assert_creator_can_read(said)


def assert_creator_can_read(text: str) -> None:
    """No editor's shorthand, no settings by their short names, no em dash."""
    assert not banned_in(text), (banned_in(text), text)
    assert not re.search(r"\b(crf|bitrate|bit rate|mbps|kbps|encod\w*|qp|aac|x264|codec)\b", text, re.I), text
    assert "—" not in text and "–" not in text, text


def test_what_claude_passes_on_about_a_file_is_in_plain_words():
    assert set(MADE) == {(STANDARD, False), (STANDARD, True), (MASTER, False), (MASTER, True)}
    for said in [*MADE.values(), *IN_WORDS.values(), reels.COPY_IS_WHOLE,
                 reels.quality_note(MASTER, STANDARD), reels.quality_note(STANDARD, MASTER)]:
        assert_creator_can_read(said)
    assert IN_WORDS[MASTER] == "for uploading: as close to your footage as an export gets, and a much bigger file"
    assert "compressed once" in MADE[(MASTER, True)] and "working file" in MADE[(MASTER, True)]
    assert "compressed twice" in MADE[(STANDARD, True)]
    assert "—" not in REEL_HOLDS_IT_BACK and "carry_edit" in REEL_HOLDS_IT_BACK


# ── The commands: standard is pinned, master is the settings chosen ───────────


def test_a_standard_cut_is_the_command_it_always_was():
    cut = plan_cut(plan_of([(0.0, 2.0)]), fps=25.0, has_sound=True)
    assert cut.crf == 19 and cut.sound == STANDARD_SOUND and cut.working is False
    assert cut_command(cut, ["-filter_complex", "GRAPH"]) == [
        *QUIET, "-t", "3.000000", "-i", "/footage/bread.mp4", "-filter_complex", "GRAPH",
        "-map", "[outv]", *X264, "-crf", "19", "-map", "[outa]", *STANDARD_SOUND, *FINISH, "/exports/bread-edit.mp4",
    ]
    named = plan_cut(plan_of([(0.0, 2.0)], crf=QUALITIES[STANDARD].crf), fps=25.0, has_sound=True, sound=QUALITIES[STANDARD].sound)
    assert cut_command(named, ["-filter_complex", "GRAPH"]) == cut_command(cut, ["-filter_complex", "GRAPH"])


def test_a_master_cut_keeps_more_of_the_picture_and_the_sound():
    master = QUALITIES[MASTER]
    cut = plan_cut(plan_of([(0.0, 2.0)], crf=master.crf), fps=25.0, has_sound=True, sound=master.sound)
    assert cut_command(cut, ["-filter_complex", "GRAPH"]) == [
        *QUIET, "-t", "3.000000", "-i", "/footage/bread.mp4", "-filter_complex", "GRAPH",
        "-map", "[outv]", *X264, "-crf", "12", "-map", "[outa]", *MASTER_SOUND, *FINISH, "/exports/bread-edit.mp4",
    ]


def test_a_working_file_loses_nothing_and_is_written_fast():
    master = QUALITIES[MASTER]
    cut = plan_cut(plan_of([(0.0, 2.0)], crf=master.crf), fps=25.0, has_sound=True, sound=master.sound, working=True)
    assert cut_command(cut, ["-filter_complex", "GRAPH"]) == [
        *QUIET, "-t", "3.000000", "-i", "/footage/bread.mp4", "-filter_complex", "GRAPH",
        "-map", "[outv]", *WORKING, "-map", "[outa]", "-c:a", "alac", "/exports/bread-edit.mp4",
    ]
    silent = plan_cut(plan_of([(0.0, 2.0)]), fps=25.0, has_sound=False, working=True)
    assert cut_command(silent, ["-filter_complex", "GRAPH"])[-11:] == ["-map", "[outv]", *WORKING, "/exports/bread-edit.mp4"]


def test_a_standard_overlay_pass_is_the_command_it_always_was(tmp_path):
    out = tmp_path / "out.mp4"
    quiet = build_graph([photo_item(), clip_item(0.0)], (1280, 720), 25.0)
    before = [*QUIET, "-i", "/w/base.mp4", *quiet.inputs, "-filter_complex", quiet.filter]
    assert overlay_command(Path("/w/base.mp4"), out, quiet, crf=19) == [
        *before, "-map", "[vout]", *X264, "-crf", "19", "-map", "0:a?", "-c:a", "copy", *FINISH, str(out),
    ]
    mixed = build_graph([clip_item(0.2)], (1280, 720), 25.0)
    before = [*QUIET, "-i", "/w/base.mp4", *mixed.inputs, "-filter_complex", mixed.filter]
    assert overlay_command(Path("/w/base.mp4"), out, mixed, crf=19) == [
        *before, "-map", "[vout]", *X264, "-crf", "19", "-map", "[aout]", *STANDARD_SOUND, *FINISH, str(out),
    ]


def test_a_master_overlay_pass_is_the_one_encode_of_picture_and_sound(tmp_path):
    out = tmp_path / "out.mp4"
    master = QUALITIES[MASTER]
    quiet = build_graph([photo_item()], (1280, 720), 25.0)
    cmd = overlay_command(Path("/w/base.mp4"), out, quiet, crf=master.crf, sound=master.sound, copy_sound=False)
    assert cmd[cmd.index("-map"):] == [
        "-map", "[vout]", *X264, "-crf", "12", "-map", "0:a?", *MASTER_SOUND, *FINISH, str(out),
    ], "the working file's sound was never encoded, so this pass does it"
    mixed = build_graph([clip_item(0.2)], (1280, 720), 25.0)
    cmd = overlay_command(Path("/w/base.mp4"), out, mixed, crf=master.crf, sound=master.sound, copy_sound=False)
    assert cmd[cmd.index("-map"):] == ["-map", "[vout]", *X264, "-crf", "12", "-map", "[aout]", *MASTER_SOUND, *FINISH, str(out)]


def test_an_old_ffmpeg_still_gets_the_quality_asked_for():
    cmd = build_ffmpeg_cmd(Path("/footage/bread.mp4"), Path("/exports/out.mp4"), [(0.0, 2.0)], crf=12, progress_url="pipe:1")
    settings = cmd.index("-c:v")
    master = written_as(cmd, 12, MASTER_SOUND, False)
    assert master[:settings] == cmd[:settings]
    assert master[settings:] == [*X264, "-crf", "12", *MASTER_SOUND, *FINISH, "/exports/out.mp4"]
    assert written_as(cmd, 12, MASTER_SOUND, True)[settings:] == [*WORKING, "-c:a", "alac", "/exports/out.mp4"]


@pytest.mark.parametrize("quality", [STANDARD, MASTER])
def test_the_slow_way_runs_clipforges_own_command_at_standard_and_only_then(quality, tmp_path, monkeypatch):
    ran: list[list[str]] = []

    def no_ffmpeg(cmd, **_kwargs):
        ran.append(list(cmd))
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(render_jobs, "this_ffmpeg", lambda: Ffmpeg(version=4, splits_sound=False))
    monkeypatch.setattr(subprocess, "Popen", no_ffmpeg)
    made_at = QUALITIES[quality]
    plan = plan_of([(0.0, 2.0), (3.0, 4.0)], out_path=tmp_path / "exports" / "out.mp4", crf=made_at.crf)
    with pytest.raises(StudioError):
        cut_edit(plan, JobHandle(), "label", sound=made_at.sound)
    clipforge = build_ffmpeg_cmd(plan.source, plan.out_path, plan.kept, source_height=720, crf=made_at.crf, progress_url="pipe:1")
    assert (ran == [clipforge]) is (quality == STANDARD)
    assert ("320k" in ran[0]) is (quality == MASTER)


# ── Real renders: which settings each pass got, and what the result says ──────


@pytest.fixture
def passes(monkeypatch):
    """Every ffmpeg command a render runs, in order. They still run."""
    seen: list[list[str]] = []
    real = overlay_render.run_ffmpeg

    def spy(cmd, out, **kwargs):
        seen.append(list(cmd))
        return real(cmd, out, **kwargs)

    monkeypatch.setattr(render_jobs, "run_ffmpeg", spy)
    monkeypatch.setattr(overlay_render, "run_ffmpeg", spy)
    return seen


def settings_of(cmd: list[str]) -> list[str]:
    """A command from its first ``-map`` on, without the file it writes."""
    return cmd[cmd.index("-map"):-1]


def finish(jobs, job) -> dict:
    status = jobs.wait(job["job_id"], timeout=120)
    assert status["status"] == "done", status["error"]
    return status["result"]


def save_a_photo(video: Path, folder: Path, start: float = 1.0, end: float = 3.0) -> None:
    folder.mkdir(exist_ok=True)
    saved = overlay_tools.set_overlays(str(video), [
        {"file": str(make_image(folder / "loaf.jpg", (400, 300), "white")), "start": start, "end": end,
         "motion": "still", "place": "top_right", "size": 0.3},
    ], picture=False)
    assert not saved["rejected"], saved


def receipts(video: Path, step: str) -> list[dict]:
    lines = open_project(str(video)).receipts_path.read_text().splitlines()
    return [r for r in map(json.loads, lines) if r["step"] == step]


def test_a_standard_render_runs_the_commands_it_always_ran(video, jobs, passes, tmp_path):
    tools.set_edit(str(video), CUTS, silences=no_silences)
    answer = tools.render(str(video), jobs=jobs)
    assert answer["quality"] == "standard" and "reel_note" not in answer
    result = finish(jobs, answer)
    assert [settings_of(c) for c in passes] == [
        ["-map", "[outv]", *X264, "-crf", "19", "-map", "[outa]", *STANDARD_SOUND, *FINISH],
    ]
    assert result["quality"] == "standard" and result["made"] == MADE[(STANDARD, False)]

    del passes[:]
    save_a_photo(video, tmp_path / "media")
    result = finish(jobs, tools.render(str(video), "standard", jobs=jobs))
    assert [settings_of(c) for c in passes] == [
        ["-map", "[outv]", *X264, "-crf", "19", "-map", "[outa]", *STANDARD_SOUND, *FINISH],
        ["-map", "[vout]", *X264, "-crf", "19", "-map", "0:a?", "-c:a", "copy", *FINISH],
    ]
    assert result["made"] == MADE[(STANDARD, True)]


def test_a_master_render_says_what_was_made_in_its_result_its_receipt_and_get_edit(video, jobs, passes):
    tools.set_edit(str(video), CUTS, silences=no_silences)
    answer = tools.render(str(video), "master", jobs=jobs)
    assert answer["quality"] == "master"
    result = finish(jobs, answer)
    assert [settings_of(c) for c in passes] == [
        ["-map", "[outv]", *X264, "-crf", "12", "-map", "[outa]", *MASTER_SOUND, *FINISH],
    ]
    out = Path(result["output_path"])
    assert result["quality"] == "master" and result["made"] == MADE[(MASTER, False)]
    assert result["size_mb"] == round(out.stat().st_size / 1e6, 1)
    assert result["video_bitrate_mbps"] == file_facts(out)["video_bitrate_mbps"] > 0

    receipt = receipts(video, "render")[-1]
    assert receipt["status"] == "done"
    assert {k: receipt["result"][k] for k in ("quality", "size_mb", "video_bitrate_mbps", "made")} == {
        k: result[k] for k in ("quality", "size_mb", "video_bitrate_mbps", "made")
    }
    assert tools.get_edit(str(video), jobs=jobs)["export"] == {
        "job_id": answer["job_id"], "status": "done", "edit_changed_since": False,
        "quality": "master", "size_mb": result["size_mb"], "video_bitrate_mbps": result["video_bitrate_mbps"],
    }

    standard = finish(jobs, tools.render(str(video), jobs=jobs))
    assert standard["quality"] == "standard" and standard["output_path"] != result["output_path"]
    assert out.exists(), "nothing is overwritten: the master export is still there"
    assert result["video_bitrate_mbps"] > standard["video_bitrate_mbps"]
    assert tools.get_edit(str(video), jobs=jobs)["export"]["quality"] == "standard", "the latest one"


def test_a_master_render_with_a_photo_over_it_writes_a_working_file_and_encodes_once(video, jobs, passes, tmp_path):
    tools.set_edit(str(video), CUTS, silences=no_silences)
    save_a_photo(video, tmp_path / "media")
    result = finish(jobs, tools.render(str(video), "master", jobs=jobs))
    first, second = passes
    assert settings_of(first) == ["-map", "[outv]", *WORKING, "-map", "[outa]", "-c:a", "alac"]
    assert "-before-overlays" in first[-1] and Path(first[-1]).parent.name == "work"
    assert settings_of(second) == ["-map", "[vout]", *X264, "-crf", "12", "-map", "0:a?", *MASTER_SOUND, *FINISH]
    assert second[second.index("-i") + 1] == first[-1], "the pass that draws reads the working file"
    assert result["made"] == MADE[(MASTER, True)] and result["overlays"]["drawn"] == ["ov1"]
    assert not any((open_project(str(video)).root / "work").iterdir()), "and the working file is gone again"
    sound = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name", "-of", "csv=p=0", result["output_path"]],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert sound == "aac", "the finished video's sound is AAC, not the working file's"


def test_a_master_render_whose_photo_never_shows_is_one_ordinary_master_pass(video, jobs, passes, tmp_path):
    tools.set_edit(str(video), CUTS, silences=no_silences)
    save_a_photo(video, tmp_path / "media", start=8.2, end=10.2)  # inside the cut
    result = finish(jobs, tools.render(str(video), "master", jobs=jobs))
    assert [settings_of(c) for c in passes] == [
        ["-map", "[outv]", *X264, "-crf", "12", "-map", "[outa]", *MASTER_SOUND, *FINISH],
    ], "nothing is drawn, so the first pass is the result and must not be a working file"
    assert result["overlays"]["not_shown"] == ["ov1"] and result["made"] == MADE[(MASTER, False)]


def test_a_preview_is_made_as_it_always_was(video, jobs, passes):
    tools.set_edit(str(video), CUTS, silences=no_silences)
    result = finish(jobs, tools.preview(str(video), at=9.0, pad=2.0, jobs=jobs))
    assert [settings_of(c) for c in passes] == [
        ["-map", "[outv]", *X264, "-crf", "23", "-map", "[outa]", *STANDARD_SOUND, *FINISH],
    ]
    assert "quality" not in result


# ── Starting a render at a quality ────────────────────────────────────────────


class StandIn:
    """A render that encodes nothing and waits to be let go. It notes the quality it was asked for."""

    def __init__(self) -> None:
        self.go = threading.Event()
        self.running = threading.Event()
        self.asked: list[dict] = []

    def __call__(self, project, removed, words, handle, **asked):
        self.asked.append(asked)
        self.running.set()
        assert self.go.wait(10), "the test never let the render finish"
        return {"output_path": str(project.exports_dir / "bread-edit.mp4")}


@pytest.fixture
def stand_in(monkeypatch):
    work = StandIn()
    monkeypatch.setattr(render_jobs, "render_full", work)
    yield work
    work.go.set()


def test_a_standard_render_is_started_as_it_always_was_and_a_master_one_names_its_quality(video, jobs, stand_in):
    tools.set_edit(str(video), CUTS, silences=no_silences)
    project = open_project(str(video))
    first, _ = start_full_render(project, jobs=jobs)
    assert stand_in.running.wait(5)
    assert stand_in.asked == [{}] and "quality" not in jobs.get(first).about
    assert tools.get_edit(str(video), jobs=jobs)["export"]["quality"] == "standard"

    late = tools.render(str(video), "master", jobs=jobs)
    assert late["job_id"] == first and late["already_running"] is True and late["quality"] == "standard"
    assert "standard export, not the master one" in late["note"] and "quality master" in late["note"]
    same = tools.render(str(video), jobs=jobs)
    assert same["quality"] == "standard" and "not the" not in same["note"]

    stand_in.go.set()
    jobs.wait(first, timeout=10)
    stand_in.go.clear()
    stand_in.running.clear()
    second = tools.render(str(video), "master", jobs=jobs)
    assert set(second) == {"job_id", "quality"} and second["quality"] == "master"
    assert stand_in.running.wait(5)
    assert stand_in.asked[-1] == {"quality": "master"} and jobs.get(second["job_id"]).about["quality"] == "master"
    assert tools.get_edit(str(video), jobs=jobs)["export"] == {
        "job_id": second["job_id"], "status": "running", "edit_changed_since": False, "quality": "master",
    }


@pytest.mark.parametrize("bad", ["best", "Master quality", "", None])
def test_render_and_assemble_refuse_a_quality_that_is_not_one_and_start_nothing(bad, video, jobs, stand_in):
    tools.set_edit(str(video), CUTS, silences=no_silences)
    project = open_project(str(video))
    calls = (
        lambda: tools.render(str(video), bad, jobs=jobs),
        lambda: start_full_render(project, jobs=jobs, quality=bad),
        lambda: tools.assemble("bread week 5", [{"clip": str(video), "start": 1, "end": 4}], bad, jobs=jobs),
    )
    for call in calls:
        with pytest.raises(StudioError) as refused:
            call()
        assert f"quality {bad!r} is not one Lumr Studio has" in str(refused.value)
        assert_creator_can_read(str(refused.value))
    assert jobs.latest(RENDER_KIND, project) is None and stand_in.asked == []
    assert not list(project.exports_dir.glob("*")), "nothing was exported"
    assert not (reels.lumr_home() / "studio" / "reels").exists(), "and no reel was begun"


# ── The page stays as it is, and the docs say where a master export comes from ─

PLUGIN_DIR = Path(__file__).resolve().parents[2]


def said_in(relative: str) -> str:
    """A shipped text file as one line, so a phrase is found across its line breaks."""
    return " ".join((PLUGIN_DIR / relative).read_text(encoding="utf-8").split())


def test_the_page_offers_no_quality_and_the_docs_say_a_master_export_is_asked_for_through_claude():
    page = said_in("server/lumr_studio/treatment_page/index.html").lower()
    assert "master" not in page and "quality" not in page, "the page the creator approved is untouched"
    assert "A master export is asked for through Claude" in said_in("TOOLS.md")
    assert "a master one is asked for through Claude" in said_in("GUIDE.md")
    assert "The page's Export video button always makes a standard one" in said_in("skills/tight-cut/SKILL.md")
    for skill in ("tight-cut", "assemble-takes", "splice-clips"):
        text = said_in(f"skills/{skill}/SKILL.md")
        assert '`quality: "master"`' in text and "can only be as good as" in text, skill
        assert "is for upload" in text and "best quality" in text, skill
        assert "carry_edit" in text, skill
    for doc in ("TOOLS.md", "GUIDE.md", "README.md", "server/README.md", *(f"skills/{s}/SKILL.md" for s in ("tight-cut", "assemble-takes", "splice-clips"))):
        assert "—" not in said_in(doc), doc


def test_the_numbers_in_the_contract_are_the_ones_the_code_uses():
    contract = said_in("TOOLS.md")
    master, standard = QUALITIES[MASTER], QUALITIES[STANDARD]
    assert f"| Picture, x264's CRF (lower keeps more) | {standard.crf} | {master.crf} |" in contract
    assert "| Sound | AAC 192k | AAC 320k |" in contract and master.sound[-1] == "320k" and standard.sound[-1] == "192k"
    assert f"| An encoded reel (`assemble`) | CRF {standard.reel_crf}, AAC 256k | CRF {master.reel_crf}, AAC 320k |" in contract
    assert "x264 at `-qp 0`, preset ultrafast, with ALAC sound" in contract


# ── get_edit: what the footage is, before anything is exported ────────────────


def test_get_edit_says_what_the_footage_is_and_how_much_of_it_the_edit_keeps(video, jobs):
    whole = tools.get_edit(str(video), jobs=jobs)["footage"]
    assert whole["resolution"] == "320x240" and whole["fps"] == 25.0
    assert whole["video_bitrate_mbps"] > 0
    assert whole["size_mb"] == whole["kept_mb"] == round(video.stat().st_size / 1e6, 1)
    assert set(whole) == {"resolution", "fps", "video_bitrate_mbps", "size_mb", "kept_mb"}
    edit = tools.set_edit(str(video), [{"start": 2.0, "end": 12.0, "reason": "the whole first try"}], silences=no_silences)
    footage = tools.get_edit(str(video), jobs=jobs)["footage"]
    assert 0 < footage["kept_mb"] < footage["size_mb"]
    assert footage["kept_mb"] == pytest.approx(footage["size_mb"] * edit["new_duration"] / 20.0, abs=0.06)


def test_a_file_that_is_no_video_has_no_facts(tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("flour, water, salt")
    assert file_facts(notes) == {} and file_facts(tmp_path / "gone.mp4") == {}

    def no_ffprobe(*_a, **_k):
        raise FileNotFoundError("ffprobe")

    assert file_facts(notes, run=no_ffprobe) == {}


def test_a_picture_shown_on_its_side_reads_as_it_is_shown(tmp_path):
    video = tmp_path / "phone.mov"
    video.write_bytes(b"x" * 2_500_000)
    stream = {"width": 1920, "height": 1080, "avg_frame_rate": "30000/1001", "bit_rate": "24000000",
              "side_data_list": [{"rotation": -90}]}

    def phone(cmd, **_k):
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"streams": [stream]}), stderr="")

    assert file_facts(video, run=phone) == {
        "resolution": "1080x1920", "fps": 29.97, "video_bitrate_mbps": 24.0, "size_mb": 2.5,
    }


# ── Reels: a master reel is another reel ──────────────────────────────────────

NAME = "bread week 5"


def stretches(video: Path) -> list[dict]:
    """Two stretches of one clip: a start and an end make the reel an encode."""
    return [{"clip": str(video), "start": 1.0, "end": 4.0}, {"clip": str(video), "start": 10.0, "end": 12.0}]


def assembled(jobs, pieces, *quality) -> dict:
    started = tools.assemble(NAME, pieces, *quality, jobs=jobs)
    assert set(started) == {"job_id"}, started
    job = jobs.wait(started["job_id"], timeout=120)
    assert job["status"] == "done", job["error"]
    return job["result"]


def id_before_there_was_a_choice(name: str, pieces: list[reels.Piece]) -> str:
    """The id a sequence of plain clips had before ``quality`` existed, worked out here from scratch."""
    parts = [name, [[str(p.clip), p.size, p.mtime_ns, p.start, p.end] for p in pieces]]
    return hashlib.sha1(json.dumps(parts).encode("utf-8")).hexdigest()[:8]


def test_a_standard_reel_keeps_the_id_and_the_command_it_always_had(video):
    pieces = reels.check_pieces(stretches(video))
    old = id_before_there_was_a_choice(NAME, pieces)
    assert reels.sequence_id(NAME, pieces) == reels.sequence_id(NAME, pieces, "standard") == old
    assert reels.plan_reel(NAME, stretches(video)).reel.name == f"bread-week-5-{old}.mp4"
    assert reels.sequence_id(NAME, pieces, "master") != old

    command = reels.encode_command(pieces, Path("out.mp4"))
    assert command == reels.encode_command(pieces, Path("out.mp4"), "standard")
    assert command[command.index("-map"):] == [
        "-map", "[v]", "-map", "[a]", *X264, "-crf", "16", "-c:a", "aac", "-b:a", "256k", *FINISH, "out.mp4",
    ]
    master = reels.encode_command(pieces, Path("out.mp4"), "master")
    assert master[:master.index("-map")] == command[:command.index("-map")], "the same join"
    assert master[master.index("-map"):] == [
        "-map", "[v]", "-map", "[a]", *X264, "-crf", "12", *MASTER_SOUND, *FINISH, "out.mp4",
    ]


def test_a_master_reel_is_built_beside_the_standard_one_and_never_replaces_it(video, jobs):
    standard = assembled(jobs, stretches(video))
    reel = Path(standard["reel"])
    before = reel.read_bytes()
    assert (standard["status"], standard["how"], standard["quality"]) == ("built", "encode", "standard")
    assert standard["size_mb"] == round(len(before) / 1e6, 1)

    master = assembled(jobs, stretches(video), "master")
    other = Path(master["reel"])
    assert (master["status"], master["how"], master["quality"]) == ("built", "encode", "master")
    assert other != reel and other.parent == reel.parent and master["sequence_file"] != standard["sequence_file"]
    assert reel.read_bytes() == before, "the standard reel is as it was"
    assert other.stat().st_size > len(before), "a master reel keeps more, so it is bigger"
    assert master["duration"] == standard["duration"] and master["pieces"] == standard["pieces"]
    assert sorted(p.name for p in reel.parent.iterdir()) == sorted(
        [reel.name, other.name, Path(standard["sequence_file"]).name, Path(master["sequence_file"]).name]
    )
    assert json.loads(Path(standard["sequence_file"]).read_text())["quality"] == "standard"
    assert json.loads(Path(master["sequence_file"]).read_text())["quality"] == "master"

    again = tools.assemble(NAME, stretches(video), jobs=jobs)
    assert (again["status"], again["reel"], again["quality"]) == ("exists", str(reel), "standard")
    again = tools.assemble(NAME, stretches(video), "master", jobs=jobs)
    assert (again["status"], again["reel"], again["quality"]) == ("exists", str(other), "master")
    assert reels.reel_for(reel)["quality"] == "standard" and reels.reel_for(other)["quality"] == "master"
    assert tools.get_edit(str(other), jobs=jobs)["reel"]["how"] == "encode"

    made = receipts(other, "assemble")[-1]["result"]
    assert (made["quality"], made["size_mb"]) == ("master", master["size_mb"])


def test_a_reel_encoded_before_there_was_a_choice_is_still_this_sequence_at_standard(video, jobs):
    built = assembled(jobs, stretches(video))
    sequence = Path(built["sequence_file"])
    record = json.loads(sequence.read_text())
    del record["quality"]  # as every sequence file was written before
    sequence.write_text(json.dumps(record))
    reel = Path(built["reel"])
    before = reel.read_bytes()

    again = tools.assemble(NAME, stretches(video), jobs=jobs)
    assert (again["status"], again["reel"], again["quality"]) == ("exists", built["reel"], "standard")
    assert reels.reel_for(reel) == {"name": NAME, "how": "encode", "quality": "standard", "pieces": again["pieces"]}
    assert reel_holds_back(reel, QUALITIES[MASTER]) == REEL_HOLDS_IT_BACK

    master = assembled(jobs, stretches(video), "master")
    assert master["reel"] != built["reel"] and reel.read_bytes() == before


def test_a_copy_loses_nothing_so_it_is_one_reel_at_either_quality_and_says_so(video, jobs):
    twin = video.parent / "second rise.mp4"
    shutil.copy(video, twin)
    whole = [{"clip": str(video)}, {"clip": str(twin)}]
    standard = assembled(jobs, whole)
    assert standard["how"] == "copy" and "quality" not in standard and "quality" not in standard["note"]
    before = Path(standard["reel"]).read_bytes()

    master = tools.assemble(NAME, whole, "master", jobs=jobs)
    assert (master["status"], master["reel"], master["how"]) == ("exists", standard["reel"], "copy")
    assert "quality" not in master and master["note"].endswith(reels.COPY_IS_WHOLE)
    assert Path(standard["reel"]).read_bytes() == before
    assert "quality" not in json.loads(Path(standard["sequence_file"]).read_text())
    assert reels.reel_for(Path(standard["reel"])) == {"name": NAME, "how": "copy", "pieces": master["pieces"]}
    assert len(list(Path(standard["reel"]).parent.glob("*.mp4"))) == 1

    pieces = reels.check_pieces(whole)
    assert reels.sequence_id(NAME, pieces, "master") == reels.sequence_id(NAME, pieces) == id_before_there_was_a_choice(NAME, pieces)


def test_a_reel_at_another_quality_than_asked_says_so_and_what_to_do():
    assert reels.quality_note("standard", "standard") == reels.quality_note("master", "master") == ""
    assert reels.quality_note("standard", None) == "" and reels.quality_note("master", None) == reels.COPY_IS_WHOLE
    said = reels.quality_note("master", "standard")
    assert "joined before at standard quality" in said and "For a master one" in said and "another name" in said


def test_a_master_render_of_a_standard_reel_says_the_reel_holds_it_back(video, jobs, stand_in):
    assert reel_holds_back(video, QUALITIES[MASTER]) is None, "a plain video is its own footage"
    standard = Path(assembled(jobs, stretches(video))["reel"])
    master = Path(assembled(jobs, stretches(video), "master")["reel"])
    twin = video.parent / "second rise.mp4"
    shutil.copy(video, twin)
    copied = Path(assembled(jobs, [{"clip": str(video)}, {"clip": str(twin)}])["reel"])
    assert reel_holds_back(standard, QUALITIES[MASTER]) == REEL_HOLDS_IT_BACK
    assert reel_holds_back(standard, QUALITIES[STANDARD]) is None
    assert reel_holds_back(master, QUALITIES[MASTER]) is None and reel_holds_back(copied, QUALITIES[MASTER]) is None

    answer = tools.render(str(standard), "master", jobs=jobs)
    assert answer["reel_note"] == REEL_HOLDS_IT_BACK and answer["quality"] == "master"
    assert "reel_note" not in tools.render(str(master), "master", jobs=jobs)
    assert "assemble at quality master" in REEL_HOLDS_IT_BACK and "can only be as good as that reel" in REEL_HOLDS_IT_BACK


def test_an_edit_made_on_a_standard_reel_is_carried_onto_its_master_twin_whole(video, jobs):
    takes = [{"clip": str(video), "start": 0.0, "end": 7.0}, {"clip": str(video), "start": 11.4, "end": 17.4}]
    built = assembled(jobs, takes)
    old = Path(built["reel"])
    words = []
    for piece in built["pieces"]:
        shift = piece["reel_start"] - piece["start"]
        words += [
            {**w, "start": round(w["start"] + shift, 3), "end": round(w["end"] + shift, 3)}
            for w in make_words() if piece["start"] <= (w["start"] + w["end"]) / 2 < piece["end"]
        ]
    old.with_suffix(".words.json").write_text(json.dumps(words))
    quiet = {"silences": no_silences, "labels": tools.unmeasured_labels}
    measuring = tools.transcribe(str(old), aligner=FakeAligner(), jobs=jobs, **quiet)
    assert jobs.wait(measuring["job_id"], timeout=60)["status"] == "done"
    cuts = [{"start": 3.85, "end": 6.65, "reason": "Starts the line and then starts it again", "kind": "false_start"}]
    saved = tools.set_edit(str(old), cuts, **quiet)
    assert not saved["rejected"]

    new = Path(assembled(jobs, takes, "master")["reel"])
    heard: list[Path] = []
    carrying = tools.carry_edit(
        str(old), str(new), transcriber=lambda path, force: heard.append(path), aligner=FakeAligner(), jobs=jobs, **quiet,
    )
    report = jobs.wait(carrying["job_id"], timeout=120)
    assert report["status"] == "done", report["error"]
    report = report["result"]
    assert heard == [], "the same footage in the same places: nothing is transcribed again"
    assert (report["cuts"]["total"], report["cuts"]["carried"]) == (1, 1)
    assert not report["cuts"]["dropped"] and not report["cuts"]["split"] and not report["cuts"]["shortened"]
    assert report["new_duration"] == saved["new_duration"]
    before, after = (tools.get_edit(str(reel), jobs=jobs, labels=tools.unmeasured_labels) for reel in (old, new))
    assert [(c["start"], c["end"], c["reason"]) for c in after["cuts"]] == [(c["start"], c["end"], c["reason"]) for c in before["cuts"]]
    assert (before["reel"]["quality"], after["reel"]["quality"]) == ("standard", "master")
    assert reel_holds_back(old, QUALITIES[MASTER]) and reel_holds_back(new, QUALITIES[MASTER]) is None


# ── One real comparison: master is clearly closer to its source ───────────────


def footage_with_detail(folder: Path, seconds: float = 4.0) -> Path:
    """A made-up clip with fine detail that moves, written the way a camera writes: far more data than an export keeps."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "bread.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "mandelbrot=size=640x360:rate=25,format=yuv420p",
            "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=48000",
            "-t", str(seconds), "-c:v", "libx264", "-preset", "ultrafast", "-crf", "4", "-pix_fmt", "yuv420p",
            "-ac", "2", "-c:a", "aac", "-b:a", "256k", str(path),
        ],
        check=True,
    )
    return path


def psnr(made: Path, source: Path, start: float = 0.0) -> float:
    """How close ``made`` is to ``source`` from ``start`` on, in dB, by ffmpeg's own psnr filter. Higher is closer."""
    said = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-ss", f"{start:.3f}", "-i", str(made), "-ss", f"{start:.3f}", "-i", str(source),
         "-lavfi", "[0:v][1:v]psnr", "-an", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    ).stderr
    found = re.search(r"PSNR .*?average:([\d.]+)", said)
    assert found, said[-400:]
    return float(found.group(1))


def test_a_master_export_is_clearly_closer_to_its_source_than_a_standard_one(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    source = footage_with_detail(tmp_path / "footage")
    project = open_project(str(source))

    plain = {q: render_full(project, [], [], JobHandle(), quality=q) for q in (STANDARD, MASTER)}
    closeness = {q: psnr(Path(made["output_path"]), source) for q, made in plain.items()}
    assert closeness[MASTER] > closeness[STANDARD] + CLEARLY_CLOSER, closeness
    assert plain[MASTER]["video_bitrate_mbps"] > plain[STANDARD]["video_bitrate_mbps"]
    assert plain[MASTER]["size_mb"] > plain[STANDARD]["size_mb"]

    # With a photo over the first second, the rest of the picture is what the edit alone would be:
    # a master export pays for one encode there, a standard one for two.
    save_a_photo(source, tmp_path / "media", start=0.2, end=1.4)
    after_the_photo = 2.0
    drawn = {q: render_full(project, [], [], JobHandle(), quality=q) for q in (STANDARD, MASTER)}
    assert all(made["overlays"]["drawn"] == ["ov1"] for made in drawn.values())
    over = {q: psnr(Path(made["output_path"]), source, after_the_photo) for q, made in drawn.items()}
    alone = {q: psnr(Path(made["output_path"]), source, after_the_photo) for q, made in plain.items()}
    assert over[MASTER] == pytest.approx(alone[MASTER], abs=0.3), "the working file cost the picture nothing"
    assert over[STANDARD] < alone[STANDARD] - 0.3, "a standard export pays for its second encode"
    assert over[MASTER] > over[STANDARD] + CLEARLY_CLOSER, (over, alone)
