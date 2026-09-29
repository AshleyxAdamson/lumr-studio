"""Starting a full render and reading it as an export, and how the cut is planned. No video is encoded here."""

import json
import subprocess
import threading
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest
from lumr_studio.engine.render import AUDIO_JOIN_FADE, RenderPlan, build_filter_complex

from lumr_studio import render_jobs, tools
from lumr_studio.errors import StudioError
from lumr_studio.jobs import JobHandle
from lumr_studio.project import open_project
from lumr_studio.render_jobs import (
    EXPORT_FAILED,
    NOTHING_EXPORTED,
    OLD_FFMPEG_NOTE,
    RENDER_KIND,
    Ffmpeg,
    cut_command,
    cut_edit,
    edit_mark,
    export_status,
    first_pass_share,
    Picture,
    Shown,
    frame_rate,
    picture_chain,
    plan_cut,
    plan_picture,
    read_from,
    sound_chain,
    start_full_render,
    this_ffmpeg,
)
from lumr_studio.silences import no_silences

EXPORT_KEYS = {"state", "progress", "file", "folder", "message", "holds_earlier_edit"}
CUTS = [{"start": 7.9, "end": 10.75, "reason": "second take of the opening line", "kind": "repeat"}]
QUIET = {"silences": no_silences, "labels": tools.unmeasured_labels}
EXPORTED_NAME = "talk-edit-20260101-120000.mp4"


class StandInRender:
    """A render that encodes nothing. It reports 42%, then waits for the test to let it finish."""

    def __init__(self) -> None:
        self.go = threading.Event()
        self.running = threading.Event()
        self.calls: list[list[tuple[float, float]]] = []
        self.fail_with: Exception | None = None

    def __call__(self, project, removed, words, handle):
        self.calls.append(list(removed))
        handle.set_progress(0.42)
        self.running.set()
        assert self.go.wait(10), "the test never let the export finish"
        if self.fail_with is not None:
            raise self.fail_with
        return {"output_path": str(project.exports_dir / EXPORTED_NAME)}

    def finish(self, jobs, job_id) -> dict:
        self.go.set()
        return jobs.wait(job_id, timeout=10)


@pytest.fixture
def render():
    stand_in = StandInRender()
    yield stand_in
    stand_in.go.set()


@pytest.fixture
def project(video):
    tools.set_edit(str(video), CUTS, **QUIET)
    return open_project(str(video))


def _edit(project) -> dict:
    return json.loads(project.edit_path.read_text())


def status(project, jobs) -> dict:
    return export_status(project, jobs, edit=_edit(project))


# ── starting ──────────────────────────────────────────────────────────────────


def test_a_full_render_gets_the_saved_edits_removed_spans(project, jobs, render):
    job_id, started = start_full_render(project, jobs=jobs, render=render)
    assert started and job_id.startswith("render-")
    assert render.running.wait(5)
    assert [(round(a, 2), round(b, 2)) for a, b in render.calls[0]] == [
        (round(c["start"], 2), round(c["end"], 2)) for c in _edit(project)["cuts"]
    ]
    assert jobs.get(job_id).kind == RENDER_KIND
    assert jobs.get(job_id).about == {"edit_mark": edit_mark(_edit(project))}


def test_only_one_full_render_runs_per_video(project, jobs, render):
    first, _ = start_full_render(project, jobs=jobs, render=render)
    assert start_full_render(project, jobs=jobs, render=render) == (first, False)
    assert render.running.wait(5) and len(render.calls) == 1
    render.finish(jobs, first)
    second, started = start_full_render(project, jobs=jobs, render=render)
    assert started and second != first


def test_an_edit_that_cannot_be_read_starts_nothing(project, jobs, render):
    project.edit_path.write_text("{not json")
    with pytest.raises(StudioError, match="not valid JSON"):
        start_full_render(project, jobs=jobs, render=render)
    assert jobs.latest(RENDER_KIND, project) is None and render.calls == []


def test_the_real_render_is_the_default_work(project, jobs, monkeypatch):
    seen = []
    monkeypatch.setattr(render_jobs, "render_full", lambda p, removed, words, handle: seen.append(p) or {"ok": 1})
    job_id, _ = start_full_render(project, jobs=jobs)
    assert jobs.wait(job_id, timeout=5)["result"] == {"ok": 1} and seen == [project]


# ── the export, as the page reads it ──────────────────────────────────────────


def test_with_no_render_the_export_is_idle(project, jobs):
    assert status(project, jobs) == {
        "state": "idle", "progress": None, "file": None, "folder": None,
        "message": NOTHING_EXPORTED, "holds_earlier_edit": False,
    }


def test_a_running_export_shows_its_progress(project, jobs, render):
    start_full_render(project, jobs=jobs, render=render)
    assert render.running.wait(5)
    now = status(project, jobs)
    assert set(now) == EXPORT_KEYS
    assert (now["state"], now["progress"], now["file"], now["folder"]) == ("running", 0.42, None, None)
    assert now["message"] == "Exporting. 42% done."


def test_a_running_export_with_no_progress_yet_says_so_plainly(project, jobs):
    hold = threading.Event()
    jobs.start(RENDER_KIND, project, lambda _h: hold.wait(5) and {})
    now = status(project, jobs)
    hold.set()
    assert (now["state"], now["progress"], now["message"]) == ("running", None, "Exporting.")


def test_a_finished_export_names_the_file_and_its_folder(project, jobs, render):
    job_id, _ = start_full_render(project, jobs=jobs, render=render)
    render.finish(jobs, job_id)
    now = status(project, jobs)
    assert (now["state"], now["progress"], now["file"]) == ("done", 1.0, EXPORTED_NAME)
    assert now["folder"] == str(project.exports_dir) and now["message"] == f"Exported {EXPORTED_NAME}."


@pytest.mark.parametrize("failure", [
    StudioError("ffmpeg could not encode talk-edit.mp4: Conversion failed! at 412.30"),
    KeyError("start"),
])
def test_a_failed_export_says_what_to_do_next_and_keeps_the_details_for_claude(project, jobs, render, failure):
    render.fail_with = failure
    job_id, _ = start_full_render(project, jobs=jobs, render=render)
    seen_by_claude = render.finish(jobs, job_id)
    now = status(project, jobs)
    assert (now["state"], now["progress"], now["file"], now["folder"]) == ("failed", None, None, None)
    assert now["message"] == EXPORT_FAILED and "Export video" in now["message"]
    assert "ffmpeg" not in now["message"] and "412" not in now["message"]
    assert seen_by_claude["status"] == "failed" and seen_by_claude["error"]


def test_a_render_with_no_file_to_show_reads_as_failed(project, jobs):
    job_id = jobs.start(RENDER_KIND, project, lambda _h: {"duration": 1.0})
    jobs.wait(job_id, timeout=5)
    assert status(project, jobs)["state"] == "failed"


MORE_CUTS = [*CUTS, {"start": 12.3, "end": 13.1, "reason": "a few words tripped over", "kind": "false_start"}]


def test_an_export_holds_the_edit_as_saved_when_it_started(project, jobs, render):
    video = str(project.video)
    job_id, _ = start_full_render(project, jobs=jobs, render=render)
    assert status(project, jobs)["holds_earlier_edit"] is False
    tools.set_edit(video, MORE_CUTS, **QUIET)
    assert status(project, jobs)["holds_earlier_edit"] is True
    assert tools.get_edit(video, jobs=jobs)["export"]["edit_changed_since"] is True
    render.finish(jobs, job_id)
    after = status(project, jobs)
    assert after["state"] == "done" and after["holds_earlier_edit"] is True


def test_a_change_made_and_undone_leaves_the_export_up_to_date(project, jobs, render):
    video = str(project.video)
    start_full_render(project, jobs=jobs, render=render)
    tools.set_edit(video, MORE_CUTS, **QUIET)
    tools.set_edit(video, CUTS, **QUIET)
    assert status(project, jobs)["holds_earlier_edit"] is False


def test_two_saves_in_the_same_second_still_tell_apart(project):
    first = _edit(project)
    tools.set_edit(str(project.video), MORE_CUTS, **QUIET)
    second = _edit(project)
    assert edit_mark(first) != edit_mark(second)
    assert edit_mark({**first, "updated_at": "another time"}) == edit_mark(first)
    assert edit_mark({"cuts": []}) == edit_mark({"cuts": [], "keep": [{"start": 1.0, "end": 2.0}]})


def test_a_render_started_with_no_record_of_its_edit_is_never_called_behind(project, jobs):
    job_id = jobs.start(RENDER_KIND, project, lambda _h: {"output_path": "/x/out.mp4"})
    jobs.wait(job_id, timeout=5)
    tools.set_edit(str(project.video), MORE_CUTS, **QUIET)
    assert status(project, jobs)["holds_earlier_edit"] is False


def test_another_videos_export_is_not_this_ones(project, jobs, render, tmp_path):
    other = tmp_path / "other.mp4"
    other.write_bytes(project.video.read_bytes())
    start_full_render(project, jobs=jobs, render=render)
    assert export_status(open_project(str(other)), jobs, edit={"cuts": []})["state"] == "idle"


# ── Claude's side: render and get_edit ────────────────────────────────────────


def test_the_render_tool_starts_nothing_while_an_export_runs_and_says_so(project, jobs, render, monkeypatch):
    monkeypatch.setattr(render_jobs, "render_full", render)
    page_job, _ = start_full_render(project, jobs=jobs)
    answer = tools.render(str(project.video), jobs=jobs)
    assert answer["job_id"] == page_job and answer["already_running"] is True
    assert "already running" in answer["note"] and "job_status" in answer["note"]
    assert render.running.wait(5) and len(render.calls) == 1
    render.finish(jobs, page_job)
    again = tools.render(str(project.video), jobs=jobs)
    assert again["job_id"] != page_job and set(again) == {"job_id"}


def test_job_status_sees_an_export_the_page_started(project, jobs, render):
    job_id, _ = start_full_render(project, jobs=jobs, render=render)
    assert render.running.wait(5)
    assert tools.job_status(job_id, jobs=jobs)["status"] == "running"
    render.go.set()
    done = tools.job_status(job_id, wait=5, jobs=jobs)
    assert done["status"] == "done" and done["result"]["output_path"].endswith(EXPORTED_NAME)


def test_get_edit_names_the_latest_export_so_claude_can_follow_it(project, jobs, render):
    video = str(project.video)
    assert "export" not in tools.get_edit(video, jobs=jobs)
    job_id, _ = start_full_render(project, jobs=jobs, render=render)
    assert tools.get_edit(video, jobs=jobs)["export"] == {
        "job_id": job_id, "status": "running", "edit_changed_since": False,
    }
    render.finish(jobs, job_id)
    assert tools.get_edit(video, jobs=jobs)["export"]["status"] == "done"


# ── the cut: placing the picture's frames ─────────────────────────────────────

FPS = Fraction(25)
FRAME = 1 / 25


def many_pieces(count: int, every: float = 0.731, keeps: float = 0.517) -> list[tuple[float, float]]:
    """``count`` kept pieces whose edges fall between frames."""
    return [(round(0.013 + i * every, 3), round(0.013 + i * every + keeps, 3)) for i in range(count)]


def plan_of(kept, **fields) -> RenderPlan:
    base = dict(
        source=Path("/footage/talk.mp4"), out_path=Path("/exports/talk-edit.mp4"), kept=kept, removed=[],
        duration=1300.0, out_height=720, source_width=1280, source_height=720, source_note="test",
    )
    return RenderPlan(**{**base, **fields})


def piece_of(run: Shown, kept) -> int:
    """The kept piece that the run's first frame was recorded in."""
    return next(n for n, (start, end) in enumerate(kept) if start <= run.first * FRAME < end)


def test_the_picture_is_as_long_as_the_sound():
    kept = many_pieces(700)
    picture = plan_picture(kept, FPS)
    assert picture.frames * FRAME == pytest.approx(sum(e - s for s, e in kept), abs=FRAME / 2 + 1e-9)
    assert picture.shown[-1].lands_on + picture.shown[-1].count <= picture.frames


def test_only_frames_recorded_inside_a_kept_piece_are_shown():
    kept = many_pieces(700)
    for run in plan_picture(kept, FPS).shown:
        start, end = kept[piece_of(run, kept)]
        assert start <= run.first * FRAME and (run.first + run.count - 1) * FRAME < end


def test_every_frame_shows_within_half_a_frame_of_its_sound():
    kept = many_pieces(700)
    sound_before = [sum(e - s for s, e in kept[:n]) for n in range(len(kept))]
    for run in plan_picture(kept, FPS).shown:
        n = piece_of(run, kept)
        its_sound_plays_at = sound_before[n] + run.first * FRAME - kept[n][0]
        assert abs(run.lands_on * FRAME - its_sound_plays_at) <= FRAME / 2 + 1e-9


def test_no_place_is_filled_twice_and_a_held_frame_is_held_briefly():
    shown = plan_picture(many_pieces(700), FPS).shown
    assert shown[0].lands_on <= 1
    gaps = [after.lands_on - (run.lands_on + run.count) for run, after in zip(shown, shown[1:])]
    assert min(gaps) >= 0 and max(gaps) <= 1 and 0 < gaps.count(1) < len(gaps)


def test_when_two_frames_want_one_place_the_earlier_piece_keeps_it():
    # The second piece's last frame, 38, fills place 26. The third's first frame, 50, is due there too.
    picture = plan_picture([(0.0, 0.5), (1.0, 1.536), (2.0, 3.0)], FPS)
    assert picture.shown == [Shown(0, 13, 0), Shown(25, 14, 13), Shown(51, 24, 27)] and picture.frames == 51


def test_a_piece_with_no_frame_of_its_own_keeps_its_sound_and_shows_none():
    kept = [(0.0, 1.0), (1.405, 1.435), (3.0, 4.0)]
    picture = plan_picture(kept, FPS)
    assert picture.shown == [Shown(0, 25, 0), Shown(75, 25, 26)] and picture.frames == 51
    assert "asegment=timestamps=1.000000|1.405000|1.435000|3.000000|4.000000" in sound_chain(kept, Fraction(0))


def test_the_five_millisecond_piece_that_opens_the_real_take():
    picture = plan_picture([(0.0, 0.005), (2.657, 3.84)], FPS)
    assert picture.shown == [Shown(0, 1, 0), Shown(67, 29, 1)] and picture.frames == 30


def test_a_frame_rate_stays_exact():
    assert frame_rate(30000 / 1001) == Fraction(30000, 1001)
    assert frame_rate(25.0) == 25 and frame_rate(59.94005994) == Fraction(60000, 1001)
    assert frame_rate(None) == frame_rate(0.0) == frame_rate(90000.0) == frame_rate(float("nan")) == 25


def test_the_picture_follows_the_sound_at_thirty_frames_less_a_bit():
    fps = Fraction(30000, 1001)
    kept = many_pieces(300)
    picture = plan_picture(kept, fps)
    assert float(picture.frames / fps) == pytest.approx(sum(e - s for s, e in kept), abs=0.5 / 29.97)
    chain = picture_chain(picture, fps)
    assert chain.startswith("[0:v]fps=30000/1001,select=") and ",fps=30000/1001:start_time=0," in chain


# ── the cut: the graph and the command ────────────────────────────────────────


def test_the_picture_chain_keeps_the_runs_moves_them_and_holds_over_the_gaps():
    picture = Picture(shown=[Shown(10, 10, 1), Shown(30, 5, 12), Shown(50, 10, 17)], frames=29)
    assert picture_chain(picture, FPS) == (
        "[0:v]fps=25/1,"
        "select='between(pts,10,19)+between(pts,30,34)+between(pts,50,59)',"
        "setpts='PTS-(9+gte(PTS,30)*9+gte(PTS,50)*15)',"
        "fps=25/1:start_time=0,tpad=stop_mode=clone:stop=-1,trim=end_frame=29[outv]"
    )


def test_the_picture_chain_counts_from_where_the_read_starts_and_scales_last():
    chain = picture_chain(Picture(shown=[Shown(100, 25, 0)], frames=25), FPS, origin=75, scale="scale=-2:1440")
    assert "select='between(pts,25,49)',setpts='PTS-(25)'" in chain
    assert chain.endswith("trim=end_frame=25,scale=-2:1440[outv]")


def test_the_sound_is_split_once_and_the_removed_stretches_dropped():
    assert sound_chain([(1.0, 2.5), (4.0, 6.0)], Fraction(0)) == (
        "[0:a]asegment=timestamps=1.000000|2.500000|4.000000|6.000000[s0][s1][s2][s3][s4];"
        "[s0]anullsink;"
        "[s1]asetpts=PTS-STARTPTS,afade=t=out:st=1.480:d=0.020[a0];"
        "[s2]anullsink;"
        "[s3]asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.020[a1];"
        "[s4]anullsink;"
        "[a0][a1]concat=n=2:v=0:a=1[outa]"
    )


def test_a_take_kept_from_its_first_moment_has_no_stretch_before_it():
    chain = sound_chain([(0.0, 2.0), (3.0, 4.0)], Fraction(0))
    assert chain.startswith("[0:a]asegment=timestamps=2.000000|3.000000|4.000000[s0][s1][s2][s3];[s0]asetpts")


def test_the_joins_fade_by_clipforges_rule():
    kept = [(0.0, 1.0), (2.0, 2.03), (3.0, 3.5), (4.0, 4.015)]
    ours = sound_chain(kept, Fraction(0))
    theirs = build_filter_complex(kept)
    for n in range(len(kept)):
        our_fades = ours.split(f"[a{n}];")[0].rsplit("asetpts=PTS-STARTPTS", 1)[1]
        their_fades = theirs.split(f"[a{n}];")[0].rsplit("asetpts=PTS-STARTPTS", 1)[1]
        assert our_fades == their_fades
    assert f"d={AUDIO_JOIN_FADE:.3f}" in ours


def test_an_edit_that_opens_late_is_read_from_a_frame_a_second_before():
    assert read_from([(0.4, 5.0)], FPS) == 0 and read_from([(1.0, 5.0)], FPS) == 0
    origin = read_from([(612.377, 640.0)], FPS)
    assert origin == 15284 and 1.0 <= 612.377 - origin * FRAME < 1.0 + FRAME
    cut = plan_cut(plan_of([(612.377, 613.377), (620.0, 621.0)]), fps=25.0, has_sound=True)
    # The first frame recorded inside the first piece is 15310, which is 26 after the frame the read starts on.
    assert "select='between(pts,26,50)+between(pts,217,240)',setpts='PTS-(25+gte(PTS,217)*166)'" in cut.graph
    assert "trim=end_frame=50" in cut.graph
    assert "asegment=timestamps=1.017000|2.017000|8.640000|9.640000" in cut.graph
    cmd = cut_command(cut, ["-filter_complex", cut.graph])
    assert cmd[cmd.index("-ss"):cmd.index("-i") + 2] == ["-ss", "611.360000", "-t", "10.640000", "-i", "/footage/talk.mp4"]


def test_the_cut_encodes_as_clipforge_does():
    cut = plan_cut(plan_of(many_pieces(3), crf=19), fps=25.0, has_sound=True)
    cmd = cut_command(cut, ["-filter_complex", cut.graph])
    assert "-ss" not in cmd and cmd[-1] == "/exports/talk-edit.mp4"
    encode = cmd[cmd.index("-map"):]
    assert encode == [
        "-map", "[outv]", "-c:v", "libx264", "-preset", "medium", "-profile:v", "high", "-pix_fmt", "yuv420p",
        "-crf", "19", "-map", "[outa]", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
        "/exports/talk-edit.mp4",
    ]
    assert cut.seconds == pytest.approx(3 * 0.517, abs=FRAME / 2)


def test_a_take_with_no_sound_is_cut_with_no_sound():
    cut = plan_cut(plan_of(many_pieces(3)), fps=25.0, has_sound=False)
    assert "asegment" not in cut.graph and "[outa]" not in cut_command(cut, ["-filter_complex", cut.graph])


def test_a_source_taller_than_the_cap_is_scaled_down():
    tall = plan_cut(plan_of(many_pieces(3), source_height=2160), fps=25.0, has_sound=True)
    assert ",scale=-2:1440[outv]" in tall.graph


@pytest.mark.parametrize("kept", [[(3.0, 3.01)], [(3.005, 3.035)]])
def test_an_edit_that_keeps_less_than_a_frame_is_refused(kept):
    with pytest.raises(StudioError, match="less than one frame"):
        plan_cut(plan_of(kept), fps=25.0, has_sound=True)


# ── the cut: what this ffmpeg can do ──────────────────────────────────────────

FILTERS = """Filters:
 .. asegment          A->N       Segment audio stream.
 T. atrim             A->A       Pick one continuous section from the input.
"""


def asked(version_line: str, filters: str = FILTERS):
    calls = []

    def run(cmd, **_kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout=version_line if "-version" in cmd else filters)
    return run, calls


@pytest.fixture
def unasked(monkeypatch):
    monkeypatch.setattr(render_jobs, "_ffmpeg", [])


@pytest.mark.parametrize("line, version, flag", [
    ("ffmpeg version 8.0.1 Copyright (c) 2000-2025 the FFmpeg developers", 8, "-/filter_complex"),
    ("ffmpeg version n7.1-12-gabc Copyright", 7, "-/filter_complex"),
    ("ffmpeg version 6.1.1-3ubuntu5 Copyright", 6, "-filter_complex_script"),
    ("ffmpeg version N-118000-g0123abcd Copyright", None, "-/filter_complex"),
])
def test_the_ffmpeg_on_this_machine_is_asked_once(unasked, line, version, flag):
    run, calls = asked(line)
    found = this_ffmpeg(run)
    assert found == Ffmpeg(version=version, splits_sound=True) and found.graph_file_flag == flag
    assert this_ffmpeg(run) == found and len(calls) == 2


def test_an_ffmpeg_older_than_five_cannot_split_the_sound(unasked):
    run, _ = asked("ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright", " T. atrim             A->A       Pick\n")
    assert this_ffmpeg(run) == Ffmpeg(version=4, splits_sound=False)


def test_a_missing_ffmpeg_is_asked_again_next_time(unasked):
    def missing(*_a, **_k):
        raise FileNotFoundError("ffmpeg")
    assert this_ffmpeg(missing).splits_sound is False
    run, _ = asked("ffmpeg version 8.0.1 Copyright")
    assert this_ffmpeg(run).splits_sound is True


# ── the cut: running it ───────────────────────────────────────────────────────


class StandInFfmpeg:
    """Stands in for run_ffmpeg: keeps the command, reads the graph file, reports half way, writes nothing."""

    def __init__(self) -> None:
        self.cmd: list[str] = []
        self.graph_file: Path | None = None
        self.graph_read = ""
        self.said: dict = {}

    def __call__(self, cmd, out, *, total, on_progress, **said):
        self.cmd, self.said = cmd, {"total": total, **said}
        flag = next((a for a in cmd if a in ("-/filter_complex", "-filter_complex_script")), None)
        if flag:
            self.graph_file = Path(cmd[cmd.index(flag) + 1])
            self.graph_read = self.graph_file.read_text()
        on_progress(0.5)


@pytest.fixture
def stand_in(monkeypatch, unasked):
    ffmpeg = StandInFfmpeg()
    monkeypatch.setattr(render_jobs, "run_ffmpeg", ffmpeg)
    monkeypatch.setattr(render_jobs, "this_ffmpeg", lambda: Ffmpeg(version=8, splits_sound=True))
    monkeypatch.setattr(
        render_jobs, "probe_media", lambda _path: SimpleNamespace(fps=25.0, has_sound=True)
    )
    return ffmpeg


def test_the_cut_fills_its_share_of_the_progress(stand_in, tmp_path):
    handle = JobHandle()
    plan = plan_of(many_pieces(5), out_path=tmp_path / "exports" / "out.mp4")
    assert cut_edit(plan, handle, "label", share=0.6) is None
    assert handle.progress() == pytest.approx(0.3)
    assert stand_in.said["total"] == pytest.approx(5 * 0.517, abs=FRAME / 2)
    assert "-filter_complex" in stand_in.cmd and stand_in.graph_file is None
    assert stand_in.said["doing"] == "encode" and "source video" in stand_in.said["check"]


def test_a_long_graph_goes_to_ffmpeg_in_a_file_that_is_gone_afterwards(stand_in, tmp_path, monkeypatch):
    monkeypatch.setattr(render_jobs, "GRAPH_INLINE_LIMIT", 200)
    plan = plan_of(many_pieces(40), out_path=tmp_path / "exports" / "out.mp4")
    cut_edit(plan, JobHandle(), "label")
    assert "-filter_complex" not in stand_in.cmd
    assert stand_in.graph_read == plan_cut(plan, fps=25.0, has_sound=True).graph
    assert not stand_in.graph_file.exists() and not stand_in.graph_file.parent.exists()


def test_seven_hundred_pieces_fit_one_argument_or_go_to_a_file():
    graph = plan_cut(plan_of(many_pieces(700)), fps=25.0, has_sound=True).graph
    assert len(graph) > render_jobs.GRAPH_INLINE_LIMIT  # so the real take's Max stop runs from a file
    assert render_jobs.GRAPH_INLINE_LIMIT < 128 * 1024


def test_an_old_ffmpeg_cuts_the_slow_way_and_says_so(video, monkeypatch, unasked):
    monkeypatch.setattr(render_jobs, "this_ffmpeg", lambda: Ffmpeg(version=4, splits_sound=False))
    out = video.parent.parent / "exports" / "out.mp4"
    plan = plan_of([(4.0, 6.0), (8.0, 9.0)], source=video, out_path=out, duration=20.0, source_height=240)
    handle = JobHandle()
    assert cut_edit(plan, handle, "label") == OLD_FFMPEG_NOTE
    assert "asegment" in OLD_FFMPEG_NOTE and handle.progress() == 1.0
    seconds = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(out)],
        capture_output=True, text=True, check=True,
    ).stdout)
    assert seconds == pytest.approx(3.0, abs=0.2)


# ── two passes: each fills its share of the time ──────────────────────────────


def test_the_cut_takes_about_half_of_a_two_pass_render():
    whole_take = [(0.0, 1307.0)]
    assert first_pass_share(whole_take) == pytest.approx(0.52, abs=0.01)


def test_the_more_that_is_removed_the_larger_the_cuts_share():
    standard = [(0.0, 500.0), (600.0, 1307.0)]
    hard = [(0.0, 300.0), (900.0, 1307.0)]
    assert 0.5 < first_pass_share(standard) < first_pass_share(hard) < 0.7
