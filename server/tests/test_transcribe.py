import json
import subprocess
import sys

import pytest
from conftest import spread_evenly

from lumr_studio import tools, transcription
from lumr_studio.errors import StudioError
from lumr_studio.silences import no_silences
from lumr_studio.transcription import run_speech_model, speech_transcriber

# The fixture forbids ``transcription.run_speech_model``; keep the real one from before it did.
real_run_speech_model = run_speech_model


def test_existing_transcript_is_reported_not_redone(video, jobs):
    result = tools.transcribe(str(video), jobs=jobs)
    assert result["status"] == "exists"
    assert result["word_count"] == 25
    assert result["words_path"].endswith("talk.words.json")


def test_transcription_job_validates_what_the_speech_model_wrote(video, jobs, words):
    sidecar = video.with_suffix(".words.json")
    sidecar.unlink()

    def fake_transcriber(path, force):
        path.with_suffix(".words.json").write_text(json.dumps(words))

    job = tools.transcribe(str(video), transcriber=fake_transcriber, jobs=jobs)
    status = jobs.wait(job["job_id"], timeout=10)
    assert status["status"] == "done", status
    assert status["result"]["word_count"] == 25


def test_transcription_job_that_writes_nothing_fails(video, jobs):
    video.with_suffix(".words.json").unlink()
    job = tools.transcribe(str(video), transcriber=lambda _p, _f: None, jobs=jobs)
    status = jobs.wait(job["job_id"], timeout=10)
    assert status["status"] == "failed"
    assert "no transcript appeared" in status["error"]


def test_relative_and_missing_paths(tmp_path):
    with pytest.raises(StudioError, match="relative"):
        tools.get_edit("clip.mp4")
    with pytest.raises(StudioError, match="No file at"):
        tools.get_edit(str(tmp_path / "missing.mp4"))


def _write_approximate_sidecar(video):
    words = spread_evenly([
        ("Hello everyone and welcome to the show.", 0.5, 3.0),
        ("Today we talk about editing video.", 2.8, 6.0),
        ("This is the part that matters most.", 6.0, 9.5),
    ])
    video.with_suffix(".words.json").write_text(json.dumps(words))
    return words


def test_exists_response_reports_measured_timing(video, jobs):
    result = tools.transcribe(str(video), jobs=jobs)
    assert result["timing_quality"]["quality"] == "measured"
    assert "warning" not in result


def test_approximate_timings_warn_everywhere_but_do_not_block(video, jobs):
    _write_approximate_sidecar(video)
    exists = tools.transcribe(str(video), jobs=jobs)
    assert exists["timing_quality"]["quality"] == "approximate"
    assert "force=true" in exists["warning"]

    report = tools.analyze_take(str(video), silences=no_silences)
    assert report["warning"] == exists["warning"]

    edit = tools.set_edit(str(video), [{"start": 6.0, "end": 9.6, "reason": "tangent"}], silences=no_silences)
    assert edit["warning"] == exists["warning"]
    assert edit["applied"]  # still saved


def test_finished_job_result_carries_the_warning(video, jobs):
    words = _write_approximate_sidecar(video)
    video.with_suffix(".words.json").unlink()

    def fake_transcriber(path, force):
        path.with_suffix(".words.json").write_text(json.dumps(words))

    status = jobs.wait(tools.transcribe(str(video), transcriber=fake_transcriber, jobs=jobs)["job_id"], timeout=10)
    assert status["result"]["timing_quality"]["quality"] == "approximate"
    assert "warning" in status["result"]


def done(returncode: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def test_the_speech_model_runs_in_a_process_of_its_own_that_cannot_read_the_servers_input(monkeypatch, video):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.update(cmd=cmd, kwargs=kwargs)
        return done(0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    real_run_speech_model(video)
    assert seen["cmd"] == [sys.executable, "-m", "lumr_studio.speech", str(video)]
    assert seen["kwargs"]["stdin"] is subprocess.DEVNULL  # the server's own stdin is the protocol stream


def test_the_transcriber_runs_the_speech_model_and_stops_when_it_succeeds(monkeypatch, video):
    video.with_suffix(".words.json").unlink()
    ran = []
    monkeypatch.setattr(transcription, "run_speech_model", lambda v: ran.append(v) or done(0))
    speech_transcriber(video, False)
    assert ran == [video]


def test_a_failed_speech_model_says_what_it_said(monkeypatch, video):
    video.with_suffix(".words.json").unlink()
    monkeypatch.setattr(transcription, "run_speech_model", lambda v: done(1, stderr="no such model\n" * 100))
    with pytest.raises(StudioError, match="(?s)could not transcribe talk.mp4: .*no such model") as raised:
        speech_transcriber(video, False)
    assert len(str(raised.value)) < 900  # only the tail of what it said


def test_a_transcript_that_is_there_is_left_alone_unless_forced(monkeypatch, video):
    ran = []
    monkeypatch.setattr(transcription, "run_speech_model", lambda v: ran.append(v) or done(0))
    speech_transcriber(video, False)
    assert ran == []
    speech_transcriber(video, True)
    assert ran == [video]
