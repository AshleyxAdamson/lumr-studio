"""``transcribe`` and the models: it asks before anything downloads, then fetches on a yes (fake network only).

The models are tiny stand-ins in temp folders (``tiny_models``), the transcriber
and the aligner are fakes, and the network is ``FakeNet``. No test loads a model
or reaches the internet.
"""

from __future__ import annotations

import json
import threading

import pytest
from conftest import FakeAligner
from tiny_models import FakeNet, content_of, tiny_set

from lumr_studio import models, tools, word_times
from lumr_studio.project import open_project
from lumr_studio.silences import no_silences
from lumr_studio.tools import unmeasured_labels

QUIET = {"silences": no_silences, "labels": unmeasured_labels}


@pytest.fixture
def both(tmp_path):
    return tiny_set(tmp_path / "models")


@pytest.fixture
def net(both):
    return FakeNet({**content_of(both.speech), **content_of(both.aligner)})


@pytest.fixture
def no_transcript(video):
    video.with_suffix(".words.json").unlink()
    return video


def fake_transcriber(words):
    def transcribe(path, force):
        path.with_suffix(".words.json").write_text(json.dumps(words))

    return transcribe


def ask(video, jobs, both, net, **more):
    return tools.transcribe(str(video), jobs=jobs, model_set=both, opener=net, aligner=FakeAligner(), **QUIET, **more)


def fill(both, net):
    """Put both tiny models on disk the way a consented download does."""
    models.download([both.speech, both.aligner], net)


# ── the question ──────────────────────────────────────────────────────────────


def test_with_no_transcript_and_no_models_it_names_both_and_downloads_nothing(no_transcript, jobs, both, net):
    answer = ask(no_transcript, jobs, both, net)
    assert answer["status"] == "needs_models"
    assert [m["name"] for m in answer["models"]] == ["Tiny speech model", "Tiny aligner"]
    assert [(m["source_host"], m["license"]) for m in answer["models"]] == [
        ("hub.example.test", "CC-BY-4.0"), ("files.example.test", "MIT"),
    ]
    assert all(set(m) == {"name", "size_mb", "source_host", "license", "folder"} for m in answer["models"])
    assert answer["total_mb"] == sum(m["size_mb"] for m in answer["models"])
    assert "ask" in answer["note"] and "download_models" in answer["note"] and "Nothing has downloaded" in answer["note"]
    assert "job_id" not in answer
    assert net.requests == [], "no byte moved"
    assert not both.speech.folder.exists() and not both.aligner.folder.exists(), "no folder made either"
    assert jobs.latest("download_models", open_project(str(no_transcript))) is None


def test_the_answer_a_stranger_gets_carries_the_real_sizes_hosts_and_licenses(no_transcript, jobs, net):
    """The real catalog, an empty cache, and a network that must not be touched."""
    def forbidden(url, start):
        raise AssertionError("asking must not download")

    real = models.ModelSet(models.SPEECH, models.ALIGNER)
    answer = tools.transcribe(str(no_transcript), jobs=jobs, model_set=real, opener=forbidden)
    speech, aligner = answer["models"]
    assert "Parakeet" in speech["name"] and (speech["size_mb"], speech["source_host"], speech["license"]) == (
        2472, "huggingface.co", "CC-BY-4.0")
    assert "wav2vec" in aligner["name"] and (aligner["size_mb"], aligner["source_host"], aligner["license"]) == (
        378, "download.pytorch.org", "MIT")
    assert answer["total_mb"] == 2850
    assert speech["folder"].endswith(f"snapshots/{models.SPEECH_REVISION}")
    assert aligner["folder"].endswith("hub/checkpoints")


def test_with_a_transcript_only_the_word_timing_model_is_asked_for_and_the_transcript_is_reported(video, jobs, both, net):
    answer = ask(video, jobs, both, net)
    assert answer["status"] == "needs_models"
    assert [m["name"] for m in answer["models"]] == ["Tiny aligner"], "the speech model isn't needed for a transcript that exists"
    assert answer["word_count"] == 25 and answer["word_times"] == "estimated"
    assert "estimated" in answer["word_times_note"] and "If they say no" in answer["note"]
    assert net.requests == []


def test_a_model_already_there_is_not_asked_for_again(no_transcript, jobs, both, net):
    models.download([both.speech], net)
    answer = ask(no_transcript, jobs, both, net)
    assert [m["name"] for m in answer["models"]] == ["Tiny aligner"]


def test_a_video_with_measured_times_needs_no_model(video, jobs, both, net):
    word_times.align_project(open_project(str(video)), aligner=FakeAligner(), silences=no_silences)
    answer = ask(video, jobs, both, net)
    assert answer["status"] == "exists" and answer["word_times"] == "measured"


def test_transcribing_again_with_force_needs_the_speech_model_too(video, jobs, both, net):
    answer = ask(video, jobs, both, net, force=True)
    assert [m["name"] for m in answer["models"]] == ["Tiny speech model", "Tiny aligner"]


# ── the yes ───────────────────────────────────────────────────────────────────


def test_after_a_yes_the_job_downloads_shows_progress_and_the_next_call_goes_on(no_transcript, jobs, both, net, words):
    started = ask(no_transcript, jobs, both, net, download_models=True)
    assert started["status"] == "downloading" and started["job_id"].startswith("download_models-")
    assert [m["name"] for m in started["models"]] == ["Tiny speech model", "Tiny aligner"]
    done = jobs.wait(started["job_id"], timeout=10)
    assert done["status"] == "done", done
    assert done["progress"] == 1.0 and done["kind"] == "download_models"
    assert done["result"]["models_downloaded"] == ["Tiny speech model", "Tiny aligner"]
    assert "transcribe again" in done["result"]["next"]
    assert both.speech.present() and both.aligner.present()
    # The same call now goes on: transcribes and measures, in one job, with nothing left to ask.
    again = tools.transcribe(
        str(no_transcript), jobs=jobs, model_set=both, opener=net, transcriber=fake_transcriber(words),
        aligner=FakeAligner(), **QUIET,
    )
    assert set(again) == {"job_id"}
    result = jobs.wait(again["job_id"], timeout=10)["result"]
    assert result["word_count"] == 25 and result["word_times"] == "measured"


def test_progress_reads_between_zero_and_one_while_the_files_arrive(no_transcript, jobs, both, net):
    started_reading, release = threading.Event(), threading.Event()

    class Held(FakeNet):
        def _chunks(self, data, size=1000):
            for i, chunk in enumerate(super()._chunks(data, size)):
                if i == 3:
                    started_reading.set()
                    release.wait(5)
                yield chunk

    held = Held({**content_of(both.speech), **content_of(both.aligner)})
    started = ask(no_transcript, jobs, both, held, download_models=True)
    assert started_reading.wait(5)
    running = tools.job_status(started["job_id"], jobs=jobs)
    assert running["status"] == "running" and 0.0 < running["progress"] < 1.0
    release.set()
    assert jobs.wait(started["job_id"], timeout=10)["progress"] == 1.0


def test_asking_again_for_a_download_that_is_running_joins_it(no_transcript, jobs, both):
    started_reading, release = threading.Event(), threading.Event()

    class Held(FakeNet):
        def _chunks(self, data, size=1000):
            started_reading.set()
            release.wait(5)
            yield from super()._chunks(data, size)

    held = Held({**content_of(both.speech), **content_of(both.aligner)})
    first = ask(no_transcript, jobs, both, held, download_models=True)
    assert started_reading.wait(5)
    second = ask(no_transcript, jobs, both, held, download_models=True)
    release.set()
    assert second["status"] == "downloading" and second["job_id"] == first["job_id"]
    assert jobs.wait(first["job_id"], timeout=10)["status"] == "done"


def test_a_yes_when_nothing_is_missing_downloads_nothing_and_goes_on(no_transcript, jobs, both, net, words):
    fill(both, net)
    before = list(net.requests)
    answer = tools.transcribe(
        str(no_transcript), jobs=jobs, model_set=both, opener=net, download_models=True,
        transcriber=fake_transcriber(words), aligner=FakeAligner(), **QUIET,
    )
    assert set(answer) == {"job_id"} and net.requests == before
    assert jobs.wait(answer["job_id"], timeout=10)["result"]["word_times"] == "measured"


def test_a_yes_for_the_word_timing_model_alone_fetches_only_that(video, jobs, both, net):
    started = ask(video, jobs, both, net, download_models=True)
    assert jobs.wait(started["job_id"], timeout=10)["status"] == "done"
    assert net.urls == [f.url for f in both.aligner.files]
    assert not both.speech.folder.exists()
    again = ask(video, jobs, both, net)
    assert again["status"] == "aligning", "the transcript is there, so the job measures it now"
    assert jobs.wait(again["job_id"], timeout=10)["result"]["word_times"] == "measured"


# ── when it goes wrong ────────────────────────────────────────────────────────


def test_a_bad_checksum_fails_the_job_with_the_reason_and_installs_nothing(no_transcript, jobs, both, net):
    url = both.aligner.files[0].url
    corrupt = FakeNet({**net.served, url: bytes(reversed(net.served[url]))})
    started = ask(no_transcript, jobs, both, corrupt, download_models=True)
    done = jobs.wait(started["job_id"], timeout=10)
    assert done["status"] == "failed"
    assert "didn't match its checksum" in done["error"] and "Tiny aligner" in done["error"]
    assert not both.aligner.present()
    # Tried again on a good network, it lands.
    again = ask(no_transcript, jobs, both, net, download_models=True)
    assert jobs.wait(again["job_id"], timeout=10)["status"] == "done" and both.aligner.present()


def test_a_download_that_stops_fails_the_job_and_the_next_yes_resumes_it(no_transcript, jobs, both, net):
    cut = FakeNet(net.served, cut_after=3000)
    started = ask(no_transcript, jobs, both, cut, download_models=True)
    done = jobs.wait(started["job_id"], timeout=10)
    assert done["status"] == "failed" and "picks up where it left off" in done["error"]
    resumed = FakeNet(net.served)
    again = ask(no_transcript, jobs, both, resumed, download_models=True)
    assert jobs.wait(again["job_id"], timeout=10)["status"] == "done"
    assert (resumed.requests[0][1], both.speech.present(), both.aligner.present()) == (3000, True, True)


def test_the_download_is_written_to_the_receipts_of_the_video(no_transcript, jobs, both, net):
    started = ask(no_transcript, jobs, both, net, download_models=True)
    jobs.wait(started["job_id"], timeout=10)
    receipts = open_project(str(no_transcript)).root / "receipts.jsonl"
    lines = [json.loads(line) for line in receipts.read_text().splitlines()]
    assert [(line["step"], line["status"]) for line in lines] == [("download_models", "done")]
