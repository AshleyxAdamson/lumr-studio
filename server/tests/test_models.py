"""The models the plugin fetches: what is pinned, where the files go, and the download (fake network only)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import torchaudio
from tiny_models import WEIGHTS, FakeNet, content_of, tiny_model, tiny_set

from lumr_studio import models
from lumr_studio.errors import StudioError
from lumr_studio.models import PART_SUFFIX, download

# ── what is pinned ────────────────────────────────────────────────────────────


def test_the_speech_model_is_pinned_to_one_full_commit_and_names_its_license_and_host():
    assert len(models.SPEECH_REVISION) == 40 and set(models.SPEECH_REVISION) <= set("0123456789abcdef")
    assert models.SPEECH_REVISION.startswith("8ae15530")
    assert models.SPEECH.license == "CC-BY-4.0" and models.SPEECH.source_host == "huggingface.co"
    for file in models.SPEECH.files:
        assert file.url == f"https://huggingface.co/{models.SPEECH_REPO}/resolve/{models.SPEECH_REVISION}/{file.name}"
        assert len(file.sha256) == 64


def test_the_aligner_file_is_pinned_and_torchaudio_looks_for_it_by_that_name():
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    assert models.ALIGNER_FILE.name == bundle._path
    assert models.ALIGNER_FILE.url == f"https://download.pytorch.org/torchaudio/models/{bundle._path}"
    assert models.ALIGNER_FILE.size == 377_664_473 and len(models.ALIGNER_FILE.sha256) == 64


def test_what_the_creator_is_told_about_each_model(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "ALIGNER", dataclasses.replace(models.ALIGNER, where=lambda: tmp_path))
    speech, aligner = models.SPEECH.describe(), models.ALIGNER.describe()
    assert set(speech) == {"name", "size_mb", "source_host", "license", "folder"}
    assert (speech["size_mb"], speech["source_host"], speech["license"]) == (2472, "huggingface.co", "CC-BY-4.0")
    assert (aligner["size_mb"], aligner["source_host"], aligner["license"]) == (378, "download.pytorch.org", "MIT")
    assert aligner["folder"] == str(tmp_path)
    assert "Parakeet" in speech["name"] and "wav2vec" in aligner["name"]


def test_the_folders_follow_the_cache_environment_the_way_the_libraries_do(tmp_path):
    """The one place the plugin works out where a model lives, held to what huggingface_hub and torch say."""
    env = {k: v for k, v in os.environ.items() if k not in {"HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "XDG_CACHE_HOME"}}
    env |= {"HF_HOME": str(tmp_path / "hf"), "TORCH_HOME": str(tmp_path / "torch"), "HF_HUB_OFFLINE": "1"}
    script = (
        "import json, torch.hub\n"
        "from huggingface_hub import constants\n"
        "from lumr_studio import models\n"
        "print(json.dumps([constants.HF_HUB_CACHE, str(models.hf_hub_cache()), torch.hub.get_dir(),"
        " str(models.torch_checkpoints()), str(models.SPEECH.folder), str(models.ALIGNER.folder)]))\n"
    )
    done = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True)
    library_hf, ours_hf, library_torch, ours_torch, speech, aligner = json.loads(done.stdout.splitlines()[-1])
    assert ours_hf == library_hf == str(tmp_path / "hf" / "hub")
    assert ours_torch == str(Path(library_torch) / "checkpoints") == str(tmp_path / "torch" / "hub" / "checkpoints")
    assert speech == f"{ours_hf}/models--mlx-community--parakeet-tdt-0.6b-v2/snapshots/{models.SPEECH_REVISION}"
    assert aligner == ours_torch


@pytest.mark.parametrize("var, expected", [("HF_HUB_CACHE", "hub-cache"), ("HUGGINGFACE_HUB_CACHE", "old-cache")])
def test_the_hub_cache_can_be_moved_by_either_variable(monkeypatch, tmp_path, var, expected):
    monkeypatch.setenv(var, str(tmp_path / expected))
    assert models.hf_hub_cache() == tmp_path / expected


def test_the_hub_cache_defaults_under_the_home_folder(monkeypatch):
    monkeypatch.delenv("HF_HOME")
    assert models.hf_hub_cache() == Path.home() / ".cache" / "huggingface" / "hub"


# ── is it there ───────────────────────────────────────────────────────────────


def test_a_model_is_present_when_every_file_is_there_at_its_exact_size(tmp_path):
    model = tiny_set(tmp_path).speech
    assert not model.present()
    model.folder.mkdir(parents=True)
    (model.folder / "config.json").write_bytes(content_of(model)[model.files[0].url])
    assert not model.present(), "half of it is not enough"
    (model.folder / "speech.bin").write_bytes(b"x" * (model.files[1].size - 1))
    assert not model.present(), "a file cut short is not there"
    (model.folder / "speech.bin").write_bytes(WEIGHTS)
    assert model.present()


def test_a_file_that_is_a_link_into_the_hub_cache_counts_at_the_size_of_what_it_points_to(tmp_path):
    """The layout Hugging Face's own tools leave: the snapshot holds links into blobs."""
    model = tiny_model(tmp_path / "snapshot")
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "abc").write_bytes(WEIGHTS)
    model.folder.mkdir()
    (model.folder / "weights.bin").symlink_to(blobs / "abc")
    assert model.present()


# ── downloading ───────────────────────────────────────────────────────────────


def test_a_download_writes_every_file_checked_and_renamed_and_leaves_nothing_half_done(tmp_path):
    model = tiny_set(tmp_path).speech
    net = FakeNet(content_of(model))
    fetched = download([model], net)
    assert fetched == [model] and model.present()
    assert (model.folder / "speech.bin").read_bytes() == WEIGHTS
    assert sorted(p.name for p in model.folder.iterdir()) == ["config.json", "speech.bin"]
    assert net.urls == [f.url for f in model.files]


def test_progress_is_the_share_of_all_the_bytes_on_disk_and_ends_at_one(tmp_path):
    both = tiny_set(tmp_path)
    net = FakeNet({**content_of(both.speech), **content_of(both.aligner)})
    seen: list[float] = []
    download([both.speech, both.aligner], net, seen.append)
    assert seen == sorted(seen), "it never goes backwards"
    assert seen[0] == 0.0 and seen[-1] == 1.0 and 0.0 < seen[1] < 0.5


def test_what_is_already_there_is_not_fetched_again(tmp_path):
    model = tiny_set(tmp_path).aligner
    download([model], FakeNet(content_of(model)))
    net = FakeNet(content_of(model))
    assert download([model], net) == [] and net.requests == []


def test_a_model_half_there_fetches_only_the_missing_file_and_counts_the_other_as_done(tmp_path):
    model = tiny_set(tmp_path).speech
    model.folder.mkdir(parents=True)
    (model.folder / "config.json").write_bytes(content_of(model)[model.files[0].url])
    net, seen = FakeNet(content_of(model)), []
    download([model], net, seen.append)
    assert net.urls == [model.files[1].url] and model.present()
    assert seen[0] >= model.files[0].size / model.size_bytes, "the file already there counts from the start"


def test_a_file_that_fails_its_checksum_is_refused_and_nothing_is_installed(tmp_path):
    model = tiny_set(tmp_path).aligner
    url = model.files[0].url
    corrupt = bytearray(content_of(model)[url])
    corrupt[100] ^= 0xFF
    with pytest.raises(StudioError, match="didn't match its checksum.*not installed"):
        download([model], FakeNet({url: bytes(corrupt)}))
    assert not model.present() and not (model.folder / "aligner.pth").exists()
    assert not list(model.folder.glob(f"*{PART_SUFFIX}")), "the bad partial file is deleted"


def test_a_file_of_the_wrong_length_is_refused_too(tmp_path):
    model = tiny_set(tmp_path).aligner
    url = model.files[0].url
    with pytest.raises(StudioError, match="checksum"):
        download([model], FakeNet({url: content_of(model)[url] + b"extra"}))
    assert not model.present() and not list(model.folder.glob("*"))


def test_a_refused_download_can_be_tried_again_and_then_lands(tmp_path):
    model = tiny_set(tmp_path).aligner
    url = model.files[0].url
    with pytest.raises(StudioError):
        download([model], FakeNet({url: b"garbage" * 1500}))
    download([model], FakeNet(content_of(model)))
    assert model.present()


def test_a_stopped_download_keeps_what_arrived_and_the_next_call_picks_up_there(tmp_path):
    model = tiny_set(tmp_path).aligner
    target = model.folder / "aligner.pth"
    with pytest.raises(StudioError, match="stopped.*picks up where it left off"):
        download([model], FakeNet(content_of(model), cut_after=4000))
    part = target.with_name(target.name + PART_SUFFIX)
    assert part.stat().st_size == 4000 and not target.exists() and not model.present()
    net = FakeNet(content_of(model))
    download([model], net)
    assert net.requests == [(model.files[0].url, 4000)], "it asked for the rest, not the whole file"
    assert target.read_bytes() == WEIGHTS[::-1]
    assert not part.exists()


def test_a_stopped_download_is_resumed_through_the_progress_too(tmp_path):
    model = tiny_set(tmp_path).aligner
    with pytest.raises(StudioError):
        download([model], FakeNet(content_of(model), cut_after=4000))
    seen: list[float] = []
    download([model], FakeNet(content_of(model)), seen.append)
    assert seen[0] == pytest.approx(4000 / model.size_bytes), "it starts from what was kept"
    assert seen == sorted(seen) and seen[-1] == 1.0


def test_a_server_that_ignores_the_range_makes_it_start_over(tmp_path):
    model = tiny_set(tmp_path).aligner
    with pytest.raises(StudioError):
        download([model], FakeNet(content_of(model), cut_after=4000))
    net, seen = FakeNet(content_of(model), honor_range=False), []
    download([model], net, seen.append)
    assert net.requests == [(model.files[0].url, 4000)]
    assert (model.folder / "aligner.pth").read_bytes() == WEIGHTS[::-1], "no bytes doubled"
    assert min(seen) == 0.0 and seen[-1] == 1.0, "the share drops back when the file starts again, then completes"


def test_a_partial_file_longer_than_the_file_is_thrown_away(tmp_path):
    model = tiny_set(tmp_path).aligner
    model.folder.mkdir(parents=True)
    (model.folder / f"aligner.pth{PART_SUFFIX}").write_bytes(b"z" * (model.size_bytes + 50))
    net = FakeNet(content_of(model))
    download([model], net)
    assert net.requests == [(model.files[0].url, 0)] and model.present()


def test_a_whole_partial_file_that_checks_out_is_installed_without_asking_the_network(tmp_path):
    """A run that died between the last byte and the rename."""
    model = tiny_set(tmp_path).aligner
    model.folder.mkdir(parents=True)
    (model.folder / f"aligner.pth{PART_SUFFIX}").write_bytes(WEIGHTS[::-1])
    net = FakeNet({})
    download([model], net)
    assert net.requests == [] and model.present()


def test_a_whole_partial_file_that_fails_its_check_is_fetched_again_from_the_start(tmp_path):
    model = tiny_set(tmp_path).aligner
    model.folder.mkdir(parents=True)
    (model.folder / f"aligner.pth{PART_SUFFIX}").write_bytes(b"y" * model.size_bytes)
    net = FakeNet(content_of(model))
    download([model], net)
    assert net.requests == [(model.files[0].url, 0)] and model.present()


def test_a_connection_that_closes_early_without_an_error_is_a_stopped_download_not_a_bad_file(tmp_path):
    """urllib can end a body short with no error: the length check tells that from a corrupt file."""
    model = tiny_set(tmp_path).aligner
    url = model.files[0].url
    with pytest.raises(StudioError, match="stopped"):
        download([model], FakeNet({url: content_of(model)[url][:5000]}))
    assert (model.folder / f"aligner.pth{PART_SUFFIX}").stat().st_size == 5000


def test_a_server_that_answers_an_error_status_is_a_stopped_download(tmp_path):
    model = tiny_set(tmp_path).aligner
    with pytest.raises(StudioError, match="the server answered 503"):
        download([model], lambda url, start: (503, iter(())))
    assert not model.present()


def test_a_download_without_the_disk_room_says_so_and_fetches_nothing(tmp_path, monkeypatch):
    model = tiny_set(tmp_path).aligner
    monkeypatch.setattr(models.shutil, "disk_usage", lambda _p: type("Usage", (), {"free": 100})())
    net = FakeNet(content_of(model))
    with pytest.raises(StudioError, match="free disk"):
        download([model], net)
    assert net.requests == []


def test_the_room_check_counts_what_a_partial_file_already_holds(tmp_path, monkeypatch):
    model = tiny_set(tmp_path).aligner
    model.folder.mkdir(parents=True)
    (model.folder / f"aligner.pth{PART_SUFFIX}").write_bytes(WEIGHTS[::-1][:8000])
    left = model.size_bytes - 8000
    monkeypatch.setattr(models.shutil, "disk_usage", lambda _p: type("Usage", (), {"free": left})())
    download([model], FakeNet(content_of(model)))
    assert model.present()


def test_the_downloads_take_turns(tmp_path):
    """Two jobs writing the same partial file would corrupt it: the second waits, then finds it done."""
    model = tiny_set(tmp_path).aligner
    started, gate = threading.Event(), threading.Event()

    class Slow(FakeNet):
        def __call__(self, url, start):
            started.set()
            gate.wait(5)
            return super().__call__(url, start)

    first, second = Slow(content_of(model)), FakeNet(content_of(model))
    runs = [threading.Thread(target=download, args=([model], net)) for net in (first, second)]
    runs[0].start()
    assert started.wait(5)
    runs[1].start()
    gate.set()
    for run in runs:
        run.join(10)
    assert model.present() and first.requests and second.requests == [], "the second found it whole"


def test_the_production_opener_asks_for_a_range_and_says_who_it_is(monkeypatch):
    """No network: the request urllib would send is looked at, and the reply is a fake."""
    sent = []

    class Reply:
        status = 206

        def read(self, _n):
            return b""

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def urlopen(request, timeout):
        sent.append((request.full_url, dict(request.header_items()), timeout))
        return Reply()

    monkeypatch.setattr(models.urllib.request, "urlopen", urlopen)
    status, chunks = models.open_url("https://example.test/file", 1234)
    assert status == 206 and list(chunks) == []
    url, headers, timeout = sent[0]
    assert url == "https://example.test/file" and headers["Range"] == "bytes=1234-"
    assert headers["User-agent"] == models.USER_AGENT and timeout == models.NETWORK_TIMEOUT_SECONDS
    models.open_url("https://example.test/file", 0)
    assert "Range" not in sent[1][1]


def test_sha256_of_reads_the_file_in_pieces(tmp_path):
    path = tmp_path / "big.bin"
    data = os.urandom(3 * models.CHUNK_BYTES // 2)
    path.write_bytes(data)
    assert models.sha256_of(path) == hashlib.sha256(data).hexdigest()
