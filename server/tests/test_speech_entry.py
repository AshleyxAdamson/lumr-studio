"""``python -m lumr_studio.speech``: the entry the transcribe job runs. The model never loads here."""

import subprocess
import sys

import pytest
from tiny_models import tiny_model

from lumr_studio import models
from lumr_studio.speech import __main__ as entry
from lumr_studio.speech import transcribe
# The suite forbids ``transcription.run_speech_model``; keep the real one from before it did.
from lumr_studio.transcription import run_speech_model


@pytest.fixture
def speech_model(tmp_path, monkeypatch):
    """A speech model that is on this machine, small, in a temp folder."""
    tiny = tiny_model(tmp_path / "speech-snapshot", "Tiny speech model")
    tiny.folder.mkdir()
    for file in tiny.files:
        (tiny.folder / file.name).write_bytes(b"x" * file.size)
    monkeypatch.setattr(models, "SPEECH", tiny)
    return tiny


def test_the_entry_hands_the_video_and_the_parakeet_settings_to_the_engine(speech_model, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(entry, "transcribe_audio", lambda *a, **k: calls.append((a, k)))
    assert entry.main([str(tmp_path / "talk.mp4")]) == 0
    assert calls == [((tmp_path / "talk.mp4", entry.parakeet_settings()), {"word_timestamps": True})]
    assert entry.parakeet_settings()["transcription_model"] == str(speech_model.folder)


def test_the_model_it_hands_over_is_the_local_folder_the_download_fills():
    """No repo name goes to Hugging Face: the folder is pinned to one revision, and the library falls back to it."""
    folder = entry.parakeet_settings()["transcription_model"]
    assert folder.endswith(f"models--mlx-community--parakeet-tdt-0.6b-v2/snapshots/{models.SPEECH_REVISION}")


def test_without_the_model_the_entry_says_how_to_get_it_and_loads_nothing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(entry, "transcribe_audio", lambda *a, **k: pytest.fail("no model, nothing to run"))
    assert entry.main([str(tmp_path / "talk.mp4")]) == 3
    assert "say yes" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [[], ["a.mp4", "b.mp4"]])
def test_the_entry_refuses_anything_but_one_video(argv, capsys):
    assert entry.main(argv) == 2
    assert "usage" in capsys.readouterr().err


def test_the_fixed_settings_are_the_ones_the_copied_code_sends_to_parakeet(monkeypatch, tmp_path):
    """A sync that renames a setting in the source would break the entry silently. This catches it."""
    reached = []
    monkeypatch.setattr(transcribe, "_transcribe_parakeet_mlx", lambda *a: reached.append(a) or ("", "0:00"))
    transcribe.transcribe_audio(tmp_path / "talk.mp4", entry.parakeet_settings(), word_timestamps=True)
    assert reached == [(tmp_path / "talk.mp4", str(models.SPEECH.folder), True)]


def test_run_as_a_module_with_no_video_it_exits_2_and_says_usage():
    done = subprocess.run([sys.executable, "-m", "lumr_studio.speech"], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert done.returncode == 2 and "usage" in done.stderr


def test_the_speech_process_is_told_to_stay_offline(monkeypatch, tmp_path):
    """The real ``run_speech_model``: whatever is missing, Hugging Face's library never goes looking online."""
    seen = {}
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: seen.update(cmd=cmd, **kw))
    run_speech_model(tmp_path / "talk.mp4")
    assert seen["env"]["HF_HUB_OFFLINE"] == "1"
    assert seen["cmd"][1:3] == ["-m", "lumr_studio.speech"]
