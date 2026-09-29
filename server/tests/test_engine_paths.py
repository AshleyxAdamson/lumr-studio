"""``engine/paths.py``: the Lumr home and the silence cache, resolved the way the Mac app does.

``paths.py`` is hand-written, so ``sync-engine.sh --check`` can't see it drift from the
ClipForge code it stands in for. The last tests here import ClipForge and compare. They
skip when the repo's ``clipforge/`` isn't beside the plugin, as in the public tree.
"""

import importlib
import json
import sys
from pathlib import Path

import pytest

from lumr_studio.engine import paths
from lumr_studio.engine.audio_boundaries import Silence

# The fixture forbids ``paths.silences_for``; keep the real one from before it did.
silences_for = paths.silences_for

CLIPFORGE_SOURCE = Path(__file__).resolve().parents[5] / "clipforge"
needs_clipforge = pytest.mark.skipif(
    not (CLIPFORGE_SOURCE / "clipforge" / "config.py").is_file(),
    reason="no clipforge beside this plugin",
)


def _forget_clipforge() -> None:
    for name in [n for n in sys.modules if n == "clipforge" or n.startswith("clipforge.")]:
        del sys.modules[name]


@pytest.fixture
def clipforge(monkeypatch, tmp_path):
    """ClipForge's ``config``, ``pipeline`` and ``audio_boundaries``, imported fresh.

    ClipForge fixes its home and cache folder when ``config`` loads, so it loads here
    after ``Path.home`` and ``LUMR_HOME`` point at temp folders, and is forgotten after.
    The folder is there, so an import that fails is a real failure, not a skip.
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.syspath_prepend(str(CLIPFORGE_SOURCE))
    _forget_clipforge()
    loaded = [importlib.import_module(f"clipforge.{name}") for name in ("config", "pipeline", "audio_boundaries")]
    yield loaded
    _forget_clipforge()


@pytest.fixture
def no_ffmpeg(monkeypatch):
    calls = []

    def detect(video, noise_db, min_silence):
        calls.append((video.name, noise_db, min_silence))
        return [Silence(1.0, 1.5)]

    monkeypatch.setattr(paths, "detect_silences", detect)
    return calls


def a_video(tmp_path: Path, size: int = 10) -> Path:
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"x" * size)
    return video


def test_the_home_is_the_environment_first(monkeypatch, tmp_path):
    monkeypatch.setenv("LUMR_HOME", str(tmp_path / "chosen"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    saved = tmp_path / ".config" / "lumr" / "home.txt"
    saved.parent.mkdir(parents=True)
    saved.write_text(str(tmp_path / "saved"))
    assert paths.lumr_home() == tmp_path / "chosen"


def test_without_the_environment_the_home_is_the_one_the_app_saved(monkeypatch, tmp_path):
    monkeypatch.delenv("LUMR_HOME")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    saved = tmp_path / ".config" / "lumr" / "home.txt"
    saved.parent.mkdir(parents=True)
    saved.write_text(f"{tmp_path / 'saved'}\n")
    assert paths.lumr_home() == tmp_path / "saved"


@pytest.mark.parametrize("saved_text", [None, "", "  \n"])
def test_with_nothing_chosen_the_home_is_lumr_in_the_home_folder(monkeypatch, tmp_path, saved_text):
    monkeypatch.delenv("LUMR_HOME")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    if saved_text is not None:
        saved = tmp_path / ".config" / "lumr" / "home.txt"
        saved.parent.mkdir(parents=True)
        saved.write_text(saved_text)
    assert paths.lumr_home() == tmp_path / "Lumr"


def test_silences_are_measured_once_then_read_from_the_cache(tmp_path, no_ffmpeg):
    video = a_video(tmp_path)
    first = silences_for(video)
    again = silences_for(video)
    assert first == again == [Silence(1.0, 1.5)]
    assert no_ffmpeg == [("talk.mp4", -25.0, 0.05)]  # the engine's defaults, asked once


def test_the_cache_file_is_named_the_way_the_mac_app_names_it(tmp_path, no_ffmpeg):
    silences_for(a_video(tmp_path, size=10))
    cached = sorted((paths.lumr_home() / "media_cache" / "pipeline_state").iterdir())
    # sha256("talk.mp4:10:-25.0:0.05")[:16]: pinned so the app and the plugin keep sharing measurements.
    assert [p.name for p in cached] == ["talk_1d3b7e59ca6f154c_silences.json"]
    assert json.loads(cached[0].read_text()) == [[1.0, 1.5]]


def test_a_measurement_the_app_already_made_is_read_not_repeated(tmp_path, no_ffmpeg):
    folder = paths.lumr_home() / "media_cache" / "pipeline_state"
    folder.mkdir(parents=True)
    (folder / "talk_1d3b7e59ca6f154c_silences.json").write_text(json.dumps([[2.0, 2.4]]))
    assert silences_for(a_video(tmp_path, size=10)) == [Silence(2.0, 2.4)]
    assert no_ffmpeg == []


def test_a_different_setting_or_a_different_file_misses_the_cache(tmp_path, no_ffmpeg):
    video = a_video(tmp_path)
    silences_for(video)
    silences_for(video, noise_db=-40.0)
    silences_for(video, min_silence=0.2)
    video.write_bytes(b"x" * 11)  # the same name, replaced by a longer file
    silences_for(video)
    assert no_ffmpeg == [("talk.mp4", -25.0, 0.05), ("talk.mp4", -40.0, 0.05), ("talk.mp4", -25.0, 0.2), ("talk.mp4", -25.0, 0.05)]
    assert len(list((paths.lumr_home() / "media_cache" / "pipeline_state").iterdir())) == 4


def test_the_real_silence_detection_reads_the_synthetic_video(video):
    """A tone with no pauses has no silences: the engine's ffmpeg call works end to end, in this env."""
    assert silences_for(video) == []


@needs_clipforge
@pytest.mark.parametrize("chosen", ["environment", "saved", "blank", "nothing"])
def test_the_home_is_the_one_clipforge_resolves(clipforge, monkeypatch, tmp_path, chosen):
    config = clipforge[0]
    if chosen == "environment":
        monkeypatch.setenv("LUMR_HOME", str(tmp_path / "chosen"))
    else:
        monkeypatch.delenv("LUMR_HOME")
    if chosen in ("saved", "blank"):
        saved = tmp_path / "home" / ".config" / "lumr" / "home.txt"
        saved.parent.mkdir(parents=True)
        saved.write_text(f"{tmp_path / 'saved'}\n" if chosen == "saved" else " \n")
    assert paths.lumr_home() == config._resolve_lumr_home()


@needs_clipforge
@pytest.mark.parametrize("settings", [{}, {"noise_db": -40.0, "min_silence": 0.2}], ids=["defaults", "tuned"])
def test_the_silence_cache_is_shared_with_clipforge_both_ways(clipforge, monkeypatch, tmp_path, no_ffmpeg, settings):
    """One file each way: what ClipForge measured is read here, and what this measured is read there."""
    config, pipeline, audio = clipforge
    theirs: list[str] = []

    def their_detect(video, noise_db, min_silence):
        theirs.append(video.name)
        return [audio.Silence(1.0, 1.5)]

    def their_detect_forbidden(*_a, **_k):
        raise AssertionError("ClipForge measured again instead of reading the plugin's cache file")

    assert paths.silences_dir() == config.PIPELINE_STATE_DIR
    folder = config.PIPELINE_STATE_DIR

    # ClipForge measures "talk.mp4" and writes its cache file. The plugin must find it.
    monkeypatch.setattr(audio, "detect_silences", their_detect)
    pipeline._silences_for(a_video(tmp_path), **settings)
    made_there = sorted(p.name for p in folder.iterdir())
    assert len(made_there) == 1 and theirs == ["talk.mp4"]
    assert silences_for(a_video(tmp_path), **settings) == [Silence(1.0, 1.5)]
    assert no_ffmpeg == [] and sorted(p.name for p in folder.iterdir()) == made_there

    # The plugin measures a second file. ClipForge must read what it wrote.
    other = tmp_path / "other.mp4"
    other.write_bytes(b"y" * 7)
    silences_for(other, **settings)
    assert no_ffmpeg == [("other.mp4", settings.get("noise_db", -25.0), settings.get("min_silence", 0.05))]
    monkeypatch.setattr(audio, "detect_silences", their_detect_forbidden)
    assert [(s.start, s.end) for s in pipeline._silences_for(other, **settings)] == [(1.0, 1.5)]
    assert len(list(folder.iterdir())) == 2
