"""Shared fixtures. Every test runs against temp folders, never the user's LUMR_HOME."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from lumr_studio import models, transcription, word_times
from lumr_studio.engine import alignment, paths
from lumr_studio.jobs import JobRegistry

VIDEO_SECONDS = 20.0

# (word, start, end[, "event"]). A 20 s talk with pauses, a filler, an event,
# and a retake ("today we talk about editing" said twice).
WORD_ROWS = [
    ("Hello", 0.50, 0.90), ("everyone", 0.95, 1.50), ("and", 1.60, 1.80), ("welcome.", 1.85, 2.40),
    ("um", 3.40, 3.70), ("today", 3.90, 4.30), ("we", 4.35, 4.50), ("talk", 4.55, 4.90),
    ("about", 4.95, 5.30), ("editing", 5.35, 6.00), ("[vocalization]", 6.00, 6.60, "event"),
    ("today", 8.00, 8.40), ("we", 8.45, 8.60), ("talk", 8.65, 9.00), ("about", 9.05, 9.40),
    ("editing", 9.45, 10.10), ("video.", 10.15, 10.70),
    ("this", 12.00, 12.30), ("is", 12.35, 12.50), ("the", 12.55, 12.70), ("part", 12.75, 13.10),
    ("that", 13.15, 13.40), ("matters.", 13.45, 14.20),
    ("thanks", 15.50, 16.00), ("for", 16.05, 16.20), ("watching.", 16.25, 17.00),
]


def make_words() -> list[dict]:
    words = []
    for row in WORD_ROWS:
        entry = {"word": row[0], "start": row[1], "end": row[2], "energy_rms": 0.1}
        if len(row) == 4:
            entry["type"] = row[3]
        words.append(entry)
    return words


def spread_evenly(sentences: list[tuple[str, float, float]]) -> list[dict]:
    """Words with SYNTHESIZED timings: each sentence's words share its span equally.

    Reproduces the Hammy bug: every word in a sentence gets the same duration,
    and neighbouring sentences may overlap.
    """
    words = []
    for text, start, end in sentences:
        tokens = text.split()
        step = (end - start) / len(tokens)
        words += [
            {"word": t, "start": round(start + i * step, 3), "end": round(start + (i + 1) * step, 3)}
            for i, t in enumerate(tokens)
        ]
    return sorted(words, key=lambda w: w["start"])


def spoken_run(text: str, start: float, step: float = 0.4, gap: float = 0.05) -> list[dict]:
    """Measured-looking words back to back: durations vary, gaps under a pause."""
    words = []
    t = start
    for i, token in enumerate(text.split()):
        dur = step - gap + (i % 3) * 0.03
        words.append({"word": token, "start": round(t, 3), "end": round(t + dur, 3)})
        t += dur + gap
    return words


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Point every root at temp folders and forbid the speech model, cached silences and the real aligner.

    No test loads the aligner's model: to the production aligner this
    machine lacks it, and a test that aligns passes a fake (``FakeAligner``).
    No test downloads a model: ``transcribe`` finds no model missing unless a
    test passes its own (``model_set=``, with a fake ``opener=``).
    No soft sound is measured unless a test passes its own (``soft=``): the
    stand-in answers None, "not measured", so words get no soft edges and
    nothing is said about them.
    """
    monkeypatch.setenv("LUMR_STUDIO_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("LUMR_HOME", str(tmp_path / "lumr-home"))
    # The model caches are temp folders too: a test never reads or fills the ones on this machine.
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    # No test sends feedback to a real server, or reads a plugin folder Claude Code named: each sets its own.
    # An empty LUMR_SHARE_URL turns sharing off, so the team's real server is never the default here.
    monkeypatch.setenv("LUMR_SHARE_URL", "")
    monkeypatch.delenv("CLAUDE_PLUGIN_ROOT", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch-home"))

    def forbidden(*_a, **_k):
        raise AssertionError("tests must not run the speech model, align for real, or write silences under LUMR_HOME")

    monkeypatch.setattr(paths, "silences_for", forbidden)
    monkeypatch.setattr(transcription, "run_speech_model", forbidden)
    monkeypatch.setattr(alignment, "align_words", forbidden)
    monkeypatch.setattr(word_times, "aligning_lacks", lambda: NO_MODEL_IN_TESTS)
    # ``transcribe`` looks for no model unless a test hands it some (``model_set=``).
    monkeypatch.setattr(models, "PRODUCTION", models.ModelSet(speech=None, aligner=None))
    monkeypatch.setattr(word_times, "measured_soft_silences", lambda _video: None)
    return tmp_path


NO_MODEL_IN_TESTS = "Tests never load the model that measures word times."


class FakeAligner:
    """An aligner for tests: each spoken word shrinks to the middle half of its span, as aligning tightens words.

    ``lacks`` makes it a machine that can't align. ``calls`` counts the runs.
    """

    def __init__(self, lacks: str = "") -> None:
        self._lacks = lacks
        self.calls = 0

    def lacks(self) -> str:
        return self._lacks

    def align(self, video, words, silences):
        self.calls += 1
        out = []
        for w in words:
            entry = dict(w)
            if w.get("type") != "event":
                quarter = (w["end"] - w["start"]) / 4
                entry["start"], entry["end"] = round(w["start"] + quarter, 3), round(w["end"] - quarter, 3)
            out.append(entry)
        return out


@pytest.fixture
def words() -> list[dict]:
    return make_words()


@pytest.fixture
def jobs() -> JobRegistry:
    return JobRegistry()


@pytest.fixture(scope="session")
def synthetic_master(tmp_path_factory) -> Path:
    """A 20 s test-pattern video with a sine tone, generated once per session."""
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    out = tmp_path_factory.mktemp("master") / "talk.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc2=size=320x240:rate=25:duration={VIDEO_SECONDS}",
            "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={VIDEO_SECONDS}",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(out),
        ],
        check=True,
    )
    return out


@pytest.fixture
def video(tmp_path, synthetic_master) -> Path:
    """A fresh copy of the synthetic video with its words.json sidecar beside it."""
    folder = tmp_path / "footage"
    folder.mkdir()
    path = folder / "talk.mp4"
    shutil.copy(synthetic_master, path)
    (folder / "talk.words.json").write_text(json.dumps(make_words()))
    return path
