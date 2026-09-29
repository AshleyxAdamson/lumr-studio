"""The production aligner, torchaudio's wav2vec 2.0 base 960h, with a fake model.

The model is a stand-in that "hears" scripted words at scripted times, and the
CTC alignment on top of it is torchaudio's real one. No test loads the model
file or fetches anything (docs/11.03 is where the numbers behind the two
tunings come from).
"""

from __future__ import annotations

import ast
import urllib.request
from pathlib import Path

import pytest
import torch
import torchaudio
from conftest import make_words

from lumr_studio import aligner, models
from lumr_studio.aligner import START_EARLIER_SECONDS, TAIL_PAD_SECONDS, Wav2Vec2Aligner
from lumr_studio.engine.audio_boundaries import Silence
from lumr_studio.errors import StudioError

REAL_BUNDLE = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
LABELS = REAL_BUNDLE.get_labels()
INDEX = {letter: i for i, letter in enumerate(LABELS)}
FRAME = 0.02  # the model hears the sound in frames of 20 ms
SAMPLES_A_FRAME = 320  # 16 kHz


class FakeVoice:
    """Stands in for the wav2vec2 model: it hears each scripted word's letters, two frames apart, then a word gap.

    ``said`` is ``(text, seconds)``, where ``seconds`` is where the word's
    first letter sounds, from the start of the video. ``window_start`` reads a
    window's own start off the audio it is handed, so a word is found in the
    window that holds it.
    """

    def __init__(self, said: list[tuple[str, float]], window_start=lambda _waveform: 0.0) -> None:
        self.said = said
        self.window_start = window_start
        self.heard_windows: list[float] = []

    def to(self, _device):
        return self

    def eval(self):
        return self

    def __call__(self, waveform):
        frames = waveform.shape[1] // SAMPLES_A_FRAME
        start = self.window_start(waveform)
        self.heard_windows.append(start)
        logits = torch.full((1, frames, len(LABELS)), -20.0)
        logits[..., INDEX["-"]] = 0.0
        for text, at in self.said:
            first = round((at - start) / FRAME)
            for k, letter in enumerate(text.upper() + "|"):
                frame = first + 2 * k
                if 0 <= frame < frames:
                    logits[0, frame, :] = -20.0
                    logits[0, frame, INDEX[letter]] = 0.0
        return logits, None


class FakeBundle:
    sample_rate = 16000

    def __init__(self, model) -> None:
        self._model = model

    def get_labels(self):
        return LABELS

    def get_model(self):
        return self._model


def heard(text: str, at: float) -> tuple[float, float]:
    """The first and last instant the fake hears ``text`` sounding from ``at``: a frame each letter, two apart."""
    return at, at + FRAME * (2 * (len(text) - 1) + 1)


def word(text: str, start: float, end: float, **more) -> dict:
    return {"word": text, "start": start, "end": end, **more}


def align(voice, words, *, video=Path("unused.mp4"), silences=(), monkeypatch=None, windows=None):
    """Run the aligner on ``words``; the audio is a fake that says where its window starts, and records the asks."""
    def decode(_path, rate, start=0.0, dur=None):
        if windows is not None:
            windows.append((start, dur))
        return torch.full((1, round(dur * rate)), float(start))

    monkeypatch.setattr(aligner, "_decode_segment", decode)
    return Wav2Vec2Aligner(bundle=FakeBundle(voice), device="cpu").align(video, words, list(silences))


def window_start_of(waveform) -> float:
    return float(waveform[0, 0])


# ── what comes out ────────────────────────────────────────────────────────────


def test_each_spoken_word_moves_to_where_the_model_hears_it_and_the_rest_is_kept(monkeypatch):
    transcript = [
        word("Hello", 0.20, 1.10, energy_rms=0.1),
        word("[laugh]", 1.10, 1.60, type="event"),
        word("--", 1.60, 1.70),
        word("there.", 1.90, 2.90, energy_rms=0.2),
    ]
    voice = FakeVoice([("hello", 0.50), ("there", 2.00)])
    out = align(voice, transcript, monkeypatch=monkeypatch)
    assert [(w["word"], w.get("type")) for w in out] == [(w["word"], w.get("type")) for w in transcript]
    assert [w.get("energy_rms") for w in out] == [0.1, None, None, 0.2]
    # A sound and a word with no letter in it keep the times the transcript gave them.
    assert (out[1]["start"], out[1]["end"]) == (1.10, 1.60) and (out[2]["start"], out[2]["end"]) == (1.60, 1.70)
    for got, (text, at) in ((out[0], ("hello", 0.50)), (out[3], ("there", 2.00))):
        first, last = heard(text, at)
        assert got["start"] == pytest.approx(first - START_EARLIER_SECONDS, abs=0.005)
        assert got["end"] == pytest.approx(last, abs=0.005)
    assert transcript[0]["start"] == 0.20, "the words it was given are left as they were"


def test_a_word_of_repeated_letters_and_an_apostrophe_is_placed_whole(monkeypatch):
    out = align(FakeVoice([("didn't", 1.00), ("well", 2.00)]), [word("Didn't", 0, 1), word("well", 1, 2)], monkeypatch=monkeypatch)
    assert out[0]["end"] == pytest.approx(heard("didn't", 1.00)[1], abs=0.005)
    assert out[1]["end"] == pytest.approx(heard("well", 2.00)[1], abs=0.005)


def test_a_transcript_with_no_spoken_word_is_handed_back_and_no_model_is_asked_for(monkeypatch):
    class NoModel(FakeBundle):
        def get_model(self):
            raise AssertionError("nothing to align, so nothing to load")

    transcript = [word("[laugh]", 1.0, 2.0, type="event"), word("--", 2.0, 2.1)]
    out = Wav2Vec2Aligner(bundle=NoModel(None), device="cpu").align(Path("unused.mp4"), transcript, [])
    assert out == transcript and out is not transcript


def test_on_a_real_decode_of_a_real_video_the_shape_is_the_same(video):
    words = make_words()
    on_a_frame = lambda seconds: round(seconds / FRAME) * FRAME  # noqa: E731
    said = [(w["word"].strip(".").lower(), on_a_frame(w["start"])) for w in words if w.get("type") != "event"]
    out = Wav2Vec2Aligner(bundle=FakeBundle(FakeVoice(said)), device="cpu").align(video, words, [])
    assert [(w["word"], w.get("type")) for w in out] == [(w["word"], w.get("type")) for w in words]
    spoken = [w for w in out if w.get("type") != "event"]
    assert all(0 <= w["start"] < w["end"] for w in spoken)
    assert out[0]["start"] == pytest.approx(on_a_frame(0.50) - START_EARLIER_SECONDS, abs=0.005)
    assert [w["start"] for w in spoken] == sorted(w["start"] for w in spoken)


# ── the two tunings (docs/11.03) ──────────────────────────────────────────────


def test_a_word_starts_one_frame_before_the_frame_it_is_heard_in(monkeypatch):
    assert START_EARLIER_SECONDS == FRAME
    out = align(FakeVoice([("take", 3.00)]), [word("take", 2.9, 3.4)], monkeypatch=monkeypatch)
    assert out[0]["start"] == pytest.approx(3.00 - START_EARLIER_SECONDS, abs=0.005)


def test_a_word_heard_in_the_first_frame_does_not_start_before_zero(monkeypatch):
    out = align(FakeVoice([("hi", 0.00)]), [word("hi", 0.0, 0.3)], monkeypatch=monkeypatch)
    assert out[0]["start"] == 0.0 and out[0]["end"] > 0.0


def test_only_half_a_second_of_audio_is_decoded_past_each_window(monkeypatch):
    """A word at the edge of a window is not clipped, and the next sentence's words are not in the pad."""
    assert TAIL_PAD_SECONDS == 0.5
    windows: list = []
    transcript = [word("early", 1.0, 1.5), word("late", 44.0, 44.5), word("last", 54.5, 55.0)]
    voice = FakeVoice([("early", 1.0), ("late", 44.0), ("last", 54.5)], window_start=window_start_of)
    out = align(voice, transcript, monkeypatch=monkeypatch, windows=windows)
    assert windows == [(0.0, 40.0 + TAIL_PAD_SECONDS), (40.0, 15.0 + TAIL_PAD_SECONDS)]
    assert voice.heard_windows == [0.0, 40.0]
    # A word in the second window is placed on the video's clock, not the window's.
    assert out[1]["start"] == pytest.approx(44.0 - START_EARLIER_SECONDS, abs=0.005)
    assert out[2]["end"] == pytest.approx(heard("last", 54.5)[1], abs=0.005)


def test_a_word_is_aligned_in_the_window_its_transcript_start_falls_in(monkeypatch):
    """A word that starts at 39.9 s is heard in window one, though the pad reaches past 40 s."""
    voice = FakeVoice([("edge", 39.90), ("next", 40.60)], window_start=window_start_of)
    out = align(voice, [word("edge", 39.9, 40.3), word("next", 40.5, 40.9)], monkeypatch=monkeypatch)
    assert voice.heard_windows == [0.0, 40.0]
    assert out[0]["start"] == pytest.approx(39.90 - START_EARLIER_SECONDS, abs=0.005)
    assert out[1]["start"] == pytest.approx(40.60 - START_EARLIER_SECONDS, abs=0.005)


def test_a_word_is_held_to_the_next_measured_silence_as_before(monkeypatch):
    """The silence clamp is ClipForge's: a word never runs on past a pause that starts inside it."""
    text, at = "because", 1.00
    first, last = heard(text, at)
    pause = Silence(start=first + 0.20, end=first + 0.80)
    out = align(FakeVoice([(text, at)]), [word(text, 0.9, 2.0)], silences=[pause], monkeypatch=monkeypatch)
    assert last > pause.start
    assert out[0]["end"] == pytest.approx(pause.start, abs=0.001)


# ── nothing is fetched, and MMS_FA is never loaded ────────────────────────────


class Tripwire:
    """Stands in for MMS_FA: any use of it fails the test."""

    def __getattr__(self, name):
        raise AssertionError(f"the plugin touched MMS_FA ({name})")


def test_without_the_model_file_the_aligner_refuses_to_fetch_it_and_never_touches_mms(monkeypatch):
    """The real torchaudio bundle, an empty torch cache, no network: the download hook refuses, and nothing goes out."""
    def no_network(*_a, **_k):
        raise AssertionError("the aligner reached for the network")

    fetch = torch.hub.download_url_to_file
    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    monkeypatch.setattr(torchaudio.pipelines, "MMS_FA", Tripwire())
    with pytest.raises(StudioError, match=f"tried to download {models.ALIGNER_FILE.url}"):
        Wav2Vec2Aligner(device="cpu").align(Path("unused.mp4"), [word("hello", 0.5, 0.9)], [])
    assert torch.hub.download_url_to_file is fetch, "torch's downloader is put back"


def test_the_default_bundle_is_the_mit_one_and_its_file_is_the_one_the_plugin_downloads():
    assert REAL_BUNDLE._path == models.ALIGNER_FILE.name
    assert f"https://download.pytorch.org/torchaudio/models/{REAL_BUNDLE._path}" == models.ALIGNER_FILE.url
    assert models.ALIGNER.license == "MIT" and models.ALIGNER.source_host == "download.pytorch.org"


# The plugin's own code, without the generated engine and speech copies.
PACKAGE = Path(aligner.__file__).parent
GENERATED_HEADER = "# GENERATED by studio/tools/sync-engine.sh"
# What loads MMS_FA: the bundle itself, the tokenizer and aligner it brings, and
# the copied function that loads it (``engine/alignment.py``, generated from
# ClipForge, which keeps MMS_FA).
MMS_NAMES = {"MMS_FA", "get_tokenizer", "get_aligner", "align_words", "_load_aligner", "_align_window"}


def own_modules() -> list[Path]:
    return [p for p in PACKAGE.rglob("*.py") if not p.read_text().startswith(GENERATED_HEADER)]


def test_no_plugin_code_reaches_mms_fa():
    """Nothing hand-written names MMS_FA or calls the generated copy's loader. Docstrings and comments may say why."""
    found = []
    for path in own_modules():
        for node in ast.walk(ast.parse(path.read_text())):
            name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
            if name in MMS_NAMES or (isinstance(node, ast.alias) and node.name in MMS_NAMES):
                found.append(f"{path.relative_to(PACKAGE)}:{node.lineno} {name or node.name}")
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in MMS_NAMES:
                found.append(f"{path.relative_to(PACKAGE)}:{node.lineno} {node.value!r}")
    assert found == []


def test_the_scan_sees_the_generated_copy_that_still_holds_the_loader():
    """The check above is only worth something if it would notice: the copy does hold the loader, and is left out on purpose."""
    copy = (PACKAGE / "engine" / "alignment.py").read_text()
    assert copy.startswith(GENERATED_HEADER) and "MMS_FA" in copy
    assert PACKAGE / "engine" / "alignment.py" not in own_modules()
