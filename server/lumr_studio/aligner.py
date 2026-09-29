"""The production aligner: torchaudio's wav2vec 2.0 base 960h pins each word to the sound.

The weights are MIT. They replace MMS_FA, whose weights are CC-BY-NC 4.0, and
the plugin promises creators they can sell what they make (docs/11.03). The
plugin never loads MMS_FA, and nothing here fetches a model: ``lacks`` says
what is missing, and ``models.download`` is the one place a model arrives.

The windowing, the tail pad's job and the silence clamp are ClipForge's
(``engine/alignment.py``, a generated copy). torchaudio's English bundles have
no tokenizer or aligner object the way MMS_FA has, so this class does what
they do in about 40 lines: uppercase letters with a ``|`` between words,
``log_softmax`` on the model's output, CTC forced alignment, and a word's span
from its first letter's first frame to its last letter's last frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lumr_studio import word_times
from lumr_studio.engine.alignment import _clamp_ends_to_silence, _decode_segment, normalize_word
from lumr_studio.engine.audio_boundaries import Silence

# Long audio is aligned a window at a time: one pass over a 20 minute take
# would blow up the model's attention memory.
WINDOW_SECONDS = 40.0
# Audio decoded past a window's end, so a word at the edge isn't clipped. It
# was ClipForge's 2 s, and 2 s is too much: the audio then holds the start of
# the next sentence, and a window's last word can land there. Measured on the
# test take, "you." landed 1.6 s late at 2 s and right at 0.5 s (docs/11.03
# section 6).
TAIL_PAD_SECONDS = 0.5
# wav2vec2 starts a word a median frame (20 ms) later than MMS_FA did. A filler
# cut ends where the next word starts, so a late start clips the word's onset:
# "Grinder" and "men" on the test take. One frame earlier put the words lost
# back to MMS_FA's count (docs/11.03 section 6). Tuned on one take.
START_EARLIER_SECONDS = 0.02


def _alignable(word: dict[str, Any]) -> bool:
    """Whether the aligner places ``word``: a spoken word with times and at least one letter."""
    return (
        word.get("type") != "event" and "start" in word and "end" in word
        and bool(normalize_word(str(word.get("word", ""))))
    )


def _place_words(model: Any, device: str, waveform: Any, words: list[str], labels: tuple[str, ...], rate: int) -> list[tuple[float, float]]:
    """``(start, end)`` in seconds from the start of ``waveform``, for each of ``words``, in order.

    ``words`` are normalized (``normalize_word``). Raises RuntimeError when the
    alignment doesn't give one span for each letter.
    """
    import torch
    import torchaudio

    with torch.inference_mode():
        logits, _ = model(waveform.to(device))
        emission = torch.log_softmax(logits.float(), dim=-1)[0].cpu()
    ratio = waveform.size(1) / emission.size(0)  # samples in a frame
    index = {letter: i for i, letter in enumerate(labels)}
    blank, gap = index["-"], index["|"]
    targets: list[int] = []
    owner: list[int] = []  # the word each target belongs to; -1 for a gap
    for k, word in enumerate(words):
        if k:
            targets.append(gap)
            owner.append(-1)
        for letter in word.upper():
            targets.append(index[letter])
            owner.append(k)
    # torchaudio's forced_align has no MPS kernel: this step runs on the CPU.
    paths, scores = torchaudio.functional.forced_align(
        emission.unsqueeze(0), torch.tensor([targets], dtype=torch.int32), blank=blank,
    )
    spans = torchaudio.functional.merge_tokens(paths[0], scores[0].exp(), blank=blank)
    if len(spans) != len(targets):
        raise RuntimeError(f"the aligner made {len(spans)} spans for {len(targets)} letters")
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    for span, k in zip(spans, owner):
        if k >= 0:
            first.setdefault(k, span.start)
            last[k] = span.end
    return [(first[k] * ratio / rate, last[k] * ratio / rate) for k in range(len(words))]


class Wav2Vec2Aligner:
    """Aligns with ``torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H``. Tests pass a fake ``bundle``."""

    def __init__(self, bundle: Any = None, device: str | None = None) -> None:
        self._bundle = bundle
        self._device = device

    def lacks(self) -> str:
        return word_times.aligning_lacks()

    def align(self, video: Path, words: word_times.Words, silences: list[Silence]) -> word_times.Words:
        """``words`` with each spoken word's start and end moved onto the sound, ends held to the next measured silence."""
        import torch
        import torchaudio

        out = [dict(w) for w in words]
        speech = [(i, w) for i, w in enumerate(words) if _alignable(w)]
        if not speech:
            return out
        bundle = self._bundle or torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        device = self._device or ("mps" if torch.backends.mps.is_available() else "cpu")
        # torchaudio fetches the model when its file is missing. ``lacks``
        # checked the file a moment ago; this holds even if it went since.
        fetch = torch.hub.download_url_to_file
        torch.hub.download_url_to_file = word_times.refuse_download
        try:
            model = bundle.get_model().to(device).eval()
        finally:
            torch.hub.download_url_to_file = fetch
        labels, rate = bundle.get_labels(), int(bundle.sample_rate)
        end_of_speech = max(float(w["end"]) for _, w in speech)
        t = 0.0
        while t < end_of_speech:
            t1 = min(t + WINDOW_SECONDS, end_of_speech)
            window = [(i, w) for i, w in speech if t <= float(w["start"]) < t1]
            if window:
                waveform = _decode_segment(video, rate, start=t, dur=(t1 - t) + TAIL_PAD_SECONDS)
                spans = _place_words(model, device, waveform, [normalize_word(str(w["word"])) for _, w in window], labels, rate)
                for (i, _), (start, end) in zip(window, spans):
                    out[i]["start"] = round(max(0.0, round(t + start, 3) - START_EARLIER_SECONDS), 3)
                    out[i]["end"] = round(t + end, 3)
            t = t1
        _clamp_ends_to_silence(out, silences)
        return out
