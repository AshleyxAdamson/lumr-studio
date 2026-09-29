import json
import os

import numpy as np
import pytest

from conftest import spoken_run
from lumr_studio.errors import StudioError
from lumr_studio.project import open_project
from lumr_studio.sounds import (
    SAMPLE_RATE,
    SOUNDS_FILE,
    classify_events,
    decode_span,
    is_filler,
    load_or_measure_labels,
    measure_events,
    measure_samples,
    protected_spans,
)

RNG = np.random.default_rng(7)


# ── Synthetic audio ───────────────────────────────────────────────────────────


def voice(seconds: float, pitch: float = 200.0, level: float = 0.5) -> np.ndarray:
    """A pitched tone with a few harmonics, like a held vowel."""
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    wave = sum(np.sin(2 * np.pi * pitch * k * t) / k for k in (1, 2, 3))
    return level * wave / np.abs(wave).max()


def breath(seconds: float, level: float) -> np.ndarray:
    """Unpitched noise: the "h" between laugh bursts."""
    return level * RNG.uniform(-1, 1, int(seconds * SAMPLE_RATE))


def laugh(bursts: int = 8, period: float = 0.2, voiced_part: float = 0.07) -> np.ndarray:
    """Short pitched bursts separated by quieter breathy noise, ``1 / period`` a second."""
    pieces = []
    for _ in range(bursts):
        pieces += [voice(voiced_part, pitch=280.0), breath(period - voiced_part, 0.12)]
    return np.concatenate(pieces)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE))


# ── Measuring ─────────────────────────────────────────────────────────────────


def test_laugh_measures_as_a_breathy_burst_train():
    m = measure_samples(laugh(bursts=8, period=0.2))
    assert m["bursts"] >= 6
    assert 4.0 <= m["burst_rate"] <= 6.0
    assert 0.15 <= m["voiced"] <= 0.5


def test_held_vowel_measures_as_one_steady_voiced_sound():
    uhh = voice(1.0, pitch=180.0) * (1 + 0.1 * np.sin(2 * np.pi * 3 * np.arange(SAMPLE_RATE) / SAMPLE_RATE))
    m = measure_samples(np.concatenate([silence(0.2), uhh, silence(0.2)]))
    assert m["bursts"] <= 1
    assert m["burst_rate"] == 0.0
    assert m["voiced"] > 0.8


def test_breath_noise_has_no_pitch():
    m = measure_samples(breath(1.0, 0.2))
    assert m["voiced"] < 0.15


def test_bursts_too_far_apart_are_not_a_train():
    # Four syllables a full second apart: separate sounds, not laughter.
    clip = np.concatenate([np.concatenate([voice(0.1), silence(0.9)]) for _ in range(4)])
    assert measure_samples(clip)["bursts"] <= 1


def test_silence_and_tiny_clips_measure_as_nothing():
    assert measure_samples(silence(1.0))["bursts"] == 0
    assert measure_samples(np.zeros(10)) == {"bursts": 0, "burst_rate": 0.0, "voiced": 0.0}


def test_measure_events_decodes_each_span_in_order():
    seen = []

    def fake_decode(video, start, end, rate):
        seen.append((start, end, rate))
        return laugh() if start > 5 else voice(end - start)

    out = measure_events("talk.mp4", [(1.0, 2.0), (8.0, 10.0)], decode=fake_decode)
    assert seen == [(1.0, 2.0, SAMPLE_RATE), (8.0, 10.0, SAMPLE_RATE)]
    assert out[0]["voiced"] > 0.8
    assert out[1]["bursts"] >= 6


def test_decode_span_reads_the_video_audio(video):
    samples = decode_span(video, 2.0, 3.0)
    assert abs(len(samples) - SAMPLE_RATE) < SAMPLE_RATE * 0.05
    assert measure_samples(samples)["voiced"] > 0.8  # the fixture's steady 440 Hz tone


def test_decode_span_failure_says_what_to_check(tmp_path):
    bad = tmp_path / "broken.mp4"
    bad.write_text("not a video")
    with pytest.raises(StudioError, match="Check that the source video plays"):
        decode_span(bad, 0.0, 1.0)


# ── Classifying ───────────────────────────────────────────────────────────────


def event(start: float, end: float) -> dict:
    return {"word": "[vocalization]", "start": start, "end": end, "type": "event"}


def talk() -> list[dict]:
    """A punchline, a long event after it, a mid-sentence event, and a short event after a sentence.

    Events: 4.0-6.5 after "punchline.", 7.8-10.0 after "want", 11.3-11.6 after "this.".
    """
    words = spoken_run("Here is setup.", 0.0)                # ends 1.24
    words += spoken_run("That is the punchline.", 1.5)      # 1.5 to 3.14
    words.append(event(4.0, 6.5))
    words += spoken_run("Next I want", 6.6)                  # ends 7.84
    words.append(event(7.8, 10.0))
    words += spoken_run("to say this.", 10.1)                # ends 11.34
    words.append(event(11.3, 11.6))
    words += spoken_run("The end.", 11.7)
    return words


LAUGH_MEASURE = {"bursts": 8, "burst_rate": 5.0, "voiced": 0.35}
UHH_MEASURE = {"bursts": 1, "burst_rate": 0.0, "voiced": 0.9}


def test_without_measures_only_long_events_after_a_sentence_are_possible_laughs():
    labels = classify_events(talk())
    assert [(l["kind"], l["confidence"]) for l in labels] == [
        ("laugh", "possible"), ("sound", "possible"), ("sound", "possible"),
    ]
    assert labels[0]["punchline"] == [1.5, pytest.approx(3.14)]
    assert labels[1]["punchline"] is None


def test_label_shape_and_order():
    labels = classify_events(talk())
    assert [l["start"] for l in labels] == [4.0, 7.8, 11.3]
    assert labels[0] == {
        "start": 4.0, "end": 6.5, "seconds": 2.5, "kind": "laugh",
        "confidence": "possible", "voiced": None, "bursts": None, "burst_rate": None,
        "punchline": labels[0]["punchline"],
    }


def test_measured_laugh_after_punchline_is_likely():
    labels = classify_events(talk(), measures=[LAUGH_MEASURE, UHH_MEASURE, UHH_MEASURE])
    assert (labels[0]["kind"], labels[0]["confidence"]) == ("laugh", "likely")
    assert labels[1]["kind"] == "sound"
    assert labels[1]["voiced"] == 0.9


def test_is_filler_reads_the_voiced_share_of_a_sound_not_a_laugh():
    # Her take, measured: voiced 0.8 and up is an unmistakable held "uhh"; under 0.5 is breathy, no held
    # pitch; between the two, still a filler by ear. A laugh never counts, whatever its voiced share.
    held = {"kind": "sound", "voiced": 0.86}
    mixed = {"kind": "sound", "voiced": 0.62}
    breath = {"kind": "sound", "voiced": 0.31}
    unmeasured = {"kind": "sound", "voiced": None}
    a_laugh = {"kind": "laugh", "voiced": 0.86}
    assert is_filler(held) and is_filler(mixed)
    assert not is_filler(breath) and not is_filler(unmeasured) and not is_filler(a_laugh)


def test_is_filler_refuses_a_voiced_laugh_train_judged_as_a_sound():
    # A giggle can measure breathy enough to read as pitched throughout (voiced 0.6, over
    # FILLER_VOICED_MIN) without being breathy enough for _judge to call it a laugh (LAUGH_VOICED
    # tops out at 0.5); its burst train still gives it away.
    giggle = {"kind": "sound", "voiced": 0.6, "bursts": 6, "burst_rate": 4.5}
    assert not is_filler(giggle), "a burst train in laugh's rate is a laugh, whatever it measured as"
    # A held "uhh" has no train (bursts under LIKELY_MIN_BURSTS) and is still a filler.
    uhh = {"kind": "sound", "voiced": 0.6, "bursts": 1, "burst_rate": 0.0}
    assert is_filler(uhh)
    # A voiced train too fast or slow for laughter is not one either.
    too_fast = {"kind": "sound", "voiced": 0.6, "bursts": 6, "burst_rate": 9.0}
    assert is_filler(too_fast)


def test_laugh_train_mid_sentence_is_only_possible():
    labels = classify_events(talk(), measures=[UHH_MEASURE, LAUGH_MEASURE, UHH_MEASURE])
    assert labels[0]["kind"] == "sound"  # long and after a punchline, but a steady voice
    assert (labels[1]["kind"], labels[1]["confidence"]) == ("laugh", "possible")
    assert labels[1]["punchline"] is None


def test_breathy_but_unpitched_or_short_events_stay_sounds():
    breath_only = {"bursts": 0, "burst_rate": 0.0, "voiced": 0.0}
    short_breathy = {"bursts": 1, "burst_rate": 0.0, "voiced": 0.4}
    labels = classify_events(talk(), measures=[breath_only, UHH_MEASURE, short_breathy])
    assert [l["kind"] for l in labels] == ["sound", "sound", "sound"]


def test_breathy_long_event_after_punchline_is_possible():
    labels = classify_events(
        talk(), measures=[{"bursts": 1, "burst_rate": 0.0, "voiced": 0.4}, UHH_MEASURE, UHH_MEASURE]
    )
    assert (labels[0]["kind"], labels[0]["confidence"]) == ("laugh", "possible")


def test_fast_or_slow_trains_are_not_laughter():
    too_fast = {"bursts": 8, "burst_rate": 9.0, "voiced": 0.35}
    labels = classify_events(talk(), measures=[too_fast, UHH_MEASURE, UHH_MEASURE])
    assert labels[0]["confidence"] != "likely"


def test_unmeasured_entry_falls_back_to_length_and_position():
    labels = classify_events(talk(), measures=[{}, UHH_MEASURE, UHH_MEASURE])
    assert (labels[0]["kind"], labels[0]["confidence"]) == ("laugh", "possible")


def test_measures_must_match_the_events():
    with pytest.raises(StudioError, match="2 sound measures for 3 events"):
        classify_events(talk(), measures=[LAUGH_MEASURE, UHH_MEASURE])


def test_one_connecting_word_between_punchline_and_laugh_is_allowed():
    words = spoken_run("I dropped the pie.", 0.0)        # ends 1.64
    words.append({"word": "And", "start": 1.6, "end": 1.8})
    words.append(event(1.9, 3.9))
    labels = classify_events(words)
    assert labels[0]["kind"] == "laugh"
    assert labels[0]["punchline"] == [0.0, pytest.approx(1.64)]


def test_connecting_word_long_after_the_sentence_breaks_the_link():
    words = spoken_run("I dropped the pie.", 0.0)
    words.append({"word": "So", "start": 3.5, "end": 3.8})
    words.append(event(3.9, 6.0))
    assert classify_events(words)[0]["kind"] == "sound"


def test_no_events_no_labels():
    assert classify_events(spoken_run("Just talk.", 0.0)) == []


# ── Protecting ────────────────────────────────────────────────────────────────


def test_protected_span_runs_from_the_beat_before_the_punchline_to_the_pause_after_the_laugh():
    words = talk()
    labels = classify_events(words, measures=[LAUGH_MEASURE, UHH_MEASURE, UHH_MEASURE])
    # The setup ends at 1.24 and the punchline starts at 1.5: that beat is protected.
    # Laugh 4.0-6.5, next word at 6.6: the 0.1 s pause is protected whole.
    assert protected_spans(labels, words) == [(pytest.approx(1.24), pytest.approx(6.6))]


def test_pause_before_the_punchline_is_capped_at_one_second():
    words = spoken_run("Setup.", 0.0) + spoken_run("That is the joke.", 5.0)
    words.append(event(7.0, 9.0))
    words += spoken_run("Moving on.", 9.1)
    labels = classify_events(words, measures=[LAUGH_MEASURE])
    assert protected_spans(labels, words) == [(pytest.approx(4.0), pytest.approx(9.1))]


def test_pause_after_laugh_is_capped_at_one_second():
    words = spoken_run("That is the joke.", 0.0)
    words.append(event(2.0, 4.0))
    words += spoken_run("Moving on.", 7.0)
    labels = classify_events(words)
    assert protected_spans(labels, words) == [(0.0, 5.0)]


def test_laugh_without_punchline_protects_itself_and_sounds_protect_nothing():
    words = talk()
    labels = classify_events(words, measures=[UHH_MEASURE, LAUGH_MEASURE, UHH_MEASURE])
    assert protected_spans(labels, words) == [(7.8, pytest.approx(10.1))]


def run_on(start: float, seconds: float) -> list[dict]:
    """One sentence of half-second words lasting ``seconds``, ending in a full stop."""
    n = int(seconds / 0.5)
    return [
        {"word": "joke." if i == n - 1 else "word", "start": start + i * 0.5, "end": start + i * 0.5 + 0.45}
        for i in range(n)
    ]


def test_punchline_of_a_run_on_sentence_starts_at_its_last_pause():
    words = spoken_run("Setup.", 0.0) + run_on(2.0, 12.0)   # 2.0 to 13.95
    for w in words:
        if w["start"] >= 10.0:                                # a 0.6s pause before the last 4s
            w["start"] += 0.6
            w["end"] += 0.6
    words.append(event(14.6, 16.6))
    label = classify_events(words, measures=[LAUGH_MEASURE])[0]
    assert label["punchline"] == [pytest.approx(10.6), pytest.approx(14.55)]


def test_punchline_of_a_run_on_sentence_without_a_pause_is_capped():
    words = spoken_run("Setup.", 0.0) + run_on(2.0, 12.0)
    words.append(event(14.0, 16.0))
    label = classify_events(words, measures=[LAUGH_MEASURE])[0]
    assert label["punchline"] == [pytest.approx(6.0), pytest.approx(13.95)]


def test_a_pause_inside_a_short_punchline_does_not_split_it():
    words = spoken_run("Setup.", 0.0)
    words += [
        {"word": "Wait", "start": 2.0, "end": 2.4}, {"word": "for", "start": 2.45, "end": 2.7},
        {"word": "it.", "start": 3.6, "end": 4.0},
    ]
    words.append(event(4.1, 6.1))
    label = classify_events(words, measures=[LAUGH_MEASURE])[0]
    assert label["punchline"] == [pytest.approx(2.0), pytest.approx(4.0)]


def test_overlapping_protected_spans_merge():
    labels = [
        {"start": 2.0, "end": 3.0, "seconds": 1.0, "kind": "laugh", "confidence": "likely", "punchline": [0.0, 1.9]},
        {"start": 3.5, "end": 4.0, "seconds": 0.5, "kind": "laugh", "confidence": "possible", "punchline": [3.1, 3.4]},
    ]
    words = spoken_run("One two three four.", 0.0) + spoken_run("Five.", 3.1) + spoken_run("Six.", 4.5)
    assert protected_spans(labels, words) == [(0.0, 4.5)]


# ── Caching ───────────────────────────────────────────────────────────────────


class CountingMeasure:
    def __init__(self):
        self.calls = []

    def __call__(self, video, spans):
        self.calls.append(spans)
        return [UHH_MEASURE for _ in spans]


def test_labels_are_measured_once_then_cached(video, words):
    project = open_project(str(video))
    measure = CountingMeasure()
    first = load_or_measure_labels(project, words, measure=measure)
    second = load_or_measure_labels(project, words, measure=measure)
    assert first == second
    assert measure.calls == [[(6.0, 6.6)]]
    cached = json.loads((project.root / SOUNDS_FILE).read_text())
    assert cached["labels"] == first
    assert cached["measures"] == [UHH_MEASURE]


def test_retranscribe_invalidates_the_cache(video, words):
    project = open_project(str(video))
    measure = CountingMeasure()
    load_or_measure_labels(project, words, measure=measure)
    stat = project.words_path.stat()
    os.utime(project.words_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    load_or_measure_labels(project, words, measure=measure)
    assert len(measure.calls) == 2


def test_corrupt_or_old_cache_is_rebuilt(video, words):
    project = open_project(str(video))
    measure = CountingMeasure()
    (project.root / SOUNDS_FILE).write_text("{not json")
    load_or_measure_labels(project, words, measure=measure)
    data = json.loads((project.root / SOUNDS_FILE).read_text())
    data["version"] = 0
    (project.root / SOUNDS_FILE).write_text(json.dumps(data))
    load_or_measure_labels(project, words, measure=measure)
    assert len(measure.calls) == 2


def test_no_events_means_no_measuring(video):
    project = open_project(str(video))
    measure = CountingMeasure()
    assert load_or_measure_labels(project, spoken_run("Only words.", 0.0), measure=measure) == []
    assert measure.calls == []


def test_sounds_are_judged_again_once_the_words_have_room_from_the_measures_already_saved(video):
    # A punchline is found from the words before a laugh, and those words end later once they have their room.
    import short_word
    from lumr_studio import word_times

    laugh_after = short_word.transcript() + [{"word": "[vocalization]", "start": 7.5, "end": 9.5, "type": "event"}]
    video.with_suffix(".words.json").write_text(json.dumps(laugh_after))
    project = open_project(str(video))

    class WithTheLaugh(short_word.ShortWordAligner):
        def align(self, video, words, silences):
            return super().align(video, [w for w in words if w.get("type") != "event"], silences) + [dict(laugh_after[-1])]

    quiet = short_word.silences()[:-1]  # the take ends on the laugh, not on silence
    word_times.align_project(project, aligner=WithTheLaugh(), silences=lambda _video: quiet)
    measure = CountingMeasure()
    roomy = load_or_measure_labels(project, word_times.load_words(project), measure=measure)
    # The labels as the plugin cached them before words had room: judged on measured times, and no word of how those were made.
    path = project.root / SOUNDS_FILE
    cached = json.loads(path.read_text())
    cached.pop("word_times_made_as")
    cached["labels"] = [{**label, "punchline": [6.5, 7.0]} for label in cached["labels"]]
    path.write_text(json.dumps(cached))
    again = load_or_measure_labels(project, word_times.load_words(project), measure=measure)
    assert again == roomy and again != cached["labels"], "judged again on the words as they are now"
    assert len(measure.calls) == 1, "the sound sits where it sat, so its measures are used again"
    assert json.loads(path.read_text())["word_times_made_as"] == word_times.ALIGNED_VERSION
    load_or_measure_labels(project, word_times.load_words(project), measure=measure)
    assert len(measure.calls) == 1
