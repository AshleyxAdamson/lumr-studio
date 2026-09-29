from lumr_studio import tools
from lumr_studio.analysis import analyze
from lumr_studio.retakes import find_retakes


def _w(text, start, step=0.3):
    return [
        {"word": t, "start": start + i * step, "end": start + i * step + 0.25}
        for i, t in enumerate(text.split())
    ]


def test_finds_a_restarted_phrase(words):
    (retake,) = find_retakes(words)
    assert retake.phrase == "today we talk about editing"
    assert retake.first_start == 3.90
    assert retake.repeat_start == 8.00


def test_short_repeats_and_distinct_phrases_are_not_retakes():
    words = _w("so the plan is simple", 0) + _w("the plan changed today", 5)
    assert find_retakes(words) == []


def test_repeat_beyond_sixty_seconds_is_not_a_retake():
    words = _w("this is my favourite part", 0) + _w("this is my favourite part", 70)
    assert find_retakes(words) == []
    assert len(find_retakes(words, max_gap=120)) == 1


def test_punctuation_and_case_do_not_hide_a_retake():
    words = _w("So this is the plan.", 0) + _w("so this is the plan", 4)
    assert len(find_retakes(words)) == 1


def test_report_contents(words):
    report = analyze(words, 20.0, [])
    assert report["word_count"] == 25 and report["event_count"] == 1
    assert report["longest_pauses"][0] == {"start": 6.6, "end": 8.0, "seconds": 1.4}
    assert report["fillers"][0]["word"] == "um"
    assert report["retake_count"] == 1
    assert report["auto_microcuts"]["count"] > 0
    assert report["auto_microcuts"]["seconds_saved"] > 0
    assert report["timing_quality"]["quality"] == "measured"
    assert "warning" not in report


def test_analyze_take_tool(video):
    report = tools.analyze_take(str(video), silences=lambda _v: [])
    assert report["duration"] == 20.0
