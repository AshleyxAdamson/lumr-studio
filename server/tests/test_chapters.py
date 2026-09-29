import pytest

from lumr_studio import tools
from lumr_studio.errors import StudioError
from lumr_studio.silences import no_silences
from lumr_studio.timeline import edited_duration, source_spans_for_window, to_edited_time

KEPT = [(0.0, 5.0), (8.0, 12.0), (15.0, 20.0)]  # cuts at 5-8 and 12-15


@pytest.mark.parametrize(
    ("source", "edited", "in_cut"),
    [
        (0.0, 0.0, False),
        (4.0, 4.0, False),
        (9.0, 6.0, False),     # 5 s kept before, then 1 s into the second segment
        (16.0, 10.0, False),
        (6.5, 5.0, True),      # inside the first cut: start of the next kept segment
        (13.0, 9.0, True),
    ],
)
def test_source_to_edited(source, edited, in_cut):
    assert to_edited_time(source, KEPT) == (pytest.approx(edited), in_cut)


def test_edited_duration():
    assert edited_duration(KEPT) == 14.0


def test_window_maps_back_to_source_spans():
    assert source_spans_for_window(KEPT, 3.0, 7.0) == [(3.0, 5.0), (8.0, 10.0)]


def test_chapter_times_tool_uses_the_saved_edit(video):
    tools.set_edit(str(video), [{"start": 7.9, "end": 10.8, "reason": "retake"}], silences=no_silences)
    cut = tools.get_edit(str(video))["cuts"][0]
    result = tools.chapter_times(str(video), [0.0, 9.0, 16.0])
    first, inside, after = result["times"]
    assert first == {"source": 0.0, "edited": 0.0, "inside_cut": False}
    assert inside["inside_cut"] is True
    assert inside["edited"] == pytest.approx(cut["start"], abs=0.001)
    assert after["edited"] == pytest.approx(16.0 - (cut["end"] - cut["start"]), abs=0.001)


def test_chapter_time_outside_video_is_an_error(video):
    with pytest.raises(StudioError, match="outside the video"):
        tools.chapter_times(str(video), [25.0])
