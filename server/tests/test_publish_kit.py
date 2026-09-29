import json
from pathlib import Path

import pytest

from lumr_studio import tools
from lumr_studio.errors import StudioError
from lumr_studio.publish_kit import format_timestamp, validate_chapters, write_publish_kit

GOOD = [{"time": 0, "label": "Intro"}, {"time": 30, "label": "Setup"}, {"time": 75, "label": "Result"}]


def test_first_chapter_must_start_at_zero():
    with pytest.raises(StudioError, match="must start at 0"):
        validate_chapters([{"time": 2, "label": "a"}, *GOOD[1:], {"time": 90, "label": "b"}], 120)


def test_needs_three_chapters():
    with pytest.raises(StudioError, match="at least 3 chapters; got 2"):
        validate_chapters(GOOD[:2], 120)


def test_chapters_closer_than_ten_seconds_rejected():
    close = [{"time": 0, "label": "a"}, {"time": 8, "label": "b"}, {"time": 40, "label": "c"}]
    with pytest.raises(StudioError, match="Chapter 0 .* lasts 8.0s"):
        validate_chapters(close, 120)


def test_last_chapter_needs_ten_seconds_before_the_end():
    with pytest.raises(StudioError, match="last chapter"):
        validate_chapters(GOOD, 80)


def test_timestamps():
    assert format_timestamp(0) == "0:00"
    assert format_timestamp(75.9) == "1:15"
    assert format_timestamp(3725) == "1:02:05"


def test_title_count_checked(tmp_path):
    with pytest.raises(StudioError, match="2 to 5 title"):
        write_publish_kit(tmp_path, titles=["one"], description="d", chapters=GOOD, tags=[], edited_duration=120)


def test_writes_the_kit(tmp_path):
    files = write_publish_kit(
        tmp_path, titles=["A", "B"], description="About.", chapters=GOOD,
        tags=["editing", "video"], edited_duration=120,
    )
    description = Path(files["description"]).read_text()
    assert description.endswith("0:00 Intro\n0:30 Setup\n1:15 Result\n")
    assert json.loads(Path(files["kit"]).read_text())["tags"] == ["editing", "video"]


def test_tool_uses_edited_duration(video):
    # The 20 s test video: chapters every 5 s break the 10 s rule.
    with pytest.raises(StudioError, match="at least 10s"):
        tools.save_publish_kit(
            str(video), ["A", "B"], "d",
            [{"time": 0, "label": "a"}, {"time": 5, "label": "b"}, {"time": 15, "label": "c"}], [],
        )
