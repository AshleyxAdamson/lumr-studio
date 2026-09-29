"""The join picture: size limits, returned fields, the no-cut case, and label layout."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import VIDEO_SECONDS
from lumr_studio import look
from lumr_studio.errors import StudioError
from lumr_studio.look import Label, axis_labels, layout_labels, plan_window, render_join_picture

CUT = (7.9, 10.8)


def _render(video: Path, words, tmp_path: Path, at: float, removed=(CUT,), **kwargs) -> dict:
    out = tmp_path / "pictures" / f"join-{at}.png"
    return render_join_picture(video, list(removed), words, VIDEO_SECONDS, at, out, **kwargs)


def _assert_limits(result: dict) -> Image.Image:
    path = Path(result["path"])
    assert path.exists()
    assert path.stat().st_size < look.MAX_FILE_BYTES
    image = Image.open(path)
    assert image.format == "PNG"
    assert image.size == (result["width"], result["height"])
    assert max(image.size) <= look.MAX_LONG_EDGE
    return image.convert("RGB")


def test_join_picture_fields_and_limits(video, words, tmp_path):
    result = _render(video, words, tmp_path, at=CUT[0])
    image = _assert_limits(result)
    assert result["join"] == [CUT[0], CUT[1]]
    assert result["edited_at"] == pytest.approx(CUT[0])
    assert result["frames"] == look.STRIP_FRAMES + 2  # the strip plus the seam pair
    assert result["seconds_shown"] == pytest.approx(6.0)
    # The join line runs down the centre, through the gap between the seam frames.
    seam_gap_y = look.MARGIN + look.LABEL_SIZE + 8 + look.SEAM_HEIGHT // 2
    assert image.getpixel((look.CANVAS_WIDTH // 2, seam_gap_y)) == look.JOIN


def test_at_inside_a_cut_picks_that_cut(video, words, tmp_path):
    result = _render(video, words, tmp_path, at=9.0)
    assert result["join"] == [CUT[0], CUT[1]]


def test_no_cut_nearby_draws_plain_source(video, words, tmp_path):
    joined = _render(video, words, tmp_path, at=CUT[0])
    result = _render(video, words, tmp_path, at=16.0)
    _assert_limits(result)
    assert result["join"] is None
    assert result["frames"] == look.STRIP_FRAMES  # no seam pair
    assert result["edited_at"] == pytest.approx(16.0 - (CUT[1] - CUT[0]))
    assert result["height"] < joined["height"]


def test_edge_of_video_cut_shows_one_side(video, words, tmp_path):
    result = _render(video, words, tmp_path, at=0.5, removed=[(0.0, 3.3)])
    _assert_limits(result)
    assert result["join"] == [0.0, 3.3]
    assert result["seconds_shown"] == pytest.approx(3.0)  # nothing plays before the start
    assert result["frames"] == look.STRIP_FRAMES // 2 + 1  # right half of the strip, and the after frame


def test_laugh_labels_become_blocks(video, words, tmp_path):
    labels = [{"start": 6.0, "end": 6.6, "seconds": 0.6, "kind": "laugh", "confidence": "likely",
               "punchline": [3.9, 6.0]}]
    result = _render(video, words, tmp_path, at=CUT[0], labels=labels)
    _assert_limits(result)
    window = plan_window([CUT], VIDEO_SECONDS, CUT[0], 3.0)
    kinds = {item.text: item.kind for item in axis_labels(words, window, 3.0, labels)}
    assert kinds["laugh 0.6s"] == "laugh"


def test_nothing_written_beside_the_source(video, words, tmp_path):
    before = sorted(p.name for p in video.parent.iterdir())
    _render(video, words, tmp_path, at=CUT[0])
    assert sorted(p.name for p in video.parent.iterdir()) == before
    with pytest.raises(StudioError, match="beside the source video"):
        render_join_picture(video, [CUT], words, VIDEO_SECONDS, CUT[0], video.parent / "join.png")


def test_bad_arguments_raise(video, words, tmp_path):
    with pytest.raises(StudioError, match="outside the video"):
        _render(video, words, tmp_path, at=VIDEO_SECONDS + 1)
    with pytest.raises(StudioError, match="span"):
        _render(video, words, tmp_path, at=5.0, span=0)
    with pytest.raises(StudioError, match="removes the whole video"):
        _render(video, words, tmp_path, at=5.0, removed=[(0.0, VIDEO_SECONDS)])


def test_ffmpeg_failure_says_what_to_check(words, tmp_path):
    fake = tmp_path / "footage" / "broken.mp4"
    fake.parent.mkdir()
    fake.write_bytes(b"not a video at all")
    with pytest.raises(StudioError, match="playable video"):
        render_join_picture(fake, [CUT], words, VIDEO_SECONDS, CUT[0], tmp_path / "out" / "join.png")


class SilentFootage(look.Footage):
    """Grey stills and no sound track, without running ffmpeg."""

    def has_audio(self, video):
        return False

    def frames(self, video, start, end, times):
        return [Image.new("RGB", (512, 288), (90, 90, 90)) for _ in times]

    def audio(self, video, start, end):
        raise AssertionError("audio must not be read from a video with no sound track")


def test_video_without_sound_draws_flat_wave(words, tmp_path):
    result = render_join_picture(
        tmp_path / "footage" / "talk.mp4", [CUT], words, VIDEO_SECONDS, CUT[0],
        tmp_path / "out" / "join.png", footage=SilentFootage(),
    )
    _assert_limits(result)
    assert result["frames"] == look.STRIP_FRAMES + 2


# ── Pure parts ────────────────────────────────────────────────────────────────


def test_plan_window_picks_the_nearest_cut_within_span():
    removed = [(2.0, 3.0), (8.0, 9.0)]
    assert plan_window(removed, 20.0, 7.0, 3.0).join == (8.0, 9.0)
    assert plan_window(removed, 20.0, 3.5, 3.0).join == (2.0, 3.0)
    far = plan_window(removed, 20.0, 15.0, 3.0)
    assert far.join is None and far.pieces[0].src_start == 12.0


def test_plan_window_pieces_play_in_edited_order():
    window = plan_window([(8.0, 9.0)], 20.0, 8.0, 3.0)
    assert [(p.src_start, p.src_end, p.at) for p in window.pieces] == [(5.0, 8.0, -3.0), (9.0, 12.0, 0.0)]


def _dense_labels(join_x: float) -> list[Label]:
    """Sixty short words in six seconds either side of a join: far more than fit in one row."""
    labels = []
    for i in range(60):
        x0 = look.AXIS_LEFT + i * (look.AXIS_RIGHT - look.AXIS_LEFT) / 60
        labels.append(Label(f"word{i}", x0, x0 + 20, "word", x0 + 20 <= join_x))
    return labels


def test_dense_labels_never_overlap_and_stay_off_the_join():
    join_x = look.CANVAS_WIDTH / 2
    items = _dense_labels(join_x)
    size, rows, dropped = layout_labels(items, join_x)
    placed = [item for item in items if item.row >= 0]
    assert len(placed) + dropped == len(items)
    assert rows > 1 and size < look.WORD_LAYOUTS[0][0]
    for a in placed:
        assert look.AXIS_LEFT <= a.left and a.left + a.width <= look.AXIS_RIGHT
        if a.left_of_join:
            assert a.left + a.width <= join_x - look.JOIN_CLEAR
        else:
            assert a.left >= join_x + look.JOIN_CLEAR
        for b in placed:
            if a is not b and a.row == b.row:
                assert a.left + a.width <= b.left or b.left + b.width <= a.left


def test_sparse_labels_use_one_row_at_full_size():
    items = [Label(w, 100 + 150 * i, 140 + 150 * i, "word", False) for i, w in enumerate("one two three".split())]
    assert layout_labels(items, None) == (look.WORD_LAYOUTS[0][0], 1, 0)


def test_join_levels_read_either_side_of_the_centre():
    audio = np.concatenate([np.zeros(800, np.float32), np.full(800, 0.5, np.float32)])
    before, after = look.join_levels(audio)
    assert before < -100 and after == pytest.approx(-6.0, abs=0.1)
