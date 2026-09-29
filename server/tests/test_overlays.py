"""Checking, saving and placing overlays: every field, ids, layers, and the anchor rules.

Media files are made here: small pictures with Pillow, small clips with
ffmpeg's test sources. The other overlay tests import ``make_image`` and
``make_clip`` from this file.
"""

import json
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from lumr_studio.errors import StudioError
from lumr_studio.overlays import (
    HIDDEN,
    IMAGE,
    MAX_OVERLAYS,
    MIN_SCREEN_SECONDS,
    SHORTENED,
    SHOWN,
    VIDEO,
    Media,
    OverlayRejected,
    anchor_words,
    build_overlays,
    load_overlays,
    overlays_path,
    page_row,
    parse_overlay,
    place_on_edit,
    place_overlay,
    probe_media,
    quote,
    recheck_files,
    save_overlays,
)
from lumr_studio.project import open_project

DURATION = 20.0
FRAME = (320, 240)


def make_image(path: Path, size=(400, 300), color="red", exif_orientation: int | None = None) -> Path:
    """A plain picture with a blue corner, so a crop or a turn is visible."""
    img = Image.new("RGB", size, color)
    img.paste("blue", (0, 0, size[0] // 4, size[1] // 4))
    kwargs = {}
    if exif_orientation:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        kwargs["exif"] = exif
    img.save(path, **kwargs)
    return path


def make_clip(path: Path, seconds=4.0, size="160x120", sound=True, rotate: int | None = None) -> Path:
    """A test-pattern clip, with a tone when ``sound``."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"mandelbrot=size={size}:rate=25"]
    if sound:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=880:sample_rate=48000"]
    cmd += ["-t", str(seconds), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if sound:
        cmd += ["-c:a", "aac"]
    cmd.append(str(path))
    subprocess.run(cmd, check=True)
    if rotate is not None:  # display rotation is an input option: remux to write it into the file
        turned = path.with_name(f"turned-{path.name}")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-display_rotation", str(rotate), "-i", str(path),
             "-c", "copy", str(turned)],
            check=True,
        )
        turned.replace(path)
    return path


@pytest.fixture(scope="module")
def media(tmp_path_factory) -> dict[str, Path]:
    folder = tmp_path_factory.mktemp("media")
    return {
        "photo": make_image(folder / "photo.jpg", (400, 300)),
        "tall": make_image(folder / "tall.png", (300, 600)),
        "wide": make_image(folder / "wide.png", (640, 480)),
        "clip": make_clip(folder / "clip.mp4", 4.0),
        "silent": make_clip(folder / "silent.mp4", 3.0, sound=False),
        "text": _write(folder / "notes.jpg", "not a picture"),
        "folder": folder,
    }


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def parse(raw, **kw):
    return parse_overlay(raw, duration=DURATION, frame=FRAME, **kw)


def ov(media, key="photo", **fields):
    return {"file": str(media[key]), "start": 2.0, "end": 6.0, **fields}


# ── Probing ───────────────────────────────────────────────────────────────────


def test_probe_reads_kind_size_length_and_sound(media):
    photo = probe_media(media["photo"])
    assert (photo.kind, photo.width, photo.height, photo.duration) == (IMAGE, 400, 300, None)
    clip = probe_media(media["clip"])
    assert clip.kind == VIDEO and (clip.width, clip.height) == (160, 120)
    assert clip.duration == pytest.approx(4.0, abs=0.1) and clip.has_sound
    assert not probe_media(media["silent"]).has_sound


def test_probe_ignores_the_extension(media, tmp_path):
    disguised = tmp_path / "really-a-png.mp4"
    disguised.write_bytes(media["tall"].read_bytes())
    assert probe_media(disguised).kind == IMAGE
    with pytest.raises(StudioError, match="ffprobe|no picture"):
        probe_media(media["text"])


def test_probe_reports_the_size_as_displayed(tmp_path):
    turned = probe_media(make_image(tmp_path / "phone.jpg", (400, 200), exif_orientation=6))
    assert (turned.width, turned.height) == (200, 400)
    clip = probe_media(make_clip(tmp_path / "upright.mp4", 1.0, size="320x180", rotate=90))
    assert (clip.width, clip.height) == (180, 320)


# ── Every field ───────────────────────────────────────────────────────────────


def test_a_photo_gets_its_defaults(media):
    o = parse(ov(media))
    assert o["kind"] == IMAGE and o["place"] == "full" and o["fit"] == "fill"
    assert o["motion"] == "zoom_in" and o["layer"] == 1 and o["fade"] is True
    assert o["size"] is None and o["clip_in"] is None and o["volume"] is None
    assert o["media"] == {"width": 400, "height": 300, "duration": None, "has_sound": False}


def test_a_very_different_shape_is_shown_whole_by_default(media):
    assert parse(ov(media, "tall"))["fit"] == "fit"
    assert parse(ov(media, "tall", fit="fill"))["fit"] == "fill"


def test_a_clip_gets_its_defaults(media):
    o = parse(ov(media, "clip"))
    assert o["kind"] == VIDEO and o["clip_in"] == 0 and o["clip_out"] == pytest.approx(4.0, abs=0.1)
    assert o["volume"] == 0 and o["motion"] is None


@pytest.mark.parametrize(
    "fields, match",
    [
        ({"file": None}, "has no file"),
        ({"file": "photo.jpg"}, "relative"),
        ({"file": "/nowhere/photo.jpg"}, "no file at"),
        ({"start": "2"}, "numbers of seconds"),
        ({"start": -1.0}, "before 0"),
        ({"end": 25.0}, "past the end of the video"),
        ({"start": 5.0, "end": 5.5}, "at least 1s"),
        ({"place": "top"}, "place 'top' is not one of"),
        ({"place": "top_right", "size": 0.9}, "size 0.9 must be a number from 0.2 to 0.6"),
        ({"size": 0.3}, "size only applies to a box"),
        ({"place": "top_right", "fit": "fill"}, "fit only applies to full-frame"),
        ({"fit": "stretch"}, "fit 'stretch' is not one of"),
        ({"motion": "spin"}, "motion 'spin' is not one of"),
        ({"clip_in": 1.0}, "clip_in only applies to video clips"),
        ({"volume": 0.2}, "volume only applies to video clips"),
        ({"layer": 0}, "layer 0 must be a whole number from 1 to 9"),
        ({"layer": 1.5}, "whole number"),
        ({"fade": "yes"}, "fade must be true or false"),
        ({"note": "x" * 300}, "at most 200"),
        ({"id": "Photo One"}, "id 'Photo One' must be"),
        ({"position": "top_right"}, "unknown field 'position'"),
    ],
)
def test_photo_fields_are_checked(media, fields, match):
    with pytest.raises(OverlayRejected, match=match):
        parse(ov(media, **fields))


@pytest.mark.parametrize(
    "key, fields, match",
    [
        ("clip", {"motion": "still"}, "motion only applies to photos"),
        ("clip", {"clip_in": -1.0}, "clip_in -1.0 must be a number"),
        ("clip", {"clip_out": 9.0}, "past the end of clip.mp4"),
        ("clip", {"clip_in": 2.0, "clip_out": 2.5}, "Play at least 1s"),
        ("clip", {"volume": 0.8}, "volume 0.8 must be a number from 0 to 0.5"),
        ("silent", {"volume": 0.2}, "has no sound. Set volume to 0"),
        ("text", {}, "notes.jpg"),
        ("folder", {}, "is a folder"),
    ],
)
def test_clip_and_file_fields_are_checked(media, key, fields, match):
    with pytest.raises(OverlayRejected, match=match):
        parse(ov(media, key, **fields))


def test_tiny_and_huge_pictures_are_rejected(media, tmp_path):
    with pytest.raises(OverlayRejected, match="too small"):
        parse(ov({"photo": make_image(tmp_path / "dot.png", (8, 8))}))
    huge = Media(IMAGE, 12_000, 10_000, None, False)
    with pytest.raises(OverlayRejected, match="megapixels"):
        parse(ov(media), probe=lambda _p: huge)


def test_an_end_just_past_the_video_is_the_end(media):
    assert parse(ov(media, end=DURATION + 0.03))["end"] == DURATION


def test_read_back_fields_are_accepted_and_ignored(media):
    first = parse(ov(media))
    again = parse({**first, "clock": "0:02-0:06", "status": "shown", "notes": [], "over": "hi"})
    assert again["start"] == first["start"] and again["place"] == first["place"]


# ── The whole list: ids and layers ────────────────────────────────────────────


def build(raws, **kw):
    return build_overlays(raws, duration=DURATION, frame=FRAME, **kw)


def test_new_overlays_get_ids_and_given_ids_are_kept(media):
    out = build([ov(media), ov(media, start=8.0, end=10.0, id="ov7"), ov(media, start=12.0, end=14.0)])
    # A new id is always past every id seen, so ov7 given means new ones start at ov8.
    assert [o["id"] for o in out.overlays] == ["ov8", "ov7", "ov9"]
    assert out.next_id == 10


def test_ids_are_never_reused(media):
    out = build([ov(media)], next_id=5)
    assert out.overlays[0]["id"] == "ov5" and out.next_id == 6


def test_a_repeated_id_is_rejected(media):
    out = build([ov(media, id="a"), ov(media, start=8.0, end=10.0, id="a")])
    assert len(out.overlays) == 1
    assert "repeats id 'a'" in out.rejected[0]["why"]


def test_overlap_on_one_layer_is_rejected_with_a_fix(media):
    out = build([ov(media), ov(media, "clip", start=4.0, end=8.0)])
    assert [o["id"] for o in out.overlays] == ["ov1"]
    why = out.rejected[0]["why"]
    assert why.startswith("Overlay 1 overlaps ov1 (0:02-0:06)") and "higher layer" in why


def test_overlap_on_different_layers_is_allowed(media):
    out = build([ov(media), ov(media, "clip", start=4.0, end=8.0, place="top_right", layer=2)])
    assert not out.rejected and len(out.overlays) == 2


def test_touching_spans_do_not_clash(media):
    assert not build([ov(media), ov(media, start=6.0, end=9.0)]).rejected


def test_one_bad_overlay_does_not_sink_the_rest(media):
    out = build([ov(media, place="middle"), ov(media, start=8.0, end=10.0)])
    assert len(out.overlays) == 1 and out.rejected[0]["index"] == 0


def test_too_many_overlays(media):
    with pytest.raises(StudioError, match=f"more than {MAX_OVERLAYS}"):
        build([ov(media)] * (MAX_OVERLAYS + 1))


# ── Words under an overlay ────────────────────────────────────────────────────


def test_anchor_words_use_the_midpoint_rule(words):
    # "today"(3.90-4.30) starts it; "editing"(5.35-6.00) ends it; the event is skipped.
    anchors = anchor_words(3.8, 6.7, words)
    assert anchors["starts_on"] == {"word": "today", "start": 3.9}
    assert anchors["ends_on"] == {"word": "editing", "start": 5.35}
    assert anchor_words(18.0, 19.5, words) == {"starts_on": None, "ends_on": None}


def test_quote_leaves_out_cut_words(words):
    assert quote(3.8, 10.8, words, removed=[(7.9, 10.8)]) == "today we talk about editing"


# ── Placing on the edited video: the hard cases ───────────────────────────────


def saved(media_kind=IMAGE, start=2.0, end=6.0, **extra):
    o = {"id": "ov1", "kind": media_kind, "start": start, "end": end, "layer": 1,
         "file": "/x.png", "starts_on": None}
    if media_kind == VIDEO:
        o.update(clip_in=0.0, clip_out=4.0)
    return {**o, **extra}


WHOLE = [(0.0, DURATION)]


def test_no_cuts_places_it_where_it_was_set(words):
    p = place_overlay(saved(), WHOLE, words)
    assert (p.at, p.seconds, p.status, p.notes) == (2.0, 4.0, SHOWN, ())


def test_cuts_before_it_move_it_earlier(words):
    p = place_overlay(saved(start=12.0, end=15.0), [(0.0, 3.0), (8.0, DURATION)], words)
    assert p.at == pytest.approx(7.0) and p.seconds == pytest.approx(3.0) and p.status == SHOWN


def test_a_start_inside_a_cut_waits_for_the_talk_to_resume(words):
    p = place_overlay(saved(start=3.0, end=9.0), [(0.0, 2.5), (3.85, DURATION)], words)
    assert p.at == pytest.approx(2.5) and p.seconds == pytest.approx(5.15)
    assert "inside a cut, so it appears when the talk resumes at 0:03 on 'today'" in p.notes[0]


def test_a_cut_inside_the_span_shortens_it_and_says_it_hides_the_jump(words):
    p = place_overlay(saved(start=2.0, end=12.0), [(0.0, 6.9), (11.9, DURATION)], words)
    assert p.seconds == pytest.approx(5.0) and p.status == SHORTENED
    assert "1 cut falls under it" in p.notes[0] and "hides the jump" in p.notes[0]


def test_an_overlay_whose_words_are_all_cut_is_hidden(words):
    p = place_overlay(saved(start=8.0, end=10.5), [(0.0, 7.9), (10.8, DURATION)], words)
    assert p.status == HIDDEN and not p.shown and p.seconds == 0
    assert "is not shown" in p.notes[0] and "Move it" in p.notes[0]


def test_less_than_the_minimum_left_is_hidden(words):
    left = MIN_SCREEN_SECONDS / 2
    p = place_overlay(saved(start=8.0, end=10.0), [(0.0, 7.0), (10.0 - left, DURATION)], words)
    assert p.status == HIDDEN and f"only {left:.1f}s" in p.notes[0]


def test_a_clip_shorter_than_its_span_ends_early_and_says_where(words):
    p = place_overlay(saved(VIDEO, start=2.0, end=9.0), WHOLE, words)
    assert p.seconds == pytest.approx(4.0) and p.status == SHORTENED
    assert "talk shows again from 0:06" in p.notes[0] and "end the overlay at 6.00" in p.notes[0]


def test_a_span_running_into_the_cut_tail_ends_with_the_video(words):
    p = place_overlay(saved(start=15.0, end=20.0), [(0.0, 17.0)], words)
    assert p.at == pytest.approx(15.0) and p.seconds == pytest.approx(2.0)


def test_a_changed_edit_moves_it_but_it_keeps_its_words(words):
    """The pace dial changes every trim; the overlay follows its words."""
    o = saved(start=12.0, end=14.2)
    natural = place_overlay(o, [(0.0, 7.9), (10.8, DURATION)], words)
    fast = place_overlay(o, [(0.0, 3.3), (3.8, 7.9), (10.8, DURATION)], words)
    assert natural.at == pytest.approx(9.1) and fast.at == pytest.approx(8.6)
    assert natural.seconds == fast.seconds == pytest.approx(2.2)


def test_a_moved_transcript_is_flagged(words):
    o = saved(start=12.0, end=14.2, starts_on={"word": "this", "start": 12.0})
    assert not place_overlay(o, WHOLE, words).notes
    moved = saved(start=12.0, end=14.2, starts_on={"word": "that", "start": 12.0})
    assert "transcript changed" in place_overlay(moved, WHOLE, words).notes[0]


def test_drawing_order_is_bottom_layer_first_then_time(words):
    overlays = [
        saved(id="top", layer=2, start=3.0, end=5.0),
        saved(id="late", start=8.0, end=9.5),
        saved(id="early", start=1.0, end=6.0),
    ]
    order = [p.overlay["id"] for p in place_on_edit(overlays, WHOLE, words)]
    assert order == ["early", "late", "top"]


# ── As the treatment page shows them ──────────────────────────────────────────


def test_the_page_gets_the_name_the_place_in_time_and_a_tag_but_never_the_folder(words):
    photo = saved(start=12.0, end=15.0, file="/somewhere/private/berlin.jpg")
    row = page_row(place_overlay(photo, WHOLE, words))
    assert row == {
        "id": "ov1", "name": "berlin.jpg", "kind": "photo", "start": 12.0, "end": 15.0,
        "clock": "0:12", "status": SHOWN, "tag": "Photo: berlin.jpg",
    }
    assert "somewhere" not in json.dumps(row)


def test_the_page_calls_a_video_file_a_clip(words):
    row = page_row(place_overlay(saved(VIDEO, start=2.0, end=9.0, file="/x/walk.mp4"), WHOLE, words))
    assert (row["kind"], row["tag"], row["status"]) == ("clip", "Clip: walk.mp4", SHORTENED)


def test_the_page_row_keeps_source_times_when_a_cut_hides_it(words):
    row = page_row(place_overlay(saved(start=8.0, end=10.5), [(0.0, 7.9), (10.8, DURATION)], words))
    assert (row["status"], row["start"], row["end"], row["clock"]) == (HIDDEN, 8.0, 10.5, "0:08")


# ── Saving and reading back ───────────────────────────────────────────────────


def test_save_and_load_round_trip(video, media, words):
    project = open_project(str(video))
    assert load_overlays(project)["overlays"] == []
    out = build([ov(media)], words=words)
    save_overlays(project, out, duration=DURATION, requested=[ov(media)])
    data = load_overlays(project)
    assert data["overlays"][0]["id"] == "ov1" and data["next_id"] == 2
    assert overlays_path(project).parent == project.root
    assert not (project.root / "edit.json").exists()  # overlays never touch the edit


def test_an_unusable_file_says_how_to_fix_it(video):
    project = open_project(str(video))
    overlays_path(project).write_text("{not json")
    with pytest.raises(StudioError, match="not valid JSON. Call set_overlays"):
        load_overlays(project)
    overlays_path(project).write_text(json.dumps({"overlays": [{"id": "ov1"}]}))
    with pytest.raises(StudioError, match="bad overlay at index 0"):
        load_overlays(project)


def test_recheck_catches_a_moved_or_changed_file(media, tmp_path):
    photo = make_image(tmp_path / "moving.png", (400, 300))
    overlay = parse(ov({"photo": photo}))
    overlay["id"] = "ov1"
    recheck_files([overlay])
    make_image(photo, (500, 300))
    with pytest.raises(StudioError, match="ov1's file moving.png changed"):
        recheck_files([overlay])
    photo.unlink()
    with pytest.raises(StudioError, match="Overlay ov1 has no file at .* call set_overlays without ov1"):
        recheck_files([overlay])
