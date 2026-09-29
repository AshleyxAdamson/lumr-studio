"""set_overlays and get_overlays, as plain functions and as MCP tools."""

import asyncio
import json
from pathlib import Path

import pytest
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from PIL import Image, ImageStat

from lumr_studio import overlay_mcp, overlay_tools, tools
from lumr_studio.errors import StudioError
from lumr_studio.overlays import overlays_path
from lumr_studio.project import open_project
from lumr_studio.silences import no_silences
from test_overlays import make_clip, make_image

CUTS = [{"start": 7.9, "end": 10.8, "reason": "retake of the intro line"}]


@pytest.fixture
def media(tmp_path):
    folder = tmp_path / "creator-media"
    folder.mkdir()
    return {
        "photo": make_image(folder / "photo.jpg", (400, 300), "white"),
        "shot": make_image(folder / "screenshot.png", (300, 600), "green"),
        "clip": make_clip(folder / "broll.mp4", 3.0),
    }


def no_stills(*_a, **_k):
    raise AssertionError("pass picture=False or a fake still drawer")


def test_set_overlays_saves_and_says_where_each_lands(video, media):
    tools.set_edit(str(video), CUTS, silences=no_silences)
    result = overlay_tools.set_overlays(str(video), [
        {"file": str(media["photo"]), "start": 1.0, "end": 5.0, "note": "the clinic"},
        {"file": str(media["clip"]), "start": 12.0, "end": 14.5, "place": "top_right", "layer": 2},
    ], picture=False, stills=no_stills)
    assert [r["id"] for r in result["saved"]] == ["ov1", "ov2"]
    assert result["saved"][1]["on_screen"]["at"] == pytest.approx(7.1)  # the cut before it moved it up
    assert result["rejected"] == [] and "picture" not in result
    saved = json.loads(overlays_path(open_project(str(video))).read_text())
    assert saved["overlays"][0]["note"] == "the clinic" and saved["next_id"] == 3
    assert not (video.parent / "photo.jpg").exists()  # nothing copied anywhere


def test_the_picture_shows_each_overlay_as_the_viewer_sees_it(video, media):
    result = overlay_tools.set_overlays(str(video), [
        {"file": str(media["photo"]), "start": 1.0, "end": 5.0, "motion": "still"},
        {"file": str(media["shot"]), "start": 12.0, "end": 16.0, "place": "bottom_right"},
    ])
    picture = result["picture"]
    path = Path(picture["path"])
    assert path.parent.name == "looks" and picture["shows"] == ["ov1", "ov2"] and picture["left_out"] == 0
    with Image.open(path) as img:
        assert img.format == "JPEG"
        left = img.crop((100, 100, 300, 200)).convert("L")
        assert ImageStat.Stat(left).mean[0] > 200  # the white photo fills the first still


def test_a_still_shows_everything_on_screen_at_that_moment(video, media):
    result = overlay_tools.set_overlays(str(video), [
        {"file": str(media["photo"]), "start": 1.0, "end": 6.0, "motion": "still"},
        {"file": str(media["shot"]), "start": 3.0, "end": 8.0, "place": "top_left", "layer": 2},
    ])
    with Image.open(result["picture"]["path"]) as img:
        box_still = img.crop((640, 0, 1280, 480))
    # At the box's middle (5.5s) the white photo is still up, so it shows behind the box.
    corner = box_still.crop((440, 330, 620, 460)).convert("L")
    assert ImageStat.Stat(corner).mean[0] > 200


def test_the_picture_gets_the_real_times(video, media):
    seen = {}

    def fake(video_path, placements, source_at, out):
        seen["times"] = [source_at(p.at + p.seconds / 2) for p in placements]
        return {"path": None, "shows": [], "left_out": 0}

    tools.set_edit(str(video), CUTS, silences=no_silences)
    result = overlay_tools.set_overlays(
        str(video), [{"file": str(media["clip"]), "start": 12.0, "end": 14.0}], stills=fake
    )
    assert seen["times"] == [pytest.approx(13.0, abs=0.01)]  # middle of 12-14 in the source
    assert not any(open_project(str(video)).looks_dir.glob("*.jpg"))  # no empty picture left
    assert result["picture"]["path"] is None


def test_every_overlay_rejected_saves_nothing(video, media):
    overlay_tools.set_overlays(str(video), [{"file": str(media["photo"]), "start": 1.0, "end": 5.0}], picture=False)
    with pytest.raises(StudioError, match="Every overlay was rejected.*Overlay 0 has no file"):
        overlay_tools.set_overlays(str(video), [{"file": "/gone.jpg", "start": 1.0, "end": 5.0}], picture=False)
    assert [o["id"] for o in overlay_tools.get_overlays(str(video))["overlays"]] == ["ov1"]


def test_an_empty_list_clears_them(video, media):
    overlay_tools.set_overlays(str(video), [{"file": str(media["photo"]), "start": 1.0, "end": 5.0}], picture=False)
    result = overlay_tools.set_overlays(str(video), [], picture=False)
    assert result["cleared"] is True and overlay_tools.get_overlays(str(video))["overlays"] == []


def test_read_back_passes_straight_to_set_and_ids_hold(video, media):
    overlay_tools.set_overlays(str(video), [
        {"file": str(media["photo"]), "start": 1.0, "end": 5.0},
        {"file": str(media["clip"]), "start": 12.0, "end": 14.0},
    ], picture=False)
    rows = overlay_tools.get_overlays(str(video))["overlays"]
    rows[0]["start"] = 2.0
    new = {"file": str(media["shot"]), "start": 16.0, "end": 18.0}
    again = overlay_tools.set_overlays(str(video), [*rows, new], picture=False)
    assert not again["rejected"]
    assert [r["id"] for r in again["saved"]] == ["ov1", "ov2", "ov3"]
    dropped = overlay_tools.set_overlays(str(video), [rows[1]], picture=False)
    later = overlay_tools.set_overlays(str(video), [rows[1], new], picture=False)
    assert [r["id"] for r in dropped["saved"]] == ["ov2"]
    assert [r["id"] for r in later["saved"]] == ["ov2", "ov4"]  # ov1 and ov3 are never handed out again


def test_overlays_follow_their_words_when_the_edit_changes(video, media):
    """The creator moves the pace dial after placing a photo: it stays on its words."""
    overlay_tools.set_overlays(
        str(video), [{"file": str(media["photo"]), "start": 12.0, "end": 14.2}], picture=False
    )
    before = overlay_tools.get_overlays(str(video))["overlays"][0]
    assert before["on_screen"] == {"at": 12.0, "seconds": 2.2} and before["starts_on"]["word"] == "this"

    tools.set_edit(str(video), CUTS, silences=no_silences)
    after = overlay_tools.get_overlays(str(video))["overlays"][0]
    assert after["start"] == 12.0 and after["on_screen"]["at"] == pytest.approx(7.1)
    assert after["over"] == "this is the part that matters." and after["status"] == "shown"

    tools.set_edit(str(video), [{"start": 11.9, "end": 14.3, "reason": "drop the line"}], silences=no_silences)
    gone = overlay_tools.get_overlays(str(video))
    assert gone["not_shown"] == ["ov1"] and gone["overlays"][0]["status"] == "hidden"
    assert "is not shown" in gone["notes"][0]


def test_an_unreadable_edit_places_them_on_the_uncut_video(video, media):
    overlay_tools.set_overlays(str(video), [{"file": str(media["photo"]), "start": 1.0, "end": 5.0}], picture=False)
    open_project(str(video)).edit_path.write_text("{broken")
    result = overlay_tools.get_overlays(str(video))
    assert "edit can't be read" in result["notes"][0] and result["overlays"][0]["on_screen"]["at"] == 1.0


def test_overlays_never_touch_the_edit(video, media):
    tools.set_edit(str(video), CUTS, silences=no_silences)
    edit_before = open_project(str(video)).edit_path.read_text()
    overlay_tools.set_overlays(str(video), [{"file": str(media["photo"]), "start": 1.0, "end": 5.0}], picture=False)
    assert open_project(str(video)).edit_path.read_text() == edit_before


# ── As MCP tools ──────────────────────────────────────────────────────────────


def test_register_adds_two_annotated_tools_that_answer(video, media):
    server = MCPServer("test")
    overlay_mcp.register(server)

    async def talk():
        listed = {t.name: t for t in await server.list_tools()}
        saved = await server.call_tool("set_overlays", {
            "video_path": str(video),
            "overlays": [{"file": str(media["photo"]), "start": 1.0, "end": 5.0}],
        })
        read = await server.call_tool("get_overlays", {"video_path": str(video)})
        with pytest.raises(ToolError, match="relative"):
            await server.call_tool("get_overlays", {"video_path": "relative.mp4"})
        return listed, saved, read

    listed, saved, read = asyncio.run(talk())
    assert set(listed) == {"set_overlays", "get_overlays"}
    assert listed["set_overlays"].annotations.read_only_hint is False
    assert listed["get_overlays"].annotations.read_only_hint is True
    assert all(t.annotations.destructive_hint is False and t.annotations.title for t in listed.values())
    assert [c.type for c in saved.content] == ["image", "text"]
    assert saved.content[0].mime_type == "image/jpeg"
    assert json.loads(read.content[0].text)["overlays"][0]["id"] == "ov1"
