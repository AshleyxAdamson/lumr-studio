"""The review page's two share routes: preview what would be sent, then send exactly that.

Built on the changed video and the pretend team server from ``test_shapes`` and ``test_feedback``.
"""

import json

import pytest

from lumr_studio import shapes, treatment
from lumr_studio.project import open_project
from test_feedback import TEAM, Answer, team  # noqa: F401
from test_review_server import call, error_of, post_json, render, server  # noqa: F401
from test_shapes import changed, ctx_for, span_of


@pytest.fixture
def sharing(monkeypatch):
    monkeypatch.setenv("LUMR_SHARE_URL", TEAM)


@pytest.fixture
def page(video, server, sharing):
    """A review server on a video the creator changed, and the address of its page."""
    ctx = changed(video)
    return server.url_for(ctx.project)


def test_the_page_state_says_sharing_is_on_only_when_there_is_somewhere_to_send(video, server, monkeypatch):
    ctx = changed(video)
    base = server.url_for(ctx.project)
    assert json.loads(call(base + "api/treatment")[2])["share"] == {"on": False}
    monkeypatch.setenv("LUMR_SHARE_URL", TEAM)
    assert json.loads(call(base + "api/treatment")[2])["share"] == {"on": True}


def test_preview_then_send_posts_exactly_what_was_previewed(page, team, video):
    status, _, body = post_json(page + "api/share/preview", {})
    assert status == 200
    preview = json.loads(body)
    assert set(preview) == {"lines", "send"} and len(preview["lines"]) == len(preview["send"]["records"])
    assert shapes.validate_send(preview["send"]) is None
    assert shapes.describe(preview["send"]) == preview["lines"]
    status, _, body = post_json(page + "api/share/send", {"send_id": preview["send"]["send_id"], "removed": [1]})
    assert status == 200, body
    n = len(preview["send"]["records"]) - 1
    assert json.loads(body) == {"send_id": preview["send"]["send_id"], "records": n}
    expected = {**preview["send"], "records": [r for i, r in enumerate(preview["send"]["records"]) if i != 1]}
    assert team.sent[-1][0].data == json.dumps(expected, sort_keys=True).encode("utf-8")
    line = json.loads((open_project(str(video)).root / "shares.jsonl").read_text())
    assert line["send_id"] == preview["send"]["send_id"]


def test_a_preview_is_sent_once_and_the_edit_changing_after_it_changes_nothing_sent(page, team, video):
    preview = json.loads(post_json(page + "api/share/preview", {})[2])
    treatment.add_cut(ctx_for(video), span_of(0, 1))
    status, _, body = post_json(page + "api/share/send", {"send_id": preview["send"]["send_id"], "removed": []})
    assert status == 200
    assert json.loads(team.sent[-1][0].data) == preview["send"], "what was previewed, not a rebuilt one"
    status, _, body = post_json(page + "api/share/send", {"send_id": preview["send"]["send_id"], "removed": []})
    assert status == 400 and "preview" in error_of(body).lower() and len(team.sent) == 1


def test_a_send_with_an_unknown_id_or_no_preview_is_an_error(page, team):
    status, _, body = post_json(page + "api/share/send", {"send_id": "A" * 22, "removed": []})
    assert status == 400 and "Open the preview again" in error_of(body)
    preview = json.loads(post_json(page + "api/share/preview", {})[2])
    status, _, body = post_json(page + "api/share/send", {"send_id": "B" * 22, "removed": []})
    assert status == 400 and "Open the preview again" in error_of(body)
    for bad in ({}, {"send_id": 5, "removed": []}, {"send_id": preview["send"]["send_id"], "removed": "0"},
                {"send_id": preview["send"]["send_id"], "removed": [], "records": []}):
        status, _, body = post_json(page + "api/share/send", bad)
        assert status == 400, bad
    assert team.sent == []


def test_a_failed_send_keeps_the_preview_so_the_creator_can_try_again(page, team):
    preview = json.loads(post_json(page + "api/share/preview", {})[2])
    team.answer = Answer(503)
    status, _, body = post_json(page + "api/share/send", {"send_id": preview["send"]["send_id"], "removed": []})
    assert status == 400 and "Try again later" in error_of(body)
    team.answer = Answer(201)
    status, _, _ = post_json(page + "api/share/send", {"send_id": preview["send"]["send_id"], "removed": []})
    assert status == 200 and len(team.sent) == 2


def test_the_routes_do_nothing_when_sharing_is_off(video, server):
    ctx = changed(video)
    base = server.url_for(ctx.project)
    for route, body in (("api/share/preview", {}), ("api/share/send", {"send_id": "A" * 22, "removed": []})):
        status, _, answer = post_json(base + route, body)
        assert status == 400 and "isn't set up" in error_of(answer)


def test_the_routes_answer_posts_from_this_page_only(page):
    assert call(page + "api/share/preview")[0] == 404
    assert post_json(page + "api/share/preview", {}, Origin="http://evil.example")[0] == 403
    assert post_json(page + "api/share/preview", {"x": 1})[0] == 400
