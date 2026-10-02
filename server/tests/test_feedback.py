"""Send feedback: the form's checks, the one request to the team's server, and what the page is told.

No test reaches the internet: ``urllib.request.urlopen`` is replaced, and only
the request to the pretend team server is answered by the stand-in.
"""

import json
import logging
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from lumr_studio import feedback, review_server, share, tools, treatment
from lumr_studio.errors import StudioError
from lumr_studio.project import open_project
from lumr_studio.silences import no_silences

from test_review_server import TREATMENT_CUTS, call, error_of, post_json, project, render, server  # noqa: F401

PLUGIN_DIR = Path(__file__).resolve().parents[2]
TEAM = "https://share.example.test"
ENDPOINT = TEAM + "/v1/feedback"
REAL_URLOPEN = urllib.request.urlopen


class Answer:
    """What ``urlopen`` returns: a status, and a close."""

    def __init__(self, status=200):
        self.status = status
        self.closed = False

    def close(self):
        self.closed = True


class TeamServer:
    """Stands in for ``urlopen``: records each request to the team's address; anything else goes where it was going."""

    def __init__(self, answer=None):
        self.answer = answer if answer is not None else Answer()
        self.sent = []

    def __call__(self, request, timeout=None, **kwargs):
        if getattr(request, "full_url", "").startswith(TEAM):
            self.sent.append((request, timeout))
            if isinstance(self.answer, BaseException):
                raise self.answer
            return self.answer
        return REAL_URLOPEN(request, timeout=timeout, **kwargs)

    @property
    def payload(self):
        return json.loads(self.sent[-1][0].data)


@pytest.fixture
def team(monkeypatch):
    monkeypatch.setenv("LUMR_SHARE_URL", TEAM)
    stand_in = TeamServer()
    monkeypatch.setattr(urllib.request, "urlopen", stand_in)
    return stand_in


# ── where feedback goes ───────────────────────────────────────────────────────


def test_the_team_server_is_the_default_and_the_environment_overrides_it(monkeypatch):
    monkeypatch.delenv("LUMR_SHARE_URL")
    assert share.DEFAULT_SHARE_URL == "https://feedback.lumr-studio.workers.dev"
    assert share.share_url() == "https://feedback.lumr-studio.workers.dev"
    monkeypatch.setenv("LUMR_SHARE_URL", "https://share.example.test/")
    assert share.share_url() == "https://share.example.test"
    monkeypatch.setenv("LUMR_SHARE_URL", "https://share.example.test///")
    assert share.share_url() == "https://share.example.test"


def test_an_empty_environment_value_turns_sharing_off(monkeypatch):
    for off in ("", "   "):
        monkeypatch.setenv("LUMR_SHARE_URL", off)
        assert share.share_url() is None


def test_the_module_constant_is_the_fallback_when_the_environment_says_nothing(monkeypatch):
    monkeypatch.delenv("LUMR_SHARE_URL")
    monkeypatch.setattr(share, "DEFAULT_SHARE_URL", "https://team.example.test/")
    assert share.share_url() == "https://team.example.test"
    monkeypatch.setenv("LUMR_SHARE_URL", "https://mine.example.test")
    assert share.share_url() == "https://mine.example.test"


# ── the plugin version ────────────────────────────────────────────────────────


def test_the_version_is_the_one_in_plugin_json():
    shipped = json.loads((PLUGIN_DIR / ".claude-plugin" / "plugin.json").read_text())["version"]
    assert feedback.plugin_version() == shipped
    assert feedback.plugin_root() == PLUGIN_DIR


def test_claude_code_names_the_plugin_folder_and_the_version_comes_from_there(monkeypatch, tmp_path):
    (tmp_path / ".claude-plugin").mkdir()
    (tmp_path / ".claude-plugin" / "plugin.json").write_text('{"version": "9.8.7"}')
    monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path))
    assert feedback.plugin_version() == "9.8.7"


@pytest.mark.parametrize("text", ["{not json", "[1]", '{"version": 3}', '{"version": ""}', "{}"])
def test_a_version_that_cannot_be_read_is_none(monkeypatch, tmp_path, text):
    (tmp_path / ".claude-plugin").mkdir()
    (tmp_path / ".claude-plugin" / "plugin.json").write_text(text)
    monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path))
    assert feedback.plugin_version() is None


def test_with_no_plugin_folder_the_version_is_none(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path / "nowhere"))
    assert feedback.plugin_version() is None


# ── what the form may send ────────────────────────────────────────────────────


def test_a_message_alone_is_enough_and_extra_fields_are_dropped():
    assert feedback.clean({"message": "  Loved it.  ", "extra": "x", "schema": 9}) == {
        "message": "Loved it.", "name": None, "email": None,
    }


def test_a_name_and_an_email_are_kept_when_given_and_empty_ones_are_none():
    assert feedback.clean({"message": "hi", "name": " Sam ", "email": "sam@example.com"}) == {
        "message": "hi", "name": "Sam", "email": "sam@example.com",
    }
    assert feedback.clean({"message": "hi", "name": "  ", "email": ""}) == {"message": "hi", "name": None, "email": None}
    assert feedback.clean({"message": "hi", "name": None, "email": None})["name"] is None


@pytest.mark.parametrize("body, why", [
    ({}, "Write a few words first."),
    ({"message": ""}, "Write a few words first."),
    ({"message": "  \n "}, "Write a few words first."),
    ({"message": 5}, "Write a few words first."),
    ({"message": "x" * 5001}, "over 5000 characters"),
    ({"message": "hi", "name": "n" * 101}, "name is over 100 characters"),
    ({"message": "hi", "name": 7}, "name must be text"),
    ({"message": "hi", "email": "e" * 201}, "over 200 characters"),
    ({"message": "hi", "email": 7}, "must be text"),
    ({"message": "hi", "email": "not an email"}, "doesn't look right"),
    ({"message": "hi", "email": "a@b"}, "doesn't look right"),
    ({"message": "hi", "email": "a b@c.com"}, "doesn't look right"),
])
def test_the_form_is_checked_and_the_answer_is_plain(body, why):
    with pytest.raises(StudioError, match=why):
        feedback.clean(body)


def test_the_limits_are_inclusive():
    assert feedback.clean({"message": "x" * 5000, "name": "n" * 100, "email": "a@" + "b" * 190 + ".co"})["message"] == "x" * 5000


# ── the one request ───────────────────────────────────────────────────────────


def test_the_exact_json_goes_to_the_teams_server_once(team):
    got = feedback.send_feedback({"message": "Nice page.", "name": "Sam", "email": "sam@example.com", "ignored": 1})
    (request, timeout), = team.sent
    version = json.loads((PLUGIN_DIR / ".claude-plugin" / "plugin.json").read_text())["version"]
    assert request.full_url == ENDPOINT and request.get_method() == "POST"
    assert timeout == 15
    assert request.get_header("Content-type") == "application/json"
    assert request.get_header("User-agent") == f"lumr-studio/{version}"
    payload = team.payload
    assert set(payload) == {"schema", "feedback_id", "plugin_version", "message", "name", "email"}
    assert payload["schema"] == 1 and payload["plugin_version"] == version
    assert (payload["message"], payload["name"], payload["email"]) == ("Nice page.", "Sam", "sam@example.com")
    assert got == {"feedback_id": payload["feedback_id"]}
    assert len(payload["feedback_id"]) == 22 and payload["feedback_id"].replace("-", "").replace("_", "").isalnum()
    assert team.answer.closed


def test_a_message_alone_sends_null_for_name_and_email(team):
    feedback.send_feedback({"message": "Just this."})
    assert team.payload["name"] is None and team.payload["email"] is None


def test_each_send_has_its_own_id(team):
    a = feedback.send_feedback({"message": "one"})["feedback_id"]
    b = feedback.send_feedback({"message": "two"})["feedback_id"]
    assert a != b and len(team.sent) == 2


def test_a_version_that_cannot_be_read_goes_as_zero(team, monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path))
    feedback.send_feedback({"message": "hi"})
    assert team.payload["plugin_version"] == "0.0.0"
    assert team.sent[0][0].get_header("User-agent") == "lumr-studio/0.0.0"


def test_unicode_goes_through_as_it_was_typed(team):
    feedback.send_feedback({"message": "Café, 日本語, and a tab\tand a line\nbreak"})
    assert team.payload["message"] == "Café, 日本語, and a tab\tand a line\nbreak"


@pytest.mark.parametrize("status", [199, 300, 302, 400, 404, 500, 503])
def test_an_answer_that_is_not_a_success_is_a_plain_error(team, status):
    team.answer = Answer(status)
    with pytest.raises(StudioError, match=rf"answered with an error \({status}\)\. Try again later\."):
        feedback.send_feedback({"message": "hi"})
    assert len(team.sent) == 1, "no retry"


@pytest.mark.parametrize("status", [200, 201, 202, 204])
def test_any_2xx_is_a_success(team, status):
    team.answer = Answer(status)
    assert "feedback_id" in feedback.send_feedback({"message": "hi"})


def test_an_http_error_raised_by_urlopen_is_the_same_plain_error(team):
    team.answer = urllib.error.HTTPError(ENDPOINT, 422, "no", {}, None)
    with pytest.raises(StudioError, match=r"answered with an error \(422\)"):
        feedback.send_feedback({"message": "hi"})
    assert len(team.sent) == 1


@pytest.mark.parametrize("failure", [
    urllib.error.URLError("no route"), TimeoutError("timed out"), ConnectionResetError("reset"), OSError("down"),
])
def test_a_server_that_cannot_be_reached_is_a_plain_error_with_no_retry(team, failure):
    team.answer = failure
    with pytest.raises(StudioError, match="Couldn't reach the feedback server. Check your connection and try again."):
        feedback.send_feedback({"message": "hi"})
    assert len(team.sent) == 1


def test_a_bad_address_is_the_same_plain_error(monkeypatch):
    monkeypatch.setenv("LUMR_SHARE_URL", "not a url")
    with pytest.raises(StudioError, match="Couldn't reach"):
        feedback.send_feedback({"message": "hi"})


def test_with_no_team_server_nothing_is_sent(monkeypatch):
    calls = []
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: calls.append(a))
    with pytest.raises(StudioError, match="can't be sent from here yet"):
        feedback.send_feedback({"message": "hi"})
    assert calls == []


def test_a_message_that_fails_its_checks_is_never_sent(team):
    with pytest.raises(StudioError):
        feedback.send_feedback({"message": ""})
    assert team.sent == []


def test_what_the_creator_wrote_is_never_logged(team, caplog):
    team.answer = Answer(500)
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(StudioError):
            feedback.send_feedback({"message": "a private sentence", "name": "Sam Private", "email": "sam@private.test"})
    assert "private" not in caplog.text.lower()


def test_the_words_the_creator_can_read_are_plain():
    from creator_words import banned_in, decimal_times_in

    lines = [
        "Write a few words first.", "Feedback can't be sent from here yet. Use the GitHub link the page offers instead.",
        "The feedback server answered with an error (500). Try again later.",
        "Couldn't reach the feedback server. Check your connection and try again.",
        "The message is over 5000 characters. Shorten it a bit.", "That email address doesn't look right. Check it, or leave it empty.",
        "The name must be text.", "The name is over 100 characters. Shorten it a bit.",
    ]
    for line in lines:
        assert not banned_in(line) and not decimal_times_in(line) and "—" not in line, line


# ── what the page is told ─────────────────────────────────────────────────────


def edited(video):
    tools.set_edit(str(video), [{"start": 3.3, "end": 3.8, "reason": "filler", "kind": "other"}], silences=no_silences,
                   labels=tools.unmeasured_labels)
    return treatment.load_context(open_project(str(video)), duration=20.0, silences=no_silences)


def test_the_page_state_carries_the_version_and_that_feedback_is_not_direct_yet(video):
    state = treatment.page_state(edited(video))
    assert state["version"] == json.loads((PLUGIN_DIR / ".claude-plugin" / "plugin.json").read_text())["version"]
    assert state["feedback"] == {"direct": False}


def test_with_a_team_server_the_page_state_says_direct(video, monkeypatch):
    monkeypatch.setenv("LUMR_SHARE_URL", TEAM)
    assert treatment.page_state(edited(video))["feedback"] == {"direct": True}


def test_a_version_that_cannot_be_read_is_null_on_the_page(video, monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path))
    assert treatment.page_state(edited(video))["version"] is None


# ── the route ─────────────────────────────────────────────────────────────────


@pytest.fixture
def treated(server, video):  # noqa: F811
    tools.set_edit(str(video), TREATMENT_CUTS, auto_tighten=True, silences=no_silences, labels=tools.unmeasured_labels)
    return server.url_for(open_project(str(video)))


def test_the_route_sends_the_message_and_answers_the_id(treated, team):
    status, _h, body = post_json(treated + "api/feedback", {"message": "Works well.", "name": "", "email": "sam@example.com"})
    got = json.loads(body)
    assert status == 200 and set(got) == {"feedback_id"}
    assert team.payload["message"] == "Works well." and team.payload["name"] is None
    assert team.payload["email"] == "sam@example.com" and team.payload["feedback_id"] == got["feedback_id"]


def test_the_route_validates_before_it_sends(treated, team):
    for body, why in (({}, "Write a few words first."), ({"message": "x" * 5001}, "over 5000"),
                      ({"message": "hi", "email": "nope"}, "doesn't look right")):
        status, _h, raw = post_json(treated + "api/feedback", body)
        assert status == 400 and why in error_of(raw), body
    assert team.sent == []


def test_a_refusal_from_the_team_is_a_plain_400_on_the_page(treated, team):
    team.answer = Answer(503)
    status, _h, raw = post_json(treated + "api/feedback", {"message": "hi"})
    assert status == 400 and "answered with an error (503)" in error_of(raw)


def test_with_no_team_server_the_route_refuses(treated):
    status, _h, raw = post_json(treated + "api/feedback", {"message": "hi"})
    assert status == 400 and "can't be sent from here yet" in error_of(raw)


def test_the_route_is_for_this_pages_own_origin_and_json_only(treated, team):
    status, _h, raw = post_json(treated + "api/feedback", {"message": "hi"}, Origin="http://evil.example")
    assert status == 403 and "other sites" in error_of(raw)
    status, _h, _b = call(treated + "api/feedback", method="POST", body=b"{}", headers={"Content-Type": "text/plain"})
    assert status == 415
    assert team.sent == []


def test_the_route_needs_a_live_review_link(server, team):
    status, _h, _b = post_json(f"http://127.0.0.1:{server.port}/nope/api/feedback", {"message": "hi"})
    assert status == 404 and team.sent == []


def test_feedback_is_not_one_of_the_edits_changes():
    assert review_server.FEEDBACK_ROUTE == "api/feedback"
    assert review_server.FEEDBACK_ROUTE not in review_server.TREATMENT_ACTIONS
