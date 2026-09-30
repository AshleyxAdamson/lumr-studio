"""The treatment page's local server, exercised over real HTTP on a free port."""

import base64
import http.client
import json
import shutil
import urllib.error
import urllib.request

import pytest

from lumr_studio import render_jobs, review, review_server, tools, treatment
from lumr_studio.jobs import JOBS, JobRegistry
from lumr_studio.project import open_project, projects_root, write_json_atomic
from lumr_studio.render_jobs import EXPORT_FAILED, RENDER_KIND
from lumr_studio.review_server import MAX_BODY_BYTES, UNSATISFIABLE, ReviewServer, parse_range
from lumr_studio.silences import no_silences
from test_render_jobs import EXPORT_KEYS, EXPORTED_NAME, StandInRender

DURATION = 20.0
CUTS = [
    {"start": 2.45, "end": 3.85, "reason": "auto: filler: um", "source": "auto"},
    {"start": 3.95, "end": 7.95, "reason": "retake of the opening line", "source": "claude"},
    {"start": 10.75, "end": 11.95, "reason": "auto: pause: 1.3s", "source": "auto"},
]
FAKE_PEAKS = bytes(range(0, 250, 5))


def fake_join_rows(cuts, words, duration, *, labels=None):
    return [
        {"cut": i, "start": c["start"], "end": c["end"], "seconds": c["end"] - c["start"], "source": c["source"],
         "reason": c["reason"], "removed_words": 0, "before": "", "removed": "", "after": "",
         "flags": ["mid_sentence_in"] if c["source"] == "claude" else [], "note": ""}
        for i, c in enumerate(cuts)
    ]


@pytest.fixture
def project(video):
    p = open_project(str(video))
    write_json_atomic(p.edit_path, {"version": 1, "video": str(video), "duration": DURATION, "cuts": CUTS})
    return p


@pytest.fixture
def render():
    """The export's work, standing in for ffmpeg: no test here encodes video."""
    stand_in = StandInRender()
    yield stand_in
    stand_in.go.set()


@pytest.fixture
def server(jobs, render):
    s = ReviewServer(join_rows=fake_join_rows, silences=no_silences, measure_peaks=lambda _v: FAKE_PEAKS,
                     jobs=jobs, render=render)
    s.start()
    yield s
    s.stop()


@pytest.fixture
def base(server, project):
    return server.url_for(project)


def call(url, *, method="GET", headers=None, body=None):
    """(status, headers, body bytes) for one request; HTTP errors are returned, not raised."""
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.headers, resp.read()
    except urllib.error.HTTPError as err:
        return err.code, err.headers, err.read()


def post_json(url, data, **headers):
    return call(url, method="POST", body=json.dumps(data).encode(),
                headers={"Content-Type": "application/json", **headers})


def error_of(body: bytes) -> str:
    return json.loads(body)["error"]


# ── lifecycle and access ─────────────────────────────────────────────────────


def test_url_shape_and_one_token_per_project(server, project, base):
    assert base.startswith(f"http://127.0.0.1:{server.port}/") and base.endswith("/")
    token = base.rstrip("/").rsplit("/", 1)[1]
    assert len(token) >= 32
    assert server.url_for(project) == base


def test_binds_loopback_only(server):
    assert server._httpd.server_address[0] == "127.0.0.1"


def test_a_dropped_connection_is_not_logged_but_a_real_failure_is(server, caplog, capsys):
    for exc, logged in ((ConnectionResetError(54, "reset"), False), (BrokenPipeError(), False), (ValueError("bad"), True)):
        caplog.clear()
        try:
            raise exc
        except Exception:
            server._httpd.handle_error(None, ("127.0.0.1", 1))
        assert bool(caplog.records) is logged, exc
    assert "Traceback" not in capsys.readouterr().err


def test_page_is_served_with_a_policy_that_blocks_outside_requests(base):
    status, headers, body = call(base)
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    csp = headers["Content-Security-Policy"]
    assert "default-src 'none'" in csp and "connect-src 'self'" in csp
    page = body.decode()
    assert "http://" not in page
    # The only address in the page is the GitHub issue that Send feedback opens in a new tab on a click.
    assert page.count("https://") == 1 and "https://github.com/AshleyxAdamson/lumr-studio/issues/new" in page
    assert "New York" not in page
    assert '<video id="vid" src="video"' in page


def test_address_without_the_slash_redirects(server, base):
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    path = base.split(str(server.port), 1)[1].rstrip("/")
    conn.request("GET", path)
    resp = conn.getresponse()
    assert resp.status == 308
    assert resp.getheader("Location") == path + "/"
    conn.close()


@pytest.mark.parametrize("path", ["", "nottoken/", "nottoken/api/queue", "nottoken/video"])
def test_wrong_token_is_not_found(server, base, path):
    status, _h, body = call(f"http://127.0.0.1:{server.port}/{path}")
    assert status == 404
    assert "expired" in error_of(body)


def test_unknown_route_under_a_good_token_is_not_found(base):
    assert call(base + "secrets")[0] == 404
    assert call(base + "../edit.json")[0] == 404


@pytest.mark.parametrize("host", ["evil.example:80", "127.0.0.1", "127.0.0.1:1", "attacker.localhost"])
def test_a_foreign_host_header_is_refused(base, host):
    status, _h, body = call(base + "api/queue", headers={"Host": host})
    assert status == 403
    assert "not allowed" in error_of(body)


def test_localhost_host_header_is_accepted(server, base):
    status, _h, _b = call(base + "api/queue", headers={"Host": f"localhost:{server.port}"})
    assert status == 200


def test_stop_releases_the_port(project):
    s = ReviewServer(join_rows=fake_join_rows, silences=no_silences)
    s.start()
    url = s.url_for(project)
    s.stop()
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(url, timeout=2)
    s.stop()  # a second stop is harmless


# ── the queue and peaks ──────────────────────────────────────────────────────


def test_queue_has_rows_counts_and_durations(base):
    status, headers, body = call(base + "api/queue")
    assert status == 200 and headers["Cache-Control"] == "no-store"
    q = json.loads(body)
    assert [r["section"] for r in q["rows"]] == ["flagged", "auto", "auto"]
    assert q["counts"] == {"rows": 3, "decided": 0, "by_state": {"open": 3, "accepted": 0, "restored": 0, "edited": 0}}
    assert q["durations"]["full"] == pytest.approx(DURATION, abs=0.1)
    assert q["video"]["name"] == "talk"
    assert len(q["words"]) == 26


def test_peaks_come_back_base64(base, project):
    status, _h, body = call(base + "api/peaks")
    assert status == 200
    data = json.loads(body)
    assert base64.b64decode(data["peaks"]) == FAKE_PEAKS
    assert data["seconds_per_peak"] == 0.02
    assert list((project.root / "review").glob("peaks-*.bin"))


# ── video and Range ──────────────────────────────────────────────────────────


def test_full_video_says_it_accepts_ranges(base, video):
    status, headers, body = call(base + "video")
    assert status == 200
    assert headers["Accept-Ranges"] == "bytes"
    assert headers["Content-Type"] == "video/mp4"
    assert body == video.read_bytes()


def test_range_returns_206_with_the_right_bytes(base, video):
    data = video.read_bytes()
    status, headers, body = call(base + "video", headers={"Range": "bytes=100-199"})
    assert status == 206
    assert headers["Content-Range"] == f"bytes 100-199/{len(data)}"
    assert headers["Content-Length"] == "100"
    assert body == data[100:200]


def test_open_ended_range_runs_to_the_end(base, video):
    data = video.read_bytes()
    status, headers, body = call(base + "video", headers={"Range": "bytes=1000-"})
    assert status == 206
    assert headers["Content-Range"] == f"bytes 1000-{len(data) - 1}/{len(data)}"
    assert body == data[1000:]


def test_suffix_range_returns_the_tail(base, video):
    data = video.read_bytes()
    status, headers, body = call(base + "video", headers={"Range": "bytes=-500"})
    assert status == 206
    assert headers["Content-Range"] == f"bytes {len(data) - 500}-{len(data) - 1}/{len(data)}"
    assert body == data[-500:]


def test_range_past_the_end_is_416(base, video):
    size = video.stat().st_size
    status, headers, _b = call(base + "video", headers={"Range": f"bytes={size}-"})
    assert status == 416
    assert headers["Content-Range"] == f"bytes */{size}"


def test_head_sends_headers_only(server, base, video):
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    conn.request("HEAD", base.split(str(server.port), 1)[1] + "video", headers={"Range": "bytes=0-9"})
    resp = conn.getresponse()
    assert resp.status == 206 and resp.getheader("Content-Length") == "10"
    assert resp.read() == b""
    conn.close()


@pytest.mark.parametrize("header, size, expected", [
    (None, 1000, None),
    ("bytes=0-99", 1000, (0, 99)),
    ("bytes=990-2000", 1000, (990, 999)),
    ("bytes=500-", 1000, (500, 999)),
    ("bytes=-100", 1000, (900, 999)),
    ("bytes=-5000", 1000, (0, 999)),
    ("bytes=1000-", 1000, UNSATISFIABLE),
    ("bytes=-0", 1000, UNSATISFIABLE),
    ("bytes=0-", 0, UNSATISFIABLE),
    ("bytes=9-3", 1000, None),
    ("bytes=0-1,5-6", 1000, None),
    ("items=0-1", 1000, None),
    ("bytes=a-b", 1000, None),
])
def test_parse_range(header, size, expected):
    assert parse_range(header, size) == expected


# ── frames ───────────────────────────────────────────────────────────────────


def test_frame_is_a_jpeg_cached_in_the_project(base, project):
    status, headers, body = call(base + "frame?t=5.0")
    assert status == 200 and headers["Content-Type"] == "image/jpeg"
    assert body[:2] == b"\xff\xd8"
    assert (project.root / "review" / "frames" / "00000500.jpg").exists()


@pytest.mark.parametrize("query", ["frame", "frame?t=abc", "frame?t=-1", "frame?t=25", "frame?t=nan", "frame?t=inf"])
def test_bad_frame_times_are_refused(base, query):
    status, _h, body = call(base + query)
    assert status == 400
    assert "t" in error_of(body)


# ── decisions ────────────────────────────────────────────────────────────────


def ids():
    return [review.row_id(c["start"], c["end"]) for c in CUTS]


def test_decisions_round_trip_through_a_reload(base, project):
    decisions = {ids()[0]: {"state": "accepted"}, ids()[2]: {"state": "edited", "start": 10.75, "end": 11.5}}
    status, _h, body = post_json(base + "api/decisions", {"decisions": decisions})
    assert status == 200
    assert json.loads(body)["decided"] == 1
    assert json.loads((project.root / "review.json").read_text())["decisions"] == decisions
    q = json.loads(call(base + "api/queue")[2])
    states = {r["id"]: (r["state"], r["edges"]) for r in q["rows"]}
    assert states[ids()[0]] == ("accepted", None)
    assert states[ids()[2]] == ("edited", [10.75, 11.5])


def test_a_bad_decision_is_a_400_that_says_why(base, project):
    status, _h, body = post_json(base + "api/decisions", {"decisions": {ids()[1]: {"state": "maybe"}}})
    assert status == 400
    msg = error_of(body)
    assert ids()[1] in msg and "maybe" in msg and "accepted" in msg
    assert not (project.root / "review.json").exists()


def test_decisions_outside_the_video_are_refused(base):
    status, _h, body = post_json(base + "api/decisions",
                                 {"decisions": {ids()[2]: {"state": "edited", "start": 10.0, "end": 99.0}}})
    assert status == 400 and "outside the video" in error_of(body)


def test_decisions_for_an_unknown_row_are_refused(base):
    status, _h, body = post_json(base + "api/decisions", {"decisions": {"c1.00-2.00": {"state": "accepted"}}})
    assert status == 400 and "reload" in error_of(body)


def test_decisions_body_needs_the_decisions_key(base):
    status, _h, body = post_json(base + "api/decisions", {"rows": {}})
    assert status == 400 and "decisions" in error_of(body)


def test_post_needs_json_content_type(base):
    status, _h, body = call(base + "api/decisions", method="POST", body=b'{"decisions": {}}',
                            headers={"Content-Type": "text/plain"})
    assert status == 415 and "application/json" in error_of(body)


def test_post_body_must_be_json(base):
    status, _h, body = call(base + "api/decisions", method="POST", body=b"{nope",
                            headers={"Content-Type": "application/json"})
    assert status == 400 and "not valid JSON" in error_of(body)


def test_post_body_over_the_limit_is_refused(server, base):
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    conn.putrequest("POST", base.split(str(server.port), 1)[1] + "api/decisions")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", str(MAX_BODY_BYTES + 1))
    conn.endheaders()
    resp = conn.getresponse()
    assert resp.status == 413
    conn.close()


def test_post_from_another_origin_is_refused(base):
    status, _h, body = post_json(base + "api/decisions", {"decisions": {}}, Origin="http://evil.example")
    assert status == 403 and "other sites" in error_of(body)


def test_post_from_the_page_origin_is_accepted(server, base):
    status, _h, _b = post_json(base + "api/decisions", {"decisions": {}}, Origin=f"http://127.0.0.1:{server.port}")
    assert status == 200


def test_get_on_a_post_route_and_post_on_a_get_route_are_not_found(base):
    assert call(base + "api/decisions")[0] == 404
    assert post_json(base + "api/queue", {})[0] == 404


# ── apply ────────────────────────────────────────────────────────────────────


def test_apply_changes_the_edit(base, project):
    before = json.loads(call(base + "api/queue")[2])["durations"]["saved_edit"]
    decisions = {ids()[1]: {"state": "restored"}, ids()[2]: {"state": "edited", "start": 10.75, "end": 11.5}}
    assert post_json(base + "api/decisions", {"decisions": decisions})[0] == 200
    status, _h, body = post_json(base + "api/apply", {})
    assert status == 200
    result = json.loads(body)
    assert result["restored"] == 1 and result["edited"] == 1
    assert result["new_duration"] == pytest.approx(before + 4.0 + 0.45, abs=0.01)

    saved = json.loads(project.edit_path.read_text())
    assert [(c["start"], c["end"]) for c in saved["cuts"]] == [(2.45, 3.85), (10.75, 11.5)]
    assert saved["keep"][0]["start"] == 3.95

    q = json.loads(call(base + "api/queue")[2])
    assert len(q["rows"]) == 2
    assert q["durations"]["saved_edit"] == pytest.approx(result["new_duration"])


def test_apply_with_nothing_decided_leaves_the_cuts(base, project):
    status, _h, body = post_json(base + "api/apply", {})
    assert status == 200 and json.loads(body)["restored"] == 0
    assert [c["start"] for c in json.loads(project.edit_path.read_text())["cuts"]] == [2.45, 3.95, 10.75]


# ── the treatment page ───────────────────────────────────────────────────────

TREATMENT_CUTS = [
    {"start": 3.3, "end": 3.8, "reason": "filler before the line", "kind": "other"},
    {"start": 7.9, "end": 10.75, "reason": "second take of the opening line", "kind": "repeat"},
]


@pytest.fixture
def treated(server, video):
    """The page address for a video Claude has edited through set_edit."""
    tools.set_edit(str(video), TREATMENT_CUTS, auto_tighten=True, silences=no_silences,
                   labels=tools.unmeasured_labels)
    return server.url_for(open_project(str(video)))


def state_of(body: bytes) -> dict:
    return json.loads(body)


def test_the_treatment_page_is_served_at_the_token_root(base, monkeypatch, tmp_path):
    page = tmp_path / "index.html"
    page.write_text("<!doctype html><title>Treatment</title>")
    monkeypatch.setattr(review_server, "PAGE_PATH", page)
    status, headers, body = call(base)
    assert status == 200 and body == page.read_bytes()
    assert "default-src 'none'" in headers["Content-Security-Policy"]


def test_a_missing_page_file_says_to_reinstall(base, monkeypatch, tmp_path):
    monkeypatch.setattr(review_server, "PAGE_PATH", tmp_path / "gone.html")
    status, _headers, body = call(base)
    assert status == 500 and "Reinstall the plugin" in error_of(body)


def test_get_treatment_answers_the_whole_state(treated):
    status, headers, body = call(treated + "api/treatment")
    assert status == 200 and headers["Content-Type"].startswith("application/json")
    state = state_of(body)
    assert {"settings", "paces", "take_out", "need_a_look", "groups", "rows", "removed", "keeps", "samples",
            "clusters", "busy", "overlays", "export", "durations", "changed", "words"} <= set(state)
    assert state["export"]["state"] == "idle" and state["overlays"] == []
    assert state["changed"] is None and len(state["rows"]) == 2


def test_get_treatment_works_on_an_edit_saved_before_treatments(base):
    status, _h, body = call(base + "api/treatment")
    assert status == 200 and state_of(body)["rows"] == []


def test_post_treatment_changes_the_pace_and_says_what_changed(treated, video):
    status, _h, body = post_json(treated + "api/treatment", {"pace": "fast", "take_out": {"fillers": False}})
    assert status == 200
    state = state_of(body)
    assert state["settings"]["pace"] == "fast" and state["settings"]["take_out"]["fillers"] is False
    assert state["changed"] is not None
    saved = json.loads(open_project(str(video)).edit_path.read_text())
    assert saved["treatment"]["pace"] == "fast"


@pytest.mark.parametrize("route, body, message", [
    ("api/treatment", {"pace": "brutal"}, "is not a level"),
    ("api/treatment", {}, "Send a pace"),
    ("api/treatment", {"take_out": {"fillers": 1}}, "true or false"),
    ("api/cut", {"id": "c0.00-1.00", "state": "put_back"}, "reload the page"),
    ("api/cut", {"id": "c0.00-1.00"}, "state must be"),
    ("api/keep", {"start": 5, "end": 2}, "must come before"),
    ("api/keep", {"start": 1, "end": 99}, "outside the video"),
    ("api/keep/remove", {"id": "k1.00-2.00"}, "Reload the page"),
    ("api/keep", {"start": 6.7, "end": 7.5, "exact": True}, "No word sits between"),
    ("api/cut/add", {"start": 6.7, "end": 7.5}, "No word sits between"),
    ("api/cut/add", {"start": 1, "end": 99}, "outside the video"),
    ("api/cut/remove", {"id": "y1.00-2.00"}, "Reload the page"),
    ("api/cut", {"id": "y1.00-2.00", "state": "put_back"}, "Take it back instead"),
    ("api/undo", {}, "nothing to undo"),
    ("api/treatment", {"fine": {"gap_length": 0.37}}, "pause length is not on the slider"),
    ("api/treatment", {"fine": {"rhythm": 9}}, "rhythm is not on the slider"),
    ("api/treatment", {"pace": "fast", "fine": {"rhythm": 3}}, "Send a pace or fine, not both"),
    ("api/treatment", {"pace": "custom"}, "can't be picked by name"),
    ("api/treatment", {"take_out": {"likes": 1}}, "true or false"),
    ("api/samples", {"more": True}, "Unknown field"),
    ("api/usual", {"pace": "fast"}, "Unknown field"),
])
def test_bad_treatment_posts_are_a_400_that_says_what_to_do(treated, route, body, message):
    status, _h, raw = post_json(treated + route, body)
    assert status == 400 and message in error_of(raw)


def test_put_back_and_keep_out_over_http(treated):
    rows = state_of(call(treated + "api/treatment")[2])["rows"]
    rid = next(r["id"] for r in rows if r["group"] == "repeat")
    state = state_of(post_json(treated + "api/cut", {"id": rid, "state": "put_back"})[2])
    assert next(r for r in state["rows"] if r["id"] == rid)["state"] == "put_back"
    state = state_of(post_json(treated + "api/cut", {"id": rid, "state": "kept_out"})[2])
    assert next(r for r in state["rows"] if r["id"] == rid)["state"] == "kept_out"


def test_keep_and_remove_over_http(treated):
    status, _h, body = post_json(treated + "api/keep", {"start": 9.0, "end": 9.2, "note": "the good take"})
    assert status == 200
    (kept,) = state_of(body)["keeps"]
    status, _h, body = post_json(treated + "api/keep/remove", {"id": kept["id"]})
    assert status == 200 and state_of(body)["keeps"] == []


def test_cutting_a_word_bringing_one_back_and_undo_over_http(treated):
    status, _h, body = post_json(treated + "api/cut/add", {"start": 16.05, "end": 16.2})
    state = state_of(body)
    assert status == 200 and state["changed"]["words_cut"] == 1 and state["can_undo"] is True
    assert [(r["id"], r["by"], r["removed"]) for r in state["rows"] if r["by"] == "you"] == [("y16.05-16.20", "you", "for")]
    assert [w[4] for w in state["words"] if w[0] == "for"] == [3]

    state = state_of(post_json(treated + "api/keep", {"start": 8.65, "end": 9.0, "exact": True})[2])
    assert [(k["text"], k["exact"]) for k in state["keeps"]] == [("talk", True)]
    assert state["changed"]["words_back"] == 1

    state = state_of(post_json(treated + "api/undo", {})[2])
    assert state["keeps"] == [] and state["can_undo"] is False

    status, _h, body = post_json(treated + "api/cut/remove", {"id": "y16.05-16.20"})
    assert status == 200 and state_of(body)["cut_counts"]["yours"] == 0


def test_taking_back_one_word_of_a_longer_cut_over_http(treated):
    state = state_of(post_json(treated + "api/cut/add", {"start": 15.5, "end": 17.0})[2])
    assert [r["removed"] for r in state["rows"] if r["by"] == "you"] == ["thanks for watching."]
    status, _h, body = post_json(treated + "api/cut/remove", {"id": "y15.50-17.00", "start": 16.05, "end": 16.2})
    state = state_of(body)
    assert status == 200 and state["changed"]["words_back"] == 1
    assert [(r["id"], r["removed"]) for r in state["rows"] if r["by"] == "you"] == [
        ("y15.50-16.00", "thanks"), ("y16.25-17.00", "watching."),
    ]
    assert state["cut_counts"]["yours"] == 2 and state["can_undo"] is True


def test_the_state_says_why_the_word_times_are_estimates(treated):
    state = state_of(call(treated + "api/treatment")[2])
    assert state["word_times"] == "estimated" and state["word_times_note"].startswith("The word times are estimates")


def test_the_pace_takes_all_six_stops_over_http(treated):
    for name in ("natural", "standard", "fast", "tight", "hard", "max"):
        status, _h, body = post_json(treated + "api/treatment", {"pace": name})
        assert status == 200 and state_of(body)["settings"]["pace"] == name
    assert len(state_of(body)["paces"]) == 6


def test_the_sliders_over_http(treated):
    state = state_of(call(treated + "api/treatment")[2])
    assert state["settings"]["fine"] == {"gap_length": 0.6, "rhythm": 3.0} and state["custom"] is None
    assert [len(state["fine_ranges"][k]["positions"]) for k in ("gap_length", "rhythm")] == [28, 9]
    status, _h, body = post_json(treated + "api/treatment", {"fine": {"gap_length": 0.35, "rhythm": 3.5}})
    state = state_of(body)
    assert status == 200 and state["settings"]["pace"] == "custom"
    assert state["settings"]["fine"] == {"gap_length": 0.35, "rhythm": 3.5}
    assert state["custom"]["label"] == "Custom" and len(state["paces"]) == 6
    # The page sends a drag as a run of moves; each answers the whole state.
    for gap in (0.3, 0.25, 0.2):
        status, _h, body = post_json(treated + "api/treatment", {"fine": {"gap_length": gap}})
        assert status == 200 and state_of(body)["settings"]["fine"] == {"gap_length": gap, "rhythm": 3.5}
    state = state_of(post_json(treated + "api/usual", {})[2])
    assert state["usual"]["pace"] == "custom" and state["usual"]["fine"] == {"gap_length": 0.2, "rhythm": 3.5}
    state = state_of(post_json(treated + "api/treatment", {"pace": "hard"})[2])
    assert state["settings"]["pace"] == "hard" and state["custom"] is None


def test_the_filler_likes_switch_over_http(server, video):
    tools.set_edit(str(video), TREATMENT_CUTS, auto_tighten=True, silences=no_silences, labels=tools.unmeasured_labels)
    url = server.url_for(open_project(str(video)))
    state = state_of(call(url + "api/treatment")[2])
    assert state["take_out"][3]["key"] == "likes" and state["take_out"][3]["ready"] is False
    tools.set_edit(str(video), TREATMENT_CUTS, auto_tighten=True, silences=no_silences, labels=tools.unmeasured_labels,
                   picks=[{"id": "w16.050-16.200", "reason": "stands in for a filler word"}])
    state = state_of(call(url + "api/treatment")[2])
    likes = state["take_out"][3]
    assert (likes["ready"], likes["count"], likes["picked"]) == (True, 1, 1)
    assert [w[4] for w in state["words"] if w[0] == "for"] == [4]
    state = state_of(post_json(url + "api/treatment", {"take_out": {"likes": False}})[2])
    assert state["settings"]["take_out"]["likes"] is False and state["changed"]["likes"] == -1
    assert [w[4] for w in state["words"] if w[0] == "for"] == [0]
    state = state_of(post_json(url + "api/keep", {"start": 16.05, "end": 16.2, "exact": True})[2])
    state = state_of(post_json(url + "api/treatment", {"take_out": {"likes": True}})[2])
    assert [w[4] for w in state["words"] if w[0] == "for"] == [0] and state["take_out"][3]["kept_by_you"] == 1


def test_samples_and_usual_over_http(treated):
    status, _h, body = post_json(treated + "api/samples", {})
    assert status == 200 and len(state_of(body)["samples"]) == 3
    post_json(treated + "api/treatment", {"pace": "natural"})
    status, _h, body = post_json(treated + "api/usual", {})
    assert status == 200 and state_of(body)["usual"]["pace"] == "natural"
    assert json.loads((projects_root() / treatment.USUAL_FILE).read_text())["pace"] == "natural"


def test_rating_a_cut_over_http(treated, video):
    state = state_of(call(treated + "api/treatment")[2])
    row = next(r for r in state["rows"] if r["by"] == "claude")
    assert row["rating"] is None
    status, _h, body = post_json(treated + "api/rate", {"id": row["id"], "rating": "good"})
    got = next(r for r in state_of(body)["rows"] if r["id"] == row["id"])
    assert status == 200 and got["rating"] == "good" and got["state"] == row["state"]
    status, _h, body = post_json(treated + "api/rate", {"id": row["id"], "rating": "bad"})
    got = next(r for r in state_of(body)["rows"] if r["id"] == row["id"])
    assert status == 200 and got["rating"] == "bad" and got["state"] == "put_back"
    saved = json.loads(open_project(str(video)).edit_path.read_text())["ratings"]
    assert [(r["row"], r["rating"], r["source"]) for r in saved] == [(row["id"], "bad", "claude")]
    status, _h, body = post_json(treated + "api/rate", {"id": row["id"], "rating": None})
    got = next(r for r in state_of(body)["rows"] if r["id"] == row["id"])
    assert status == 200 and got["rating"] is None and got["state"] == "put_back"
    status, _h, raw = post_json(treated + "api/rate", {"id": row["id"], "rating": "meh"})
    assert status == 400 and "rating must be" in error_of(raw)
    status, _h, raw = post_json(treated + "api/rate", {"id": "c1.00-2.00", "rating": "good"})
    assert status == 400 and "not one of Claude's cuts" in error_of(raw)


def test_the_page_has_no_route_to_forget_what_was_learned(treated):
    assert "api/taste/forget" not in review_server.TREATMENT_ACTIONS
    status, _h, _b = post_json(treated + "api/taste/forget", {})
    assert status == 404


@pytest.mark.parametrize("route", sorted(review_server.TREATMENT_ACTIONS))
def test_every_treatment_post_refuses_another_origin(treated, route):
    status, _h, body = post_json(treated + route, {}, Origin="http://evil.example")
    assert status == 403 and "other sites" in error_of(body)


@pytest.mark.parametrize("route", sorted(review_server.TREATMENT_ACTIONS))
def test_every_treatment_post_needs_json(treated, route):
    status, _h, _b = call(treated + route, method="POST", body=b"{}", headers={"Content-Type": "text/plain"})
    assert status == 415


def test_treatment_routes_need_the_right_method(treated):
    assert call(treated + "api/cut")[0] == 404
    assert post_json(treated + "api/peaks", {})[0] == 404


# ── export ────────────────────────────────────────────────────────────────────


def export_of(body: bytes) -> dict:
    answer = json.loads(body)
    assert set(answer) == {"export"} and set(answer["export"]) == EXPORT_KEYS
    return answer["export"]


def running_job(jobs, video):
    return jobs.latest(RENDER_KIND, open_project(str(video)))


def test_a_server_made_with_no_registry_uses_the_one_the_tools_use():
    assert ReviewServer().jobs is JOBS and tools.job_status.__kwdefaults__["jobs"] is JOBS


def test_before_any_export_the_answer_is_idle(treated):
    status, headers, body = call(treated + "api/export")
    assert status == 200 and headers["Cache-Control"] == "no-store"
    assert export_of(body) == {"state": "idle", "progress": None, "file": None, "folder": None,
                               "message": "Nothing exported yet.", "holds_earlier_edit": False}


def test_post_export_starts_the_render_of_the_saved_edit(treated, video, jobs, render):
    status, _h, body = post_json(treated + "api/export", {})
    assert status == 200 and export_of(body)["state"] == "running"
    assert render.running.wait(5)
    saved = json.loads(open_project(str(video)).edit_path.read_text())
    assert [round(a, 2) for a, _ in render.calls[0]] == [round(c["start"], 2) for c in saved["cuts"]]
    now = export_of(call(treated + "api/export")[2])
    assert (now["state"], now["progress"], now["message"]) == ("running", 0.42, "Exporting. 42% done.")


def test_a_post_while_one_runs_answers_the_running_one_and_starts_nothing(treated, video, jobs, render):
    post_json(treated + "api/export", {})
    first = running_job(jobs, video).job_id
    for _ in range(3):
        status, _h, body = post_json(treated + "api/export", {})
        assert status == 200 and export_of(body)["state"] == "running"
    assert render.running.wait(5) and len(render.calls) == 1
    assert running_job(jobs, video).job_id == first


def test_a_render_claude_started_counts_as_the_running_export(treated, video, jobs, render, monkeypatch):
    monkeypatch.setattr(render_jobs, "render_full", render)
    claude = tools.render(str(video), jobs=jobs)
    assert set(claude) == {"job_id"}
    assert render.running.wait(5)
    assert export_of(call(treated + "api/export")[2])["state"] == "running"
    assert export_of(post_json(treated + "api/export", {})[2])["state"] == "running"
    assert len(render.calls) == 1 and running_job(jobs, video).job_id == claude["job_id"]


def test_job_status_sees_the_export_the_page_started(treated, video, jobs, render):
    post_json(treated + "api/export", {})
    found = tools.get_edit(str(video), jobs=jobs)["export"]
    assert found["status"] == "running"
    assert tools.job_status(found["job_id"], jobs=jobs)["kind"] == "render"
    render.go.set()
    done = tools.job_status(found["job_id"], wait=5, jobs=jobs)
    assert done["status"] == "done" and done["result"]["output_path"].endswith(EXPORTED_NAME)


def test_a_finished_export_names_the_file_and_a_reload_still_shows_it(treated, video, jobs, render):
    post_json(treated + "api/export", {})
    render.finish(jobs, running_job(jobs, video).job_id)
    done = export_of(call(treated + "api/export")[2])
    assert (done["state"], done["progress"], done["file"]) == ("done", 1.0, EXPORTED_NAME)
    assert done["folder"] == str(open_project(str(video)).exports_dir)
    assert state_of(call(treated + "api/treatment")[2])["export"] == done


def test_a_post_after_one_finished_starts_a_new_export(treated, video, jobs, render):
    post_json(treated + "api/export", {})
    first = running_job(jobs, video).job_id
    render.finish(jobs, first)
    render.go.clear()
    assert export_of(post_json(treated + "api/export", {})[2])["state"] == "running"
    assert running_job(jobs, video).job_id != first


def test_a_failed_export_says_what_to_do_next(treated, video, jobs, render):
    render.fail_with = RuntimeError("ffmpeg exited 1 at 412.30")
    post_json(treated + "api/export", {})
    job_id = running_job(jobs, video).job_id
    render.finish(jobs, job_id)
    failed = export_of(call(treated + "api/export")[2])
    assert (failed["state"], failed["file"], failed["message"]) == ("failed", None, EXPORT_FAILED)
    assert "ffmpeg" in tools.job_status(job_id, jobs=jobs)["error"]  # Claude still gets the details
    render.fail_with = None
    render.go.clear()
    assert export_of(post_json(treated + "api/export", {})[2])["state"] == "running"  # she can try again


def test_changing_the_edit_while_it_exports_is_allowed_and_the_export_says_it_is_behind(treated, render):
    rows = state_of(call(treated + "api/treatment")[2])["rows"]
    rid = next(r["id"] for r in rows if r["group"] == "repeat")
    post_json(treated + "api/export", {})
    assert render.running.wait(5)
    status, _h, body = post_json(treated + "api/cut", {"id": rid, "state": "put_back"})
    assert status == 200
    state = state_of(body)
    assert next(r for r in state["rows"] if r["id"] == rid)["state"] == "put_back"
    assert (state["export"]["state"], state["export"]["holds_earlier_edit"]) == ("running", True)
    assert export_of(call(treated + "api/export")[2])["holds_earlier_edit"] is True
    assert len(render.calls) == 1
    # Undoing the change brings the edit back to what the export holds.
    state = state_of(post_json(treated + "api/cut", {"id": rid, "state": "kept_out"})[2])
    assert state["export"]["holds_earlier_edit"] is False


def test_every_post_answer_carries_the_export(treated, render):
    post_json(treated + "api/export", {})
    for route, body in [("api/treatment", {"pace": "natural"}), ("api/samples", {}), ("api/usual", {}),
                        ("api/keep", {"start": 9.0, "end": 9.2})]:
        state = state_of(post_json(treated + route, body)[2])
        assert state["export"]["state"] == "running", route


def test_export_takes_no_settings(treated, jobs, video):
    status, _h, body = post_json(treated + "api/export", {"quality": "high"})
    assert status == 400 and "Send an empty JSON object" in error_of(body)
    assert running_job(jobs, video) is None


def test_export_with_an_edit_that_cannot_be_read_is_a_400_and_starts_nothing(treated, jobs, video):
    open_project(str(video)).edit_path.write_text("{not json")
    status, _h, body = post_json(treated + "api/export", {})
    assert status == 400 and "not valid JSON" in error_of(body)
    assert running_job(jobs, video) is None


@pytest.mark.parametrize("headers, body, status", [
    ({"Content-Type": "application/json", "Origin": "http://evil.example"}, b"{}", 403),
    ({"Content-Type": "application/json", "Host": "evil.example:80"}, b"{}", 403),
    ({"Content-Type": "text/plain"}, b"{}", 415),
    ({"Content-Type": "application/x-www-form-urlencoded"}, b"", 415),
    ({"Content-Type": "application/json"}, b"{nope", 400),
    ({"Content-Type": "application/json"}, b"[]", 400),
])
def test_post_export_keeps_every_rule_the_other_posts_have(treated, jobs, video, headers, body, status):
    assert call(treated + "api/export", method="POST", body=body, headers=headers)[0] == status
    assert running_job(jobs, video) is None


def test_post_export_over_the_body_limit_is_refused(server, treated, jobs, video):
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    conn.putrequest("POST", treated.split(str(server.port), 1)[1] + "api/export")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", str(MAX_BODY_BYTES + 1))
    conn.endheaders()
    assert conn.getresponse().status == 413
    conn.close()
    assert running_job(jobs, video) is None


def test_export_needs_the_token(server, treated, jobs, video):
    wrong = f"http://127.0.0.1:{server.port}/nottoken/api/export"
    assert call(wrong)[0] == 404 and post_json(wrong, {})[0] == 404
    assert call(treated + "api/export", headers={"Host": "evil.example:80"})[0] == 403
    assert running_job(jobs, video) is None


def test_one_videos_export_does_not_show_on_another(server, treated, video, tmp_path, render):
    other = tmp_path / "other.mp4"
    other.write_bytes(video.read_bytes())
    other.with_suffix(".words.json").write_text(video.with_suffix(".words.json").read_text())
    other_page = server.url_for(open_project(str(other)))
    post_json(treated + "api/export", {})
    assert export_of(call(other_page + "api/export")[2])["state"] == "idle"
    assert export_of(post_json(other_page + "api/export", {})[2])["state"] == "running"
    assert render.running.wait(5)
