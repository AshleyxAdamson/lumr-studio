"""The decision queue's data: rows, decisions on disk, and applying them to the edit."""

import copy
import json
import math

import pytest

from lumr_studio import review
from lumr_studio.errors import StudioError
from lumr_studio.project import open_project, write_json_atomic

DURATION = 20.0

# Cuts on the conftest transcript: a filler trim, Claude's retake cut, two pause trims.
FILLER = {"start": 2.45, "end": 3.85, "reason": "auto: filler: um", "source": "auto"}
RETAKE = {"start": 3.85, "end": 7.95, "reason": "retake of the opening line", "source": "claude"}
PAUSE_A = {"start": 10.75, "end": 11.95, "reason": "auto: pause: 1.3s", "source": "auto"}
PAUSE_B = {"start": 14.25, "end": 15.45, "reason": "auto: pause: 1.3s", "source": "auto"}


def make_edit() -> dict:
    return {
        "version": 1, "video": "/footage/talk.mp4", "duration": DURATION,
        "updated_at": "2026-09-27T10:00:00+00:00", "auto_tighten": True, "gap_length": 0.6,
        "cuts": [dict(c) for c in (FILLER, RETAKE, PAUSE_A, PAUSE_B)],
    }


def fake_join_rows(cuts, words=None, duration=None, *, labels=None, flagged=None):
    """Rows of the join-report shape, one per cut, flags only where asked."""
    if flagged is None:
        flagged = {1: ["mid_sentence_in", "long_jump"]}
    return [
        {"cut": i, "start": c["start"], "end": c["end"], "seconds": round(c["end"] - c["start"], 3),
         "source": c["source"], "reason": c["reason"], "removed_words": 0,
         "before": "", "removed": "", "after": "",
         "flags": flagged.get(i, []), "note": "Check this join." if i in flagged else ""}
        for i, c in enumerate(cuts)
    ]


def rid(cut: dict) -> str:
    return review.row_id(cut["start"], cut["end"])


# ── row ids ──────────────────────────────────────────────────────────────────


def test_row_id_comes_from_rounded_edges():
    assert review.row_id(412.3, 431.8) == "c412.30-431.80"
    assert review.row_id(412.30000001, 431.7999999) == "c412.30-431.80"


def test_row_ids_are_the_same_on_every_build(words):
    edit = make_edit()
    a = review.build_queue(edit, words, DURATION, join_rows=fake_join_rows(edit["cuts"]))
    b = review.build_queue(copy.deepcopy(edit), words, DURATION, join_rows=fake_join_rows(edit["cuts"]))
    assert [r["id"] for r in a["rows"]] == [r["id"] for r in b["rows"]]
    assert len({r["id"] for r in a["rows"]}) == 4


# ── the queue ────────────────────────────────────────────────────────────────


def test_sections_run_flagged_then_claude_then_auto_each_in_time_order(words):
    edit = make_edit()
    joins = fake_join_rows(edit["cuts"], flagged={1: ["mid_sentence_in"], 3: ["tight"]})
    q = review.build_queue(edit, words, DURATION, join_rows=joins)
    assert [(r["section"], r["start"]) for r in q["rows"]] == [
        ("flagged", 3.85), ("flagged", 14.25), ("auto", 2.45), ("auto", 10.75),
    ]
    assert [(s["key"], s["count"]) for s in q["sections"]] == [("flagged", 2), ("claude", 0), ("auto", 2)]


def test_unflagged_claude_cut_goes_in_claudes_section(words):
    edit = make_edit()
    q = review.build_queue(edit, words, DURATION, join_rows=fake_join_rows(edit["cuts"], flagged={}))
    assert [r["section"] for r in q["rows"]] == ["claude", "auto", "auto", "auto"]


def test_flags_read_as_plain_words(words):
    edit = make_edit()
    q = review.build_queue(edit, words, DURATION, join_rows=fake_join_rows(edit["cuts"]))
    row = next(r for r in q["rows"] if r["source"] == "claude")
    assert row["flags"] == ["mid_sentence_in", "long_jump"]
    assert row["flag_labels"] == ["picks up mid-sentence", "takes out 20 seconds or more"]
    for label in row["flag_labels"]:
        assert "_" not in label


def test_row_carries_reason_seconds_source_class_and_removed_words(words):
    edit = make_edit()
    q = review.build_queue(edit, words, DURATION, join_rows=fake_join_rows(edit["cuts"]))
    by_id = {r["id"]: r for r in q["rows"]}
    retake = by_id[rid(RETAKE)]
    assert retake["reason"] == "retake of the opening line"
    assert retake["seconds"] == pytest.approx(4.1)
    assert retake["source"] == "claude" and retake["cls"] == "content"
    assert retake["removed_words"] == 5  # "today we talk about editing", the event is not a word
    assert by_id[rid(FILLER)]["cls"] == "filler"
    assert by_id[rid(PAUSE_A)]["cls"] == "pause"
    assert all(r["state"] == "open" and r["edges"] is None for r in q["rows"])


def test_join_rows_match_on_index_when_edges_differ(words):
    edit = make_edit()
    joins = fake_join_rows(edit["cuts"])
    for j in joins:
        j["start"] += 0.5  # a self-check that rounded differently still lands on the right cut
    q = review.build_queue(edit, words, DURATION, join_rows=joins)
    assert next(r for r in q["rows"] if r["source"] == "claude")["section"] == "flagged"


def test_bulk_sets_name_their_class_and_leave_claude_and_flagged_rows_out(words):
    edit = make_edit()
    q = review.build_queue(edit, words, DURATION, join_rows=fake_join_rows(edit["cuts"], flagged={3: ["tight"]}))
    assert q["bulk"] == [
        {"cls": "filler", "label": "filler trims", "ids": [rid(FILLER)]},
        {"cls": "pause", "label": "pause trims", "ids": [rid(PAUSE_A)]},
    ]


def test_decisions_set_states_edges_counts_and_lengths(words):
    edit = make_edit()
    decisions = {
        rid(FILLER): {"state": "accepted"},
        rid(RETAKE): {"state": "restored"},
        rid(PAUSE_A): {"state": "edited", "start": 10.75, "end": 11.5},
    }
    q = review.build_queue(edit, words, DURATION, join_rows=fake_join_rows(edit["cuts"]),
                           review={"decisions": decisions})
    by_id = {r["id"]: r for r in q["rows"]}
    assert by_id[rid(FILLER)]["state"] == "accepted"
    assert by_id[rid(RETAKE)]["state"] == "restored"
    assert by_id[rid(PAUSE_A)]["edges"] == [10.75, 11.5]
    assert q["counts"]["decided"] == 2  # edited still needs a listen
    assert q["counts"]["by_state"] == {"open": 1, "accepted": 1, "restored": 1, "edited": 1}
    removed_before = 1.4 + 4.1 + 1.2 + 1.2
    assert q["durations"]["saved_edit"] == pytest.approx(DURATION - removed_before)
    assert q["durations"]["with_decisions"] == pytest.approx(DURATION - (1.4 + 0.75 + 1.2))
    # two open rows, each auditioned 2s either side
    assert q["durations"]["listen_left"] == pytest.approx(8.0)


def test_page_words_name_sounds_plainly(words):
    plain = review.page_words(words)
    event = next(w for w in plain if w[3] == 1)
    assert event == [6.0, 6.6, "(sound)", 1]
    labelled = review.page_words(words, [{"start": 6.0, "end": 6.6, "kind": "laugh", "confidence": "likely"}])
    assert next(w for w in labelled if w[3] == 1)[2] == "(laugh)"
    possible = review.page_words(words, [{"start": 6.0, "end": 6.6, "kind": "laugh", "confidence": "possible"}])
    assert next(w for w in possible if w[3] == 1)[2] == "(laugh?)"


def test_times_read_as_minutes_seconds_tenths():
    assert review.fmt_time(0) == "0:00.0"
    assert review.fmt_time(638.12) == "10:38.1"
    assert review.fmt_time(59.96) == "1:00.0"


# ── validating decisions ─────────────────────────────────────────────────────


@pytest.fixture
def cuts_by_id():
    return {rid(c): c for c in make_edit()["cuts"]}


def test_valid_decisions_are_normalized_and_open_ones_dropped(cuts_by_id, words):
    clean = review.validate_decisions({
        rid(FILLER): {"state": "accepted"},
        rid(RETAKE): {"state": "open"},
        rid(PAUSE_A): {"state": "edited", "start": 10.75, "end": 11.5},
    }, cuts_by_id, DURATION, words)
    assert clean == {rid(FILLER): {"state": "accepted"}, rid(PAUSE_A): {"state": "edited", "start": 10.75, "end": 11.5}}


@pytest.mark.parametrize("decision, says", [
    ({"state": "vetoed"}, "is not one of"),
    ({"state": "edited"}, "needs its new start and end"),
    ({"state": "accepted", "start": 10.8}, "both start and end"),
    ({"state": "edited", "start": float("nan"), "end": 11.0}, "must be numbers"),
    ({"state": "edited", "start": "10", "end": 11.0}, "must be numbers"),
    ({"state": "edited", "start": 10.9, "end": 25.0}, "outside the video"),
    ({"state": "edited", "start": 11.0, "end": 10.9}, "is not before"),
    ({"state": "accepted", "why": "x"}, "unknown field"),
    ("accepted", "must be an object"),
])
def test_bad_decisions_are_refused_with_the_reason(cuts_by_id, decision, says):
    with pytest.raises(StudioError, match=says):
        review.validate_decisions({rid(PAUSE_A): decision}, cuts_by_id, DURATION)


def test_unknown_row_says_to_reload(cuts_by_id):
    with pytest.raises(StudioError, match="reload the page"):
        review.validate_decisions({"c1.00-2.00": {"state": "accepted"}}, cuts_by_id, DURATION)


def test_edges_that_clip_a_word_without_removing_one_are_refused(cuts_by_id, words):
    # 12.80-12.90 lies inside "part" without taking the whole word
    with pytest.raises(StudioError, match="removes no whole word"):
        review.validate_decisions({rid(PAUSE_A): {"state": "edited", "start": 12.80, "end": 12.90}},
                                  cuts_by_id, DURATION, words)


def test_decisions_must_be_a_map(cuts_by_id):
    with pytest.raises(StudioError, match="must be an object"):
        review.validate_decisions([], cuts_by_id, DURATION)


# ── applying ─────────────────────────────────────────────────────────────────


def test_apply_drops_restored_cuts_and_records_them_as_keep(words):
    edit = make_edit()
    out = review.apply_review(edit, {"decisions": {rid(RETAKE): {"state": "restored"}}}, words, DURATION)
    assert [c["start"] for c in out["cuts"]] == [2.45, 10.75, 14.25]
    assert out["keep"] == [{"start": 3.85, "end": 7.95, "note": out["keep"][0]["note"]}]
    assert "retake of the opening line" in out["keep"][0]["note"]
    assert edit["cuts"][1] == RETAKE and "keep" not in edit  # the input is left alone


def test_apply_places_edited_edges_word_safe(words):
    edit = make_edit()
    # widen the pause trim into "this is": the end lands mid-word and gets placed
    out = review.apply_review(edit, {"decisions": {rid(PAUSE_A): {"state": "edited", "start": 10.75, "end": 12.45}}},
                              words, DURATION)
    cut = next(c for c in out["cuts"] if c["start"] < 12.2 < c["end"])
    assert 10.70 <= cut["start"] <= 12.0  # after "video." and its natural tail
    assert 12.50 <= cut["end"] <= 12.55  # "this is" go (midpoint rule), "the" stays: the end sits before it
    assert cut["edited_in_review"] is True
    assert cut["source"] == "auto"


def test_apply_keeps_accepted_and_open_cuts_unchanged(words):
    edit = make_edit()
    out = review.apply_review(edit, {"decisions": {rid(FILLER): {"state": "accepted"}}}, words, DURATION)
    assert [(c["start"], c["end"]) for c in out["cuts"]] == [(c["start"], c["end"]) for c in edit["cuts"]]


def test_apply_merges_an_edit_that_grows_into_its_neighbour(words):
    edit = make_edit()
    out = review.apply_review(edit, {"decisions": {rid(FILLER): {"state": "edited", "start": 2.45, "end": 4.60}}},
                              words, DURATION)
    first = out["cuts"][0]
    assert 2.40 <= first["start"] <= 3.40 and first["end"] == 7.95  # placed after "welcome.", merged into the retake
    assert first["source"] == "claude"
    assert len(out["cuts"]) == 3


def test_apply_twice_does_not_duplicate_keep(words):
    edit = make_edit()
    once = review.apply_review(edit, {"decisions": {rid(RETAKE): {"state": "restored"}}}, words, DURATION)
    once["cuts"].insert(1, dict(RETAKE))  # Claude cuts it again; the creator restores it again
    twice = review.apply_review(once, {"decisions": {rid(RETAKE): {"state": "restored"}}}, words, DURATION)
    assert len(twice["keep"]) == 1


def test_apply_refuses_edges_that_cannot_be_placed(words):
    edit = make_edit()
    with pytest.raises(StudioError, match="can't be placed"):
        review.apply_review(edit, {"decisions": {rid(PAUSE_A): {"state": "edited", "start": 12.80, "end": 12.90}}},
                            words, DURATION)


# ── the project files ────────────────────────────────────────────────────────


@pytest.fixture
def project(video):
    p = open_project(str(video))
    write_json_atomic(p.edit_path, {**make_edit(), "video": str(video)})
    return p


def test_load_review_is_empty_before_any_decision(project):
    assert review.load_review(project) == {}


def test_save_then_load_brings_back_the_same_decisions(project, words):
    decisions = {rid(FILLER): {"state": "accepted"}, rid(PAUSE_B): {"state": "edited", "start": 14.25, "end": 15.0}}
    result = review.save_decisions(project, decisions, duration=DURATION)
    assert result == {"saved": 2, "decided": 1, "rows": 4, "with_decisions": pytest.approx(DURATION - (1.4 + 4.1 + 1.2 + 0.75))}
    assert review.load_review(project)["decisions"] == decisions
    q = review.queue_for_project(project, join_rows=fake_join_rows, duration=DURATION)
    states = {r["id"]: r["state"] for r in q["rows"]}
    assert states[rid(FILLER)] == "accepted" and states[rid(PAUSE_B)] == "edited"


def test_a_bad_save_writes_nothing(project):
    review.save_decisions(project, {rid(FILLER): {"state": "accepted"}}, duration=DURATION)
    with pytest.raises(StudioError):
        review.save_decisions(project, {rid(FILLER): {"state": "maybe"}}, duration=DURATION)
    assert review.load_review(project)["decisions"] == {rid(FILLER): {"state": "accepted"}}


def test_unreadable_review_file_says_how_to_fix(project):
    review.review_path(project).write_text("{not json")
    with pytest.raises(StudioError, match="Delete it"):
        review.load_review(project)


def test_queue_for_project_passes_labels_to_the_self_check(project):
    seen = {}

    def join_rows(cuts, words, duration, *, labels=None):
        seen["labels"] = labels
        seen["duration"] = duration
        return fake_join_rows(cuts)

    labels = [{"start": 6.0, "end": 6.6, "seconds": 0.6, "kind": "laugh", "confidence": "likely", "punchline": None}]
    q = review.queue_for_project(project, join_rows=join_rows, labels_for=lambda p, w: labels, duration=DURATION)
    assert seen == {"labels": labels, "duration": DURATION}
    assert q["video"]["name"] == "talk"
    assert any(w[2] == "(laugh)" for w in q["words"])


def test_apply_to_project_saves_the_edit_and_carries_decisions(project):
    review.save_decisions(project, {
        rid(FILLER): {"state": "accepted"},
        rid(RETAKE): {"state": "restored"},
        rid(PAUSE_A): {"state": "edited", "start": 10.75, "end": 11.5},
    }, duration=DURATION)
    result = review.apply_to_project(project, duration=DURATION)
    assert result["restored"] == 1 and result["edited"] == 1
    assert result["new_duration"] > result["previous_duration"]

    saved = json.loads(project.edit_path.read_text())
    assert [c["start"] for c in saved["cuts"]] == [2.45, 10.75, 14.25]
    assert saved["keep"][0]["start"] == 3.85

    carried = review.load_review(project)["decisions"]
    assert carried[rid(FILLER)] == {"state": "accepted"}  # untouched cut keeps its id and state
    assert rid(RETAKE) not in carried
    edited = next(c for c in saved["cuts"] if c["start"] == 10.75)
    assert carried[rid(edited)]["state"] == "edited"

    receipt = json.loads(project.receipts_path.read_text().splitlines()[-1])
    assert receipt["step"] == "review_apply" and receipt["restored"] == 1


def test_ids_survive_a_second_apply(project):
    review.save_decisions(project, {rid(FILLER): {"state": "accepted"}}, duration=DURATION)
    review.apply_to_project(project, duration=DURATION)
    review.apply_to_project(project, duration=DURATION)
    q = review.queue_for_project(project, join_rows=fake_join_rows, duration=DURATION)
    assert {r["id"] for r in q["rows"]} == {rid(c) for c in (FILLER, RETAKE, PAUSE_A, PAUSE_B)}
    assert next(r for r in q["rows"] if r["id"] == rid(FILLER))["state"] == "accepted"


def test_edited_seconds_ignores_restored_and_uses_new_edges():
    cuts = [dict(FILLER), dict(PAUSE_A)]
    got = review.edited_seconds(cuts, {rid(FILLER): {"state": "restored"},
                                       rid(PAUSE_A): {"state": "edited", "start": 10.0, "end": 12.0}}, DURATION)
    assert math.isclose(got, DURATION - 2.0)
