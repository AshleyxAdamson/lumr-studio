"""The creator's hand on the words: her own cuts, the words she brings back, and undo.

Synthetic words only. The talk is conftest's 20 seconds: an "um", a line said
twice, "this is the part that matters." and a sign-off.
"""

import json

import pytest
from lumr_studio.engine.render import finalize_removed_ranges

from conftest import spoken_run
from lumr_studio import edit as edits
from lumr_studio import tools, treatment
from lumr_studio.edit import CutRejected, SaidOrder, build_edit, finalize_near, place_creator_cut
from lumr_studio.errors import StudioError
from lumr_studio.pace import PACES
from lumr_studio.project import open_project
from lumr_studio.silences import no_silences
from lumr_studio.tools import unmeasured_labels

QUIET = {"silences": no_silences, "labels": unmeasured_labels}
DURATION = 20.0

UM = {"start": 3.3, "end": 3.8, "reason": "filler before the line", "kind": "other"}
RETAKE = {"start": 7.9, "end": 10.75, "reason": "second take of the opening line", "kind": "repeat"}
# Words of the conftest talk, by their own times.
WE, TALK, ABOUT = (4.35, 4.5), (4.55, 4.9), (4.95, 5.3)
TALK_AGAIN = (8.65, 9.0)     # inside RETAKE
THE, PART = (12.55, 12.7), (12.75, 13.1)
FOR = (16.05, 16.2)
THE_UM = (3.4, 3.7)


@pytest.fixture(autouse=True)
def fresh_recipes():
    treatment.forget_recipes()
    yield
    treatment.forget_recipes()


def ctx_for(video):
    return treatment.load_context(open_project(str(video)), duration=DURATION, silences=no_silences)


def edited(video, cuts=(UM, RETAKE), **kwargs):
    tools.set_edit(str(video), list(cuts), **kwargs, **QUIET)
    return ctx_for(video)


def saved(video):
    return json.loads(open_project(str(video)).edit_path.read_text())


def span(pair):
    return {"start": pair[0], "end": pair[1]}


def why_of(state, start):
    return next(w[4] for w in state["words"] if w[1] == start)


def yours(state):
    return [r for r in state["rows"] if r["by"] == "you"]


def word(text, start, end):
    return {"word": text, "start": start, "end": end}


# ── where a cut by hand lands ─────────────────────────────────────────────────


def placed(words, a, b, duration=10.0):
    start, end, count = place_creator_cut(a, b, SaidOrder.of(words), duration)
    return round(start, 3), round(end, 3), count


def test_a_cut_by_hand_takes_the_word_and_nothing_beside_it(words):
    assert placed(words, *WE, DURATION) == (4.35, 4.5, 1)
    assert placed(words, WE[0], ABOUT[1], DURATION) == (4.35, 5.3, 3)


def test_a_word_under_the_smallest_cut_widens_into_the_quiet_beside_it():
    talk = [word("one", 0.0, 1.0), word("a", 1.05, 1.11), word("two", 1.16, 2.0)]
    assert placed(talk, 1.05, 1.11) == (1.03, 1.13, 1)


def test_the_quiet_is_taken_from_the_side_that_has_it():
    talk = [word("one", 0.0, 1.0), word("a", 1.0, 1.04), word("two", 1.2, 2.0)]
    assert placed(talk, 1.0, 1.04) == (1.0, 1.1, 1)


def test_a_short_word_with_no_quiet_beside_it_is_cut_as_short_as_it_is():
    talk = [word("one", 0.0, 1.0), word("a", 1.0, 1.06), word("two", 1.06, 2.0)]
    assert placed(talk, 1.0, 1.06) == (1.0, 1.06, 1)


def test_widening_stops_at_the_neighbouring_words():
    talk = [word("one", 0.0, 1.0), word("a", 1.01, 1.03), word("two", 1.04, 2.0)]
    start, end, _ = placed(talk, 1.01, 1.03)
    assert (start, end) == (1.0, 1.04) and end - start < edits.MIN_CUT_SECONDS


def test_a_cut_never_takes_a_piece_of_a_word_that_overlaps_it():
    talk = [word("one", 0.0, 1.05), word("a", 1.0, 1.3), word("two", 1.25, 2.0)]
    assert placed(talk, 1.0, 1.3) == (1.05, 1.25, 1)


def test_a_word_with_no_length_and_no_quiet_cannot_be_cut_alone():
    talk = [word("one", 0.0, 1.0), word("a", 1.0, 1.0), word("two", 1.0, 2.0)]
    with pytest.raises(CutRejected, match="no length"):
        placed(talk, 1.0, 1.0)
    assert placed(talk, 1.0, 2.0) == (1.0, 2.0, 2)  # picked with the word beside it


def test_a_word_with_no_length_is_cut_from_the_quiet_around_it():
    talk = [word("one", 0.0, 1.0), word("a", 1.2, 1.2), word("two", 1.4, 2.0)]
    assert placed(talk, 1.2, 1.2) == (1.15, 1.25, 1)


def test_a_span_that_holds_no_word_is_not_a_cut(words):
    with pytest.raises(CutRejected, match="holds no word"):
        placed(words, 6.7, 7.5, DURATION)


def test_the_ends_count_when_picking_words(words):
    said = SaidOrder.of(words)
    middle = (WE[0] + WE[1]) / 2
    assert len(said.picked(middle, middle)) == 1 and len(said.picked(WE[1], TALK[0])) == 0


# ── building the edit ─────────────────────────────────────────────────────────


def built(words, cuts=(), **kwargs):
    return build_edit(list(cuts), words, DURATION, **kwargs)


def test_her_cut_is_saved_as_hers(words):
    out = built(words, creator_cuts=[span(WE)])
    assert out.cuts == [{"start": 4.35, "end": 4.5, "reason": "Cut by you", "source": "you", "kind": "yours"}]
    assert out.creator_placed == [
        {"id": "y4.35-4.50", "start": 4.35, "end": 4.5, "word_start": 4.35, "word_end": 4.5, "words": 1}
    ]
    assert out.applied == [] and out.rejected == []


def test_a_kept_span_does_not_stop_her_cut(words):
    out = built(words, creator_cuts=[span(THE)], keeps=[(12.0, 14.2)])
    assert [(c["start"], c["end"]) for c in out.cuts] == [THE]


def test_her_cut_inside_claudes_cut_removes_nothing_more(words):
    alone = built(words, [RETAKE])
    both = built(words, [RETAKE], creator_cuts=[span(TALK_AGAIN)])
    assert [(c["start"], c["end"], c["source"]) for c in both.cuts] == \
        [(c["start"], c["end"], c["source"]) for c in alone.cuts]
    assert both.cuts[0]["creator_cuts"] == 1


def test_a_cut_whose_words_are_gone_is_counted_and_skipped(words):
    out = built(words, creator_cuts=[{"start": 6.7, "end": 7.5}, span(WE)])
    assert out.creator_lost == 1 and len(out.cuts) == 1


def test_claudes_cut_splits_around_a_word_she_brought_back(words):
    out = built(words, [RETAKE], exact_keeps=[TALK_AGAIN])
    assert [(p["index"], p["put_back"]) for p in out.placed] == [(0, False), (0, False)]
    first, second = out.cuts
    assert first["end"] <= TALK_AGAIN[0] and second["start"] >= TALK_AGAIN[1]
    assert {c["reason"] for c in out.cuts} == {RETAKE["reason"]} and {c["kind"] for c in out.cuts} == {"repeat"}
    assert out.rejected == []
    assert all(a["why"].startswith("split around") for a in out.adjusted)


def test_a_cut_with_every_word_brought_back_reads_as_put_back(words):
    out = built(words, [UM], exact_keeps=[THE_UM])
    assert [p["put_back"] for p in out.placed] == [True] and out.cuts == []
    assert "restored" in out.rejected[0]["why"]


def test_claude_can_cut_over_a_word_brought_back_only_when_the_creator_asks(words):
    out = built(words, [RETAKE], exact_keeps=[TALK_AGAIN], override_keeps=True)
    assert len(out.cuts) == 1 and out.cuts[0]["start"] < TALK_AGAIN[0] < out.cuts[0]["end"]


def test_an_automatic_trim_is_shortened_around_a_word_she_brought_back(words):
    trims = [{"start": 2.45, "end": 3.85, "reason": "filler: um (+1 more)", "kind": "fillers"}]
    whole = built(words, auto_cuts=trims)
    around = built(words, auto_cuts=trims, exact_keeps=[THE_UM])
    assert [c["kind"] for c in whole.cuts] == ["fillers"]
    before, after = around.cuts
    # ClipForge lands each edge on the nearest word edge, so the word is whole and the pause is gone.
    assert (before["start"], before["end"], after["start"], after["end"]) == (2.4, 3.4, 3.7, 3.9)
    assert [(c["kind"], c["source"], c["reason"]) for c in around.cuts] == [
        ("pauses", "auto", "auto: pause: 0.9s"), ("pauses", "auto", "auto: pause: 0.1s"),
    ]
    assert around.auto_skipped_kept == 0


def test_a_trim_with_nothing_left_beside_the_word_goes(words):
    trims = [{"start": 3.38, "end": 3.72, "reason": "filler: um", "kind": "fillers"}]
    out = built(words, auto_cuts=trims, exact_keeps=[THE_UM])
    assert out.cuts == [] and out.auto_skipped_kept == 1


def test_finalizing_near_each_trim_gives_what_clipforge_gives_for_the_whole_take():
    talk = []
    for n in range(40):
        talk += spoken_run("this is one more line of the talk.", 10.0 * n + 0.3, step=0.5)
    # Trims with edges in pauses, on word edges, and inside words.
    trims = [(10.0 * n + 4.2, 10.0 * n + 10.2) for n in range(39)]
    trims += [(10.0 * n + 1.45, 10.0 * n + 1.95) for n in range(0, 40, 3)]
    by_hand = [(10.0 * n + 2.0, 10.0 * n + 2.4) for n in range(0, 40, 5)]
    whole = finalize_removed_ranges(trims, by_hand, talk, silences=[])
    ranges, unsafe = finalize_near(trims, by_hand, talk, [])
    assert ranges == whole.ranges and unsafe == whole.unsafe and len(ranges) > 40


# ── cutting on the page ───────────────────────────────────────────────────────


def test_a_double_click_cuts_one_word(video):
    ctx = edited(video)
    state = treatment.add_cut(ctx, span(WE))
    assert [(r["id"], r["removed"], r["reason"], r["group"], r["state"]) for r in yours(state)] == [
        ("y4.35-4.50", "we", "Cut by you", "yours", "kept_out")
    ]
    assert state["groups"][0]["key"] == "yours" and state["groups"][0]["rows"] == ["y4.35-4.50"]
    assert state["groups"][0]["label"] == "Your cuts" and state["groups"][0]["count"] == 1
    assert state["cut_counts"] == {"yours": 1, "claude": 2}
    assert why_of(state, WE[0]) == treatment.REMOVED_BY_YOU and why_of(state, TALK[0]) == treatment.PLAYING
    changed = state["changed"]
    assert (changed["words_cut"], changed["your_cuts"], changed["keeps_changed"]) == (1, 1, 0)
    assert changed["seconds"] == pytest.approx(-0.2, abs=0.11)
    assert saved(video)["creator_cuts"] == [
        {"id": "y4.35-4.50", "start": 4.35, "end": 4.5, "text": "we", "words": 1}
    ]
    assert state["can_undo"] is True


def test_cutting_a_selection_takes_exactly_the_picked_words(video):
    state = treatment.add_cut(edited(video), {"start": WE[0], "end": ABOUT[1]})
    assert yours(state)[0]["removed"] == "we talk about" and state["changed"]["words_cut"] == 3
    assert why_of(state, 3.9) == treatment.PLAYING and why_of(state, 5.35) == treatment.PLAYING


def test_a_cut_beside_one_of_hers_joins_it(video):
    ctx = edited(video)
    treatment.add_cut(ctx, span(WE))
    state = treatment.add_cut(ctx, span(TALK))
    assert [(r["id"], r["removed"]) for r in yours(state)] == [("y4.35-4.90", "we talk")]
    assert (state["changed"]["your_cuts"], state["changed"]["words_cut"]) == (0, 1)


def test_a_cut_that_overlaps_one_of_hers_joins_it_and_counts_only_the_new_words(video):
    ctx = edited(video)
    treatment.add_cut(ctx, {"start": WE[0], "end": TALK[1]})
    state = treatment.add_cut(ctx, {"start": TALK[0], "end": ABOUT[1]})
    assert [r["removed"] for r in yours(state)] == ["we talk about"]
    assert state["changed"]["words_cut"] == 1


def test_a_cut_between_two_of_hers_joins_all_three(video):
    ctx = edited(video)
    treatment.add_cut(ctx, span(WE))
    treatment.add_cut(ctx, span(ABOUT))
    assert len(yours(treatment.page_state(ctx))) == 2
    state = treatment.add_cut(ctx, span(TALK))
    assert [r["removed"] for r in yours(state)] == ["we talk about"] and state["changed"]["your_cuts"] == -1


def test_a_cut_over_claudes_cut_shows_as_hers_and_changes_no_time(video):
    ctx = edited(video)
    state = treatment.add_cut(ctx, span(TALK_AGAIN))
    assert why_of(state, TALK_AGAIN[0]) == treatment.REMOVED_BY_YOU
    assert why_of(state, 8.0) == treatment.REMOVED_BY_CLAUDE
    assert state["changed"]["seconds"] == 0.0 and state["changed"]["words_cut"] == 1
    assert {r["by"] for r in state["rows"]} == {"you", "claude"}


def test_a_sound_can_be_cut_like_a_word(video):
    state = treatment.add_cut(edited(video), {"start": 6.0, "end": 6.6})
    assert why_of(state, 6.0) == treatment.REMOVED_BY_YOU and state["changed"]["words_cut"] == 1


def test_a_cut_inside_a_part_she_kept_wins_and_the_kept_part_splits(video):
    ctx = edited(video)
    treatment.add_keep(ctx, {"start": 12.1, "end": 12.4, "note": "the point"})
    state = treatment.add_cut(ctx, span(THE))
    assert [(k["id"], k["text"], k["note"]) for k in state["keeps"]] == [
        ("k12.00-12.50", "this is", "the point"), ("k12.75-14.20", "part that matters.", "the point"),
    ]
    assert state["changed"]["keeps_changed"] == 1 and why_of(state, THE[0]) == treatment.REMOVED_BY_YOU


def test_a_cut_at_the_edge_of_a_kept_part_shrinks_it(video):
    ctx = edited(video)
    treatment.add_keep(ctx, {"start": 12.1, "end": 12.4})
    state = treatment.add_cut(ctx, {"start": 12.0, "end": 12.3})
    assert [k["id"] for k in state["keeps"]] == ["k12.35-14.20"]


def test_a_cut_inside_a_cut_she_put_back_wins_and_the_rest_stays_back(video):
    ctx = edited(video)
    retake = next(r for r in treatment.page_state(ctx)["rows"] if r["group"] == "repeat")
    treatment.set_cut_state(ctx, {"id": retake["id"], "state": "put_back"})
    state = treatment.add_cut(ctx, span(TALK_AGAIN))
    assert next(r for r in state["rows"] if r["group"] == "repeat")["state"] == "put_back"
    assert why_of(state, TALK_AGAIN[0]) == treatment.REMOVED_BY_YOU
    assert why_of(state, 8.0) == treatment.PLAYING and why_of(state, 9.05) == treatment.PLAYING
    assert state["changed"]["keeps_changed"] == 1


def test_her_cuts_outlive_a_new_pace_a_switch_and_a_new_edit_from_claude(video):
    ctx = edited(video, auto_tighten=True)
    treatment.add_cut(ctx, span(FOR))
    for body in ({"pace": "max"}, {"pace": "natural"}, {"take_out": {"pauses": False}}):
        state = treatment.change_treatment(ctx, body)
        assert [r["id"] for r in yours(state)] == ["y16.05-16.20"], body
        assert why_of(state, FOR[0]) == treatment.REMOVED_BY_YOU
    result = tools.set_edit(str(video), [UM], auto_tighten=True, pace="fast", **QUIET)
    assert result["creator_cuts_kept"] == 1
    state = treatment.page_state(ctx_for(video))
    assert [r["id"] for r in yours(state)] == ["y16.05-16.20"] and state["cut_counts"] == {"yours": 1, "claude": 1}


def test_the_join_check_runs_on_her_cuts(video):
    def flag_hers(cuts, words, duration, *, labels=None):
        return [{"start": c["start"], "end": c["end"], "source": c["source"],
                 "flags": ["tight"] if c["source"] == "you" else [], "note": "A note.", "why": "Why."} for c in cuts]

    tools.set_edit(str(video), [UM], **QUIET)
    ctx = treatment.load_context(open_project(str(video)), duration=DURATION, silences=no_silences, join_rows=flag_hers)
    state = treatment.add_cut(ctx, span(FOR))
    assert yours(state)[0]["flags"] == ["tight"] and yours(state)[0]["why"] == "Why."
    assert state["groups"][0]["need_a_look"] == 1 and state["need_a_look"] == 1
    assert state["changed"]["new_flags"] == 1


@pytest.mark.parametrize("body, message", [
    ({"start": 6.7, "end": 7.5}, "No word sits between those times"),
    ({"start": 5.0, "end": 4.0}, "start must come before the end"),
    ({"start": 1.0, "end": 25.5}, "outside the video"),
    ({"start": "1", "end": 2}, "must be a number"),
    ({"start": 1.0, "end": 2.0, "note": "x"}, "Unknown field 'note'"),
    ([1.0, 2.0], "Send a JSON object"),
])
def test_a_bad_cut_says_how_to_fix_it(video, body, message):
    with pytest.raises(StudioError, match=message):
        treatment.add_cut(edited(video), body)
    assert "creator_cuts" not in saved(video)


# ── taking a cut back ─────────────────────────────────────────────────────────


def test_taking_back_a_cut_removes_its_row_and_keeps_nothing(video):
    ctx = edited(video)
    treatment.add_cut(ctx, span(WE))
    state = treatment.remove_cut(ctx, {"id": "y4.35-4.50"})
    assert yours(state) == [] and state["keeps"] == [] and "creator_cuts" not in saved(video)
    assert why_of(state, WE[0]) == treatment.PLAYING
    assert (state["changed"]["words_back"], state["changed"]["your_cuts"]) == (1, -1)


def test_taking_back_a_cut_that_sat_on_claudes_leaves_the_word_to_claude(video):
    ctx = edited(video)
    treatment.add_cut(ctx, span(TALK_AGAIN))
    state = treatment.remove_cut(ctx, {"id": "y8.65-9.00"})
    assert why_of(state, TALK_AGAIN[0]) == treatment.REMOVED_BY_CLAUDE


def test_taking_back_one_word_of_a_longer_cut_shortens_it(video):
    ctx = edited(video)
    treatment.add_cut(ctx, {"start": WE[0], "end": ABOUT[1]})
    state = treatment.remove_cut(ctx, {"id": "y4.35-5.30", **span(ABOUT)})
    assert [(r["id"], r["removed"]) for r in yours(state)] == [("y4.35-4.90", "we talk")]
    assert state["changed"]["words_back"] == 1


def test_taking_back_a_word_from_the_middle_splits_the_cut(video):
    ctx = edited(video)
    treatment.add_cut(ctx, {"start": WE[0], "end": ABOUT[1]})
    state = treatment.remove_cut(ctx, {"id": "y4.35-5.30", **span(TALK)})
    assert [r["removed"] for r in yours(state)] == ["we", "about"] and state["changed"]["your_cuts"] == 1
    assert why_of(state, TALK[0]) == treatment.PLAYING


@pytest.mark.parametrize("body, message", [
    ({"id": "y1.00-2.00"}, "not in this video any more"),
    ({"id": "c7.95-10.70"}, "one of Claude's. Put it back instead"),
    ({"id": 7}, "Send the id as a string"),
    ({"id": "y4.35-4.50", "start": 12.55, "end": 12.7}, "not part of that cut"),
])
def test_a_bad_take_back_says_how_to_fix_it(video, body, message):
    ctx = edited(video)
    treatment.add_cut(ctx, span(WE))
    with pytest.raises(StudioError, match=message):
        treatment.remove_cut(ctx, body)


def test_her_cut_cannot_be_put_back_like_one_of_claudes(video):
    ctx = edited(video)
    treatment.add_cut(ctx, span(WE))
    with pytest.raises(StudioError, match="one of yours. Take it back instead"):
        treatment.set_cut_state(ctx, {"id": "y4.35-4.50", "state": "put_back"})


# ── bringing words back ───────────────────────────────────────────────────────


def exact(pair, **more):
    return {**span(pair), "exact": True, **more}


def test_a_word_brought_back_from_claudes_cut_leaves_the_rest_cut(video):
    ctx = edited(video)
    state = treatment.add_keep(ctx, exact(TALK_AGAIN))
    rows = [r for r in state["rows"] if r["group"] == "repeat"]
    assert [r["removed"] for r in rows] == ["today we", "about editing video."]
    assert {r["reason"] for r in rows} == {RETAKE["reason"]} and {r["state"] for r in rows} == {"kept_out"}
    assert why_of(state, TALK_AGAIN[0]) == treatment.PLAYING
    assert why_of(state, 8.0) == treatment.REMOVED_BY_CLAUDE and why_of(state, 9.05) == treatment.REMOVED_BY_CLAUDE
    assert state["keeps"] == [{
        "id": "k8.65-9.00", "start": 8.65, "end": 9.0, "clock": "0:08-0:09", "text": "talk",
        "note": "brought back by you", "exact": True,
    }]
    assert state["changed"]["words_back"] == 1 and state["changed"]["seconds"] > 0


def test_an_exact_keep_does_not_widen_to_the_sentence(video):
    state = treatment.add_keep(edited(video), exact(PART))
    assert [(k["start"], k["end"]) for k in state["keeps"]] == [PART]


def test_bringing_back_every_word_of_claudes_cut_reads_as_put_back_and_can_be_left_out_again(video):
    ctx = edited(video)
    state = treatment.add_keep(ctx, exact(THE_UM))
    um = next(r for r in state["rows"] if r["group"] == "other")
    assert um["state"] == "put_back" and why_of(state, THE_UM[0]) == treatment.PLAYING
    state = treatment.set_cut_state(ctx, {"id": um["id"], "state": "kept_out"})
    assert next(r for r in state["rows"] if r["group"] == "other")["state"] == "kept_out"
    assert state["keeps"] == [] and why_of(state, THE_UM[0]) == treatment.REMOVED_BY_CLAUDE


def um_trim(*_args, **_kwargs):
    """A stand-in plan: at every pace one filler trim takes the "um" and the pause before it."""
    return [{"start": 2.45, "end": 3.85, "reason": "filler: um (+1 more)", "kind": "fillers"}]


def test_a_word_brought_back_from_an_automatic_removal_stays_back_at_every_pace(video, monkeypatch):
    monkeypatch.setattr(treatment, "plan_auto_cuts", um_trim)
    ctx = edited(video, cuts=[], auto_tighten=True)
    assert why_of(treatment.page_state(ctx), THE_UM[0]) == treatment.REMOVED_AUTO
    state = treatment.add_keep(ctx, exact(THE_UM))
    assert why_of(state, THE_UM[0]) == treatment.PLAYING and state["changed"]["words_back"] == 1
    for name in PACES:
        state = treatment.change_treatment(ctx, {"pace": name})
        assert why_of(state, THE_UM[0]) == treatment.PLAYING, name
        assert [t[2] for t in state["trims"]] and set(t[2] for t in state["trims"]) == {"pauses"}, name


def test_a_word_brought_back_from_her_own_cut_comes_out_of_it(video):
    ctx = edited(video)
    treatment.add_cut(ctx, {"start": WE[0], "end": ABOUT[1]})
    state = treatment.add_keep(ctx, exact(TALK))
    assert [r["removed"] for r in yours(state)] == ["we", "about"]
    assert why_of(state, TALK[0]) == treatment.PLAYING and state["changed"]["words_back"] == 1


def test_words_brought_back_side_by_side_are_one_kept_part(video):
    ctx = edited(video)
    treatment.add_keep(ctx, exact(TALK_AGAIN))
    state = treatment.add_keep(ctx, exact((9.05, 9.4)))
    assert [(k["id"], k["text"]) for k in state["keeps"]] == [("k8.65-9.40", "talk about")]


def test_a_word_brought_back_can_be_let_go_again(video):
    ctx = edited(video)
    treatment.add_keep(ctx, exact(TALK_AGAIN))
    state = treatment.remove_keep(ctx, {"id": "k8.65-9.00"})
    assert state["keeps"] == [] and why_of(state, TALK_AGAIN[0]) == treatment.REMOVED_BY_CLAUDE
    assert len([r for r in state["rows"] if r["group"] == "repeat"]) == 1


def test_a_plain_keep_takes_back_her_cut_inside_the_picked_words(video):
    ctx = edited(video)
    treatment.add_cut(ctx, span(THE))
    state = treatment.add_keep(ctx, {"start": 12.4, "end": 12.8})
    assert yours(state) == [] and [k["id"] for k in state["keeps"]] == ["k12.00-14.20"]
    assert why_of(state, THE[0]) == treatment.PLAYING


def test_a_plain_keep_leaves_her_cut_alone_where_she_did_not_pick(video):
    ctx = edited(video)
    treatment.add_cut(ctx, span(THE))
    state = treatment.add_keep(ctx, {"start": 13.2, "end": 13.5})
    assert [r["removed"] for r in yours(state)] == ["the"]
    assert [k["id"] for k in state["keeps"]] == ["k12.00-12.50", "k12.75-14.20"]
    assert why_of(state, THE[0]) == treatment.REMOVED_BY_YOU


@pytest.mark.parametrize("body, message", [
    ({"start": 6.7, "end": 7.5, "exact": True}, "No word sits between those times"),
    ({"start": 4.0, "end": 5.0, "exact": "yes"}, "exact must be true or false"),
    ({"start": 5.0, "end": 5.0}, "start must come before the end"),
])
def test_a_bad_exact_keep_says_how_to_fix_it(video, body, message):
    with pytest.raises(StudioError, match=message):
        treatment.add_keep(edited(video), body)


# ── undo ──────────────────────────────────────────────────────────────────────


def lists_of(video):
    edit = saved(video)
    return edit.get("requested"), edit.get("keep"), edit.get("creator_cuts", [])


def test_undo_takes_back_the_last_cut(video):
    ctx = edited(video)
    before = lists_of(video)
    treatment.add_cut(ctx, span(WE))
    state = treatment.undo(ctx, {})
    assert lists_of(video) == before and yours(state) == [] and state["can_undo"] is False
    assert state["changed"]["your_cuts"] == -1


def test_undo_puts_a_split_kept_part_back_together(video):
    ctx = edited(video)
    treatment.add_keep(ctx, {"start": 12.1, "end": 12.4, "note": "the point"})
    before = lists_of(video)
    treatment.add_cut(ctx, span(THE))
    state = treatment.undo(ctx, {})
    assert lists_of(video) == before and [k["id"] for k in state["keeps"]] == ["k12.00-14.20"]


def test_undo_takes_back_a_bring_back_a_take_back_and_a_keep(video):
    ctx = edited(video)
    treatment.add_cut(ctx, {"start": WE[0], "end": ABOUT[1]})
    for action, body in (
        (treatment.add_keep, exact(TALK_AGAIN)),
        (treatment.remove_cut, {"id": "y4.35-5.30"}),
        (treatment.add_keep, {"start": 12.1, "end": 12.4}),
    ):
        before = lists_of(video)
        action(ctx, body)
        assert lists_of(video) != before
        treatment.undo(ctx, {})
        assert lists_of(video) == before, action.__name__


def test_undo_is_one_step(video):
    ctx = edited(video)
    treatment.add_cut(ctx, span(WE))
    treatment.add_cut(ctx, span(FOR))
    state = treatment.undo(ctx, {})
    assert [r["removed"] for r in yours(state)] == ["we"]
    with pytest.raises(StudioError, match="nothing to undo"):
        treatment.undo(ctx, {})


def test_a_pace_change_leaves_the_last_cut_undoable_and_undo_leaves_the_pace(video):
    ctx = edited(video, auto_tighten=True)
    treatment.add_cut(ctx, span(WE))
    assert treatment.change_treatment(ctx, {"pace": "hard"})["can_undo"] is True
    state = treatment.undo(ctx, {})
    assert yours(state) == [] and state["settings"]["pace"] == "hard"


def test_a_new_edit_from_claude_clears_undo(video):
    ctx = edited(video)
    treatment.add_cut(ctx, span(WE))
    tools.set_edit(str(video), [UM], **QUIET)
    assert treatment.page_state(ctx_for(video))["can_undo"] is False


def test_undo_takes_no_fields(video):
    ctx = edited(video)
    treatment.add_cut(ctx, span(WE))
    with pytest.raises(StudioError, match="Unknown field"):
        treatment.undo(ctx, {"steps": 2})


# ── what the switch takes out ─────────────────────────────────────────────────


def test_the_filler_switch_lists_the_words_it_takes_out_most_first(video, monkeypatch):
    talk = spoken_run("so you know it is like a thing you know and like it works.", 0.5, step=0.5)
    video.with_suffix(".words.json").write_text(json.dumps(talk))

    def over(first, last, reason):
        return {"start": talk[first]["start"], "end": talk[last]["end"], "reason": reason, "kind": "fillers"}

    def plan(*_a, **_k):
        # the second holds "like" and the word before it; the last was filed by a plan from before
        return [over(0, 0, "filler: so"), over(1, 2, "filler: you know"), over(4, 5, "filler: like"),
                over(8, 9, "filler: you know"), over(11, 11, "filler: like"), over(13, 13, "filler: works")]

    monkeypatch.setattr(treatment, "plan_auto_cuts", plan)
    tools.set_edit(str(video), [], auto_tighten=True, pace="fast", **QUIET)
    state = treatment.page_state(ctx_for(video))
    fillers = state["take_out"][1]
    assert fillers["label"] == "Filler words" and fillers["count"] == 3, "counted by the word"
    assert fillers["words"] == [["you know", 2], ["so", 1]], "a word on no filler list is never named: not 'is', not 'works'"
    playing = [w[0] for w in state["words"] if not w[3]]
    assert "is" in playing and "works." in playing, "and never taken"
    assert playing.count("like") == 2, "and neither is a like: that word is Claude's to pick"
    off = treatment.change_treatment(ctx_for(video), {"take_out": {"fillers": False}})
    assert off["take_out"][1]["words"] == fillers["words"], "the list shows what the switch would take, on or off"
    assert off["take_out"][1]["count"] == 3 and not [w for w in off["words"] if w[3]]


def test_the_filler_switch_names_only_the_fillers_of_the_pace(video, monkeypatch):
    talk = spoken_run("so you know it is like a thing you know and like it works.", 0.5, step=0.5)
    video.with_suffix(".words.json").write_text(json.dumps(talk))

    def plan(*_a, **_k):
        return [{"start": talk[i]["start"], "end": talk[j]["end"], "reason": "filler", "kind": "fillers"}
                for i, j in ((1, 2), (5, 5), (10, 10))]

    monkeypatch.setattr(treatment, "plan_auto_cuts", plan)
    # Standard reads ClipForge's third list, which holds "and" and no "you know".
    tools.set_edit(str(video), [], auto_tighten=True, pace="standard", **QUIET)
    state = treatment.page_state(ctx_for(video))
    assert state["take_out"][1]["words"] == [["and", 1]]
    assert [w[0] for w in state["words"] if w[3]] == ["and"]
    assert state["take_out"][1]["by_pace"] == {"natural": 0, "standard": 1, "fast": 2, "tight": 2, "hard": 2, "max": 2}


def test_a_switch_that_takes_no_words_has_an_empty_list(video):
    state = treatment.page_state(edited(video, cuts=[], auto_tighten=True))
    assert [t["words"] for t in state["take_out"]] == [[], [], [], []]


# ── what Claude reads ─────────────────────────────────────────────────────────


def test_get_edit_tells_claude_what_she_cut_and_brought_back_apart_from_its_own(video):
    ctx = edited(video)
    for pair in (WE, THE, FOR):
        treatment.add_cut(ctx, span(pair))
    treatment.add_keep(ctx, exact(TALK_AGAIN))
    treatment.add_keep(ctx, {"start": 13.2, "end": 13.5})
    summary = tools.get_edit(str(video))
    creator = summary["creator"]
    assert [(c["clock"], c["text"]) for c in creator["cuts"]] == [("0:04-0:04", "we"), ("0:12-0:12", "the"), ("0:16-0:16", "for")]
    assert sorted(creator["cut_words"]) == [["for", 1], ["the", 1], ["we", 1]]
    assert [(k["text"], k["start"], k["end"]) for k in creator["brought_back"]] == [("talk", 8.65, 9.0)]
    assert [k["text"] for k in creator["kept"]] == ["this is", "part that matters."]
    assert all(c["reason"] != "Cut by you" for c in summary["cuts"]), "her cuts are not listed as Claude's"
    assert summary["cut_count"] == len(saved(video)["cuts"])


def test_cut_words_count_what_she_cut_most_first():
    cuts = [{"text": t} for t in ("like", "Like,", "you know", "so", "like", "this is the part", "So")]
    assert treatment.cut_word_counts(cuts) == [
        ["like", 3], ["so", 2], ["you know", 1], ["this", 1], ["is", 1], ["the", 1], ["part", 1],
    ]


def test_the_review_tool_counts_her_cuts_apart(video):
    ctx = edited(video)
    treatment.add_cut(ctx, span(WE))
    result = tools.review(str(video), open_browser=False, address_for=lambda _p: "http://127.0.0.1:1/t/", **QUIET)
    assert (result["claude_cuts"], result["creator_cuts"]) == (2, 1)


# ── a cut changes what she cut, and nothing else ──────────────────────────────


def pause_trims(*_args, **_kwargs):
    """A stand-in plan: the pause before "this" and the pause before "thanks" are trimmed at every pace."""
    return [
        {"start": 10.75, "end": 11.95, "reason": "pause: 1.3s", "kind": "pauses"},
        {"start": 14.25, "end": 15.45, "reason": "pause: 1.3s", "kind": "pauses"},
    ]


THIS = (12.0, 12.3)  # the word right after the first trimmed pause


@pytest.mark.parametrize("pair", [WE, THIS, FOR, THE_UM, TALK_AGAIN])
def test_cut_then_undo_gives_back_the_very_same_edit(video, monkeypatch, pair):
    monkeypatch.setattr(treatment, "plan_auto_cuts", pause_trims)
    ctx = edited(video, auto_tighten=True)
    before = treatment.page_state(ctx)
    treatment.add_cut(ctx, span(pair))
    after = treatment.undo(ctx, {})
    assert after["removed"] == before["removed"] and after["trims"] == before["trims"]
    assert after["durations"] == before["durations"]
    assert [r["id"] for r in after["rows"]] == [r["id"] for r in before["rows"]]


@pytest.mark.parametrize("name", list(PACES))
def test_cut_then_undo_gives_back_the_very_same_edit_with_the_real_planner(video, name):
    ctx = edited(video, auto_tighten=True, pace=name)
    before = treatment.page_state(ctx)
    for pair in (WE, THIS, FOR):
        treatment.add_cut(ctx, span(pair))
        after = treatment.undo(ctx, {})
        assert after["removed"] == before["removed"], (name, pair)
        assert after["durations"] == before["durations"]


def test_a_word_cut_beside_a_trimmed_pause_takes_the_word_and_keeps_the_pause_the_pace_leaves(video, monkeypatch):
    monkeypatch.setattr(treatment, "plan_auto_cuts", pause_trims)
    ctx = edited(video, cuts=[], auto_tighten=True, pace="standard")
    before = treatment.page_state(ctx)
    state = treatment.add_cut(ctx, span(THIS))
    changed = state["changed"]
    # "video." ends a sentence; standard leaves 0.4 s after it, with the word there or gone.
    assert changed["seconds"] == pytest.approx(-(THIS[1] - THIS[0]), abs=0.11)
    assert (changed["trims"], changed["trims_added"], changed["trims_dropped"]) == (0, 0, 0)
    assert changed["words_cut"] == 1
    joined = next(c for c in saved(video)["cuts"] if c["source"] == "you")
    assert joined["auto_trims"] == 1 and joined["end"] == pytest.approx(12.35, abs=0.06)
    assert len(state["removed"]) == len(before["removed"])


def test_an_edit_the_recipe_would_now_make_otherwise_is_made_again_before_anything_else(video, monkeypatch):
    monkeypatch.setattr(treatment, "plan_auto_cuts", pause_trims)
    ctx = edited(video, cuts=[UM], auto_tighten=True)
    fresh = saved(video)
    # As an earlier version of the plugin might have saved it: one trim two seconds longer.
    stale = json.loads(json.dumps(fresh))
    trim = next(c for c in stale["cuts"] if c["source"] == "auto")
    trim["start"] = round(trim["start"] - 0.2, 3)
    open_project(str(video)).edit_path.write_text(json.dumps(stale))
    ctx = ctx_for(video)
    assert saved(video)["cuts"] == fresh["cuts"], "loading the page put the edit right"
    state = treatment.add_cut(ctx, span(WE))
    assert state["changed"]["seconds"] == pytest.approx(-(WE[1] - WE[0]), abs=0.06)
    assert state["changed"]["trims_added"] == 0 and state["changed"]["trims_dropped"] == 0


def test_an_edit_saved_by_hand_is_left_as_it_was_saved(video):
    project = open_project(str(video))
    by_hand = {"version": 1, "video": str(video), "duration": DURATION,
               "cuts": [{"start": 2.5, "end": 3.3, "reason": "auto: pause: 0.8s", "source": "auto"}]}
    project.edit_path.write_text(json.dumps(by_hand))
    ctx_for(video)
    assert json.loads(project.edit_path.read_text()) == by_hand
