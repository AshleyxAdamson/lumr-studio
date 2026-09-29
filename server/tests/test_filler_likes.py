"""Filler likes: Claude finds every place a word is said, picks the fillers by reading, and the creator has one switch.

Synthetic words only. The talk says "like" four times: a verb, a clean filler, one
that runs into its neighbours, and a sentence starter.
"""

import json

import pytest
from lumr_studio.engine.audio_boundaries import Silence

from conftest import FakeAligner
from lumr_studio import edit as edits
from lumr_studio import tools, treatment, word_finder, word_times
from lumr_studio.edit import CutRejected, SaidOrder, SilenceIndex, build_edit, parse_pick, pick_id, why_not_clean
from lumr_studio.errors import StudioError
from lumr_studio.pace import PACES
from lumr_studio.project import open_project
from lumr_studio.silences import no_silences
from lumr_studio.tools import unmeasured_labels

QUIET = {"silences": no_silences, "labels": unmeasured_labels}
DURATION = 20.0

ROWS = [
    ("I", 0.50, 0.70), ("like", 0.80, 1.00), ("jazz.", 1.10, 1.60),
    ("So", 2.50, 2.70), ("like,", 2.80, 3.00), ("three", 3.10, 3.40), ("sections.", 3.45, 4.00),
    ("It", 5.00, 5.20), ("was", 5.25, 5.45), ("a", 5.50, 5.60), ("bit", 5.65, 5.90), ("like", 5.91, 6.10),
    ("a", 6.11, 6.20), ("mess.", 6.25, 6.80),
    ("Like,", 8.00, 8.30), ("how", 8.40, 8.60), ("am", 8.65, 8.80), ("I", 8.85, 8.95), ("here?", 9.00, 9.50),
    ("You", 11.00, 11.20), ("know", 11.25, 11.50), ("it", 11.60, 11.80), ("works.", 11.85, 12.40),
    ("Thanks", 14.00, 14.50), ("for", 14.55, 14.70), ("watching.", 14.75, 15.50),
]
TALK = [{"word": w, "start": a, "end": b} for w, a, b in ROWS]
VERB, FILLER, WEDGED, STARTER = "w0.800-1.000", "w2.800-3.000", "w5.910-6.100", "w8.000-8.300"
THREE = {"start": 3.1, "end": 3.4}
OPENING_LINE = {"start": 0.4, "end": 1.7, "reason": "the opening line, said again later", "kind": "repeat"}


@pytest.fixture(autouse=True)
def fresh_recipes():
    treatment.forget_recipes()
    yield
    treatment.forget_recipes()


@pytest.fixture
def talk(video):
    video.with_suffix(".words.json").write_text(json.dumps(TALK))
    return video


def pick(pid, reason="a pause word"):
    return {"id": pid, "reason": reason}


def ctx_for(video):
    return treatment.load_context(open_project(str(video)), duration=DURATION, silences=no_silences)


def picked(video, ids=(FILLER, STARTER), cuts=(), **kwargs):
    result = tools.set_edit(str(video), list(cuts), picks=[pick(i) for i in ids], **kwargs, **QUIET)
    return ctx_for(video), result


def saved(video):
    return json.loads(open_project(str(video)).edit_path.read_text())


def likes(state):
    return state["take_out"][3]


def why_of(state, start):
    return next(w[4] for w in state["words"] if w[1] == start)


# ── whether a cut would be clean ──────────────────────────────────────────────


def clean(start, end, silences=()):
    said = SaidOrder.of(TALK)
    return why_not_clean(said.picked(start, end), said, SilenceIndex(list(silences)), DURATION)


def test_a_word_with_quiet_either_side_is_clean_to_cut():
    assert clean(2.8, 3.0) == "" and clean(0.8, 1.0) == "" and clean(8.0, 8.3) == ""


def test_a_word_that_runs_into_its_neighbour_is_not():
    assert clean(5.91, 6.1) == "cutting it would clip the word before"
    talk = [dict(w) for w in TALK]
    talk[11]["start"] = 5.95  # now clear of "bit", still 0.01 s from "a"
    said = SaidOrder.of(talk)
    assert why_not_clean(said.picked(5.95, 6.1), said, SilenceIndex([]), DURATION) == "cutting it would clip the next word"


def test_a_measured_silence_at_the_edge_makes_that_side_clean():
    assert clean(5.91, 6.1, [Silence(5.895, 5.915)]) == "cutting it would clip the next word"
    assert clean(5.91, 6.1, [Silence(5.895, 5.915), Silence(6.095, 6.12)]) == ""
    assert clean(5.91, 6.1, [Silence(5.0, 5.5), Silence(7.0, 7.5)]) != "", "a silence elsewhere does not count"


def test_a_word_stretched_over_a_pause_is_not_trusted():
    talk = [{"word": "well", "start": 1.0, "end": 1.3}, {"word": "like", "start": 1.5, "end": 3.4},
            {"word": "learn", "start": 3.6, "end": 4.0}]
    said = SaidOrder.of(talk)
    assert why_not_clean(said.picked(1.5, 3.4), said, SilenceIndex([]), DURATION) == \
        "its time in the transcript is too long to be right"


def test_clean_is_judged_by_the_edges_the_aligner_gave_and_the_cut_takes_the_words_whole_room():
    # "like," with room for its sound: the sound runs from "So" into it and from it into "three", so by
    # their rooms the three words touch. The aligner placed them 0.1 s apart, and that is what counts.
    import short_word
    from lumr_studio.edit import place_creator_cut

    roomy = [dict(w) for w in TALK]
    so, like, three = roomy[3], roomy[4], roomy[5]
    so.update(end=2.75, aligned_start=2.5, aligned_end=2.7)
    like.update(start=2.75, end=3.05, aligned_start=2.8, aligned_end=3.0)
    three.update(start=3.05, aligned_start=3.1, aligned_end=3.4)
    said = SaidOrder.of(roomy)
    picked = said.picked(2.75, 3.05)
    assert [w["word"] for w in said.tokens[picked.start:picked.stop]] == ["like,"]
    assert why_not_clean(picked, said, SilenceIndex([]), DURATION) == ""
    assert place_creator_cut(2.75, 3.05, said, DURATION)[:2] == (2.75, 3.05), "none of the word is left behind"
    # On the made-up take "take" is clean to cut too, and its cut takes all of its sound.
    take = word_times.with_room(short_word.aligned(), short_word.silences())
    said = SaidOrder.of(take)
    assert why_not_clean(said.picked(*short_word.TAKE), said, SilenceIndex(short_word.silences()), DURATION) == ""
    assert place_creator_cut(*short_word.TAKE, said, DURATION)[:2] == short_word.TAKE


def test_a_word_whose_room_is_too_long_to_be_one_word_is_not_trusted():
    roomy = [dict(w) for w in TALK]
    roomy[4].update(end=4.2 - 1.15, aligned_start=2.8, aligned_end=3.0)  # 0.25 s: fine
    said = SaidOrder.of(roomy)
    assert why_not_clean(said.picked(2.8, 3.05), said, SilenceIndex([]), DURATION) == ""
    roomy[14].update(end=9.4 - 0.05, aligned_start=8.0, aligned_end=8.3)  # "Like," with 1.35 s of room
    roomy = [w for w in roomy if w["word"] not in ("how", "am", "I", "here?") or w["start"] < 8]
    said = SaidOrder.of(roomy)
    assert why_not_clean(said.picked(8.0, 9.35), said, SilenceIndex([]), DURATION) == \
        "its time in the transcript is too long to be right"


def test_the_first_and_the_last_word_have_one_neighbour_only():
    said = SaidOrder.of(TALK)
    assert why_not_clean(said.picked(0.5, 0.7), said, SilenceIndex([]), DURATION) == ""
    assert why_not_clean(said.picked(14.75, 15.5), said, SilenceIndex([]), DURATION) == ""


# ── find_words ────────────────────────────────────────────────────────────────


def lines_of(found):
    return [line for line in found["text"].splitlines() if line.startswith("w")]


def test_find_words_lists_every_place_with_the_words_around_it(talk):
    found = tools.find_words(str(talk), ["like"], **QUIET)
    assert found["words"] == [{"word": "like", "said": 4, "clean": 3, "already_out": 0}]
    assert found["listed"] == 4 and found["next_start"] is None and found["word_times"] == "estimated"
    assert lines_of(found) == [
        "w0.800-1.000 0:00 I [like] jazz. So like, three sections. It q.10/.10",
        "w2.800-3.000 0:02 I like jazz. So [like,] three sections. It was a bit q.10/.10",
        "w5.910-6.100 0:05 three sections. It was a bit [like] a mess. Like, how am I q.01/.01"
        " NOT CLEAN: cutting it would clip the word before",
        "w8.000-8.300 0:08 was a bit like a mess. [Like,] how am I here? You know q1.20/.10",
    ]
    text = found["text"].splitlines()
    assert text[0] == "like: said 4 times, 3 clean to cut, 1 not, 0 already out."
    assert text[1].startswith("Each line: id, clock") and text[-1] == "END"


def test_find_words_takes_phrases_and_several_words_at_once(talk):
    found = tools.find_words(str(talk), ["you know", "So", "right"], **QUIET)
    assert [(w["word"], w["said"]) for w in found["words"]] == [("you know", 1), ("so", 1), ("right", 0)]
    assert [line.split()[0] for line in lines_of(found)] == ["w2.500-2.700", "w11.000-11.500"]
    assert "[You know]" in lines_of(found)[1]


def test_find_words_says_what_the_saved_edit_removes_already_and_who_does(talk):
    ctx, _ = picked(talk, [STARTER], cuts=[OPENING_LINE])
    treatment.add_cut(ctx, {"start": 2.8, "end": 3.0})
    found = tools.find_words(str(talk), ["like"], **QUIET)
    notes = {line.split()[0]: line.split(" q", 1)[1] for line in lines_of(found)}
    assert notes[VERB].endswith("OUT:claude") and notes[FILLER].endswith("OUT:creator")
    assert notes[STARTER].endswith("OUT:pick") and "OUT" not in notes[WEDGED]
    assert found["words"][0]["already_out"] == 3


def test_find_words_works_before_any_edit_is_saved(talk):
    assert tools.find_words(str(talk), ["like"], **QUIET)["listed"] == 4
    assert not open_project(str(talk)).edit_path.exists()


def test_a_long_answer_stops_at_its_budget_and_says_where_to_go_on():
    words = []
    for i in range(300):
        t = i * 2.0
        words += [{"word": "and", "start": t, "end": t + 0.3}, {"word": "like", "start": t + 0.4, "end": t + 0.6},
                  {"word": "this.", "start": t + 0.7, "end": t + 1.2}]
    places = word_finder.find_places(words, [["like"]], [], 600.0)
    first = word_finder.pack_places(places, [["like"]], budget_chars=6000)
    assert len(first["text"]) <= 6000 and first["listed"] < 300 and first["words"][0]["said"] == 300
    assert first["text"].splitlines()[-1] == f"NEXT {first['next_start']!r}"
    rest = word_finder.pack_places(places, [["like"]], start=first["next_start"], budget_chars=10**6)
    assert first["listed"] + rest["listed"] == 300 and rest["text"].endswith("END")


def test_a_hundred_and_twelve_places_fit_in_one_answer():
    words = []
    for i in range(112):
        t = i * 11.0
        words += [{"word": w, "start": round(t + 0.5 * k, 2), "end": round(t + 0.5 * k + 0.4, 2)}
                  for k, w in enumerate("and honestly it really always felt like something completely different altogether somehow.".split())]
    packed = word_finder.pack_places(word_finder.find_places(words, [["like"]], [], 1300.0), [["like"]])
    assert packed["listed"] == 112 and packed["next_start"] is None
    assert len(packed["text"]) < word_finder.PACK_BUDGET_CHARS


@pytest.mark.parametrize("words, message", [
    ([], "one to 5 words or short phrases"),
    ("like", "one to 5 words or short phrases"),
    (["a", "b", "c", "d", "e", "f"], "at most 5 at a time"),
    (["..."], "holds no word to look for"),
    (["one two three four five"], "at most 4 words"),
    ([7], "one to 5 words or short phrases"),
])
def test_a_bad_list_of_words_says_how_to_fix_it(talk, words, message):
    with pytest.raises(StudioError, match=message):
        tools.find_words(str(talk), words, **QUIET)


# ── handing the picks back ────────────────────────────────────────────────────


def test_an_id_holds_the_words_own_times():
    assert pick_id(312.395, 312.541) == "w312.395-312.541"
    assert edits.span_of_pick_id("w312.395-312.541") == (312.395, 312.541)


@pytest.mark.parametrize("raw, message", [
    ("w2.800-3.000", "is not an object"),
    ({"id": "c2.80-3.00", "reason": "x"}, "not an id from find_words"),
    ({"id": "w2.8", "reason": "x"}, "not an id from find_words"),
    ({"id": "w3.0-2.8", "reason": "x"}, "not an id from find_words"),
    ({"id": FILLER}, "has no reason"),
    ({"id": FILLER, "reason": "x" * 201}, "reason over 200 characters"),
    ({"id": "w2.850-3.000", "reason": "x"}, "names no word as the transcript is timed now"),
    ({"id": "w6.900-7.100", "reason": "x"}, "names no word as the transcript is timed now"),
    ({"id": "w2.800-3.300", "reason": "x"}, "names no word"),
])
def test_a_bad_pick_says_why(raw, message):
    with pytest.raises(CutRejected, match=message):
        parse_pick(0, raw, SaidOrder.of(TALK))
    picks = {p["id"] for p in [parse_pick(0, pick(FILLER), SaidOrder.of(TALK))]}
    assert picks == {FILLER}


def test_a_pick_is_placed_the_way_her_own_word_cuts_are():
    out = build_edit([], TALK, DURATION, picks=[{"id": FILLER, "start": 2.8, "end": 3.0}])
    hers = build_edit([], TALK, DURATION, creator_cuts=[{"start": 2.8, "end": 3.0}])
    assert [(c["start"], c["end"]) for c in out.cuts] == [(c["start"], c["end"]) for c in hers.cuts] == [(2.8, 3.0)]
    assert out.cuts[0] == {"start": 2.8, "end": 3.0, "reason": "A filler word Claude picked", "source": "pick", "kind": "likes"}
    assert out.picks == [{"id": FILLER, "start": 2.8, "end": 3.0, "word_start": 2.8, "word_end": 3.0,
                          "status": "out", "why": ""}]


def test_a_pick_under_the_smallest_cut_is_cut_and_not_refused():
    talk = [{"word": "into", "start": 1.0, "end": 1.3}, {"word": "like", "start": 1.4, "end": 1.46},
            {"word": "three", "start": 1.56, "end": 1.9}]
    out = build_edit([], talk, DURATION, picks=[{"start": 1.4, "end": 1.46}])
    assert [(c["start"], c["end"]) for c in out.cuts] == [(1.38, 1.48)] and out.rejected == []


def test_a_pick_that_is_not_clean_is_left_in_with_why():
    out = build_edit([], TALK, DURATION, picks=[{"id": WEDGED, "start": 5.91, "end": 6.1}])
    assert out.cuts == []
    assert [(p["status"], p["why"]) for p in out.picks] == [("left_in", "cutting it would clip the word before")]


def test_set_edit_takes_picks_under_a_name_of_their_own(talk):
    ctx, result = picked(talk, [FILLER, WEDGED, STARTER], cuts=[OPENING_LINE])
    assert [c["reason"] for c in result["applied"]] == [OPENING_LINE["reason"]]
    assert result["picks"] == {
        "picked": 3, "switch_on": True, "out": 2,
        "left_in": [{"id": WEDGED, "clock": "0:05", "text": "like", "why": "cutting it would clip the word before"}],
        "kept_by_the_creator": [], "rejected": [],
    }
    edit = saved(talk)
    assert [p["id"] for p in edit["picks"]] == [FILLER, WEDGED, STARTER]
    assert edit["picks"][0] == {"id": FILLER, "start": 2.8, "end": 3.0, "text": "like,", "reason": "a pause word"}
    assert [r["start"] for r in edit["requested"]] == [0.4], "picks are no part of the cuts"
    assert [c["source"] for c in edit["cuts"]] == ["claude", "pick", "pick"]


def test_a_refused_pick_is_reported_and_the_rest_are_saved(talk):
    result = tools.set_edit(str(talk), [], picks=[pick(FILLER), pick("w7.000-7.200"), pick(FILLER, "again")], **QUIET)
    assert result["picks"]["picked"] == 1 and result["picks"]["out"] == 1
    assert [r["index"] for r in result["picks"]["rejected"]] == [1]
    assert "Call find_words again" in result["picks"]["rejected"][0]["why"]


def test_when_every_pick_is_refused_nothing_is_saved(talk):
    picked(talk, [FILLER])
    before = saved(talk)
    with pytest.raises(StudioError, match="Every pick was refused, so the saved edit was left unchanged"):
        tools.set_edit(str(talk), [], picks=[pick("w7.000-7.200")], **QUIET)
    assert saved(talk) == before


def test_a_new_list_replaces_the_old_one_and_leaving_it_out_keeps_it(talk):
    picked(talk, [FILLER, STARTER])
    tools.set_edit(str(talk), [OPENING_LINE], **QUIET)
    assert [p["id"] for p in saved(talk)["picks"]] == [FILLER, STARTER]
    tools.set_edit(str(talk), [OPENING_LINE], picks=[pick(STARTER)], **QUIET)
    assert [p["id"] for p in saved(talk)["picks"]] == [STARTER]
    result = tools.set_edit(str(talk), [OPENING_LINE], picks=[], **QUIET)
    assert saved(talk)["picks"] == [] and result["picks"]["picked"] == 0 and "cleared" not in result


def test_get_edit_lists_the_picks_apart_from_the_cuts(talk):
    picked(talk, [FILLER, STARTER], cuts=[OPENING_LINE])
    summary = tools.get_edit(str(talk))
    assert [c["reason"] for c in summary["cuts"]] == [OPENING_LINE["reason"]]
    assert summary["picks"] == {"switch_on": True, "picked": 2, "picks": [
        {"id": FILLER, "clock": "0:02", "text": "like,", "reason": "a pause word"},
        {"id": STARTER, "clock": "0:08", "text": "Like,", "reason": "a pause word"},
    ]}
    assert "creator" not in summary
    tools.set_edit(str(talk), [], **QUIET)
    fresh = open_project(str(talk))
    fresh.edit_path.unlink()
    assert "picks" not in tools.get_edit(str(talk))


# ── the switch on the page ────────────────────────────────────────────────────


def test_before_claude_has_picked_the_switch_is_not_ready(talk):
    tools.set_edit(str(talk), [OPENING_LINE], **QUIET)
    state = treatment.page_state(ctx_for(talk))
    assert [t["key"] for t in state["take_out"]] == ["pauses", "fillers", "repeats", "likes"]
    assert likes(state) == {
        "key": "likes", "label": "Filler likes", "count": 0, "seconds": 0, "by_pace": {name: 0 for name in PACES},
        "words": [], "picked": 0, "said": 4, "left_in": 0, "kept_by_you": 0, "ready": False, "word": "like",
    }
    assert state["settings"]["take_out"]["likes"] is True
    assert all(w[4] != 4 for w in state["words"])


def test_once_claude_has_picked_the_switch_is_on_and_counts_them(talk):
    ctx, _ = picked(talk, [FILLER, WEDGED, STARTER])
    state = treatment.page_state(ctx)
    entry = likes(state)
    assert (entry["ready"], entry["count"], entry["picked"], entry["said"], entry["left_in"], entry["kept_by_you"]) == \
        (True, 2, 3, 4, 1, 0)
    assert entry["seconds"] == 0.5 and entry["words"] == [["like", 2]]
    assert entry["picked"] == entry["count"] + entry["left_in"] + entry["kept_by_you"]
    assert state["settings"]["take_out"]["likes"] is True
    assert [why_of(state, t) for t in (0.8, 2.8, 5.91, 8.0)] == [0, 4, 0, 4]
    assert [[2.8, 3.0], [8.0, 8.3]] == state["removed"] and state["trims"] == []


def test_an_empty_list_from_claude_is_ready_with_nothing_picked(talk):
    ctx, _ = picked(talk, [])
    assert (likes(treatment.page_state(ctx))["ready"], likes(treatment.page_state(ctx))["picked"]) == (True, 0)


def test_picked_words_are_no_rows_under_claudes_cuts(talk):
    ctx, _ = picked(talk, [FILLER, STARTER], cuts=[OPENING_LINE])
    state = treatment.page_state(ctx)
    assert [r["reason"] for r in state["rows"]] == [OPENING_LINE["reason"]]
    assert state["cut_counts"] == {"yours": 0, "claude": 1}
    assert sum(g["count"] for g in state["groups"]) == 1


def test_a_flag_on_a_picked_word_is_not_counted_where_no_row_can_show_it(talk):
    def flag_everything(cuts, words, duration, *, labels=None):
        return [{"start": c["start"], "end": c["end"], "source": c["source"], "flags": ["tight"], "note": "", "why": "Why."}
                for c in cuts]

    tools.set_edit(str(talk), [], picks=[pick(FILLER)], **QUIET)
    ctx = treatment.load_context(open_project(str(talk)), duration=DURATION, silences=no_silences, join_rows=flag_everything)
    assert treatment.page_state(ctx)["need_a_look"] == 0


def test_switching_likes_off_puts_every_pick_back_and_on_takes_them_out_again(talk):
    ctx, _ = picked(talk, [FILLER, STARTER])
    off = treatment.change_treatment(ctx, {"take_out": {"likes": False}})
    assert off["removed"] == [] and off["settings"]["take_out"]["likes"] is False
    assert off["changed"]["likes"] == -2 and off["changed"]["seconds"] == pytest.approx(0.5)
    assert (likes(off)["count"], likes(off)["picked"], likes(off)["ready"]) == (2, 2, True), "the count shows on or off"
    assert all(w[4] == 0 for w in off["words"])
    assert [p["id"] for p in saved(talk)["picks"]] == [FILLER, STARTER], "the list is kept while the switch is off"
    on = treatment.change_treatment(ctx, {"take_out": {"likes": True}})
    assert on["removed"] == [[2.8, 3.0], [8.0, 8.3]] and on["changed"]["likes"] == 2


def test_the_switch_leaves_alone_what_she_cut_by_hand_and_what_she_brought_back(talk):
    ctx, _ = picked(talk, [FILLER, STARTER])
    treatment.add_cut(ctx, {"start": 0.8, "end": 1.0})                      # one Claude kept: hers now
    back = treatment.add_keep(ctx, {"start": 8.0, "end": 8.3, "exact": True})  # one Claude picked: back in
    assert [why_of(back, t) for t in (0.8, 2.8, 8.0)] == [3, 4, 0]
    assert (likes(back)["count"], likes(back)["kept_by_you"], back["changed"]["likes"]) == (1, 1, -1)
    off = treatment.change_treatment(ctx, {"take_out": {"likes": False}})
    assert [why_of(off, t) for t in (0.8, 2.8, 8.0)] == [3, 0, 0]
    on = treatment.change_treatment(ctx, {"take_out": {"likes": True}})
    assert [why_of(on, t) for t in (0.8, 2.8, 8.0)] == [3, 4, 0], "the word she brought back stays back"
    assert [k["id"] for k in on["keeps"]] == ["k8.00-8.30"] and [r["id"] for r in on["rows"]] == ["y0.80-1.00"]


def test_a_picked_word_in_a_part_she_kept_stays_in(talk):
    ctx, _ = picked(talk, [FILLER])
    state = treatment.add_keep(ctx, {"start": 3.1, "end": 3.3})  # widens to the whole sentence
    assert why_of(state, 2.8) == 0 and (likes(state)["count"], likes(state)["kept_by_you"]) == (0, 1)
    state = treatment.remove_keep(ctx, {"id": state["keeps"][0]["id"]})
    assert why_of(state, 2.8) == 4 and likes(state)["count"] == 1


def test_she_can_cut_by_hand_one_that_claude_picked_and_take_her_cut_back(talk):
    ctx, _ = picked(talk, [FILLER])
    state = treatment.add_cut(ctx, {"start": 2.8, "end": 3.0})
    assert why_of(state, 2.8) == 3 and state["changed"]["seconds"] == 0.0
    state = treatment.remove_cut(ctx, {"id": "y2.80-3.00"})
    assert why_of(state, 2.8) == 4


def test_a_picked_word_inside_one_of_claudes_cuts_reads_as_that_cut(talk):
    ctx, _ = picked(talk, [VERB, FILLER], cuts=[OPENING_LINE])
    state = treatment.page_state(ctx)
    assert [why_of(state, t) for t in (0.5, 0.8, 1.1, 2.8)] == [2, 2, 2, 4]
    rid = state["rows"][0]["id"]
    state = treatment.set_cut_state(ctx, {"id": rid, "state": "put_back"})
    assert [why_of(state, t) for t in (0.5, 0.8, 1.1)] == [0, 0, 0], "putting the cut back keeps the whole line"
    assert likes(state)["kept_by_you"] == 1


def test_picks_outlive_a_new_pace_a_slider_move_and_a_switch(talk):
    ctx, _ = picked(talk, [FILLER, STARTER], auto_tighten=True)
    for body in ({"pace": "max"}, {"pace": "natural"}, {"fine": {"gap_length": 0.35, "rhythm": 3.5}},
                 {"take_out": {"pauses": False}}):
        state = treatment.change_treatment(ctx, body)
        assert [why_of(state, t) for t in (2.8, 8.0)] == [4, 4], body
        assert likes(state)["count"] == 2


def test_what_she_did_by_hand_outlives_a_new_pick_list(talk):
    ctx, _ = picked(talk, [FILLER, STARTER])
    treatment.add_cut(ctx, {"start": 14.55, "end": 14.7})
    treatment.add_keep(ctx, {"start": 8.0, "end": 8.3, "exact": True})
    result = tools.set_edit(str(talk), [], picks=[pick(VERB), pick(STARTER)], **QUIET)
    assert result["creator_cuts_kept"] == 1
    assert [k["id"] for k in result["picks"]["kept_by_the_creator"]] == [STARTER]
    state = treatment.page_state(ctx_for(talk))
    assert [why_of(state, t) for t in (0.8, 2.8, 8.0, 14.55)] == [4, 0, 0, 3]


def filler_trim(*_args, **_kwargs):
    """A stand-in plan: one trim ClipForge named for "like," at 2.8, and one pause trim the pause at 7."""
    return [{"start": 2.75, "end": 3.05, "reason": "filler: like", "kind": "fillers"},
            {"start": 6.85, "end": 7.95, "reason": "pause: 1.2s", "kind": "pauses"}]


def test_the_pace_has_no_say_over_a_like_that_claude_picked_or_left(talk, monkeypatch):
    monkeypatch.setattr(treatment, "plan_auto_cuts", filler_trim)
    ctx, _ = picked(talk, [FILLER], auto_tighten=True)
    on = treatment.page_state(ctx)
    fillers = on["take_out"][1]
    assert (likes(on)["count"], fillers["count"], fillers["words"]) == (1, 0, [])
    assert why_of(on, 2.8) == 4 and len([r for r in on["removed"] if r[0] < 4]) == 1
    off = treatment.change_treatment(ctx, {"take_out": {"likes": False}})
    assert (off["take_out"][1]["count"], off["take_out"][1]["words"]) == (0, []), "Filler words never holds a like"
    assert why_of(off, 2.8) == 0, "with the picks off the word plays: the pace takes no like"
    assert off["changed"]["likes"] == -1 and [r for r in off["removed"] if r[0] < 4] == []
    for name in PACES:
        assert why_of(treatment.change_treatment(ctx_for(talk), {"pace": name}), 2.8) == 0, name


def test_a_picked_word_beside_a_trimmed_pause_leaves_the_pause_the_pace_leaves(talk, monkeypatch):
    monkeypatch.setattr(treatment, "plan_auto_cuts", filler_trim)
    ctx, _ = picked(talk, [STARTER], auto_tighten=True, pace="standard")
    state = treatment.page_state(ctx)
    joined = next(c for c in saved(talk)["cuts"] if c["source"] == "pick")
    assert joined["auto_trims"] == 1 and joined["end"] >= 8.3
    # "mess." ends a sentence at 6.8, and "how" starts at 8.4: standard leaves 0.4 s between them.
    left = (joined["start"] - 6.8) + (8.4 - joined["end"])
    assert left == pytest.approx(PACES["standard"].sentence_pause, abs=0.01)
    assert why_of(state, 8.0) == 4 and state["take_out"][0]["count"] == 0


def test_cut_then_undo_gives_back_the_same_edit_with_picks_in_it(talk):
    ctx, _ = picked(talk, [FILLER, STARTER], auto_tighten=True)
    before = treatment.page_state(ctx)
    treatment.add_keep(ctx, {"start": 2.8, "end": 3.0, "exact": True})
    after = treatment.undo(ctx, {})
    assert after["removed"] == before["removed"] and likes(after) == likes(before)


def test_the_picks_move_with_their_words_when_the_word_times_are_measured(talk):
    picked(talk, [FILLER, STARTER])
    word_times.align_project(open_project(str(talk)), aligner=FakeAligner(), silences=no_silences)
    state = treatment.page_state(ctx_for(talk))
    assert [p["id"] for p in saved(talk)["picks"]] == ["w2.850-2.950", "w8.075-8.225"]
    assert [why_of(state, t) for t in (2.85, 8.075)] == [4, 4] and likes(state)["count"] == 2


@pytest.mark.parametrize("body, message", [
    ({"take_out": {"likes": "off"}}, "take_out.likes must be true or false"),
    ({"take_out": {"like": False}}, "'like', which is not a kind. Use pauses, fillers, repeats, likes"),
])
def test_a_bad_switch_says_how_to_fix_it(talk, body, message):
    ctx, _ = picked(talk)
    with pytest.raises(StudioError, match=message):
        treatment.change_treatment(ctx, body)


def test_get_edit_says_when_she_switched_the_picks_off(talk):
    ctx, _ = picked(talk)
    treatment.change_treatment(ctx, {"take_out": {"likes": False}})
    summary = tools.get_edit(str(talk))
    assert summary["creator"]["take_out"] == {"likes": False} and summary["picks"]["switch_on"] is False
    result = tools.set_edit(str(talk), [], picks=[pick(FILLER)], **QUIET)
    assert result["picks"]["switch_on"] is False and result["picks"]["out"] == 1 and "switched Filler likes off" in result["picks"]["note"]
    assert treatment.page_state(ctx_for(talk))["removed"] == [], "her switch stays off through a new pick list"
