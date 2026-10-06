"""The taste profile: what the creator's changes on other videos teach the next cut.

Synthetic words only. The talk is conftest's 20 seconds: an "um", a line said
twice, "this is the part that matters." and a sign-off. Extra videos are
copies of the fixture under other names, since a project's folder is named
from its video's path.
"""

import json
import os
import shutil
import time

import pytest

from creator_words import banned_in, decimal_times_in
from lumr_studio import edit as edits
from lumr_studio import taste, tools, treatment
from lumr_studio.errors import StudioError
from lumr_studio.project import open_project, projects_root, write_json_atomic
from lumr_studio.silences import no_silences
from lumr_studio.tools import unmeasured_labels
from lumr_studio.treatment import Treatment

QUIET = {"silences": no_silences, "labels": unmeasured_labels}
DURATION = 20.0

UM = {"start": 3.3, "end": 3.8, "reason": "filler before the line", "kind": "other"}
RETAKE = {"start": 7.9, "end": 10.75, "reason": "second take of the opening line", "kind": "repeat"}
WE = {"start": 4.35, "end": 4.5}
ABOUT = {"start": 4.95, "end": 5.3}


@pytest.fixture(autouse=True)
def fresh_recipes():
    treatment.forget_recipes()
    yield
    treatment.forget_recipes()


def another(video, name):
    """A copy of ``video`` under ``name`` with its own words, so it is a project of its own."""
    path = video.with_name(f"{name}.mp4")
    shutil.copy(video, path)
    shutil.copy(video.with_suffix(".words.json"), path.with_suffix(".words.json"))
    return path


def ctx_for(video):
    return treatment.load_context(open_project(str(video)), duration=DURATION, silences=no_silences)


def edited(video, cuts=(UM, RETAKE), **kwargs):
    tools.set_edit(str(video), list(cuts), **kwargs, **QUIET)
    return ctx_for(video)


def row_id(ctx, kind):
    return next(r["id"] for r in treatment.page_state(ctx)["rows"] if r["group"] == kind)


def taught(video, name, *, put_back=True, cut=(WE,)):
    """Edit a copy of ``video`` the way a creator who fixes things would: put the retake back and cut by hand."""
    ctx = edited(another(video, name))
    if put_back:
        treatment.set_cut_state(ctx, {"id": row_id(ctx, "repeat"), "state": "put_back"})
    for span in cut:
        treatment.add_cut(ctx, span)
    return ctx


def plant(name, edit):
    """Write an ``edit.json`` by hand into a project folder of its own."""
    path = projects_root() / name / "edit.json"
    write_json_atomic(path, edit)
    return path


def planted(*, put_back=(), requested=(), cut_words=(), brought=(), kept=0, pace=None, off=()):
    """The parts of a saved edit that ``creator_changes`` reads, built by hand."""
    asked = Treatment(pace="standard")
    now = Treatment(pace=pace or "standard", take_out=frozenset(k for k in edits.SWITCHES if k not in off))
    keep = [{"id": f"k{s}", "start": s, "end": e, "origin": "put_back", "cut": "c", "note": "Put back by the creator."}
            for s, e in put_back]
    keep += [{"id": f"b{i}", "start": 30.0 + i, "end": 30.5 + i, "origin": "exact", "text": text}
             for i, text in enumerate(brought)]
    keep += [{"id": f"f{i}", "start": 40.0 + i, "end": 41.0 + i, "origin": "flagged", "text": "a part"} for i in range(kept)]
    cuts = [{"start": 50.0 + i, "end": 50.4 + i, "text": text} for i, text in enumerate(cut_words)]
    return {
        "requested": [{"start": s, "end": e, "reason": "why", "kind": k} for s, e, k in requested],
        "keep": keep, "creator_cuts": cuts,
        "set_by_claude": asked.as_dict(), "treatment": now.as_dict(),
    }


# ── nothing learned ───────────────────────────────────────────────────────────


def test_with_no_other_projects_get_edit_has_no_taste(video):
    edited(video)
    assert taste.learn() is None
    assert "taste" not in tools.get_edit(str(video))


def test_other_videos_the_creator_never_changed_teach_nothing(video):
    edited(another(video, "quiet"))
    edited(video)
    assert taste.learn(exclude=open_project(str(video)).root) is None
    assert "taste" not in tools.get_edit(str(video))


# ── what two videos teach a third ─────────────────────────────────────────────


def test_two_videos_teach_the_third(video):
    taught(video, "first")
    taught(video, "second")
    edited(video)
    got = tools.get_edit(str(video))["taste"]
    assert got["videos"] == 2
    assert got["put_back"] == {"repeat": {"put_back": 2, "proposed": 2}}
    assert got["cut_by_hand"] == [["we", 2]]
    assert got["brought_back"] == [] and got["kept_parts"] == 0
    assert got["lessons"] == [
        "Put back 2 of 2 restated-point cuts (repeat) across 2 videos.",
        'Cut "we" by hand 2 times across 2 videos.',
    ]


def test_the_current_video_is_not_counted_twice(video):
    ctx = taught(video, "first")
    project = open_project(str(ctx.project.video))
    assert taste.learn(exclude=project.root) is None
    assert taste.learn()["videos"] == 1
    assert "taste" not in tools.get_edit(str(ctx.project.video))


def test_a_cut_the_creator_put_back_is_counted_by_the_kind_it_overlaps_most(video):
    ctx = edited(another(video, "mixed"), cuts=(UM, RETAKE))
    treatment.set_cut_state(ctx, {"id": row_id(ctx, "other"), "state": "put_back"})
    treatment.set_cut_state(ctx, {"id": row_id(ctx, "repeat"), "state": "put_back"})
    got = taste.learn()
    assert got["put_back"] == {"other": {"put_back": 1, "proposed": 1}, "repeat": {"put_back": 1, "proposed": 1}}


def test_a_put_back_that_matches_none_of_claudes_cuts_is_unknown():
    kept = {"start": 100.0, "end": 101.0}
    assert taste._kind_of(kept, [{"start": 1.0, "end": 2.0, "kind": "repeat"}]) == "unknown"
    assert taste._kind_of(kept, []) == "unknown"
    # a cut with no kind reads as other
    assert taste._kind_of({"start": 1.0, "end": 1.5}, [{"start": 1.0, "end": 2.0}]) == "other"


def test_the_largest_overlap_names_the_kind():
    requested = [{"start": 0.0, "end": 2.0, "kind": "false_start"}, {"start": 2.0, "end": 9.0, "kind": "off_topic"}]
    assert taste._kind_of({"start": 1.5, "end": 6.0}, requested) == "off_topic"


def test_words_brought_back_and_parts_kept_and_the_pace_are_counted(video):
    for name in ("one", "two"):
        ctx = edited(another(video, name), pace="standard", auto_tighten=True)
        treatment.change_treatment(ctx, {"pace": "fast", "take_out": {"fillers": False}})
        treatment.add_keep(ctx, {"start": 15.6, "end": 15.8, "note": "the sign-off"})
        treatment.add_keep(ctx, {"start": 4.35, "end": 4.5, "exact": True})
    got = taste.learn()
    assert got["kept_parts"] == 2
    assert got["brought_back"] == [["we", 2]]
    assert got["pace"] == {"claude": {"standard": 2}, "creator": {"fast": 2}}
    assert got["take_out"] == {"fillers": {"off": 2, "on": 0}}
    assert "Switched filler words off in 2 of 2 videos." in got["lessons"]
    assert "Chose a faster pace than Claude in 2 videos: fast." in got["lessons"]
    assert 'Brought back "we" 2 times.' in got["lessons"]


# ── the lessons ───────────────────────────────────────────────────────────────


def test_a_lesson_needs_two_videos_or_three_events():
    plant("a", planted(cut_words=["like"] * 2, put_back=[(1.0, 2.0)], requested=[(1.0, 2.0, "repeat")]))
    got = taste.learn()
    assert got["cut_by_hand"] == [["like", 2]] and got["put_back"] == {"repeat": {"put_back": 1, "proposed": 1}}
    assert got["lessons"] == []  # two events in one video is not yet a habit
    plant("b", planted(cut_words=["like"]))
    assert taste.learn()["lessons"] == ['Cut "like" by hand 3 times across 2 videos.']


def test_three_events_in_one_video_are_a_lesson():
    plant("a", planted(cut_words=["like"] * 3, brought=["actually"] * 3))
    got = taste.learn()
    assert got["lessons"] == ['Cut "like" by hand 3 times across 1 video.', 'Brought back "actually" 3 times.']


def test_the_lessons_in_full():
    requested = [(0.0, 1.0, "repeat"), (2.0, 3.0, "repeat"), (4.0, 5.0, "repeat")]
    plant("a", planted(requested=requested, put_back=[(0.0, 1.0), (2.0, 3.0)], cut_words=["like"] * 8 + ["so"] * 2,
                       brought=["Actually,"], pace="fast", off=["fillers", "likes"]))
    plant("b", planted(requested=requested + [(6.0, 7.0, "repeat")], put_back=[(4.0, 5.0)],
                       cut_words=["like"] * 6 + ["so"] * 3, brought=["actually", "actually"], pace="fast", off=["fillers"]))
    plant("c", planted(cut_words=["so"], pace="tight"))
    got = taste.learn()
    assert got["videos"] == 3
    assert got["put_back"] == {"repeat": {"put_back": 3, "proposed": 7}}
    assert got["cut_by_hand"] == [["like", 14], ["so", 6]]
    assert got["brought_back"] == [["actually", 3]]
    assert got["pace"] == {"claude": {"standard": 3}, "creator": {"fast": 2, "tight": 1}}
    assert got["take_out"] == {"fillers": {"off": 2, "on": 0}, "likes": {"off": 1, "on": 0}}
    assert got["lessons"] == [
        "Put back 3 of 7 restated-point cuts (repeat) across 2 videos.",
        'Cut "like" by hand 14 times across 2 videos.',
        'Cut "so" by hand 6 times across 3 videos.',
        'Brought back "actually" 3 times.',
        "Switched filler words off in 2 of 3 videos.",
        "Chose a faster pace than Claude in 3 videos: fast.",
    ]


def test_a_gentler_pace_is_a_lesson_too():
    for name in ("a", "b"):
        plant(name, planted(pace="natural", cut_words=["um"]))
    assert "Chose a slower pace than Claude in 2 videos: natural." in taste.learn()["lessons"]


def test_the_lists_are_the_top_ten_most_first():
    words = [f"w{i:02d}" for i in range(12)]
    plant("a", planted(cut_words=[w for i, w in enumerate(words) for _ in range(i + 1)]))
    top = taste.learn()["cut_by_hand"]
    assert len(top) == 10 and top[0] == ["w11", 12] and top[-1] == ["w02", 3]


def test_only_short_bring_backs_count_as_words():
    plant("a", planted(brought=["Actually,", "you know", "this is a whole sentence"]))
    assert taste.learn()["brought_back"] == [["actually", 1], ["you know", 1]]


# ── a project that can't be read ──────────────────────────────────────────────


def test_a_corrupt_edit_in_another_project_is_skipped(video):
    taught(video, "good")
    broken = projects_root() / "broken"
    broken.mkdir(parents=True)
    (broken / "edit.json").write_text("{not json")
    (projects_root() / "list").mkdir()
    (projects_root() / "list" / "edit.json").write_text("[1, 2]")
    (projects_root() / "stray-file.txt").write_text("not a project")
    (projects_root() / "empty-folder").mkdir()
    plant("odd", {"keep": "nope", "creator_cuts": 5, "set_by_claude": {"pace": 7}})
    edited(video)
    got = tools.get_edit(str(video))["taste"]
    assert got["videos"] == 1


# ── forgetting ────────────────────────────────────────────────────────────────


def folder_of(ctx):
    """The name a project's folder goes by in taste.json."""
    return ctx.project.root.name


def marker():
    return json.loads((projects_root() / taste.TASTE_FILE).read_text())


def test_forgetting_everything_writes_ids_and_the_treatment_and_no_date(video):
    ctx = taught(video, "first")
    got = taste.forget_all()
    assert got == {"videos": 1}
    data = marker()
    assert set(data) == {"version", "forgotten", "ignored_words", "ignored_kinds"}
    assert data["version"] == 2 and data["ignored_words"] == [] and data["ignored_kinds"] == []
    kept = data["forgotten"][folder_of(ctx)]
    assert set(kept) == {"keep", "creator_cuts", "ratings", "treatment"}
    assert kept["keep"] == [k["id"] for k in ctx.edit["keep"]] and len(kept["keep"]) == 1
    assert kept["creator_cuts"] == [c["id"] for c in ctx.edit["creator_cuts"]] and len(kept["creator_cuts"]) == 1
    assert kept["ratings"] == [] and kept["treatment"] == ctx.edit["treatment"]


def test_forgetting_everything_leaves_every_edit_as_it_was(video):
    ctx = taught(video, "first")
    before = open_project(str(ctx.project.video)).edit_path.read_bytes()
    taste.forget_all()
    assert taste.learn() is None
    assert open_project(str(ctx.project.video)).edit_path.read_bytes() == before


def test_forgetting_sticks_when_the_old_video_is_edited_again(video):
    """The file's date moves on a new edit; what it taught before must not come back."""
    first = taught(video, "first")
    taste.forget_all()
    path = open_project(str(first.project.video)).edit_path
    later = time.time() + 5
    os.utime(path, (later, later))
    assert taste.learn() is None
    # A new change of the same video counts, and only that one.
    treatment.add_cut(first, {"start": 4.95, "end": 5.3})
    got = taste.learn()
    assert got["videos"] == 1 and got["cut_by_hand"] == [["about", 1]]
    assert got["put_back"] == {}, "the put-back she forgot stays forgotten"


def test_the_pace_and_switches_come_back_only_when_they_change(video):
    ctx = edited(another(video, "first"), pace="standard", auto_tighten=True)
    treatment.change_treatment(ctx, {"pace": "fast", "take_out": {"fillers": False}})
    assert taste.learn()["pace"] == {"claude": {"standard": 1}, "creator": {"fast": 1}}
    taste.forget_all()
    assert taste.learn() is None
    treatment.change_treatment(ctx, {"pace": "tight"})
    got = taste.learn()
    assert got["pace"] == {"claude": {"standard": 1}, "creator": {"tight": 1}}
    assert got["take_out"] == {"fillers": {"off": 1, "on": 0}}


def test_a_change_made_after_forgetting_counts_and_an_earlier_one_does_not(video):
    taught(video, "first")
    taste.forget_all()
    taught(video, "second")
    got = taste.learn()
    assert got["videos"] == 1
    assert got["cut_by_hand"] == [["we", 1]] and got["put_back"] == {"repeat": {"put_back": 1, "proposed": 1}}


def test_forgetting_twice_keeps_the_first_ids_and_adds_the_new(video):
    ctx = taught(video, "first")
    first = marker_after(taste.forget_all)["forgotten"][folder_of(ctx)]
    treatment.add_cut(ctx, {"start": 4.95, "end": 5.3})
    second = marker_after(taste.forget_all)["forgotten"][folder_of(ctx)]
    assert set(first["creator_cuts"]) < set(second["creator_cuts"]) and len(second["creator_cuts"]) == 2
    assert second["keep"] == first["keep"]
    assert taste.learn() is None


def marker_after(action):
    action()
    return marker()


def test_a_project_added_after_forgetting_counts(video):
    taught(video, "first")
    taste.forget_all()
    assert len(marker()["forgotten"]) == 1
    taught(video, "second")
    assert taste.learn()["videos"] == 1


def test_forgetting_a_word_drops_it_from_cut_and_brought_back(video):
    for name in ("a", "b"):
        plant(name, planted(cut_words=["like", "so"], brought=["Actually,", "you know"]))
    assert taste.forget_word("Like,") == "like"
    assert taste.forget_word("actually") == "actually"
    assert taste.forget_word("Like") == "like"
    assert marker()["ignored_words"] == ["like", "actually"]
    got = taste.learn()
    assert got["cut_by_hand"] == [["so", 2]] and got["brought_back"] == [["you know", 2]]
    assert not any("like" in line or "actually" in line for line in got["lessons"])


def test_a_project_with_only_forgotten_words_no_longer_counts():
    plant("a", planted(cut_words=["like"]))
    taste.forget_word("like")
    assert taste.learn() is None and taste.summary_for_page() is None


def test_forgetting_a_kind_drops_it_from_put_back():
    requested = [(0.0, 1.0, "repeat"), (2.0, 3.0, "false_start")]
    for name in ("a", "b"):
        plant(name, planted(requested=requested, put_back=[(0.0, 1.0), (2.0, 3.0)]))
    assert set(taste.learn()["put_back"]) == {"repeat", "false_start"}
    assert taste.forget_kind("repeat") == "repeat"
    got = taste.learn()
    assert set(got["put_back"]) == {"false_start"}
    assert all("(repeat)" not in line for line in got["lessons"])


def test_only_a_real_kind_can_be_forgotten():
    for kind in (*edits.CUT_KINDS, "likes"):
        taste.forget_kind(kind)
    assert marker()["ignored_kinds"] == [*edits.CUT_KINDS, "likes"]
    for bad in ("pace", "", "Repeat", "unknown"):
        with pytest.raises(StudioError, match="is not a kind of cut"):
            taste.forget_kind(bad)
    with pytest.raises(StudioError, match="Send the word"):
        taste.forget_word("  ,, ")


def test_a_marker_that_cannot_be_read_forgets_nothing(video):
    taught(video, "first")
    for text in ("{not json", "[1, 2]", '{"forgotten": 5, "ignored_words": "so", "ignored_kinds": [1]}', "{}"):
        (projects_root() / taste.TASTE_FILE).write_text(text)
        assert taste.learn()["videos"] == 1, text
    # a damaged marker is replaced by the next forget
    (projects_root() / taste.TASTE_FILE).write_text("{not json")
    taste.forget_word("so")
    assert marker()["ignored_words"] == ["so"] and marker()["version"] == 2


def test_learning_works_with_no_marker_at_all(video):
    taught(video, "first")
    assert not (projects_root() / taste.TASTE_FILE).exists()
    assert taste.learn()["videos"] == 1
    assert not (projects_root() / taste.TASTE_FILE).exists(), "learning never writes"


# ── the forget_taste tool ─────────────────────────────────────────────────────


def test_the_tool_forgets_one_word_and_says_what_is_left(video):
    for name in ("a", "b"):
        plant(name, planted(cut_words=["like", "so"]))
    got = tools.forget_taste(word="Like")
    assert got["forgot"] == 'the word "like"'
    assert got["taste"]["cut_by_hand"] == [["so", 2]]


def test_the_tool_forgets_one_kind():
    for name in ("a", "b"):
        plant(name, planted(requested=[(0.0, 1.0, "repeat")], put_back=[(0.0, 1.0)], cut_words=["so"]))
    got = tools.forget_taste(kind="repeat")
    assert got["forgot"] == 'the kind "repeat"' and got["taste"]["put_back"] == {}


def test_the_tool_forgets_everything_and_taste_is_null(video):
    ctx = taught(video, "first")
    assert tools.forget_taste(everything=True) == {"forgot": "everything", "taste": None}
    assert marker()["forgotten"][folder_of(ctx)]["creator_cuts"]


def test_the_tool_takes_exactly_one_input():
    for kwargs in ({}, {"everything": True, "word": "so"}, {"word": "so", "kind": "repeat"},
                   {"everything": False}, {"everything": False, "word": "so"}):
        with pytest.raises(StudioError):
            tools.forget_taste(**kwargs)
    assert not (projects_root() / taste.TASTE_FILE).exists(), "a refused call writes nothing"
    with pytest.raises(StudioError, match="not a kind of cut"):
        tools.forget_taste(kind="nonsense")


def test_the_tool_is_forgetting_twice_the_same_as_once(video):
    taught(video, "first")
    tools.forget_taste(everything=True)
    once = marker()
    tools.forget_taste(everything=True)
    assert marker() == once
    tools.forget_taste(word="so")
    tools.forget_taste(word="so")
    assert marker()["ignored_words"] == ["so"]


def test_the_server_offers_the_tool_without_a_video():
    import asyncio

    from lumr_studio import offering, server

    assert "forget_taste" in offering.EDITOR_TOOLS
    listed = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    tool = listed["forget_taste"]
    assert set(tool.input_schema["properties"]) == {"everything", "word", "kind"}
    assert not tool.input_schema.get("required")
    assert tool.annotations.read_only_hint is False
    assert tool.annotations.destructive_hint is False
    assert tool.annotations.idempotent_hint is True
    assert "ask the creator" in tool.description.lower()
    assert "except job_status and forget_taste" in server.INSTRUCTIONS


def test_the_server_tool_answers_and_refuses_the_way_the_others_do():
    from mcp.server.mcpserver.exceptions import ToolError

    from lumr_studio import server

    plant("a", planted(cut_words=["so"] * 3))
    answered = server.forget_taste(word="so")
    assert answered.structured_content == {"forgot": 'the word "so"', "taste": None}
    with pytest.raises(ToolError, match="exactly one"):
        server.forget_taste()


# ── the page ──────────────────────────────────────────────────────────────────


def test_the_page_state_says_how_many_videos_it_learns_from(video):
    ctx = edited(video)
    assert treatment.page_state(ctx)["taste"] is None
    taught(video, "first")
    taught(video, "second")
    assert treatment.page_state(ctx)["taste"] == {"videos": 2}


def test_the_page_does_not_count_its_own_video(video):
    ctx = taught(video, "first")
    assert treatment.page_state(ctx)["taste"] is None


def test_the_page_forgets_nothing_itself():
    from lumr_studio import review_server

    assert not hasattr(treatment, "forget_taste")
    assert "api/taste/forget" not in review_server.TREATMENT_ACTIONS


def test_the_page_state_drops_out_when_everything_is_forgotten(video):
    ctx = edited(video)
    taught(video, "first")
    assert treatment.page_state(ctx)["taste"] == {"videos": 1}
    taste.forget_all()
    assert treatment.page_state(ctx)["taste"] is None


# ── rating a cut ──────────────────────────────────────────────────────────────


def row_of(ctx, rid):
    return next(r for r in treatment.page_state(ctx)["rows"] if r["id"] == rid)


def saved_ratings(ctx):
    return json.loads(open_project(str(ctx.project.video)).edit_path.read_text()).get("ratings", [])


def rated(video, name, sides, cuts=(UM, RETAKE)):
    """A copy of ``video`` where the creator rated Claude's cuts: ``sides`` maps a kind to good or bad."""
    ctx = edited(another(video, name), cuts=cuts)
    for kind, side in sides.items():
        treatment.rate_cut(ctx, {"id": row_id(ctx, kind), "rating": side})
    return ctx


def test_a_good_rating_is_saved_and_the_cut_stays(video):
    ctx = edited(video)
    rid = row_id(ctx, "repeat")
    edited_before = treatment.page_state(ctx)["durations"]["edited"]
    state = treatment.rate_cut(ctx, {"id": rid, "rating": "good"})
    row = row_of(ctx, rid)
    assert row["rating"] == "good" and row["state"] == "kept_out"
    assert state["durations"]["edited"] == edited_before
    assert saved_ratings(ctx) == [{
        "id": rid.replace("c", "r", 1), "row": rid, "rating": "good", "source": "claude", "kind": "repeat",
        "start": row["start"], "end": row["end"], "reason": RETAKE["reason"],
    }]
    assert "put_back" not in (treatment.creator_changes(ctx.edit) or {})


def test_a_wrong_rating_puts_the_cut_back_the_way_put_back_does(video):
    ctx = edited(video)
    rid = row_id(ctx, "repeat")
    before = treatment.page_state(ctx)["durations"]["edited"]
    state = treatment.rate_cut(ctx, {"id": rid, "rating": "bad"})
    row = row_of(ctx, rid)
    assert row["rating"] == "bad" and row["state"] == "put_back"
    assert state["durations"]["edited"] > before, "the words play again"
    assert state["changed"]["seconds"] > 0
    assert [k["origin"] for k in ctx.edit["keep"]] == ["put_back"]
    (rating,) = saved_ratings(ctx)
    assert rating["rating"] == "bad" and rating["kind"] == "repeat" and rating["reason"] == RETAKE["reason"]
    seen = treatment.creator_changes(ctx.edit)
    assert [p["reason"] for p in seen["put_back"]] == [RETAKE["reason"]]
    assert [r["reason"] for r in seen["rated"]["bad"]] == [RETAKE["reason"]]


def test_the_wrong_rating_and_put_back_end_in_the_same_edit(video):
    a, b = edited(another(video, "rated")), edited(another(video, "pushed"))
    treatment.rate_cut(a, {"id": row_id(a, "repeat"), "rating": "bad"})
    treatment.set_cut_state(b, {"id": row_id(b, "repeat"), "state": "put_back"})
    keep = lambda ctx: [(k["start"], k["end"], k["origin"]) for k in ctx.edit["keep"]]  # noqa: E731
    assert keep(a) == keep(b)
    assert a.edit["cuts"] == b.edit["cuts"]


def test_pressing_the_same_rating_again_clears_it(video):
    ctx = edited(video)
    rid = row_id(ctx, "other")
    treatment.rate_cut(ctx, {"id": rid, "rating": "good"})
    state = treatment.rate_cut(ctx, {"id": rid, "rating": None})
    assert row_of(ctx, rid)["rating"] is None and saved_ratings(ctx) == []
    assert "rated" not in (treatment.creator_changes(ctx.edit) or {})
    assert "ratings" not in ctx.edit and state["rows"]


def test_clearing_a_wrong_rating_does_not_cut_it_again(video):
    ctx = edited(video)
    rid = row_id(ctx, "repeat")
    treatment.rate_cut(ctx, {"id": rid, "rating": "bad"})
    treatment.rate_cut(ctx, {"id": rid, "rating": None})
    row = row_of(ctx, rid)
    assert row["rating"] is None and row["state"] == "put_back"
    treatment.set_cut_state(ctx, {"id": rid, "state": "kept_out"})
    assert row_of(ctx, rid)["state"] == "kept_out"


def test_a_new_rating_replaces_the_old_one_and_each_cut_has_its_own(video):
    ctx = edited(video)
    a, b = row_id(ctx, "other"), row_id(ctx, "repeat")
    treatment.rate_cut(ctx, {"id": a, "rating": "good"})
    treatment.rate_cut(ctx, {"id": b, "rating": "good"})
    treatment.rate_cut(ctx, {"id": a, "rating": "bad"})
    assert sorted((r["kind"], r["rating"]) for r in saved_ratings(ctx)) == [("other", "bad"), ("repeat", "good")]


def test_ratings_outlive_a_new_set_edit_and_every_page_change(video):
    ctx = edited(video)
    treatment.rate_cut(ctx, {"id": row_id(ctx, "repeat"), "rating": "good"})
    treatment.change_treatment(ctx, {"pace": "fast"})
    treatment.add_cut(ctx, WE)
    assert len(saved_ratings(ctx)) == 1
    tools.set_edit(str(video), [UM, RETAKE], auto_tighten=True, **QUIET)
    assert [r["rating"] for r in saved_ratings(ctx)] == ["good"]
    assert tools.get_edit(str(video))["creator"]["rated"]["good"][0]["kind"] == "repeat"


def test_only_claudes_cuts_can_be_rated_and_the_body_is_checked(video):
    ctx = edited(video)
    treatment.add_cut(ctx, WE)
    yours = next(r["id"] for r in treatment.page_state(ctx)["rows"] if r["by"] == "you")
    rid = row_id(ctx, "repeat")
    with pytest.raises(StudioError, match="one of yours"):
        treatment.rate_cut(ctx, {"id": yours, "rating": "good"})
    with pytest.raises(StudioError, match="not one of Claude's cuts"):
        treatment.rate_cut(ctx, {"id": "c1.00-2.00", "rating": "good"})
    with pytest.raises(StudioError, match="rating must be"):
        treatment.rate_cut(ctx, {"id": rid, "rating": "meh"})
    with pytest.raises(StudioError, match="Send a rating"):
        treatment.rate_cut(ctx, {"id": rid})
    with pytest.raises(StudioError, match="Unknown field"):
        treatment.rate_cut(ctx, {"id": rid, "rating": "good", "kind": "other"})
    with pytest.raises(StudioError, match="Send the id"):
        treatment.rate_cut(ctx, {"id": "q1", "rating": "good"})
    assert saved_ratings(ctx) == []


def test_a_wrong_rating_on_a_cut_that_is_gone_saves_nothing(video):
    ctx = edited(video)
    with pytest.raises(StudioError, match="not one of Claude's cuts"):
        treatment.rate_cut(ctx, {"id": "c99.00-100.00", "rating": "bad"})
    assert saved_ratings(ctx) == []
    assert ctx.edit.get("keep", []) == []


def test_the_rate_route_is_in_the_route_table():
    from lumr_studio import review_server

    assert review_server.TREATMENT_ACTIONS["api/rate"] is treatment.rate_cut


def test_creator_changes_lists_what_was_rated_with_kind_and_reason(video):
    ctx = rated(video, "mixed", {"other": "good", "repeat": "bad"})
    got = treatment.creator_changes(ctx.edit)["rated"]
    assert set(got) == {"good", "bad"}
    (good,), (bad,) = got["good"], got["bad"]
    assert good["kind"] == "other" and good["reason"] == UM["reason"] and set(good) == {"clock", "start", "end", "reason", "kind"}
    assert bad["kind"] == "repeat" and bad["reason"] == RETAKE["reason"]
    assert tools.get_edit(str(ctx.project.video))["creator"]["rated"] == got


def planted_ratings(edit, ratings):
    """A planted edit with ratings: ``(kind, side, source)`` each."""
    return {**edit, "ratings": [
        {"id": f"r{i}.00-{i}.50", "row": f"c{i}.00-{i}.50", "rating": side, "source": source, "kind": kind,
         "start": float(i), "end": i + 0.5, "reason": "why"}
        for i, (kind, side, source) in enumerate(ratings)
    ]}


def test_learn_adds_up_the_ratings_by_kind():
    plant("a", planted_ratings(planted(), [("repeat", "good", "claude")] * 3 + [("likes", "bad", "pick")] * 2))
    plant("b", planted_ratings(planted(), [("repeat", "good", "claude"), ("likes", "bad", "pick"), ("other", "bad", "claude")]))
    got = taste.learn()
    assert got["videos"] == 2
    assert got["ratings"] == {"repeat": {"good": 4, "bad": 0}, "likes": {"good": 0, "bad": 3}, "other": {"good": 0, "bad": 1}}
    assert got["lessons"] == [
        "Marked 4 restated-point cuts (repeat) good across 2 videos.",
        "Marked 3 filler-word picks wrong across 2 videos.",
    ]


def test_a_rating_lesson_needs_two_videos_or_three_events():
    plant("a", planted_ratings(planted(), [("off_topic", "bad", "claude")] * 2))
    assert taste.learn()["ratings"] == {"off_topic": {"good": 0, "bad": 2}} and taste.learn()["lessons"] == []
    plant("b", planted_ratings(planted(), [("off_topic", "bad", "claude")]))
    assert taste.learn()["lessons"] == ["Marked 3 off-topic cuts (off_topic) wrong across 2 videos."]
    plant("c", planted_ratings(planted(), [("false_start", "good", "claude")] * 3))
    assert "Marked 3 false-start cuts (false_start) good across 1 video." in taste.learn()["lessons"]


def test_a_project_with_only_ratings_counts_and_the_profile_has_no_ratings_without_them():
    assert taste.learn() is None
    plant("a", planted_ratings(planted(), [("repeat", "good", "claude")]))
    assert taste.learn()["videos"] == 1 and taste.summary_for_page() == {"videos": 1}
    plant("b", planted(cut_words=["so"]))
    plant("a", planted())
    assert taste.learn()["ratings"] == {}


def test_learn_reads_ratings_made_through_the_page(video):
    rated(video, "one", {"repeat": "good", "other": "bad"})
    rated(video, "two", {"repeat": "good"})
    got = taste.learn()
    assert got["ratings"] == {"repeat": {"good": 2, "bad": 0}, "other": {"good": 0, "bad": 1}}
    assert "Marked 2 restated-point cuts (repeat) good across 2 videos." in got["lessons"]


def test_a_wrong_cut_counts_once_as_a_bad_rating_and_not_as_a_put_back(video):
    rated(video, "one", {"repeat": "bad"})
    taught(video, "two", cut=())  # a plain put-back of the same kind still counts
    got = taste.learn()
    assert got["ratings"] == {"repeat": {"good": 0, "bad": 1}}
    assert got["put_back"] == {"repeat": {"put_back": 1, "proposed": 2}}, "only the plain put-back"


def test_a_put_back_over_a_bad_rating_is_not_counted_but_a_neighbour_is():
    edit = planted(put_back=[(5.0, 7.0), (7.0, 8.0), (12.0, 13.0)],
                   requested=[(5.0, 7.0, "repeat"), (7.0, 8.0, "repeat"), (12.0, 13.0, "other")])
    wrong = {"id": "r5.50-6.50", "row": "c5.00-7.00", "rating": "bad", "source": "claude", "kind": "repeat",
             "start": 5.5, "end": 6.5, "reason": "why"}
    plant("a", {**edit, "ratings": [wrong]})
    got = taste.learn()
    assert got["put_back"] == {"repeat": {"put_back": 1, "proposed": 2}, "other": {"put_back": 1, "proposed": 1}}
    assert got["ratings"] == {"repeat": {"good": 0, "bad": 1}}, "the edges that only touch do not match"


def test_forgotten_ratings_drop_out_and_new_ones_count(video):
    first = rated(video, "one", {"repeat": "good"})
    rated(video, "two", {"repeat": "good"})
    assert taste.learn()["ratings"] == {"repeat": {"good": 2, "bad": 0}}
    taste.forget_all()
    assert taste.learn() is None
    assert marker()["forgotten"][folder_of(first)]["ratings"] == [r["id"] for r in saved_ratings(first)]
    treatment.rate_cut(first, {"id": row_id(first, "other"), "rating": "good"})
    assert taste.learn()["ratings"] == {"other": {"good": 1, "bad": 0}}


def test_a_forgotten_kind_drops_its_ratings_and_a_forgotten_word_leaves_them():
    plant("a", planted_ratings(planted(cut_words=["so"]), [("repeat", "good", "claude")] * 3 + [("likes", "bad", "pick")] * 3))
    taste.forget_kind("likes")
    assert taste.learn()["ratings"] == {"repeat": {"good": 3, "bad": 0}}
    taste.forget_word("so")
    assert taste.learn()["ratings"] == {"repeat": {"good": 3, "bad": 0}}
    taste.forget_kind("repeat")
    assert taste.learn() is None


def test_a_rating_with_an_odd_kind_reads_as_other_and_broken_ones_are_skipped():
    plant("a", planted_ratings(planted(), [("mystery", "good", "claude")]))
    edit = json.loads((projects_root() / "a" / "edit.json").read_text())
    edit["ratings"] += ["nope", {"rating": "meh", "start": 1.0, "end": 2.0}, {"rating": "good", "start": "x", "end": 2}]
    plant("a", edit)
    assert taste.learn()["ratings"] == {"other": {"good": 1, "bad": 0}}


def test_the_rating_words_read_plainly():
    for name in ("a", "b"):
        plant(name, planted_ratings(planted(), [(k, side, "pick" if k == "likes" else "claude")
                                                  for k in (*edits.CUT_KINDS, "likes") for side in ("good", "bad")]))
    got = taste.learn()
    assert len(got["lessons"]) == taste.LESSONS_PER_SORT
    for line in got["lessons"]:
        assert not banned_in(line) and not decimal_times_in(line) and "—" not in line, line
    every = taste._lessons_rated(
        {k: {"good": 4, "bad": 4} for k in (*edits.CUT_KINDS, "likes")},
        {(k, s): {"a", "b"} for k in (*edits.CUT_KINDS, "likes") for s in ("good", "bad")},
    )
    assert every and all(not banned_in(line) and "—" not in line for line in every)


# ── the words the creator reads ───────────────────────────────────────────────


def test_the_lessons_use_no_editor_slang_and_no_decimal_times():
    requested = [(0.0, 1.0, "repeat"), (2.0, 3.0, "false_start"), (4.0, 5.0, "off_topic"), (6.0, 7.0, "other")]
    for name in ("a", "b"):
        plant(name, planted(requested=requested, put_back=[(s, s + 1.0) for s, _e, _k in requested],
                            cut_words=["like", "so"], brought=["actually"], pace="hard", off=list(edits.SWITCHES)))
    lessons = taste.learn()["lessons"]
    assert len(lessons) >= 8
    for line in lessons:
        assert not banned_in(line), line
        assert not decimal_times_in(line), line
        assert "—" not in line
