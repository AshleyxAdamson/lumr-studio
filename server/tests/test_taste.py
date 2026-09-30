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


def test_forget_writes_only_a_marker():
    got = taste.forget()
    data = json.loads((projects_root() / taste.TASTE_FILE).read_text())
    assert data == {"version": 1, "forgotten_at": got["forgotten_at"]}
    assert got["forgotten_at"].endswith("+00:00")


def test_forget_hides_earlier_projects_and_a_project_changed_after_counts_again(video):
    first = taught(video, "first")
    assert taste.learn()["videos"] == 1
    taste.forget()
    assert taste.learn() is None
    assert open_project(str(first.project.video)).edit_path.exists(), "the edit stays as it was"
    # Changed after forgetting: the file's time moves past the marker.
    path = open_project(str(first.project.video)).edit_path
    later = time.time() + 5
    os.utime(path, (later, later))
    assert taste.learn()["videos"] == 1
    # Edited for real after forgetting.
    taught(video, "second")
    os.utime(open_project(str(video.with_name("second.mp4"))).edit_path, (later, later))
    assert taste.learn()["videos"] == 2


def test_a_marker_that_cannot_be_read_forgets_nothing(video):
    taught(video, "first")
    (projects_root() / taste.TASTE_FILE).write_text("{not json")
    assert taste.learn()["videos"] == 1
    write_json_atomic(projects_root() / taste.TASTE_FILE, {"version": 1, "forgotten_at": "yesterday"})
    assert taste.learn()["videos"] == 1


def test_forgetting_leaves_every_edit_as_it_was(video):
    ctx = taught(video, "first")
    before = open_project(str(ctx.project.video)).edit_path.read_bytes()
    taste.forget()
    assert open_project(str(ctx.project.video)).edit_path.read_bytes() == before


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


def test_forget_on_the_page_answers_the_new_state(video):
    ctx = edited(video)
    taught(video, "first")
    assert treatment.page_state(ctx)["taste"] == {"videos": 1}
    state = treatment.forget_taste(ctx, {})
    assert state["taste"] is None
    assert taste.learn() is None


def test_forget_on_the_page_takes_no_fields(video):
    with pytest.raises(treatment.StudioError, match="Unknown field"):
        treatment.forget_taste(edited(video), {"all": True})


def test_the_forget_route_is_in_the_route_table():
    from lumr_studio import review_server

    assert review_server.TREATMENT_ACTIONS["api/taste/forget"] is treatment.forget_taste


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
