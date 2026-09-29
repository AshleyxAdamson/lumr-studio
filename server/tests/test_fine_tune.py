"""Fine tune: the two sliders under the pace, and the pace that reads custom.

Synthetic words only (conftest's 20 second talk).
"""

import json

import pytest

from lumr_studio import autocuts, pace, tools, treatment
from lumr_studio.errors import StudioError
from lumr_studio.pace import CUSTOM, PACES, check_fine, custom_level, fine_ranges, get_pace, level_for
from lumr_studio.project import open_project, projects_root
from lumr_studio.silences import no_silences
from lumr_studio.tools import unmeasured_labels
from lumr_studio.treatment import Treatment

QUIET = {"silences": no_silences, "labels": unmeasured_labels}
DURATION = 20.0
UM = {"start": 3.3, "end": 3.8, "reason": "filler before the line", "kind": "other"}
SIX_STOPS = ["natural", "standard", "fast", "tight", "hard", "max"]


@pytest.fixture(autouse=True)
def fresh_recipes():
    treatment.forget_recipes()
    yield
    treatment.forget_recipes()


def ctx_for(video):
    return treatment.load_context(open_project(str(video)), duration=DURATION, silences=no_silences)


def edited(video, cuts=(UM,), **kwargs):
    kwargs.setdefault("auto_tighten", True)
    tools.set_edit(str(video), list(cuts), **kwargs, **QUIET)
    return ctx_for(video)


def saved(video):
    return json.loads(open_project(str(video)).edit_path.read_text())


# ── the sliders ───────────────────────────────────────────────────────────────


def test_the_two_sliders_say_their_range_step_label_and_every_position():
    ranges = fine_ranges()
    gap, rhythm = ranges["gap_length"], ranges["rhythm"]
    assert (gap["min"], gap["max"], gap["step"], gap["label"]) == (0.15, 1.5, 0.05, "Cut pauses longer than")
    assert (rhythm["min"], rhythm["max"], rhythm["step"], rhythm["label"]) == (1, 5, 0.5, "Speech kept between cuts")
    assert [p["value"] for p in gap["positions"]] == [round(0.15 + 0.05 * i, 2) for i in range(28)]
    assert [p["value"] for p in rhythm["positions"]] == [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    assert all(set(p) == {"value", "shown", "said"} for p in gap["positions"] + rhythm["positions"])


def test_the_rhythm_slider_shows_the_speech_kept_and_never_a_rhythm():
    shown = [p["shown"] for p in fine_ranges()["rhythm"]["positions"]]
    assert shown == ["3 s", "2.5 s", "2 s", "1.6 s", "1.2 s", "0.85 s", "0.5 s", "0.25 s", "0 s"]
    said = [p["said"] for p in fine_ranges()["rhythm"]["positions"]]
    assert said[0] == "3 seconds" and said[-1] == "0 seconds" and said[3] == "1.6 seconds"
    # The whole numbers are the stops' own spacings.
    by_value = {p["value"]: p["shown"] for p in fine_ranges()["rhythm"]["positions"]}
    for stop in PACES.values():
        assert by_value[stop.rhythm] == f"{pace.speech_kept(stop.rhythm):g} s"


def test_the_pause_slider_shows_seconds_and_says_them_in_words():
    positions = {p["value"]: p for p in fine_ranges()["gap_length"]["positions"]}
    assert positions[0.35] == {"value": 0.35, "shown": "0.35 s", "said": "0.35 seconds"}
    assert positions[1.0]["shown"] == "1 s" and positions[1.0]["said"] == "1 second"


def test_each_slider_says_which_end_cuts_harder():
    ranges = fine_ranges()
    assert ranges["gap_length"]["harder"] == "min" and ranges["rhythm"]["harder"] == "max"


@pytest.mark.parametrize("gap, rhythm, message", [
    (0.37, 3, "That pause length is not on the slider. It goes from 0.15 to 1.5 seconds in steps of 0.05."),
    (0.1, 3, "pause length is not on the slider"),
    (1.55, 3, "pause length is not on the slider"),
    (0.4, 3.2, "That rhythm is not on the slider. It goes from 1 to 5 in steps of 0.5."),
    (0.4, 0.5, "rhythm is not on the slider"),
    (0.4, 5.5, "rhythm is not on the slider"),
])
def test_a_value_off_the_step_or_outside_the_range_is_refused_in_plain_words(gap, rhythm, message):
    with pytest.raises(StudioError, match=message):
        check_fine(gap, rhythm)


def test_a_value_a_hair_off_its_step_is_put_on_it():
    assert check_fine(0.35000000000000003, 3.5000000001) == (0.35, 3.5)
    assert check_fine(0.15 + 0.05 * 3, 1) == (0.3, 1.0)


# ── the level behind her own setting ──────────────────────────────────────────


def test_between_two_stops_the_pauses_left_lie_on_a_line_between_theirs():
    level = custom_level(0.35, 3.5)  # halfway from tight (0.3) to fast (0.4)
    assert (level.sentence_pause, level.clause_pause, level.inner_pause) == (0.215, 0.125, 0.08)
    assert (level.gap_length, level.rhythm, level.name) == (0.35, 3.5, CUSTOM)
    quarter = custom_level(0.7, 2)   # a quarter of the way from standard (0.6) to natural (1.0)
    assert quarter.sentence_pause == pytest.approx(0.45) and quarter.inner_pause == pytest.approx(0.21)


@pytest.mark.parametrize("name", SIX_STOPS)
def test_on_a_stops_pause_length_the_pauses_left_are_that_stops(name):
    stop = PACES[name]
    level = custom_level(stop.gap_length, 1)
    assert (level.sentence_pause, level.clause_pause, level.inner_pause) == \
        (stop.sentence_pause, stop.clause_pause, stop.inner_pause)


def test_past_either_end_the_pauses_left_are_that_ends():
    gentlest, hardest = custom_level(1.5, 1), custom_level(0.15, 5)
    assert gentlest.sentence_pause == PACES["natural"].sentence_pause
    assert hardest.inner_pause == PACES["max"].inner_pause


def test_her_own_setting_always_leaves_a_pause_shorter_than_the_pauses_it_cuts():
    for position in fine_ranges()["gap_length"]["positions"]:
        level = custom_level(position["value"], 5)
        assert 0 < level.inner_pause <= level.clause_pause <= level.sentence_pause < level.gap_length


def test_the_summary_names_the_stops_around_her_setting_and_holds_no_number():
    assert custom_level(0.35, 3).summary == "Your own setting, between Fast and Tight."
    assert custom_level(0.6, 5).summary == "Your own setting, close to Standard."
    assert custom_level(1.5, 1).summary == "Your own setting, close to Natural."
    assert custom_level(0.15, 5).summary == "Your own setting, close to Max."
    for position in fine_ranges()["gap_length"]["positions"]:
        assert not any(ch.isdigit() for ch in custom_level(position["value"], 3).summary)


def test_custom_is_no_stop_and_one_place_knows_it():
    assert CUSTOM not in PACES
    with pytest.raises(StudioError, match="can't be picked by name. Leave pace out to keep it"):
        get_pace("custom")
    assert level_for("custom", (0.35, 3.5)) == custom_level(0.35, 3.5)
    assert level_for("fast") is PACES["fast"] and level_for(None) is PACES["standard"]
    with pytest.raises(StudioError, match="two values are missing"):
        level_for("custom")
    with pytest.raises(StudioError, match="is not a level"):
        level_for("brutal")


def test_the_filler_list_is_the_nearest_by_pause_length_the_way_clipforge_picks_it(words, monkeypatch):
    from lumr_studio.engine.microcut_pacing import _level_for_gap_length

    assert [_level_for_gap_length(g) for g in (0.2, 0.35, 0.6, 1.2)] == [5, 4, 3, 2]
    # The plugin hands ClipForge her pause length and rhythm as they are, and ClipForge picks the list.
    asked = []

    def planner(_words, **kwargs):
        asked.append((kwargs["gap_length"], kwargs["rhythm"], kwargs["stop"]))
        return []

    monkeypatch.setattr(treatment, "plan_auto_cuts", planner)
    treatment.make_edit([], words, DURATION, silences=[], sounds=[], keeps=[],
                        treatment=Treatment(pace=CUSTOM, fine=(0.35, 3.5)))
    treatment.make_edit([], words, DURATION, silences=[], sounds=[], keeps=[], treatment=Treatment(pace="fast"))
    assert asked == [(0.35, 3.5, False), (0.4, 4, True)]


# ── moving a slider ───────────────────────────────────────────────────────────


def test_the_sliders_stand_at_the_stops_values_until_she_moves_one(video):
    state = treatment.page_state(edited(video, pace="fast"))
    assert state["settings"]["fine"] == {"gap_length": 0.4, "rhythm": 4.0}
    assert state["custom"] is None and state["settings"]["pace"] == "fast"


def test_moving_a_slider_makes_the_pace_custom(video):
    ctx = edited(video, pace="standard")
    state = treatment.change_treatment(ctx, {"fine": {"gap_length": 0.35}})
    assert state["settings"]["pace"] == "custom"
    assert state["settings"]["fine"] == {"gap_length": 0.35, "rhythm": 3.0}, "the other slider stays where it stood"
    assert state["settings"]["gap_length"] is None
    custom = state["custom"]
    assert set(custom) == {"label", "summary", "trims", "seconds_saved"} and custom["label"] == "Custom"
    assert custom["trims"] == len(state["trims"]) > 0
    assert custom["seconds_saved"] == pytest.approx(sum(b - a for a, b, _ in state["trims"]), abs=0.06)
    assert [p["pace"] for p in state["paces"]] == SIX_STOPS, "custom is no seventh entry"
    assert all(list(t["by_pace"]) == SIX_STOPS for t in state["take_out"])
    edit = saved(video)
    assert edit["pace"] == "custom" and edit["treatment"]["fine"] == {"gap_length": 0.35, "rhythm": 3.0}


def test_both_sliders_move_in_one_go_and_one_after_the_other(video):
    ctx = edited(video)
    together = treatment.change_treatment(ctx, {"fine": {"gap_length": 0.2, "rhythm": 4.5}})
    assert together["settings"]["fine"] == {"gap_length": 0.2, "rhythm": 4.5}
    after = treatment.change_treatment(ctx, {"fine": {"rhythm": 2}})
    assert after["settings"]["fine"] == {"gap_length": 0.2, "rhythm": 2.0}


def test_her_setting_reads_custom_even_on_a_stops_own_values(video):
    ctx = edited(video, pace="fast")
    state = treatment.change_treatment(ctx, {"fine": {"gap_length": 0.4, "rhythm": 4}})
    assert state["settings"]["pace"] == "custom" and state["custom"]["summary"] == "Your own setting, close to Fast."
    fast = treatment.change_treatment(ctx, {"pace": "fast"})
    assert state["removed"] == fast["removed"], "the same values cut the same"


def test_a_harder_setting_cuts_more(video):
    ctx = edited(video, cuts=[])
    gentle = treatment.change_treatment(ctx, {"fine": {"gap_length": 1.5, "rhythm": 1}})
    hard = treatment.change_treatment(ctx, {"fine": {"gap_length": 0.15, "rhythm": 5}})
    assert hard["durations"]["saved"] > gentle["durations"]["saved"]
    assert hard["changed"]["seconds"] < 0 and hard["changed"]["trims"] > 0


def test_pressing_a_stop_leaves_custom_and_sets_the_sliders_back(video):
    ctx = edited(video)
    treatment.change_treatment(ctx, {"fine": {"gap_length": 0.35, "rhythm": 3.5}})
    state = treatment.change_treatment(ctx, {"pace": "tight"})
    assert state["settings"]["pace"] == "tight" and state["custom"] is None
    assert state["settings"]["fine"] == {"gap_length": 0.3, "rhythm": 4.5}
    assert "fine" in saved(video)["treatment"] and saved(video)["pace"] == "tight"


def test_a_switch_change_leaves_her_setting_as_it_is(video):
    ctx = edited(video)
    treatment.change_treatment(ctx, {"fine": {"gap_length": 0.35}})
    state = treatment.change_treatment(ctx, {"take_out": {"pauses": False}})
    assert state["settings"]["pace"] == "custom" and state["settings"]["fine"]["gap_length"] == 0.35
    assert state["custom"]["trims"] == len(state["trims"])


def test_her_cuts_and_keeps_outlive_a_slider_move(video):
    ctx = edited(video)
    treatment.add_cut(ctx, {"start": 16.05, "end": 16.2})
    treatment.add_keep(ctx, {"start": 12.1, "end": 12.4})
    state = treatment.change_treatment(ctx, {"fine": {"gap_length": 0.15, "rhythm": 5}})
    assert [r["id"] for r in state["rows"] if r["by"] == "you"] == ["y16.05-16.20"]
    assert [k["id"] for k in state["keeps"]] == ["k12.00-14.20"]
    assert not any(a < 14.2 and b > 12.0 for a, b in state["removed"]), "nothing cuts inside the part she kept"


def test_a_slider_move_is_no_step_of_undo(video):
    ctx = edited(video)
    treatment.add_cut(ctx, {"start": 16.05, "end": 16.2})
    moved = treatment.change_treatment(ctx, {"fine": {"gap_length": 0.35}})
    assert moved["can_undo"] is True
    undone = treatment.undo(ctx, {})
    assert undone["cut_counts"]["yours"] == 0, "undo takes back the cut"
    assert undone["settings"]["pace"] == "custom" and undone["settings"]["fine"]["gap_length"] == 0.35


def test_claudes_pause_override_shows_where_the_slider_stands(video):
    ctx = edited(video, pace="natural", gap_length=0.4)
    state = treatment.page_state(ctx)
    assert state["settings"]["pace"] == "natural" and state["settings"]["gap_length"] == 0.4
    assert state["settings"]["fine"] == {"gap_length": 0.4, "rhythm": 2.0} and state["custom"] is None


@pytest.mark.parametrize("body, message", [
    ({"fine": {"gap_length": 0.37}}, "pause length is not on the slider"),
    ({"fine": {"rhythm": 6}}, "rhythm is not on the slider"),
    ({"fine": {}}, "fine must hold gap_length, rhythm, or both"),
    ({"fine": 0.35}, "fine must hold gap_length, rhythm, or both"),
    ({"fine": {"gap": 0.35}}, "'gap', which is not a slider"),
    ({"fine": {"gap_length": "0.35"}}, "fine.gap_length must be a number"),
    ({"fine": {"gap_length": True}}, "fine.gap_length must be a number"),
    ({"pace": "fast", "fine": {"gap_length": 0.35}}, "Send a pace or fine, not both"),
    ({"pace": "custom"}, "can't be picked by name"),
])
def test_a_bad_slider_move_says_how_to_fix_it_and_changes_nothing(video, body, message):
    ctx = edited(video)
    before = saved(video)
    with pytest.raises(StudioError, match=message):
        treatment.change_treatment(ctx, body)
    assert saved(video) == before


# ── her usual, and Claude ─────────────────────────────────────────────────────


def test_make_this_my_usual_saves_her_own_setting_and_the_next_video_starts_from_it(video, tmp_path, synthetic_master):
    ctx = edited(video)
    treatment.change_treatment(ctx, {"fine": {"gap_length": 0.35, "rhythm": 3.5}})
    state = treatment.save_as_usual(ctx, {})
    usual = json.loads((projects_root() / treatment.USUAL_FILE).read_text())
    assert usual["pace"] == "custom" and usual["fine"] == {"gap_length": 0.35, "rhythm": 3.5}
    assert state["usual"]["pace"] == "custom" and state["usual"]["fine"] == state["settings"]["fine"]
    assert state["usual"]["take_out"] == state["settings"]["take_out"]
    assert treatment.starting_treatment() == Treatment(pace=CUSTOM, fine=(0.35, 3.5))

    other = tmp_path / "second.mp4"
    other.write_bytes(synthetic_master.read_bytes())
    other.with_suffix(".words.json").write_text(video.with_suffix(".words.json").read_text())
    result = tools.set_edit(str(other), [UM], auto_tighten=True, **QUIET)
    assert result["auto"]["pace"] == "custom" and result["auto"]["fine"] == {"gap_length": 0.35, "rhythm": 3.5}
    assert treatment.page_state(ctx_for(other))["settings"]["pace"] == "custom"


def test_a_usual_saved_before_the_sliders_still_loads(video):
    (projects_root() / treatment.USUAL_FILE).parent.mkdir(parents=True, exist_ok=True)
    (projects_root() / treatment.USUAL_FILE).write_text(json.dumps(
        {"version": 1, "pace": "fast", "take_out": {"pauses": True, "fillers": False, "repeats": True}}))
    usual = treatment.load_usual()
    assert usual.pace == "fast" and usual.fine_values() == {"gap_length": 0.4, "rhythm": 4.0}
    assert usual.take_out == frozenset({"pauses", "repeats", "likes"})


def test_a_usual_that_lost_its_values_is_ignored(video):
    (projects_root() / treatment.USUAL_FILE).parent.mkdir(parents=True, exist_ok=True)
    (projects_root() / treatment.USUAL_FILE).write_text(json.dumps({"pace": "custom", "take_out": {}}))
    assert treatment.load_usual() is None


def test_get_edit_shows_claude_her_values_and_says_they_are_hers(video):
    ctx = edited(video, pace="standard")
    treatment.change_treatment(ctx, {"fine": {"gap_length": 0.35, "rhythm": 3.5}})
    summary = tools.get_edit(str(video))
    assert summary["pace"] == "custom"
    creator = summary["creator"]
    assert creator["pace"] == {"claude": "standard", "creator": "custom"}
    assert creator["fine"]["gap_length"] == 0.35 and creator["fine"]["rhythm"] == 3.5
    assert creator["fine"]["speech_kept_between_cuts"] == 0.85
    assert "They are hers" in creator["fine"]["note"] and "Leave pace and gap_length out" in creator["fine"]["note"]


def test_a_new_edit_from_claude_without_a_pace_keeps_her_setting(video):
    ctx = edited(video)
    before = treatment.change_treatment(ctx, {"fine": {"gap_length": 0.35, "rhythm": 3.5}})
    result = tools.set_edit(str(video), [UM], auto_tighten=True, **QUIET)
    assert result["auto"]["pace"] == "custom" and result["auto"]["trims_pauses_longer_than"] == 0.35
    after = treatment.page_state(ctx_for(video))
    assert after["settings"] == before["settings"] and after["removed"] == before["removed"]
    assert tools.get_edit(str(video))["creator"]["fine"]["gap_length"] == 0.35, "hers stays listed as hers"


def test_claude_replaces_her_setting_only_by_naming_a_pace(video):
    ctx = edited(video)
    treatment.change_treatment(ctx, {"fine": {"gap_length": 0.35}})
    with pytest.raises(StudioError, match="gap_length would change it"):
        tools.set_edit(str(video), [UM], auto_tighten=True, gap_length=0.5, **QUIET)
    with pytest.raises(StudioError, match="can't be picked by name"):
        tools.set_edit(str(video), [UM], auto_tighten=True, pace="custom", **QUIET)
    assert saved(video)["pace"] == "custom"
    result = tools.set_edit(str(video), [UM], auto_tighten=True, pace="fast", **QUIET)
    assert result["auto"]["pace"] == "fast" and saved(video)["treatment"]["fine"] == {"gap_length": 0.4, "rhythm": 4.0}


def test_her_setting_survives_an_edit_with_no_automatic_trims(video):
    ctx = edited(video)
    treatment.change_treatment(ctx, {"fine": {"gap_length": 0.35}})
    tools.set_edit(str(video), [UM], **QUIET)
    assert saved(video)["treatment"]["pace"] == "custom" and saved(video)["auto_tighten"] is False
    tools.set_edit(str(video), [UM], auto_tighten=True, **QUIET)
    assert saved(video)["pace"] == "custom" and saved(video)["treatment"]["fine"]["gap_length"] == 0.35


# ── the plans kept with the project ───────────────────────────────────────────


def plans_of(video):
    path = open_project(str(video)).root / autocuts.PLANS_FILE
    return json.loads(path.read_text())["plans"]


def test_the_stops_plans_are_always_kept_and_of_her_own_only_the_latest(video, monkeypatch):
    monkeypatch.setattr(autocuts, "CUSTOM_PLANS_KEPT", 3)
    ctx = edited(video)
    treatment.page_state(ctx)
    stops = {f"{p.gap_length}/{float(p.rhythm)}" for p in PACES.values()}
    assert set(plans_of(video)) == stops and all(plan["stop"] for plan in plans_of(video).values())
    tried = [0.25, 0.35, 0.45, 0.55, 0.65]
    for gap in tried:
        treatment.change_treatment(ctx, {"fine": {"gap_length": gap, "rhythm": 2.5}})
    plans = plans_of(video)
    assert stops <= set(plans)
    assert [name for name, plan in plans.items() if not plan["stop"]] == ["0.45/2.5", "0.55/2.5", "0.65/2.5"]


def test_a_setting_on_a_stops_values_does_not_take_a_place_of_its_own(video):
    ctx = edited(video)
    treatment.page_state(ctx)
    before = set(plans_of(video))
    treatment.change_treatment(ctx, {"fine": {"gap_length": 0.4, "rhythm": 4}})
    assert set(plans_of(video)) == before and all(plan["stop"] for plan in plans_of(video).values())


def test_plans_saved_in_the_earlier_shape_are_made_again(video):
    ctx = edited(video)
    path = open_project(str(video)).root / autocuts.PLANS_FILE
    path.write_text(json.dumps({"version": 1, "take": "x", "plans": {"0.6/3.0": [{"start": 1, "end": 2}]}}))
    treatment.forget_recipes()
    treatment.page_state(ctx)
    assert json.loads(path.read_text())["version"] == autocuts.PLANS_VERSION and len(plans_of(video)) == 6
