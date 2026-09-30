"""The treatment: one recipe for every edit, the page's state, and each action's effect on the saved edit."""

import json

import pytest
from lumr_studio.engine.audio_boundaries import Silence

import positivity
import seams
import short_word
from conftest import make_words, spoken_run
from lumr_studio import autocuts, tools, treatment, word_times
from lumr_studio.autocuts import trim_kind
from lumr_studio.errors import StudioError
from lumr_studio.overlays import overlays_path
from lumr_studio.pace import PACES
from lumr_studio.project import open_project, projects_root, write_json_atomic
from lumr_studio.render_jobs import nothing_exported
from lumr_studio.silences import no_silences
from lumr_studio.tools import unmeasured_labels
from lumr_studio.treatment import Treatment, make_edit

QUIET = {"silences": no_silences, "labels": unmeasured_labels}
DURATION = 20.0

# Claude's two cuts on the conftest talk: the "um" and the second take of the opening line.
UM = {"start": 3.3, "end": 3.8, "reason": "filler before the line", "kind": "other"}
RETAKE = {"start": 7.9, "end": 10.75, "reason": "second take of the opening line", "kind": "repeat"}

STATE_KEYS = {
    "video", "word_times", "word_times_note", "settings", "usual", "taste", "version", "feedback", "share", "paces", "custom", "fine_ranges", "take_out",
    "need_a_look",
    "cut_counts", "groups", "rows", "removed", "trims", "keeps", "samples", "clusters", "busy", "overlays",
    "export", "durations", "changed", "can_undo", "words", "sounds", "updated_at",
}
SIX_STOPS = ["natural", "standard", "fast", "tight", "hard", "max"]
CLUSTER_KEYS = {"number", "start", "end", "clock", "clock_range", "edits", "seconds_removed", "out", "level"}
OVERLAY_KEYS = {"id", "name", "kind", "start", "end", "clock", "status", "tag"}
CHANGED_KEYS = {
    "trims", "trims_added", "trims_dropped", "seconds", "need_a_look", "new_flags", "overlays_hidden",
    "your_cuts", "words_cut", "words_back", "keeps_changed", "likes",
}
ROW_KEYS = {
    "id", "start", "end", "clock", "seconds", "group", "reason", "by", "flags", "flag_labels", "note", "why",
    "state", "before", "removed", "after", "rating",
}


@pytest.fixture(autouse=True)
def fresh_recipes():
    """Each test starts with empty caches, so a stand-in planner is never hidden by a cached recipe."""
    treatment.forget_recipes()
    yield
    treatment.forget_recipes()


def ctx_for(video):
    return treatment.load_context(open_project(str(video)), duration=DURATION, silences=no_silences)


def edited(video, cuts=(UM, RETAKE), **kwargs):
    kwargs.setdefault("auto_tighten", True)
    tools.set_edit(str(video), list(cuts), **kwargs, **QUIET)
    return ctx_for(video)


def saved(video):
    project = open_project(str(video))
    return json.loads(project.edit_path.read_text())


def row(state, kind):
    return next(r for r in state["rows"] if r["group"] == kind)


def removed_at(state, t):
    return any(a <= t < b for a, b in state["removed"])


# ── kinds of trim ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("reason, kind", [
    ("pause: 1.2s", "pauses"), ("filler: um", "fillers"), ("stutter: the the", "repeats"),
    ("repeat: the", "repeats"), ("auto: filler: um (+1 more)", "fillers"), ("mystery", "pauses"),
])
def test_trim_kind_reads_clipforges_label(reason, kind):
    assert trim_kind(reason) == kind


def three_kinds(*_args, **_kwargs):
    """A stand-in plan with one well-spaced trim of each kind on the conftest talk."""
    return [
        {"start": 2.45, "end": 3.35, "reason": "pause: 1.0s", "kind": "pauses"},
        {"start": 10.75, "end": 11.95, "reason": "filler: uh", "kind": "fillers"},
        {"start": 14.25, "end": 15.45, "reason": "stutter: the the", "kind": "repeats"},
    ]


@pytest.mark.parametrize("off", ["pauses", "fillers", "repeats"])
def test_turning_a_kind_off_drops_exactly_its_trims(words, monkeypatch, off):
    monkeypatch.setattr(treatment, "plan_auto_cuts", three_kinds)
    everything = make_edit([], words, DURATION, treatment=Treatment(pace="fast"), silences=[], sounds=[], keeps=[])
    without = make_edit([], words, DURATION, silences=[], sounds=[], keeps=[],
                        treatment=Treatment(pace="fast", take_out=frozenset({"pauses", "fillers", "repeats"} - {off})))
    kinds = [c["kind"] for c in everything.outcome.cuts]
    assert sorted(kinds) == ["fillers", "pauses", "repeats"]
    assert without.outcome.cuts == [c for c in everything.outcome.cuts if c["kind"] != off]


def test_the_real_planner_files_an_um_under_fillers(words):
    talk = spoken_run("Hello everyone and welcome.", 0.2) + spoken_run("Today we will talk about", 3.0)
    talk += [{"word": "um", "start": 5.2, "end": 5.5}] + spoken_run("editing video for the web.", 5.6)
    talk += spoken_run("Thanks for watching.", 9.0)
    on = make_edit([], talk, 12.0, treatment=Treatment(pace="fast"), silences=[], sounds=[], keeps=[])
    off = make_edit([], talk, 12.0, treatment=Treatment(pace="fast", take_out=frozenset({"pauses", "repeats"})),
                    silences=[], sounds=[], keeps=[])
    assert [c["kind"] for c in on.outcome.cuts].count("fillers") == 1
    assert off.outcome.cuts == [c for c in on.outcome.cuts if c["kind"] != "fillers"]


def test_no_kinds_on_means_no_trims(words):
    made = make_edit([RETAKE], words, DURATION, treatment=Treatment(take_out=frozenset()),
                     silences=[], sounds=[], keeps=[])
    assert [c["source"] for c in made.outcome.cuts] == ["claude"]


def test_the_recipe_is_cached_and_callers_cannot_spoil_it(words):
    first = make_edit([UM], words, DURATION, treatment=Treatment(), silences=[], sounds=[], keeps=[])
    first.outcome.cuts.clear()
    again = make_edit([UM], words, DURATION, treatment=Treatment(), silences=[], sounds=[], keeps=[])
    assert again.outcome.cuts


# ── set_edit through the recipe ───────────────────────────────────────────────


def test_set_edit_keeps_each_cuts_kind_and_calls_a_missing_one_other(video):
    tools.set_edit(str(video), [RETAKE, {k: v for k, v in UM.items() if k != "kind"}], **QUIET)
    edit = saved(video)
    assert sorted(c["kind"] for c in edit["cuts"]) == ["other", "repeat"]
    assert {c.get("kind") for c in tools.get_edit(str(video))["cuts"]} == {"other", "repeat"}


def test_a_kind_not_on_the_list_is_rejected_with_the_list(video):
    result = tools.set_edit(str(video), [UM, {**RETAKE, "kind": "joke"}], **QUIET)
    assert "not one of: repeat, false_start, off_topic, other" in result["rejected"][0]["why"]


def test_set_edit_saves_the_treatment_it_used_and_what_claude_asked(video):
    tools.set_edit(str(video), [UM], auto_tighten=True, pace="fast", **QUIET)
    edit = saved(video)
    assert edit["treatment"]["pace"] == "fast" and all(edit["treatment"]["take_out"].values())
    assert edit["set_by_claude"] == edit["treatment"]


def test_set_edit_without_auto_tighten_turns_every_kind_of_trim_off(video):
    tools.set_edit(str(video), [UM], **QUIET)
    # The switch over Claude's picked filler words is the creator's, and stays as it was.
    assert saved(video)["treatment"]["take_out"] == {"pauses": False, "fillers": False, "repeats": False, "likes": True}
    assert saved(video)["auto_tighten"] is False and saved(video)["pace"] is None


def test_set_edit_without_a_pace_keeps_what_the_creator_set_on_the_page(video):
    ctx = edited(video, pace="standard")
    treatment.change_treatment(ctx, {"pace": "fast", "take_out": {"fillers": False}})
    result = tools.set_edit(str(video), [UM, RETAKE], auto_tighten=True, **QUIET)
    assert result["auto"]["pace"] == "fast" and result["auto"]["take_out"] == ["pauses", "repeats"]


def test_a_new_video_starts_from_the_usual(video):
    write_json_atomic(projects_root() / treatment.USUAL_FILE,
                      {"pace": "natural", "take_out": {"pauses": True, "fillers": False, "repeats": True}})
    result = tools.set_edit(str(video), [UM], auto_tighten=True, **QUIET)
    assert result["auto"]["pace"] == "natural" and result["auto"]["take_out"] == ["pauses", "repeats"]


def test_with_no_usual_a_new_video_starts_at_standard_with_everything_on(video):
    result = tools.set_edit(str(video), [UM], auto_tighten=True, **QUIET)
    assert result["auto"]["pace"] == "standard"
    assert result["auto"]["take_out"] == ["pauses", "fillers", "repeats"]


def test_an_edit_saved_before_treatments_reads_as_its_own_pace():
    legacy = {"cuts": [], "auto_tighten": True, "pace": "fast", "gap_length": None}
    assert treatment.saved_treatment(legacy) == Treatment(pace="fast")
    plain = treatment.saved_treatment({"cuts": [], "auto_tighten": False, "pace": None})
    assert plain.trims == frozenset() and plain.pace == "standard"


# ── the page's state ──────────────────────────────────────────────────────────


def test_state_has_every_field_the_page_reads(video):
    state = treatment.page_state(edited(video))
    assert set(state) == STATE_KEYS
    assert [g["key"] for g in state["groups"]] == ["yours", "repeat", "false_start", "off_topic", "other"]
    assert [t["key"] for t in state["take_out"]] == ["pauses", "fillers", "repeats", "likes"]
    assert [t["label"] for t in state["take_out"]] == ["Long pauses", "Filler words", "Stutters", "Filler likes"]
    assert state["custom"] is None and state["settings"]["fine"] == {"gap_length": 0.6, "rhythm": 3.0}
    assert set(state["fine_ranges"]) == {"gap_length", "rhythm"}
    assert [p["pace"] for p in state["paces"]] == SIX_STOPS
    assert [p["label"] for p in state["paces"]] == ["Natural", "Standard", "Fast", "Tight", "Hard", "Max"]
    assert state["word_times"] == "estimated" and state["word_times_note"]
    assert state["cut_counts"] == {"yours": 0, "claude": 2} and state["can_undo"] is False
    assert all(r["by"] == "claude" for r in state["rows"])
    assert [s["key"] for s in state["samples"]] == ["rhythm", "big_cut", "joke"]
    assert all(set(r) == ROW_KEYS for r in state["rows"]) and len(state["rows"]) == 2
    assert all(r["state"] == "kept_out" for r in state["rows"])
    assert state["changed"] is None and state["settings"]["pace"] == "standard"
    assert state["durations"]["full"] == DURATION
    assert state["durations"]["saved"] == pytest.approx(DURATION - state["durations"]["edited"], abs=0.11)
    assert all(len(w) == 5 for w in state["words"])


def test_the_one_needs_a_look_count_is_the_sum_of_the_groups(video):
    def flag_claudes(cuts, words, duration, *, labels=None):
        return [{"start": c["start"], "end": c["end"], "source": c["source"],
                 "flags": ["mid_sentence_in"] if c["source"] == "claude" else [], "note": "", "why": "Why."}
                for c in cuts]

    tools.set_edit(str(video), [UM, RETAKE], auto_tighten=True, **QUIET)
    ctx = treatment.load_context(open_project(str(video)), duration=DURATION, silences=no_silences,
                                 join_rows=flag_claudes)
    state = treatment.page_state(ctx)
    assert [g["need_a_look"] for g in state["groups"]] == [0, 1, 0, 0, 1]
    assert state["need_a_look"] == 2
    rid = row(state, "repeat")["id"]
    assert treatment.set_cut_state(ctx, {"id": rid, "state": "put_back"})["need_a_look"] == 1


def test_with_nothing_flagged_the_count_is_zero(video):
    ctx = treatment.load_context(open_project(str(video)), duration=DURATION, silences=no_silences,
                                 join_rows=lambda *a, **k: [])
    tools.set_edit(str(video), [UM], **QUIET)
    assert treatment.page_state(ctx)["need_a_look"] == 0


def test_row_ids_hold_still_when_the_pace_changes(video):
    ctx = edited(video, pace="natural")
    before = [r["id"] for r in treatment.page_state(ctx)["rows"]]
    after = [r["id"] for r in treatment.change_treatment(ctx, {"pace": "fast"})["rows"]]
    assert before == after


def test_counts_by_kind_show_even_when_the_switch_is_off(video):
    ctx = edited(video, cuts=[], pace="standard")
    on = {t["key"]: t["count"] for t in treatment.page_state(ctx)["take_out"]}
    off = treatment.change_treatment(ctx, {"take_out": {"pauses": False}})
    assert {t["key"]: t["count"] for t in off["take_out"]} == on
    assert off["trims"] == [] and on["pauses"] > 0
    assert next(p for p in off["paces"] if p["pace"] == "standard")["trims"] == 0


def test_each_switch_says_what_every_pace_would_take_out(video):
    # A switch showing 0 at this pace can still do something at another; the page says so.
    state = treatment.page_state(edited(video, cuts=[], pace="standard"))
    for kind in state["take_out"]:
        assert list(kind["by_pace"]) == SIX_STOPS
        assert kind["by_pace"]["standard"] == kind["count"]
    pauses = next(t for t in state["take_out"] if t["key"] == "pauses")["by_pace"]
    assert [pauses[name] for name in SIX_STOPS] == sorted(pauses.values())


# ── changing the pace and switches ────────────────────────────────────────────


def test_a_new_pace_rebuilds_and_saves_the_edit_and_says_what_changed(video):
    ctx = edited(video, pace="natural")
    state = treatment.change_treatment(ctx, {"pace": "fast"})
    edit = saved(video)
    assert edit["treatment"]["pace"] == "fast" and edit["pace"] == "fast"
    assert state["settings"]["pace"] == "fast"
    changed = state["changed"]
    assert changed["trims"] == len(state["trims"]) - 0 and changed["trims_added"] == len(state["trims"])
    assert changed["seconds"] < 0 and set(changed) == CHANGED_KEYS
    assert changed["overlays_hidden"] == 0


def test_a_new_pace_clears_claudes_pause_override(video):
    tools.set_edit(str(video), [UM], auto_tighten=True, pace="standard", gap_length=0.4, **QUIET)
    ctx = ctx_for(video)
    assert treatment.page_state(ctx)["settings"]["gap_length"] == 0.4
    assert treatment.change_treatment(ctx, {"pace": "fast"})["settings"]["gap_length"] is None


@pytest.mark.parametrize("body, message", [
    ({}, "Send a pace"),
    ({"pace": "brutal"}, "is not a level"),
    ({"pace": 3}, "pace must be one of"),
    ({"speed": "fast"}, "Unknown field 'speed'"),
    ({"take_out": {"ums": False}}, "not a kind"),
    ({"take_out": {"fillers": "no"}}, "true or false"),
    ({"take_out": ["fillers"]}, "take_out must be an object"),
    ([1, 2], "Send a JSON object"),
])
def test_a_bad_treatment_change_says_how_to_fix_it(video, body, message):
    ctx = edited(video)
    with pytest.raises(StudioError, match=message):
        treatment.change_treatment(ctx, body)


# ── Claude's cuts: put back and kept out ──────────────────────────────────────


def test_putting_a_cut_back_saves_a_kept_span_and_plays_its_words(video):
    ctx = edited(video)
    rid = row(treatment.page_state(ctx), "repeat")["id"]
    state = treatment.set_cut_state(ctx, {"id": rid, "state": "put_back"})
    assert row(state, "repeat")["state"] == "put_back" and row(state, "repeat")["flags"] == []
    assert not removed_at(state, 9.0)
    keep = saved(video)["keep"]
    assert len(keep) == 1 and keep[0]["origin"] == "put_back" and keep[0]["cut"] == rid
    assert "second take" in keep[0]["note"]
    assert state["keeps"] == []  # a cut put back is a row, not a kept part
    assert state["changed"]["seconds"] > 0


def test_a_cut_put_back_outlives_a_new_pace_and_a_new_set_edit(video):
    ctx = edited(video)
    rid = row(treatment.page_state(ctx), "repeat")["id"]
    treatment.set_cut_state(ctx, {"id": rid, "state": "put_back"})
    state = treatment.change_treatment(ctx, {"pace": "fast"})
    assert row(state, "repeat")["state"] == "put_back" and not removed_at(state, 9.0)
    again = tools.set_edit(str(video), [UM, RETAKE], auto_tighten=True, **QUIET)
    assert "the creator restored" in again["rejected"][0]["why"]


def test_keeping_a_cut_out_again_removes_its_kept_span(video):
    ctx = edited(video)
    rid = row(treatment.page_state(ctx), "repeat")["id"]
    treatment.set_cut_state(ctx, {"id": rid, "state": "put_back"})
    state = treatment.set_cut_state(ctx, {"id": rid, "state": "kept_out"})
    assert row(state, "repeat")["state"] == "kept_out" and removed_at(state, 9.0)
    assert saved(video)["keep"] == []


def test_putting_back_a_cut_claude_forced_through_a_keep_clears_the_force(video):
    ctx = edited(video)
    rid = row(treatment.page_state(ctx), "repeat")["id"]
    treatment.set_cut_state(ctx, {"id": rid, "state": "put_back"})
    tools.set_edit(str(video), [UM, RETAKE], auto_tighten=True, override_keeps=True, **QUIET)
    ctx = ctx_for(video)
    assert row(treatment.page_state(ctx), "repeat")["state"] == "kept_out"
    state = treatment.set_cut_state(ctx, {"id": rid, "state": "put_back"})
    assert row(state, "repeat")["state"] == "put_back"


@pytest.mark.parametrize("body, message", [
    ({"id": "c1.00-2.00", "state": "put_back"}, "is not one of Claude's cuts"),
    ({"id": "k1.00-2.00", "state": "put_back"}, "starting with 'c'"),
    ({"id": 7, "state": "put_back"}, "starting with 'c'"),
    ({"id": "c1.00-2.00", "state": "restored"}, "state must be"),
    ({"id": "c1.00-2.00", "state": "put_back", "why": "x"}, "Unknown field 'why'"),
])
def test_a_bad_cut_change_says_how_to_fix_it(video, body, message):
    ctx = edited(video)
    with pytest.raises(StudioError, match=message):
        treatment.set_cut_state(ctx, body)


# ── keep no matter what ───────────────────────────────────────────────────────


def test_a_kept_part_widens_to_whole_sentences_and_puts_back_what_it_covers(video):
    ctx = edited(video, pace="standard")
    state = treatment.add_keep(ctx, {"start": 9.0, "end": 9.2, "note": "the good take"})
    (kept,) = state["keeps"]
    assert (kept["start"], kept["end"]) == (8.0, 10.7)
    assert kept["text"] == "today we talk about editing video." and kept["note"] == "the good take"
    assert kept["id"] == "k8.00-10.70" and kept["clock"] == "0:08-0:10"
    assert row(state, "repeat")["state"] == "put_back"
    assert not any(8.0 < t[0] < 10.7 or 8.0 < t[1] < 10.7 for t in state["trims"])
    assert saved(video)["keep"][0]["origin"] == "flagged"


def test_a_kept_part_drops_the_trims_inside_it(video):
    ctx = edited(video, cuts=[UM], pace="standard")
    trims = treatment.page_state(ctx)["trims"]
    inside = [t for t in trims if 10.0 < t[0] < 12.5]
    assert inside
    state = treatment.add_keep(ctx, {"start": 10.0, "end": 12.5})
    assert not [t for t in state["trims"] if 10.0 < t[0] < 12.5]
    assert state["changed"]["trims_dropped"] == len(inside)


def test_a_kept_part_survives_a_new_pace(video):
    ctx = edited(video, pace="standard")
    treatment.add_keep(ctx, {"start": 9.0, "end": 9.2})
    for pace in ("fast", "natural", "standard"):
        state = treatment.change_treatment(ctx, {"pace": pace})
        assert len(state["keeps"]) == 1 and row(state, "repeat")["state"] == "put_back"


def test_overlapping_kept_parts_merge(video):
    ctx = edited(video)
    treatment.add_keep(ctx, {"start": 0.6, "end": 0.7, "note": "hello"})
    state = treatment.add_keep(ctx, {"start": 2.0, "end": 3.5})
    (kept,) = state["keeps"]
    assert kept["start"] == 0.5 and kept["end"] >= 6.6 and "hello" in kept["note"]


def test_a_cut_inside_a_kept_part_cannot_be_kept_out(video):
    ctx = edited(video)
    state = treatment.add_keep(ctx, {"start": 9.0, "end": 9.2})
    with pytest.raises(StudioError, match="Remove that keep first"):
        treatment.set_cut_state(ctx, {"id": row(state, "repeat")["id"], "state": "kept_out"})


def test_removing_a_kept_part_rebuilds_the_edit(video):
    ctx = edited(video)
    state = treatment.add_keep(ctx, {"start": 9.0, "end": 9.2})
    state = treatment.remove_keep(ctx, {"id": state["keeps"][0]["id"]})
    assert state["keeps"] == [] and row(state, "repeat")["state"] == "kept_out" and removed_at(state, 9.0)


def test_a_kept_part_wins_over_claudes_old_override(video):
    ctx = edited(video)
    rid = row(treatment.page_state(ctx), "repeat")["id"]
    treatment.set_cut_state(ctx, {"id": rid, "state": "put_back"})
    tools.set_edit(str(video), [UM, RETAKE], auto_tighten=True, override_keeps=True, **QUIET)
    state = treatment.add_keep(ctx_for(video), {"start": 9.0, "end": 9.2})
    assert row(state, "repeat")["state"] == "put_back"


@pytest.mark.parametrize("body, message", [
    ({"start": 5.0, "end": 4.0}, "must come before"),
    ({"start": -1, "end": 4.0}, "outside the video"),
    ({"start": 1.0, "end": 25.0}, "outside the video"),
    ({"start": "1", "end": 4.0}, "must be a number"),
    ({"start": 1.0, "end": 4.0, "note": "x" * 201}, "at most 200"),
    ({"start": 1.0}, "end must be a number"),
])
def test_a_bad_keep_says_how_to_fix_it(video, body, message):
    ctx = edited(video)
    with pytest.raises(StudioError, match=message):
        treatment.add_keep(ctx, body)


def test_removing_an_unknown_keep_says_to_reload(video):
    with pytest.raises(StudioError, match="Reload the page"):
        treatment.remove_keep(edited(video), {"id": "k1.00-2.00"})


# ── samples hold still ────────────────────────────────────────────────────────


def spans(state):
    return [(s["start"], s["end"]) for s in state["samples"]]


def test_samples_are_saved_once_and_hold_still_when_settings_change(video):
    ctx = edited(video, pace="natural")
    first = spans(treatment.page_state(ctx))
    assert (open_project(str(video)).root / treatment.TREATMENT_FILE).exists()
    assert spans(treatment.change_treatment(ctx, {"pace": "fast"})) == first
    assert spans(treatment.change_treatment(ctx, {"take_out": {"pauses": False}})) == first
    assert spans(treatment.add_keep(ctx, {"start": 9.0, "end": 9.2})) == first


def test_picking_new_samples_moves_them_and_records_the_history(video):
    ctx = edited(video)
    first = spans(treatment.page_state(ctx))
    state = treatment.new_samples(ctx, {})
    store = treatment.load_samples_store(ctx.project)
    assert spans(state) == [(s["start"], s["end"]) for s in store["samples"]]
    # A 20 s video runs out of fresh stretches at once, so the history starts over.
    assert len(store["used"]) >= 3 and all(s["why"] for s in state["samples"])
    assert state["changed"] is None and len(first) == 3


def test_new_samples_take_no_fields(video):
    with pytest.raises(StudioError, match="Unknown field"):
        treatment.new_samples(edited(video), {"count": 3})


# ── the usual ─────────────────────────────────────────────────────────────────


def test_save_as_usual_writes_one_file_for_every_video(video):
    ctx = edited(video, pace="fast")
    treatment.change_treatment(ctx, {"take_out": {"repeats": False}})
    state = treatment.save_as_usual(ctx, {})
    usual = json.loads((projects_root() / treatment.USUAL_FILE).read_text())
    switches = {"pauses": True, "fillers": True, "repeats": False, "likes": True}
    assert usual["pace"] == "fast" and usual["take_out"] == switches
    assert state["usual"] == {"pace": "fast", "take_out": switches, "fine": {"gap_length": 0.4, "rhythm": 4.0}}
    assert treatment.starting_treatment() == Treatment(pace="fast", take_out=frozenset({"pauses", "fillers", "likes"}))


def test_an_unreadable_usual_is_ignored(video):
    (projects_root()).mkdir(parents=True, exist_ok=True)
    (projects_root() / treatment.USUAL_FILE).write_text("{not json")
    assert treatment.starting_treatment() == Treatment()


# ── what Claude learns ────────────────────────────────────────────────────────


def test_get_edit_tells_claude_what_the_creator_changed(video):
    ctx = edited(video, pace="standard")
    assert "creator" not in tools.get_edit(str(video))
    rid = row(treatment.page_state(ctx), "repeat")["id"]
    treatment.change_treatment(ctx, {"pace": "fast", "take_out": {"fillers": False}})
    treatment.set_cut_state(ctx, {"id": rid, "state": "put_back"})
    treatment.add_keep(ctx, {"start": 15.6, "end": 15.8, "note": "the sign-off"})
    result = tools.get_edit(str(video))
    creator = result["creator"]
    assert "review" not in result  # the older page's report stays for the older page
    assert creator["pace"] == {"claude": "standard", "creator": "fast"}
    assert creator["take_out"] == {"fillers": False}
    assert creator["put_back"][0]["reason"] == RETAKE["reason"]
    assert creator["kept"][0]["note"] == "the sign-off" and creator["kept"][0]["text"] == "thanks for watching."


def test_once_claude_follows_the_creators_pace_it_is_no_longer_a_change(video):
    ctx = edited(video, pace="standard")
    treatment.change_treatment(ctx, {"pace": "fast"})
    tools.set_edit(str(video), [UM, RETAKE], auto_tighten=True, **QUIET)
    assert "creator" not in tools.get_edit(str(video))


# ── what changed ──────────────────────────────────────────────────────────────


def test_what_changed_matches_trims_that_only_moved():
    before = treatment.Snapshot(trims=[(1.0, 2.0), (5.0, 6.0)], edited=100.0, flagged=[(10.0, 12.0)])
    after = treatment.Snapshot(trims=[(1.1, 1.9), (8.0, 9.0), (9.5, 9.8)], edited=98.5,
                               flagged=[(10.05, 12.0), (20.0, 21.0)])
    assert treatment.what_changed(before, after) == {
        "trims": 1, "trims_added": 2, "trims_dropped": 1, "seconds": -1.5, "need_a_look": 2, "new_flags": 1,
        "overlays_hidden": 0, "your_cuts": 0, "words_cut": 0, "words_back": 0, "keeps_changed": 0, "likes": 0,
    }


def test_what_changed_counts_only_the_photos_a_change_newly_hid():
    def snap(hidden):
        return treatment.Snapshot(trims=[], edited=100.0, flagged=[], hidden=hidden)

    assert treatment.what_changed(snap(["ov1"]), snap(["ov1", "ov2", "ov3"]))["overlays_hidden"] == 2
    assert treatment.what_changed(snap(["ov1"]), snap(["ov1"]))["overlays_hidden"] == 0
    assert treatment.what_changed(snap(["ov1", "ov2"]), snap([]))["overlays_hidden"] == 0  # shown again


def test_each_word_says_whether_the_edit_skips_it(video):
    state = treatment.page_state(edited(video, cuts=[RETAKE], auto_tighten=False))
    flags = {(text, start): cut for text, start, _end, cut, _why in state["words"]}
    assert flags[("video.", 10.15)] == 1 and flags[("today", 8.0)] == 1
    assert flags[("today", 3.9)] == 0 and flags[("this", 12.0)] == 0
    assert ("(sound)", 6.0) in flags  # sounds read as words, in brackets


# ── clusters ──────────────────────────────────────────────────────────────────


def long_talk(video, minutes=6):
    """Swap the fixture's transcript for a longer one: a sentence every 5.4 s. The edit is saved by hand."""
    words, t = [], 0.0
    while t + 5.0 <= minutes * 60:
        for i, text in enumerate(("one", "two", "three", "four.")):
            words.append({"word": text, "start": round(t + i * 1.25, 3), "end": round(t + (i + 1) * 1.25 - 0.05, 3)})
        t += 5.4
    video.with_suffix(".words.json").write_text(json.dumps(words))
    return minutes * 60.0


def save_cuts(video, duration, spans):
    project = open_project(str(video))
    write_json_atomic(project.edit_path, {
        "version": 1, "video": str(video), "duration": duration,
        "cuts": [{"start": a, "end": b, "reason": why, "source": source} for a, b, why, source in spans],
    })
    return treatment.load_context(project, duration=duration, silences=no_silences, join_rows=lambda *a, **k: [])


def pauses(start, count):
    """``count`` short removals in the gaps between sentences, from ``start`` on."""
    first = 5.4 * round(start / 5.4)
    return [(round(first + 5.4 * i + 5.0, 2), round(first + 5.4 * i + 5.35, 2), "auto: pause: 0.4s", "auto")
            for i in range(count)]


def test_each_cluster_is_numbered_timed_counted_and_shaded(video):
    duration = long_talk(video)
    ctx = save_cuts(video, duration, pauses(20.0, 4) + pauses(250.0, 12) + [(100.0, 104.0, "a tangent", "claude")])
    state = treatment.page_state(ctx)
    found = state["clusters"]
    assert len(found) >= 2 and all(set(c) == CLUSTER_KEYS for c in found)
    assert [c["number"] for c in found] == list(range(1, len(found) + 1))
    assert [c["start"] for c in found] == sorted(c["start"] for c in found)
    busiest = max(found, key=lambda c: c["edits"])
    assert busiest["level"] == 3 and busiest["edits"] >= 5 and busiest["start"] >= 240.0
    assert {c["level"] for c in found} <= {1, 2, 3} and min(c["level"] for c in found) < 3
    for c in found:
        assert c["clock_range"].startswith(c["clock"] + " to ")
        assert c["out"].endswith(" out") and ":" in c["out"] and "." not in c["out"]
        assert c["start"] < c["end"] <= duration


def test_a_clusters_time_out_reads_as_whole_seconds_on_a_clock(video):
    duration = long_talk(video)
    ctx = save_cuts(video, duration, [(100.0, 114.4, "a tangent", "claude")])
    (cluster,) = treatment.page_state(ctx)["clusters"]
    assert (cluster["edits"], cluster["seconds_removed"], cluster["out"]) == (1, 14.4, "0:14 out")
    assert cluster["level"] == 2  # one cluster has nothing to compare against


def test_busy_stays_one_more_round_with_the_same_entries(video):
    duration = long_talk(video)
    state = treatment.page_state(save_cuts(video, duration, pauses(20.0, 4) + pauses(250.0, 12)))
    assert state["busy"] == [{**c, "cuts": c["edits"]} for c in state["clusters"]] and state["busy"]


def test_an_edit_that_removes_nothing_has_no_clusters(video):
    tools.set_edit(str(video), [], **QUIET)
    state = treatment.page_state(ctx_for(video))
    assert state["clusters"] == [] and state["busy"] == []


@pytest.mark.parametrize("seconds, shown", [
    (14.2, "0:14 out"), (3.6, "0:04 out"), (75.0, "1:15 out"), (0.4, "0:00 out"),
    (0.5, "0:01 out"), (14.5, "0:15 out"), (15.5, "0:16 out"), (59.6, "1:00 out"),
])
def test_time_out_is_a_clock_to_the_nearest_second_with_a_half_going_up(seconds, shown):
    assert treatment.time_out(seconds) == shown


def test_a_cluster_of_many_tiny_removals_never_reads_as_no_time_out(video):
    duration = long_talk(video)
    tiny = [(a, round(a + 0.03, 2), why, source) for a, _b, why, source in pauses(250.0, 8)]
    found = treatment.page_state(save_cuts(video, duration, tiny))["clusters"]
    assert found and all(c["seconds_removed"] < 0.5 and c["out"] == "0:01 out" for c in found)


# ── what each sample says ─────────────────────────────────────────────────────


def sample_details(state):
    return {s["key"]: s["detail"] for s in state["samples"]}


def save_samples(project, samples):
    write_json_atomic(project.root / treatment.TREATMENT_FILE, {"version": 1, "samples": samples, "used": []})


def sample(key, start, end, *, anchor=None, fallback=False, why="Why this stretch."):
    return {"key": key, "start": start, "end": end, "why": why, "anchor": anchor, "fallback": fallback}


def out_between(state, start, end):
    """The time the edit removes between two source times, as the page would word it."""
    seconds = sum(max(0.0, min(b, end) - max(a, start)) for a, b in state["removed"])
    assert seconds >= 2.0, "the fixture's cut should take a few seconds out of this stretch"
    return f"0:{round(seconds):02d} out"


def test_no_sample_detail_or_why_says_trim(video):
    ctx = edited(video, pace="fast")
    for state in (treatment.page_state(ctx), treatment.new_samples(ctx, {}),
                  treatment.change_treatment(ctx, {"take_out": {"pauses": False}})):
        for s in state["samples"]:
            assert "trim" not in s["detail"].lower() and "trim" not in s["why"].lower(), s


def test_a_sample_says_how_much_time_comes_out_of_it(video):
    ctx = edited(video, cuts=[RETAKE], auto_tighten=False)
    save_samples(ctx.project, [
        sample("rhythm", 3.9, 14.2), sample("big_cut", 0.5, 2.4), sample("joke", 12.0, 17.0),
    ])
    state = treatment.page_state(ctx)
    assert sample_details(state) == {
        "rhythm": out_between(state, 3.9, 14.2), "big_cut": "nothing out", "joke": "nothing out",
    }


def test_the_big_cut_sample_says_put_back_once_the_creator_put_it_back(video):
    ctx = edited(video, cuts=[RETAKE], auto_tighten=False)
    anchor = [row(treatment.page_state(ctx), "repeat")[k] for k in ("start", "end")]
    save_samples(ctx.project, [
        sample("rhythm", 0.5, 2.4), sample("big_cut", 3.9, 14.2, anchor=anchor), sample("joke", 15.5, 17.0),
    ])
    state = treatment.page_state(ctx)
    assert sample_details(state)["big_cut"] == out_between(state, 3.9, 14.2)
    rid = row(state, "repeat")["id"]
    state = treatment.set_cut_state(ctx, {"id": rid, "state": "put_back"})
    assert sample_details(state)["big_cut"] == "put back"


def test_the_joke_sample_says_whether_its_laugh_plays_whole(video):
    ctx = edited(video, cuts=[UM], auto_tighten=False)
    samples = [sample("rhythm", 12.0, 17.0), sample("big_cut", 8.0, 10.7),
               sample("joke", 3.9, 6.6, anchor=[6.0, 6.6])]
    save_samples(ctx.project, samples)
    assert sample_details(treatment.page_state(ctx))["joke"] == "laugh whole"
    tools.set_edit(str(video), [UM, {"start": 5.4, "end": 6.4, "reason": "cuts into the laugh", "kind": "other"}],
                   **QUIET)
    save_samples(ctx.project, samples)
    assert sample_details(treatment.page_state(ctx_for(video)))["joke"] == "laugh cut"


def test_a_stand_in_joke_sample_says_the_time_out_and_nothing_about_a_laugh(video):
    ctx = edited(video, cuts=[RETAKE], auto_tighten=False)
    save_samples(ctx.project, [
        sample("rhythm", 0.5, 2.4), sample("big_cut", 12.0, 14.2),
        sample("joke", 3.9, 14.2, anchor=None, fallback=True),
    ])
    state = treatment.page_state(ctx)
    assert sample_details(state)["joke"] == out_between(state, 3.9, 14.2)


def test_samples_saved_with_the_earlier_wording_stay_put_and_read_plain(video):
    ctx = edited(video)
    save_samples(ctx.project, [
        sample("rhythm", 0.5, 2.4, why="The stretch with the most pause trims."),
        sample("big_cut", 3.9, 10.7), sample("joke", 12.0, 17.0),
    ])
    rhythm = treatment.page_state(ctx)["samples"][0]
    assert rhythm["why"] == "The stretch where the most pauses come out."
    assert (rhythm["start"], rhythm["end"]) == (0.5, 2.4)


# ── photos and clips, view only ───────────────────────────────────────────────


def overlay(oid, start, end, *, kind="image", file="/somewhere/private/berlin.jpg", layer=1, **extra):
    o = {"id": oid, "file": file, "kind": kind, "start": start, "end": end, "layer": layer, "starts_on": None}
    if kind == "video":
        o.update(clip_in=0.0, clip_out=end - start)
    return {**o, **extra}


def save_overlays(video, overlays):
    write_json_atomic(overlays_path(open_project(str(video))), {"version": 1, "overlays": overlays, "next_id": 9})


def test_with_no_photos_saved_the_list_is_empty(video):
    assert treatment.page_state(edited(video))["overlays"] == []


def test_the_state_lists_saved_photos_and_clips_with_their_status_under_this_edit(video):
    ctx = edited(video, cuts=[RETAKE], auto_tighten=False)
    save_overlays(video, [
        overlay("ov3", 15.5, 17.0, kind="video", file="/somewhere/private/walk.mp4", layer=2),
        overlay("ov1", 8.2, 10.4),                                   # inside the cut
        overlay("ov2", 5.0, 13.0, file="/elsewhere/messages.png"),   # the cut falls under it
    ])
    shown = treatment.page_state(ctx)["overlays"]
    assert all(set(o) == OVERLAY_KEYS for o in shown)
    assert [(o["id"], o["name"], o["kind"], o["status"]) for o in shown] == [
        ("ov2", "messages.png", "photo", "shortened"),
        ("ov1", "berlin.jpg", "photo", "hidden"),
        ("ov3", "walk.mp4", "clip", "shown"),
    ]
    assert [o["tag"] for o in shown] == ["Photo: messages.png", "Photo: berlin.jpg", "Clip: walk.mp4"]
    assert [(o["start"], o["end"], o["clock"]) for o in shown][1] == (8.2, 10.4, "0:08")
    assert "somewhere" not in json.dumps(shown) and "elsewhere" not in json.dumps(shown)


def test_a_change_that_hides_a_photo_says_so_once(video):
    ctx = edited(video, cuts=[RETAKE], auto_tighten=False)
    save_overlays(video, [overlay("ov1", 8.2, 10.4), overlay("ov2", 12.0, 14.0)])
    rid = row(treatment.page_state(ctx), "repeat")["id"]
    back = treatment.set_cut_state(ctx, {"id": rid, "state": "put_back"})
    assert back["changed"]["overlays_hidden"] == 0
    assert [o["status"] for o in back["overlays"]] == ["shown", "shown"]
    out = treatment.set_cut_state(ctx, {"id": rid, "state": "kept_out"})
    assert out["changed"]["overlays_hidden"] == 1
    assert [o["status"] for o in out["overlays"]] == ["hidden", "shown"]
    still = treatment.change_treatment(ctx, {"pace": "natural"})
    assert still["changed"]["overlays_hidden"] == 0 and still["overlays"][0]["status"] == "hidden"


def test_no_page_action_writes_the_saved_photos(video):
    ctx = edited(video)
    save_overlays(video, [overlay("ov1", 8.2, 10.4)])
    path = overlays_path(open_project(str(video)))
    before = path.read_bytes()
    treatment.change_treatment(ctx, {"pace": "fast"})
    treatment.add_keep(ctx, {"start": 9.0, "end": 9.2})
    treatment.new_samples(ctx, {})
    treatment.save_as_usual(ctx, {})
    assert path.read_bytes() == before


@pytest.mark.parametrize("saved_file", [
    "{not json",
    json.dumps({"overlays": [{"id": "ov1"}]}),
    json.dumps({"overlays": [{"id": "ov1", "file": "/x/a.mp4", "kind": "video", "start": 1.0, "end": 5.0,
                              "layer": 1}]}),   # a clip with no clip_in or clip_out
    json.dumps({"overlays": [{"id": "ov1", "file": "/x/a.jpg", "kind": "image", "start": "soon", "end": 5.0,
                              "layer": 1}]}),
])
def test_a_saved_photo_list_that_cannot_be_read_shows_none_and_the_page_still_works(video, saved_file, caplog):
    ctx = edited(video)
    overlays_path(open_project(str(video))).write_text(saved_file)
    state = treatment.page_state(ctx)
    assert state["overlays"] == [] and len(state["rows"]) == 2
    assert "can't be read" in caplog.text
    assert treatment.change_treatment(ctx, {"pace": "fast"})["changed"]["overlays_hidden"] == 0


# ── the export in the state ───────────────────────────────────────────────────


def test_with_no_way_to_read_exports_the_state_says_nothing_was_exported(video):
    assert treatment.page_state(edited(video))["export"] == nothing_exported()


def test_the_state_reads_the_export_against_the_edit_as_saved_now(video):
    seen = []

    def export_for(edit):
        seen.append(edit["treatment"]["pace"])
        return {**nothing_exported(), "state": "running"}

    tools.set_edit(str(video), [UM, RETAKE], auto_tighten=True, **QUIET)
    ctx = treatment.load_context(open_project(str(video)), duration=DURATION, silences=no_silences,
                                 export_for=export_for)
    assert treatment.page_state(ctx)["export"]["state"] == "running"
    state = treatment.change_treatment(ctx, {"pace": "fast"})
    assert state["export"]["state"] == "running"
    assert seen == ["standard", "fast"]  # read after the change was saved, not before


# ── a word protects its sound ─────────────────────────────────────────────────
#
# The take is ``short_word``: the aligner gave "take" 1 ms in the pause after "just", and
# "um," sits alone between two pauses.

HEARD = {"silences": short_word.silences, "labels": unmeasured_labels}
# The pause either side of "um," as one removal, the way the pace lands one on a word said quietly.
PAUSE_OVER_THE_WORD_ALONE = {"start": 5.135, "end": 6.485, "reason": "pause: 1.4s", "kind": "pauses"}
PAUSE_OVER_TAKE = {"start": 1.512, "end": 2.328, "reason": "pause: 0.8s", "kind": "pauses"}


def heard(video):
    return treatment.load_context(open_project(str(video)), duration=DURATION, silences=short_word.silences)


def overlapping(state, span):
    return [r for r in state["removed"] if r[0] < span[1] and span[0] < r[1]]


def plays(state, text):
    return [w[3] for w in state["words"] if w[0] == text] == [0]


@pytest.mark.parametrize("name", list(PACES))
def test_no_pace_takes_a_word_that_is_no_filler(video, name):
    short_word.on_video(video)
    tools.set_edit(str(video), [], auto_tighten=True, pace=name, **HEARD)
    state = treatment.page_state(heard(video))
    assert state["word_times"] == "measured" and state["durations"]["saved"] > 10
    assert [w[0] for w in state["words"] if w[3]] in ([], ["um,"]), "nothing goes but the filler"
    assert not overlapping(state, short_word.TAKE), "the sound of \"take\" plays"
    fillers = state["take_out"][1]
    assert fillers["words"] == ([["um", 1]] if not plays(state, "um,") else []), "and the switch says so"


def test_a_removal_planned_over_a_word_is_shortened_around_it(video, monkeypatch):
    # As ClipForge planned it on the real take from Fast on: one removal over the pause that holds "take".
    short_word.on_video(video)
    monkeypatch.setattr(treatment, "plan_auto_cuts", lambda *_a, **_k: [dict(PAUSE_OVER_TAKE)])
    tools.set_edit(str(video), [], auto_tighten=True, pace="max", **HEARD)
    state = treatment.page_state(heard(video))
    assert plays(state, "take") and not overlapping(state, short_word.TAKE)
    assert state["removed"] == [[1.507, 1.932], [2.162, 2.303]], "the quiet either side of it still goes"
    assert state["take_out"][1]["count"] == 0 and state["take_out"][1]["words"] == []


def test_switching_filler_words_off_brings_back_a_filler_inside_a_pause(video, monkeypatch):
    short_word.on_video(video)
    monkeypatch.setattr(treatment, "plan_auto_cuts", lambda *_a, **_k: [dict(PAUSE_OVER_THE_WORD_ALONE)])
    tools.set_edit(str(video), [], auto_tighten=True, pace="max", **HEARD)
    on = treatment.page_state(heard(video))
    assert not plays(on, "um,") and [w[4] for w in on["words"] if w[0] == "um,"] == [1]
    pauses, fillers = on["take_out"][0], on["take_out"][1]
    assert (fillers["count"], fillers["words"]) == (1, [["um", 1]]), "the switch covers a filler a pause took"
    assert pauses["count"] == 1 and on["paces"][-1]["trims"] == 1, "one removal, filed as a long pause"
    assert fillers["seconds"] == pytest.approx(0.1, abs=0.05), "the word's own length"
    assert pauses["seconds"] + fillers["seconds"] == pytest.approx(on["paces"][-1]["seconds_saved"], abs=0.11)

    off = treatment.change_treatment(heard(video), {"take_out": {"fillers": False}})
    assert plays(off, "um,") and not overlapping(off, short_word.ALONE_ROOM), "the word is back, whole"
    assert len(off["removed"]) == 2, "and the pause either side of it still goes"
    assert off["changed"]["seconds"] > 0 and off["paces"][-1]["trims"] == 2
    assert (off["take_out"][1]["count"], off["take_out"][1]["words"]) == (1, [["um", 1]]), "what it would take, on again"

    again = treatment.change_treatment(heard(video), {"take_out": {"fillers": True}})
    assert again["removed"] == on["removed"]


def test_with_long_pauses_off_the_filler_switch_counts_what_is_left_for_it(video, monkeypatch):
    short_word.on_video(video)
    named = {"start": 5.585, "end": 5.735, "reason": "filler: um", "kind": "fillers"}
    monkeypatch.setattr(treatment, "plan_auto_cuts", lambda *_a, **_k: [dict(PAUSE_OVER_TAKE), dict(named)])
    tools.set_edit(str(video), [], auto_tighten=True, pace="max", **HEARD)
    state = treatment.change_treatment(heard(video), {"take_out": {"pauses": False}})
    assert [w[0] for w in state["words"] if w[3]] == ["um,"]
    assert state["take_out"][1]["count"] == 1 and state["paces"][-1]["trims"] == 1
    assert state["take_out"][0]["count"] == 2, "what Long pauses would take, on again: the quiet either side of \"take\""


def test_a_word_she_brings_back_plays_whole(video):
    short_word.on_video(video)
    sentence = {"start": 0.9, "end": 3.5, "reason": "said better later", "kind": "repeat"}
    tools.set_edit(str(video), [sentence], **HEARD)
    ctx = heard(video)
    before = treatment.page_state(ctx)
    struck = next(w for w in before["words"] if w[0] == "take")
    assert struck[1:] == [*short_word.TAKE, 1, treatment.REMOVED_BY_CLAUDE], "the page holds the word with its room"
    # A double-click on the struck word sends the word's own times.
    state = treatment.add_keep(ctx, {"start": struck[1], "end": struck[2], "exact": True})
    assert state["changed"]["words_back"] == 1 and plays(state, "take")
    assert not overlapping(state, short_word.TAKE), "no removed span is left over any of its sound"
    assert not overlapping(state, short_word.TAKE_SOUNDS)
    assert [r["removed"] for r in state["rows"]] == ["You can just", "one step at a time."], "the rest of the cut stays cut"


def test_a_filler_she_brings_back_stays_back_at_every_pace(video, monkeypatch):
    short_word.on_video(video)
    monkeypatch.setattr(treatment, "plan_auto_cuts", lambda *_a, **_k: [dict(PAUSE_OVER_THE_WORD_ALONE)])
    tools.set_edit(str(video), [], auto_tighten=True, pace="hard", **HEARD)
    ctx = heard(video)
    struck = next(w for w in treatment.page_state(ctx)["words"] if w[0] == "um,")
    state = treatment.add_keep(ctx, {"start": struck[1], "end": struck[2], "exact": True})
    assert plays(state, "um,") and not overlapping(state, short_word.ALONE_ROOM)
    assert state["take_out"][1]["count"] == 0, "the switch has nothing left to take here"
    for name in PACES:
        moved = treatment.change_treatment(heard(video), {"pace": name})
        assert plays(moved, "um,") and not overlapping(moved, short_word.ALONE_ROOM), name


# ── no removal inside a word ──────────────────────────────────────────────────
#
# The take is ``positivity``: on the real take every stop cut "-tivity" as a pause.

# Every stop, and settings of her own from the gentlest to the hardest the sliders make.
SETTINGS = [Treatment(pace=name) for name in PACES] + [
    Treatment(pace="custom", fine=fine) for fine in ((0.5, 3.5), (0.15, 5.0), (0.3, 1.0), (1.5, 2.0))
]


def edges_inside_words(cuts, words):
    """The edges of automatic removals that sit inside a spoken word, to the millisecond."""
    spoken = [w for w in words if w.get("type") != "event"]
    return [
        (t, w["word"]) for c in cuts if c.get("source") == "auto" or c.get("auto_trims")
        for t in (c["start"], c["end"]) for w in spoken if w["start"] + 0.001 < t < w["end"] - 0.001
    ]


def takes():
    """``(words, silences, Claude's cuts, length)`` of three takes: two measured, one on the transcript's times."""
    return {
        "positivity": (
            word_times.with_room(positivity.aligned(), positivity.silences()), positivity.silences(), [],
            positivity.DURATION,
        ),
        "positivity, soft edges": (
            word_times.with_room(positivity.aligned(), positivity.silences(), positivity.soft()),
            positivity.silences(), [], positivity.DURATION,
        ),
        "short_word": (
            word_times.with_room(short_word.aligned(), short_word.silences()), short_word.silences(), [], DURATION,
        ),
        "estimated": (make_words(), [], [UM, RETAKE], DURATION),
    }


@pytest.mark.parametrize("setting", SETTINGS, ids=lambda t: t.pace if t.fine is None else f"custom-{t.fine}")
@pytest.mark.parametrize("take", ["positivity", "positivity, soft edges", "short_word", "estimated"])
def test_no_automatic_removal_starts_or_ends_inside_a_word(setting, take):
    words, quiet, cuts, length = takes()[take]
    made = make_edit(cuts, words, length, treatment=setting, silences=quiet, sounds=[], keeps=[])
    assert edges_inside_words(made.outcome.cuts, words) == []
    assert made.outcome.cuts or setting.shortest_pause() > 1.0, "every take has a pause of a second to cut"
    if take.startswith("positivity"):
        whole = positivity.times_of(words, "positivity")
        assert not [c for c in made.outcome.cuts if c["start"] < whole[1] and whole[0] < c["end"]], "the word plays whole"


@pytest.mark.parametrize("setting", SETTINGS, ids=lambda t: t.pace if t.fine is None else f"custom-{t.fine}")
def test_the_room_alone_keeps_every_pace_off_positivity(setting, monkeypatch):
    # With the last check off, what keeps "-tivity" in is the room the word was given.
    monkeypatch.setattr(treatment.edits, "keep_words_whole", lambda cuts, _words, _held: (cuts, 0))
    words, quiet, _, length = takes()["positivity"]
    made = make_edit([], words, length, treatment=setting, silences=quiet, sounds=[], keeps=[])
    whole = positivity.WHOLE
    assert not [c for c in made.outcome.cuts if c["start"] < whole[1] and whole[0] < c["end"]]


def test_a_removal_planned_into_the_quiet_inside_a_word_gives_the_word_back(monkeypatch):
    # A plan made on other word times: it starts in the quiet between the "s" and the "-tivity" of "positivity".
    # ClipForge leaves an edge that sits in measured silence where it is, inside a word or not.
    words, quiet, _, _ = takes()["positivity"]
    stale = {"start": 23.75, "end": 24.95, "reason": "pause: 1.2s", "kind": "pauses"}
    monkeypatch.setattr(treatment, "plan_auto_cuts", lambda *_a, **_k: [dict(stale)])
    made = make_edit([], words, positivity.DURATION, treatment=Treatment(pace="max"), silences=quiet, sounds=[], keeps=[])
    assert [(c["start"], c["end"]) for c in made.outcome.cuts] == [(positivity.WHOLE[1], 24.95)]
    assert made.edges_moved_out_of_words == 1


def test_claude_is_told_when_a_removal_had_to_give_a_word_back(video, monkeypatch):
    # The pause after "just", planned to end in the closed mouth before the "t" of "take".
    short_word.on_video(video)
    stale = {"start": 1.6, "end": 2.0, "reason": "pause: 0.4s", "kind": "pauses"}
    monkeypatch.setattr(treatment, "plan_auto_cuts", lambda *_a, **_k: [dict(stale)])
    result = tools.set_edit(str(video), [], auto_tighten=True, pace="max", **HEARD)
    assert result["auto"]["edges_moved_out_of_words"] == 1
    state = treatment.page_state(heard(video))
    assert plays(state, "take") and not overlapping(state, short_word.TAKE)


def test_her_saved_edit_gives_back_the_word_a_pause_took_when_it_is_next_opened(video):
    project = positivity.on_video_as_saved_before(video)
    ctx = treatment.load_context(project, duration=positivity.DURATION, silences=positivity.silences)
    saved = json.loads(project.edit_path.read_text())
    assert saved["word_times_made_as"] == word_times.ALIGNED_VERSION, "placed again on the word times as made now"
    whole, pause = positivity.WITH_ITS_SOFT_END, positivity.PAUSE_AFTER
    assert not [c for c in saved["cuts"] if c["start"] < whole[1] and whole[0] < c["end"]], "no cut touches the word"
    assert [c for c in saved["cuts"] if pause[0] <= c["start"] and c["end"] <= pause[1]], "the pause after it still goes"
    state = treatment.page_state(ctx)
    assert next(w for w in state["words"] if w[0] == "positivity")[1:4] == [*whole, 0], "it plays, whole"


@pytest.mark.parametrize("name", SIX_STOPS)
def test_no_pace_takes_part_of_a_sound_of_the_transcript(name):
    # On the real take, at Standard, the pause trim took 0.368 s of this 0.64 s "[vocalization]": the page
    # showed it struck while the rest of it played.
    words = [{"word": "now", "start": 82.921, "end": 83.041}, {"word": "season,", "start": 83.041, "end": 83.938},
             {"word": "[vocalization]", "start": 84.08, "end": 84.72, "type": "event"},
             {"word": "after", "start": 84.72, "end": 85.142}, {"word": "Devon", "start": 85.142, "end": 85.543},
             {"word": "winter,", "start": 85.543, "end": 86.103}]
    quiet = [Silence(82.0, 82.921), Silence(83.7713, 84.4533), Silence(84.6801, 84.7743), Silence(84.8301, 84.9309),
             Silence(85.4826, 85.5457), Silence(86.103, 88.0)]
    made = make_edit([], words, 88.0, treatment=Treatment(pace=name), silences=quiet, sounds=[], keeps=[])
    sound = (84.08, 84.72)
    part = [c for c in made.outcome.cuts if c["start"] < sound[1] and sound[0] < c["end"]
            and not (c["start"] <= sound[0] and sound[1] <= c["end"])]
    assert part == [] and made.outcome.cuts, "the pause before it still goes"


@pytest.mark.parametrize("name", ["natural", "standard"])
def test_shortening_a_trim_for_its_pause_never_eats_into_the_filler_it_holds(name):
    # On the real take: "And" (ends 497.12) then a "[vocalization]" the transcript heard from 497.12 to
    # 497.68, voiced 0.54. Measured -25 dB silence dips inside the sound itself (497.213-497.277,
    # 497.330-497.649): leave_pauses used to read those as "air" it could give back from inside the
    # trim, and the last check then found an edge inside the sound and moved it, dropping the cut. The
    # trim is held now, the way her own cuts are, so shortening for the pause never reaches past it.
    words = [
        {"word": "there.", "start": 496.31, "end": 496.57}, {"word": "And", "start": 496.953, "end": 497.12},
        {"word": "[vocalization]", "start": 497.12, "end": 497.68, "type": "event"},
        {"word": "we", "start": 497.68, "end": 498.009}, {"word": "were", "start": 498.009, "end": 498.269},
        {"word": "a", "start": 498.269, "end": 498.402}, {"word": "reliable", "start": 498.402, "end": 499.042},
    ]
    quiet = [
        Silence(496.219, 496.310), Silence(496.509, 496.953), Silence(497.213, 497.277),
        Silence(497.330, 497.649), Silence(497.649, 497.741), Silence(498.327, 498.402),
    ]
    filler = {"start": 497.12, "end": 497.68, "seconds": 0.56, "kind": "sound", "confidence": "possible",
              "voiced": 0.54, "punchline": None}
    made = make_edit([], words, 500.0, treatment=Treatment(pace=name), silences=quiet, sounds=[filler], keeps=[])
    sound = (497.12, 497.68)
    covering = [c for c in made.outcome.cuts if c["start"] <= sound[0] + 0.002 and sound[1] - 0.002 <= c["end"]]
    touching = [c for c in made.outcome.cuts if c["start"] < sound[1] and sound[0] < c["end"]]
    assert covering, f"the umm still plays at {name}: {touching}"
    assert touching == covering, "never only part of it"


def test_a_short_run_on_with_no_gap_from_a_word_is_its_tail_not_a_filler():
    # On the real take: "in" (room-given end 299.44) runs straight into a "[vocalization]" the
    # transcript logged right at that instant, 299.44-299.559 (0.119 s), voiced 0.5, the edge of held.
    # No gap at all and short: this is "in" decaying, not a fresh "uhh", so Filler words leaves it, at
    # every pace (max included, the hardest).
    words = [
        {"word": "season", "start": 298.509, "end": 299.12}, {"word": "in", "start": 299.12, "end": 299.44},
        {"word": "[vocalization]", "start": 299.44, "end": 299.559, "type": "event"},
        {"word": "the", "start": 299.694, "end": 300.09}, {"word": "fields,", "start": 300.09, "end": 300.396},
    ]
    quiet = [Silence(299.559, 299.694)]
    label = {"start": 299.44, "end": 299.559, "seconds": 0.12, "kind": "sound", "confidence": "possible",
             "voiced": 0.5, "punchline": None}
    trims = autocuts.vocalization_filler_trims([label], words)
    assert trims == [], "a run-on this short with no gap is the word's own tail"
    made = make_edit([], words, 301.0, treatment=Treatment(pace="max"), silences=quiet, sounds=[label], keeps=[])
    sound = (299.44, 299.559)
    from_fillers = [c for c in made.outcome.cuts if c.get("kind") == "fillers"
                    and c["start"] < sound[1] and sound[0] < c["end"]]
    assert from_fillers == [], "not taken as a filler, whatever a pause trim later does with the same air"


def test_a_long_run_on_with_no_gap_from_a_word_is_still_a_filler():
    # The real "And" -> "[vocalization]" case above: no gap either (497.12 to 497.12), but 0.56 s, well
    # past the short-run-on cutoff. It is a filler.
    words = [{"word": "And", "start": 496.953, "end": 497.12},
             {"word": "[vocalization]", "start": 497.12, "end": 497.68, "type": "event"},
             {"word": "we", "start": 497.68, "end": 498.009}]
    label = {"start": 497.12, "end": 497.68, "seconds": 0.56, "kind": "sound", "confidence": "possible",
             "voiced": 0.54, "punchline": None}
    trims = autocuts.vocalization_filler_trims([label], words)
    assert trims == [{"start": 497.12, "end": 497.68, "reason": "filler: sound", "kind": "fillers"}]


@pytest.mark.parametrize("name", SIX_STOPS)
def test_a_filler_vocalization_is_removed_whole_by_filler_words(name):
    # The same real "[vocalization]", trimmed to where it really sounds (word_times narrows 84.08-84.72
    # to 84.396-84.72), measured voiced 0.86: an unmistakable held "uhh". Filler words takes it whole,
    # whatever the pace; the pause around it is left to the pace as usual.
    words = [{"word": "now", "start": 82.921, "end": 83.041}, {"word": "season,", "start": 83.041, "end": 83.938},
             {"word": "[vocalization]", "start": 84.396, "end": 84.72, "type": "event"},
             {"word": "after", "start": 84.72, "end": 85.142}, {"word": "Devon", "start": 85.142, "end": 85.543},
             {"word": "winter,", "start": 85.543, "end": 86.103}]
    quiet = [Silence(82.0, 82.921), Silence(83.7713, 84.396), Silence(84.6801, 84.7743), Silence(84.8301, 84.9309),
             Silence(85.4826, 85.5457), Silence(86.103, 88.0)]
    filler = {"start": 84.396, "end": 84.72, "seconds": 0.32, "kind": "sound", "confidence": "possible",
              "voiced": 0.86, "punchline": None}
    made = make_edit([], words, 88.0, treatment=Treatment(pace=name), silences=quiet, sounds=[filler], keeps=[])
    sound = (84.396, 84.72)
    covering = [c for c in made.outcome.cuts if c["start"] <= sound[0] and sound[1] <= c["end"]]
    partial = [c for c in made.outcome.cuts if c["start"] < sound[1] and sound[0] < c["end"]
               and not (c["start"] <= sound[0] and sound[1] <= c["end"])]
    assert covering, f"the umm should be cut whole at {name}"
    assert partial == [], "never only part of it"


def test_a_breath_like_vocalization_is_left_alone_by_filler_words():
    # The same shape, but breathy, not held: voiced 0.2, under FILLER_VOICED_MIN. Filler words leaves it.
    words = [{"word": "now", "start": 82.921, "end": 83.041}, {"word": "season,", "start": 83.041, "end": 83.938},
             {"word": "[vocalization]", "start": 84.396, "end": 84.72, "type": "event"},
             {"word": "after", "start": 84.72, "end": 85.142}]
    quiet = [Silence(82.0, 82.921), Silence(83.7713, 84.396), Silence(84.6801, 84.7743)]
    breath = {"start": 84.396, "end": 84.72, "seconds": 0.32, "kind": "sound", "confidence": "possible",
              "voiced": 0.2, "punchline": None}
    made = make_edit([], words, 86.0, treatment=Treatment(pace="max"), silences=quiet, sounds=[breath], keeps=[])
    sound = (84.396, 84.72)
    touching = [c for c in made.outcome.cuts if c["start"] < sound[1] and sound[0] < c["end"]]
    assert touching == [], "a breath is not a filler"


def test_a_filler_vocalization_counts_and_lists_as_sound_on_the_page():
    words = [{"word": "season,", "start": 83.041, "end": 83.938},
             {"word": "[vocalization]", "start": 84.396, "end": 84.72, "type": "event"},
             {"word": "after", "start": 84.72, "end": 85.142}]
    quiet = [Silence(83.7713, 84.396), Silence(84.6801, 84.7743)]
    filler = {"start": 84.396, "end": 84.72, "seconds": 0.32, "kind": "sound", "confidence": "possible",
              "voiced": 0.86, "punchline": None}
    made = make_edit([], words, 86.0, treatment=Treatment(pace="max"), silences=quiet, sounds=[filler], keeps=[])
    taken = treatment.taken_by_kind(made, words)["fillers"]
    assert taken.count == 1
    assert taken.listed() == [[treatment.SOUND_FILLER_LABEL, 1]]


def test_the_filler_words_switch_off_leaves_the_vocalization_in():
    words = [{"word": "season,", "start": 83.041, "end": 83.938},
             {"word": "[vocalization]", "start": 84.396, "end": 84.72, "type": "event"},
             {"word": "after", "start": 84.72, "end": 85.142}]
    quiet = [Silence(83.7713, 84.396), Silence(84.6801, 84.7743)]
    filler = {"start": 84.396, "end": 84.72, "seconds": 0.32, "kind": "sound", "confidence": "possible",
              "voiced": 0.86, "punchline": None}
    off = frozenset(treatment.edits.SWITCHES) - {"fillers"}
    made = make_edit([], words, 86.0, treatment=Treatment(pace="max", take_out=off), silences=quiet,
                     sounds=[filler], keeps=[])
    sound = (84.396, 84.72)
    assert not [c for c in made.outcome.cuts if c["start"] < sound[1] and sound[0] < c["end"]], \
        "the switch is off; the umm stays"


@pytest.mark.parametrize("pace", ["standard", "max"])
@pytest.mark.parametrize("text", list(seams.RUN_OVER))
def test_a_double_click_on_a_word_the_aligner_ran_over_takes_that_word_and_only_it(video, text, pace):
    seams.on_video(video)
    tools.set_edit(str(video), [], auto_tighten=True, pace=pace, silences=seams.silences, labels=unmeasured_labels)
    ctx = treatment.load_context(open_project(str(video)), duration=DURATION, silences=seams.silences)
    struck = next(w for w in treatment.page_state(ctx)["words"] if w[0] == text)
    state = treatment.add_cut(ctx, {"start": struck[1], "end": struck[2]})
    assert [w[0] for w in state["words"] if w[3]] == [text]
    before = next(w for w in state["words"] if w[0] == seams.RUN_OVER[text][0])
    assert not overlapping(state, before[1:3]), f"\"{before[0]}\" plays whole"
    assert [r for r in state["removed"] if r[0] <= struck[1] and struck[2] <= r[1]], "all of it is cut"


# ── a word judged by reading ──────────────────────────────────────────────────
#
# The same take with "like," alone between the two pauses.


def like_id(video):
    """The id ``find_words`` gives the one "like," of the take."""
    found = tools.find_words(str(video), ["like"], **HEARD)
    assert found["words"] == [{"word": "like", "said": 1, "clean": 1, "already_out": 0}], "the pace took none"
    return next(line.split(" ")[0] for line in found["text"].splitlines() if line.startswith("w"))


@pytest.mark.parametrize("name", list(PACES))
def test_no_pace_takes_a_like(video, monkeypatch, name):
    short_word.on_video(video, alone="like,")
    named = {"start": 5.585, "end": 5.735, "reason": "filler: like", "kind": "fillers"}
    monkeypatch.setattr(treatment, "plan_auto_cuts", lambda *_a, **_k: [dict(PAUSE_OVER_THE_WORD_ALONE), named])
    tools.set_edit(str(video), [], auto_tighten=True, pace=name, **HEARD)
    state = treatment.page_state(heard(video))
    assert plays(state, "like,") and not overlapping(state, short_word.ALONE_ROOM)
    fillers = state["take_out"][1]
    assert (fillers["count"], fillers["words"], set(fillers["by_pace"].values())) == (0, [], {0})
    assert state["removed"], "the pause beside it still goes"
    assert all(kind == "pauses" for _a, _b, kind in state["trims"]), "what ClipForge named for the word is a pause now"


def test_a_like_is_claudes_to_pick_and_the_creators_to_switch(video, monkeypatch):
    short_word.on_video(video, alone="like,")
    monkeypatch.setattr(treatment, "plan_auto_cuts", lambda *_a, **_k: [dict(PAUSE_OVER_THE_WORD_ALONE)])
    tools.set_edit(str(video), [], auto_tighten=True, pace="max", **HEARD)
    result = tools.set_edit(str(video), [], auto_tighten=True, pace="max",
                            picks=[{"id": like_id(video), "reason": "a pause word"}], **HEARD)
    assert (result["picks"]["out"], result["picks"]["left_in"]) == (1, [])
    on = treatment.page_state(heard(video))
    assert [w[4] for w in on["words"] if w[0] == "like,"] == [treatment.REMOVED_AS_PICKED]
    assert on["take_out"][3]["count"] == 1 and on["take_out"][1]["count"] == 0, "Filler likes holds it, Filler words does not"
    assert len(on["removed"]) == 1, "the picked word and the pause around it go as one"

    no_fillers = treatment.change_treatment(heard(video), {"take_out": {"fillers": False}})
    assert no_fillers["removed"] == on["removed"], "Filler words has no say over it"
    off = treatment.change_treatment(heard(video), {"take_out": {"likes": False}})
    assert plays(off, "like,") and not overlapping(off, short_word.ALONE_ROOM), "with Filler likes off the pace takes none of it"
    assert off["changed"]["likes"] == -1 and len(off["removed"]) == 2
