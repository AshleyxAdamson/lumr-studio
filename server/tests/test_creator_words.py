"""Every string the engine writes for the creator reads as plain words.

The treatment page shows labels, samples, the self-check's ``why`` and error
messages exactly as the server writes them. These tests fail when one of them
holds a decimal time such as 412.30 or a word from ``creator_words``.

One thing she reads may be a decimal: a length on a Fine tune slider. The
tests of that exception are here too, with the texts it lets through.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from creator_words import banned_in, decimal_times_in, decimals_off_the_slider, is_slider_length
from lumr_studio import samples, tools, treatment
from lumr_studio.autocuts import TRIM_KINDS
from lumr_studio import pace, word_times
from lumr_studio.edit import CREATOR_KIND_LABEL, CREATOR_REASON, CUT_KINDS, SWITCHES
from lumr_studio.errors import StudioError
from lumr_studio.joins import WHO_AUTO, WHO_CUT, WHYS
from lumr_studio.pace import PACES
from lumr_studio.project import open_project
from lumr_studio.render_jobs import EXPORT_FAILED, NOTHING_EXPORTED, export_status, start_full_render
from lumr_studio.review import FLAG_LABELS
from lumr_studio.samples import SAMPLE_LABELS
from lumr_studio.silences import no_silences
from lumr_studio.tools import unmeasured_labels
from test_filler_likes import FILLER, STARTER, TALK as LIKES_TALK, WEDGED
from test_render_jobs import StandInRender
from test_treatment import long_talk, overlay, pauses, save_cuts, save_overlays
from test_treatment_page import creator_reads, needs_node

DURATION = 20.0
QUIET = {"silences": no_silences, "labels": unmeasured_labels}
# Claude's cuts on the conftest talk: a filler, the second take of the
# opening line, and a cut inside "this is the part that matters." so a join
# is flagged and the page has a ``why`` to show.
CUTS = [
    {"start": 3.3, "end": 3.8, "reason": "An um before the line.", "kind": "other"},
    {"start": 7.9, "end": 10.75, "reason": "You say the opening line a second time.", "kind": "repeat"},
    {"start": 12.3, "end": 13.1, "reason": "A few words you trip over.", "kind": "false_start"},
]


def assert_plain(text: str, where: str) -> None:
    assert not decimal_times_in(text), f"{where} shows a decimal time: {text!r}"
    assert not banned_in(text), f"{where} uses {banned_in(text)}: {text!r}"


@pytest.fixture(autouse=True)
def fresh_recipes():
    treatment.forget_recipes()
    yield
    treatment.forget_recipes()


def ctx_for(video):
    return treatment.load_context(open_project(str(video)), duration=DURATION, silences=no_silences)


@pytest.fixture
def ctx(video):
    tools.set_edit(str(video), CUTS, auto_tighten=True, **QUIET)
    return ctx_for(video)


def test_every_fixed_label_and_sentence_is_plain():
    fixed = {
        "take-out label": [*TRIM_KINDS.values(), *SWITCHES.values()],
        "slider label": [pace.GAP_LABEL, pace.RHYTHM_LABEL, pace.CUSTOM_LABEL],
        "summary of her own setting": [pace.custom_summary(g) for g in pace.GAP_POSITIONS],
        "cut group label": [CREATOR_KIND_LABEL, *CUT_KINDS.values()],
        "the reason on her own cut": [CREATOR_REASON],
        "pace label": [treatment.pace_label(name) for name in PACES],
        "word times note": [word_times.NOTE_NOT_ALIGNED, word_times.NOTE_TRANSCRIPT_CHANGED],
        "pace summary": [p.summary for p in PACES.values()],
        "flag label": list(FLAG_LABELS.values()),
        "sample label": list(SAMPLE_LABELS.values()),
        "sample why": [v for k, v in vars(samples).items() if k.startswith("WHY_")],
        "self-check why": [w.format(who=who, word="that", length="25 seconds")
                           for w in WHYS.values() for who in (WHO_CUT, WHO_AUTO)],
        "keep note": [treatment.KEPT_BY_YOU, treatment.BROUGHT_BACK_BY_YOU],
        "export message": [NOTHING_EXPORTED, EXPORT_FAILED],
    }
    for where, texts in fixed.items():
        assert texts, where
        for text in texts:
            assert_plain(text, where)


def shown_in(state: dict[str, Any]) -> list[str]:
    """Every string in the page's state that the page puts on screen as the server wrote it.

    All but one kind: what a slider shows at each position is a decimal
    length, and ``slider_texts`` holds those.
    """
    return [
        *(p[k] for p in state["paces"] for k in ("label", "summary")),
        *(t["label"] for t in state["take_out"]),
        *(text for t in state["take_out"] for text, _count in t["words"]),
        state["take_out"][3]["word"],
        *(g["label"] for g in state["groups"]),
        *(r["why"] for r in state["rows"]),
        *(label for r in state["rows"] for label in r["flag_labels"]),
        *(s[k] for s in state["samples"] for k in ("label", "why", "detail", "clock")),
        *(c[k] for c in state["clusters"] for k in ("clock", "clock_range", "out")),
        *(o[k] for o in state["overlays"] for k in ("tag", "clock")),
        state["export"]["message"],
        state["word_times_note"],
        *([state["custom"]["label"], state["custom"]["summary"]] if state["custom"] else []),
        *(r["label"] for r in state["fine_ranges"].values()),
        *(k["note"] for k in state["keeps"]),
        *(k["clock"] for k in state["keeps"]),
        *(r["clock"] for r in state["rows"]),
        *(r["reason"] for r in state["rows"] if r["by"] == "you"),
    ]


def slider_texts(state: dict[str, Any]) -> list[str]:
    """What the two sliders show and say at every position: the one place a decimal reaches her."""
    return [p[k] for r in state["fine_ranges"].values() for p in r["positions"] for k in ("shown", "said")]


# ── the rule on decimals, and its one exception ───────────────────────────────


def test_a_time_in_the_video_is_never_a_decimal():
    for text in ("Cut at 412.30", "9.7 seconds in", "0.35 s", "The pause at 6:52 is 1.5 s long.", "from 3.84 to 5.68"):
        assert decimal_times_in(text), text
    for text in ("6:52", "0:04 out", "0:09.7", "12 edits", "Cluster 3 of 8 · 7:30 to 8:15 · 11 edits", "1s", "25 seconds"):
        assert not decimal_times_in(text), text


def test_a_length_on_a_slider_may_be_a_decimal_and_nothing_else_may():
    for text in ("0.35 s", "0.15 s", "1.5 s", "0.85 s", "2 s", "0 s", "0.35 seconds", "1 second", "2.5 seconds"):
        assert is_slider_length(text), text
    # a time in the video, a length of no slider, and a length with anything around it
    for text in ("412.30 s", "12.5 s", "0.355 s", "0.35", "0.35s", "0.35 sec", ".35 s", "0.35 s shorter", "at 0.35 s",
                 "Cut at 0.35 s", "0:09.7", "0.35 s\n412.30 s", "0.35 minutes", ""):
        assert not is_slider_length(text), text


def test_a_refusal_may_name_the_sliders_own_numbers_and_no_others():
    ends = [0.15, 1.5, 0.05]
    assert decimals_off_the_slider("It goes from 0.15 to 1.5 seconds in steps of 0.05.", ends) == []
    assert decimals_off_the_slider("It goes from 0.15 to 1.5 seconds. You sent 0.37 at 412.30.", ends) == ["0.37", "412.30"]
    assert decimals_off_the_slider("It goes from 0.15 to 1.5 seconds.", []) == ["0.15", "1.5"], "with no slider named, no decimal passes"


def test_every_position_of_both_sliders_reads_as_a_length(ctx):
    state = treatment.page_state(ctx)
    ranges = state["fine_ranges"]
    assert {key: len(r["positions"]) for key, r in ranges.items()} == {"gap_length": 28, "rhythm": 9}
    for key, r in ranges.items():
        assert_plain(r["label"], f"the {key} slider's label")
        for p in r["positions"]:
            assert is_slider_length(p["shown"]) and p["shown"].endswith(" s"), (key, p)
            assert is_slider_length(p["said"]) and p["said"].endswith(("second", "seconds")), (key, p)
            assert p["shown"].removesuffix(" s") == p["said"].split(" ")[0], "both name the same length"
            assert not banned_in(p["shown"] + " " + p["said"])
    assert any(decimal_times_in(text) for text in slider_texts(state)), "the exception is in use, so it is needed"
    for text in shown_in(state):
        assert not is_slider_length(text), f"a slider length outside the sliders: {text!r}"


# ── a state with every new thing in it ────────────────────────────────────────

# On the talk of the filler likes tests: a cut of Claude's, a filler the pace takes, and a pause.
LIKES_CUTS = [{"start": 0.4, "end": 1.7, "reason": "You say this line again later.", "kind": "repeat"}]
PLANNED = [
    {"start": 10.95, "end": 11.55, "reason": "filler: you know", "kind": "fillers"},
    {"start": 12.6, "end": 13.8, "reason": "pause: 1.2s", "kind": "pauses"},
]


@pytest.fixture
def full(video, monkeypatch):
    """The page's states after each thing she can do, on a talk where Claude picked filler likes.

    Every list the word checks read holds something: her own setting, the
    picked likes with one left in, a filler the pace takes, her cut, a word
    she brought back, a flagged join, and estimated word times.
    """
    video.with_suffix(".words.json").write_text(json.dumps(LIKES_TALK))
    monkeypatch.setattr(treatment, "plan_auto_cuts", lambda *_a, **_k: [dict(t) for t in PLANNED])
    tools.set_edit(str(video), LIKES_CUTS, auto_tighten=True,
                   picks=[{"id": i, "reason": "a pause word"} for i in (FILLER, WEDGED, STARTER)], **QUIET)

    def did(action, body):
        return action(ctx_for(video), body)

    answers = {
        "a pace": did(treatment.change_treatment, {"pace": "max"}),
        "a slider": did(treatment.change_treatment, {"fine": {"gap_length": 0.35}}),
        "a switch": did(treatment.change_treatment, {"take_out": {"likes": False}}),
        "the switch again": did(treatment.change_treatment, {"take_out": {"likes": True}}),
        "a cut by hand": did(treatment.add_cut, {"start": 3.1, "end": 3.4}),
        "a word brought back": did(treatment.add_keep, {"start": 8.0, "end": 8.3, "exact": True}),
        "a kept part": did(treatment.add_keep, {"start": 14.0, "end": 15.5, "note": "the goodbye"}),
        "undo": did(treatment.undo, {}),
        "a cut by hand again": did(treatment.add_cut, {"start": 5.0, "end": 5.45}),
    }
    return answers


def test_the_fixture_holds_every_new_thing_she_reads(full):
    state = full["a cut by hand again"]
    likes = state["take_out"][3]
    assert [p["label"] for p in state["paces"]] == ["Natural", "Standard", "Fast", "Tight", "Hard", "Max"]
    assert all(p["summary"] for p in state["paces"])
    assert state["settings"]["pace"] == "custom" and state["custom"]["label"] and state["custom"]["summary"]
    assert state["word_times"] == "estimated" and state["word_times_note"]
    assert likes["ready"] and likes["count"] > 0 and likes["left_in"] > 0 and likes["kept_by_you"] > 0 and likes["words"]
    assert state["take_out"][1]["count"] > 0 and state["take_out"][1]["words"] == [["you know", 1]]
    assert [r["reason"] for r in state["rows"] if r["by"] == "you"] == ["Cut by you", "Cut by you"]
    assert any(k["exact"] for k in state["keeps"])
    for name, answer in full.items():
        assert answer["changed"], f"{name} says what changed"
    assert full["a switch"]["changed"]["likes"] < 0 < full["the switch again"]["changed"]["likes"]
    assert full["a cut by hand"]["changed"]["words_cut"] == 1 and full["a word brought back"]["changed"]["words_back"] == 1


def test_every_new_thing_the_engine_writes_is_plain(full):
    for name, state in full.items():
        texts = shown_in(state)
        assert len(texts) > 40, name
        for text in texts:
            assert_plain(text, f"the state after {name}")


@needs_node
def test_every_sentence_the_page_builds_from_the_engines_answers_is_plain(full):
    for name, state in full.items():
        built = [text for text in creator_reads(state) if text]
        assert len(built) > 40, name
        for text in built:
            assert_plain(text, f"what the page says after {name}")
            assert not is_slider_length(text), f"a slider length outside the sliders, after {name}: {text!r}"


def test_the_page_state_shows_only_plain_words(ctx):
    treatment.add_keep(ctx, {"start": 15.6, "end": 15.8})
    treatment.add_cut(ctx, {"start": 12.55, "end": 12.7})
    treatment.add_keep(ctx, {"start": 8.65, "end": 9.0, "exact": True})
    treatment.change_treatment(ctx, {"fine": {"gap_length": 0.35}})
    state = treatment.page_state(ctx_for(ctx.project.video))
    assert state["custom"], "the fixture needs her own setting to check its words"
    assert [r for r in state["rows"] if r["by"] == "you"] and any(k["exact"] for k in state["keeps"])
    assert any(r["why"] for r in state["rows"]), "the fixture needs a flagged join to check its why"
    for text in shown_in(state):
        assert_plain(text, "the page state")


def test_clusters_and_photos_read_as_plain_words(video):
    duration = long_talk(video)
    busy = pauses(20.0, 4) + pauses(250.0, 12) + [(100.0, 104.0, "a tangent", "claude")]
    save_overlays(video, [overlay("ov1", 101.0, 103.0),
                          overlay("ov2", 30.0, 36.0, kind="video", file="/somewhere/private/walk.mp4")])
    state = treatment.page_state(save_cuts(video, duration, busy))
    assert len(state["clusters"]) >= 2 and len(state["overlays"]) == 2, "the fixture needs both to check their words"
    for text in shown_in(state):
        assert_plain(text, "the page state")


def test_the_export_reads_as_plain_words_in_every_state(video, jobs):
    tools.set_edit(str(video), CUTS, auto_tighten=True, **QUIET)
    project = open_project(str(video))
    edit = json.loads(project.edit_path.read_text())
    render = StandInRender()
    seen = [export_status(project, jobs, edit=edit)]
    try:
        job_id, _ = start_full_render(project, jobs=jobs, render=render)
        assert render.running.wait(5)
        seen.append(export_status(project, jobs, edit=edit))
        render.finish(jobs, job_id)
    finally:
        render.go.set()
    seen.append(export_status(project, jobs, edit=edit))
    assert [s["state"] for s in seen] == ["idle", "running", "done"]
    for status in seen:
        assert status["message"], status
        assert_plain(status["message"], f"the {status['state']} export")


def test_every_row_that_needs_a_look_says_why(ctx):
    for r in treatment.page_state(ctx)["rows"]:
        assert bool(r["why"]) == bool(r["flags"]), r
        assert r["why"][:1].isupper() if r["why"] else True


def _message(action, ctx, body) -> str:
    with pytest.raises(StudioError) as caught:
        action(ctx, body)
    return str(caught.value)


def test_errors_a_click_can_cause_are_plain(ctx):
    rows = treatment.page_state(ctx)["rows"]
    messages = [
        _message(treatment.change_treatment, ctx, {"pace": "brutal"}),
        _message(treatment.set_cut_state, ctx, {"id": "c1.23-4.56", "state": "put_back"}),
        _message(treatment.add_keep, ctx, {"start": 5.0, "end": 4.0}),
        _message(treatment.add_keep, ctx, {"start": 1.0, "end": 25.5}),
        _message(treatment.add_keep, ctx, {"start": 1.0, "end": 4.0, "note": "x" * 201}),
        _message(treatment.remove_keep, ctx, {"id": "k1.23-4.56"}),
        _message(treatment.add_cut, ctx, {"start": 6.7, "end": 7.5}),
        _message(treatment.add_cut, ctx, {"start": 5.0, "end": 4.0}),
        _message(treatment.add_keep, ctx, {"start": 6.7, "end": 7.5, "exact": True}),
        _message(treatment.remove_cut, ctx, {"id": "y1.23-4.56"}),
        _message(treatment.remove_cut, ctx, {"id": rows[0]["id"]}),
        _message(treatment.set_cut_state, ctx, {"id": "y1.23-4.56", "state": "put_back"}),
        _message(treatment.undo, ctx, {}),
    ]
    # A cut inside a part the creator flagged can't be left out again.
    retake = next(r for r in rows if r["group"] == "repeat")
    treatment.add_keep(ctx, {"start": retake["start"], "end": retake["end"]})
    messages.append(_message(treatment.set_cut_state, ctx_for(ctx.project.video), {"id": retake["id"], "state": "kept_out"}))
    for text in messages:
        assert_plain(text, "an error")
    # A slider's refusal names the slider's own ends and step ("from 0.15 to 1.5 seconds in steps
    # of 0.05"). Those may be decimals, and no other number in it may.
    ranges = pace.fine_ranges()
    on_a_slider = [r[k] for r in ranges.values() for k in ("min", "max", "step")]
    for body in ({"fine": {"gap_length": 0.37}}, {"fine": {"rhythm": 9}}, {"pace": "fast", "fine": {"rhythm": 3}},
                 {"pace": "custom"}):
        text = _message(treatment.change_treatment, ctx, body)
        assert not banned_in(text), text
        assert decimals_off_the_slider(text, on_a_slider) == [], text
