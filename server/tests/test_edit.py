import json

import pytest
from lumr_studio.engine.audio_boundaries import Silence

from lumr_studio import edit as edits
from lumr_studio import tools
from lumr_studio.edit import CutRejected, build_edit, place_cut
from lumr_studio.errors import StudioError
from lumr_studio.project import open_project, write_json_atomic
from lumr_studio.silences import no_silences

DURATION = 20.0


def _clips_a_kept_word(start, end, words):
    """True if an edge lands strictly inside a word the cut keeps (midpoint outside)."""
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        kept = not (start < mid < end)
        if kept and (w["start"] < start < w["end"] or w["start"] < end < w["end"]):
            return True
    return False


# ── Edge placement ────────────────────────────────────────────────────────────


def test_edge_inside_word_moves_to_word_boundary_gap(words):
    # 1.20 sits inside "everyone" (0.95-1.50, mid 1.225): it is removed.
    # 2.10 sits inside "welcome." (1.85-2.40, mid 2.125): it is kept.
    start, end = place_cut(1.20, 2.10, words, DURATION)
    assert 0.90 <= start <= 0.95  # after "Hello", before "everyone"
    assert 1.80 <= end <= 1.85    # after "and", before "welcome."
    assert not _clips_a_kept_word(start, end, words)


def test_cut_clipping_one_word_without_removing_it_is_rejected(words):
    with pytest.raises(CutRejected, match="Nearest word boundaries are 0.95 and 1.50"):
        place_cut(1.0, 1.2, words, DURATION)


def test_cut_inside_a_pause_is_left_alone(words):
    assert place_cut(2.5, 3.3, words, DURATION) == (2.5, 3.3)


def test_cut_at_the_head_keeps_the_requested_start(words):
    start, end = place_cut(0.0, 1.0, words, DURATION)  # removes "Hello" only
    assert start == 0.0
    assert 0.90 <= end <= 0.95


def test_measured_silence_pulls_the_edge_into_it(words):
    # Gap between "and" (ends 1.80) and "welcome." (starts 1.85); silence inside it.
    silent = [Silence(1.83, 1.845)]
    _, end = place_cut(1.20, 2.10, words, DURATION, silent)
    assert 1.83 <= end <= 1.845


# ── build_edit: validation, merge, reasons ────────────────────────────────────


def test_out_of_range_and_malformed_cuts_are_rejected(words):
    outcome = build_edit(
        [
            {"start": -1, "end": 2, "reason": "x"},
            {"start": 15, "end": 25, "reason": "x"},
            {"start": 5, "end": 5, "reason": "x"},
            {"start": 12, "end": 13, "reason": " "},
            {"start": "a", "end": 13, "reason": "x"},
        ],
        words, DURATION,
    )
    whys = [r["why"] for r in outcome.rejected]
    assert [r["index"] for r in outcome.rejected] == [0, 1, 2, 3, 4]
    assert "before 0" in whys[0]
    assert "past the end" in whys[1]
    assert "not before end" in whys[2]
    assert "no reason" in whys[3]
    assert "numbers" in whys[4]
    assert outcome.cuts == []


def test_end_just_past_video_end_is_clamped_not_rejected(words):
    outcome = build_edit([{"start": 15.2, "end": DURATION + 0.03, "reason": "outro"}], words, DURATION)
    assert not outcome.rejected
    assert outcome.cuts[-1]["end"] == DURATION


def test_overlapping_cuts_merge_with_both_reasons(words):
    outcome = build_edit(
        [
            {"start": 3.35, "end": 6.05, "reason": "first take"},
            {"start": 5.0, "end": 6.7, "reason": "flub"},
        ],
        words, DURATION,
    )
    assert len(outcome.applied) == 1
    merged = outcome.applied[0]
    assert merged["from_cuts"] == [0, 1]
    assert merged["reason"] == "first take; flub"
    assert not _clips_a_kept_word(merged["start"], merged["end"], words)


def test_adjusted_reports_requested_and_final(words):
    outcome = build_edit([{"start": 1.2, "end": 2.1, "reason": "trim"}], words, DURATION)
    (adj,) = outcome.adjusted
    assert adj["requested"] == {"start": 1.2, "end": 2.1}
    assert adj["final"]["start"] != 1.2 and adj["final"]["end"] != 2.1


def test_rejected_cut_does_not_block_the_others(words):
    outcome = build_edit(
        [{"start": 1.0, "end": 1.2, "reason": "clip"}, {"start": 12.0, "end": 14.3, "reason": "tangent"}],
        words, DURATION,
    )
    assert [r["index"] for r in outcome.rejected] == [0]
    assert outcome.applied[0]["reason"] == "tangent"


def test_removing_everything_fails_loud(words):
    with pytest.raises(StudioError, match="remove the whole video"):
        build_edit([{"start": 0, "end": DURATION, "reason": "all"}], words, DURATION)


# ── Spans that are off limits ─────────────────────────────────────────────────


def test_cut_in_a_span_the_creator_restored_is_rejected(words):
    # The creator restored the retake at 8.0-10.7 on the review page.
    outcome = build_edit(
        [{"start": 7.9, "end": 10.75, "reason": "retake"}], words, DURATION, keeps=[(7.9, 10.75)]
    )
    assert outcome.cuts == []
    why = outcome.rejected[0]["why"]
    assert "the creator restored" in why and "override_keeps=true" in why


def test_override_keeps_lets_the_cut_through(words):
    outcome = build_edit(
        [{"start": 7.9, "end": 10.75, "reason": "retake"}], words, DURATION,
        keeps=[(7.9, 10.75)], override_keeps=True,
    )
    assert len(outcome.cuts) == 1 and outcome.rejected == []


def test_cut_beside_a_restored_span_is_fine(words):
    outcome = build_edit(
        [{"start": 3.3, "end": 3.8, "reason": "filler"}], words, DURATION, keeps=[(7.9, 10.75)]
    )
    assert len(outcome.cuts) == 1 and outcome.rejected == []


def test_automatic_trims_leave_jokes_and_restored_spans_alone(words):
    auto = [
        {"start": 2.6, "end": 3.2, "reason": "pause"},    # free
        {"start": 6.8, "end": 7.8, "reason": "pause"},    # inside the protected joke
        {"start": 10.9, "end": 11.8, "reason": "pause"},  # inside the creator's keep
    ]
    outcome = build_edit(
        [], words, DURATION, auto_cuts=auto, keeps=[(10.8, 12.0)], protected=[(5.3, 7.9)]
    )
    assert outcome.auto_skipped_protected == 1 and outcome.auto_skipped_kept == 1
    assert len(outcome.cuts) == 1 and 2.4 <= outcome.cuts[0]["start"] <= 3.4


def test_claude_may_cut_a_protected_span_on_purpose(words):
    outcome = build_edit(
        [{"start": 3.3, "end": 3.8, "reason": "filler"}], words, DURATION, protected=[(3.0, 4.0)]
    )
    assert len(outcome.cuts) == 1


# ── set_edit through tools: saving ────────────────────────────────────────────


def test_set_edit_persists_reasons_and_get_edit_reads_them(video):
    result = tools.set_edit(
        str(video),
        [{"start": 7.9, "end": 10.8, "reason": "retake of the intro line"}],
        silences=no_silences,
    )
    assert result["removed_seconds"] > 2.5
    assert result["new_duration"] == pytest.approx(DURATION - result["removed_seconds"], abs=0.05)

    saved = json.loads(open_project(str(video)).edit_path.read_text())
    assert saved["cuts"][0]["reason"] == "retake of the intro line"
    assert saved["requested"][0]["reason"] == "retake of the intro line"

    summary = tools.get_edit(str(video))
    assert summary["cuts"][0]["reason"] == "retake of the intro line"
    assert summary["new_duration"] == result["new_duration"]


def test_set_edit_replaces_the_previous_edit(video):
    tools.set_edit(str(video), [{"start": 7.9, "end": 10.8, "reason": "a"}], silences=no_silences)
    tools.set_edit(str(video), [], silences=no_silences)
    assert tools.get_edit(str(video))["cuts"] == []


def test_auto_tighten_adds_auto_cuts_summarized_in_get_edit(video):
    result = tools.set_edit(str(video), [], auto_tighten=True, gap_length=0.4, silences=no_silences)
    count = result["auto"]["cuts_after_merge"]
    assert count > 0 and "cleared" not in result
    summary = tools.get_edit(str(video))
    assert summary["cuts"] == []  # automatic trims are not listed by default
    assert summary["auto"]["count"] == count and summary["auto"]["seconds_removed"] > 0
    assert "cuts" not in summary["auto"]
    assert summary["auto_tighten"] is True and summary["gap_length"] == 0.4
    assert summary["cut_count"] == count

    listed = tools.get_edit(str(video), include_auto=True)["auto"]
    assert len(listed["cuts"]) == count and listed["not_shown"] == 0
    assert all(c["reason"].startswith("auto:") and set(c) == {"start", "end", "clock", "seconds", "reason"}
               for c in listed["cuts"])


def test_include_auto_is_capped(video, monkeypatch):
    monkeypatch.setattr(edits, "AUTO_CUTS_SHOWN", 1)
    tools.set_edit(str(video), [], auto_tighten=True, gap_length=0.4, silences=no_silences)
    listed = tools.get_edit(str(video), include_auto=True)["auto"]
    assert len(listed["cuts"]) == 1
    assert listed["not_shown"] == listed["count"] - 1


def test_claude_cut_merged_with_auto_trims_keeps_a_short_reason(words):
    auto = [{"start": 2.45, "end": 3.35, "reason": "pause: 1.0s"}, {"start": 3.4, "end": 3.7, "reason": "filler: um"}]
    outcome = build_edit([{"start": 3.35, "end": 6.05, "reason": "first take"}], words, DURATION, auto_cuts=auto)
    (applied,) = outcome.applied
    assert applied["reason"] == "first take (+2 automatic trims)"
    assert set(applied) == {"start", "end", "clock", "seconds", "reason", "kind", "from_cuts"}
    saved = next(c for c in outcome.cuts if c["source"] == "claude")
    assert saved["reason"] == "first take" and saved["auto_trims"] == 2


def test_cut_under_a_tenth_of_a_second_is_rejected(words):
    outcome = build_edit([{"start": 2.60, "end": 2.66, "reason": "tiny gap"}], words, DURATION)
    (rejected,) = outcome.rejected
    assert "too short to matter, under 0.1s" in rejected["why"]


def test_all_rejected_leaves_the_saved_edit_alone(video):
    tools.set_edit(str(video), [{"start": 7.9, "end": 10.8, "reason": "retake"}], silences=no_silences)
    before = tools.get_edit(str(video))
    with pytest.raises(StudioError, match="left unchanged") as err:
        tools.set_edit(str(video), [{"start": 9.0, "end": 8.0, "reason": "typo"}], silences=no_silences)
    assert "Cut 0 start 9.0 is not before end 8.0" in str(err.value)
    assert tools.get_edit(str(video)) == before


def test_some_rejected_still_saves_the_good_ones(video):
    result = tools.set_edit(
        str(video),
        [{"start": 9.0, "end": 8.0, "reason": "typo"}, {"start": 12.0, "end": 14.3, "reason": "tangent"}],
        silences=no_silences,
    )
    assert [r["index"] for r in result["rejected"]] == [0]
    assert tools.get_edit(str(video))["cuts"][0]["reason"] == "tangent"


def test_empty_cuts_clear_the_edit_explicitly(video):
    tools.set_edit(str(video), [{"start": 7.9, "end": 10.8, "reason": "retake"}], silences=no_silences)
    result = tools.set_edit(str(video), [], silences=no_silences)
    assert result["cleared"] is True
    assert tools.get_edit(str(video))["cut_count"] == 0


def test_gap_length_needs_auto_tighten_and_a_sane_value(video):
    with pytest.raises(StudioError, match="only applies with auto_tighten"):
        tools.set_edit(str(video), [], gap_length=0.4, silences=no_silences)
    with pytest.raises(StudioError, match="out of range"):
        tools.set_edit(str(video), [], auto_tighten=True, gap_length=9, silences=no_silences)


def test_atomic_save_keeps_the_old_file_on_failure(tmp_path):
    path = tmp_path / "edit.json"
    write_json_atomic(path, {"cuts": [1]})
    with pytest.raises(TypeError):
        write_json_atomic(path, {"cuts": {object()}})
    assert json.loads(path.read_text()) == {"cuts": [1]}
    assert [p.name for p in tmp_path.iterdir()] == ["edit.json"]  # no temp files left


def test_edit_for_a_different_video_length_is_refused(video):
    tools.set_edit(str(video), [], silences=no_silences)
    project = open_project(str(video))
    data = json.loads(project.edit_path.read_text())
    data["duration"] = 99.0
    project.edit_path.write_text(json.dumps(data))
    with pytest.raises(StudioError, match="video changed"):
        tools.get_edit(str(video))


# ── No automatic removal inside a word ────────────────────────────────────────

# "positivity" with its room, and "at" after the pause, as in positivity.py.
POSITIVITY_AT = [{"word": "positivity", "start": 23.251, "end": 24.202}, {"word": "at", "start": 25.07, "end": 25.177}]


def trimmed(start, end, **more):
    return {"start": start, "end": end, "reason": "auto: pause: 1.5s", "source": "auto", "kind": "pauses", **more}


def test_an_automatic_removal_never_starts_or_ends_inside_a_word(caplog):
    assert edits.keep_words_whole([trimmed(23.521, 24.89)], POSITIVITY_AT, []) == ([trimmed(24.202, 24.89)], 1), \
        "the cut the saved edit made: it gives the rest of the word back and keeps the pause"
    assert edits.keep_words_whole([trimmed(24.3, 25.12)], POSITIVITY_AT, []) == ([trimmed(24.3, 25.07)], 1)
    assert edits.keep_words_whole([trimmed(23.5, 25.12)], POSITIVITY_AT, []) == ([trimmed(24.202, 25.07)], 2)
    assert "now 24.202-25.070" in caplog.text, "a move never goes unseen"


def test_an_automatic_removal_with_nothing_worth_cutting_outside_a_word_goes(caplog):
    assert edits.keep_words_whole([trimmed(23.4, 24.0)], POSITIVITY_AT, []) == ([], 1)
    assert edits.keep_words_whole([trimmed(24.15, 24.3)], POSITIVITY_AT, []) == ([], 1), "0.098 s of pause is left"
    assert "what is left goes" in caplog.text and "now" not in caplog.text


def test_a_removal_that_takes_a_word_whole_or_sits_on_its_edge_is_left_alone():
    filler = trimmed(23.2, 24.25, kind="fillers")
    assert edits.keep_words_whole([filler], POSITIVITY_AT, []) == ([filler], 0)
    on_the_edge = trimmed(24.2015, 25.0705)
    assert edits.keep_words_whole([on_the_edge], POSITIVITY_AT, []) == ([on_the_edge], 0), "saved to the millisecond"


def test_the_parts_claude_and_the_creator_chose_are_never_moved():
    claudes = {"start": 23.4, "end": 24.0, "reason": "a false start", "source": "claude", "kind": "false_start"}
    assert edits.keep_words_whole([claudes], POSITIVITY_AT, [(23.4, 24.0)]) == ([claudes], 0), \
        "only what the pace planned is checked"
    joined = {"start": 22.126, "end": 23.6, "reason": "Cut by you", "source": "you", "kind": "yours", "auto_trims": 1}
    assert edits.keep_words_whole([joined], POSITIVITY_AT, [(22.126, 22.866)]) == ([{**joined, "end": 23.251}], 1), \
        "the trim that joined her cut gives the word back; her words stay cut"


# A word, then a "[vocalization]" the transcript heard from 84.08 to 84.72, then another word.
AROUND_A_SOUND = [{"word": "season,", "start": 83.041, "end": 83.938},
                  {"word": "[vocalization]", "start": 84.08, "end": 84.72, "type": "event"},
                  {"word": "after", "start": 84.72, "end": 85.142}]


def test_an_automatic_removal_takes_a_sound_of_the_transcript_whole_or_not_at_all():
    # At Standard the pause trim took 84.08-84.448 of the sound: the page struck it, and the rest played.
    assert edits.keep_words_whole([trimmed(83.938, 84.448)], AROUND_A_SOUND, []) == ([trimmed(83.938, 84.08)], 1)
    whole = trimmed(83.938, 84.72)
    assert edits.keep_words_whole([whole], AROUND_A_SOUND, []) == ([whole], 0), "a trim may take a sound whole"
    hers = {"start": 84.3, "end": 84.72, "reason": "Cut by you", "source": "you", "kind": "yours", "auto_trims": 1}
    assert edits.keep_words_whole([hers], AROUND_A_SOUND, [(84.3, 84.72)]) == ([hers], 0), "her own cut is hers"


def test_a_struck_sound_is_never_only_partly_removed():
    """Whatever an automatic cut first covers of the sound, what ``keep_words_whole`` saves either misses
    it or holds it whole. The page strikes a sound when a removal covers its midpoint; this is the promise
    behind that mark: struck never means only part of it plays.
    """
    sound = (84.08, 84.72)
    mid = (sound[0] + sound[1]) / 2
    candidates = [
        trimmed(83.938, 84.2), trimmed(84.5, 84.9), trimmed(84.08, 84.72),
        trimmed(84.0, 84.1), trimmed(84.6, 85.142), trimmed(84.3, 84.5),
    ]
    for candidate in candidates:
        cuts, _ = edits.keep_words_whole([candidate], AROUND_A_SOUND, [])
        struck = any(c["start"] <= mid <= c["end"] for c in cuts)
        if struck:
            assert any(c["start"] <= sound[0] and sound[1] <= c["end"] for c in cuts), \
                f"{candidate} left the sound struck but not fully removed"


def test_her_cut_of_a_word_inside_another_stays_cut_when_a_trim_joins_it():
    # On a measured take the aligner put "we" (640.24-640.433) inside "plan" (639.979-640.433).
    # She cut "we" by hand, and at Max the pause trim after it joined her cut.
    plan_we = [{"word": "plan", "start": 639.979, "end": 640.433}, {"word": "we", "start": 640.24, "end": 640.433},
               {"word": "now", "start": 641.142, "end": 641.4}]
    joined = {"start": 640.24, "end": 641.142, "reason": "Cut by you", "source": "you", "kind": "yours", "auto_trims": 1}
    assert edits.keep_words_whole([joined], plan_we, [(640.24, 640.433)]) == ([joined], 0), \
        "her edge sits inside \"plan\", and it is hers: \"we\" stays cut"


def test_a_cut_of_hers_that_a_trim_joined_is_saved_with_her_word_whatever_is_left():
    # She cut a 50 ms "a" against "B"; the trim of the pause after joined it and ends inside "B".
    a_b = [{"word": "a", "start": 1.5, "end": 1.55}, {"word": "B", "start": 1.55, "end": 2.0}]
    joined = {"start": 1.5, "end": 1.9, "reason": "Cut by you", "source": "you", "kind": "yours", "auto_trims": 1}
    assert edits.keep_words_whole([joined], a_b, [(1.5, 1.55)]) == ([{**joined, "end": 1.55}], 1), \
        "\"B\" plays whole, and \"a\" stays cut though only 50 ms is left"


def test_no_claude_cut_or_hand_cut_is_touched_while_a_trim_is_kept_off_a_word(monkeypatch):
    from lumr_studio import treatment

    # Estimated times with "W" overlapping "P". She cut "W"; a pause trim after it joins her cut.
    words = [{"word": "P", "start": 1.0, "end": 1.6}, {"word": "W", "start": 1.5, "end": 1.7},
             {"word": "N", "start": 3.0, "end": 3.4}, {"word": "Z", "start": 3.5, "end": 4.0}]
    monkeypatch.setattr(treatment, "plan_auto_cuts", lambda *_a, **_k: [
        {"start": 1.7, "end": 2.9, "reason": "pause: 1.2s", "kind": "pauses"},
    ])
    treatment.forget_recipes()
    made = treatment.make_edit([], words, 5.0, treatment=treatment.Treatment(pace="max"), silences=[], sounds=[],
                               keeps=[], creator_cuts=[{"start": 1.5, "end": 1.7}])
    placed = [(c["start"], c["end"]) for c in made.outcome.creator_placed]
    hers = [c for c in made.outcome.cuts if c["source"] == "you"]
    assert len(hers) == 1 and all(hers[0]["start"] <= a and b <= hers[0]["end"] for a, b in placed), \
        "the saved cut holds all of her cut as placed"
    treatment.forget_recipes()


# ── Times ready to show ───────────────────────────────────────────────────────


def test_clock_never_shows_sixty_seconds():
    assert edits.clock(239.9) == "3:59"
    assert edits.clock(240.0) == "4:00"
    assert edits.clock(240.4) == "4:00"
    assert edits.clock(59.99) == "0:59"
    assert edits.clock(0) == "0:00"
    assert edits.clock(1307.05) == "21:47"


def test_cuts_carry_clock_times(words):
    outcome = build_edit([{"start": 7.9, "end": 10.75, "reason": "retake"}], words, DURATION)
    cut = outcome.cuts[0]
    shown = edits.display_cut(cut)
    assert shown["clock"] == f"{edits.clock(cut['start'])}-{edits.clock(cut['end'])}"
    assert shown["clock"] == "0:06-0:11"
