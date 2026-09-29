"""The pace dial: which pauses get trimmed, and the pause left at each join."""

import pytest
from lumr_studio.engine.audio_boundaries import Silence

from lumr_studio.engine.microcut_pacing import GAP_LENGTH_MIN, RHYTHM_MAX

from lumr_studio import tools
from lumr_studio.errors import StudioError
from lumr_studio.pace import PACES, air_spans, describe_paces, get_pace, leave_pauses, settle_trims
from lumr_studio.silences import no_silences
from lumr_studio.tools import unmeasured_labels

QUIET = {"silences": no_silences, "labels": unmeasured_labels}


def word(text: str, start: float, end: float) -> dict:
    return {"word": text, "start": start, "end": end}


def trim(start: float, end: float) -> dict:
    return {"start": start, "end": end, "reason": "pause"}


# "One two." then a 2.0s pause, then "Three four," a 1.0s pause, "five" a 1.0s pause "six."
TALK = [
    word("One", 0.0, 0.4), word("two.", 0.4, 1.0),
    word("Three", 3.0, 3.4), word("four,", 3.4, 4.0),
    word("five", 5.0, 5.5),
    word("six.", 6.5, 7.0),
]


# ── The levels ────────────────────────────────────────────────────────────────


SIX_STOPS = ["natural", "standard", "fast", "tight", "hard", "max"]


def test_there_are_six_stops_gentlest_first():
    assert list(PACES) == SIX_STOPS


def test_levels_get_tighter_in_order():
    levels = [PACES[n] for n in SIX_STOPS]
    for a, b in zip(levels, levels[1:]):
        assert a.gap_length > b.gap_length
        assert a.rhythm <= b.rhythm
        assert a.sentence_pause > b.sentence_pause
        assert a.clause_pause > b.clause_pause
        assert a.inner_pause >= b.inner_pause


def test_the_hardest_stop_reaches_as_far_as_clipforge_goes_and_still_leaves_a_pause():
    hardest = PACES["max"]
    assert hardest.gap_length == GAP_LENGTH_MIN and hardest.rhythm == RHYTHM_MAX
    assert (hardest.sentence_pause, hardest.clause_pause, hardest.inner_pause) == (0.1, 0.05, 0.03)


def test_the_three_earlier_names_kept_their_numbers():
    # A usual saved before there were six stops names one of these.
    assert (PACES["natural"].gap_length, PACES["standard"].gap_length, PACES["fast"].gap_length) == (1.0, 0.6, 0.4)
    assert (PACES["natural"].rhythm, PACES["standard"].rhythm, PACES["fast"].rhythm) == (2, 3, 4)


def test_a_stop_between_two_rhythms_keeps_a_spacing_between_theirs():
    kept = {p["pace"]: p["speech_kept_between_cuts"] for p in describe_paces()}
    assert kept == {"natural": 2.0, "standard": 1.2, "fast": 0.5, "tight": 0.25, "hard": 0.0, "max": 0.0}


def test_from_hard_on_the_summary_says_what_the_creator_gives_up():
    assert "Almost no pause is left between sentences" in PACES["hard"].summary
    assert "choppy" in PACES["max"].summary and "no pause" not in PACES["max"].summary.lower()
    for name in ("tight", "hard", "max"):
        summary = PACES[name].summary
        assert summary.count(".") == 1 and summary.endswith("."), "one plain sentence"


def test_every_level_leaves_more_pause_between_sentences_than_inside_one():
    for pace in PACES.values():
        assert pace.sentence_pause > pace.clause_pause > pace.inner_pause > 0


def test_a_pace_never_trims_a_pause_shorter_than_the_pause_it_leaves():
    for pace in PACES.values():
        assert pace.gap_length > pace.sentence_pause


def test_unknown_pace_names_the_choices():
    with pytest.raises(StudioError, match="natural, standard, fast, tight, hard, max"):
        get_pace("brutal")


def test_default_pace_is_standard():
    # The first live test on real footage went well at standard.
    assert get_pace(None).name == "standard"
    assert [p["pace"] for p in describe_paces()] == SIX_STOPS


# ── Measuring air ─────────────────────────────────────────────────────────────


def test_air_is_the_gaps_between_words_when_nothing_was_measured():
    assert air_spans(TALK, [], 0.7, 3.2) == [(1.0, 3.0)]


def test_air_adds_measured_silence_the_timestamps_hide():
    # "One two." runs to 1.0 on paper; the audio went silent at 0.8.
    assert air_spans(TALK, [Silence(0.8, 2.5)], 0.7, 3.2) == [(0.8, 3.0)]


def test_air_keeps_a_pause_filled_by_a_breath():
    # Only 1.2-1.5 measured silent; the rest of the gap is a breath. It is still a pause.
    assert air_spans(TALK, [Silence(1.2, 1.5)], 0.7, 3.2) == [(1.0, 3.0)]


# ── Leaving the pause ─────────────────────────────────────────────────────────


def left_after(trims, start_word, end_word, silences=()):
    quiet = air_spans(TALK, list(silences), (start_word["start"] + start_word["end"]) / 2,
                      (end_word["start"] + end_word["end"]) / 2)
    total = sum(b - a for a, b in quiet)
    removed = sum(max(0.0, min(b, t["end"]) - max(a, t["start"])) for t in trims for a, b in quiet)
    return total - removed


@pytest.mark.parametrize("name", ["natural", "standard", "fast"])
def test_a_trim_that_takes_the_whole_pause_is_shortened_to_leave_the_level_s_pause(name):
    pace = get_pace(name)
    kept, report = leave_pauses([trim(1.0, 3.0)], TALK, [], pace)
    assert report.shortened == 1 and report.dropped == 0
    assert left_after(kept, TALK[1], TALK[2]) == pytest.approx(pace.sentence_pause)
    assert kept[0]["start"] == 1.0                     # the pause comes back before the next word
    assert kept[0]["end"] == pytest.approx(3.0 - pace.sentence_pause)


def test_the_pause_left_depends_on_the_boundary():
    pace = get_pace("standard")
    kept, _ = leave_pauses([trim(1.0, 3.0), trim(4.0, 5.0), trim(5.5, 6.5)], TALK, [], pace)
    assert left_after(kept[:1], TALK[1], TALK[2]) == pytest.approx(pace.sentence_pause)
    assert left_after(kept[1:2], TALK[3], TALK[4]) == pytest.approx(pace.clause_pause)
    assert left_after(kept[2:], TALK[4], TALK[5]) == pytest.approx(pace.inner_pause)


def test_a_trim_that_already_leaves_enough_is_untouched():
    planned = [trim(1.5, 2.2)]
    kept, report = leave_pauses(planned, TALK, [], get_pace("standard"))
    assert kept == planned and report.shortened == 0


def test_a_trim_with_nothing_left_to_cut_is_dropped():
    # A 0.65s pause at natural pace (0.6s left between sentences) leaves 0.05s to trim.
    talk = [word("One.", 0.0, 1.0), word("Two.", 1.65, 2.0)]
    kept, report = leave_pauses([trim(1.0, 1.65)], talk, [], get_pace("natural"))
    assert kept == [] and report.dropped == 1


def test_measured_silence_counts_the_quiet_the_timestamps_hide():
    # The words butt together on paper; the audio has 1.0s of silence across the join.
    talk = [word("One.", 0.0, 1.5), word("Two.", 1.5, 3.0)]
    silences = [Silence(1.0, 2.0)]
    kept, report = leave_pauses([trim(1.0, 2.0)], talk, silences, get_pace("standard"))
    assert report.shortened == 1
    assert kept[0]["end"] - kept[0]["start"] == pytest.approx(1.0 - get_pace("standard").sentence_pause)


def test_a_trim_too_short_to_hear_is_dropped():
    # On the real take at Max: 68 ms between the soft end of "out." and the soft start of "It".
    talk = [word("out.", 49.9, 50.463), word("It", 50.531, 50.8), word("goes", 50.8, 51.2)]
    kept, report = leave_pauses([trim(50.463, 50.531)], talk, [], PACES["max"])
    assert (kept, report.dropped) == ([], 1)
    held = [(50.47, 50.52)]
    kept, report = leave_pauses([trim(50.463, 50.531)], talk, [], PACES["max"], held)
    assert len(kept) == 1 and kept[0]["start"] <= 50.47 and 50.52 <= kept[0]["end"], "a trim holding a word she cut stays"


def test_a_filler_the_trim_takes_stays_taken():
    # "um" sits in the pause. Quiet comes back around it, never the word itself.
    talk = [word("One.", 0.0, 1.0), word("um", 1.4, 1.7), word("Two.", 2.0, 3.0)]
    kept, _ = leave_pauses([trim(1.0, 2.0)], talk, [], get_pace("standard"))
    assert kept[0]["start"] <= 1.4 and kept[0]["end"] >= 1.7


def test_a_trim_at_the_very_start_is_left_as_planned():
    planned = [trim(0.0, 0.0 + 0.3)]
    talk = [word("Hello.", 0.5, 1.0)]
    assert leave_pauses(planned, talk, [], get_pace("natural"))[0] == planned


def test_settling_touches_only_automatic_trims():
    cuts = [
        {"start": 1.0, "end": 3.0, "reason": "auto: pause", "source": "auto"},
        {"start": 4.0, "end": 5.0, "reason": "tangent", "source": "claude"},
    ]
    settled, report = settle_trims(cuts, TALK, [], get_pace("standard"))
    assert report.shortened == 1
    assert settled[1] == cuts[1]
    assert settled[0]["end"] == pytest.approx(2.6) and settled[0]["source"] == "auto"


# ── Through the tools ─────────────────────────────────────────────────────────


def test_set_edit_reports_the_pace_it_used(video):
    result = tools.set_edit(str(video), [], auto_tighten=True, pace="standard", **QUIET)
    auto = result["auto"]
    assert auto["pace"] == "standard" and auto["trims_pauses_longer_than"] == 0.6
    assert tools.get_edit(str(video))["pace"] == "standard"


def test_auto_tighten_without_a_pace_is_standard_when_no_usual_is_saved(video):
    result = tools.set_edit(str(video), [], auto_tighten=True, **QUIET)
    assert result["auto"]["pace"] == "standard"


def test_a_faster_pace_removes_at_least_as_much(video):
    saved = [
        tools.set_edit(str(video), [], auto_tighten=True, pace=name, **QUIET)["removed_seconds"]
        for name in SIX_STOPS
    ]
    assert saved == sorted(saved) and saved[-1] > saved[0]


def test_gap_length_overrides_only_the_shortest_pause(video):
    result = tools.set_edit(str(video), [], auto_tighten=True, pace="natural", gap_length=0.4, **QUIET)
    assert result["auto"]["pace"] == "natural" and result["auto"]["trims_pauses_longer_than"] == 0.4


def test_pace_needs_auto_tighten_and_a_real_name(video):
    with pytest.raises(StudioError, match="only applies with auto_tighten"):
        tools.set_edit(str(video), [], pace="fast", **QUIET)
    with pytest.raises(StudioError, match="is not a level"):
        tools.set_edit(str(video), [], auto_tighten=True, pace="brutal", **QUIET)


def test_analyze_take_says_what_each_pace_would_save(video):
    paces = tools.analyze_take(str(video), silences=no_silences)["paces"]
    assert [p["pace"] for p in paces] == SIX_STOPS
    assert all({"trims", "seconds_saved", "pause_left", "summary"} <= set(p) for p in paces)
    assert paces[0]["seconds_saved"] <= paces[-1]["seconds_saved"]


# ── a word cut by hand beside a pause ─────────────────────────────────────────


def by_hand(start: float, end: float, trims: int = 1) -> dict:
    return {"start": start, "end": end, "reason": "Cut by you", "source": "you", "auto_trims": trims}


def test_a_word_cut_by_hand_beside_a_trimmed_pause_still_leaves_the_pause():
    # "five" (5.0-5.5) cut by hand, joined by the trim of the pause after it (5.5-6.5).
    # The pause before the word is still there (4.0-5.0), more than the pace leaves: nothing to give back.
    cut = by_hand(5.0, 6.5)
    settled, report = settle_trims([cut], TALK, [], PACES["standard"], held=[(5.0, 5.5)])
    assert settled == [cut] and report.shortened == 0
    # With the pause before it trimmed too, the join would have none. "four," ends a clause.
    (tighter,), report = settle_trims([by_hand(4.0, 6.5, trims=2)], TALK, [], PACES["standard"], held=[(5.0, 5.5)])
    assert tighter["start"] == 4.0 and tighter["end"] == pytest.approx(6.5 - PACES["standard"].clause_pause)
    assert tighter["source"] == "you" and report.shortened == 1


def test_her_words_stay_cut_however_much_pause_the_pace_wants():
    (settled,), report = settle_trims([by_hand(5.0, 5.6)], TALK, [], PACES["natural"], held=[(5.0, 5.5)])
    assert settled["start"] == 5.0 and settled["end"] >= 5.5 and report.dropped == 0


def test_a_cut_by_hand_that_no_trim_joined_is_left_as_she_made_it():
    cut = {"start": 5.0, "end": 5.5, "reason": "Cut by you", "source": "you"}
    settled, report = settle_trims([cut], TALK, [], PACES["natural"], held=[(5.0, 5.5)])
    assert settled == [cut] and report.shortened == 0


# ── plans kept with the project ───────────────────────────────────────────────


def test_a_plan_saved_with_the_project_is_read_and_not_made_again(tmp_path, monkeypatch):
    from lumr_studio import autocuts

    talk = TALK + [word("seven", 9.0, 9.4), word("eight.", 9.4, 10.0)]
    first = autocuts.plan_auto_cuts(talk, duration=12.0, silences=[], gap_length=0.6, rhythm=3, saved_in=tmp_path)
    assert first and (tmp_path / autocuts.PLANS_FILE).exists()
    autocuts.forget_plans()

    def never(*_a, **_k):
        raise AssertionError("the plan was saved; planning again is wasted work")

    monkeypatch.setattr(autocuts, "plan_microcuts", never)
    assert autocuts.plan_auto_cuts(talk, duration=12.0, silences=[], gap_length=0.6, rhythm=3, saved_in=tmp_path) == first
    autocuts.forget_plans()
    # Another pace, or other words, is another plan.
    with pytest.raises(AssertionError, match="wasted work"):
        autocuts.plan_auto_cuts(talk, duration=12.0, silences=[], gap_length=0.4, rhythm=4, saved_in=tmp_path)
    with pytest.raises(AssertionError, match="wasted work"):
        autocuts.plan_auto_cuts(talk[:-1], duration=12.0, silences=[], gap_length=0.6, rhythm=3, saved_in=tmp_path)


def test_a_plan_file_that_cannot_be_read_is_planned_again(tmp_path):
    from lumr_studio import autocuts

    (tmp_path / autocuts.PLANS_FILE).write_text("not json")
    planned = autocuts.plan_auto_cuts(TALK, duration=8.0, silences=[], gap_length=0.6, rhythm=3, saved_in=tmp_path)
    autocuts.forget_plans()
    assert planned and planned == autocuts.plan_auto_cuts(TALK, duration=8.0, silences=[], gap_length=0.6, rhythm=3)


# ── a trim is filed by what it takes ──────────────────────────────────────────


def test_a_filler_trim_that_takes_no_word_is_filed_as_a_pause():
    from lumr_studio.autocuts import file_by_what_it_takes

    talk = [w if w["word"] != "five" else word("so", 5.0, 5.5) for w in TALK]
    talk.append({"word": "[vocalization]", "start": 7.2, "end": 7.6, "type": "event"})
    trims = [
        {"start": 4.95, "end": 5.55, "reason": "filler: so"},            # takes "so" (5.0 to 5.5)
        {"start": 5.55, "end": 6.45, "reason": "filler: so (+2 more)"},  # landed on the pause after it
        {"start": 1.1, "end": 2.9, "reason": "pause: 2.0s"},
        {"start": 7.1, "end": 7.7, "reason": "stutter: six six"},          # holds a sound, and no word
    ]
    file_by_what_it_takes(trims, talk)
    assert [t["kind"] for t in trims] == ["fillers", "pauses", "pauses", "pauses"]


def test_a_filler_trim_that_takes_no_filler_is_filed_as_a_pause():
    # On the real take ClipForge named a trim for "just", landed it on the pause after
    # "just", and that pause held "take", which the aligner had given a length of 1 ms.
    from lumr_studio.autocuts import file_by_what_it_takes

    talk = [word("You", 0.0, 0.2), word("can", 0.2, 0.4), word("just", 0.4, 0.6),
            word("take", 1.041, 1.042), word("one", 1.5, 1.7), word("step.", 1.7, 2.0),
            word("Well,", 3.0, 3.2), word("maybe", 3.25, 3.5), word("so.", 4.0, 4.3)]
    trims = [
        {"start": 0.61, "end": 1.47, "reason": "filler: just (+3 more)"},  # takes "take" and no filler
        {"start": 2.95, "end": 3.55, "reason": "filler: well (+1 more)"},  # takes a filler and one more word
        {"start": 1.45, "end": 1.75, "reason": "stutter: one one"},        # a stutter is not judged by the filler lists
    ]
    file_by_what_it_takes(trims, talk)
    assert [t["kind"] for t in trims] == ["pauses", "fillers", "repeats"]


def test_only_words_on_a_filler_list_are_named_as_fillers():
    from lumr_studio.autocuts import FILLER_PAIRS, FILLER_WORDS, fillers_in

    said = [word(text, i, i + 0.5) for i, text in enumerate(["You", "know,", "take", "So,", "sort", "of", "know", "um."])]
    assert fillers_in(said) == ["you know", "so", "sort of", "um"], "a pair reads as one, and its halves alone are no filler"
    assert fillers_in([word("take", 0, 1), word("maybe", 1, 2)]) == []
    assert fillers_in([]) == []
    assert {"and", "so", "um", "just"} <= FILLER_WORDS and "take" not in FILLER_WORDS
    assert {"you know", "sort of"} <= FILLER_PAIRS


def test_a_word_judged_by_reading_is_no_filler_to_the_pace():
    from lumr_studio.engine.longform import EDIT_LEVELS

    from lumr_studio.autocuts import FILLER_WORDS, JUDGED_BY_READING, engine_level, file_by_what_it_takes, filler_lists, fillers_in

    assert JUDGED_BY_READING == {"like"}
    assert any("like" in level.filler_words for level in EDIT_LEVELS.values()), "ClipForge lists it, and plans removals for it"
    assert "like" not in FILLER_WORDS
    assert fillers_in([word("Like,", 0, 0.5), word("um", 1, 1.5)]) == ["um"]
    for gap_length in (NATURAL, STANDARD, FAST, GAP_LENGTH_MIN):
        singles, _pairs = filler_lists(engine_level(gap_length))
        assert "like" not in singles
        assert fillers_in([word("like", 0, 0.5)], engine_level(gap_length)) == []
    # A removal ClipForge named for the word is a pause, whatever it lands on.
    trims = [{"start": 1.15, "end": 1.5, "reason": "filler: like"}]
    file_by_what_it_takes(trims, AND_LIKE, engine_level(FAST))
    assert trims[0]["kind"] == "pauses"


# ── what a trim may take ──────────────────────────────────────────────────────

EVERY_KIND = ("pauses", "fillers", "repeats")
# The shortest pause cut at three paces, and the ClipForge level each reads its fillers from.
NATURAL, STANDARD, FAST = 1.0, 0.6, 0.4


def taken(trims, talk, *, gap_length=FAST, on=EVERY_KIND):
    from lumr_studio.autocuts import trims_taken

    left = trims_taken([dict(t) for t in trims], talk, gap_length=gap_length, on=frozenset(on))
    return [(t["start"], t["end"], t["kind"]) for t in left]


def pause(start, end):
    return {"start": start, "end": end, "reason": f"pause: {end - start:.1f}s", "kind": "pauses"}


JUST_TAKE = [word("You", 0.0, 0.2), word("can", 0.2, 0.4), word("just", 0.4, 0.6),
             word("take", 1.0, 1.25), word("one", 1.7, 1.9), word("step.", 1.9, 2.2)]
AND_LIKE = [word("works.", 0.0, 0.5), word("like,", 1.2, 1.45), word("it", 2.2, 2.4), word("helps.", 2.4, 2.9)]
AND_UM = [word("works.", 0.0, 0.5), word("um,", 1.2, 1.45), word("it", 2.2, 2.4), word("helps.", 2.4, 2.9)]


@pytest.mark.parametrize("gap_length", [NATURAL, STANDARD, FAST, GAP_LENGTH_MIN])
def test_a_trim_never_takes_a_word_that_no_switch_owns(gap_length):
    # The pause after "just" holds "take". The trim is shortened around the word at every pace.
    assert taken([pause(0.61, 1.69)], JUST_TAKE, gap_length=gap_length) == [
        (0.61, 1.0, "pauses"), (1.25, 1.69, "pauses"),
    ]


def test_what_is_left_of_a_trim_beside_a_word_goes_only_when_it_is_worth_cutting():
    from lumr_studio.autocuts import SHORTEST_PIECE_SECONDS

    assert SHORTEST_PIECE_SECONDS == 0.1
    assert taken([pause(0.95, 1.69)], JUST_TAKE) == [(1.25, 1.69, "pauses")], "0.05 s before the word is not worth a cut"
    assert taken([pause(0.95, 1.3)], JUST_TAKE) == [], "and with 0.05 s either side nothing is cut"


def test_a_trim_that_holds_no_word_is_left_as_planned():
    trims = [pause(0.61, 0.99), pause(1.26, 1.69)]
    assert taken(trims, JUST_TAKE) == [(0.61, 0.99, "pauses"), (1.26, 1.69, "pauses")]


def test_a_filler_inside_a_pause_belongs_to_the_filler_switch():
    whole = [pause(0.55, 2.15)]
    assert taken(whole, AND_UM) == [(0.55, 2.15, "pauses")], "on, the pause takes the filler with it"
    assert taken(whole, AND_UM, on=("pauses", "repeats")) == [
        (0.55, 1.2, "pauses"), (1.45, 2.15, "pauses"),
    ], "off, the word stays and the pause either side of it still goes"


def test_with_long_pauses_off_a_pause_goes_nowhere_and_takes_no_filler_with_it():
    assert taken([pause(0.55, 2.15)], AND_UM, on=("fillers", "repeats")) == []


@pytest.mark.parametrize("gap_length", [NATURAL, STANDARD, FAST, GAP_LENGTH_MIN])
@pytest.mark.parametrize("on", [EVERY_KIND, ("pauses", "repeats"), ("pauses",)])
def test_a_pause_never_takes_a_like_with_it(gap_length, on):
    assert taken([pause(0.55, 2.15)], AND_LIKE, gap_length=gap_length, on=on) == [
        (0.55, 1.2, "pauses"), (1.45, 2.15, "pauses"),
    ], "the word stays at every pace, and the pause either side of it still goes"


def test_a_removal_clipforge_named_for_a_like_takes_the_pause_beside_it_and_leaves_the_word():
    named = [{"start": 1.15, "end": 2.15, "reason": "filler: like", "kind": "fillers"}]
    assert taken(named, AND_LIKE) == [(1.45, 2.15, "pauses")]
    assert taken(named, AND_LIKE, on=("fillers", "repeats")) == [], "and that pause waits for the pause switch"
    word_only = [{"start": 1.2, "end": 1.45, "reason": "filler: like", "kind": "fillers"}]
    assert taken(word_only, AND_LIKE) == []


def test_a_like_said_twice_is_still_judged_by_reading():
    # On the real take one of the two pairs is the end of a sentence and the start of the next.
    talk = [word("felt", 0.0, 0.4), word("like.", 1.0, 1.2), word("Like,", 1.25, 1.45), word("begin", 1.5, 2.0)]
    assert taken([pause(0.45, 1.22)], talk, gap_length=FAST) == [(0.45, 1.0, "pauses")]
    stutter = [{"start": 0.95, "end": 1.22, "reason": "stutter: like like", "kind": "repeats"}]
    assert taken(stutter, talk, gap_length=FAST) == []
    assert taken(stutter, talk, gap_length=GAP_LENGTH_MIN) == []


def test_a_word_is_a_filler_by_the_list_of_the_pace():
    so = [word("works.", 0.0, 0.5), word("So", 1.2, 1.45), word("it", 2.2, 2.4)]
    # ClipForge counts "so" as a filler from its fourth level on, which Fast reads. Standard reads the third.
    assert taken([pause(0.55, 2.15)], so, gap_length=FAST) == [(0.55, 2.15, "pauses")]
    assert taken([pause(0.55, 2.15)], so, gap_length=STANDARD) == [(0.55, 1.2, "pauses"), (1.45, 2.15, "pauses")]
    # "um" is on the third level's list and on none at Natural, where no word goes at all.
    assert taken([pause(0.55, 2.15)], AND_UM, gap_length=STANDARD) == [(0.55, 2.15, "pauses")]
    assert taken([pause(0.55, 2.15)], AND_UM, gap_length=NATURAL) == [(0.55, 1.2, "pauses"), (1.45, 2.15, "pauses")]


def test_a_two_word_filler_goes_whole_or_stays_whole():
    talk = [word("works.", 0.0, 0.5), word("you", 1.2, 1.4), word("know,", 1.4, 1.7), word("it", 2.4, 2.6)]
    assert taken([pause(0.55, 2.35)], talk, gap_length=FAST) == [(0.55, 2.35, "pauses")]
    assert taken([pause(0.55, 2.35)], talk, gap_length=FAST, on=("pauses",)) == [(0.55, 1.2, "pauses"), (1.7, 2.35, "pauses")]
    # "you" alone is no filler: a trim that holds it and not "know," is shortened around it.
    assert taken([pause(0.55, 1.41)], talk, gap_length=FAST) == [(0.55, 1.2, "pauses")]


def test_a_filler_trim_goes_with_its_own_switch_as_before():
    named = [{"start": 1.15, "end": 2.15, "reason": "filler: um", "kind": "fillers"}]
    assert taken(named, AND_UM) == [(1.15, 2.15, "fillers")]
    assert taken(named, AND_UM, on=("fillers",)) == [(1.15, 2.15, "fillers")], "with the pause beside the word, as ClipForge planned it"
    assert taken(named, AND_UM, on=("pauses", "repeats")) == []


def test_what_is_left_of_a_filler_trim_once_its_word_stays_in_is_a_pause():
    talk = [word("works.", 0.0, 0.5), word("um,", 1.0, 1.25), word("maybe", 1.6, 1.9), word("it", 2.6, 2.8)]
    named = [{"start": 0.95, "end": 2.55, "reason": "filler: um (+1 more)", "kind": "fillers"}]
    assert taken(named, talk) == [(0.95, 1.6, "fillers"), (1.9, 2.55, "pauses")], "\"maybe\" is nobody's to take"
    assert taken(named, talk, on=("fillers",)) == [(0.95, 1.6, "fillers")], "and its pause waits for the pause switch"


def test_the_first_of_a_word_said_twice_belongs_to_the_stutter_switch():
    talk = [word("see", 0.0, 0.4), word("the", 1.0, 1.2), word("the", 1.25, 1.45), word("point.", 1.5, 2.0)]
    assert taken([pause(0.45, 1.22)], talk, gap_length=FAST) == [(0.45, 1.22, "pauses")]
    assert taken([pause(0.45, 1.22)], talk, gap_length=FAST, on=("pauses", "fillers")) == [(0.45, 1.0, "pauses")]
    # ClipForge cuts words said twice from its fourth level on. At Standard the word is nobody's.
    assert taken([pause(0.45, 1.22)], talk, gap_length=STANDARD) == [(0.45, 1.0, "pauses")]
    stutter = [{"start": 0.95, "end": 1.22, "reason": "stutter: the the", "kind": "repeats"}]
    assert taken(stutter, talk, gap_length=FAST) == [(0.95, 1.22, "repeats")]
    assert taken(stutter, talk, gap_length=FAST, on=("pauses", "fillers")) == []


@pytest.mark.parametrize("name", SIX_STOPS)
@pytest.mark.parametrize("room", [False, True])
def test_no_plan_takes_a_word_that_no_switch_owns(name, room):
    import short_word
    from lumr_studio import autocuts, word_times

    talk = short_word.aligned()
    if room:
        talk = word_times.with_room(talk, short_word.silences())
    level = PACES[name]
    planned = autocuts.plan_auto_cuts(talk, duration=20.0, silences=short_word.silences(), gap_length=level.gap_length, rhythm=level.rhythm)
    said = autocuts.Said.of(talk)
    lists = autocuts.engine_level(level.gap_length)
    for t in planned:
        held = said.held(t["start"], t["end"])
        assert None not in said.owners(held, lists), (t, [said.words[i]["word"] for i in held])


def test_a_plan_saved_by_an_earlier_version_is_planned_again(tmp_path):
    import json

    from lumr_studio import autocuts

    assert autocuts.PLANS_VERSION == 6
    autocuts.forget_plans()
    mark = autocuts.take_mark(TALK, [], 8.0)
    stale = [{"start": 1.1, "end": 2.9, "reason": "pause: 2.0s", "kind": "pauses"}, pause(0.05, 0.95)]
    (tmp_path / autocuts.PLANS_FILE).write_text(json.dumps({
        "version": 4, "take": mark, "plans": {"0.6/3.0": {"stop": True, "trims": stale}},
    }))
    planned = autocuts.plan_auto_cuts(TALK, duration=8.0, silences=[], gap_length=0.6, rhythm=3, saved_in=tmp_path)
    assert planned and planned != stale, "the saved plan held a trim over \"One two.\""
    saved = json.loads((tmp_path / autocuts.PLANS_FILE).read_text())
    assert saved["version"] == 6 and saved["plans"]["0.6/3.0"]["trims"] == planned
