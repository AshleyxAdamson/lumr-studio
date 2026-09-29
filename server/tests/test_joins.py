"""The join self-check, on synthetic transcripts built word by word."""

import re

import pytest

from lumr_studio.errors import StudioError
from lumr_studio.joins import (
    FLAGS,
    MAX_LISTED,
    check_joins,
    join_rows,
)

STEP = 0.4  # every synthetic word lasts this long, back to back like the real transcript


class Talk:
    """Builds a transcript: words back to back, pauses as gaps, events as vocalizations."""

    def __init__(self):
        self.words: list[dict] = []
        self.t = 0.0

    def say(self, text: str) -> "Talk":
        for token in text.split():
            self.words.append({"word": token, "start": round(self.t, 3), "end": round(self.t + STEP, 3)})
            self.t += STEP
        return self

    def pause(self, seconds: float) -> "Talk":
        self.t += seconds
        return self

    def sound(self, seconds: float) -> "Talk":
        self.words.append({"word": "[vocalization]", "type": "event",
                           "start": round(self.t, 3), "end": round(self.t + seconds, 3)})
        self.t += seconds
        return self

    def at(self, word: str, nth: int = 0) -> dict:
        """The nth transcript entry whose text is ``word``."""
        return [w for w in self.words if w["word"] == word][nth]


def cut(start, end, source="claude", reason="test"):
    return {"start": start, "end": end, "reason": reason, "source": source}


def span(talk, first, last, first_nth=0, last_nth=0, source="claude"):
    """A cut from the start of word ``first`` to the end of word ``last``."""
    return cut(talk.at(first, first_nth)["start"], talk.at(last, last_nth)["end"], source)


def rows_for(talk, cuts, labels=None, duration=None):
    return join_rows(cuts, talk.words, duration or talk.t + 1.0, labels=labels)


def only_row(talk, cuts, labels=None):
    rows = rows_for(talk, cuts, labels)
    assert len(rows) == len(cuts)
    return rows[0]


def story() -> Talk:
    return (Talk().say("I went to the store.").pause(0.6)
            .say("Then I bought some milk.").pause(0.6)
            .say("We walked home together."))


# ── mid_sentence_out ──────────────────────────────────────────────────────────


def test_cut_that_leaves_a_sentence_unfinished_is_mid_sentence_out():
    talk = story()
    row = only_row(talk, [span(talk, "bought", "milk.")])
    assert row["flags"] == ["mid_sentence_out"]
    then = talk.at("Then")["start"]
    assert f"start at {then:.2f}" in row["note"].lower()  # drop the unfinished sentence


def test_a_flag_has_a_fix_for_claude_and_a_plain_why_for_the_creator():
    talk = story()
    row = only_row(talk, [span(talk, "bought", "milk.")])
    # Claude can move the cut's start, so the note names a source time to pass back.
    assert row["note"].startswith("Sentence left unfinished. Start at ")
    # The creator can only put the cut back or leave it out: no times, no fix.
    assert row["why"] == "The sentence before this cut never finishes."


def test_cut_of_whole_sentences_is_clean():
    talk = story()
    row = only_row(talk, [span(talk, "Then", "milk.")])
    assert row["flags"] == []
    assert row["note"] == "" and row["why"] == ""


# ── mid_sentence_in ───────────────────────────────────────────────────────────


def test_cut_that_opens_mid_sentence_is_mid_sentence_in():
    talk = story()
    row = only_row(talk, [span(talk, "Then", "walked")])
    assert row["flags"] == ["mid_sentence_in"]
    # Either skip past "home together." (nothing follows, so no such option),
    # or end at "We" to keep that sentence whole.
    assert f"at {talk.at('We')['start']:.2f} to keep it whole" in row["note"].lower()


def test_mid_sentence_in_offers_the_next_sentence_start():
    talk = story()
    row = only_row(talk, [span(talk, "Then", "I", last_nth=1)])
    assert row["flags"] == ["mid_sentence_in"]
    assert f"end at {talk.at('We')['start']:.2f}" in row["note"].lower()


def test_dropping_only_a_lead_in_so_is_not_mid_sentence_in():
    talk = Talk().say("That was the end.").pause(0.6).say("So we walked home.")
    row = only_row(talk, [span(talk, "So", "So")])
    assert "mid_sentence_in" not in row["flags"]


# ── splice ────────────────────────────────────────────────────────────────────


def test_cut_inside_one_sentence_is_a_splice_with_a_clause_edge_fix():
    talk = Talk().say("After the storm, I can't tell if you are here, but I have fixed it.")
    row = only_row(talk, [span(talk, "you", "here,")])
    assert row["flags"] == ["splice"]
    # Move the start back to just after "storm," so the join reads "storm, | but".
    assert f"start at {talk.at('I')['start']:.2f}" in row["note"].lower()


def test_retake_splice_is_listed_and_called_a_retake():
    talk = Talk().say("We said I went to the, I went to the store.")
    row = only_row(talk, [span(talk, "I", "the,")])
    assert row["flags"] == ["splice"]
    assert row["note"].startswith("Retake fix")


def test_retake_at_the_start_of_a_sentence_is_clean():
    talk = Talk().say("I went to the, I went to the store.")
    assert only_row(talk, [span(talk, "I", "the,")])["flags"] == []


def test_claude_cut_of_only_filler_words_is_clean():
    talk = Talk().say("So um we went home.")
    assert only_row(talk, [span(talk, "um", "um")])["flags"] == []


def test_retake_cut_that_also_takes_a_word_before_the_repeat_says_where_to_start():
    talk = Talk().say("a wall of books that I was trying to, that was trying to come down.")
    # Starts one word too early: "books" goes as well as the false start.
    row = only_row(talk, [span(talk, "books", "that", last_nth=1)])
    assert row["flags"] == ["splice"]
    assert '"books' in row["note"]
    assert f"start at {talk.at('was')['start']:.2f}" in row["note"].lower()


# ── re_entry ──────────────────────────────────────────────────────────────────


def _long_aside(opener: str) -> Talk:
    return (Talk().say("Let us begin.").pause(0.5)
            .say("There is so much to say about this long journey of ours right now.").pause(0.5)
            .say(f"{opener} goes on forever."))


def test_back_pointing_opener_after_a_long_cut_is_re_entry():
    talk = _long_aside("It")
    row = only_row(talk, [span(talk, "There", "now.")])
    assert row["seconds"] >= 5
    assert row["flags"] == ["re_entry"]
    assert "drop the cut" in row["note"].lower()


def test_back_pointing_opener_after_a_short_cut_is_not_re_entry():
    talk = Talk().say("Let us begin.").pause(0.3).say("Right now.").pause(0.3).say("It goes on forever.")
    row = only_row(talk, [span(talk, "Right", "now.")])
    assert row["flags"] == []


def test_so_with_a_comma_announces_a_new_part_and_is_not_re_entry():
    talk = _long_aside("So, it")
    row = only_row(talk, [span(talk, "There", "now.")])
    assert "re_entry" not in row["flags"]


def test_an_answer_word_is_re_entry_even_after_a_short_cut():
    talk = Talk().say("That was fun.").pause(0.3).say("Go out there.").pause(0.3).say("Yes, it is risky.")
    row = only_row(talk, [span(talk, "Go", "there.")])
    assert row["seconds"] < 5
    assert row["flags"] == ["re_entry"]


# ── long_jump ─────────────────────────────────────────────────────────────────


def _speech(n_words: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n_words - 1)) + f" {prefix}end."


def test_twenty_seconds_removed_is_a_long_jump():
    talk = Talk().say("Start here.").say(_speech(48)).say("Last aside here.").say("We carry on.")  # 51 words = 20.4s
    row = only_row(talk, [span(talk, "w0", "here.", last_nth=1)])
    assert row["seconds"] >= 20
    assert row["flags"] == ["long_jump"]
    assert f"end at {talk.at('Last')['start']:.2f} to keep the last cut sentence" in row["note"].lower()


def test_nineteen_seconds_removed_is_not_a_long_jump():
    talk = Talk().say("Start here.").say(_speech(47)).say("We carry on.")  # 18.8s
    row = only_row(talk, [span(talk, "w0", "wend.")])
    assert row["flags"] == []


# ── fragment ──────────────────────────────────────────────────────────────────


def test_two_words_kept_between_two_cuts_is_a_fragment():
    talk = (Talk().say("I went out.").pause(0.6).say("Then I bought some milk.").pause(0.6)
            .say("Okay fine.").pause(0.6).say("We walked home together.").pause(0.6).say("The end."))
    cuts = [span(talk, "Then", "milk."), span(talk, "We", "together.")]
    row = rows_for(talk, cuts)[0]
    assert row["flags"] == ["fragment"]
    assert f"end at {talk.at('together.')['end']:.2f}" in row["note"].lower()


def test_a_whole_sentence_kept_between_two_cuts_is_not_a_fragment():
    talk = story()
    cuts = [span(talk, "I", "store."), span(talk, "We", "together.")]
    assert "fragment" not in rows_for(talk, cuts)[0]["flags"]


def test_words_beside_a_pause_trim_are_not_a_fragment():
    talk = Talk().say("One two three.").pause(0.6).say("Four five.").pause(0.8).say("Six seven eight.")
    cuts = [span(talk, "One", "three."),
            cut(talk.at("five.")["end"] + 0.1, talk.at("Six")["start"] - 0.1, source="auto")]
    assert "fragment" not in rows_for(talk, cuts)[0]["flags"]


# ── tight ─────────────────────────────────────────────────────────────────────


def _paused() -> Talk:
    return (Talk().say("One two three.").pause(0.5)
            .say("Four five.").pause(0.4)
            .say("Six seven.")
            .say("Eight nine."))


def test_cut_that_removes_a_real_pause_and_leaves_no_silence_is_tight():
    talk = _paused()
    c = cut(talk.at("three.")["end"], talk.at("Eight")["start"])
    row = only_row(talk, [c])
    assert row["flags"] == ["tight"]
    assert f"start at {talk.at('five.')['end']:.2f}" in row["note"].lower()


def test_joining_words_that_already_touched_is_not_tight():
    talk = Talk().say("One two three four five.")
    row = only_row(talk, [span(talk, "two", "two")])
    assert "tight" not in row["flags"]


def test_cut_that_leaves_some_of_the_pause_is_not_tight():
    talk = _paused()
    c = cut(talk.at("three.")["end"] + 0.2, talk.at("Eight")["start"])
    assert "tight" not in only_row(talk, [c])["flags"]


# ── Laughs: removes_laugh and clips_beat ──────────────────────────────────────


def _joke() -> tuple[Talk, list[dict]]:
    talk = (Talk().say("Let me tell you something.").pause(0.4)
            .say("I met a chef.").pause(0.3)
            .say("He was not chef material.").pause(0.3)
            .sound(2.0).pause(0.6)
            .say("Anyway we moved on.").pause(0.6)
            .say("Later that year I moved out.").pause(0.6)
            .say("Life went on."))
    laugh = talk.words[[w.get("type") for w in talk.words].index("event")]
    punch = (talk.at("He")["start"], talk.at("material.")["end"])
    labels = [{"start": laugh["start"], "end": laugh["end"], "seconds": 2.0, "kind": "laugh",
               "confidence": "likely", "punchline": list(punch)}]
    return talk, labels


def test_cut_that_removes_the_laugh_is_removes_laugh():
    talk, labels = _joke()
    laugh = labels[0]
    row = only_row(talk, [cut(laugh["start"] - 0.1, laugh["end"] + 0.1)], labels)
    assert row["flags"] == ["removes_laugh"]
    assert f"{laugh['start']:.2f}" in row["note"]


def test_cut_that_removes_the_punchline_is_removes_laugh():
    talk, labels = _joke()
    row = only_row(talk, [span(talk, "He", "material.")], labels)
    assert row["flags"][0] == "removes_laugh"
    assert "punchline" in row["note"]


def test_cut_that_removes_the_setup_right_before_the_punchline_is_removes_laugh():
    talk, labels = _joke()
    row = only_row(talk, [span(talk, "I", "chef.")], labels)
    assert row["flags"] == ["removes_laugh"]
    assert "setup" in row["note"]


def test_cut_well_away_from_the_joke_is_not_removes_laugh():
    talk, labels = _joke()
    row = only_row(talk, [span(talk, "Later", "out.")], labels)
    assert row["flags"] == []


def test_laugh_flags_need_labels():
    talk, _ = _joke()
    laugh = [w for w in talk.words if w.get("type") == "event"][0]
    row = only_row(talk, [cut(laugh["start"] - 0.1, laugh["end"] + 0.1)])
    assert row["flags"] == []
    assert "No sound labels" in check_joins([], talk.words, talk.t).get("text")


def test_automatic_trim_in_the_pause_after_a_laugh_clips_the_beat():
    talk, labels = _joke()
    laugh = labels[0]
    trim = cut(laugh["end"] + 0.1, talk.at("Anyway")["start"] - 0.1, source="auto")
    row = only_row(talk, [trim], labels)
    assert row["flags"] == ["clips_beat"]
    assert "ask the creator to keep" in row["note"].lower()
    assert row["why"] == "An automatic cut shortens the pause around a laugh."


def test_automatic_trim_in_a_pause_away_from_the_joke_is_clean():
    talk, labels = _joke()
    trim = cut(talk.at("on.")["end"] + 0.1, talk.at("Later")["start"] - 0.1, source="auto")
    assert only_row(talk, [trim], labels)["flags"] == []


def test_same_kind_of_claude_cut_in_that_pause_is_not_clips_beat():
    talk, labels = _joke()
    laugh = labels[0]
    c = cut(laugh["end"] + 0.1, talk.at("Anyway")["start"] - 0.1)
    assert "clips_beat" not in only_row(talk, [c], labels)["flags"]


# ── Edges of the video, pauses, automatic trims ───────────────────────────────


def test_cut_at_the_very_start_has_one_side_only():
    talk = story()
    rows = rows_for(talk, [cut(0.0, talk.at("store.")["end"])])
    assert rows[0]["flags"] == []
    assert rows[0]["before"] == ""
    text = check_joins([cut(0.0, talk.at("to")["end"])], talk.words, talk.t + 1)["text"]
    assert "(no words before) | the store." in text
    assert "[mid_sentence_in]" in text


def test_cut_at_the_very_end_has_one_side_only():
    talk = story()
    end = talk.t + 1.0
    row = rows_for(talk, [cut(talk.at("home")["start"], end)], duration=end)[0]
    assert row["flags"] == ["mid_sentence_out"]
    assert row["after"] == ""
    text = check_joins([cut(talk.at("home")["start"], end)], talk.words, end)["text"]
    assert "| (no words after)" in text


def test_cut_wholly_inside_a_pause_gets_no_sentence_or_jump_flags():
    talk = Talk().say("There was a pause.").pause(25.0).say("It was long.")
    c = cut(talk.at("pause.")["end"] + 0.5, talk.at("It")["start"] - 0.5)
    row = only_row(talk, [c])
    assert row["removed_words"] == 0
    assert row["flags"] == []


def test_automatic_filler_trim_mid_sentence_is_clean():
    talk = Talk().say("I was um going home.")
    row = only_row(talk, [span(talk, "um", "um", source="auto")])
    assert row["removed_words"] == 1
    assert row["flags"] == []


def test_automatic_pause_trims_are_not_judged_from_word_gaps():
    # Word timestamps butt words together, so a gap read from them says nothing
    # about the pause a trim leaves. pace.py measures that from the audio.
    talk = Talk().say("One two.").pause(0.8).say("Three four.")
    trim = cut(talk.at("two.")["end"], talk.at("Three")["start"], source="auto")
    out = check_joins([trim], talk.words, talk.t + 1)
    assert out["flagged"] == 0
    assert "automatic trims leave" not in out["text"]


def test_zero_length_words_do_not_make_a_trim_remove_speech():
    talk = Talk().say("First part.").pause(0.3).say("So,")
    t = talk.t
    talk.words += [{"word": "while", "start": t, "end": t}, {"word": "we", "start": t, "end": t}]
    talk.pause(0.2).say("cook, it matters.")
    cuts = [span(talk, "First", "part."), cut(t, t + 0.2, source="auto")]
    assert "fragment" not in rows_for(talk, cuts)[0]["flags"]


# ── check_joins: summary, order, text ─────────────────────────────────────────


def test_empty_cut_list():
    talk = story()
    out = check_joins([], talk.words, talk.t)
    assert out["joins"] == out["flagged"] == out["clean"] == out["not_listed"] == 0
    assert out["rows"] == [] and out["by_flag"] == {}
    assert out["text"].startswith("0 joins, 0 flagged, 0 clean.")
    assert join_rows([], talk.words, talk.t) == []


def test_rows_have_the_contract_shape():
    talk = story()
    row = only_row(talk, [span(talk, "bought", "milk.")])
    assert set(row) == {"cut", "start", "end", "seconds", "source", "reason", "removed_words",
                        "before", "removed", "after", "flags", "note", "why"}
    assert row["before"].endswith("Then I")
    assert row["removed"] == "bought some milk."
    assert row["after"].startswith("We walked")
    assert row["removed_words"] == 3


def test_join_rows_lists_every_cut_in_time_order():
    talk = story()
    cuts = [span(talk, "We", "together."), span(talk, "bought", "milk."), span(talk, "went", "went")]
    rows = rows_for(talk, cuts)
    assert [r["cut"] for r in rows] == [0, 1, 2]
    assert [r["start"] for r in rows] == sorted(r["start"] for r in rows)


def test_worst_first_and_counts():
    talk = (Talk().say("A retake, a retake and on.").pause(0.6)
            .say("I went to the store.").pause(0.6)
            .say("Then I bought milk.").pause(0.6)
            .say("We walked home."))
    cuts = [span(talk, "a", "and"),              # splice (retake)
            span(talk, "bought", "milk.")]       # mid_sentence_out
    out = check_joins(cuts, talk.words, talk.t + 1)
    assert out["joins"] == 2 and out["flagged"] == 2 and out["clean"] == 0
    assert [r["flags"][0] for r in out["rows"]] == ["mid_sentence_out", "splice"]
    assert set(out["by_flag"]) <= set(FLAGS)
    assert out["by_flag"] == {"mid_sentence_out": 1, "splice": 1}


def test_max_listed_caps_rows_and_reports_the_rest():
    talk = Talk()
    for i in range(30):
        talk.say(f"Part{i} goes here now.").pause(0.6)
    cuts = [span(talk, "goes", "now.", i, i) for i in range(30)]  # each leaves "PartN" unfinished
    out = check_joins(cuts, talk.words, talk.t + 1)
    assert out["flagged"] == 30
    assert len(out["rows"]) == MAX_LISTED and out["not_listed"] == 5
    assert out["text"].endswith("+5 more flagged, not listed.")
    few = check_joins(cuts, talk.words, talk.t + 1, max_listed=3)
    assert len(few["rows"]) == 3 and few["not_listed"] == 27


def test_text_form_rows_match_the_contract_shape():
    talk = story()
    out = check_joins([span(talk, "bought", "milk.")], talk.words, talk.t + 1)
    lines = out["text"].splitlines()
    row = lines.index(next(line for line in lines if line.startswith("#")))
    assert re.fullmatch(r"#0 \d+\.\d\d-\d+\.\d\d -\d+\.\ds claude \[mid_sentence_out\]", lines[row])
    assert lines[row + 1] == "   ...the store. Then I | We walked home together...."
    assert lines[row + 2].startswith("   fix: ")


def test_text_for_25_flagged_rows_stays_compact():
    talk = Talk()
    for i in range(30):
        talk.say(f"Part{i} of the story goes right here.").pause(0.6)
    cuts = [span(talk, "goes", "here.", i, i) for i in range(30)]
    out = check_joins(cuts, talk.words, talk.t + 1)
    assert len(out["rows"]) == 25
    # The contract aims for 3,000. Each row carries its header, both sides of
    # the join and a fix with times, about 160 characters, so 25 rows land
    # near 4,100. The real edits flag 9 to 17 rows, about 2,000 to 3,300.
    assert len(out["text"]) < 4200, len(out["text"])


# ── Bad input ─────────────────────────────────────────────────────────────────


def test_overlapping_cuts_are_refused():
    talk = story()
    with pytest.raises(StudioError, match="overlap"):
        check_joins([cut(1.0, 3.0), cut(2.0, 4.0)], talk.words, talk.t)


def test_cut_past_the_end_is_refused():
    talk = story()
    with pytest.raises(StudioError, match="past the video's end"):
        join_rows([cut(1.0, talk.t + 5)], talk.words, talk.t)


def test_negative_max_listed_is_refused():
    talk = story()
    with pytest.raises(StudioError, match="max_listed"):
        check_joins([], talk.words, talk.t, max_listed=-1)


# ── the creator's own cuts ────────────────────────────────────────────────────


def test_the_creators_cut_inside_one_sentence_says_so_in_plain_words():
    talk = story()
    hers = only_row(talk, [span(talk, "bought", "bought", source="you")])
    claudes = only_row(talk, [span(talk, "bought", "bought")])
    assert hers["flags"] == claudes["flags"] == ["splice"] and hers["source"] == "you"
    assert hers["why"].startswith("This takes words out of the middle of a sentence.")
    assert claudes["why"] == "This joins the start of one sentence to the end of another."


def test_the_creators_cut_of_a_filler_word_is_clean():
    talk = Talk().say("I went there and like it was really good.").pause(0.4).say("Then home.")
    assert only_row(talk, [span(talk, "like", "like", source="you")])["flags"] == []


def test_the_creators_cut_of_a_laugh_is_flagged_like_any_cut():
    talk, labels = _joke()
    sound = next(w for w in talk.words if w.get("type") == "event")
    row = only_row(talk, [cut(sound["start"], sound["end"], source="you")], labels)
    assert "removes_laugh" in row["flags"] and row["why"] == "This cut takes out a laugh."
