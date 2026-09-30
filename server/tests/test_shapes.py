"""What the creator can choose to send: the shape of their changes, and never a word from the video.

Every transcript here is made of distinctive words ("zebra", "Wellington", "Priya"), so a word that
leaks into a send, a preview line or a saved file can be found by name. ``urllib.request.urlopen`` is
replaced by the stand-in from ``test_feedback``: no test reaches the internet.
"""

import copy
import json
import re
import urllib.error

import pytest

from creator_words import banned_in, decimal_times_in
from lumr_studio import shapes, tools, treatment
from lumr_studio.errors import StudioError
from lumr_studio.project import open_project, write_json_atomic
from lumr_studio.silences import no_silences
from lumr_studio.tools import unmeasured_labels
from lumr_studio.treatment import Treatment
from test_feedback import TEAM, Answer, team  # noqa: F401

QUIET = {"silences": no_silences, "labels": unmeasured_labels}
DURATION = 20.0
SENTENCES = [
    "Priya flew to Wellington yesterday.",
    "Um the zebra escaped from the zoo.",
    "The zebra escaped from the zoo again.",
    "You know Wellington has 1987 seaside cafes.",
    "Thanks for watching Priya.",
    "Goodbye from Auckland.",
]


def layout():
    """Words at half-second steps, 0.6 s between sentences, and each sentence's first and last word."""
    rows, spans, t = [], [], 0.5
    for sentence in SENTENCES:
        first = len(rows)
        for token in sentence.split():
            rows.append({"word": token, "start": round(t, 2), "end": round(t + 0.4, 2), "energy_rms": 0.1})
            t += 0.5
        spans.append((first, len(rows) - 1))
        t += 0.6
    return rows, spans


ROWS, SPANS = layout()


def span_of(first, last=None):
    """The source seconds from word ``first`` to word ``last`` (a word index in ROWS)."""
    return {"start": ROWS[first]["start"], "end": ROWS[first if last is None else last]["end"]}


def sentence(n):
    return span_of(*SPANS[n])


def talk(video, rows=ROWS):
    video.with_suffix(".words.json").write_text(json.dumps(rows))
    return video


def ctx_for(video):
    return treatment.load_context(open_project(str(video)), duration=DURATION, silences=no_silences)


def row_of(ctx, kind, near):
    """The id of the page row of ``kind`` that starts at ``near`` seconds."""
    rows = treatment.page_state(ctx)["rows"]
    return min((r for r in rows if r["group"] == kind), key=lambda r: abs(r["start"] - near))["id"]


LAST_WORD_OF_S4 = SPANS[3][1]


def changed(video):
    """A video Claude cut four ways and the creator changed every way there is. Returns its context."""
    talk(video)
    cuts = [
        {**sentence(1), "reason": "said twice", "kind": "repeat"},
        {**span_of(SPANS[3][0], SPANS[3][0] + 1), "reason": "filler", "kind": "other"},
        {**sentence(5), "reason": "off the point", "kind": "off_topic"},
        {**span_of(LAST_WORD_OF_S4 - 1, LAST_WORD_OF_S4), "reason": "false start", "kind": "false_start"},
    ]
    tools.set_edit(str(video), cuts, **QUIET)
    ctx = ctx_for(video)
    treatment.set_cut_state(ctx, {"id": row_of(ctx, "repeat", sentence(1)["start"]), "state": "put_back"})
    treatment.rate_cut(ctx, {"id": row_of(ctx, "other", cuts[1]["start"]), "rating": "good"})
    treatment.rate_cut(ctx, {"id": row_of(ctx, "off_topic", cuts[2]["start"]), "rating": "bad"})
    treatment.add_cut(ctx, span_of(SPANS[4][0] + 2, SPANS[4][1]))
    treatment.add_keep(ctx, {**span_of(*SPANS[0]), "note": "the opening"})
    treatment.add_keep(ctx, {**span_of(LAST_WORD_OF_S4), "exact": True})
    return ctx


def built(video, **more):
    return shapes.build_send(open_project(str(video)), duration=DURATION, **more)


def actions(send):
    return [r["creator"]["action"] for r in send["records"]]


def transcript_tokens(rows=ROWS):
    """Every word of the transcript as a whole lower-case token, punctuation off."""
    return {t for row in rows for t in re.findall(r"[a-z0-9']+", row["word"].lower())}


def tokens_of(text):
    return set(re.findall(r"[a-z0-9_']+", text.lower()))


# ── the privacy test: no word of the video is ever in a send ─────────────────


def test_a_send_holds_no_word_from_the_video_outside_the_named_fillers(video):
    changed(video)
    send = built(video)
    assert len(send["records"]) >= 7
    text = json.dumps(send)
    leaked = (transcript_tokens() - shapes.FILLER) & tokens_of(text)
    assert not leaked, f"words from the video in the send: {leaked}"
    for word in ("zebra", "wellington", "priya", "seaside", "cafes", "1987", "escaped", "watching", "thanks"):
        assert word not in text.lower()
    # the words a shape may name are the fillers, and "you know" and "um" are two of them
    named = {c["filler"] for r in send["records"] for side in ("before", "after") for c in r["context"][side] if c["filler"]}
    assert named <= shapes.FILLER and named, "a filler next to a cut is named"


def test_the_send_says_every_kind_of_thing_the_creator_did(video):
    changed(video)
    send = built(video)
    got = actions(send)
    assert set(got) == {"put_back", "rated_good", "rated_bad", "brought_back_words", "cut_by_hand", "kept_part", "kept"}
    assert got.count("rated_bad") == 1 and got.count("put_back") == 1


def test_a_wrong_cut_is_one_record_the_rating_and_not_also_a_put_back(video):
    talk(video)
    tools.set_edit(str(video), [{**sentence(1), "reason": "said twice", "kind": "repeat"}], **QUIET)
    ctx = ctx_for(video)
    treatment.rate_cut(ctx, {"id": row_of(ctx, "repeat", sentence(1)["start"]), "rating": "bad"})
    assert treatment.creator_changes(ctx.edit)["put_back"], "the edit did put the cut back"
    send = built(video)
    assert actions(send) == ["rated_bad"]
    assert send["records"][0]["proposed"]["kind"] == "repeat"


def test_a_plain_put_back_is_a_put_back(video):
    talk(video)
    tools.set_edit(str(video), [{**sentence(1), "reason": "said twice", "kind": "repeat"}], **QUIET)
    ctx = ctx_for(video)
    treatment.set_cut_state(ctx, {"id": row_of(ctx, "repeat", sentence(1)["start"]), "state": "put_back"})
    (record,) = built(video)["records"]
    assert record["creator"]["action"] == "put_back"
    assert record["proposed"] == {"source": "claude", "kind": "repeat", "length_s": record["proposed"]["length_s"], "words": 7}
    assert record["pace"] in shapes.PACE


def test_a_send_is_the_shape_of_each_change(video):
    changed(video)
    by = {r["creator"]["action"]: r for r in built(video)["records"]}
    hand = by["cut_by_hand"]
    assert hand["proposed"]["source"] == "creator" and hand["proposed"]["kind"] == "other" and hand["proposed"]["words"] == 2
    assert hand["context"]["sentence_position"] == "end", "the last words of a sentence"
    part = by["kept_part"]
    assert part["proposed"]["source"] == "creator" and part["context"]["sentence_position"] == "whole"
    left = by["kept"]
    assert left["proposed"]["kind"] == "false_start" and left["proposed"]["words"] >= 1
    assert by["rated_good"]["proposed"]["kind"] == "other", "a filler cut Claude filed under other"


def test_off_topic_is_sent_as_other_until_the_server_takes_the_name(video):
    changed(video)
    send = built(video)
    assert "off_topic" not in json.dumps(send)
    bad = next(r for r in send["records"] if r["creator"]["action"] == "rated_bad")
    assert bad["proposed"]["kind"] == "other"
    # the server refuses any name that holds "topic", so the check refuses it too
    record = copy.deepcopy(send["records"][0])
    record["proposed"]["kind"] = "off_topic"
    assert shapes.validate_send({**send, "records": [record]}) == "Record 0: proposed: kind contains prohibited text"


def test_kinds_are_named_the_way_the_schema_names_them():
    for internal, sent in {"pauses": "pause", "fillers": "filler", "repeats": "stutter", "repeat": "repeat",
                           "false_start": "false_start", "other": "other", "likes": "likes",
                           "something new": "other", None: "other", "yours": "other"}.items():
        assert shapes._kind(internal) == sent
    assert set(shapes.SENDABLE_KIND.values()) <= shapes.KIND


# ── the words either side of a cut ───────────────────────────────────────────


def test_context_words_are_shapes_up_to_three_a_side(video):
    changed(video)
    send = built(video)
    for r in send["records"]:
        for side in ("before", "after"):
            assert len(r["context"][side]) <= 3
            for c in r["context"][side]:
                assert set(c) == {"pos", "dur", "gap_after", "pitch", "filler"}
                assert c["pitch"] == "unknown" and c["dur"] >= 0 and c["gap_after"] >= 0
    hand = next(r for r in send["records"] if r["creator"]["action"] == "cut_by_hand")
    assert [c["pos"] for c in hand["context"]["before"]] == ["other", "other", "prep"], "cafes, Thanks, for"
    thanks = hand["context"]["before"][1]
    assert thanks["dur"] == 0.4 and thanks["gap_after"] == 0.1, "0.4 s long, and 0.1 s before the next word"
    assert [c["pos"] for c in hand["context"]["after"]] == ["other", "prep", "other"], "Goodbye, from, Auckland"


def test_words_of_a_cut_are_not_context_and_played_words_are():
    words = [{"word": w, "start": i * 0.5, "end": i * 0.5 + 0.4} for i, w in enumerate("one two three four five six".split())]
    talk_ = shapes._Talk(words, [(1.0, 1.9)])  # "three" and "four" are cut
    before, after = talk_.around(1.0, 1.9)
    assert [w["pos"] for w in before] == ["num", "num"], "one, two"
    assert [w["pos"] for w in after] == ["num", "num"]
    assert len(talk_.inside(1.0, 1.9)) == 2
    assert talk_.around(1.0, 1.9) == talk_.around(1.0, 1.9)
    # a cut somewhere else in the video does not sit between the words either side
    skipping = shapes._Talk(words, [(0.5, 0.9), (1.0, 1.4)])
    before, after = skipping.around(1.0, 1.4)
    assert len(before) == 1, "the words cut elsewhere are not played, so they are not context"


def test_the_lexicon_is_closed_and_fillers_win():
    said = {"i": "pron", "they": "pron", "the": "det", "this": "det", "with": "prep", "into": "prep", "but": "conj",
            "because": "conj", "seven": "num", "1987": "num", "3rd": "num", "wow": "interj", "okay": "interj",
            "um": "filler", "like": "filler", "so": "filler", "right": "filler", "zebra": "other", "wellington": "other"}
    for word, pos in said.items():
        assert shapes.part_of_speech(word) == pos, word
    assert shapes.POS >= set(said.values())


def test_a_two_word_filler_counts_when_the_words_are_side_by_side():
    words = [{"word": w, "start": i * 0.5, "end": i * 0.5 + 0.4} for i, w in enumerate("well you know it works".split())]
    talk_ = shapes._Talk(words, [])
    assert [talk_.filler_at(i) for i in range(5)] == ["well", "you know", "you know", None, None]
    assert [talk_.shape(i)["pos"] for i in range(5)] == ["filler", "filler", "filler", "pron", "other"]
    apart = shapes._Talk([{"word": "you", "start": 0, "end": 0.3}, {"word": "really", "start": 0.4, "end": 0.7},
                          {"word": "know", "start": 0.8, "end": 1.0}], [])
    assert [apart.filler_at(i) for i in range(3)] == [None, None, None]


def test_the_sentence_position_comes_from_the_punctuation():
    words = [{"word": w, "start": i * 0.5, "end": i * 0.5 + 0.4}
             for i, w in enumerate("First one here. Second one goes here. Third.".split())]
    talk_ = shapes._Talk(words, [])
    at = lambda a, b: talk_.sentence_position(words[a]["start"], words[b]["end"])  # noqa: E731
    assert at(0, 2) == "whole" and at(3, 6) == "whole" and at(7, 7) == "whole"
    assert at(3, 4) == "start" and at(5, 6) == "end" and at(4, 5) == "middle"
    assert talk_.sentence_position(1.45, 1.55) == "end", "a pause right after a sentence"
    assert talk_.sentence_position(0.95, 1.05) == "middle", "a pause inside one"


def test_a_laugh_near_a_cut_is_its_distance_and_none_is_null(video):
    changed(video)
    plain = built(video)
    assert all(r["context"]["laugh_within_s"] is None for r in plain["records"])
    hand = span_of(SPANS[4][0] + 2, SPANS[4][1])
    laugh = [{"start": hand["end"] + 2.0, "end": hand["end"] + 2.5, "kind": "laugh"},
             {"start": 0.0, "end": 0.1, "kind": "sound"}]
    near = {r["creator"]["action"]: r for r in built(video, sounds=laugh)["records"]}
    assert near["cut_by_hand"]["context"]["laugh_within_s"] == 2.0
    assert near["kept_part"]["context"]["laugh_within_s"] is not None
    inside = built(video, sounds=[{"start": hand["start"], "end": hand["end"], "kind": "laugh"}])
    assert next(r for r in inside["records"] if r["creator"]["action"] == "cut_by_hand")["context"]["laugh_within_s"] == 0.0


def test_the_laugh_labels_are_read_from_the_cache_and_never_measured(video):
    changed(video)
    project = open_project(str(video))
    (project.root / "sounds.json").write_text(json.dumps({"labels": [{"start": 17.0, "end": 17.6, "kind": "laugh"}]}))
    send = built(video)
    assert any(r["context"]["laugh_within_s"] is not None for r in send["records"])
    (project.root / "sounds.json").write_text("not json")
    assert all(r["context"]["laugh_within_s"] is None for r in built(video)["records"])


def test_the_join_is_the_silence_a_played_cut_leaves_and_the_checks_flags(video):
    changed(video)
    send = built(video)
    left = next(r for r in send["records"] if r["creator"]["action"] == "kept")
    assert left["join"]["gap_left_s"] is not None and 0 <= left["join"]["gap_left_s"] <= 60
    put_back = next(r for r in send["records"] if r["creator"]["action"] == "rated_bad")
    assert put_back["join"] == {"gap_left_s": None, "flags": []}, "a cut put back has no join"
    flagged = built(video, join_rows=lambda cuts, words, duration, *, labels=None: [
        {"start": c["start"], "end": c["end"], "flags": ["splice", "not_a_flag", "removes_laugh"]} for c in cuts])
    assert next(r for r in flagged["records"] if r["creator"]["action"] == "kept")["join"]["flags"] == ["splice", "removes_laugh"]
    broken = built(video, join_rows=lambda *a, **k: 1 / 0)
    assert all(r["join"]["flags"] == [] for r in broken["records"]), "a join check that fails costs the flags and nothing else"


def test_the_gap_left_is_the_pause_before_plus_the_pause_after():
    words = [{"word": "a", "start": 0.0, "end": 0.4}, {"word": "b", "start": 1.0, "end": 1.4},
             {"word": "c", "start": 2.0, "end": 2.4}, {"word": "d", "start": 3.0, "end": 3.4}]
    talk_ = shapes._Talk(words, [(0.9, 2.5)])
    assert talk_.silence_left(0.9, 2.5) == round((0.9 - 0.4) + (3.0 - 2.5), 2) == 1.0
    assert shapes._Talk(words, [(0.0, 0.5)]).silence_left(0.0, 0.5) is None, "nothing before it"


def test_numbers_are_two_decimals_and_kept_in_range():
    assert shapes._two(0.1 + 0.2, 10) == 0.3
    assert shapes._two(-1, 10) == 0.0 and shapes._two(99, 10) == 10.0
    assert shapes._two(1 / 3, 10) == 0.33


def test_the_pace_is_the_saved_one_or_custom(video):
    ctx = changed(video)
    path = ctx.project.edit_path
    saved = json.loads(path.read_text())
    assert {r["pace"] for r in built(video)["records"]} == {"standard"}
    for treatment_dict, pace in ((Treatment(pace="hard").as_dict(), "hard"),
                                 (Treatment(pace="custom", fine=(0.4, 4.5)).as_dict(), "custom")):
        write_json_atomic(path, {**saved, "treatment": treatment_dict})
        assert {r["pace"] for r in built(video)["records"]} == {pace}
    assert shapes.PACE == {"natural", "standard", "fast", "tight", "hard", "max", "custom"}


# ── the check ────────────────────────────────────────────────────────────────


def test_every_built_send_passes_the_check_and_the_sets_are_the_schemas(video):
    assert shapes.SCHEMA == 1
    assert shapes.KIND == {"pause", "filler", "stutter", "repeat", "false_start", "off_topic", "likes", "other"}
    assert shapes.ACTION == {"kept", "put_back", "rated_good", "rated_bad", "brought_back_words", "cut_by_hand", "kept_part"}
    assert shapes.FILLER == {"um", "uh", "er", "ah", "hmm", "like", "so", "you know", "i mean", "basically", "literally",
                             "right", "actually", "well"}
    changed(video)
    send = built(video)
    assert shapes.validate_send(send) is None
    assert re.fullmatch(r"[A-Za-z0-9_-]{22}", send["send_id"]) and re.fullmatch(r"\d+\.\d+\.\d+", send["plugin_version"])
    assert send["schema"] == 1 and set(send) == {"schema", "send_id", "plugin_version", "records"}
    assert built(video)["send_id"] != send["send_id"]


def tampered(send, edit):
    """A copy of ``send`` after ``edit`` changed it."""
    out = copy.deepcopy(send)
    edit(out)
    return out


def test_the_check_turns_away_what_the_server_turns_away(video):
    changed(video)
    send = built(video)
    first = next(r for r in send["records"] if r["context"]["before"] and r["context"]["after"])
    i = send["records"].index(first)
    wordy = lambda out: out["records"][i]["context"]["before"][0]  # noqa: E731
    cases = {
        "an unknown key at the top": lambda o: o.update(text="zebra"),
        "an unknown key in a record": lambda o: o["records"][i].update(note="Priya"),
        "an unknown key in proposed": lambda o: o["records"][i]["proposed"].update(text="zebra"),
        "an unknown key in context": lambda o: o["records"][i]["context"].update(words=["zebra"]),
        "an unknown key in creator": lambda o: o["records"][i]["creator"].update(note="zebra"),
        "an unknown key in join": lambda o: o["records"][i]["join"].update(why="zebra"),
        "an unknown key in a word": lambda o: wordy(o).update(word="zebra"),
        "a smuggled word for a part of speech": lambda o: wordy(o).update(pos="Wellington"),
        "a smuggled word for a filler": lambda o: wordy(o).update(filler="zebra"),
        "a smuggled word for a flag": lambda o: o["records"][i]["join"].update(flags=["zebra"]),
        "a smuggled word for the pace": lambda o: o["records"][i].update(pace="Priya"),
        "a bad source": lambda o: o["records"][i]["proposed"].update(source="you"),
        "a bad kind": lambda o: o["records"][i]["proposed"].update(kind="zebra"),
        "a bad action": lambda o: o["records"][i]["creator"].update(action="deleted"),
        "a bad pitch": lambda o: wordy(o).update(pitch="high"),
        "a bad sentence position": lambda o: o["records"][i]["context"].update(sentence_position="beginning"),
        "a length with three decimals": lambda o: o["records"][i]["proposed"].update(length_s=1.234),
        "a length past an hour": lambda o: o["records"][i]["proposed"].update(length_s=3600.5),
        "a negative length": lambda o: o["records"][i]["proposed"].update(length_s=-1),
        "a length that is text": lambda o: o["records"][i]["proposed"].update(length_s="4"),
        "a fraction of a word": lambda o: o["records"][i]["proposed"].update(words=2.5),
        "a true for a number": lambda o: o["records"][i]["proposed"].update(words=True),
        "too many words": lambda o: o["records"][i]["proposed"].update(words=10001),
        "a word held too long": lambda o: wordy(o).update(dur=10.5),
        "a gap that is too long": lambda o: wordy(o).update(gap_after=61),
        "a laugh too far away": lambda o: o["records"][i]["context"].update(laugh_within_s=601),
        "a gap left too long": lambda o: o["records"][i]["join"].update(gap_left_s=60.5),
        "four words before": lambda o: o["records"][i]["context"].update(before=[wordy(o)] * 4),
        "flags that are not a list": lambda o: o["records"][i]["join"].update(flags="splice"),
        "a missing key": lambda o: o["records"][i]["context"].pop("laugh_within_s"),
        "a missing filler": lambda o: wordy(o).pop("filler"),
        "a missing gap left": lambda o: o["records"][i]["join"].pop("gap_left_s"),
        "a record that is not an object": lambda o: o["records"].append("zebra"),
        "a wrong schema": lambda o: o.update(schema=2),
        "a schema that is true": lambda o: o.update(schema=True),
        "a short id": lambda o: o.update(send_id="abc"),
        "an id with a space": lambda o: o.update(send_id="a" * 21 + " "),
        "an id with a newline after it": lambda o: o.update(send_id="a" * 22 + "\n"),
        "a version with a suffix": lambda o: o.update(plugin_version="0.2.0-beta"),
        "a version with a newline": lambda o: o.update(plugin_version="0.2.0\n"),
        "records that are not a list": lambda o: o.update(records={}),
        "five hundred and one records": lambda o: o.update(records=[o["records"][0]] * 501),
    }
    for what, edit in cases.items():
        assert shapes.validate_send(tampered(send, edit)), f"{what} was let through"
    assert shapes.validate_send("zebra") and shapes.validate_send(None) and shapes.validate_send([send])
    assert shapes.validate_send(send) is None


def test_the_check_allows_what_the_server_allows(video):
    changed(video)
    send = built(video)
    ok = lambda edit: shapes.validate_send(tampered(send, edit)) is None  # noqa: E731
    assert ok(lambda o: o.update(records=[]))
    tiny = {"pace": "tight", "proposed": {"source": "claude", "kind": "other", "length_s": 1.0, "words": 1},
            "context": {"before": [], "after": [], "sentence_position": "middle", "laugh_within_s": None},
            "creator": {"action": "kept"}, "join": {"gap_left_s": None, "flags": []}}
    assert ok(lambda o: o.update(records=[tiny] * 500))
    assert ok(lambda o: o["records"][0]["proposed"].update(length_s=3600, words=0))
    assert ok(lambda o: o["records"][0]["proposed"].update(words=4.0)), "4.0 is 4 to the server"
    assert ok(lambda o: o["records"][0]["context"].update(laugh_within_s=600, before=[], after=[]))
    assert ok(lambda o: o["records"][0]["join"].update(gap_left_s=None, flags=["splice", "splice"]))
    assert ok(lambda o: o["records"][0]["proposed"].update(length_s=0.3))
    assert ok(lambda o: o.update(send_id="A" * 16)) and ok(lambda o: o.update(send_id="a_-" * 10 + "ab"))
    every_pos = [{"pos": p, "dur": 0.1, "gap_after": 0, "pitch": "flat", "filler": None} for p in sorted(shapes.POS)][:3]
    assert ok(lambda o: o["records"][0]["context"].update(before=every_pos))
    for f in sorted(shapes.FILLER):
        assert ok(lambda o: o["records"][0]["context"].update(after=[{"pos": "filler", "dur": 1, "gap_after": 1, "pitch": "rising", "filler": f}]))


def test_a_send_over_the_size_limit_is_turned_away_and_a_built_one_fits(video):
    changed(video)
    send = built(video)
    full = {**send, "records": [copy.deepcopy(send["records"][0]) for _ in range(500)]}
    for r in full["records"]:
        r["context"]["before"] = r["context"]["after"] = [{"pos": "other", "dur": 0.12, "gap_after": 0.12, "pitch": "unknown", "filler": None}] * 3
        r["join"]["flags"] = ["splice"] * 40
    assert len(shapes.body_of(full)) > shapes.MAX_BODY_BYTES
    assert shapes.validate_send(full) == "Body too large"
    fitted = shapes._fitted(full["records"])
    assert 0 < len(fitted) < 500
    assert shapes.validate_send({**send, "records": fitted}) is None
    assert len(shapes.body_of({**send, "records": fitted})) <= shapes.MAX_BODY_BYTES


def test_at_most_five_hundred_records_and_what_the_creator_did_comes_first():
    keep = [{"pace": "tight", "proposed": {"source": "claude", "kind": "other", "length_s": 1.0, "words": 1},
             "context": {"before": [], "after": [], "sentence_position": "middle", "laugh_within_s": None},
             "creator": {"action": "kept"}, "join": {"gap_left_s": None, "flags": []}}] * 600
    assert len(shapes._fitted(keep)) == 500


# ── the preview ──────────────────────────────────────────────────────────────


def test_the_preview_says_what_each_record_is_without_a_word_from_the_video(video):
    changed(video)
    send = built(video, sounds=[{"start": 17.0, "end": 17.6, "kind": "laugh"}])
    lines = shapes.describe(send)
    assert len(lines) == len(send["records"])
    said = tokens_of(" ".join(lines))
    everyday = {"you", "the", "a", "to", "for", "from", "has", "know"}
    assert not ((transcript_tokens() - shapes.FILLER - everyday) & said), "the lines are made of names and numbers"
    for line in lines:
        assert line[0].isupper() and line.endswith("."), line
        assert not banned_in(line) and not decimal_times_in(line) and "—" not in line, line
        assert not re.search(r"\d\.\d", line)
    assert any("You put back a restated-point cut (" in line for line in lines)
    assert any(line.startswith("You marked a cut as good") for line in lines)
    assert any(line.startswith("You marked a cut as wrong") for line in lines)
    assert any(line.startswith("You brought back words that were cut") for line in lines)
    assert any(line.startswith("You cut words by hand") for line in lines)
    assert any(line.startswith("You kept a part no matter what") for line in lines)
    assert any(line.startswith("You left a false-start cut in place") for line in lines)


def test_the_preview_line_reads_like_the_example():
    rec = {"pace": "tight", "proposed": {"source": "claude", "kind": "repeat", "length_s": 4.2, "words": 11},
           "context": {"before": [], "after": [], "sentence_position": "whole", "laugh_within_s": 0.4},
           "creator": {"action": "put_back"}, "join": {"gap_left_s": None, "flags": []}}
    send = {"records": [rec]}
    assert shapes.describe(send) == ["You put back a restated-point cut (4 s, 11 words) near a laugh."]
    far = copy.deepcopy(rec)
    far["context"]["laugh_within_s"] = 40.0
    far["proposed"].update(length_s=0.3, words=1)
    far["creator"]["action"] = "rated_bad"
    silent = copy.deepcopy(rec)
    silent["context"]["laugh_within_s"] = None
    silent["proposed"].update(words=0, length_s=1.5, kind="pause")
    silent["creator"]["action"] = "kept"
    assert shapes.describe({"records": [far, silent]}) == [
        "You marked a restated-point cut as wrong (under 1 s, 1 word).",
        "You left a long-pause cut in place (2 s).",
    ]
    for kind in shapes.KIND:
        for action in shapes.ACTION:
            line = shapes.describe({"records": [{**rec, "proposed": {**rec["proposed"], "kind": kind}, "creator": {"action": action}}]})[0]
            assert not banned_in(line) and not decimal_times_in(line), line


# ── sending ──────────────────────────────────────────────────────────────────


def test_with_sharing_off_send_raises_and_posts_nothing(video, monkeypatch):
    changed(video)
    send = built(video)
    posted = []
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: posted.append(a))
    with pytest.raises(StudioError, match="isn't set up"):
        shapes.send(open_project(str(video)), send, [])
    assert posted == []
    assert not (open_project(str(video)).root / "shares.jsonl").exists()


def test_what_is_posted_is_what_was_previewed_minus_what_was_removed(video, team):
    changed(video)
    project = open_project(str(video))
    send = built(video)
    previewed = copy.deepcopy(send)
    answer = shapes.send(project, send, [0, 2])
    expected = {**previewed, "records": [r for i, r in enumerate(previewed["records"]) if i not in (0, 2)]}
    (request, timeout), = team.sent
    assert request.data == json.dumps(expected, sort_keys=True).encode("utf-8")
    assert request.full_url == TEAM + "/v1/sends" and request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json" and timeout == 15
    assert request.get_header("User-agent") == f"lumr-studio/{send['plugin_version']}"
    assert set(request.headers) == {"Content-type", "User-agent"}, "nothing that names this machine"
    assert answer == {"send_id": send["send_id"], "records": len(expected["records"])}
    assert send == previewed, "the preview itself is not changed"
    assert shapes.validate_send(json.loads(request.data)) is None


def test_a_send_is_noted_in_shares_jsonl_with_the_id_and_the_date_only(video, team):
    changed(video)
    project = open_project(str(video))
    send = built(video)
    shapes.send(project, send, [])
    (line,) = (project.root / "shares.jsonl").read_text().splitlines()
    noted = json.loads(line)
    assert set(noted) == {"send_id", "sent_at", "records"}
    assert noted["send_id"] == send["send_id"] and noted["records"] == len(send["records"])
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", noted["sent_at"])
    shapes.send(project, built(video), [])
    assert len((project.root / "shares.jsonl").read_text().splitlines()) == 2, "each send adds a line"


@pytest.mark.parametrize("answer", [Answer(500), Answer(404), Answer(302), Answer(100)])
def test_a_reply_that_is_not_a_success_raises_and_writes_nothing(video, team, answer):
    changed(video)
    project = open_project(str(video))
    team.answer = answer
    with pytest.raises(StudioError, match="Nothing was saved"):
        shapes.send(project, built(video), [])
    assert not (project.root / "shares.jsonl").exists()
    assert len(team.sent) == 1, "never retried"


def test_a_refusal_or_a_lost_connection_raises_in_plain_words_and_writes_nothing(video, team):
    changed(video)
    project = open_project(str(video))
    send = built(video)
    team.answer = urllib.error.HTTPError(TEAM, 429, "Too Many Requests", {}, None)
    with pytest.raises(StudioError, match=r"\(429\)"):
        shapes.send(project, send, [])
    for why in (urllib.error.URLError("no route"), TimeoutError("slow"), OSError("down")):
        team.answer = why
        with pytest.raises(StudioError, match="Couldn't reach the Lumr Studio server"):
            shapes.send(project, send, [])
    assert not (project.root / "shares.jsonl").exists()
    assert len(team.sent) == 4


def test_a_bad_list_of_removed_lines_or_nothing_left_sends_nothing(video, team):
    changed(video)
    project = open_project(str(video))
    send = built(video)
    n = len(send["records"])
    for bad in ([n], [-1], ["0"], [True], [1.5], "0", 3):
        with pytest.raises(StudioError, match="not in the list"):
            shapes.send(project, send, bad)
    with pytest.raises(StudioError, match="nothing left"):
        shapes.send(project, send, list(range(n)))
    with pytest.raises(StudioError, match="nothing left"):
        shapes.send(project, {**send, "records": []}, [])
    assert team.sent == []
    shapes.send(project, send, [0, 0])
    assert len(team.sent) == 1 and len(team.payload["records"]) == n - 1


def test_a_send_that_breaks_the_check_is_never_posted(video, team):
    changed(video)
    project = open_project(str(video))
    send = built(video)
    send["records"][0]["context"]["before"] = [{"pos": "Wellington", "dur": 0.1, "gap_after": 0, "pitch": "flat", "filler": None}]
    with pytest.raises(StudioError, match="Nothing was sent"):
        shapes.send(project, send, [])
    assert team.sent == [] and not (project.root / "shares.jsonl").exists()


def test_a_failed_note_in_shares_jsonl_does_not_undo_a_send_that_went(video, team, caplog):
    changed(video)
    project = open_project(str(video))
    (project.root / "shares.jsonl").mkdir()
    send = built(video)
    assert shapes.send(project, send, [])["send_id"] == send["send_id"]
    assert len(team.sent) == 1


def test_a_video_the_creator_never_changed_still_builds_and_has_only_kept_cuts(video):
    talk(video)
    tools.set_edit(str(video), [{**sentence(1), "reason": "said twice", "kind": "repeat"}], **QUIET)
    send = built(video)
    assert actions(send) == ["kept"] and shapes.validate_send(send) is None
    fresh = open_project(str(video))
    fresh.edit_path.unlink()
    assert built(video)["records"] == []


def test_the_plugin_version_falls_back_when_it_is_not_a_version(monkeypatch):
    monkeypatch.setattr("lumr_studio.feedback.plugin_version", lambda: None)
    assert shapes._version() == "0.0.0"
    monkeypatch.setattr("lumr_studio.feedback.plugin_version", lambda: "0.2.0-beta")
    assert shapes._version() == "0.0.0"
    monkeypatch.setattr("lumr_studio.feedback.plugin_version", lambda: "0.2.0")
    assert shapes._version() == "0.2.0"
