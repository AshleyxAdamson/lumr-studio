"""Ums: Claude sees every one in its sentence, picks the ones that do not help, and no rule takes one.

Every take here is made up. The first is sound: tones for words, a train of
pitched bursts for a giggle, and a held tone for each "uhh" the speech model
wrote no word for. The transcript also holds two "um"s it did write.
"""

import json
import re
import shutil
import subprocess
import wave

import numpy as np
import pytest

from lumr_studio import edit as edits
from lumr_studio import tools, treatment, word_finder, word_times
from lumr_studio.errors import StudioError
from lumr_studio.pace import PACES
from lumr_studio.project import open_project
from lumr_studio.silences import no_silences
from lumr_studio.sounds import SAMPLE_RATE, UM_WORDS, classify_events, is_um_word
from lumr_studio.transcript import pack_transcript
from test_sound_finding import SameTimes, giggle, tone

RNG = np.random.default_rng(7)
TAKE_SECONDS = 18.5

# (word, start, end). A giggle after "plan.", an "uhh" that opens "Next we got lunch", an "uhh" tucked
# between "lunch" and "and", an "Um," the speech model wrote at the start of a sentence, and an "um,"
# it wrote inside one.
SAID = [
    ("So", 0.5, 0.8), ("that", 0.85, 1.15), ("was", 1.2, 1.45), ("the", 1.5, 1.7), ("plan.", 1.75, 2.3),
    ("Anyway", 6.7, 7.2), ("we", 7.25, 7.45), ("moved", 7.5, 7.85), ("on.", 7.9, 8.25),
    ("Next", 9.6, 10.0), ("we", 10.05, 10.2), ("got", 10.25, 10.55), ("lunch", 10.6, 11.0),
    ("and", 11.8, 12.05), ("left.", 12.1, 12.6),
    ("Um,", 13.6, 13.9), ("that", 14.0, 14.3), ("was", 14.35, 14.6), ("all.", 14.65, 15.0),
    ("It", 16.0, 16.2), ("was,", 16.25, 16.5), ("um,", 16.6, 16.85), ("fine.", 16.95, 17.4),
]
GIGGLE = (2.7, 5.3)
UHH_OPENING = (8.9, 9.4)
UHH_INSIDE = (11.15, 11.65)
UM_OPENING = (13.6, 13.9)
UM_INSIDE = (16.6, 16.85)
EVERY_UM = (UHH_OPENING, UHH_INSIDE, UM_OPENING, UM_INSIDE)


def said() -> list[dict]:
    return [{"word": word, "start": start, "end": end} for word, start, end in SAID]


def take_audio() -> np.ndarray:
    audio = 0.003 * RNG.uniform(-1, 1, round(TAKE_SECONDS * SAMPLE_RATE))  # the room
    sounds = [((start, end), tone(end - start, 0.5)) for _word, start, end in SAID]
    sounds += [(GIGGLE, giggle(GIGGLE[1] - GIGGLE[0]))]
    sounds += [(span, tone(span[1] - span[0], 0.3, pitch=180.0)) for span in (UHH_OPENING, UHH_INSIDE)]
    for (start, _end), sound in sounds:
        at = round(start * SAMPLE_RATE)
        audio[at:at + len(sound)] += sound
    return audio


@pytest.fixture(scope="module")
def take_master(tmp_path_factory):
    """The made-up take as a video file, made once: a black picture over the audio."""
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    folder = tmp_path_factory.mktemp("ums")
    with wave.open(str(folder / "take.wav"), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes((np.clip(take_audio(), -1, 1) * 32767).astype("<i2").tobytes())
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c=black:size=64x64:rate=5:duration={TAKE_SECONDS}",
            "-i", str(folder / "take.wav"),
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-shortest", str(folder / "take.mp4"),
        ],
        check=True,
    )
    return folder / "take.mp4"


@pytest.fixture(autouse=True)
def fresh_recipes():
    treatment.forget_recipes()
    yield
    treatment.forget_recipes()


@pytest.fixture
def take(tmp_path, take_master, jobs):
    """The take, transcribed: its words measured, and the sounds between them found."""
    folder = tmp_path / "footage"
    folder.mkdir()
    video = folder / "take.mp4"
    shutil.copy(take_master, video)
    video.with_suffix(".words.json").write_text(json.dumps(said()))
    job = tools.transcribe(str(video), aligner=SameTimes(), silences=no_silences, jobs=jobs)
    assert jobs.wait(job["job_id"], timeout=60)["status"] == "done"
    return video


def ctx_for(video):
    return treatment.load_context(open_project(str(video)), silences=no_silences, labels_for=tools.measured_labels)


def find_ums(video, asked=("um",)) -> dict:
    return tools.find_words(str(video), list(asked), silences=no_silences)


def lines_of(found) -> list[str]:
    return [line for line in found["text"].splitlines() if line.startswith("w")]


def line_at(found, span) -> str:
    """The line of the um that sits in ``span``, give or take the room a found sound gets."""
    for line in lines_of(found):
        start, end = edits.span_of_pick_id(line.split()[0])
        if abs(start - span[0]) < 0.1 and abs(end - span[1]) < 0.1:
            return line
    raise AssertionError(f"no um listed at {span}: {found['text']}")


def id_at(found, span) -> str:
    return line_at(found, span).split()[0]


def taken(video, span) -> float:
    """Seconds of ``span`` the saved edit removes."""
    cuts = json.loads(open_project(str(video)).edit_path.read_text())["cuts"]
    return round(sum(max(0.0, min(c["end"], span[1]) - max(c["start"], span[0])) for c in cuts), 3)


def picks_for(video, spans, reason="opens the line after a pause"):
    found = find_ums(video)
    return [{"id": id_at(found, span), "reason": reason} for span in spans]


def likes(state):
    return state["take_out"][3]


def why_at(state, span):
    return next(w[4] for w in state["words"] if abs(w[1] - span[0]) < 0.1 and abs(w[2] - span[1]) < 0.1)


# ── Claude sees every um ──────────────────────────────────────────────────────


def test_find_words_lists_every_um_written_or_heard_and_says_where_each_sits(take):
    found = find_ums(take)
    assert found["words"] == [
        {"word": "um", "said": 4, "clean": 4, "already_out": 0, "opens": 2, "inside": 2, "sounds": 2},
    ]
    assert found["listed"] == 4 and len(lines_of(found)) == 4
    first = found["text"].splitlines()[0]
    assert first.startswith("um: 4 in this video, 2 open a sentence, 2 inside one. 2 of them are sounds")
    # A held "uhh" in the gap before a sentence opens it; so does the "Um," the speech model wrote.
    for span in (UHH_OPENING, UM_OPENING):
        assert " OPENS A SENTENCE, QUIET " in line_at(found, span)
    # One tucked between two words of a sentence is inside it, written or heard.
    for span in (UHH_INSIDE, UM_INSIDE):
        assert " INSIDE A SENTENCE, QUIET " in line_at(found, span)
    assert re.search(r"moved on\. \[\(um 0\.\ds\)\] Next we got lunch", line_at(found, UHH_OPENING))
    assert re.search(r"we got lunch \[\(um 0\.\ds\)\] and left\.", line_at(found, UHH_INSIDE))
    assert "and left. [Um,] that was all." in line_at(found, UM_OPENING)
    assert "It was, [um,] fine." in line_at(found, UM_INSIDE)


def test_every_um_line_says_the_quiet_either_side_in_seconds(take):
    found = find_ums(take)
    quiet = {}
    for span in EVERY_UM:
        said = re.search(r"QUIET (\d\.\d\d)s BEFORE, (\d\.\d\d)s AFTER: ", line_at(found, span))
        assert said, line_at(found, span)
        quiet[span] = (float(said.group(1)), float(said.group(2)))
    assert quiet[UHH_OPENING] == (pytest.approx(0.65, abs=0.1), pytest.approx(0.2, abs=0.1)), "a clear pause before it"
    assert quiet[UHH_INSIDE] == (pytest.approx(0.15, abs=0.1), pytest.approx(0.15, abs=0.1)), "tucked between two words"
    assert quiet[UM_OPENING] == (pytest.approx(1.0, abs=0.05), pytest.approx(0.1, abs=0.05))
    assert quiet[UM_INSIDE] == (pytest.approx(0.1, abs=0.05), pytest.approx(0.1, abs=0.05))
    assert "QUIET" in found["text"].split("\nw")[0], "and the lines above say how to read it"


def test_a_giggle_is_never_listed_as_an_um(take):
    found = find_ums(take)
    for line in lines_of(found):
        start, end = edits.span_of_pick_id(line.split()[0])
        assert end <= GIGGLE[0] or start >= GIGGLE[1], line
    labels = tools.measured_labels(open_project(str(take)), word_times.load_words(open_project(str(take))))
    assert [lb["kind"] for lb in labels] == ["laugh", "um", "um"]


def test_any_spelling_asks_for_every_um_and_other_words_are_found_beside_them(take):
    assert find_ums(take, ["uh", "Umm", "er"])["words"] == find_ums(take)["words"], "one um, however it is spelled"
    both = find_ums(take, ["um", "that"])
    assert [(w["word"], w["said"]) for w in both["words"]] == [("um", 4), ("that", 2)]
    assert "opens" not in both["words"][1], "only an um says where it sits"
    plain = next(line for line in lines_of(both) if "[that]" in line)
    assert "QUIET" not in plain and re.search(r" q\.\d\d/\.\d\d$", plain), "every other word reads as it did"


def test_the_transcript_shows_an_um_in_its_sentence_and_a_laugh_on_a_line_of_its_own(take):
    lines = tools.read_transcript(str(take))["text"].splitlines()
    assert any(re.fullmatch(r"\[[\d.]+-[\d.]+\] \(laugh 2\.\ds\)", line) for line in lines), "the giggle reads as it did"
    opening = next(line for line in lines if "Next we got lunch" in line)
    assert re.search(r"\] \(um 0\.\ds\) Next we got lunch \(um 0\.\ds\) and left\.", opening), opening
    assert not any(re.fullmatch(r"\[[\d.]+-[\d.]+\] \((um|sound) [\d.]+s\)", line) for line in lines), "no um on a line of its own"
    assert any("Um, that was all." in line for line in lines) and any("It was, um, fine." in line for line in lines)


# ── no rule takes one ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("pace", list(PACES))
def test_no_pace_and_no_switch_removes_an_um_nobody_picked(take, pace):
    result = tools.set_edit(str(take), [], auto_tighten=True, pace=pace, silences=no_silences)
    assert result["auto"]["take_out"] == ["pauses", "fillers", "repeats"], "every switch is on"
    for span in EVERY_UM:
        assert taken(take, span) == 0, (pace, span)
    assert taken(take, GIGGLE) == 0
    state = treatment.page_state(ctx_for(take))
    assert state["take_out"][1]["words"] == [], "Filler words names no um"
    assert likes(state)["count"] == 0 and likes(state)["ready"] is False and likes(state)["said"] == 4
    found = find_ums(take)
    assert found["words"][0]["already_out"] == 0 and "OUT:" not in found["text"].split("\nw", 1)[1]


def test_the_hardest_pace_still_cuts_the_pauses_around_them(take):
    tools.set_edit(str(take), [], auto_tighten=True, pace="max", silences=no_silences)
    assert taken(take, (12.6, 13.6)) > 0.5, "the pause before \"Um,\" goes, down to what the pace leaves"
    assert taken(take, UM_OPENING) == 0


# ── a pick takes one, and the creator has the last word ───────────────────────


def test_a_picked_um_is_cut_whether_it_is_a_word_or_a_sound(take):
    picks = picks_for(take, (UHH_OPENING, UM_OPENING))
    result = tools.set_edit(str(take), [], picks=picks, silences=no_silences)
    assert (result["picks"]["picked"], result["picks"]["out"], result["picks"]["left_in"]) == (2, 2, [])
    assert result["picks"]["counts"] == "2 ums" and result["picks"]["out_counts"] == "2 ums"
    assert taken(take, UHH_OPENING) == pytest.approx(0.5, abs=0.1) and taken(take, UM_OPENING) == pytest.approx(0.3, abs=0.01)
    assert taken(take, UHH_INSIDE) == 0 and taken(take, UM_INSIDE) == 0, "the two inside a sentence stay"
    saved = json.loads(open_project(str(take)).edit_path.read_text())
    assert [p["text"] for p in saved["picks"]] == ["(um)", "Um,"]
    assert [(c["source"], c["kind"]) for c in saved["cuts"]] == [("pick", "likes")] * 2

    summary = tools.get_edit(str(take))["picks"]
    assert (summary["picked"], summary["by_word"], summary["counts"]) == (2, [["um", 2]], "2 ums")
    found = find_ums(take)
    assert line_at(found, UHH_OPENING).endswith("OUT:pick") and "OUT" not in line_at(found, UHH_INSIDE)
    shown = tools.read_transcript(str(take), show_cuts=True)["text"]
    assert re.search(r"CUT \[[\d.]+-[\d.]+\] \(um 0\.\ds\)\n", shown), shown


def test_a_picked_um_stays_out_at_every_pace_and_joins_the_pause_beside_it(take):
    picks = picks_for(take, (UHH_OPENING, UM_OPENING))
    for pace in PACES:
        tools.set_edit(str(take), [], auto_tighten=True, pace=pace, picks=picks, silences=no_silences)
        assert taken(take, UHH_OPENING) == pytest.approx(0.5, abs=0.1), pace
        assert taken(take, UM_OPENING) == pytest.approx(0.3, abs=0.01), pace
        assert taken(take, UHH_INSIDE) == 0 and taken(take, UM_INSIDE) == 0 and taken(take, GIGGLE) == 0, pace


def test_the_page_counts_picked_ums_and_one_switch_takes_them_all_on_or_off(take):
    tools.set_edit(str(take), [], picks=picks_for(take, (UHH_OPENING, UM_OPENING, UM_INSIDE)), silences=no_silences)
    ctx = ctx_for(take)
    state = treatment.page_state(ctx)
    entry = likes(state)
    assert (entry["key"], entry["label"], entry["ready"]) == ("likes", "Ums and likes", True)
    assert (entry["count"], entry["picked"], entry["said"], entry["words"]) == (3, 3, 4, [["um", 3]])
    assert [w[0] for w in state["words"] if w[4] == treatment.REMOVED_AS_PICKED] == ["(um)", "Um,", "um,"]
    assert [w[0] for w in state["words"] if w[0].startswith("(")] == ["(laugh)", "(um)", "(um)"]

    off = treatment.change_treatment(ctx, {"take_out": {"likes": False}})
    assert off["removed"] == [] and off["changed"]["likes"] == -3 and likes(off)["count"] == 3
    on = treatment.change_treatment(ctx, {"take_out": {"likes": True}})
    assert on["removed"] == state["removed"] and on["changed"]["likes"] == 3


def test_a_double_click_brings_one_picked_um_back_and_it_stays_back(take):
    tools.set_edit(str(take), [], picks=picks_for(take, (UHH_OPENING, UM_OPENING)), silences=no_silences)
    ctx = ctx_for(take)
    sound = next(w for w in treatment.page_state(ctx)["words"] if w[0] == "(um)" and w[3])
    # What the page sends for a double-click on a struck word: keep exactly that word.
    back = treatment.add_keep(ctx, {"start": sound[1], "end": sound[2], "exact": True})
    assert why_at(back, UHH_OPENING) == treatment.PLAYING and why_at(back, UM_OPENING) == treatment.REMOVED_AS_PICKED
    assert (likes(back)["count"], likes(back)["kept_by_you"], back["changed"]["words_back"]) == (1, 1, 1)
    assert back["keeps"][0]["text"] == "(um)", "the page says what she brought back"
    assert taken(take, UHH_OPENING) == 0
    for body in ({"take_out": {"likes": False}}, {"take_out": {"likes": True}}, {"pace": "max"}):
        state = treatment.change_treatment(ctx, body)
        assert why_at(state, UHH_OPENING) == treatment.PLAYING, body
    again = tools.set_edit(str(take), [], picks=picks_for(take, (UHH_OPENING, UM_OPENING)), silences=no_silences)
    assert [k["text"] for k in again["picks"]["kept_by_the_creator"]] == ["(um)"], "a new pick list leaves it in too"
    assert tools.get_edit(str(take))["creator"]["brought_back"][0]["text"] == "(um)"


def test_she_can_cut_by_hand_an_um_claude_left_in(take):
    tools.set_edit(str(take), [], picks=picks_for(take, (UM_OPENING,)), silences=no_silences)
    ctx = ctx_for(take)
    sound = next(w for w in treatment.page_state(ctx)["words"] if w[0] == "(um)" and abs(w[1] - UHH_INSIDE[0]) < 0.1)
    state = treatment.add_cut(ctx, {"start": sound[1], "end": sound[2]})
    assert why_at(state, UHH_INSIDE) == treatment.REMOVED_BY_YOU and taken(take, UHH_INSIDE) == pytest.approx(0.5, abs=0.1)
    assert tools.get_edit(str(take))["creator"]["cut_words"] == [["um", 1]]


def test_a_laugh_can_never_be_picked(take):
    labels = tools.measured_labels(open_project(str(take)), word_times.load_words(open_project(str(take))))
    laugh = edits.pick_id(labels[0]["start"], labels[0]["end"])
    with pytest.raises(StudioError, match="names a laugh or another sound, not a word or an um"):
        tools.set_edit(str(take), [], picks=[{"id": laugh, "reason": "dead air"}], silences=no_silences)
    assert not open_project(str(take)).edit_path.exists()
    mixed = picks_for(take, (UHH_OPENING,)) + [{"id": laugh, "reason": "dead air"}]
    result = tools.set_edit(str(take), [], picks=mixed, silences=no_silences)
    assert result["picks"]["picked"] == 1 and [r["index"] for r in result["picks"]["rejected"]] == [1]
    assert "cut that covers all of it" in result["picks"]["rejected"][0]["why"]
    assert taken(take, GIGGLE) == 0


# ── made-up words alone: where an um sits, and what counts as one ─────────────


def um_sound(start, end, **more):
    return {"word": "[vocalization]", "start": start, "end": end, "type": "event", **more}


HELD = {"bursts": 0, "burst_rate": 0.0, "voiced": 0.9}
IN_BURSTS = {"bursts": 5, "burst_rate": 4.5, "voiced": 0.7}
BREATH = {"bursts": 0, "burst_rate": 0.0, "voiced": 0.05}


def places(words, measures):
    labels = classify_events(words, measures=measures)
    return word_finder.find_places(words, [["um"]], [], 60.0, labels=labels), labels


def w(text, start, end):
    return {"word": text, "start": start, "end": end}


def test_an_um_opens_a_sentence_after_a_sentence_end_or_a_pause_and_is_inside_one_otherwise():
    words = [
        um_sound(0.2, 0.6),                                              # the first thing said
        w("Mix", 0.7, 1.0), w("the", 1.05, 1.2), w("flour.", 1.25, 1.7),
        um_sound(1.8, 2.2), w("Then", 2.3, 2.6), w("add", 2.65, 2.9),    # after a sentence end, no pause
        um_sound(2.95, 3.3), w("the", 3.35, 3.5), w("salt", 3.55, 3.9),  # between two words
        um_sound(4.6, 5.0), w("and", 5.05, 5.3), w("water.", 5.35, 5.8),  # no sentence end, but 0.7 s of quiet before
    ]
    found, labels = places(words, [HELD] * 4)
    assert [lb["kind"] for lb in labels] == ["um"] * 4
    assert [p.position for p in found] == ["opens", "opens", "inside", "opens"]
    assert [(p.quiet_before, p.quiet_after) for p in found] == [(0.2, 0.1), (0.1, 0.1), (0.05, 0.05), (0.7, 0.05)]
    assert all(p.sound and p.clean for p in found)


def test_ums_said_one_after_another_sit_where_the_first_one_does():
    words = [w("Rest", 0.5, 0.8), w("it.", 0.85, 1.1), w("Um,", 1.2, 1.4), um_sound(1.45, 1.8), w("so", 1.85, 2.0),
             w("we", 2.05, 2.2), w("uh,", 2.25, 2.4), w("um", 2.45, 2.6), w("wait.", 2.65, 3.0)]
    found, _ = places(words, [HELD])
    assert [(p.said, p.position) for p in found] == [
        ("Um,", "opens"), ("(um 0.3s)", "opens"), ("uh,", "inside"), ("um", "inside"),
    ]


def test_a_laugh_a_breath_and_an_unmeasured_sound_are_never_ums():
    words = [w("the", 0.5, 0.8), w("joke.", 0.9, 1.4), um_sound(1.6, 2.9), w("So", 3.2, 3.5),
             um_sound(3.6, 4.0), w("anyway", 4.1, 4.6), um_sound(4.7, 5.1), w("next.", 5.2, 5.6)]
    found, labels = places(words, [IN_BURSTS, BREATH, {}])
    assert [lb["kind"] for lb in labels] == ["laugh", "sound", "sound"] and found == []
    # Without a measure nothing is called an um on a guess, and a transcript read with no labels lists words only.
    assert word_finder.find_places(words + [w("um", 6.0, 6.2)], [["um"]], [], 60.0)[0].said == "um"
    line = word_finder.pack_places(found, [["um"]])["text"].splitlines()[0]
    assert line.startswith("um: 0 in this video")


def test_a_sound_can_only_be_picked_when_it_is_an_um():
    words = [w("the", 0.5, 0.8), w("joke.", 0.9, 1.4), um_sound(1.6, 2.9), w("So", 3.2, 3.5),
             um_sound(3.6, 4.0), w("anyway", 4.1, 4.6), um_sound(4.7, 5.1), w("next.", 5.2, 5.6)]
    labels = classify_events(words, measures=[IN_BURSTS, BREATH, HELD])
    said = edits.SaidOrder.of(words)
    from lumr_studio.sounds import um_spans

    ums = um_spans(labels)
    assert ums == [(4.7, 5.1)]
    assert edits.parse_pick(0, {"id": "w4.700-5.100", "reason": "opens the line"}, said, ums)["text"] == "(um)"
    for other in ("w1.600-2.900", "w3.600-4.000"):
        with pytest.raises(edits.CutRejected, match="names a laugh or another sound"):
            edits.parse_pick(0, {"id": other, "reason": "x"}, said, ums)
    with pytest.raises(edits.CutRejected, match="names a laugh or another sound"):
        edits.parse_pick(0, {"id": "w4.700-5.100", "reason": "x"}, said), "with no labels no sound is an um"
    # A sound picked as an um and judged a laugh since stays in, whole.
    out = edits.build_edit([], words, 60.0, picks=[{"id": "w1.600-2.900", "start": 1.6, "end": 2.9}], whole_sounds=[(1.6, 2.9)])
    assert out.cuts == [] and [(p["status"], p["why"]) for p in out.picks] == [("left_in", edits.IS_A_SOUND_TO_KEEP)]


def test_the_words_a_speech_model_writes_for_a_hesitation_are_all_one_um():
    assert {"um", "umm", "uh", "uhh", "uhm", "er", "erm", "ah"} <= UM_WORDS
    assert not {"mhm", "huh", "oh", "so", "well", "like", "a"} & UM_WORDS, "each of these says something"
    assert is_um_word(w("Umm,", 0, 1)) and is_um_word(w("uh.", 0, 1)) and not is_um_word(w("umbrella", 0, 1))
    assert not is_um_word(um_sound(0, 1)), "a sound is an um by its label, not by its word"
    assert edits.pick_word("Um,") == edits.pick_word("(um)") == edits.pick_word("uhh") == "um"
    assert edits.pick_word("Like,") == "like" and edits.pick_word("You know,") == "you know"
    by_word = edits.picks_by_word(["like,", "(um)", "Like", "Um,", "like", "you know"])
    assert by_word == [["um", 2], ["like", 3], ["you know", 1]]
    assert edits.counted(by_word) == '2 ums, 3 likes, 1 "you know"' and edits.counted([["um", 1]]) == "1 um"
    assert edits.counted([]) == ""


def test_the_packed_transcript_puts_an_um_among_the_words_and_leaves_other_sounds_as_they_were():
    words = [w("Mix", 0.7, 1.0), w("the", 1.05, 1.2), w("flour.", 1.25, 1.7), um_sound(1.8, 2.2), w("Then", 2.3, 2.6),
             w("add", 2.65, 2.9), um_sound(2.95, 3.3), w("the", 3.35, 3.5), w("salt.", 3.55, 3.9),
             um_sound(4.0, 4.4), w("Done.", 4.5, 4.9)]
    labels = classify_events(words, measures=[HELD, HELD, BREATH])
    lines = pack_transcript(words, labels=labels)["text"].splitlines()
    assert lines == [
        "[0.70-3.90] Mix the flour. (um 0.4s) Then add (um 0.3s) the salt.",
        "[4.00-4.40] (sound 0.4s)",
        "[4.50-4.90] Done.",
        "END",
    ]
    # With no labels every sound reads as it always did, on a line of its own.
    assert [ln for ln in pack_transcript(words)["text"].splitlines() if "[vocalization]" in ln] == [
        "[1.80-2.20] [vocalization]", "[2.95-3.30] [vocalization]", "[4.00-4.40] [vocalization]",
    ]
    # A picked um is marked like any cut word.
    cut = pack_transcript(words, labels=labels, removed=[(1.75, 2.25)])["text"].splitlines()
    assert cut[:3] == ["[0.70-1.70] Mix the flour.", "CUT [1.80-2.20] (um 0.4s)", "[2.30-3.90] Then add (um 0.3s) the salt."]


# ── saved edits from before ───────────────────────────────────────────────────

TALK = [w("I", 0.5, 0.7), w("like", 0.8, 1.0), w("bread.", 1.1, 1.6), w("So", 2.5, 2.7), w("like,", 2.8, 3.0),
        w("three", 3.1, 3.4), w("loaves.", 3.45, 4.0), w("Like,", 8.0, 8.3), w("how", 8.4, 8.6), w("many?", 8.65, 9.2)]


def edit_saved_before(video, likes_on: bool) -> dict:
    """An edit as the plugin saved it before ums could be picked: two picked likes, a rating, and the switch."""
    pick = lambda start, end, text: {"id": edits.pick_id(start, end), "start": start, "end": end, "text": text,
                                     "reason": "a pause word"}
    cut = lambda start, end: {"start": start, "end": end, "reason": "A filler word Claude picked", "source": "pick",
                              "kind": "likes"}
    switches = {"pauses": False, "fillers": False, "repeats": False, "likes": likes_on}
    settings = {"pace": "standard", "take_out": switches, "gap_length": None, "fine": {"gap_length": 0.6, "rhythm": 3.0}}
    return {
        "version": 1, "video": str(video), "duration": 20.0, "updated_at": "2026-01-05T10:00:00+00:00",
        "auto_tighten": False, "gap_length": None, "pace": None,
        "cuts": [cut(2.8, 3.0), cut(8.0, 8.3)] if likes_on else [], "requested": [], "rejected": [], "keep": [],
        "treatment": settings, "set_by_claude": {**settings, "take_out": {**switches, "likes": True}},
        "word_times": "estimated", "picks": [pick(2.8, 3.0, "like,"), pick(8.0, 8.3, "Like,")],
        "ratings": [{"id": "r2.80-3.00", "start": 2.8, "end": 3.0, "rating": "good", "source": "pick", "kind": "likes"}],
    }


@pytest.mark.parametrize("likes_on", [True, False])
def test_an_edit_saved_before_ums_could_be_picked_opens_as_it_was(video, likes_on):
    video.with_suffix(".words.json").write_text(json.dumps(TALK))
    project = open_project(str(video))
    before = edit_saved_before(video, likes_on)
    project.edit_path.write_text(json.dumps(before))
    written = project.edit_path.read_text()

    ctx = treatment.load_context(project, duration=20.0, silences=no_silences)
    state = treatment.page_state(ctx)
    assert project.edit_path.read_text() == written, "opening it writes nothing"
    assert state["removed"] == ([[2.8, 3.0], [8.0, 8.3]] if likes_on else [])
    assert state["settings"]["take_out"] == before["treatment"]["take_out"], "her switch stands as she left it"
    entry = likes(state)
    assert (entry["key"], entry["label"], entry["count"], entry["picked"], entry["words"]) == \
        ("likes", "Ums and likes", 2, 2, [["like", 2]])
    summary = tools.get_edit(str(video))
    assert (summary["picks"]["switch_on"], summary["picks"]["counts"]) == (likes_on, "2 likes")
    assert [p["id"] for p in summary["picks"]["picks"]] == [p["id"] for p in before["picks"]]
    assert [r["kind"] for r in summary["creator"]["rated"]["good"]] == ["likes"], "her rating of a pick is still read"
    assert summary["creator"].get("take_out") == (None if likes_on else {"likes": False})
    found = tools.find_words(str(video), ["like"], silences=no_silences, labels=tools.unmeasured_labels)
    assert found["words"] == [{"word": "like", "said": 3, "clean": 3, "already_out": 2 if likes_on else 0}]
    assert json.loads(project.edit_path.read_text())["ratings"] == before["ratings"]


def test_a_usual_saved_before_reads_with_the_same_switches():
    usual = {"pace": "fast", "take_out": {"pauses": True, "fillers": True, "repeats": False, "likes": False}}
    assert treatment.Treatment.from_dict(usual).take_out == frozenset({"pauses", "fillers"})
    assert treatment.Treatment.from_dict({"pace": "fast", "take_out": {"pauses": True}}).picks_on, "a file with no such switch has it on"
    assert edits.PICK_KIND == "likes" and list(edits.SWITCHES) == ["pauses", "fillers", "repeats", "likes"]


def test_an_edit_whose_filler_switch_took_an_um_sound_gives_it_back_and_keeps_the_rest(video):
    # Before, the Filler words switch took every held um, all or none. Such an edit is made again when it
    # is opened: the um plays until Claude picks it, and every cut somebody chose is where it was.
    words = [w("Mix", 0.7, 1.0), w("the", 1.05, 1.2), w("flour.", 1.25, 1.7), um_sound(2.4, 2.8), w("Then", 3.0, 3.3),
             w("add", 3.35, 3.6), w("salt.", 3.65, 4.1), w("Rest", 6.0, 6.3), w("it", 6.35, 6.5), w("now.", 6.55, 7.0)]
    video.with_suffix(".words.json").write_text(json.dumps(words))
    project = open_project(str(video))
    labels = lambda _project, said: classify_events(said, measures=[HELD])
    cut = {"start": 5.9, "end": 7.1, "reason": "said again later", "kind": "repeat"}
    tools.set_edit(str(video), [cut], auto_tighten=True, pace="natural", silences=no_silences, labels=labels)
    saved = json.loads(project.edit_path.read_text())
    assert not [c for c in saved["cuts"] if c["start"] < 2.8 and 2.4 < c["end"]], "today no rule takes it"
    # As it was saved before: the same edit, with the um taken by the rule.
    saved["cuts"] = sorted(saved["cuts"] + [{"start": 2.4, "end": 2.8, "reason": "auto: filler: sound", "source": "auto",
                                             "kind": "fillers"}], key=lambda c: c["start"])
    project.edit_path.write_text(json.dumps(saved))

    ctx = treatment.load_context(project, duration=20.0, silences=no_silences, labels_for=labels)
    state = treatment.page_state(ctx)
    assert not [r for r in state["removed"] if r[0] < 2.8 and 2.4 < r[1]], "the um plays again"
    assert [r["reason"] for r in state["rows"]] == ["said again later"] and state["rows"][0]["state"] == "kept_out"
    found = tools.find_words(str(video), ["um"], silences=no_silences, labels=labels)
    assert found["words"][0]["said"] == 1 and found["words"][0]["already_out"] == 0
    assert " OPENS A SENTENCE, QUIET 0.70s BEFORE, 0.20s AFTER: Mix the flour. [(um 0.4s)] Then add salt." in found["text"]
