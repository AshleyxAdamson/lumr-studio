"""carry_edit: a saved edit taken from one video onto a new reel made from the same footage.

Made-up media only. The talk is conftest's 20 seconds: a hello, an "um", a
line said twice, "this is the part that matters." and a sign-off. The old
reel is four takes of it; the new reel drops the last take and has another
clip joined in before the third.
"""

import json
import os
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from conftest import FakeAligner, make_words
from creator_words import banned_in, decimal_times_in
from tiny_models import tiny_set

from lumr_studio import carry, edit as edits, overlay_tools, taste, tools, treatment, word_times
from lumr_studio.carry import Part, Stretch, at_their_paths_now, fill_from, move_span, time_map
from lumr_studio.engine.render import probe_duration
from lumr_studio.errors import StudioError
from lumr_studio.overlays import load_overlays
from lumr_studio.project import open_project
from lumr_studio.silences import no_silences
from lumr_studio.tools import unmeasured_labels

QUIET = {"silences": no_silences, "labels": unmeasured_labels}


@pytest.fixture(autouse=True)
def fresh_recipes():
    treatment.forget_recipes()
    yield
    treatment.forget_recipes()


# ── the map ───────────────────────────────────────────────────────────────────


def sequence(*parts: tuple[str, float, float]) -> list[dict]:
    """Pieces as a sequence file lists them, from ``(clip, start, end)`` in play order."""
    pieces, at = [], 0.0
    for clip, start, end in parts:
        pieces.append({"clip": clip, "start": start, "end": end, "reel_start": at, "reel_end": at + end - start})
        at += end - start
    return pieces


def as_tuples(stretches: list[Stretch]) -> list[tuple[float, float, float]]:
    return [(round(s.old_start, 3), round(s.old_end, 3), round(s.new_start, 3)) for s in stretches]


def spans(moved) -> list[tuple[float, float]]:
    return [(round(p.start, 3), round(p.end, 3)) for p in moved.parts]


OLD = sequence(("a", 0, 30))
ADDED_IN_THE_MIDDLE = sequence(("a", 0, 12), ("x", 0, 5), ("a", 12, 30))


def test_a_clip_added_in_the_middle_shifts_what_plays_after_it():
    assert as_tuples(time_map(OLD, ADDED_IN_THE_MIDDLE)) == [(0, 12, 0), (12, 30, 17)]


def test_a_clip_added_at_the_start_shifts_everything():
    assert as_tuples(time_map(OLD, sequence(("x", 0, 5), ("a", 0, 30)))) == [(0, 30, 5)]


def test_a_clip_taken_out_leaves_no_stretch_for_its_seconds():
    old = sequence(("a", 0, 10), ("b", 0, 10), ("c", 0, 10))
    assert as_tuples(time_map(old, sequence(("a", 0, 10), ("c", 0, 10)))) == [(0, 10, 0), (20, 30, 10)]


def test_clips_in_another_order_each_go_where_they_now_play():
    old = sequence(("a", 0, 10), ("b", 0, 10), ("c", 0, 10))
    assert as_tuples(time_map(old, sequence(("c", 0, 10), ("a", 0, 10), ("b", 0, 10)))) == [
        (0, 10, 10), (10, 20, 20), (20, 30, 0),
    ]


def test_two_takes_of_one_clip_are_told_apart_by_their_seconds():
    old = sequence(("a", 0, 10), ("a", 40, 50))
    new = sequence(("a", 40, 50), ("x", 0, 3), ("a", 0, 10))
    assert as_tuples(time_map(old, new)) == [(0, 10, 13), (10, 20, 0)]


def test_a_take_made_longer_or_shorter_shares_the_seconds_both_use():
    assert as_tuples(time_map(sequence(("a", 5, 20)), sequence(("a", 0, 12)))) == [(0, 7, 5)]


def test_footage_that_plays_twice_in_the_new_video_gives_two_stretches():
    assert as_tuples(time_map(sequence(("a", 0, 10)), sequence(("a", 0, 10), ("a", 0, 10)))) == [(0, 10, 0), (0, 10, 10)]


def test_videos_with_no_clip_in_common_share_nothing():
    assert time_map(sequence(("a", 0, 10)), sequence(("b", 0, 10))) == []
    assert time_map(sequence(("a", 0, 10)), sequence(("a", 10, 20))) == []


def test_a_piece_that_ends_a_frame_early_ends_its_stretch_there():
    new = [{"clip": "a", "start": 0, "end": 10.02, "reel_start": 0, "reel_end": 10.0}]
    assert as_tuples(time_map(sequence(("a", 0, 30)), new)) == [(0, 10, 0)]


def test_a_clip_that_moved_is_the_new_videos_clip_of_its_name_and_length(tmp_path):
    """A sequence file names clips by path; after a move the same footage has another one."""
    here = tmp_path / "talk.mp4"
    here.write_bytes(b"footage")
    new = [{**p, "clip": str(here), "clip_duration": 30.0} for p in sequence(("x", 0, 12))]
    gone = [{**p, "clip": "/another/mac/talk.mp4", "clip_duration": 30.02} for p in sequence(("x", 0, 30))]
    assert [p["clip"] for p in at_their_paths_now(gone, new)] == [str(here)]
    assert as_tuples(time_map(at_their_paths_now(gone, new), new)) == [(0, 12, 0)]

    other_length = [{**p, "clip_duration": 31.0} for p in gone]
    assert [p["clip"] for p in at_their_paths_now(other_length, new)] == ["/another/mac/talk.mp4"]
    other_name = [{**p, "clip": "/another/mac/walk.mp4"} for p in gone]
    assert [p["clip"] for p in at_their_paths_now(other_name, new)] == ["/another/mac/walk.mp4"]
    # A clip still at its path is never taken for another: a copy does not get the original's edit.
    still = tmp_path / "kept" / "talk.mp4"
    still.parent.mkdir()
    still.write_bytes(b"footage")
    still_there = [{**p, "clip": str(still)} for p in gone]
    assert [p["clip"] for p in at_their_paths_now(still_there, new)] == [str(still)]
    # Two clips of the new video that both fit: nothing is guessed.
    twin = tmp_path / "twin" / "talk.mp4"
    two = new + [{**new[0], "clip": str(twin)}]
    assert [p["clip"] for p in at_their_paths_now(gone, two)] == ["/another/mac/talk.mp4"]


# ── moving one span ───────────────────────────────────────────────────────────

MAP = time_map(OLD, ADDED_IN_THE_MIDDLE)


def test_a_span_inside_carried_footage_moves_whole():
    assert (spans(move_span(3, 6, MAP)), move_span(3, 6, MAP).how) == ([(3, 6)], carry.WHOLE)
    after = move_span(20, 24.5, MAP)
    assert (spans(after), after.how, after.why()) == ([(25, 29.5)], carry.WHOLE, "")


def test_a_span_across_the_point_where_a_clip_went_in_becomes_two():
    moved = move_span(10, 14, MAP)
    assert moved.parts == [Part(start=10, end=12, was=10), Part(start=17, end=19, was=12)]
    assert (moved.how, moved.missing) == (carry.SPLIT, 0)
    assert "in 2 parts" in moved.why()


def test_an_end_moves_as_an_end_and_a_start_as_a_start():
    """A cut that ended where the piece ends still ends there: it does not reach over the new clip."""
    assert spans(move_span(8, 12, MAP)) == [(8, 12)]
    assert spans(move_span(12, 15, MAP)) == [(17, 20)]


def test_a_span_in_footage_that_is_gone_is_dropped_and_one_half_in_it_is_shortened():
    gone = time_map(sequence(("a", 0, 10), ("b", 0, 10)), sequence(("a", 0, 10)))
    dropped = move_span(12, 15, gone)
    assert (dropped.parts, dropped.how, dropped.why()) == ([], carry.DROPPED, carry.WHY_DROPPED)
    half = move_span(8, 13, gone)
    assert (spans(half), half.how, round(half.missing, 3), half.why()) == ([(8, 10)], carry.SHORTENED, 3, carry.WHY_SHORTENED)


def test_a_span_over_two_pieces_that_still_play_back_to_back_stays_one():
    split_only = time_map(OLD, sequence(("a", 0, 12), ("a", 12, 30)))
    moved = move_span(10, 14, split_only)
    assert (spans(moved), moved.how) == ([(10, 14)], carry.WHOLE)
    # The second piece lands a frame late in the new reel: still one span.
    late = [Stretch(0, 12, 0), Stretch(12, 30, 12.04)]
    assert spans(move_span(10, 14, late)) == [(10, 14.04)]


def test_a_sliver_clipped_off_by_an_edge_is_no_part():
    moved = move_span(8, 12.01, MAP)
    assert (spans(moved), moved.how) == ([(8, 12)], carry.WHOLE)


def test_a_span_whose_footage_plays_twice_lands_twice():
    twice = time_map(sequence(("a", 0, 10)), sequence(("a", 0, 10), ("a", 0, 10)))
    moved = move_span(2, 3, twice)
    assert (spans(moved), moved.how, moved.why()) == ([(2, 3), (12, 13)], carry.SPLIT, carry.WHY_TWICE)


def test_a_point_lands_in_the_stretch_that_holds_it():
    assert spans(move_span(20, 20, MAP)) == [(25, 25)]
    assert move_span(12, 12, MAP).parts[0].start == 12  # the end of the first stretch, not the start of the next


# ── moving the words ──────────────────────────────────────────────────────────


def test_words_move_with_their_footage_and_sounds_stay_sounds():
    words = [
        {"word": "one", "start": 1.0, "end": 1.4, "energy_rms": 0.2},
        {"word": "[vocalization]", "start": 11.0, "end": 11.6, "type": "event"},
        {"word": "two", "start": 13.0, "end": 13.5},
    ]
    moved, free = fill_from(words, MAP, [(0.0, 35.0)])
    assert moved == [
        {"word": "one", "start": 1.0, "end": 1.4, "energy_rms": 0.2},
        {"word": "[vocalization]", "start": 11.0, "end": 11.6, "type": "event"},
        {"word": "two", "start": 18.0, "end": 18.5},
    ]
    assert free == [(12, 17)]  # the clip that went in is all that still has no words


def test_a_word_never_reaches_past_the_footage_it_came_with():
    words = [{"word": "edge", "start": 11.7, "end": 12.2}]
    moved, _ = fill_from(words, MAP, [(0.0, 35.0)])
    assert moved == [{"word": "edge", "start": 11.7, "end": 12.0}]


def test_a_word_whose_end_runs_on_through_a_pause_stays_with_the_footage_it_starts_in():
    # A raw transcript stretches the last word before a pause up to the next word. Found on a real
    # video: a clip excerpt ended inside such a pause, the word's middle fell past the excerpt, the
    # word was left out, and a cut then slid back over it.
    words = [{"word": "crust,", "start": 11.2, "end": 16.0}, {"word": "and", "start": 16.0, "end": 16.2}]
    moved, _ = fill_from(words, MAP, [(0.0, 35.0)])
    assert moved == [
        {"word": "crust,", "start": 11.2, "end": 12.0},   # with the footage it starts in, and no further
        {"word": "and", "start": 21.0, "end": 21.2},
    ]


def test_a_word_that_starts_before_all_the_footage_goes_by_its_middle():
    late = time_map(sequence(("a", 0, 30)), sequence(("a", 5, 30)))
    moved, _ = fill_from([{"word": "half", "start": 4.8, "end": 5.6}, {"word": "gone", "start": 4.0, "end": 4.9}], late, [(0.0, 25.0)])
    assert moved == [{"word": "half", "start": 0.0, "end": 0.6}]


def test_footage_that_played_twice_in_the_old_video_fills_its_place_once():
    old = sequence(("a", 0, 10), ("a", 0, 10))
    words = [{"word": "hi", "start": 1.0, "end": 1.4}, {"word": "hi", "start": 11.0, "end": 11.4}]
    moved, free = fill_from(words, time_map(old, sequence(("a", 0, 10))), [(0.0, 10.0)])
    assert (moved, free) == ([{"word": "hi", "start": 1.0, "end": 1.4}], [])


# ── a real reel carried onto a real reel ──────────────────────────────────────

# The old reel: four takes of the talk. Each length is a whole number of frames at 25 fps.
TAKES = [(0.0, 7.0), (7.4, 11.0), (11.4, 14.6), (15.0, 17.4)]
FALSE_START = {"start": 3.85, "end": 6.65, "reason": "Starts the line and then starts it again", "kind": "false_start"}
ACROSS = {"start": 9.72, "end": 11.53, "reason": "Trails off into the next thought", "kind": "other"}
SIGN_OFF = {"start": 14.25, "end": 15.85, "reason": "The sign-off adds nothing", "kind": "off_topic"}
HEARD = [{"word": "Fresh", "start": 0.5, "end": 0.9}, {"word": "bit.", "start": 1.0, "end": 1.6}]


def run_job(jobs, started: dict) -> dict:
    assert set(started) == {"job_id"}, started
    job = jobs.wait(started["job_id"], timeout=120)
    assert job["status"] == "done", job["error"]
    return job["result"]


def assemble(jobs, name: str, pieces: list[dict]) -> dict:
    return run_job(jobs, tools.assemble(name, pieces, jobs=jobs))


def other_clip(folder: Path, name: str = "extra.mp4", seconds: float = 4.0) -> Path:
    """A clip that is not the talk, the same size and frame rate."""
    path = folder / name
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c=blue:size=320x240:rate=25:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=880:sample_rate=48000:duration={seconds}",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path),
        ],
        check=True,
    )
    return path


def photo(folder: Path) -> Path:
    path = folder / "loaf.png"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=red:size=64x64", "-frames:v", "1", str(path)],
        check=True,
    )
    return path


def words_of(result: dict, talk: Path) -> list[dict]:
    """The talk's words as they play in a reel, worked out from the reel's own answer."""
    out = []
    for p in result["pieces"]:
        if Path(p["clip"]) != talk.resolve():
            continue
        for w in make_words():
            if p["start"] <= (w["start"] + w["end"]) / 2 < p["end"]:
                shift = p["reel_start"] - p["start"]
                out.append({**w, "start": round(w["start"] + shift, 3), "end": round(w["end"] + shift, 3)})
    return out


def measure(video: Path, jobs) -> None:
    """Measure the word times of a video that has a transcript, with the fake aligner."""
    started = tools.transcribe(str(video), aligner=FakeAligner(), jobs=jobs, **QUIET)
    assert started["status"] == "aligning", started
    assert jobs.wait(started["job_id"], timeout=60)["status"] == "done"


def old_reel(video: Path, jobs) -> Path:
    """The four takes joined, transcribed and measured."""
    result = assemble(jobs, "bread", [{"clip": str(video), "start": a, "end": b} for a, b in TAKES])
    reel = Path(result["reel"])
    reel.with_suffix(".words.json").write_text(json.dumps(words_of(result, video)))
    measure(reel, jobs)
    return reel


def word(video: Path, text: str, nth: int = 0) -> dict:
    return [w for w in word_times.load_words(open_project(str(video))) if w["word"] == text][nth]


def pick(video: Path, text: str, nth: int = 0) -> dict:
    w = word(video, text, nth)
    return {"id": edits.pick_id(w["start"], w["end"]), "reason": "a pause word"}


def ctx_for(video: Path):
    return treatment.load_context(open_project(str(video)), silences=no_silences, labels_for=unmeasured_labels)


def edited_old_reel(video: Path, jobs) -> Path:
    """The old reel with everything a finished edit holds: Claude's cuts and picks, and what the creator did on the page."""
    reel = old_reel(video, jobs)
    tools.set_edit(
        str(reel), [FALSE_START, ACROSS, SIGN_OFF], auto_tighten=True, pace="fast",
        picks=[pick(reel, "um"), pick(reel, "for")], **QUIET,
    )
    ctx = ctx_for(reel)
    treatment.add_keep(ctx, {"start": 0.6, "end": 2.2, "note": "the hello stays"})
    treatment.add_cut(ctx, {"start": 8.7, "end": 8.95})  # "about", in the second take
    treatment.change_treatment(ctx, {"pace": "tight", "take_out": {"repeats": False}})
    first = next(r for r in treatment.page_state(ctx)["rows"] if r["reason"] == FALSE_START["reason"])
    treatment.rate_cut(ctx, {"id": first["id"], "rating": "good"})
    picture = photo(video.parent)
    overlay_tools.set_overlays(
        str(reel),
        [{"file": str(picture), "start": 0.5, "end": 2.5}, {"file": str(picture), "start": 14.0, "end": 16.0}],
        picture=False,
    )
    return reel


def new_reel(video: Path, jobs) -> tuple[Path, dict]:
    """The first three takes with another clip joined in before the third. The last take is gone."""
    extra = other_clip(video.parent)
    pieces = [{"clip": str(video), "start": a, "end": b} for a, b in TAKES[:3]]
    pieces.insert(2, {"clip": str(extra), "start": 0.0, "end": 3.0})
    result = assemble(jobs, "bread again", pieces)
    return Path(result["reel"]), result


class FakeSpeech:
    """A speech model that says the same two words whatever it hears, and notes how long each stretch was."""

    def __init__(self) -> None:
        self.heard: list[float] = []

    def __call__(self, path: Path, force: bool) -> None:
        self.heard.append(round(probe_duration(path), 2))
        path.with_suffix(".words.json").write_text(json.dumps(HEARD))


def carried(old: Path, new: Path, jobs, speech=None, **more) -> dict:
    return run_job(jobs, tools.carry_edit(
        str(old), str(new), transcriber=speech or FakeSpeech(), aligner=FakeAligner(), jobs=jobs, **QUIET, **more,
    ))


@pytest.fixture
def scene(video, jobs):
    """The edited old reel, the new reel, and the report of carrying one onto the other."""
    old = edited_old_reel(video, jobs)
    new, built = new_reel(video, jobs)
    speech = FakeSpeech()
    before = {p: p.read_bytes() for p in [*open_project(str(old)).root.rglob("*.json"), old.with_suffix(".words.json")]}
    report = carried(old, new, jobs, speech)
    return {"old": old, "new": new, "built": built, "speech": speech, "report": report, "before": before}


def said_in(video: Path, start: float, end: float) -> list[str]:
    words = word_times.load_words(open_project(str(video)))
    return [w["word"] for w in words if start < (w["start"] + w["end"]) / 2 < end]


def claude_cuts(video: Path) -> list[dict]:
    return tools.get_edit(str(video), labels=unmeasured_labels)["cuts"]


def test_the_new_transcript_is_the_old_words_shifted_and_only_the_new_clip_is_heard(scene):
    transcript = scene["report"]["transcript"]
    new_words = json.loads(scene["new"].with_suffix(".words.json").read_text())
    old_words = json.loads(scene["old"].with_suffix(".words.json").read_text())

    # Takes one and two play where they did; take three plays 3 s later; take four is gone.
    expected = [w for w in old_words if w["end"] <= 10.6]
    expected += [{**w, "start": round(w["start"] + 3.0, 3), "end": round(w["end"] + 3.0, 3)} for w in old_words if 10.6 <= w["start"] < 13.8]
    fresh = [{**w, "start": w["start"] + 10.6, "end": w["end"] + 10.6} for w in HEARD]
    assert new_words == sorted(expected + fresh, key=lambda w: w["start"])
    assert any(w.get("type") == "event" for w in new_words), "the sound in take one came along"

    assert scene["speech"].heard == [3.0], "the speech model heard the new clip and nothing else"
    assert transcript["status"] == "written"
    assert (transcript["carried_words"], transcript["word_count"]) == (len(expected), len(expected) - 1 + 2)
    assert transcript["heard"] == [{"clock": "0:10-0:13", "start": 10.6, "end": 13.6, "words": 2}]
    assert scene["report"]["word_times"] == "measured"


def test_a_cut_inside_carried_footage_lands_on_the_same_words(scene):
    old_cut = claude_cuts(scene["old"])[0]
    new_cut = claude_cuts(scene["new"])[0]
    # The picked "um" before it and the pause between them join the cut into one removal, in both videos.
    gone = ["um", "today", "we", "talk", "about", "editing", "[vocalization]"]
    assert said_in(scene["old"], old_cut["start"], old_cut["end"]) == gone
    assert said_in(scene["new"], new_cut["start"], new_cut["end"]) == gone
    assert (new_cut["start"], new_cut["end"]) == (old_cut["start"], old_cut["end"]), "take one plays where it did"
    assert (new_cut["reason"], new_cut["kind"]) == (old_cut["reason"], "false_start")
    assert new_cut["reason"].startswith(FALSE_START["reason"])
    assert scene["report"]["cuts"]["carried"] == 1


def test_a_cut_across_the_point_where_the_clip_went_in_becomes_two(scene):
    cuts = claude_cuts(scene["new"])
    assert [c["reason"].split(" (+")[0] for c in cuts] == [FALSE_START["reason"], ACROSS["reason"], ACROSS["reason"]]
    before, after = cuts[1], cuts[2]
    assert said_in(scene["new"], before["start"], before["end"]) == ["video."]
    assert said_in(scene["new"], after["start"], after["end"]) == ["this"]
    asked = json.loads(open_project(str(scene["new"])).edit_path.read_text())["requested"]
    assert [(c["start"], c["end"]) for c in asked[1:]] == [(9.72, 10.6), (13.6, 14.53)], "each half stops at the new clip"
    removed = [w for c in cuts for w in said_in(scene["new"], c["start"], c["end"])]
    assert "Fresh" not in removed and "bit." not in removed, "no word of the new clip is cut"

    split = scene["report"]["cuts"]["split"]
    assert [(s["was"]["clock"], [n["clock"] for n in s["now"]], s["reason"]) for s in split] == [
        ("0:09-0:11", ["0:09-0:10", "0:13-0:14"], ACROSS["reason"]),
    ]
    assert "in 2 parts" in split[0]["why"]


def test_a_cut_in_footage_that_is_gone_is_dropped_and_named(scene):
    report = scene["report"]["cuts"]
    assert report["total"] == 3 and report["shortened"] == [] and report["not_placed"] == []
    assert [(d["was"]["clock"], d["now"], d["reason"], d["why"]) for d in report["dropped"]] == [
        ("0:14-0:15", [], SIGN_OFF["reason"], carry.WHY_DROPPED),
    ]
    assert "watching." not in said_in(scene["new"], 0, 99)


def test_picks_are_found_again_on_the_word_they_name(scene):
    um = word(scene["new"], "um")
    saved = json.loads(open_project(str(scene["new"])).edit_path.read_text())["picks"]
    assert [(p["id"], p["text"], p["reason"]) for p in saved] == [(edits.pick_id(um["start"], um["end"]), "um", "a pause word")]
    report = scene["report"]["picks"]
    assert (report["total"], report["carried"], report["switch_on"]) == (2, 1, True)
    assert [(d["text"], d["why"]) for d in report["dropped"]] == [("for", carry.WHY_DROPPED)]
    assert tools.get_edit(str(scene["new"]), labels=unmeasured_labels)["picks"]["picked"] == 1


def test_what_the_creator_did_on_the_page_arrives(scene):
    old = tools.get_edit(str(scene["old"]), labels=unmeasured_labels)
    new = tools.get_edit(str(scene["new"]), labels=unmeasured_labels)
    creator = new["creator"]

    # Her own cut: the same word, in the second take, which plays where it did.
    assert [c["text"] for c in creator["cuts"]] == ["about"]
    assert creator["cuts"] == old["creator"]["cuts"]
    # The part she kept.
    assert [(k["text"], k["note"]) for k in creator["kept"]] == [("Hello everyone and welcome.", "the hello stays")]
    # Her pace and her switches, and what Claude had asked for.
    assert creator["pace"] == {"claude": "fast", "creator": "tight"}
    assert creator["take_out"] == {"repeats": False}
    assert new["pace"] == "tight"
    # Her rating of Claude's first cut, on the row that cut has here.
    assert [r["reason"] for r in creator["rated"]["good"]] == [FALSE_START["reason"]]
    row = next(r for r in treatment.page_state(ctx_for(scene["new"]))["rows"] if r["reason"] == FALSE_START["reason"])
    assert row["rating"] == "good"

    report = scene["report"]
    assert (report["creator_cuts"]["carried"], report["kept_parts"]["carried"], report["ratings"]["carried"]) == (1, 1, 1)
    assert report["treatment"]["pace"] == "tight" and report["treatment"]["take_out"]["repeats"] is False


def test_her_photos_and_clips_arrive_where_their_words_now_play(scene):
    saved = load_overlays(open_project(str(scene["new"])))
    assert [(o["id"], o["start"], o["end"], o["starts_on"]["word"]) for o in saved["overlays"]] == [("ov1", 0.5, 2.5, "Hello")]
    assert saved["next_id"] == 3, "an id is never handed out twice"
    report = scene["report"]["overlays"]
    assert (report["total"], report["carried"]) == (2, 1)
    assert [(d["id"], d["file"], d["why"]) for d in report["dropped"]] == [("ov2", "loaf.png", carry.WHY_DROPPED)]


def test_the_report_gives_the_new_length_and_the_join_check(scene):
    report = scene["report"]
    new = tools.get_edit(str(scene["new"]), check=True, labels=unmeasured_labels)
    assert report["status"] == "carried"
    assert (report["from_video"], report["to_video"]) == (str(scene["old"]), str(scene["new"]))
    assert report["duration"] == pytest.approx(16.8, abs=0.05)
    assert (report["new_duration"], report["removed_seconds"], report["cut_count"]) == (
        new["new_duration"], new["removed_seconds"], new["cut_count"],
    )
    assert report["joins"]["joins"] == new["joins"]["joins"] and "rows" not in report["joins"]
    assert "joins" in report["note"]


def test_nothing_of_the_old_video_changes(scene):
    for path, content in scene["before"].items():
        assert path.read_bytes() == content, path.name
    assert not (open_project(str(scene["old"])).root / taste.CARRIED_FILE).exists()


def test_the_words_a_creator_may_hear_read_plainly(scene):
    lines = [
        carry.WHY_DROPPED, carry.WHY_SHORTENED, carry.WHY_TWICE, carry.WHY_SPLIT.format(n=2), carry.WHY_NO_WORD,
        carry.WHY_OTHER_WORD, carry.WHY_LONGEST_PART, carry.WHY_TOO_SHORT, carry.WHY_NOT_READ,
        carry.TRANSCRIPT_WAS_THERE, carry.OVERLAYS_WERE_THERE,
    ]
    for group in ("cuts", "picks", "creator_cuts", "kept_parts", "overlays"):
        for how in ("split", "shortened", "dropped"):
            lines += [entry["why"] for entry in scene["report"][group][how]]
    for line in lines:
        assert not banned_in(line) and not decimal_times_in(line) and chr(0x2014) not in line, line


def test_a_change_the_creator_made_counts_once_after_it_is_carried(scene, video):
    """The old reel's project stops teaching once its edit lives on in the new one."""
    another = video.with_name("another.mp4")
    another.write_bytes(video.read_bytes())
    another.with_suffix(".words.json").write_text(json.dumps(make_words()))
    learned = tools.get_edit(str(another), labels=unmeasured_labels)["taste"]
    assert learned["videos"] == 1
    assert learned["cut_by_hand"] == [["about", 1]]


def test_what_she_asked_to_forget_stays_forgotten_after_it_is_carried(video, jobs):
    """Carried changes get new ids in a new folder; the note of what to forget goes with them."""
    old = edited_old_reel(video, jobs)
    taste.forget_all()
    another = video.with_name("another.mp4")
    another.write_bytes(video.read_bytes())
    another.with_suffix(".words.json").write_text(json.dumps(make_words()))
    assert "taste" not in tools.get_edit(str(another), labels=unmeasured_labels)
    new, _ = new_reel(video, jobs)

    carried(old, new, jobs)

    assert "taste" not in tools.get_edit(str(another), labels=unmeasured_labels), "the carry brought nothing back"
    # A change she makes on the new reel after that counts, as one made after forgetting always does.
    treatment.add_cut(ctx_for(new), {"start": 0.95, "end": 1.5})  # "everyone"
    learned = tools.get_edit(str(another), labels=unmeasured_labels)["taste"]
    assert learned["videos"] == 1 and learned["cut_by_hand"] == [["everyone", 1]]


def test_a_carry_with_nothing_forgotten_writes_no_note_of_it(scene):
    assert not taste.taste_path().exists()


# ── after a move ──────────────────────────────────────────────────────────────


def test_the_edit_of_a_video_that_moved_since_is_carried(video, jobs, tmp_path):
    """The old video's project sits in a folder named for the path it had; it is found as every tool finds it."""
    tools.set_edit(str(video), [FALSE_START], **QUIET)
    made_for = open_project(str(video)).root
    there = tmp_path / "elsewhere"
    there.mkdir()
    moved = there / video.name
    shutil.move(video, moved)
    shutil.move(video.with_suffix(".words.json"), moved.with_suffix(".words.json"))
    built = assemble(jobs, "moved since", [{"clip": str(moved)}, {"clip": str(other_clip(there)), "start": 0.0, "end": 3.0}])
    reel = Path(built["reel"])

    report = carried(moved, reel, jobs)

    assert report["cuts"]["carried"] == 1 and report["transcript"]["carried_words"] == len(make_words())
    cut = claude_cuts(reel)[0]
    assert said_in(reel, cut["start"], cut["end"]) == ["today", "we", "talk", "about", "editing", "[vocalization]"]
    record = json.loads((open_project(str(reel)).root / taste.CARRIED_FILE).read_text())
    assert record["from_project"] == made_for.name, "the project that stops teaching is the one that holds the old edit"


def test_a_reel_copied_to_another_mac_is_carried_onto_a_reel_joined_there(video, jobs, tmp_path, monkeypatch):
    """The old reel's sequence file names its clips where they were; the new reel is joined from where they are."""
    old = edited_old_reel(video, jobs)
    gone = [said_in(old, c["start"], c["end"]) for c in claude_cuts(old)]
    home, projects = Path(os.environ["LUMR_HOME"]), Path(os.environ["LUMR_STUDIO_PROJECTS_DIR"])
    there = tmp_path / "other-mac"
    (there / "footage").mkdir(parents=True)
    shutil.move(home, there / "lumr-home")
    shutil.move(projects, there / "projects")
    clip = there / "footage" / video.name
    shutil.move(video, clip)
    monkeypatch.setenv("LUMR_HOME", str(there / "lumr-home"))
    monkeypatch.setenv("LUMR_STUDIO_PROJECTS_DIR", str(there / "projects"))
    old_there = there / "lumr-home" / old.relative_to(home)
    new, _ = new_reel(clip, jobs)

    report = carried(old_there, new, jobs)

    # As on the first Mac: the first cut whole, the one across the new clip in two, the last take's dropped.
    cuts = report["cuts"]
    assert (cuts["total"], cuts["carried"], len(cuts["split"]), len(cuts["dropped"])) == (3, 1, 1, 1)
    assert [said_in(new, c["start"], c["end"]) for c in claude_cuts(new)][0] == gone[0]
    assert report["creator_cuts"]["carried"] == 1 and report["transcript"]["status"] == "written"


# ── refusals, and the other ways in ───────────────────────────────────────────


def test_a_new_video_that_already_has_an_edit_is_refused(scene, video, jobs):
    with pytest.raises(StudioError, match="already has a saved edit.*never replaces.*another name"):
        tools.carry_edit(str(scene["old"]), str(scene["new"]), jobs=jobs, **QUIET)
    # The other way round is refused too: the old reel has its edit.
    with pytest.raises(StudioError, match="already has a saved edit"):
        tools.carry_edit(str(scene["new"]), str(scene["old"]), jobs=jobs, **QUIET)


def test_a_video_with_no_edit_has_nothing_to_carry(video, jobs):
    old = old_reel(video, jobs)
    new, _ = new_reel(video, jobs)
    with pytest.raises(StudioError, match="has no saved edit, so there is nothing to carry"):
        tools.carry_edit(str(old), str(new), jobs=jobs, **QUIET)
    assert not open_project(str(new)).edit_path.exists() and not new.with_suffix(".words.json").exists()


def test_videos_that_share_no_footage_are_refused(video, jobs):
    tools.set_edit(str(video), [FALSE_START], **QUIET)
    extra = other_clip(video.parent)
    with pytest.raises(StudioError, match="share no footage"):
        tools.carry_edit(str(video), str(extra), jobs=jobs, **QUIET)
    with pytest.raises(StudioError, match="the same video"):
        tools.carry_edit(str(video), str(video), jobs=jobs, **QUIET)
    assert not extra.with_suffix(".words.json").exists()


def test_a_plain_video_is_carried_into_a_reel_that_holds_it(video, jobs):
    tools.set_edit(str(video), [FALSE_START], picks=[pick(video, "um")], **QUIET)
    extra = other_clip(video.parent)
    built = assemble(jobs, "with an opening", [{"clip": str(extra), "start": 0.0, "end": 3.0}, {"clip": str(video)}])
    reel = Path(built["reel"])
    speech = FakeSpeech()

    report = carried(video, reel, jobs, speech)

    assert speech.heard == [3.0]
    assert report["transcript"]["carried_words"] == len(make_words())
    assert (report["cuts"]["carried"], report["picks"]["carried"]) == (1, 1)
    cut = claude_cuts(reel)[0]
    assert said_in(reel, cut["start"], cut["end"]) == ["today", "we", "talk", "about", "editing", "[vocalization]"]
    assert cut["start"] > 6.0, "the cut moved with the talk, 3 s later"
    moved = json.loads(reel.with_suffix(".words.json").read_text())
    assert [w for w in moved if w["word"] == "um"] == [{"word": "um", "start": 6.4, "end": 6.7, "energy_rms": 0.1}]


def test_a_half_that_no_longer_removes_a_whole_word_is_named_as_not_placed(video, jobs):
    """The cut took "video." and ran into the start of "this". Split, its second half only clips "this"."""
    old = old_reel(video, jobs)
    clipping = {"start": 9.72, "end": 11.3, "reason": "Trails off", "kind": "other"}
    tools.set_edit(str(old), [clipping], **QUIET)
    new, _ = new_reel(video, jobs)

    report = carried(old, new, jobs)["cuts"]

    assert [said_in(new, c["start"], c["end"]) for c in claude_cuts(new)] == [["video."]]
    assert (report["total"], report["carried"], len(report["split"])) == (1, 0, 1)
    [refused] = report["not_placed"]
    assert refused["now"] == [{"clock": "0:13-0:14", "start": 13.6, "end": 14.3}]
    assert refused["reason"] == "Trails off" and refused["why"].startswith("Removes no whole word: it only clips 'this'")


def test_a_clip_with_a_transcript_of_its_own_is_not_heard_again(video, jobs):
    """Nothing that was transcribed once is transcribed again: the new clip's own words are used."""
    old = edited_old_reel(video, jobs)
    new, _ = new_reel(video, jobs)
    (video.parent / "extra.words.json").write_text(json.dumps(HEARD + [{"word": "later", "start": 3.2, "end": 3.6}]))
    speech = FakeSpeech()

    report = carried(old, new, jobs, speech)

    assert speech.heard == []
    assert (report["transcript"]["words_from_clips"], report["transcript"]["heard"]) == (2, [])
    assert said_in(new, 10.6, 13.6) == ["Fresh", "bit."]


def test_a_transcript_that_is_already_there_is_left_and_said(video, jobs):
    old = edited_old_reel(video, jobs)
    new, built = new_reel(video, jobs)
    own = words_of(built, video)
    new.with_suffix(".words.json").write_text(json.dumps(own))
    speech = FakeSpeech()

    report = carried(old, new, jobs, speech)

    assert json.loads(new.with_suffix(".words.json").read_text()) == own and speech.heard == []
    assert report["transcript"] == {
        "status": "exists", "words_path": str(new.with_suffix(".words.json")), "note": carry.TRANSCRIPT_WAS_THERE,
    }
    assert report["cuts"]["carried"] == 1 and report["picks"]["carried"] == 1


def test_a_missing_model_is_asked_for_before_anything_starts(video, jobs, tmp_path):
    old = edited_old_reel(video, jobs)
    new, _ = new_reel(video, jobs)
    both = tiny_set(tmp_path / "models")

    answer = tools.carry_edit(str(old), str(new), jobs=jobs, model_set=both, aligner=FakeAligner(), **QUIET)

    assert answer["status"] == "needs_models"
    assert [m["name"] for m in answer["models"]] == ["Tiny speech model", "Tiny aligner"]
    assert "Nothing has downloaded" in answer["note"] and "call carry_edit again" in answer["note"]
    assert set(answer) == {"status", "models", "total_mb", "note"}
    assert not new.with_suffix(".words.json").exists() and not open_project(str(new)).edit_path.exists()


def test_only_the_models_the_carry_needs_are_asked_for(video, jobs, tmp_path):
    """With every stretch covered by words there is nothing to hear, so the speech model is not needed."""
    old = edited_old_reel(video, jobs)
    built = assemble(jobs, "bread shorter", [{"clip": str(video), "start": a, "end": b} for a, b in TAKES[:3]])
    both = tiny_set(tmp_path / "models")

    answer = tools.carry_edit(str(old), built["reel"], jobs=jobs, model_set=both, aligner=FakeAligner(), **QUIET)

    assert [m["name"] for m in answer["models"]] == ["Tiny aligner"]


def test_a_second_call_while_the_job_runs_gets_the_same_job(video, jobs):
    old = edited_old_reel(video, jobs)
    new, _ = new_reel(video, jobs)
    release = threading.Event()

    def slow(path, force):
        release.wait(30)
        path.with_suffix(".words.json").write_text(json.dumps(HEARD))

    first = tools.carry_edit(str(old), str(new), transcriber=slow, aligner=FakeAligner(), jobs=jobs, **QUIET)
    second = tools.carry_edit(str(old), str(new), transcriber=slow, aligner=FakeAligner(), jobs=jobs, **QUIET)
    release.set()
    assert second == first
    assert jobs.wait(first["job_id"], timeout=60)["status"] == "done"
