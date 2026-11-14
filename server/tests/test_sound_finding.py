"""Sounds between the words are found when a video is transcribed, and no automatic cut takes a laugh.

Every take here is made up: tones for words, a train of pitched bursts for a
giggle, a held tone for an "uhh", faint noise for the room.
"""

import json
import shutil
import subprocess
import wave

import numpy as np
import pytest

from conftest import NO_MODEL_IN_TESTS
from lumr_studio import edit as edits
from lumr_studio import tools, word_times
from lumr_studio.aligner import Remembering
from lumr_studio.errors import StudioError
from lumr_studio.joins import check_joins
from lumr_studio.pace import PACES
from lumr_studio.project import open_project
from lumr_studio.silences import no_silences
from lumr_studio.sounds import (
    FIND_WINDOW_SECONDS,
    FOUND_FILE,
    NONE_FOUND,
    NOT_CHECKED,
    SAMPLE_RATE,
    find_in_gaps,
    find_sounds,
    is_filler,
    loudness_of,
    sounds_checked,
    whole_spans,
)

RNG = np.random.default_rng(11)
TAKE_SECONDS = 14.0

# The made-up take: (word, start, end). A giggle after "plan.", then "um" right behind it, a word
# said twice, a silent pause after "on.", and an "uhh" after "lunch".
SAID = [
    ("So", 0.5, 0.8), ("that", 0.85, 1.15), ("was", 1.2, 1.45), ("the", 1.5, 1.7), ("plan.", 1.75, 2.3),
    ("um", 5.5, 5.8), ("Anyway", 6.0, 6.5), ("we", 6.55, 6.75), ("we", 6.8, 7.0), ("moved", 7.05, 7.4),
    ("on.", 7.45, 7.8),
    ("Next", 9.6, 10.0), ("we", 10.05, 10.2), ("got", 10.25, 10.55), ("lunch", 10.6, 11.0),
    ("and", 12.2, 12.45), ("left.", 12.5, 13.0),
]
GIGGLE = (2.7, 5.3)
UHH = (11.3, 11.8)
QUIET_PAUSE = (7.8, 9.6)
EVERY_SWITCH = ["pauses", "fillers", "repeats"]


def said() -> list[dict]:
    """The transcript the speech model would write for the take: words only, no sounds."""
    return [{"word": word, "start": start, "end": end} for word, start, end in SAID]


def tone(seconds: float, level: float, pitch: float = 200.0) -> np.ndarray:
    t = np.arange(round(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    voice = sum(np.sin(2 * np.pi * pitch * k * t) / k for k in (1, 2, 3))
    return level * voice / np.abs(voice).max()


def giggle(seconds: float) -> np.ndarray:
    """Five pitched bursts a second with the voice dropping, not stopping, between them: voiced all the way."""
    burst = np.concatenate([tone(0.1, 0.5, pitch=320.0), tone(0.1, 0.08, pitch=320.0)])
    return np.tile(burst, round(seconds / 0.2))


def take_audio() -> np.ndarray:
    audio = 0.003 * RNG.uniform(-1, 1, round(TAKE_SECONDS * SAMPLE_RATE))  # the room
    for _word, start, end in SAID:
        at = round(start * SAMPLE_RATE)
        voice = tone(end - start, 0.5)
        audio[at:at + len(voice)] += voice
    for (start, end), sound in ((GIGGLE, giggle(GIGGLE[1] - GIGGLE[0])), (UHH, tone(UHH[1] - UHH[0], 0.3, pitch=180.0))):
        at = round(start * SAMPLE_RATE)
        audio[at:at + len(sound)] += sound
    return audio


@pytest.fixture(scope="module")
def take_master(tmp_path_factory):
    """The made-up take as a video file, made once: a black picture over the audio."""
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    folder = tmp_path_factory.mktemp("giggle")
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


@pytest.fixture
def bare_take(tmp_path, take_master):
    """The take with no transcript yet."""
    folder = tmp_path / "footage"
    folder.mkdir()
    shutil.copy(take_master, folder / "take.mp4")
    return folder / "take.mp4"


@pytest.fixture
def take(bare_take):
    """The take with the transcript an earlier plugin wrote: its words, and not one sound."""
    bare_take.with_suffix(".words.json").write_text(json.dumps(said()))
    return bare_take


class SameTimes:
    """An aligner that hears every word where the transcript has it, and counts its runs."""

    def __init__(self) -> None:
        self.calls = 0

    def lacks(self) -> str:
        return ""

    def align(self, video, words, silences):
        self.calls += 1
        return [dict(w) for w in words]


def never_transcribes(_video, _force):
    raise AssertionError("the speech model must not run for a transcript that is there")


def sounds_in(video) -> list[dict]:
    return [w for w in json.loads(video.with_suffix(".words.json").read_text()) if w.get("type") == "event"]


def taken(video, span: tuple[float, float]) -> float:
    """Seconds of ``span`` the saved edit removes."""
    cuts = json.loads(open_project(str(video)).edit_path.read_text())["cuts"]
    return round(sum(max(0.0, min(c["end"], span[1]) - max(c["start"], span[0])) for c in cuts), 3)


def tighten(video, pace: str) -> dict:
    result = tools.set_edit(str(video), [], auto_tighten=True, pace=pace, silences=no_silences)
    assert result["auto"]["take_out"] == EVERY_SWITCH
    return result


# ── Finding, in a transcribe job ──────────────────────────────────────────────


def test_a_giggle_in_a_gap_becomes_a_sound_of_the_transcript_and_reads_as_a_laugh(bare_take, jobs):
    aligner = SameTimes()

    def speech_model(path, force):
        path.with_suffix(".words.json").write_text(json.dumps(said()))

    job = tools.transcribe(str(bare_take), transcriber=speech_model, aligner=aligner, silences=no_silences, jobs=jobs)
    done = jobs.wait(job["job_id"], timeout=60)
    assert done["status"] == "done", done
    assert done["result"]["sounds"] == {"found": 2, "on": "measured"}
    assert done["result"]["word_times"] == "measured" and done["result"]["aligning"]["sounds"] == 2
    assert aligner.calls == 1, "the words are measured again around the sounds without a second run of the model"

    found = sounds_in(bare_take)
    assert [(e["word"], e["type"], e["found_by"]) for e in found] == [("[vocalization]", "event", "loudness")] * 2
    assert (found[0]["start"], found[0]["end"]) == (pytest.approx(GIGGLE[0], abs=0.08), pytest.approx(GIGGLE[1], abs=0.08))
    assert (found[1]["start"], found[1]["end"]) == (pytest.approx(UHH[0], abs=0.08), pytest.approx(UHH[1], abs=0.08))
    assert not [e for e in found if e["start"] < QUIET_PAUSE[1] and e["end"] > QUIET_PAUSE[0]], "room tone is no sound"

    project = open_project(str(bare_take))
    labels = tools.measured_labels(project, word_times.load_words(project))
    assert (labels[0]["kind"], labels[0]["confidence"]) == ("laugh", "likely")
    assert labels[0]["voiced"] > 0.5, "this giggle is pitched all the way: the old rule called it a hesitation"
    assert labels[1]["kind"] == "sound" and is_filler(labels[1])
    read = tools.read_transcript(str(bare_take))
    assert "(laugh 2." in read["text"] and "sounds_note" not in read
    assert tools.transcribe(str(bare_take), aligner=aligner, silences=no_silences, jobs=jobs)["status"] == "exists"


def test_an_old_transcript_gains_its_sounds_without_the_speech_model_and_its_edit_survives(take, jobs):
    project = open_project(str(take))
    aligner = SameTimes()
    word_times.align_project(project, aligner=aligner, silences=no_silences)  # measured before sounds were found
    cut = {"start": 12.1, "end": 13.1, "reason": "the ending drags"}
    tools.set_edit(str(take), [cut], auto_tighten=True, pace="max", silences=no_silences)
    assert taken(take, GIGGLE) > 1.0, "what went wrong before: with no sound marked the giggle went as a pause"
    assert not sounds_checked(project)

    answer = tools.transcribe(str(take), transcriber=never_transcribes, aligner=aligner, silences=no_silences, jobs=jobs)
    assert answer["status"] == "finding_sounds" and "Nothing is transcribed again" in answer["note"]
    done = jobs.wait(answer["job_id"], timeout=60)
    assert done["status"] == "done", done
    assert done["result"]["sounds"] == {"found": 2, "on": "measured"} and done["result"]["word_times"] == "measured"
    assert aligner.calls == 1, "measured again from the times it had: no model ran"
    assert len(sounds_in(take)) == 2 and sounds_checked(project)

    # The saved edit was made again: Claude's cut is still in it, and the giggle is back.
    assert "laughs and other sounds are marked" in done["result"]["edit"]["note"]
    saved = tools.get_edit(str(take), check=True)
    assert [c["reason"] for c in saved["cuts"]] == ["the ending drags"]
    assert taken(take, GIGGLE) == 0
    assert NOT_CHECKED not in saved["joins"]["text"] and "sounds_note" not in saved
    again = tools.transcribe(str(take), transcriber=never_transcribes, aligner=aligner, silences=no_silences, jobs=jobs)
    assert again["status"] == "exists" and "job_id" not in again


def test_at_every_pace_with_every_switch_on_nothing_removed_overlaps_the_giggle(take, jobs):
    aligner = SameTimes()
    job = tools.transcribe(str(take), aligner=aligner, silences=no_silences, jobs=jobs)
    assert job["status"] == "aligning" and jobs.wait(job["job_id"], timeout=60)["status"] == "done"
    project = open_project(str(take))
    labels = tools.measured_labels(project, word_times.load_words(project))
    laugh = whole_spans(labels)
    assert len(laugh) == 1 and laugh[0] == (pytest.approx(GIGGLE[0], abs=0.08), pytest.approx(GIGGLE[1], abs=0.08))
    for pace in PACES:
        result = tighten(take, pace)
        assert taken(take, laugh[0]) == 0, pace
        assert taken(take, GIGGLE) == 0, pace
    # The hardest pace did cut around it: the word said twice, the silent pause, and the "uhh", which
    # the Filler words switch takes whole. A held hesitation is the one sound a rule may cut.
    assert result["removed_seconds"] > 2.0
    assert taken(take, QUIET_PAUSE) > 1.0
    assert taken(take, UHH) == pytest.approx(UHH[1] - UHH[0], abs=0.1)


def test_room_tone_and_a_steady_tone_give_no_sound(take, synthetic_master, tmp_path):
    project = open_project(str(take))
    find_sounds(project)
    assert not [e for e in sounds_in(take) if e["start"] < QUIET_PAUSE[1] and e["end"] > QUIET_PAUSE[0]]
    # The shared fixture's video is one 440 Hz tone from end to end: nothing stands out of it.
    tone_video = tmp_path / "tone.mp4"
    shutil.copy(synthetic_master, tone_video)
    tone_video.with_suffix(".words.json").write_text(json.dumps(said()))
    before = tone_video.with_suffix(".words.json").read_text()
    assert find_sounds(open_project(str(tone_video))) == {"found": 0, "on": "estimated", "changed": False}
    assert tone_video.with_suffix(".words.json").read_text() == before, "with nothing found the transcript is not written again"


def test_a_machine_that_cannot_measure_word_times_still_finds_the_sounds_then_looks_again_once_it_can(take, jobs):
    answer = tools.transcribe(str(take), jobs=jobs)
    assert answer["status"] == "exists" and NO_MODEL_IN_TESTS in answer["word_times_note"]
    assert answer["sounds"] == {"found": 2, "on": "estimated"} and len(sounds_in(take)) == 2
    assert "sounds" not in tools.transcribe(str(take), jobs=jobs), "one look is enough while nothing changed"

    aligner = SameTimes()
    job = tools.transcribe(str(take), aligner=aligner, silences=no_silences, jobs=jobs)
    assert job["status"] == "aligning"
    done = jobs.wait(job["job_id"], timeout=60)
    assert done["result"]["sounds"] == {"found": 2, "on": "measured"}, "measured gaps are read again"
    assert len(sounds_in(take)) == 2, "what the first look found is replaced, not added to"
    assert tools.transcribe(str(take), aligner=aligner, silences=no_silences, jobs=jobs)["status"] == "exists"


def test_a_video_whose_sound_cannot_be_read_keeps_its_words_and_says_so(take, jobs):
    def broken(_project):
        raise StudioError("ffmpeg could not read the sound of take.mp4.")

    answer = tools.transcribe(str(take), jobs=jobs, finder=broken)
    assert answer["status"] == "exists" and answer["word_count"] == len(SAID)
    assert "could not read the sound" in answer["sounds_note"] and "automatic cut can take one" in answer["sounds_note"]
    assert not sounds_checked(open_project(str(take)))


def test_an_edit_carried_onto_a_transcript_nobody_looked_through_is_placed_after_the_look(take, jobs):
    """A new reel that already has a transcript and measured times, both from before sounds were found.

    The carried edit is made at the same pace on that reel. Placed before the
    look, its pause cuts took the giggle as the pause it reads as.
    """
    aligner = SameTimes()
    job = tools.transcribe(str(take), aligner=aligner, silences=no_silences, jobs=jobs)
    assert jobs.wait(job["job_id"], timeout=60)["status"] == "done"
    tools.set_edit(str(take), [{"start": 12.1, "end": 13.1, "reason": "the ending drags"}], auto_tighten=True, pace="max", silences=no_silences)
    halves = [{"clip": str(take), "start": 0.0, "end": 8.0}, {"clip": str(take), "start": 8.0, "end": TAKE_SECONDS}]
    built = jobs.wait(tools.assemble("in two halves", halves, jobs=jobs)["job_id"], timeout=60)["result"]
    reel = open_project(built["reel"])
    reel.words_path.write_text(json.dumps(said()))
    word_times.align_project(reel, aligner=aligner, silences=no_silences)
    assert word_times.source_of(reel) == "measured" and not sounds_checked(reel)

    started = tools.carry_edit(str(take), built["reel"], transcriber=never_transcribes, aligner=aligner, silences=no_silences, jobs=jobs)
    done = jobs.wait(started["job_id"], timeout=60)
    assert done["status"] == "done", done
    report = done["result"]

    assert report["transcript"]["status"] == "exists" and report["sounds"] == {"found": 2, "on": "measured"}
    assert sounds_checked(reel) and NOT_CHECKED not in report["joins"]["text"]
    assert taken(reel.video, GIGGLE) == 0, "the giggle plays whole in the carried edit"
    assert report["cuts"]["carried"] == 1
    saved = json.loads(reel.edit_path.read_text())["cuts"]
    again = tools.transcribe(built["reel"], transcriber=never_transcribes, aligner=aligner, silences=no_silences, jobs=jobs)
    assert again["status"] == "exists", "nothing is left for transcribe to find"
    assert json.loads(reel.edit_path.read_text())["cuts"] == saved


# ── What Claude is told ───────────────────────────────────────────────────────


def test_a_transcript_nobody_looked_through_says_so_wherever_sounds_are_read(take, jobs):
    assert tools.get_edit(str(take))["sounds_note"] == NOT_CHECKED
    assert tools.read_transcript(str(take))["sounds_note"] == NOT_CHECKED
    assert tools.analyze_take(str(take), silences=no_silences)["sounds_note"] == NOT_CHECKED
    made = tools.set_edit(str(take), [], auto_tighten=True, pace="standard", silences=no_silences)
    assert made["sounds_note"] == NOT_CHECKED and NOT_CHECKED in made["joins"]["text"]
    assert "transcribe" in NOT_CHECKED

    tools.transcribe(str(take), jobs=jobs)  # this machine can't measure word times: the look happens on the spot
    for answer in (
        tools.get_edit(str(take), check=True), tools.read_transcript(str(take)),
        tools.analyze_take(str(take), silences=no_silences),
    ):
        assert "sounds_note" not in answer
    assert NOT_CHECKED not in tools.get_edit(str(take), check=True)["joins"]["text"]


def test_the_join_check_tells_none_found_from_nobody_looked():
    words = said()
    assert NOT_CHECKED in check_joins([], words, TAKE_SECONDS, labels=[], sounds_checked=False)["text"]
    assert NONE_FOUND in check_joins([], words, TAKE_SECONDS, labels=[], sounds_checked=True)["text"]
    label = {"start": 2.7, "end": 5.3, "seconds": 2.6, "kind": "laugh", "confidence": "likely", "punchline": None}
    sure = check_joins([], words, TAKE_SECONDS, labels=[label], sounds_checked=True)["text"]
    assert NOT_CHECKED not in sure and NONE_FOUND not in sure
    # Labels on a transcript nobody looked through (sounds the speech model wrote) still get the warning.
    assert NOT_CHECKED in check_joins([], words, TAKE_SECONDS, labels=[label], sounds_checked=False)["text"]


# ── The finder itself ─────────────────────────────────────────────────────────

ROOM, VOICE = -58.0, -12.0


def levels_of(seconds: float, loud: list[tuple[float, float, float]]) -> np.ndarray:
    """A take's loudness: the room, with each ``(start, end, level)`` over it."""
    levels = np.full(round(seconds / FIND_WINDOW_SECONDS), ROOM)
    for start, end, level in loud:
        levels[round(start / FIND_WINDOW_SECONDS):round(end / FIND_WINDOW_SECONDS)] = level
    return levels


def two_words() -> list[dict]:
    return [{"word": "Before", "start": 1.0, "end": 2.0}, {"word": "after.", "start": 8.0, "end": 9.0}]


def spans(found: list[dict]) -> list[tuple[float, float]]:
    return [(e["start"], e["end"]) for e in found]


def test_a_sound_runs_from_its_first_loud_window_to_its_last_with_a_little_room():
    words = two_words()
    levels = levels_of(10, [(1.0, 2.0, VOICE), (8.0, 9.0, VOICE), (3.0, 4.5, -10.0)])
    found = find_in_gaps(levels, words)
    assert spans(found) == [(2.96, 4.54)]
    assert found[0]["over_speech_db"] == 2.0, "its loudest window stands 2 dB over the words"


def test_a_sound_never_reaches_past_the_gap_it_sits_in():
    levels = levels_of(10, [(1.0, 2.0, VOICE), (8.0, 9.0, VOICE), (2.0, 2.5, -15.0), (7.5, 8.0, -15.0)])
    assert spans(find_in_gaps(levels, two_words())) == [(2.0, 2.54), (7.46, 8.0)]


def test_bouts_under_a_second_apart_are_one_sound_and_further_apart_two():
    words = two_words()
    close = levels_of(10, [(1.0, 2.0, VOICE), (8.0, 9.0, VOICE), (3.0, 3.6, -14.0), (4.4, 5.0, -14.0)])
    assert spans(find_in_gaps(close, words)) == [(2.96, 5.04)]
    apart = levels_of(10, [(1.0, 2.0, VOICE), (8.0, 9.0, VOICE), (3.0, 3.6, -14.0), (4.8, 5.4, -14.0)])
    assert spans(find_in_gaps(apart, words)) == [(2.96, 3.64), (4.76, 5.44)]


def test_a_click_a_breath_and_the_room_are_no_sound():
    words = two_words()
    click = levels_of(10, [(1.0, 2.0, VOICE), (8.0, 9.0, VOICE), (3.0, 3.2, -8.0)])
    assert find_in_gaps(click, words) == [], "0.2 s over the line is too short to be a voice"
    breath = levels_of(10, [(1.0, 2.0, VOICE), (8.0, 9.0, VOICE), (3.0, 4.0, -38.0)])
    assert find_in_gaps(breath, words) == [], "26 dB under the words is a breath"
    assert find_in_gaps(levels_of(10, [(1.0, 2.0, VOICE), (8.0, 9.0, VOICE)]), words) == []


def test_in_a_loud_room_a_sound_must_stand_clear_of_the_noise():
    words = two_words()
    loud = [(1.0, 2.0, VOICE), (8.0, 9.0, VOICE)]
    noisy = np.maximum(levels_of(10, loud), -22.0)  # the room hums 10 dB under the words
    noisy[round(3.0 / FIND_WINDOW_SECONDS):round(4.0 / FIND_WINDOW_SECONDS)] = -17.0  # 5 dB over the hum
    assert find_in_gaps(noisy, words) == []
    noisy[round(3.0 / FIND_WINDOW_SECONDS):round(4.0 / FIND_WINDOW_SECONDS)] = -10.0
    assert spans(find_in_gaps(noisy, words)) == [(2.96, 4.04)]
    assert find_in_gaps(np.full(500, VOICE), words) == [], "one level from end to end has nothing standing out"


def test_only_gaps_between_two_entries_are_read_and_a_sound_already_there_is_left_alone():
    words = two_words()
    edges = levels_of(10, [(1.0, 2.0, VOICE), (8.0, 9.0, VOICE), (0.2, 0.8, -10.0), (9.2, 9.8, -10.0)])
    assert find_in_gaps(edges, words) == [], "before the first word and after the last the camera is being handled"
    theirs = {"word": "[vocalization]", "start": 3.0, "end": 4.5, "type": "event"}
    levels = levels_of(10, [(1.0, 2.0, VOICE), (8.0, 9.0, VOICE), (3.0, 4.5, -10.0), (5.6, 6.2, -10.0)])
    assert spans(find_in_gaps(levels, words + [theirs])) == [(5.56, 6.24)]
    assert find_in_gaps(levels, []) == [] and find_in_gaps(np.zeros(0), words) == []


def test_the_loudness_of_a_take_is_one_level_for_every_fiftieth_of_a_second(take, tmp_path):
    levels = loudness_of(take)
    assert len(levels) == pytest.approx(TAKE_SECONDS / FIND_WINDOW_SECONDS, abs=5)
    at = lambda t: levels[round(t / FIND_WINDOW_SECONDS)]
    assert at(8.5) < -45 and at(0.65) > -20 and at(3.14) > -20  # the room, a word, a burst of the giggle
    broken = tmp_path / "broken.mp4"
    broken.write_text("not a video")
    with pytest.raises(StudioError, match="Check that the source video plays"):
        loudness_of(broken)


def test_the_loudness_keeps_time_with_the_video_where_its_sound_has_a_hole(tmp_path):
    # A reel copied together from clips has a hole in its sound at every join. Here: one second gone.
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    holed = tmp_path / "holed.mka"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=3",
            "-af", "aselect='not(between(t,1,2))'", "-c:a", "pcm_s16le", str(holed),
        ],
        check=True,
    )
    levels = loudness_of(holed)
    at = lambda t: levels[round(t / FIND_WINDOW_SECONDS)]
    assert len(levels) == pytest.approx(3 / FIND_WINDOW_SECONDS, abs=2), "counted in samples it would be 2 s long"
    assert at(0.5) > -30 and at(1.5) < -80 and at(2.5) > -30, "what sounds at 2.5 s is found at 2.5 s"


def test_a_look_writes_the_transcript_once_and_marks_it_looked_through(take):
    project = open_project(str(take))
    assert find_sounds(project) == {"found": 2, "on": "estimated", "changed": True}
    written = take.with_suffix(".words.json").read_text()
    assert [w["word"] for w in json.loads(written) if w.get("type") != "event"] == [w for w, _, _ in SAID]
    assert json.loads((project.root / FOUND_FILE).read_text())["found"] == 2 and sounds_checked(project)
    assert find_sounds(project) == {"found": 2, "on": "estimated", "changed": False}
    assert take.with_suffix(".words.json").read_text() == written
    # A new transcript was never looked through, whatever the old one's marker says.
    take.with_suffix(".words.json").write_text(json.dumps(said()[:-1]))
    assert not sounds_checked(project)


# ── Claude's cuts and the last check ──────────────────────────────────────────


def talk_with_a_laugh() -> list[dict]:
    words = [
        {"word": "the", "start": 0.5, "end": 0.9}, {"word": "joke.", "start": 1.0, "end": 1.8},
        {"word": "[vocalization]", "start": 2.0, "end": 5.0, "type": "event"},
        {"word": "So", "start": 5.4, "end": 5.8}, {"word": "anyway", "start": 5.9, "end": 6.5},
        {"word": "next.", "start": 7.5, "end": 8.0},
    ]
    return words


LAUGH = (2.0, 5.0)


def test_a_cut_of_claudes_that_reaches_into_a_laugh_stops_at_it():
    words = talk_with_a_laugh()
    # Asked from 3.0: past the laugh's start, before its middle. The midpoint rule takes the whole laugh.
    assert edits.place_cut(3.0, 7.0, words, 10.0)[0] <= LAUGH[0]
    start, end = edits.place_cut(3.0, 7.0, words, 10.0, whole=[LAUGH])
    assert start >= LAUGH[1] and end > 6.5, "the laugh stays whole and the words after it still go"
    # Asked up to 4.5: past its middle from the other side.
    start, end = edits.place_cut(0.95, 4.5, words, 10.0, whole=[LAUGH])
    assert end <= LAUGH[0] and start < 1.0
    # A cut inside the laugh alone clips it and takes nothing whole: refused, as before.
    with pytest.raises(edits.CutRejected, match="removes no whole word"):
        edits.place_cut(2.5, 4.8, words, 10.0, whole=[LAUGH])


def test_a_cut_asked_for_over_the_whole_laugh_still_takes_it():
    words = talk_with_a_laugh()
    start, end = edits.place_cut(1.9, 7.0, words, 10.0, whole=[LAUGH])
    assert start <= LAUGH[0] and end >= LAUGH[1]
    # The transcript prints a sound's times to the hundredth: a cut that near its ends covers it.
    start, end = edits.place_cut(2.03, 4.97, words, 10.0, whole=[LAUGH])
    assert (start, end) == (pytest.approx(1.8, abs=0.21), pytest.approx(5.4, abs=0.41)) and start <= 2.0 and end >= 5.0


def test_set_edit_tells_claude_when_a_cut_stopped_at_a_laugh():
    words = talk_with_a_laugh()
    cut = {"start": 3.0, "end": 7.0, "reason": "a tangent"}
    outcome = edits.build_edit([cut], words, 10.0, whole_sounds=[LAUGH])
    assert outcome.cuts[0]["start"] >= LAUGH[1]
    assert "stopped at the laugh or long sound at 2.00-5.00" in outcome.adjusted[0]["why"]
    assert "covers all of it" in outcome.adjusted[0]["why"]


def auto(start: float, end: float, **more) -> dict:
    return {"start": start, "end": end, "source": "auto", "reason": "auto: pause: 1.0s", "kind": "pauses", **more}


def test_the_last_check_takes_a_laugh_back_out_of_an_automatic_cut():
    cuts, changed = edits.leave_sounds_whole([auto(1.0, 6.0), auto(7.0, 7.5)], [LAUGH], [])
    assert [(c["start"], c["end"]) for c in cuts] == [(1.0, 2.0), (5.0, 6.0), (7.0, 7.5)] and changed == 1
    # A piece too short to matter goes with the laugh's part.
    cuts, _ = edits.leave_sounds_whole([auto(1.95, 5.5)], [LAUGH], [])
    assert [(c["start"], c["end"]) for c in cuts] == [(5.0, 5.5)]
    # A cut that never touches one comes back as it is.
    untouched = [auto(0.0, 2.0), auto(5.0, 6.0)]
    assert edits.leave_sounds_whole(untouched, [LAUGH], []) == (untouched, 0)


def test_the_last_check_leaves_what_claude_and_the_creator_asked_for():
    by_claude = {"start": 1.5, "end": 5.5, "source": "claude", "reason": "the joke falls flat", "kind": "other"}
    assert edits.leave_sounds_whole([by_claude], [LAUGH], [(1.5, 5.5)]) == ([by_claude], 0)
    # An automatic cut that joined hers keeps her part of the laugh out, and gives back the rest.
    joined = {**by_claude, "start": 1.0, "end": 6.0, "source": "you", "auto_trims": 1}
    cuts, changed = edits.leave_sounds_whole([joined], [LAUGH], [(3.0, 4.0)])
    assert [(c["start"], c["end"]) for c in cuts] == [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)] and changed == 1


# ── Measuring twice without the model ─────────────────────────────────────────


def test_the_second_measure_of_the_same_words_runs_no_model():
    real = SameTimes()
    aligner = Remembering(real)
    words = said()
    first = aligner.align("take.mp4", words, [])
    laugh = {"word": "[vocalization]", "start": 2.7, "end": 5.3, "type": "event"}
    again = aligner.align("take.mp4", sorted(words + [laugh], key=lambda w: w["start"]), [])
    assert real.calls == 1
    assert [w for w in again if w.get("type") != "event"] == first and laugh in again
    aligner.align("take.mp4", words[:-1] + [{"word": "gone.", "start": 12.5, "end": 13.0}], [])
    assert real.calls == 2, "other words are measured for real"


def test_measured_times_saved_with_the_project_stand_in_for_a_model_that_is_gone(take):
    project = open_project(str(take))
    word_times.align_project(project, aligner=SameTimes(), silences=no_silences)

    class Gone(SameTimes):
        def lacks(self) -> str:
            return "The model file is not on this machine."

    gone = Gone()
    aligner = Remembering.of(gone, project)
    assert aligner.lacks() == ""
    made = word_times.align_project(project, aligner=aligner, silences=no_silences)
    assert made["word_times"] == "measured" and gone.calls == 0
    with pytest.raises(StudioError, match="not on this machine"):
        aligner.align(take, [{"word": "new", "start": 0.0, "end": 0.4}], [])
    assert Remembering.of(gone, open_project(str(take.with_name("other.mp4").parent / "take.mp4"))).lacks() == ""
