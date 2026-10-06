"""Word times: one source for every reader, aligning as a job, and the fall back when it can't run.

No test here loads a model or touches real footage. Aligning is a fake
(``conftest.FakeAligner``) passed in as a keyword parameter.
"""

import dataclasses
import json

import pytest
from lumr_studio.engine.audio_boundaries import Silence

import positivity
import seams
import short_word
from conftest import NO_MODEL_IN_TESTS, FakeAligner, make_words
from lumr_studio import edit as edits_module
from lumr_studio import models, sounds, tools, treatment, word_times
from lumr_studio.aligner import Wav2Vec2Aligner
from lumr_studio.errors import StudioError
from lumr_studio.project import open_project
from lumr_studio.silences import no_silences
from lumr_studio.tools import unmeasured_labels
from lumr_studio.word_times import ESTIMATED, MEASURED
from lumr_studio.word_times import aligning_lacks as real_aligning_lacks

QUIET = {"silences": no_silences, "labels": unmeasured_labels}
DURATION = 20.0
RETAKE = {"start": 7.9, "end": 10.75, "reason": "second take of the opening line", "kind": "repeat"}


@pytest.fixture(autouse=True)
def fresh_recipes():
    treatment.forget_recipes()
    yield
    treatment.forget_recipes()


def align(video, aligner=None):
    project = open_project(str(video))
    return project, word_times.align_project(project, aligner=aligner or FakeAligner(), silences=no_silences)


def times_of(words, text):
    return next((w["start"], w["end"]) for w in words if w["word"] == text)


def ctx_for(video):
    return treatment.load_context(open_project(str(video)), duration=DURATION, silences=no_silences)


# ── the one source ────────────────────────────────────────────────────────────


def test_without_aligned_times_the_words_are_the_transcripts_and_say_estimated(video):
    times = word_times.read(open_project(str(video)))
    assert times.source == ESTIMATED and times.words == make_words()
    assert "estimates" in times.note and "Ask Claude" in times.note


def test_aligning_saves_measured_times_in_the_project_folder(video, isolated_home):
    project, made = align(video)
    assert made["word_times"] == MEASURED and made["words_moved"] == 25
    saved = word_times.aligned_path(project)
    assert made["saved_to"] == str(saved) and saved.parent == project.root
    assert sorted(p.name for p in video.parent.iterdir()) == ["talk.mp4", "talk.words.json"]
    assert not (isolated_home / "lumr-home").exists()
    times = word_times.read(project)
    assert times.source == MEASURED and times.note == ""
    assert times_of(times.words, "Hello") == (0.6, 0.8)  # the fake halves each word about its middle
    assert [w["word"] for w in times.words] == [w["word"] for w in make_words()]


def test_sounds_keep_the_times_the_transcript_gave_them(video):
    class ClampsEverything(FakeAligner):
        def align(self, video, words, silences):
            return [{**w, "end": round(w["start"] + 0.1, 3)} for w in words]  # as clipforge ends a laugh early

    project, _ = align(video, ClampsEverything())
    assert times_of(word_times.read(project).words, "[vocalization]") == (6.0, 6.6)
    assert times_of(word_times.read(project).words, "Hello") == (0.5, 0.6)


def test_a_new_transcript_makes_the_aligned_times_stale(video):
    project, _ = align(video)
    words = make_words()
    words[0]["word"] = "Hi"
    video.with_suffix(".words.json").write_text(json.dumps(words))
    times = word_times.read(project)
    assert times.source == ESTIMATED and times.words[0]["word"] == "Hi"
    assert "transcript changed" in times.note
    assert word_times.source_of(project) == ESTIMATED


@pytest.mark.parametrize("content", ["not json", "[]", '{"version": 1}', '{"version": 99, "words": []}', '{"version": 7, "words": [5]}'])
def test_aligned_times_that_cannot_be_read_fall_back_to_the_transcript(video, content):
    project, _ = align(video)
    word_times.aligned_path(project).write_text(content)
    times = word_times.read(project)
    assert times.source == ESTIMATED and times.words == make_words()


def test_a_reader_may_change_the_words_it_was_given(video):
    project, _ = align(video)
    word_times.read(project).words[0]["word"] = "spoiled"
    assert word_times.read(project).words[0]["word"] == "Hello"


def test_every_reader_gets_the_same_words(video):
    tools.set_edit(str(video), [RETAKE], **QUIET)
    align(video)
    hello = [0.6, 0.8]
    assert tools.read_transcript(str(video), labels=unmeasured_labels)["text"].startswith("[0.60-")
    state = treatment.page_state(ctx_for(video))
    assert state["words"][0][:3] == ["Hello", *hello]
    assert state["word_times"] == MEASURED and state["word_times_note"] == ""
    assert tools.analyze_take(str(video), silences=no_silences)["word_times"] == MEASURED
    assert tools.get_edit(str(video))["word_times"] == MEASURED
    assert tools.set_edit(str(video), [RETAKE], **QUIET)["word_times"] == MEASURED


# ── what aligning needs ───────────────────────────────────────────────────────


def test_to_the_real_aligner_a_test_machine_lacks_the_model(video):
    assert Wav2Vec2Aligner().lacks() == NO_MODEL_IN_TESTS


def test_a_machine_without_the_model_file_is_told_so_and_nothing_is_fetched(tmp_path, monkeypatch):
    monkeypatch.setattr(word_times, "model_path", lambda: tmp_path / "model.pth")
    lacks = real_aligning_lacks()
    assert "not on this machine" in lacks and "about 380 MB" in lacks and str(tmp_path) in lacks
    assert "Transcribe downloads it when the creator says yes" in lacks
    assert not (tmp_path / "model.pth").exists()


def test_a_model_file_cut_short_counts_as_missing(tmp_path, monkeypatch):
    (tmp_path / "model.pth").write_bytes(b"half a download")
    monkeypatch.setattr(word_times, "model_path", lambda: tmp_path / "model.pth")
    assert "incomplete" in real_aligning_lacks()
    monkeypatch.setattr(word_times, "MODEL_MIN_BYTES", 4)
    assert real_aligning_lacks() == ""


def test_a_machine_without_torch_is_told_so(monkeypatch):
    def no_torch():
        raise ImportError("No module named 'torch'", name="torch")

    monkeypatch.setattr(word_times, "model_path", no_torch)
    assert "torch is not installed" in real_aligning_lacks()


def test_a_download_is_refused():
    with pytest.raises(StudioError, match="tried to download https://example.invalid/model.pth"):
        word_times.refuse_download("https://example.invalid/model.pth", "/tmp/model.pth")


def test_the_model_file_is_the_one_torchaudio_asks_for(tmp_path, monkeypatch):
    """The file ``models.download`` writes is where the aligner looks, and the size floor sits under it."""
    monkeypatch.setattr(models, "ALIGNER", dataclasses.replace(models.ALIGNER, where=lambda: tmp_path))
    assert word_times.model_path() == tmp_path / models.ALIGNER_FILE.name
    assert word_times.MODEL_MIN_BYTES < models.ALIGNER_FILE.size < 2 * word_times.MODEL_MIN_BYTES


def test_aligning_on_a_machine_that_cannot_says_why_and_saves_nothing(video):
    project = open_project(str(video))
    with pytest.raises(StudioError, match="no model here"):
        word_times.align_project(project, aligner=FakeAligner(lacks="There is no model here."), silences=no_silences)
    assert not word_times.aligned_path(project).exists()


def test_an_aligner_that_gives_back_other_words_is_not_trusted(video):
    class DropsAWord(FakeAligner):
        def align(self, video, words, silences):
            return super().align(video, words, silences)[1:]

    project = open_project(str(video))
    with pytest.raises(StudioError, match="different words"):
        word_times.align_project(project, aligner=DropsAWord(), silences=no_silences)
    assert word_times.read(project).source == ESTIMATED


# ── aligning through the transcribe tool ──────────────────────────────────────


def test_a_transcript_without_measured_times_is_reported_with_why(video, jobs):
    result = tools.transcribe(str(video), jobs=jobs)
    assert result["status"] == "exists" and result["word_times"] == ESTIMATED and "job_id" not in result
    assert NO_MODEL_IN_TESTS in result["word_times_note"] and "harder paces find less" in result["word_times_note"]


def test_a_transcript_without_measured_times_gets_them_in_one_job(video, jobs):
    aligner = FakeAligner()
    started = tools.transcribe(str(video), aligner=aligner, jobs=jobs, **QUIET)
    assert started["status"] == "aligning" and started["word_times"] == ESTIMATED
    done = jobs.wait(started["job_id"], timeout=10)
    assert done["status"] == "done", done
    result = done["result"]
    assert result["word_times"] == MEASURED and result["word_count"] == 25
    assert result["aligning"]["words_moved"] == 25 and "edit" not in result
    again = tools.transcribe(str(video), aligner=aligner, jobs=jobs, **QUIET)
    assert again["status"] == "exists" and again["word_times"] == MEASURED and "job_id" not in again
    assert aligner.calls == 1


def test_a_new_video_is_transcribed_and_aligned_in_one_job(video, jobs, words):
    video.with_suffix(".words.json").unlink()

    def fake_transcriber(path, force):
        path.with_suffix(".words.json").write_text(json.dumps(words))

    started = tools.transcribe(str(video), transcriber=fake_transcriber, aligner=FakeAligner(), jobs=jobs, **QUIET)
    assert set(started) == {"job_id"}
    result = jobs.wait(started["job_id"], timeout=10)["result"]
    assert result["word_count"] == 25 and result["word_times"] == MEASURED


def test_transcribing_again_aligns_again(video, jobs, words):
    aligner = FakeAligner()
    align(video, aligner)

    def fake_transcriber(path, force):
        path.with_suffix(".words.json").write_text(json.dumps(words[:-1]))

    started = tools.transcribe(str(video), force=True, transcriber=fake_transcriber, aligner=aligner, jobs=jobs, **QUIET)
    result = jobs.wait(started["job_id"], timeout=10)["result"]
    assert result["word_times"] == MEASURED and result["word_count"] == 24 and aligner.calls == 2
    assert len(word_times.read(open_project(str(video))).words) == 25


def test_an_aligner_that_breaks_leaves_the_job_done_on_estimated_times(video, jobs):
    class Breaks(FakeAligner):
        def align(self, video, words, silences):
            raise RuntimeError("ffmpeg decode failed")

    started = tools.transcribe(str(video), aligner=Breaks(), jobs=jobs, **QUIET)
    done = jobs.wait(started["job_id"], timeout=10)
    assert done["status"] == "done", done
    assert done["result"]["word_times"] == ESTIMATED
    assert "RuntimeError" in done["result"]["word_times_note"]
    assert word_times.read(open_project(str(video))).source == ESTIMATED


# ── a saved edit when the word times change under it ──────────────────────────


def saved(video):
    return json.loads(open_project(str(video)).edit_path.read_text())


def test_an_edit_says_which_word_times_it_was_placed_on(video):
    tools.set_edit(str(video), [RETAKE], **QUIET)
    assert saved(video)["word_times"] == ESTIMATED
    align(video)
    tools.set_edit(str(video), [RETAKE], **QUIET)
    assert saved(video)["word_times"] == MEASURED


def test_aligning_places_the_saved_edit_again_and_says_so(video, jobs):
    tools.set_edit(str(video), [RETAKE], auto_tighten=True, **QUIET)
    before = saved(video)
    started = tools.transcribe(str(video), aligner=FakeAligner(), jobs=jobs, **QUIET)
    result = jobs.wait(started["job_id"], timeout=10)["result"]
    after = saved(video)
    assert after["word_times"] == MEASURED
    # The cut moved with its words: from "today" to "video." as they are timed now.
    assert [(r["start"], r["end"], r["reason"]) for r in after["requested"]] == [(8.1, 10.562, RETAKE["reason"])]
    assert result["edit"]["before"]["cut_count"] == len(before["cuts"])
    assert result["edit"]["now"]["cut_count"] == len(after["cuts"])
    assert "placed again" in result["edit"]["note"]
    assert [c["source"] for c in after["cuts"]].count("claude") == 1
    assert after["cuts"] != before["cuts"], "the cuts sit on the measured word edges now"


def test_an_edit_saved_before_this_round_loads_and_moves_to_the_measured_times(video):
    tools.set_edit(str(video), [RETAKE], **QUIET)
    project = open_project(str(video))
    old = saved(video)
    for key in ("word_times", "creator_cuts", "undo"):
        old.pop(key, None)
    project.edit_path.write_text(json.dumps(old))
    before = treatment.page_state(ctx_for(video))
    assert before["word_times"] == ESTIMATED and [r["state"] for r in before["rows"]] == ["kept_out"]
    align(video)
    after = treatment.page_state(ctx_for(video))
    assert saved(video)["word_times"] == MEASURED
    assert [r["reason"] for r in after["rows"]] == [RETAKE["reason"]]
    assert after["rows"][0]["id"] != before["rows"][0]["id"], "row ids come from placed edges, which moved"
    assert after["rows"][0]["removed"] == before["rows"][0]["removed"], "the same words are cut"


def test_cuts_and_keeps_carry_over_when_the_word_times_change(video):
    tools.set_edit(str(video), [RETAKE], **QUIET)
    ctx = ctx_for(video)
    rid = treatment.page_state(ctx)["rows"][0]["id"]
    treatment.set_cut_state(ctx, {"id": rid, "state": "put_back"})
    treatment.add_keep(ctx, {"start": 12.1, "end": 12.4})
    treatment.add_cut(ctx, {"start": 16.05, "end": 16.2})
    align(video)
    state = treatment.page_state(ctx_for(video))
    # Each span moved with the words it holds, so it holds the same words on the new times.
    assert [(k["id"], k["text"]) for k in state["keeps"]] == [("k12.07-14.01", "this is the part that matters.")]
    claude, yours = (next(r for r in state["rows"] if r["by"] == by) for by in ("claude", "you"))
    assert claude["state"] == "put_back" and claude["id"] != rid
    assert claude["removed"] == "today we talk about editing video."
    assert yours["removed"] == "for" and yours["id"] == "y16.09-16.16"
    # The kept span still names the row's earlier id, and leaving it out still works.
    out = treatment.set_cut_state(ctx_for(video), {"id": claude["id"], "state": "kept_out"})
    assert next(r for r in out["rows"] if r["by"] == "claude")["state"] == "kept_out"


def test_a_span_moves_onto_the_measured_times_of_the_words_it_held(video):
    project, _ = align(video)
    words = word_times.read(project).words
    # "we talk" by the transcript's times, a little loosely picked.
    assert word_times.on_measured_times(4.32, 4.92, words) == (4.387, 4.812)
    assert word_times.on_measured_times(6.7, 7.5, words) == (6.7, 7.5), "a span in a pause stays where it is"
    assert word_times.on_measured_times(4.32, 4.92, make_words()) == (4.32, 4.92), "no transcript times to go by"


def test_a_cut_that_took_an_edge_word_on_estimated_times_still_takes_it_on_measured_ones(video):
    # "and welcome." picked by the transcript's times, where "and" starts at 1.60. Measured, "and" sits at
    # 1.65 to 1.75: a cut from 1.72 would miss its middle if it were only placed again.
    loose = {"start": 1.58, "end": 2.45, "reason": "a greeting said twice", "kind": "repeat"}
    tools.set_edit(str(video), [loose], **QUIET)
    before = treatment.page_state(ctx_for(video))["rows"][0]["removed"]

    class Squeezes(FakeAligner):
        def align(self, video, words, silences):
            return [{**w, "start": round(w["end"] - 0.05, 3)} if w.get("type") != "event" else dict(w) for w in words]

    align(video, Squeezes())
    after = treatment.page_state(ctx_for(video))["rows"][0]["removed"]
    assert before == after == "and welcome."


def test_an_empty_edit_is_left_alone_when_the_word_times_change(video):
    align(video)
    project = open_project(str(video))
    treatment.load_context(project, duration=DURATION, silences=no_silences)
    assert not project.edit_path.exists()


# ── sound labels follow the word times ────────────────────────────────────────


def test_sounds_are_judged_again_on_new_word_times_from_the_measures_already_saved(video):
    calls = []

    def measure(_video, spans):
        calls.append(spans)
        return [{} for _ in spans]

    project = open_project(str(video))
    first = sounds.load_or_measure_labels(project, word_times.load_words(project), measure=measure)
    align(video)
    again = sounds.load_or_measure_labels(project, word_times.load_words(project), measure=measure)
    cached = json.loads((project.root / sounds.SOUNDS_FILE).read_text())
    assert len(calls) == 1 and cached["word_times"] == MEASURED
    assert [(lb["start"], lb["end"]) for lb in again] == [(lb["start"], lb["end"]) for lb in first]
    sounds.load_or_measure_labels(project, word_times.load_words(project), measure=measure)
    assert len(calls) == 1


# ── sounds give way to measured words ─────────────────────────────────────────


def sound(start, end):
    return {"word": "[vocalization]", "start": start, "end": end, "type": "event"}


def said(text, start, end):
    return {"word": text, "start": start, "end": end}


def test_a_sound_is_shortened_to_where_no_measured_word_sits():
    words = [said("there.", 79.04, 79.2), sound(79.76, 80.4), said("So,", 80.22, 80.32), said("let's", 81.5, 81.64)]
    cleared = word_times._sounds_clear_of_words(words)
    assert [(w["word"], w["start"], w["end"]) for w in cleared] == [
        ("there.", 79.04, 79.2), ("[vocalization]", 79.76, 80.22), ("So,", 80.22, 80.32), ("let's", 81.5, 81.64),
    ]


def test_a_sound_that_was_words_all_along_is_dropped():
    words = [sound(10.0, 10.9), said("you", 10.02, 10.17), said("can", 10.2, 10.42), said("be", 10.46, 10.54),
             said("you.", 10.6, 10.9)]
    assert [w["word"] for w in word_times._sounds_clear_of_words(words)] == ["you", "can", "be", "you."]


def test_a_laugh_with_no_words_in_it_stays_whole():
    words = [said("dinner.", 481.58, 481.98), sound(482.68, 486.2), said("Anyway", 486.34, 486.52)]
    assert word_times._sounds_clear_of_words(words) == words


def test_aligning_says_how_many_sounds_turned_out_to_be_words(video):
    class PutsAWordInTheSound(FakeAligner):
        def align(self, video, words, silences):
            out = super().align(video, words, silences)
            for w in out:
                if w["word"] == "editing" and w["start"] < 6:
                    w["start"], w["end"] = 5.98, 6.62  # over the whole sound at 6.0 to 6.6
            return out

    project, made = align(video, PutsAWordInTheSound())
    assert (made["sounds"], made["sounds_that_were_words"]) == (1, 1)
    assert all(w.get("type") != "event" for w in word_times.read(project).words)


def test_sounds_that_moved_are_measured_again(video):
    calls = []

    def measure(_video, spans):
        calls.append(list(spans))
        return [{} for _ in spans]

    class ShortensTheSound(FakeAligner):
        def align(self, video, words, silences):
            out = super().align(video, words, silences)
            for w in out:
                if w["word"] == "editing" and w["start"] < 6:
                    w["start"], w["end"] = 5.6, 6.2  # the word runs into the sound at 6.0 to 6.6
            return out

    project = open_project(str(video))
    sounds.load_or_measure_labels(project, word_times.load_words(project), measure=measure)
    align(video, ShortensTheSound())
    sounds.load_or_measure_labels(project, word_times.load_words(project), measure=measure)
    assert calls == [[(6.0, 6.6)], [(6.2, 6.6)]]


# ── room for each word ────────────────────────────────────────────────────────
#
# The take is ``short_word``: the aligner gave "take" 1 ms, in the pause after "just".


def roomy():
    return word_times.with_room(short_word.aligned(), short_word.silences())


def test_a_word_the_aligner_cut_short_gets_the_room_the_transcript_gave_it():
    assert short_word.times_of(short_word.aligned(), "take") == short_word.TAKE_AS_ALIGNED
    take = next(w for w in roomy() if w["word"] == "take")
    assert (take["start"], take["end"]) == short_word.TAKE == (1.932, 2.162)
    assert take["start"] <= short_word.TAKE_SOUNDS[0] and short_word.TAKE_SOUNDS[1] <= take["end"], "its sound is inside it"
    # The room stops at the sound: the transcript gave it 1.92 to 2.24, and both ends of that are silent.
    assert take[word_times.RAW_START] < take["start"] and take["end"] < take[word_times.RAW_END]
    assert (take[word_times.ALIGNED_START], take[word_times.ALIGNED_END]) == short_word.TAKE_AS_ALIGNED


def test_the_room_stops_at_the_neighbours_measured_edges():
    words = roomy()
    assert short_word.times_of(words, "just")[1] == 1.507 <= short_word.times_of(words, "take")[0]
    assert short_word.times_of(words, "take")[1] <= short_word.times_of(words, "one")[0]
    spoken = [w for w in words if w.get("type") != "event"]
    assert all(a["end"] <= b["start"] for a, b in zip(spoken, spoken[1:])), "no two words overlap"


def test_a_word_is_cut_short_when_it_is_shorter_than_a_frame_of_the_aligner_a_letter():
    def lasting(text, seconds):
        return {"word": text, "start": 10.0, "end": round(10.0 + seconds, 3)}

    assert word_times.cut_short(lasting("take", 0.001)) and word_times.cut_short(lasting("take", 0.079))
    assert not word_times.cut_short(lasting("take", 0.08)), "four letters, four frames of 20 ms"
    assert word_times.cut_short(lasting("experiencing", 0.116)), "0.116 s is no tiny word, and still too short for twelve letters"
    assert not word_times.cut_short(lasting("I", 0.02)) and word_times.cut_short(lasting("I", 0.006))
    assert word_times.letters_of({"word": "Don't,"}) == 5 and word_times.letters_of({"word": "..."}) == 1


def test_sound_that_runs_on_from_a_word_is_the_words():
    words = roomy()
    # "You" was given 0.98 to 1.08 and "can" starts at 1.12, with sound and no silence between.
    assert short_word.times_of(words, "You") == (0.98, 1.12)
    assert short_word.times_of(words, "can") == (1.12, 1.32)
    # Sound leads into "one" from 2.333. It is 0.171 s after "take" ends, too far to be the end of "take".
    assert short_word.times_of(words, "one") == (2.333, 2.518)
    # "just" ends where a silence of 0.425 s starts, and keeps its end.
    assert short_word.times_of(words, "just") == (1.32, 1.507)
    assert word_times.ALIGNED_START not in next(w for w in words if w["word"] == "just")


def test_a_quiet_as_long_as_the_shortest_pause_the_pace_cuts_ends_a_word():
    from lumr_studio.engine.microcut_pacing import GAP_LENGTH_MIN

    assert word_times.RUNS_ON_SECONDS == GAP_LENGTH_MIN

    def after_a_quiet_of(seconds):
        words = [said("heard", 1.0, 1.3), said("next", 3.0, 3.3)]
        quiet = [Silence(0.0, 1.0), Silence(1.3, round(1.3 + seconds, 3)), Silence(round(1.5 + seconds, 3), 3.0)]
        return short_word.times_of(word_times.with_room(words, quiet), "heard")

    assert after_a_quiet_of(0.14) == (1.0, 1.64), "the sound after a short quiet is the rest of the word"
    assert after_a_quiet_of(0.15) == (1.0, 1.3), "after a pause it is a breath, or a word nobody wrote down"


def test_a_word_never_grows_into_a_sound_of_the_transcript():
    # The laugh starts at 1.4. The sound between the word and the laugh is the word's; the laugh stays whole.
    words = [said("dinner.", 1.0, 1.3), sound(1.4, 3.0), said("Anyway", 3.3, 3.5)]
    quiet = [Silence(0.0, 1.0), Silence(3.05, 3.2)]
    out = word_times.with_room(words, quiet)
    assert [(w["word"], w["start"], w["end"]) for w in out] == [
        ("dinner.", 1.0, 1.4), ("[vocalization]", 1.4, 3.0), ("Anyway", 3.2, 3.5),
    ]
    assert word_times._sounds_clear_of_words(out) == out


def test_sound_the_transcript_heard_inside_a_word_is_the_words():
    # The take is ``positivity``: the soft "s" and the closed mouth before the "t" read as 0.172 s of silence.
    words = word_times.with_room(positivity.aligned(), positivity.silences())
    assert positivity.times_of(positivity.aligned(), "positivity") == positivity.AS_ALIGNED
    assert positivity.times_of(words, "positivity") == positivity.WHOLE == (23.251, 24.202), "\"-tivity\" is in it"
    assert word_times.RUNS_ON_SECONDS < round(23.6925 - 23.5209, 3) < word_times.LONGEST_QUIET_IN_A_WORD
    assert positivity.times_of(words, "at")[0] == 25.07, "the pause before \"at\" is no part of it"
    others = [row for row in positivity.ROWS if row[0] != "positivity"]
    assert [positivity.times_of(words, row[0]) for row in others] == [row[5:] for row in others], \
        "every other word keeps the room it had"


def after_a_quiet_of(seconds, heard_until=2.5, lasting=0.2):
    """Where "heard" ends when a stretch of sound ``lasting`` this long follows it after a quiet of ``seconds``."""
    words = [{**said("heard", 1.0, 1.3), word_times.RAW_END: heard_until}, said("next", 3.0, 3.3)]
    sounds_until = round(1.3 + seconds + lasting, 3)
    quiet = [Silence(0.0, 1.0), Silence(1.3, round(1.3 + seconds, 3)), Silence(sounds_until, 3.0)]
    return short_word.times_of(word_times.with_room(words, quiet), "heard")


def test_sound_after_a_quiet_longer_than_any_in_a_word_is_not_the_words():
    assert word_times.LONGEST_QUIET_IN_A_WORD == 0.25
    assert after_a_quiet_of(0.2) == (1.0, 1.7), "the transcript heard the word go on"
    assert after_a_quiet_of(0.25) == (1.0, 1.3), "a click, a breath or a laugh after the word"
    assert after_a_quiet_of(0.2, heard_until=1.55) == (1.0, 1.3), "the transcript heard the word end before it"
    assert after_a_quiet_of(0.2, heard_until=1.65) == (1.0, 1.3), "and before the sound ends"


def test_a_word_takes_no_more_across_a_quiet_than_the_rest_of_a_word():
    assert word_times.LONGEST_REST_OF_A_WORD == 0.5
    assert after_a_quiet_of(0.2, lasting=0.3) == (1.0, 1.8), "a syllable or two"
    # On a measured take: a breath after a word (0.193 s of quiet, 0.389 s of sound), a laugh after another (0.182, 1.528).
    assert after_a_quiet_of(0.193, lasting=0.389) == (1.0, 1.3)
    assert after_a_quiet_of(0.182, lasting=1.528, heard_until=3.0) == (1.0, 1.3)


def test_a_word_keeps_the_rest_of_itself_after_a_closed_mouth():
    # On a measured take: "at most a dozen". The "st" of "most" sounds after a closed mouth, and "a" has a lead-in.
    words = [said("most", 779.109, 779.314), said("a", 779.889, 779.909), said("dozen", 779.97, 780.13)]
    quiet = [Silence(778.9, 779.109), Silence(779.314, 779.4), Silence(779.799, 779.878), Silence(780.19, 781.0)]
    out = word_times.with_room(words, quiet)
    assert short_word.times_of(out, "most") == (779.109, 779.799) and short_word.times_of(out, "a")[0] == 779.878


def test_a_stretch_both_words_reach_at_once_goes_to_the_nearer_one():
    words = [said("before", 1.0, 1.3), said("after", 1.55, 1.8)]
    quiet = [Silence(0.0, 1.0), Silence(1.3, 1.4), Silence(1.5, 1.55), Silence(1.8, 3.0)]
    out = word_times.with_room(words, quiet)
    assert short_word.times_of(out, "after")[0] == 1.4 and short_word.times_of(out, "before")[1] == 1.3


def test_a_word_keeps_its_soft_end_and_start():
    words = word_times.with_room(positivity.aligned(), positivity.silences(), positivity.soft())
    assert positivity.times_of(words, "positivity") == positivity.WITH_ITS_SOFT_END, "\"-ty\" fades out at -40 dB"
    assert positivity.times_of(words, "itself") == (18.369, 19.196), "the \"f\" goes on after -25 dB went quiet"
    assert positivity.times_of(words, "warm")[0] == 21.455, "and the soft \"w\" leads into the word"
    spoken = [w for w in words if w.get("type") != "event"]
    assert all(a["end"] <= b["start"] for a, b in zip(spoken, spoken[1:])), "no two words overlap"
    again = word_times.with_room(words, positivity.silences(), positivity.soft())
    assert [(w["start"], w["end"]) for w in again] == [(w["start"], w["end"]) for w in words], "giving room again"


def test_soft_sound_that_runs_on_is_taken_only_so_far():
    assert (word_times.SOFT_END_SECONDS, word_times.SOFT_START_SECONDS) == (0.3, 0.15)
    words = [said("books", 1.0, 1.3), said("that", 3.0, 3.2)]
    quiet = [Silence(0.0, 1.0), Silence(1.3, 3.0), Silence(3.2, 5.0)]
    # Soft sound fills the whole pause, as room noise or a breath might.
    out = word_times.with_room(words, quiet, [Silence(0.0, 1.0), Silence(3.2, 5.0)])
    assert short_word.times_of(out, "books") == (1.0, 1.6) and short_word.times_of(out, "that") == (2.85, 3.2)
    # A laugh of the transcript stays whole.
    laugh = [said("books", 1.0, 1.3), {**sound(1.45, 2.5)}, said("that", 3.0, 3.2)]
    out = word_times.with_room(laugh, quiet, [Silence(0.0, 1.0), Silence(3.2, 5.0)])
    assert short_word.times_of(out, "books") == (1.0, 1.45) and short_word.times_of(out, "that") == (2.85, 3.2)


def test_a_soft_start_is_measured_from_where_the_aligner_heard_the_word_begin():
    # On a measured take the aligner began "sending" 0.146 s inside the silence before it: its soft "s".
    words = [said("in", 264.087, 264.554), said("sending", 265.032, 265.712)]
    quiet = [Silence(263.5, 264.087), Silence(264.554, 265.178), Silence(265.712, 266.0)]
    out = word_times.with_room(words, quiet, [Silence(264.646, 264.901)])
    assert short_word.times_of(out, "sending")[0] == 264.901


# Five words the aligner ran into the next one on a measured take, each at a seam of its 40 s windows:
# (word, aligned start, aligned end, the next word, its aligned start, its aligned end).
RUN_INTO_THE_NEXT = [
    ("him", 360.039, 360.261, "now", 360.08, 360.2), ("small", 399.879, 400.339, "step", 400.06, 400.16),
    ("plan", 639.979, 640.433, "we", 640.24, 640.38), ("clever", 1039.799, 1040.204, "on", 1040.2, 1040.204),
    ("but", 1080.719, 1081.02, "so", 1080.98, 1081.02),
]


@pytest.mark.parametrize("pair", RUN_INTO_THE_NEXT, ids=lambda pair: f"{pair[0]} {pair[3]}")
def test_a_word_the_aligner_ran_into_the_next_ends_where_the_next_starts(pair):
    text, a, b, after, c, d = pair
    words = [said(text, a, b), said(after, c, d), said("later", d + 2.0, d + 2.3)]
    out = word_times.with_room(words, [Silence(d + 0.5, d + 1.5)])
    first = next(w for w in out if w["word"] == text)
    assert (first["start"], first["end"]) == (a, c) and short_word.times_of(out, after)[0] == c
    assert (first[word_times.ALIGNED_START], first[word_times.ALIGNED_END]) == (a, b), "the aligner's end is kept"
    assert word_times.with_room(out, [Silence(d + 0.5, d + 1.5)]) == out, "giving room again changes nothing"


def test_a_word_ends_in_the_sound_of_the_transcript_that_was_cut_back_to_it():
    # On a measured take "in" sounds to 887.72 and fades out by 887.78. The transcript heard a
    # "[vocalization]" start at 887.6, inside the word, and it was cut back to where the aligner ended "in".
    words = [{**said("in", 887.223, 887.644), word_times.RAW_START: 887.28, word_times.RAW_END: 887.6},
             {**sound(887.644, 888.24), word_times.RAW_START: 887.6, word_times.RAW_END: 888.24},
             {**said("who", 888.304, 888.424), word_times.RAW_START: 888.24, word_times.RAW_END: 888.88}]
    quiet = [Silence(886.473, 887.161), Silence(887.721, 888.38), Silence(888.744, 890.0)]
    soft = [Silence(886.642, 887.16), Silence(887.782, 888.239), Silence(888.8, 890.0)]
    out = word_times._sounds_clear_of_words(word_times.with_room(words, quiet, soft))
    assert short_word.times_of(out, "in") == (887.223, 887.782)
    assert short_word.times_of(out, "[vocalization]") == (887.782, 888.24)
    heard_after = [{**w, word_times.RAW_START: 887.644} if w.get("type") == "event" else w for w in words]
    walled = word_times.with_room(heard_after, quiet, soft)
    assert short_word.times_of(walled, "in") == (887.223, 887.644), "a sound heard after the word stays whole"


def aligned_and_cleared(words, quiet, soft):
    """What aligning saves: the sounds cleared of the aligner's words, room given, the sounds cleared again."""
    return word_times._sounds_clear_of_words(word_times.with_room(word_times._sounds_clear_of_words(words), quiet, soft))


def test_giving_room_twice_changes_nothing_beside_a_sound_of_the_transcript():
    # On a measured take: "in" and the "[vocalization]" cut back to its end, as above; and "window." and "Like,"
    # with a "[vocalization]" between them that "Like," grows over until too little of it is left.
    words = [{**said("in", 887.223, 887.644), word_times.RAW_START: 887.28, word_times.RAW_END: 887.6},
             {**sound(887.6, 888.24), word_times.RAW_START: 887.6, word_times.RAW_END: 888.24},
             {**said("who", 888.304, 888.424), word_times.RAW_START: 888.24, word_times.RAW_END: 888.88},
             {**said("window.", 889.244, 889.685), word_times.RAW_START: 889.2, word_times.RAW_END: 889.76},
             {**sound(889.76, 890.0), word_times.RAW_START: 889.76, word_times.RAW_END: 890.0},
             {**said("Like,", 889.885, 890.072), word_times.RAW_START: 890.0, word_times.RAW_END: 890.72}]
    quiet = [Silence(886.473, 887.161), Silence(887.721, 888.38), Silence(888.744, 889.244),
             Silence(889.8368, 889.8989), Silence(890.072, 890.781)]
    soft = [Silence(886.642, 887.16), Silence(887.782, 888.239), Silence(888.8, 889.2), Silence(890.4902, 890.6012)]
    once = aligned_and_cleared(words, quiet, soft)
    assert short_word.times_of(once, "in") == (887.223, 887.782)
    assert [w["start"] for w in once if w.get("type") == "event"] == [887.782], "the second sound was the words'"
    assert word_times._sounds_clear_of_words(word_times.with_room(once, quiet, soft)) == once


def test_the_84_08_s_sound_is_narrowed_to_where_it_really_sounds():
    # On a measured take: a "[vocalization]" the transcript heard from 84.08 to 84.72 s. It sits in a pause
    # ("season," ends 83.938, "after" starts 84.72) and really sounds only from 84.396: the rest, back
    # to 83.938, reads as silence even at the soft level (-40 dB), two measured silences a hair apart.
    words = [said("season,", 83.041, 83.938), sound(84.08, 84.72), said("after", 84.72, 85.142)]
    quiet = [Silence(83.7713, 84.4533), Silence(84.6801, 84.7743)]
    soft = [Silence(83.937574, 84.220952), Silence(84.222517, 84.395578)]
    out = word_times._sounds_clear_of_words(word_times.with_room(words, quiet, soft), soft)
    assert short_word.times_of(out, "[vocalization]") == (84.396, 84.72)
    # Left out, the sound keeps its loose, untrimmed span: trimming is opt in.
    untrimmed = word_times._sounds_clear_of_words(word_times.with_room(words, quiet, soft))
    assert short_word.times_of(untrimmed, "[vocalization]") == (84.08, 84.72)


def test_a_sound_silent_throughout_at_the_soft_level_is_dropped():
    # A "[vocalization]" the transcript heard in a pause that is, in fact, quiet even at -40 dB end to end:
    # room noise the transcriber heard as a voice, not one.
    words = [said("one", 1.0, 1.3), sound(1.4, 1.9), said("two", 2.0, 2.3)]
    quiet = [Silence(1.3, 2.0)]
    soft = [Silence(1.3, 2.0)]
    out = word_times._sounds_clear_of_words(word_times.with_room(words, quiet, soft), soft)
    assert [w["word"] for w in out] == ["one", "two"], "the sound left nothing worth keeping"


def test_the_soft_edges_say_how_much_they_took_and_when_the_room_misled_them():
    words = [said("one", 1.0, 1.3), said("two", 2.0, 2.3), said("three", 3.0, 3.3)]
    quiet = [Silence(0.0, 1.0), Silence(1.3, 2.0), Silence(2.3, 3.0), Silence(3.3, 4.0)]
    fine = word_times.with_soft_room(words, quiet, [Silence(0.0, 0.9), Silence(1.4, 1.9), Silence(2.4, 2.9),
                                                    Silence(3.4, 4.0)])[1]
    assert (fine.words, round(fine.seconds, 3), fine.warning()) == (3, 0.6, "")
    assert fine.as_dict() == {"words_given_soft_room": 3, "soft_room_seconds": 0.6, "soft_room_share": 0.19}
    # A room as loud as a soft "s": no quiet near the words at the soft level, so each takes all it may.
    loud = word_times.with_soft_room(words, quiet, [Silence(9.0, 10.0)])[1]
    assert loud.share > word_times.SOFT_SHARE_WARNED and "took 44% of the pauses" in loud.warning()
    # A room never quiet at the soft level.
    never = word_times.with_soft_room(words, quiet, [])[1]
    assert never.warning() == word_times.NOTE_NO_SOFT_QUIET and never.as_dict()["warning"] == never.warning()
    assert word_times.with_soft_room(words, quiet)[1].as_dict() == {
        "words_given_soft_room": 0, "soft_room_seconds": 0.0, "soft_room_share": 0.0,
    }, "not measured, nothing to say"


def test_measuring_word_times_tells_claude_when_the_room_misled_the_soft_edges(video, jobs, monkeypatch):
    video.with_suffix(".words.json").write_text(json.dumps(short_word.transcript()))
    monkeypatch.setattr(word_times, "measured_soft_silences", lambda _video: [])
    started = tools.transcribe(str(video), aligner=short_word.ShortWordAligner(), jobs=jobs,
                               silences=short_word.silences, labels=unmeasured_labels)
    result = jobs.wait(started["job_id"], timeout=10)["result"]
    assert result["word_times"] == MEASURED and result["word_times_warning"] == word_times.NOTE_NO_SOFT_QUIET
    assert result["aligning"]["words_given_soft_room"] == 0
    saved = json.loads(word_times.aligned_path(open_project(str(video))).read_text())
    assert saved["soft_room"]["warning"] == word_times.NOTE_NO_SOFT_QUIET


def test_without_measured_silences_the_words_keep_their_times():
    assert word_times.with_room(short_word.aligned(), []) == short_word.aligned()


def test_a_word_said_too_quietly_to_measure_gets_back_the_frames_the_aligner_gave_it():
    # "if" was cut to 4 ms and nothing in its room sounds. Two letters are two frames.
    words = [said("see", 1.0, 1.4), {**said("if", 1.404, 1.408), "raw_start": 1.44, "raw_end": 1.52}, said("you", 1.6, 1.8)]
    quiet = [Silence(0.0, 1.0), Silence(1.408, 1.6), Silence(1.8, 5.0)]
    assert short_word.times_of(word_times.with_room(words, quiet), "if") == (1.404, 1.444)


def test_giving_room_twice_changes_nothing_more():
    once = roomy()
    again = word_times.with_room(once, short_word.silences())
    assert [(w["start"], w["end"]) for w in again] == [(w["start"], w["end"]) for w in once]
    assert short_word.times_of(again, "take") == short_word.TAKE


def test_aligning_saves_each_word_with_its_room(video):
    project = short_word.on_video(video)
    saved_file = json.loads(word_times.aligned_path(project).read_text())
    assert saved_file["version"] == word_times.ALIGNED_VERSION == 7
    times = word_times.read(project)
    assert times.source == MEASURED and short_word.times_of(times.words, "take") == short_word.TAKE
    assert [w["word"] for w in times.words] == [w["word"] for w in short_word.transcript()]


def test_aligning_says_how_many_words_were_given_room(video):
    video.with_suffix(".words.json").write_text(json.dumps(short_word.transcript()))
    made = word_times.align_project(
        open_project(str(video)), aligner=short_word.ShortWordAligner(), silences=short_word.silences,
    )
    assert made["words_given_room"] == 11 and made["word_times"] == MEASURED, "all but \"just\" and \"time.\""


def test_every_reader_sees_a_file_of_an_earlier_build_as_estimated(video):
    project = short_word.on_video(video)
    short_word.as_saved_by_an_earlier_build(project)
    assert word_times.source_of(project) == ESTIMATED
    assert word_times.load_words(project) == short_word.transcript()
    assert tools.read_transcript(str(video), labels=unmeasured_labels)["text"].startswith("[1.00-"), "the transcript's own times"
    assert tools.analyze_take(str(video), silences=short_word.silences)["word_times"] == ESTIMATED


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5, 6])
def test_times_saved_by_an_earlier_build_are_not_read_as_measured(video, jobs, version):
    """Versions 1 to 5 came from unreleased builds and 6 from MMS_FA. None is kept: the file is set aside and transcribe measures again."""
    project, _ = align(video)
    path = word_times.aligned_path(project)
    path.write_text(json.dumps({**json.loads(path.read_text()), "version": version}))
    times = word_times.read(project)
    assert times.source == ESTIMATED and word_times.source_of(project) == ESTIMATED
    assert times.note == word_times.NOTE_NOT_ALIGNED
    assert json.loads(path.read_text())["version"] == version, "reading changes nothing on disk"
    aligner = FakeAligner()
    again = tools.transcribe(str(video), aligner=aligner, jobs=jobs, **QUIET)
    assert again["status"] == "aligning" and again["word_times"] == ESTIMATED
    assert jobs.wait(again["job_id"], timeout=10)["result"]["word_times"] == MEASURED
    assert json.loads(path.read_text())["version"] == word_times.ALIGNED_VERSION
    assert word_times.source_of(project) == MEASURED


def test_a_sound_of_the_transcript_comes_through_room_given_again():
    # Two breaths that sound at the soft level throughout: one in the pause
    # before "words", one in the long pause before "at".
    breaths = [sound(20.8, 20.95), sound(23.75, 24.1)]
    saved = sorted(positivity.as_saved_with_room_before() + breaths, key=lambda w: w["start"])
    roomy, _ = word_times.with_soft_room(saved, positivity.silences(), positivity.soft())
    words = word_times._sounds_clear_of_words(roomy, positivity.soft())
    assert [(w["start"], w["end"]) for w in words if w.get("type") == "event"] == [(20.8, 20.95), (23.75, 24.1)]


def test_a_span_placed_before_words_had_room_moves_with_its_words(video):
    project = short_word.on_video(video)
    words = word_times.read(project).words
    # "take" picked by the times the aligner gave it, as a double-click sent them.
    assert word_times.on_measured_times(*short_word.TAKE_AS_ALIGNED, words, MEASURED) == short_word.TAKE
    # "just take one": from the start of "just" to the end of "one", as they were.
    assert word_times.on_measured_times(1.32, 2.518, words, MEASURED) == (1.32, 2.518)
    assert word_times.on_measured_times(1.6, 1.8, words, MEASURED) == (1.6, 1.8), "a span in a pause stays where it is"


def test_a_cut_placed_in_a_pause_takes_no_part_of_a_word_that_got_room(video):
    project = short_word.on_video(video)
    words = word_times.read(project).words
    # Claude cut the quiet after "take" as the aligner had it, from 1.95 to 2.39. The word's sound is in there.
    assert word_times.on_measured_times(1.95, 2.39, words, MEASURED) == (2.162, 2.333)
    assert word_times.on_measured_times(1.95, 2.1, words, MEASURED) == (1.95, 2.1), "nothing of it is left: it is returned as it was"
    # The same goes for a span picked on the transcript's own times.
    assert word_times.on_measured_times(3.85, 4.97, words) == (3.85, 4.97)
    assert word_times.on_measured_times(4.0, 5.05, words) == (4.0, 4.99), "the start of \"And\" is no part of the pause before it"


def as_the_plugin_saved_it_before(video):
    """The saved edit put back to what the plugin wrote before words had room, and the words given room now."""
    project = open_project(str(video))
    edit = saved(video)
    edit.pop("word_times_made_as")
    project.edit_path.write_text(json.dumps(edit))
    word_times.align_project(project, aligner=short_word.ShortWordAligner(), silences=short_word.silences)
    treatment.forget_recipes()


def test_an_edit_placed_before_words_had_room_is_placed_again_and_holds_the_same_words(video):
    video.with_suffix(".words.json").write_text(json.dumps(short_word.transcript()))
    project = open_project(str(video))
    # As the plugin worked before: the aligner's times as they came, with nothing known of the sound.
    word_times.align_project(project, aligner=short_word.ShortWordAligner(), silences=no_silences)
    sentence = {"start": 4.95, "end": 7.05, "reason": "an afterthought", "kind": "off_topic"}
    tools.set_edit(str(video), [sentence], **QUIET)
    ctx = ctx_for(video)
    treatment.add_cut(ctx, dict(zip(("start", "end"), short_word.TAKE_AS_ALIGNED)))
    treatment.add_keep(ctx, {**dict(zip(("start", "end"), short_word.ALONE_AS_ALIGNED)), "exact": True})
    before = treatment.page_state(ctx_for(video))
    assert saved(video)["word_times_made_as"] == word_times.ALIGNED_VERSION
    as_the_plugin_saved_it_before(video)

    ctx = treatment.load_context(project, duration=DURATION, silences=short_word.silences)
    after = treatment.page_state(ctx)
    assert after["word_times"] == MEASURED
    assert saved(video)["word_times_made_as"] == word_times.ALIGNED_VERSION
    hers = next(r for r in after["rows"] if r["by"] == "you")
    assert (hers["removed"], hers["id"]) == ("take", "y1.93-2.16")
    assert any(a <= short_word.TAKE_SOUNDS[0] and short_word.TAKE_SOUNDS[1] <= b for a, b in after["removed"]), \
        "her cut takes the whole word now"
    assert [(k["text"], k["exact"], k["id"]) for k in after["keeps"]] == [("um,", True, "k5.59-5.73")]
    claude = [r["removed"] for r in after["rows"] if r["by"] == "claude"]
    assert claude == [r["removed"] for r in before["rows"] if r["by"] == "claude"] == ["And", "it works."]
    assert not any(a < short_word.ALONE_ROOM[1] and short_word.ALONE_ROOM[0] < b for a, b in after["removed"]), \
        "the word she brought back plays whole"


def test_a_new_edit_from_claude_keeps_her_cuts_and_keeps_placed_before_words_had_room(video):
    # Claude saves a new edit before anybody opened the page again: nothing else has read the saved edit since.
    video.with_suffix(".words.json").write_text(json.dumps(short_word.transcript()))
    project = open_project(str(video))
    word_times.align_project(project, aligner=short_word.ShortWordAligner(), silences=no_silences)
    tools.set_edit(str(video), [], auto_tighten=True, **QUIET)
    ctx = ctx_for(video)
    treatment.add_cut(ctx, dict(zip(("start", "end"), short_word.TAKE_AS_ALIGNED)))
    treatment.add_keep(ctx, {**dict(zip(("start", "end"), short_word.ALONE_AS_ALIGNED)), "exact": True})
    as_the_plugin_saved_it_before(video)

    sentence = {"start": 4.95, "end": 7.05, "reason": "an afterthought", "kind": "off_topic"}
    result = tools.set_edit(str(video), [sentence], silences=short_word.silences, labels=unmeasured_labels)
    assert result["creator_cuts_kept"] == 1
    edit = saved(video)
    assert [(c["id"], c["start"], c["end"]) for c in edit["creator_cuts"]] == [("y1.93-2.16", *short_word.TAKE)]
    assert [(k["id"], k["origin"]) for k in edit["keep"]] == [("k5.59-5.73", "exact")]
    assert edits_module.MOVED_FROM not in edit, "what says the spans moved is never saved"
    hers = [c for c in edit["cuts"] if c["source"] == "you"]
    assert [(c["start"], c["end"]) for c in hers] == [short_word.TAKE], "her cut takes the whole word"
    state = treatment.page_state(treatment.load_context(project, duration=DURATION, silences=short_word.silences))
    assert [w[0] for w in state["words"] if w[3]] == ["take", "And", "it", "works."], "\"um,\" plays: she brought it back"


def test_an_edit_is_loaded_with_its_spans_on_the_word_times_the_project_has(video):
    project = short_word.on_video(video)
    by_hand = {"version": 1, "video": str(video), "duration": DURATION, "word_times": MEASURED,
               "cuts": [{"start": 1.941, "end": 2.041, "reason": "Cut by you", "source": "you"}],
               "creator_cuts": [{"id": "y1.94-1.94", "start": 1.941, "end": 1.942, "text": "take", "words": 1}]}
    project.edit_path.write_text(json.dumps(by_hand))
    loaded = edits_module.load_edit(project, DURATION)
    assert loaded["creator_cuts"] == [{"id": "y1.93-2.16", "start": 1.932, "end": 2.162, "text": "take", "words": 1}]
    assert loaded[edits_module.MOVED_FROM] == MEASURED and loaded["cuts"] == by_hand["cuts"]
    assert json.loads(project.edit_path.read_text()) == by_hand, "loading writes nothing"
    # Placed on the times as they are made now, it is loaded as it was saved.
    by_hand["word_times_made_as"] = word_times.ALIGNED_VERSION
    project.edit_path.write_text(json.dumps(by_hand))
    assert edits_module.load_edit(project, DURATION) == by_hand


def test_a_rating_moves_with_its_words_like_her_other_marks(video):
    project = short_word.on_video(video)
    rating = {"id": "r1.94-1.94", "row": "c1.94-1.94", "rating": "good", "source": "claude", "kind": "other",
              "start": 1.941, "end": 1.942, "reason": "why"}
    project.edit_path.write_text(json.dumps({
        "version": 1, "video": str(video), "duration": DURATION, "word_times": MEASURED, "cuts": [],
        "ratings": [rating],
    }))
    (loaded,) = edits_module.load_edit(project, DURATION)["ratings"]
    assert (loaded["id"], loaded["start"], loaded["end"]) == ("r1.93-2.16", 1.932, 2.162)
    assert loaded["rating"] == "good" and loaded["reason"] == "why"
