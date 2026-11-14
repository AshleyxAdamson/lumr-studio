"""A project that moved: another folder, another home folder, another Mac.

The folder's name comes from the video's path, so after a move the video asks
for a folder that is not there. These tests copy a project the way a creator
does (the projects folder to a new place, the video and its transcript to a new
path, the transcript's time kept only to the second) and check that it opens
with its edit and its measured word times, and that nothing is guessed.

Made-up media only: the 20 s test pattern and the words in ``conftest``.
"""

import json
import os
import shutil
from pathlib import Path

import pytest

from conftest import FakeAligner, make_words
from lumr_studio import tools, treatment, word_times
from lumr_studio.project import (
    FOUND_STEP,
    Project,
    open_project,
    project_dir_name,
    project_for,
    projects_root,
)
from lumr_studio.silences import no_silences
from lumr_studio.tools import unmeasured_labels
from lumr_studio.word_times import ESTIMATED, MEASURED

QUIET = {"silences": no_silences, "labels": unmeasured_labels}
RETAKE = {"start": 7.9, "end": 10.75, "reason": "second take of the opening line", "kind": "repeat"}
THANKS = {"start": 15.4, "end": 17.1, "reason": "the sign-off says nothing", "kind": "off_topic"}


@pytest.fixture(autouse=True)
def fresh_recipes():
    treatment.forget_recipes()
    yield
    treatment.forget_recipes()


def edited(video: Path) -> Project:
    """A project with measured word times and an edit placed on them, as a creator leaves one."""
    project = open_project(str(video))
    word_times.align_project(project, aligner=FakeAligner(), silences=no_silences)
    tools.set_edit(str(video), [RETAKE], auto_tighten=True, **QUIET)
    return project


def with_the_key_of_an_earlier_build(project: Project) -> None:
    """Save the measured word times the way builds before the content hash did: transcript size and time."""
    path = word_times.aligned_path(project)
    data = json.loads(path.read_text())
    stat = project.words_path.stat()
    data["transcript"] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    path.write_text(json.dumps(data))


def to_the_whole_second(path: Path) -> None:
    """Round the file's modified time down to the second, as a copy to another Mac did."""
    stat = path.stat()
    assert stat.st_mtime_ns % 1_000_000_000, "the test needs a time with a fraction to lose"
    whole = stat.st_mtime_ns // 1_000_000_000 * 1_000_000_000
    os.utime(path, ns=(stat.st_atime_ns, whole))


def moved(project: Project, tmp_path: Path, monkeypatch, *, place: str = "other-mac") -> Path:
    """Move the video, its transcript and the whole projects folder somewhere else. Answers the video's new path.

    The project folder keeps its name, which was made from the old path. The
    old video is gone, as it is on another Mac.
    """
    there = tmp_path / place
    footage = there / "footage"
    footage.mkdir(parents=True)
    shutil.copytree(projects_root(), there / "projects")
    video = footage / project.video.name
    shutil.move(project.video, video)
    shutil.move(project.words_path, video.with_suffix(".words.json"))
    to_the_whole_second(video.with_suffix(".words.json"))
    monkeypatch.setenv("LUMR_STUDIO_PROJECTS_DIR", str(there / "projects"))
    return video


def folders() -> list[str]:
    return sorted(p.name for p in projects_root().iterdir() if p.is_dir())


def receipts(folder: Path) -> list[dict]:
    return [json.loads(line) for line in (folder / "receipts.jsonl").read_text().splitlines()]


# ── found after a move ────────────────────────────────────────────────────────


def test_a_project_copied_to_another_mac_opens_with_every_cut_and_measured_times(video, tmp_path, monkeypatch):
    project = edited(video)
    with_the_key_of_an_earlier_build(project)
    before = tools.get_edit(str(video), **{"labels": unmeasured_labels})
    saved_before = project.edit_path.read_text()
    assert before["cut_count"] > 1 and before["word_times"] == MEASURED

    new_video = moved(project, tmp_path, monkeypatch)
    after = tools.get_edit(str(new_video), labels=unmeasured_labels)

    assert after["cuts"] == before["cuts"] and after["cut_count"] == before["cut_count"]
    assert after["word_times"] == MEASURED
    assert folders() == [project.root.name], "the folder it had is used where it is; no second one is made"
    used = projects_root() / project.root.name
    assert (used / "edit.json").read_text() == saved_before, "reading places nothing again"
    assert after["moved_from"]["folder"] == str(used)
    assert after["moved_from"]["video"] == str(project.video)
    assert "Nothing was moved" in after["moved_from"]["note"]
    assert "project_note" not in after


def test_the_first_answer_after_a_move_says_so_and_a_receipt_keeps_it(video, tmp_path, monkeypatch):
    project = edited(video)
    new_video = moved(project, tmp_path, monkeypatch)
    used = projects_root() / project.root.name
    assert "moved_from" in tools.get_edit(str(new_video), labels=unmeasured_labels)
    assert "moved_from" not in tools.get_edit(str(new_video), labels=unmeasured_labels)
    found = [r for r in receipts(used) if r["step"] == FOUND_STEP]
    assert found == [{"at": found[0]["at"], "step": FOUND_STEP, "video": str(new_video.resolve()),
                      "saved_for": str(project.video), "folder": used.name}]


def test_it_opens_when_an_empty_folder_for_the_new_path_is_already_there(video, tmp_path, monkeypatch):
    project = edited(video)
    with_the_key_of_an_earlier_build(project)
    before = tools.get_edit(str(video), labels=unmeasured_labels)
    new_video = moved(project, tmp_path, monkeypatch)
    # What the build before this one left behind: a folder for the new path, holding two empty folders.
    left_behind = projects_root() / project_dir_name(new_video.resolve())
    (left_behind / "exports").mkdir(parents=True)
    (left_behind / "publish-kit").mkdir()

    after = tools.get_edit(str(new_video), labels=unmeasured_labels)
    assert after["cuts"] == before["cuts"] and after["cut_count"] == before["cut_count"]
    assert after["word_times"] == MEASURED
    assert after["moved_from"]["folder"] == str(projects_root() / project.root.name)

    tools.set_edit(str(new_video), [RETAKE, THANKS], auto_tighten=True, **QUIET)
    assert sorted(p.name for p in left_behind.rglob("*")) == ["exports", "publish-kit"], "nothing lands in the empty one"


def test_saving_after_a_move_writes_to_one_folder(video, tmp_path, monkeypatch):
    project = edited(video)
    new_video = moved(project, tmp_path, monkeypatch)
    used = projects_root() / project.root.name

    tools.set_edit(str(new_video), [RETAKE, THANKS], auto_tighten=True, **QUIET)

    assert folders() == [used.name]
    saved = json.loads((used / "edit.json").read_text())
    assert saved["video"] == str(new_video.resolve()), "the edit names the video where it is now"
    assert [r["reason"] for r in saved["requested"]] == [RETAKE["reason"], THANKS["reason"]]
    again = tools.get_edit(str(new_video), labels=unmeasured_labels)
    assert [c["reason"].startswith(mine["reason"]) for c, mine in zip(again["cuts"], [RETAKE, THANKS])] == [True, True]
    assert "moved_from" in again, "said on the first read, whichever tool ran before it"
    assert open_project(str(new_video)).root == used, "and it is still found once its edit names the new path"


def test_paths_inside_the_files_may_name_the_new_place_already(video, tmp_path, monkeypatch):
    """A creator's script rewrote the paths in ``edit.json`` after the copy. The folder's name still has the old hash."""
    project = edited(video)
    new_video = moved(project, tmp_path, monkeypatch)
    used = projects_root() / project.root.name
    saved = json.loads((used / "edit.json").read_text())
    saved["video"] = str(new_video.resolve())
    (used / "edit.json").write_text(json.dumps(saved))

    after = tools.get_edit(str(new_video), labels=unmeasured_labels)
    assert after["cut_count"] == len(saved["cuts"]) and after["word_times"] == MEASURED
    assert after["moved_from"]["folder"] == str(used) and folders() == [used.name]


def test_a_video_moved_to_another_folder_on_the_same_mac_keeps_its_project(video, tmp_path):
    project = edited(video)
    elsewhere = tmp_path / "archive"
    elsewhere.mkdir()
    new_video = elsewhere / video.name
    shutil.move(video, new_video)
    shutil.move(project.words_path, new_video.with_suffix(".words.json"))

    after = tools.get_edit(str(new_video), labels=unmeasured_labels)
    assert after["cut_count"] > 1 and after["word_times"] == MEASURED
    assert after["moved_from"]["video"] == str(project.video) and folders() == [project.root.name]


# ── never a guess ─────────────────────────────────────────────────────────────


def test_two_folders_that_fit_give_a_note_and_no_guess(video, tmp_path, monkeypatch):
    project = edited(video)
    new_video = moved(project, tmp_path, monkeypatch)
    first = projects_root() / project.root.name
    second = projects_root() / "talk-0123456789ab"
    shutil.copytree(first, second)
    kept = {folder: (folder / "edit.json").read_bytes() for folder in (first, second)}

    answer = tools.get_edit(str(new_video), labels=unmeasured_labels)
    assert answer["cut_count"] == 0 and "moved_from" not in answer
    assert first.name in answer["project_note"] and second.name in answer["project_note"]
    assert "did not pick one" in answer["project_note"] and "Tell the creator" in answer["project_note"]

    # An edit made now lands in the folder named for the video's own path, and the other two are as they were.
    tools.set_edit(str(new_video), [THANKS], **QUIET)
    own = projects_root() / project_dir_name(new_video.resolve())
    assert json.loads((own / "edit.json").read_text())["requested"][0]["reason"] == THANKS["reason"]
    assert {folder: (folder / "edit.json").read_bytes() for folder in (first, second)} == kept
    assert "project_note" not in tools.get_edit(str(new_video), labels=unmeasured_labels), "its own edit settles it"


def test_once_one_of_the_two_is_taken_away_the_other_is_used(video, tmp_path, monkeypatch):
    project = edited(video)
    new_video = moved(project, tmp_path, monkeypatch)
    first = projects_root() / project.root.name
    shutil.copytree(first, projects_root() / "talk-0123456789ab")
    assert open_project(str(new_video)).note
    shutil.move(projects_root() / "talk-0123456789ab", tmp_path / "set-aside")
    assert open_project(str(new_video)).root == first


def test_a_folder_that_fits_by_name_but_not_by_length_is_named_and_not_used(video, tmp_path, monkeypatch):
    project = edited(video)
    new_video = moved(project, tmp_path, monkeypatch)
    old = projects_root() / project.root.name
    saved = json.loads((old / "edit.json").read_text())
    saved["duration"] = 12.5
    (old / "edit.json").write_text(json.dumps(saved))

    answer = tools.get_edit(str(new_video), labels=unmeasured_labels)
    assert answer["cut_count"] == 0 and "moved_from" not in answer
    assert old.name in answer["project_note"]
    assert "12.50 s" in answer["project_note"] and "20.00 s" in answer["project_note"]


def test_a_copy_of_a_video_never_takes_the_edit_of_the_one_still_there(video, tmp_path):
    project = edited(video)
    saved = project.edit_path.read_bytes()
    copy = tmp_path / "copies" / video.name
    copy.parent.mkdir()
    shutil.copy(video, copy)
    shutil.copy(project.words_path, copy.with_suffix(".words.json"))

    answer = tools.get_edit(str(copy), labels=unmeasured_labels)
    assert answer["cut_count"] == 0 and "moved_from" not in answer
    assert project.root.name in answer["project_note"] and str(project.video) in answer["project_note"]
    tools.set_edit(str(copy), [THANKS], **QUIET)
    assert project.edit_path.read_bytes() == saved, "the original's edit is untouched"
    assert len(folders()) == 2


def test_a_new_video_gets_a_fresh_empty_project_and_no_note(video, tmp_path, synthetic_master):
    edited(video)
    # Another video beside other projects: another name, and an old name with nothing saved under it.
    new = tmp_path / "new" / "fresh.mp4"
    new.parent.mkdir()
    shutil.copy(synthetic_master, new)
    new.with_suffix(".words.json").write_text(json.dumps(make_words()))
    (projects_root() / "fresh-0123456789ab" / "exports").mkdir(parents=True)

    answer = tools.get_edit(str(new), labels=unmeasured_labels)
    assert answer["cut_count"] == 0 and answer["word_times"] == ESTIMATED
    assert "project_note" not in answer and "moved_from" not in answer
    project = open_project(str(new))
    assert project.root.name == project_dir_name(new.resolve()) and not project.moved_from and not project.note
    assert not project.edit_path.exists()


def test_a_video_with_its_own_saved_edit_looks_no_further(video, tmp_path, monkeypatch):
    """The state a creator reaches by hand: the old folder's files copied into the folder for the new path."""
    project = edited(video)
    new_video = moved(project, tmp_path, monkeypatch)
    old = projects_root() / project.root.name
    own = projects_root() / project_dir_name(new_video.resolve())
    shutil.copytree(old, own)

    answer = tools.get_edit(str(new_video), labels=unmeasured_labels)
    assert answer["cut_count"] > 1 and answer["word_times"] == MEASURED
    assert "project_note" not in answer and "moved_from" not in answer
    assert open_project(str(new_video)).root == own


def test_a_folder_is_chosen_without_making_one(video, tmp_path):
    project = project_for(video.resolve())
    assert project.root == projects_root() / project_dir_name(video.resolve())
    assert not projects_root().exists()
    planned = project_for(tmp_path / "not-made-yet.mp4")
    assert planned.root.name == project_dir_name(tmp_path / "not-made-yet.mp4") and not planned.note
