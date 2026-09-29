"""The tools that let Claude check its own cuts: joins on every edit, look, and job waits."""

import time

import pytest

from lumr_studio import tools
from lumr_studio.errors import StudioError
from lumr_studio.project import open_project, write_json_atomic
from lumr_studio.review import apply_to_project, row_id, save_decisions
from lumr_studio.silences import no_silences
from lumr_studio.tools import unmeasured_labels

QUIET = {"silences": no_silences, "labels": unmeasured_labels}


def laugh_label(start: float, end: float, punchline: list[float]) -> dict:
    return {
        "start": start, "end": end, "seconds": round(end - start, 3),
        "kind": "laugh", "confidence": "likely", "punchline": punchline,
    }


def fixture_laugh(_project, _words):
    """The fixture's one event (6.00-6.60) called a laugh, after "today we talk about editing"."""
    return [laugh_label(6.0, 6.6, [3.9, 6.0])]


# ── joins on every saved edit ─────────────────────────────────────────────────


def test_set_edit_reports_joins_without_the_row_dicts(video):
    result = tools.set_edit(str(video), [{"start": 3.3, "end": 3.8, "reason": "filler"}], **QUIET)
    joins = result["joins"]
    assert joins["joins"] == 1
    assert "rows" not in joins and joins["text"].startswith("1 joins")


def test_a_cut_that_ends_mid_sentence_is_flagged_with_a_fix(video):
    # Removes "this is the", leaving "...video. | part that matters."
    result = tools.set_edit(str(video), [{"start": 11.9, "end": 12.72, "reason": "trim"}], **QUIET)
    joins = result["joins"]
    assert joins["flagged"] == 1 and "mid_sentence_in" in joins["by_flag"]
    assert "fix:" in joins["text"]


def test_a_cut_that_takes_a_laugh_is_flagged(video):
    result = tools.set_edit(
        str(video), [{"start": 3.8, "end": 6.7, "reason": "first take"}],
        silences=no_silences, labels=fixture_laugh,
    )
    assert "removes_laugh" in result["joins"]["by_flag"]


def test_get_edit_checks_joins_only_when_asked(video):
    tools.set_edit(str(video), [{"start": 3.3, "end": 3.8, "reason": "filler"}], **QUIET)
    assert "joins" not in tools.get_edit(str(video))
    assert tools.get_edit(str(video), check=True, labels=unmeasured_labels)["joins"]["joins"] == 1


# ── jokes and restored spans ──────────────────────────────────────────────────


def late_laugh(_project, _words):
    """A laugh after "this is the part that matters." (12.00-14.20), which follows a 1.3s pause."""
    return [laugh_label(14.3, 15.0, [12.0, 14.2])]


def test_automatic_trims_stay_clear_of_a_joke(video):
    plain = tools.set_edit(str(video), [], auto_tighten=True, gap_length=0.4, **QUIET)
    # The pause trim at 10.75-11.95 shortens the beat before the punchline.
    guarded = tools.set_edit(
        str(video), [], auto_tighten=True, gap_length=0.4, silences=no_silences, labels=late_laugh
    )
    skipped = guarded["auto"]["skipped_to_protect_jokes"]
    assert skipped >= 1
    assert guarded["auto"]["cuts_after_merge"] < plain["auto"]["cuts_after_merge"]
    assert plain["auto"]["skipped_to_protect_jokes"] == 0


def test_a_span_the_creator_restored_outlives_set_edit(video):
    project = open_project(str(video))
    tools.set_edit(str(video), [{"start": 3.3, "end": 3.8, "reason": "filler"}], **QUIET)
    edit = tools.edits.load_edit(project, project.duration())
    edit["keep"] = [{"start": 7.9, "end": 10.75, "note": "restored on the review page"}]
    write_json_atomic(project.edit_path, edit)

    result = tools.set_edit(
        str(video),
        [{"start": 3.3, "end": 3.8, "reason": "filler"}, {"start": 7.9, "end": 10.75, "reason": "retake"}],
        **QUIET,
    )
    assert len(result["applied"]) == 1
    assert "the creator restored" in result["rejected"][0]["why"]
    assert tools.get_edit(str(video))["creator_keeps"] == edit["keep"]

    allowed = tools.set_edit(
        str(video), [{"start": 7.9, "end": 10.75, "reason": "retake"}], override_keeps=True, **QUIET
    )
    assert len(allowed["applied"]) == 1
    assert tools.get_edit(str(video))["creator_keeps"] == edit["keep"]


# ── look ──────────────────────────────────────────────────────────────────────


def test_look_writes_a_picture_into_looks(video):
    tools.set_edit(str(video), [{"start": 3.3, "end": 3.8, "reason": "filler"}], **QUIET)
    picture = tools.look(str(video), 3.4, labels=unmeasured_labels)
    project = open_project(str(video))
    path = project.looks_dir / picture["path"].rsplit("/", 1)[-1]
    assert path.exists() and path.suffix == ".png" and path.stat().st_size > 0
    assert picture["join"] is not None
    assert not list(video.parent.glob("*.png"))


def test_look_leaves_no_empty_file_when_it_fails(video):
    with pytest.raises(StudioError):
        tools.look(str(video), 3.4, span=99, labels=unmeasured_labels)
    assert not list(open_project(str(video)).looks_dir.glob("*"))


# ── job_status wait ───────────────────────────────────────────────────────────


def test_wait_holds_until_the_job_finishes(video, jobs):
    project = open_project(str(video))

    def slow(_handle):
        time.sleep(0.2)
        return {"ok": True}

    job_id = jobs.start("preview", project, slow)
    assert tools.job_status(job_id, jobs=jobs)["status"] == "running"
    done = tools.job_status(job_id, wait=5, jobs=jobs)
    assert done["status"] == "done" and done["result"] == {"ok": True}


def test_wait_out_of_range_is_refused(video, jobs):
    job_id = jobs.start("preview", open_project(str(video)), lambda _h: {})
    with pytest.raises(StudioError, match="between 0 and 50"):
        tools.job_status(job_id, wait=600, jobs=jobs)


# ── review ────────────────────────────────────────────────────────────────────


def open_review(video, **kwargs):
    """The review tool with a fake address and a browser that records what it was asked to open."""
    opened = []
    result = tools.review(
        str(video), address_for=lambda _p: "http://127.0.0.1:1/token/",
        opener=lambda url: opened.append(url) or True, labels=unmeasured_labels, silences=no_silences, **kwargs,
    )
    return result, opened


def test_review_returns_the_address_and_opens_the_browser(video):
    tools.set_edit(
        str(video),
        [{"start": 3.3, "end": 3.8, "reason": "filler"}, {"start": 11.9, "end": 12.72, "reason": "trim"}],
        **QUIET,
    )
    result, opened = open_review(video)
    assert result["url"] == "http://127.0.0.1:1/token/" and opened == [result["url"]]
    assert result["claude_cuts"] == 2 and result["need_a_look"] == 1 and result["trims"] == 0
    assert result["opened_in_browser"] is True


def test_review_can_leave_the_browser_alone(video):
    tools.set_edit(str(video), [{"start": 3.3, "end": 3.8, "reason": "filler"}], **QUIET)
    result, opened = open_review(video, open_browser=False)
    assert opened == [] and result["opened_in_browser"] is False


def test_review_with_no_cuts_says_what_to_do(video):
    with pytest.raises(StudioError, match="set_edit first"):
        open_review(video)


def test_get_edit_reports_what_the_creator_decided(video):
    project = open_project(str(video))
    tools.set_edit(
        str(video),
        [{"start": 3.3, "end": 3.8, "reason": "filler"}, {"start": 7.9, "end": 10.75, "reason": "retake"}],
        **QUIET,
    )
    assert "review" not in tools.get_edit(str(video))

    filler, retake = tools.edits.load_edit(project, project.duration())["cuts"]
    save_decisions(project, {
        row_id(filler["start"], filler["end"]): {"state": "accepted"},
        row_id(retake["start"], retake["end"]): {"state": "restored"},
    })
    waiting = tools.get_edit(str(video))["review"]
    assert waiting["waiting_for_apply"] == 1 and "press Apply" in waiting["note"]

    apply_to_project(project, silences=[])
    done = tools.get_edit(str(video))
    assert done["review"]["accepted"] == 1 and done["review"]["waiting_for_apply"] == 0
    assert [(k["start"], k["end"]) for k in done["review"]["restored"]] == [(retake["start"], retake["end"])]
    assert len(done["cuts"]) == 1

    again = tools.set_edit(
        str(video),
        [{"start": 3.3, "end": 3.8, "reason": "filler"}, {"start": 7.9, "end": 10.75, "reason": "retake"}],
        **QUIET,
    )
    assert "the creator restored" in again["rejected"][0]["why"]
