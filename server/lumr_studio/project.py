"""The per-video project folder: where it lives, what's in it, and how it's written.

Layout (see TOOLS.md, "Project folder"):

    <projects root>/<video stem>-<first 12 hex of sha1(resolved path)>/
        edit.json  review.json  sounds.json  receipts.jsonl
        exports/  looks/  review/  publish-kit/

The folder's name comes from the video's path, so a video that moved (to
another folder, another home folder, another Mac) asks for a folder that is
not there. ``project_for`` is the one place a project's folder is chosen, and
it finds the folder the video had before. See TOOLS.md, "A project after a
move".

The projects root is resolved at CALL time, never at import: tests change the
environment after the modules load.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lumr_studio.engine.paths import lumr_home
from lumr_studio.engine.render import probe_duration
from lumr_studio.errors import StudioError

PROJECTS_DIR_ENV = "LUMR_STUDIO_PROJECTS_DIR"
TRANSCRIPT_SUFFIX = ".words.json"
EDIT_FILE = "edit.json"
# A saved edit made for a video this close in length to another was made for that video. One
# file probed twice gives one length; the few hundredths allow for a copy whose container rounds.
SAME_LENGTH_SECONDS = 0.05
# The receipt written when a project is found in a folder made for another path.
FOUND_STEP = "project_found"


def projects_root() -> Path:
    """The folder that holds every project. The env override wins over LUMR_HOME."""
    override = os.environ.get(PROJECTS_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return lumr_home() / "studio" / "projects"


def resolve_video(video_path: str) -> Path:
    """Check ``video_path`` names an existing file and return it resolved.

    Raises StudioError when the path is relative, missing, or not a file.
    """
    if not isinstance(video_path, str) or not video_path.strip():
        raise StudioError("video_path is empty. Pass the absolute path to the source video.")
    raw = Path(video_path).expanduser()
    if not raw.is_absolute():
        raise StudioError(
            f"video_path {video_path!r} is relative. Pass the absolute path to the source video."
        )
    if not raw.exists():
        raise StudioError(f"No file at {raw}. Check the path and pass the absolute path to the video.")
    if not raw.is_file():
        raise StudioError(f"{raw} is a directory, not a video file. Pass the path to the video itself.")
    return raw.resolve()


def words_path_for(video: Path) -> Path:
    """Where the speech model writes the word-timing sidecar: beside the video, same stem."""
    return video.with_suffix(TRANSCRIPT_SUFFIX)


@dataclass(frozen=True)
class Project:
    """One video's project folder. Construct with ``open_project``."""

    video: Path
    root: Path
    # Set by ``project_for`` when ``root`` was made for this video at another path: the path
    # the edit saved there names. Empty for the folder named after the video's own path.
    moved_from: str = field(default="", compare=False)
    # Why a folder that might hold this video's edit was not used, for Claude to pass on. Else empty.
    note: str = field(default="", compare=False)

    @property
    def edit_path(self) -> Path:
        return self.root / EDIT_FILE

    @property
    def receipts_path(self) -> Path:
        return self.root / "receipts.jsonl"

    @property
    def exports_dir(self) -> Path:
        return self.root / "exports"

    @property
    def publish_kit_dir(self) -> Path:
        return self.root / "publish-kit"

    @property
    def looks_dir(self) -> Path:
        return self.root / "looks"

    @property
    def frames_dir(self) -> Path:
        return self.root / "frames"

    @property
    def words_path(self) -> Path:
        return words_path_for(self.video)

    def duration(self) -> float:
        """Source video length in seconds, probed with ffprobe.

        Raises StudioError when ffprobe cannot read a duration.
        """
        seconds = probe_duration(self.video)
        if seconds <= 0:
            raise StudioError(
                f"ffprobe could not read a duration from {self.video}. "
                "Check that it is a playable video file and that ffprobe is installed."
            )
        return seconds


def project_dir_name(video: Path) -> str:
    """``<stem>-<first 12 hex of sha1(resolved path)>``."""
    digest = hashlib.sha1(str(video).encode("utf-8")).hexdigest()[:12]
    return f"{video.stem}-{digest}"


@dataclass(frozen=True)
class _SavedElsewhere:
    """A folder named for this video's stem at another path, and what the edit saved there says."""

    folder: Path
    video: str  # the path the edit was saved for
    duration: float | None


def _saved_elsewhere(video: Path, own: Path) -> list[_SavedElsewhere]:
    """Each folder beside ``own`` named ``<this stem>-<12 hex>`` whose saved edit names a video with this file name."""
    try:
        folders = sorted(own.parent.iterdir())
    except OSError:
        return []
    named_alike = re.compile(re.escape(video.stem) + r"-[0-9a-f]{12}")
    found = []
    for folder in folders:
        if folder.name == own.name or not named_alike.fullmatch(folder.name):
            continue
        try:
            edit = json.loads((folder / EDIT_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        saved_for = edit.get("video") if isinstance(edit, dict) else None
        if not isinstance(saved_for, str) or Path(saved_for).name != video.name:
            continue
        length = edit.get("duration")
        is_length = isinstance(length, (int, float)) and not isinstance(length, bool) and math.isfinite(length)
        found.append(_SavedElsewhere(folder=folder, video=saved_for, duration=float(length) if is_length else None))
    return found


def _same_file(there: Path, video: Path) -> bool:
    """Whether ``there`` is the file at ``video`` under another spelling of its path."""
    try:
        return there.samefile(video)
    except OSError:
        return False


def _names(those: list[_SavedElsewhere]) -> str:
    return ", ".join(one.folder.name for one in those)


# What get_edit says about folders ``project_for`` set aside or used. Plain sentences, for Claude to pass on.
NOTE_SEVERAL_FIT = (
    "This video has no saved edit under its own path, but {count} folders in {root} each hold one made "
    "for a video with this file name and length: {names}. Lumr Studio did not pick one, so the edit here "
    "is empty. Tell the creator. Once only the right folder is left in the projects folder, its edit is used."
)
NOTE_OTHER_LENGTH = (
    "A saved edit made for a video with this file name but another length was not used: {names} in "
    "{root} ({lengths}; this video is {length:.2f} s). For a new recording under an old file name "
    "nothing is wrong. If the creator expects an edit here, tell them."
)
NOTE_STILL_THERE = (
    "A saved edit for a video with this file name and length was not used, because that video is still "
    "at its path: {names} in {root}, saved for {videos}. This copy starts with an empty edit. Tell the "
    "creator the saved edit is under that path."
)
NOTE_FOUND = (
    "This video's saved edit was found in a folder made for it at another path: the video was moved, or "
    "the project came from another Mac. The edit is read and saved in that folder. Nothing was moved or copied."
)


def project_for(video: Path) -> Project:
    """The project of the video at ``video``, a resolved path. The one place a project's folder is chosen. Creates nothing.

    A video with a saved edit in the folder named for its path has that
    folder, and nothing else is looked at. Without one, the video may have
    moved, and its edit then sits in a folder named for the old path. That
    folder is this video's when all of this holds:

    * it is named for the same stem, and its ``edit.json`` names a video
      with the same file name and the same length to
      ``SAME_LENGTH_SECONDS``;
    * the video it names is this one, or is no longer there. A video still
      at the path a folder was made for owns that folder, so a copy of a
      video never takes the original's edit;
    * no other folder passes the same test. A folder whose edit names this
      very path counts before one whose video is gone.

    The folder is then used where it is: ``root`` is that folder, every
    read and every save lands there, and nothing is moved, copied or
    deleted. ``moved_from`` says which path the edit there was saved for.
    When folders were set aside (several fit, or one fits by name but not
    by length, or its video is still there), ``root`` is the folder named
    for this path and ``note`` names the folders and says why, so nobody
    takes an empty edit for the whole truth.
    """
    own = projects_root() / project_dir_name(video)
    if (own / EDIT_FILE).is_file() or not video.is_file():
        return Project(video=video, root=own)
    elsewhere = _saved_elsewhere(video, own)
    length = probe_duration(video) if elsewhere else 0.0
    if length <= 0:
        return Project(video=video, root=own)
    this_path, gone, copies, other_length = [], [], [], []
    for one in elsewhere:
        same_length = one.duration is not None and abs(one.duration - length) <= SAME_LENGTH_SECONDS
        there = Path(one.video)
        if one.video == str(video) or _same_file(there, video):
            (this_path if same_length else other_length).append(one)
        elif there.exists():
            if same_length:
                copies.append(one)
        else:
            (gone if same_length else other_length).append(one)
    for fitting in (this_path, gone):
        if len(fitting) == 1:
            return Project(video=video, root=fitting[0].folder, moved_from=fitting[0].video)
        if fitting:
            note = NOTE_SEVERAL_FIT.format(count=len(fitting), root=own.parent, names=_names(fitting))
            return Project(video=video, root=own, note=note)
    said = []
    if other_length:
        lengths = ", ".join("no length saved" if one.duration is None else f"{one.duration:.2f} s" for one in other_length)
        said.append(NOTE_OTHER_LENGTH.format(root=own.parent, names=_names(other_length), lengths=lengths, length=length))
    if copies:
        videos = ", ".join(one.video for one in copies)
        said.append(NOTE_STILL_THERE.format(root=own.parent, names=_names(copies), videos=videos))
    return Project(video=video, root=own, note=" ".join(said))


def open_project(video_path: str) -> Project:
    """Validate the video path and return its project (see ``project_for``), creating the folder on first use."""
    project = project_for(resolve_video(video_path))
    for folder in (project.root, project.exports_dir, project.publish_kit_dir):
        folder.mkdir(parents=True, exist_ok=True)
    return project


def found_after_a_move(project: Project) -> dict[str, str] | None:
    """What to tell Claude, once, about a project found in a folder made for another path. None every other time.

    ``{folder, video, note}``: the folder used, the path its edit was saved
    for, and a sentence. The first call for this video's path also writes a
    ``project_found`` line in that folder's ``receipts.jsonl``, and the line
    is what makes every later call answer None.
    """
    if not project.moved_from:
        return None
    try:
        with open(project.receipts_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict) and record.get("step") == FOUND_STEP and record.get("video") == str(project.video):
                    return None
    except OSError:
        pass
    append_receipt(project, FOUND_STEP, video=str(project.video), saved_for=project.moved_from, folder=project.root.name)
    return {"folder": str(project.root), "video": project.moved_from, "note": NOTE_FOUND}


def now_iso() -> str:
    """UTC timestamp for receipts and saved files."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_text_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a temp file in the same folder, then replace.

    A reader never sees a half-written file: it sees the old one or the new one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_json_atomic(path: Path, data: Any) -> None:
    """``write_text_atomic`` for JSON (indented, UTF-8, trailing newline)."""
    write_text_atomic(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def append_receipt(project: Project, step: str, **fields: Any) -> None:
    """Append one line to ``receipts.jsonl`` recording a completed step."""
    record = {"at": now_iso(), "step": step, **fields}
    with open(project.receipts_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def reserve_unique_path(folder: Path, base: str, suffix: str) -> Path:
    """Claim a new file in ``folder`` named ``base`` + ``suffix`` and return its path.

    Adds ``-2``, ``-3``… on a clash, so an existing file is never overwritten.
    The file is created empty with exclusive-create, so two jobs started in the
    same second cannot pick the same name.
    """
    folder.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        name = f"{base}{suffix}" if n == 1 else f"{base}-{n}{suffix}"
        candidate = folder / name
        try:
            with open(candidate, "x"):
                return candidate
        except FileExistsError:
            n += 1
