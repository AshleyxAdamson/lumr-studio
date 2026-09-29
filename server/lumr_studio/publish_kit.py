"""Validate and write the publish kit: titles, description, chapters, tags.

Chapter times are EDITED seconds. The checks are YouTube's chapter rules: the
first chapter starts at 0, there are at least three, and each lasts at least
10 seconds.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from lumr_studio.errors import StudioError
from lumr_studio.project import write_text_atomic

MIN_TITLES, MAX_TITLES = 2, 5
MIN_CHAPTERS = 3
MIN_CHAPTER_SECONDS = 10.0


def format_timestamp(seconds: float) -> str:
    """YouTube chapter stamp: ``M:SS``, or ``H:MM:SS`` from an hour up."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def validate_titles(titles: list[str]) -> list[str]:
    cleaned = [t.strip() for t in titles if isinstance(t, str) and t.strip()]
    if len(cleaned) != len(titles):
        raise StudioError("Every title must be a non-empty string.")
    if not MIN_TITLES <= len(cleaned) <= MAX_TITLES:
        raise StudioError(f"Give {MIN_TITLES} to {MAX_TITLES} title options; got {len(cleaned)}.")
    return cleaned


def validate_chapters(chapters: list[dict[str, Any]], edited_duration: float) -> list[dict[str, Any]]:
    """Check chapters against YouTube's rules. Returns them as ``{time, label}``.

    Raises StudioError naming the first rule broken and how to fix it.
    """
    if len(chapters) < MIN_CHAPTERS:
        raise StudioError(
            f"YouTube needs at least {MIN_CHAPTERS} chapters; got {len(chapters)}. Add more chapters."
        )
    out: list[dict[str, Any]] = []
    for i, ch in enumerate(chapters):
        t, label = ch.get("time"), ch.get("label")
        if not isinstance(t, (int, float)) or isinstance(t, bool) or not math.isfinite(t):
            raise StudioError(f"Chapter {i}: time must be a number of edited seconds.")
        if not isinstance(label, str) or not label.strip():
            raise StudioError(f"Chapter {i}: label must be a non-empty string.")
        out.append({"time": float(t), "label": label.strip()})
    if out[0]["time"] != 0:
        raise StudioError(f"The first chapter must start at 0, not {out[0]['time']}. Set its time to 0.")
    for i, (a, b) in enumerate(zip(out, out[1:]), start=1):
        if b["time"] - a["time"] < MIN_CHAPTER_SECONDS:
            raise StudioError(
                f"Chapter {i - 1} ({a['label']!r}) lasts {b['time'] - a['time']:.1f}s; YouTube needs "
                f"each chapter to be at least {MIN_CHAPTER_SECONDS:.0f}s. Move or merge chapter {i}."
            )
    last = out[-1]
    if edited_duration - last["time"] < MIN_CHAPTER_SECONDS:
        raise StudioError(
            f"The last chapter ({last['label']!r}) starts at {last['time']:.1f}s, less than "
            f"{MIN_CHAPTER_SECONDS:.0f}s before the edited video ends ({edited_duration:.1f}s). "
            "Move it earlier or drop it. Use chapter_times to convert source times."
        )
    return out


def chapter_lines(chapters: list[dict[str, Any]]) -> str:
    return "\n".join(f"{format_timestamp(c['time'])} {c['label']}" for c in chapters)


def write_publish_kit(
    folder: Path,
    *,
    titles: list[str],
    description: str,
    chapters: list[dict[str, Any]],
    tags: list[str],
    edited_duration: float,
) -> dict[str, str]:
    """Validate everything, then write the kit files. Returns ``{name: path}``.

    ``description.txt`` carries the chapter list at its end, the way YouTube
    reads chapters. ``kit.json`` holds the whole kit for tools.
    """
    titles = validate_titles(titles)
    chapters = validate_chapters(chapters, edited_duration)
    if not isinstance(description, str):
        raise StudioError("description must be a string.")
    clean_tags = [t.strip() for t in tags if isinstance(t, str) and t.strip()]
    if len(clean_tags) != len(tags):
        raise StudioError("Every tag must be a non-empty string.")

    files = {
        "titles": folder / "titles.txt",
        "description": folder / "description.txt",
        "chapters": folder / "chapters.txt",
        "tags": folder / "tags.txt",
        "kit": folder / "kit.json",
    }
    stamps = chapter_lines(chapters)
    write_text_atomic(files["titles"], "\n".join(titles) + "\n")
    write_text_atomic(files["description"], f"{description.strip()}\n\n{stamps}\n")
    write_text_atomic(files["chapters"], stamps + "\n")
    write_text_atomic(files["tags"], ", ".join(clean_tags) + "\n")
    write_text_atomic(files["kit"], json.dumps({
        "titles": titles, "description": description.strip(),
        "chapters": chapters, "tags": clean_tags,
    }, indent=2, ensure_ascii=False) + "\n")
    return {name: str(path) for name, path in files.items()}
