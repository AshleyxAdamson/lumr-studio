"""What the plugin offers: the editor slice by default, the extras when asked.

The slice is the video editor: transcribe, measure word times, pace the cuts,
review on the page, export. The publish kit and the overlays are built and
tested but switched off. ``LUMR_STUDIO_EXTRAS`` (a comma list, empty by
default) turns them on: it adds their tools here, and their skills come back
with an entry in ``plugin.json`` (README, "Turning the extras on").

This is the one list of tool names. ``server.py`` registers a tool only when
``offers`` says so, and a test holds the skills and the tool contract to it.
"""

from __future__ import annotations

import os

EXTRAS_ENV = "LUMR_STUDIO_EXTRAS"

EDITOR_TOOLS = frozenset({
    "transcribe", "read_transcript", "analyze_take", "find_words", "get_edit", "set_edit",
    "preview", "render", "look", "review", "job_status",
})

# extra name -> the tools it adds
EXTRAS: dict[str, frozenset[str]] = {
    "publish_kit": frozenset({"chapter_times", "save_publish_kit"}),
    "overlays": frozenset({"set_overlays", "get_overlays"}),
}

# extra name -> the skill folders (under ``extras/skills/``) that go with it
EXTRA_SKILLS: dict[str, tuple[str, ...]] = {
    "publish_kit": ("publish-kit",),
    "overlays": ("add-visuals",),
}

ALL_TOOLS = EDITOR_TOOLS.union(*EXTRAS.values())


def enabled_extras(value: str | None = None) -> frozenset[str]:
    """The extras switched on: ``value``, else ``LUMR_STUDIO_EXTRAS``, as a comma list.

    Raises ValueError for a name that is not an extra, so a typo stops the
    server at start instead of leaving a tool quietly missing.
    """
    text = os.environ.get(EXTRAS_ENV, "") if value is None else value
    named = {name.strip() for name in text.split(",") if name.strip()}
    unknown = sorted(named - EXTRAS.keys())
    if unknown:
        raise ValueError(f"{EXTRAS_ENV} names {', '.join(unknown)}; the extras are {', '.join(sorted(EXTRAS))}.")
    return frozenset(named)


def offers(tool: str, extras: frozenset[str]) -> bool:
    """Whether the plugin offers ``tool`` with ``extras`` on.

    A tool that is in neither list is a mistake, not a hidden tool: a new tool
    has to say which side it is on.
    """
    if tool in EDITOR_TOOLS:
        return True
    for extra, names in EXTRAS.items():
        if tool in names:
            return extra in extras
    raise ValueError(f"{tool} is in neither EDITOR_TOOLS nor EXTRAS in offering.py.")


def offered_tools(extras: frozenset[str]) -> frozenset[str]:
    return frozenset(tool for tool in ALL_TOOLS if offers(tool, extras))
