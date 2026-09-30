"""What the plugin offers, and that its skills, manifest and launch agree with it."""

import json
import re
from pathlib import Path

import pytest

from lumr_studio import offering

PLUGIN_DIR = Path(__file__).resolve().parents[2]
ALL_ON = frozenset(offering.EXTRAS)
EVERY_CHOICE = [frozenset(), frozenset({"publish_kit"}), frozenset({"overlays"}), ALL_ON]


def offered_skills(extras: frozenset[str]) -> list[Path]:
    """The skill files a session sees: everything in ``skills/``, plus each switched-on extra's own."""
    files = sorted((PLUGIN_DIR / "skills").glob("*/SKILL.md"))
    for extra in sorted(extras):
        files += [PLUGIN_DIR / "extras" / "skills" / name / "SKILL.md" for name in offering.EXTRA_SKILLS[extra]]
    return files


def names_in_backticks(path: Path) -> set[str]:
    return set(re.findall(r"`([a-z_]+)`", path.read_text()))


def test_by_default_the_plugin_offers_the_twelve_editor_tools():
    assert len(offering.EDITOR_TOOLS) == 12
    assert offering.offered_tools(frozenset()) == offering.EDITOR_TOOLS


def test_each_extra_adds_its_own_tools_and_nothing_else():
    assert offering.offered_tools(frozenset({"publish_kit"})) - offering.EDITOR_TOOLS == {"chapter_times", "save_publish_kit"}
    assert offering.offered_tools(frozenset({"overlays"})) - offering.EDITOR_TOOLS == {"set_overlays", "get_overlays"}
    assert len(offering.offered_tools(ALL_ON)) == 16


def test_no_tool_is_on_two_lists():
    lists = [offering.EDITOR_TOOLS, *offering.EXTRAS.values()]
    assert sum(len(names) for names in lists) == len(offering.ALL_TOOLS)


def test_a_tool_on_no_list_is_an_error_not_a_hidden_tool():
    with pytest.raises(ValueError, match="neither EDITOR_TOOLS nor EXTRAS"):
        offering.offers("brand_new_tool", frozenset())


@pytest.mark.parametrize("value, expected", [
    ("", frozenset()),
    ("  ", frozenset()),
    ("publish_kit", frozenset({"publish_kit"})),
    (" overlays , publish_kit,", ALL_ON),
    ("overlays,overlays", frozenset({"overlays"})),
])
def test_the_extras_come_from_a_comma_list(value, expected):
    assert offering.enabled_extras(value) == expected


def test_the_environment_variable_is_what_switches_extras_on(monkeypatch):
    monkeypatch.delenv(offering.EXTRAS_ENV, raising=False)
    assert offering.enabled_extras() == frozenset()
    monkeypatch.setenv(offering.EXTRAS_ENV, "overlays")
    assert offering.enabled_extras() == {"overlays"}


def test_a_misspelled_extra_stops_the_server_at_start():
    with pytest.raises(ValueError, match=r"LUMR_STUDIO_EXTRAS names publishkit; the extras are overlays, publish_kit"):
        offering.enabled_extras("overlays,publishkit")


@pytest.mark.parametrize("extras", EVERY_CHOICE, ids=lambda e: ",".join(sorted(e)) or "default")
def test_an_offered_skill_names_only_tools_that_are_offered(extras):
    """The flag and the skill folders can't drift apart: a skill in view never sends Claude to a tool that isn't."""
    offered = offering.offered_tools(extras)
    for skill in offered_skills(extras):
        wrongly_named = (names_in_backticks(skill) & offering.ALL_TOOLS) - offered
        assert not wrongly_named, f"{skill.parent.name} names {sorted(wrongly_named)}, which are off with extras={sorted(extras)}"


def test_the_skill_tests_read_the_skills_they_mean_to():
    assert [p.parent.name for p in offered_skills(frozenset())] == ["tight-cut"]
    assert [p.parent.name for p in offered_skills(ALL_ON)] == ["tight-cut", "add-visuals", "publish-kit"]
    assert all(p.is_file() for p in offered_skills(ALL_ON))


def test_every_extra_skill_lives_in_extras_and_none_stays_in_skills():
    on_disk = {p.parent.name for p in (PLUGIN_DIR / "extras" / "skills").glob("*/SKILL.md")}
    assert on_disk == {name for names in offering.EXTRA_SKILLS.values() for name in names}
    assert {p.parent.name for p in (PLUGIN_DIR / "skills").glob("*/SKILL.md")}.isdisjoint(on_disk)


def test_each_extra_skill_names_a_tool_of_its_own_extra():
    """A skill that named none of its extra's tools would have no reason to sit behind the flag."""
    for extra, names in offering.EXTRA_SKILLS.items():
        for name in names:
            skill = PLUGIN_DIR / "extras" / "skills" / name / "SKILL.md"
            assert names_in_backticks(skill) & offering.EXTRAS[extra], f"{name} names none of {sorted(offering.EXTRAS[extra])}"


def test_the_manifest_and_the_launch_leave_the_extras_off():
    manifest = json.loads((PLUGIN_DIR / ".claude-plugin" / "plugin.json").read_text())
    assert not any("extras" in str(path) for path in manifest.get("skills", []))
    launch = json.loads((PLUGIN_DIR / ".mcp.json").read_text())["mcpServers"]["lumr-studio"]
    assert offering.EXTRAS_ENV not in launch.get("env", {})
