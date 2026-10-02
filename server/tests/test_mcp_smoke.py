"""Start the real server over stdio, the way a plugin host would, and talk to it."""

import asyncio
import json
import os
import shutil
from pathlib import Path

import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from lumr_studio import offering

SERVER_DIR = Path(__file__).resolve().parents[1]
PLUGIN_DIR = SERVER_DIR.parent

# The tools the plugin offers by default, each with whether it is read only.
EDITOR_TOOLS = {
    "transcribe": False, "read_transcript": True, "analyze_take": True, "get_edit": True,
    "set_edit": False, "preview": False, "render": False, "job_status": True,
    "look": False, "review": False, "find_words": True,
}
# What LUMR_STUDIO_EXTRAS=publish_kit,overlays adds.
EXTRA_TOOLS = {
    "chapter_times": True, "save_publish_kit": False, "set_overlays": False, "get_overlays": True,
}
EXTRAS_ON = "publish_kit,overlays"
# What Claude must be able to read in a tool's description to use the export
# the page can start: the word each rule hangs on.
EXPORT_WORDS = {
    "render": ("already_running", "one full render"),
    "get_edit": ("export", "job_id"),
    "review": ("export", "ask the creator"),
}


@pytest.fixture
def plugin_data(tmp_path):
    """A stand-in for ${CLAUDE_PLUGIN_DATA} whose ``venv`` is the server's own ``.venv``.

    The plugin keeps its environment in ${CLAUDE_PLUGIN_DATA}/venv. Here that name
    is a link to ``.venv``, the environment this suite already runs in, so the
    launcher sets VIRTUAL_ENV exactly as it ships and uv still finds the packages
    in ``.venv``. Nothing is built in a second place.
    """
    data = tmp_path / "plugin-data"
    data.mkdir()
    (data / "venv").symlink_to(SERVER_DIR / ".venv", target_is_directory=True)
    return data


def server_params(data_dir: Path, extras: str = "") -> StdioServerParameters:
    """The launch ``.mcp.json`` declares, with the two plugin folders filled in the way Claude Code fills them.

    The test reads the shipped command and runs the shipped launcher script, so
    it can't drift from either. The plugin folder is this checkout, and
    ``data_dir`` stands in for ${CLAUDE_PLUGIN_DATA} (see ``plugin_data``).
    """
    declared = json.loads((PLUGIN_DIR / ".mcp.json").read_text())["mcpServers"]["lumr-studio"]
    fill = {"${CLAUDE_PLUGIN_ROOT}": str(PLUGIN_DIR), "${CLAUDE_PLUGIN_DATA}": str(data_dir)}

    def resolved(arg: str) -> str:
        for name, value in fill.items():
            arg = arg.replace(name, value)
        return arg

    return StdioServerParameters(
        command=declared["command"],
        args=[resolved(arg) for arg in declared["args"]],
        env={
            **({"LUMR_STUDIO_EXTRAS": extras} if extras else {}),
            "LUMR_STUDIO_PROJECTS_DIR": os.environ["LUMR_STUDIO_PROJECTS_DIR"],
            "LUMR_HOME": os.environ["LUMR_HOME"],
        },
        cwd=SERVER_DIR,
    )


def check_tools(by_name, contract):
    """Every tool in ``contract`` says what it does and carries its title and hints."""
    for name, read_only in contract.items():
        tool = by_name[name]
        assert tool.description and tool.description.strip(), f"{name} has no description"
        ann = tool.annotations
        assert ann is not None and ann.title, name
        assert ann.read_only_hint is read_only, name
        assert ann.destructive_hint is False, name


@pytest.mark.skipif(not shutil.which("uv"), reason="uv not installed")
def test_server_lists_tools_and_answers_get_edit(video, plugin_data):
    async def talk():
        async with Client(server_params(plugin_data), read_timeout_seconds=120) as client:
            listed = (await client.list_tools()).tools
            edit = await client.call_tool("get_edit", {"video_path": str(video)})
            bad = await client.call_tool("get_edit", {"video_path": "relative.mp4"})
            text = await client.call_tool("read_transcript", {"video_path": str(video)})
            found = await client.call_tool("find_words", {"video_path": str(video), "words": ["we"]})
            return listed, edit, bad, text, found

    listed, edit, bad, text, found = asyncio.run(talk())

    by_name = {t.name: t for t in listed}
    assert len(listed) == len(by_name) == 11, sorted(by_name)
    assert set(by_name) == set(EDITOR_TOOLS)
    check_tools(by_name, EDITOR_TOOLS)
    for name, words in EXPORT_WORDS.items():
        said = by_name[name].description.lower()
        assert all(word in said for word in words), f"{name} must say {words}: {by_name[name].description!r}"
    # job_status says where an export's id comes from on the input itself.
    job_id = by_name["job_status"].input_schema["properties"]["job_id"]["description"]
    assert "export" in job_id and "get_edit" in job_id, job_id

    assert not edit.is_error
    assert edit.structured_content["cuts"] == []
    assert edit.structured_content["new_duration"] == pytest.approx(20.0, abs=0.1)
    assert json.loads(edit.content[0].text) == edit.structured_content
    assert "\n" not in edit.content[0].text and ", " not in edit.content[0].text  # compact JSON

    body = text.content[0].text
    assert body.startswith("[0.50-2.40] Hello")  # plain text, not a JSON object
    assert body.endswith("\nEND")
    assert text.structured_content["next_start"] is None

    places = found.content[0].text.splitlines()  # plain text, packed like the transcript
    assert not found.is_error and places[0] == "we: said 2 times, 2 clean to cut, 0 not, 0 already out."
    assert places[2].startswith("w4.350-4.500 0:04 ") and "[we]" in places[2] and places[-1] == "END"
    assert found.structured_content["words"][0]["said"] == 2
    picks = by_name["set_edit"].input_schema["properties"]["picks"]
    assert "find_words" in picks["description"] and "one switch" in picks["description"]

    assert bad.is_error
    message = bad.content[0].text
    assert "relative" in message and "Traceback" not in message


@pytest.mark.skipif(not shutil.which("uv"), reason="uv not installed")
def test_the_server_offers_only_the_editor_slice_by_default(video, plugin_data):
    async def talk():
        async with Client(server_params(plugin_data), read_timeout_seconds=120) as client:
            listed = (await client.list_tools()).tools
            hidden = await client.call_tool("set_overlays", {"video_path": str(video), "overlays": []})
            return listed, client.instructions, hidden

    listed, instructions, hidden = asyncio.run(talk())

    assert {t.name for t in listed} == set(EDITOR_TOOLS)
    assert "set_overlays" not in instructions and "overlays" not in instructions.lower()
    assert hidden.is_error and "Unknown tool" in hidden.content[0].text


@pytest.mark.skipif(not shutil.which("uv"), reason="uv not installed")
def test_the_extras_flag_brings_back_the_publish_kit_and_the_overlays(plugin_data):
    async def talk():
        async with Client(server_params(plugin_data, EXTRAS_ON), read_timeout_seconds=120) as client:
            return (await client.list_tools()).tools, client.instructions

    listed, instructions = asyncio.run(talk())

    by_name = {t.name: t for t in listed}
    assert len(listed) == len(by_name) == 15, sorted(by_name)
    assert set(by_name) == set(EDITOR_TOOLS) | set(EXTRA_TOOLS)
    check_tools(by_name, {**EDITOR_TOOLS, **EXTRA_TOOLS})
    assert "set_overlays shows the creator's own photos and clips" in instructions


def test_the_contract_here_is_the_list_the_plugin_offers():
    assert set(EDITOR_TOOLS) == offering.EDITOR_TOOLS
    assert set(EXTRA_TOOLS) == set().union(*offering.EXTRAS.values())
