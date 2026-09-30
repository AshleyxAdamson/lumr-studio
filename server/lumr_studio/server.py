"""MCP wiring only: each tool parses its inputs and calls the plain function in tools.py.

Speaks stdio. The SDK's ``stdio_server`` points fd 1 at stderr while it serves,
so stray prints from the engine, the speech model or ffmpeg can never corrupt the protocol
stream; logging goes to stderr too.

Results: every tool returns a ``CallToolResult`` built by mcp_results, the one
place that decides how a tool answers and fails: compact JSON (or, for
read_transcript, the packed transcript as plain text) with the full dict as
``structured_content``. ``look`` also returns the picture itself as an image
block, so the model sees it. The overlay tools live in overlay_mcp and are
added by ``overlay_mcp.register``. Which tools exist at all is ``offering``'s
call: the editor slice by default, the publish kit and the overlays with
``LUMR_STUDIO_EXTRAS``.
"""

import base64
import logging
import sys
from collections.abc import Callable
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, ImageContent
from pydantic import BaseModel, Field

from lumr_studio import offering, overlay_mcp, tools
from lumr_studio.edit import CUT_KINDS
from lumr_studio.mcp_results import tool_annotations as _annotations
from lumr_studio.mcp_results import json_text, result as _result, run as _run

EXTRAS = offering.enabled_extras()

_INSTRUCTIONS_START = (
    "Lumr Studio edits one video at a time. Every tool except job_status takes the "
    "absolute path of the source video. Times are seconds in the SOURCE video unless a "
    "field says edited. Long work returns a job_id; call job_status with wait=50 "
    "until it is done. transcribe also measures the word times, in the same job; every "
    "answer that reads words says word_times, measured or estimated. The plugin needs two models on "
    "this Mac, downloaded once: Parakeet (2.47 GB from Hugging Face) writes the transcript, and wav2vec 2.0 "
    "(0.38 GB from PyTorch's download site) pins each word to the sound so cuts never land inside a word. "
    "Unless they already have them, tell the creator about both up front, before they pick a video, and say what each does. When transcribe answers "
    "needs_models, ask the creator before you pass download_models; nothing downloads without a yes. find_words lists every "
    "place a word is said, so you can pick the filler ones by reading. set_edit saves a draft and returns joins, a check of how every "
    "cut will sound; read it and fix what it flags. "
)
_INSTRUCTIONS_OVERLAYS = (
    "set_overlays shows the creator's own photos and clips over the talk, and render and "
    "preview draw them. "
)
_INSTRUCTIONS_END = (
    "The creator can export from the page; get_edit then names that job under export. "
    "No tool deletes or overwrites a file."
)
INSTRUCTIONS = _INSTRUCTIONS_START + (_INSTRUCTIONS_OVERLAYS if "overlays" in EXTRAS else "") + _INSTRUCTIONS_END

mcp = MCPServer("lumr-studio", instructions=INSTRUCTIONS)


def _tool(**options: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """``mcp.tool(**options)`` for a tool the plugin offers; the rest stay plain functions nobody lists."""

    def register(fn: Callable[..., Any]) -> Callable[..., Any]:
        return mcp.tool(**options)(fn) if offering.offers(fn.__name__, EXTRAS) else fn

    return register


VideoPath = Annotated[str, Field(description="Absolute path to the source video.")]
# The kinds come from edit.CUT_KINDS, the one list the server validates against.
CutKind = Literal[tuple(CUT_KINDS)]  # type: ignore[valid-type]


class Cut(BaseModel):
    start: float = Field(description="Cut start, source seconds.")
    end: float = Field(description="Cut end, source seconds.")
    reason: str = Field(description="Why this span goes, in one plain sentence the creator reads on the page. Required.")
    kind: CutKind | None = Field(
        default=None,
        description=(
            "What kind of cut this is, so the creator sees your cuts grouped: repeat (says again "
            "what was already said), false_start (a start abandoned and restarted), off_topic "
            "(a tangent, plug or aside that doesn't serve the video), or other. Default other."
        ),
    )


class Pick(BaseModel):
    id: str = Field(description="The id of one place a word is said, from find_words, such as w312.395-312.541.")
    reason: str = Field(description="A few words on why this one is filler, such as: a pause word, the sentence reads the same without it.")


class Chapter(BaseModel):
    time: float = Field(description="Chapter start in EDITED seconds.")
    label: str = Field(description="Chapter title.")


def _call(fn: Callable[..., dict[str, Any]], *args: Any, **kwargs: Any) -> CallToolResult:
    return _result(_run(fn, *args, **kwargs))


@_tool(title="Transcribe", annotations=_annotations("Transcribe", read_only=False))
def transcribe(
    video_path: VideoPath,
    force: Annotated[bool, Field(description="Re-transcribe even if a transcript exists.")] = False,
    download_models: Annotated[
        bool,
        Field(description="Download the models transcribe said were missing. Pass true only after the creator said yes to what needs_models listed."),
    ] = False,
) -> CallToolResult:
    """Transcribe the video and measure its word times, in one job. Returns {job_id}; or {status: "exists", ...} when both are there; or {status: "aligning", job_id, ...} when the transcript is there and only the word times are being measured. When a model this needs is not on the machine it downloads nothing and returns {status: "needs_models", models: [{name, size_mb, source_host, license}], total_mb, note}: tell the creator what would download, how big, from where and under which license, and ask. Only when they say yes call transcribe again with download_models true: it returns {status: "downloading", job_id}, and when that job is done call transcribe again. word_times says what the edit runs on: measured, or estimated when this machine can't measure them (word_times_note says why). On estimated times the harder paces find far less to cut. When a saved edit was placed again on measured times, the job's result carries edit with the totals before and now."""
    return _call(tools.transcribe, video_path, force, download_models)


@_tool(title="Read transcript", annotations=_annotations("Read transcript", read_only=True))
def read_transcript(
    video_path: VideoPath,
    start: Annotated[float | None, Field(description="Window start, source seconds.")] = None,
    end: Annotated[float | None, Field(description="Window end, source seconds.")] = None,
    show_cuts: Annotated[bool, Field(description="Prefix phrases the saved edit removes with CUT.")] = False,
) -> CallToolResult:
    """Packed transcript as plain text, one line per sentence or two, split at pauses and sentence ends. The text ends with `NEXT <time>` (pass it as start to continue) or `END`."""
    data = _run(tools.read_transcript, video_path, start, end, show_cuts)
    return _result(data, text=data["text"])


@_tool(title="Analyze take", annotations=_annotations("Analyze take", read_only=True))
def analyze_take(video_path: VideoPath) -> CallToolResult:
    """Compact report: duration, word count, speaking rate, longest pauses, fillers, likely retakes, word_times (measured or estimated), and what each of the six pace levels (natural, standard, fast, tight, hard, max) would trim and save."""
    return _call(tools.analyze_take, video_path)


@_tool(title="Find words", annotations=_annotations("Find words", read_only=True))
def find_words(
    video_path: VideoPath,
    words: Annotated[list[str], Field(description='One to five words or short phrases to find, such as ["like"].')],
    start: Annotated[float | None, Field(description="Go on from here, source seconds: the time after NEXT.")] = None,
) -> CallToolResult:
    """Every place a word is said, as plain text, one place a line: an id, the clock time, six words before and after, the quiet either side, OUT:<who> when the saved edit removes it already, and NOT CLEAN when a cut there would clip the word beside it. Use it to pick filler words by reading: take the word out and read the sentence again; if it means the same, the word was filler. When unsure, leave it in. Hand the ids you pick to set_edit as picks. The text ends with `NEXT <time>` (pass it as start to continue) or `END`."""
    data = _run(tools.find_words, video_path, words, start)
    return _result(data, text=data["text"])


@_tool(title="Get edit", annotations=_annotations("Get edit", read_only=True))
def get_edit(
    video_path: VideoPath,
    include_auto: Annotated[
        bool, Field(description="Also list automatic trims one by one (first 50).")
    ] = False,
    check: Annotated[
        bool, Field(description="Also return joins, the self-check of how every cut will sound.")
    ] = False,
) -> CallToolResult:
    """Your saved cuts with reasons, a count of automatic trims, total removed time, and the resulting duration. Carries creator when the creator changed something on the page: the pace, the take-out switches, cuts of yours put back (put_back), parts kept no matter what (kept), words brought back one by one (brought_back), and the cuts the creator made by hand (cuts, with cut_words counting what they say). The creator's cuts are settled: never list them in your own cuts and never ask to undo them. Learn from them. Carries taste: what this creator's corrections on their other videos teach (read its lessons). creator.fine holds the two slider values when the creator set the pace by hand (the pace reads custom); they are hers, so leave pace and gap_length out of set_edit. Carries picks when you picked filler words before. Carries export when a full render has run, from render or from Export video on the page: pass its job_id to job_status."""
    return _call(tools.get_edit, video_path, include_auto, check)


@_tool(title="Set edit", annotations=_annotations("Set edit", read_only=False))
def set_edit(
    video_path: VideoPath,
    cuts: Annotated[list[Cut], Field(description="Every span to remove, each with a reason and a kind.")],
    auto_tighten: Annotated[bool, Field(description="Also apply automatic micro-cuts on pauses and fillers.")] = False,
    gap_length: Annotated[
        float | None,
        Field(description="Overrides the pace's shortest trimmed pause, seconds (0.15 to 1.5). Usually leave it out."),
    ] = None,
    override_keeps: Annotated[
        bool,
        Field(description="Allow cuts in spans the creator restored. Only when the creator asks for the cut."),
    ] = False,
    pace: Annotated[
        str | None,
        Field(description=(
            "How hard auto_tighten trims, gentlest first: natural (only long pauses go), standard "
            "(the usual talking-head cut), fast (quick and punchy), tight, hard (almost no pause "
            "left between sentences), or max (every pause that can go does). Leave it out to keep "
            "the pace already set for this video, else the creator's usual, else standard. When "
            "the creator set the pace with the sliders it reads custom: leave pace out to keep it."
        )),
    ] = None,
    picks: Annotated[
        list[Pick] | None,
        Field(description=(
            "Single filler words you picked by reading, as ids from find_words with a short reason each. "
            "Not part of cuts: the creator has one switch for all of them on the page. Leave it out to "
            "keep the picks saved before; a list replaces them; an empty list clears them."
        )),
    ] = None,
) -> CallToolResult:
    """Replace the saved edit with a draft. Nothing is rendered. Cuts are checked, merged, and moved clear of spoken words; automatic trims steer around marked laughs but can still clip one, so read the joins; nothing cuts a part the creator kept: a cut over a word the creator brought back splits around it. The cuts the creator made by hand stay in the edit whatever you send. Returns applied, adjusted (requested vs final), rejected (with why), removed_seconds, new_duration, and joins: every join read the way a viewer hears it, with the ones likely to sound wrong listed and a fix for each. With picks it returns picks: how many are out, which were left in because cutting them would clip the word beside them, which the creator kept, and which ids were refused."""
    return _call(
        tools.set_edit, video_path, [c.model_dump() for c in cuts], auto_tighten, gap_length,
        override_keeps, pace, None if picks is None else [p.model_dump() for p in picks],
    )


@_tool(title="Preview", annotations=_annotations("Preview", read_only=False))
def preview(
    video_path: VideoPath,
    at: Annotated[float, Field(description="Source time to preview around.")],
    pad: Annotated[float, Field(description="Seconds of edited video shown before and after.")] = 6.0,
) -> CallToolResult:
    """Render a short clip of the edited result around one point. Returns {job_id}; the finished job gives the clip path."""
    return _call(tools.preview, video_path, at, pad)


@_tool(title="Render", annotations=_annotations("Render", read_only=False))
def render(video_path: VideoPath) -> CallToolResult:
    """Render the full edited video into the project's exports folder under a new name. Returns {job_id}. One full render runs per video at a time: while one is running, such as an export the creator started from the page, this starts nothing and returns that job's id with already_running true."""
    return _call(tools.render, video_path)


@_tool(title="Look at a join", annotations=_annotations("Look at a join", read_only=False))
def look(
    video_path: VideoPath,
    at: Annotated[float, Field(description="Source time. The join nearest to it is shown.")],
    span: Annotated[float, Field(description="Seconds of edited video shown either side, at most 6.")] = 3.0,
) -> CallToolResult:
    """A picture of one join as the viewer gets it: the frame before and after the cut, a strip of stills, the sound wave, the words, and what was removed. A red line marks the join. Check for a jump between the two large frames and for sound running through the line."""
    data = _run(tools.look, video_path, at, span)
    with open(data["path"], "rb") as fh:
        picture = base64.b64encode(fh.read()).decode("ascii")
    return CallToolResult(
        content=[
            ImageContent(type="image", data=picture, mime_type="image/png"),
            json_text(data),
        ],
        structured_content=data,
    )


@_tool(title="Open the edit page", annotations=_annotations("Open the edit page", read_only=False))
def review(
    video_path: VideoPath,
    open_browser: Annotated[bool, Field(description="Open the page in the creator's browser.")] = True,
) -> CallToolResult:
    """Ask the creator before calling this: it opens a window in their browser. Opens the page for the saved edit and returns its address. There the creator sets the pace, by stop or with two sliders, and what to take out (long pauses, filler words, stutters, and the filler likes you picked), puts back or keeps out each of your cuts, cuts words by hand, brings removed words back, flags parts to keep no matter what, listens to three short samples, hops through the stretches with the most editing, and can export the finished video. Every change saves to the edit at once. The page runs on this machine only and plays the source video from disk. When the creator is done, call get_edit and read creator and export."""
    return _call(tools.review, video_path, open_browser)


@_tool(title="Job status", annotations=_annotations("Job status", read_only=True))
def job_status(
    job_id: Annotated[
        str, Field(description="Id returned by transcribe, preview or render, or found under export in get_edit.")
    ],
    wait: Annotated[
        float, Field(description="Seconds to wait for the job to finish before answering, at most 50.")
    ] = 0.0,
) -> CallToolResult:
    """Status (running, done, failed), progress 0 to 1 when known, and the result or error. Pass wait=50 to hold the call until the job finishes."""
    return _call(tools.job_status, job_id, wait)


@_tool(title="Chapter times", annotations=_annotations("Chapter times", read_only=True))
def chapter_times(
    video_path: VideoPath,
    source_times: Annotated[list[float], Field(description="Times in the source video.")],
) -> CallToolResult:
    """Convert source times to edited-video times using the saved edit. A time inside a cut maps to the start of the next kept segment."""
    return _call(tools.chapter_times, video_path, source_times)


@_tool(title="Save publish kit", annotations=_annotations("Save publish kit", read_only=False))
def save_publish_kit(
    video_path: VideoPath,
    titles: Annotated[list[str], Field(description="Two to five title options.")],
    description: Annotated[str, Field(description="Video description.")],
    chapters: Annotated[list[Chapter], Field(description="Chapters in EDITED seconds; first at 0.")],
    tags: Annotated[list[str], Field(description="Tags.")],
) -> CallToolResult:
    """Check YouTube's chapter rules (first at 0, at least three, each at least 10s) and write publish-kit/. Returns the file paths."""
    return _call(
        tools.save_publish_kit, video_path, titles, description,
        [c.model_dump() for c in chapters], tags,
    )


if "overlays" in EXTRAS:
    overlay_mcp.register(mcp)


def main() -> None:
    """Console entry point: serve over stdio."""
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    mcp.run("stdio")


if __name__ == "__main__":
    main()
