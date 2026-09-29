"""MCP wiring for the overlay tools: ``register(mcp)`` adds them to the server.

server.py calls ``register`` once. Answers and failures come from mcp_results,
the same helpers server.py uses: compact JSON text plus ``structured_content``,
and every failure as a readable tool error. ``set_overlays`` also returns its
stills as an image block, so the model sees the overlays the way ``look``
shows a join.
"""

from __future__ import annotations

import base64
from typing import Annotated, Any

from mcp.types import CallToolResult, ImageContent
from pydantic import BaseModel, ConfigDict, Field

from lumr_studio import overlay_tools
from lumr_studio.mcp_results import json_text, result, run, tool_annotations
from lumr_studio.overlays import BOX_PLACES, MAX_CLIP_VOLUME, MAX_LAYER

VideoPath = Annotated[str, Field(description="Absolute path to the source video.")]


class Overlay(BaseModel):
    """One photo or clip over the talk. Extra fields read back from get_overlays are allowed."""

    model_config = ConfigDict(extra="allow")

    file: str = Field(description="Absolute path to the creator's photo or clip. Never a file they didn't give you.")
    start: float = Field(description="When it appears, source seconds. Use a transcript line's start.")
    end: float = Field(description="When it goes, source seconds. At least 1s after start.")
    id: str | None = Field(None, description="Keep the id from get_overlays to update that overlay. Leave out for a new one.")
    place: str | None = Field(
        None, description=f"full (default) fills the frame, or a box at {', '.join(BOX_PLACES)}."
    )
    size: float | None = Field(None, description="Box only: share of the frame the box takes, 0.2 to 0.6. Default 0.4.")
    fit: str | None = Field(
        None, description="Full only: fill (crop to cover the frame) or fit (whole picture over a blurred copy). Default picks by shape."
    )
    motion: str | None = Field(None, description="Photos only: zoom_in (default), zoom_out, or still.")
    clip_in: float | None = Field(None, description="Clips only: seconds into the clip where it starts playing. Default 0.")
    clip_out: float | None = Field(None, description="Clips only: seconds into the clip where it stops. Default its end.")
    volume: float | None = Field(
        None, description=f"Clips only: 0 (default) is muted; 0.2 sits quietly under the voice; at most {MAX_CLIP_VOLUME}."
    )
    layer: int | None = Field(None, description=f"1 (default) to {MAX_LAYER}. Higher shows on top where two overlap.")
    fade: bool | None = Field(None, description="Short fade in and out. Default true.")
    note: str | None = Field(None, description="Why it goes here, in a few words, for the creator.")


def _with_picture(data: dict[str, Any]) -> CallToolResult:
    """The result as text, led by the stills picture when there is one."""
    content: list[Any] = []
    path = (data.get("picture") or {}).get("path")
    if path:
        with open(path, "rb") as fh:
            picture = base64.b64encode(fh.read()).decode("ascii")
        content.append(ImageContent(type="image", data=picture, mime_type="image/jpeg"))
    content.append(json_text(data))
    return CallToolResult(content=content, structured_content=data)


def register(mcp: Any) -> None:
    """Add set_overlays and get_overlays to the MCP server ``mcp``."""

    @mcp.tool(title="Set overlays", annotations=tool_annotations("Set overlays", read_only=False))
    def set_overlays(
        video_path: VideoPath,
        overlays: Annotated[
            list[Overlay], Field(description="Every photo and clip to show over the talk. Replaces the saved list.")
        ],
        picture: Annotated[
            bool, Field(description="Return stills of the overlays as the viewer will see them.")
        ] = True,
    ) -> CallToolResult:
        """Replace the saved overlays: the creator's own photos and clips shown over the talk while their voice keeps playing. Only use files the creator gave you. Each file is checked; each overlay gets an id. Returns saved (with where each lands in the edited video), rejected (with why), notes, and a picture with a still of each overlay. Look at the picture before telling the creator it's done. render and preview draw saved overlays."""
        data = run(overlay_tools.set_overlays, video_path, [o.model_dump() for o in overlays], picture)
        return _with_picture(data)

    @mcp.tool(title="Get overlays", annotations=tool_annotations("Get overlays", read_only=True))
    def get_overlays(video_path: VideoPath) -> CallToolResult:
        """The saved overlays with ids, the words each one covers (over), and where each lands in the edited video right now. status is shown, shortened (cuts under it or a short clip), or hidden (its words are cut, so it won't render). Read notes after the creator changes the cuts or the pace."""
        data = run(overlay_tools.get_overlays, video_path)
        return result(data)
