"""The review page's feedback form: check what the creator wrote, then send it to the team's server.

The page never calls the internet. It posts to ``api/feedback`` on this
machine, and this module makes the one request, when the creator presses
Submit. A message, and a name and email only if the creator typed them, go
with the plugin version. Nothing else leaves this Mac.
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import re
import secrets
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from lumr_studio import share
from lumr_studio.errors import StudioError

log = logging.getLogger(__name__)

SCHEMA = 1
FEEDBACK_PATH = "/v1/feedback"
TIMEOUT_SECONDS = 15
MAX_MESSAGE, MAX_NAME, MAX_EMAIL = 5000, 100, 200
UNKNOWN_VERSION = "0.0.0"
# A loose check: something, an at sign, something, a dot, something. The team's server checks nothing more.
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def plugin_root() -> Path:
    """The plugin's folder: ``CLAUDE_PLUGIN_ROOT`` when Claude Code set it, else two folders above this package."""
    named = os.environ.get("CLAUDE_PLUGIN_ROOT")
    return Path(named) if named else Path(__file__).resolve().parents[2]


def plugin_version() -> str | None:
    """The version in ``.claude-plugin/plugin.json``, or None when it can't be read."""
    try:
        version = json.loads((plugin_root() / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8")).get("version")
    except (OSError, ValueError, AttributeError):
        return None
    return version if isinstance(version, str) and version else None


def _optional(body: dict[str, Any], key: str, limit: int, label: str) -> str | None:
    value = body.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise StudioError(f"The {label} must be text.")
    value = value.strip()
    if len(value) > limit:
        raise StudioError(f"The {label} is over {limit} characters. Shorten it a bit.")
    return value or None


def clean(body: dict[str, Any]) -> dict[str, str | None]:
    """The message, name and email from ``body``, checked. Anything else in it is dropped.

    Raises StudioError with a plain sentence for the creator.
    """
    message = body.get("message")
    if not isinstance(message, str) or not message.strip():
        raise StudioError("Write a few words first.")
    message = message.strip()
    if len(message) > MAX_MESSAGE:
        raise StudioError(f"The message is over {MAX_MESSAGE} characters. Shorten it a bit.")
    name = _optional(body, "name", MAX_NAME, "name")
    email = _optional(body, "email", MAX_EMAIL, "email address")
    if email is not None and not EMAIL.match(email):
        raise StudioError("That email address doesn't look right. Check it, or leave it empty.")
    return {"message": message, "name": name, "email": email}


def send_feedback(body: dict[str, Any]) -> dict[str, str]:
    """Send one piece of feedback to the team's server. Answers ``{"feedback_id"}``.

    One request, 15 seconds at most, no retry. Raises StudioError, in plain
    words, when there is no server to send to, when the message fails its
    checks, or when the server can't be reached or says no. The message is
    never logged.
    """
    url = share.share_url()
    if url is None:
        raise StudioError("Feedback can't be sent from here yet. Use the GitHub link the page offers instead.")
    fields = clean(body)
    version = plugin_version() or UNKNOWN_VERSION
    feedback_id = secrets.token_urlsafe(16)[:22]
    payload = {"schema": SCHEMA, "feedback_id": feedback_id, "plugin_version": version, **fields}
    sorry = "The feedback server answered with an error{code}. Try again later."
    try:
        request = urllib.request.Request(
            url + FEEDBACK_PATH,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": f"lumr-studio/{version}"},
            method="POST",
        )
        response = urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS)  # noqa: S310 - the address is the creator's own setting
        try:
            status = getattr(response, "status", None) or response.getcode()
        finally:
            close = getattr(response, "close", None)
            if close:
                close()
    except urllib.error.HTTPError as err:
        log.info("feedback refused: HTTP %s", err.code)
        raise StudioError(sorry.format(code=f" ({err.code})")) from None
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as err:
        log.info("feedback not sent: %s", type(err).__name__)
        raise StudioError("Couldn't reach the feedback server. Check your connection and try again.") from None
    if not isinstance(status, int) or not 200 <= status < 300:
        log.info("feedback refused: HTTP %s", status)
        raise StudioError(sorry.format(code=f" ({status})" if isinstance(status, int) else ""))
    return {"feedback_id": feedback_id}
