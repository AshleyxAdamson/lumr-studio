"""Where the review page's "Send feedback" goes: the address of the Lumr Studio team's server.

It's the team's Cloudflare Worker by default. ``LUMR_SHARE_URL`` overrides it,
and an empty value turns sharing off: then the page falls back to a public
GitHub issue the creator opens themselves, and Help improve Lumr is hidden.
"""

from __future__ import annotations

import os

SHARE_URL_ENV = "LUMR_SHARE_URL"
# The team's server: the Worker on the repo's `worker` branch.
DEFAULT_SHARE_URL: str | None = "https://feedback.lumr-studio.workers.dev"


def share_url() -> str | None:
    """The server feedback is sent to, without a trailing slash, or None when there is none."""
    env = os.environ.get(SHARE_URL_ENV)
    url = (DEFAULT_SHARE_URL if env is None else env) or ""
    url = url.strip().rstrip("/")
    return url or None
