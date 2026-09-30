"""Where the review page's "Send feedback" goes: the address of the Lumr Studio team's server.

``LUMR_SHARE_URL`` names it. Until the server is deployed there is no default,
and the page falls back to a public GitHub issue the creator opens themselves.
"""

from __future__ import annotations

import os

SHARE_URL_ENV = "LUMR_SHARE_URL"
# The team's server, once it exists. None means feedback goes through GitHub instead.
DEFAULT_SHARE_URL: str | None = None


def share_url() -> str | None:
    """The server feedback is sent to, without a trailing slash, or None when there is none."""
    url = (os.environ.get(SHARE_URL_ENV) or DEFAULT_SHARE_URL or "").strip().rstrip("/")
    return url or None
