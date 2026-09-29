"""Where the engine keeps its files: the Lumr home, and the silences it measured.

Hand-written. It stands in for the two small pieces of ClipForge that the copied
files don't bring (``config._resolve_lumr_home`` and ``pipeline._silences_for``,
which sit beside code that reaches OpenCV and every LLM client). It resolves the
same home and names the cache files the same way, so a take measured by the
Mac app is already measured here, and the other way round.

Both resolve at call time, never at import: tests change the environment after
the modules load.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from lumr_studio.engine.audio_boundaries import (
    DEFAULT_MIN_SILENCE,
    DEFAULT_NOISE_DB,
    Silence,
    detect_silences,
    silences_from_pairs,
    silences_to_pairs,
)


def lumr_home() -> Path:
    """The Lumr home folder: ``LUMR_HOME``, else the folder saved in ``~/.config/lumr/home.txt``, else ``~/Lumr``."""
    env = os.environ.get("LUMR_HOME")
    if env:
        return Path(env)
    saved = Path.home() / ".config" / "lumr" / "home.txt"
    if saved.exists():
        folder = saved.read_text().strip()
        if folder:
            return Path(folder)
    return Path.home() / "Lumr"


def silences_dir() -> Path:
    """Where measured silences are cached, one file per source and setting."""
    return lumr_home() / "media_cache" / "pipeline_state"


def silences_for(video: Path, noise_db: float | None = None, min_silence: float | None = None) -> list[Silence]:
    """The measured silences of ``video``, from ffmpeg once and from the cache after.

    The cache key holds the source's name and size and the resolved detection
    settings, so a changed setting misses the old file instead of reading it.
    ``None`` means the engine's default.
    """
    noise = DEFAULT_NOISE_DB if noise_db is None else float(noise_db)
    shortest = DEFAULT_MIN_SILENCE if min_silence is None else float(min_silence)
    key = f"{video.name}:{video.stat().st_size}:{noise}:{shortest}"
    slug = hashlib.sha256(key.encode()).hexdigest()[:16]
    cache = silences_dir() / f"{video.stem}_{slug}_silences.json"
    if cache.exists():
        return silences_from_pairs(json.loads(cache.read_text(encoding="utf-8")))
    silences = detect_silences(video, noise_db=noise, min_silence=shortest)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(silences_to_pairs(silences)), encoding="utf-8")
    return silences
