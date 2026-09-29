"""Where measured audio silences come from.

Edge placement and the automatic micro-cuts both get better with the real
waveform's silences. Every function that uses them takes a ``SilenceProvider``
so tests can pass fixed silences (or none) and never touch the user's LUMR_HOME.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from lumr_studio.engine import paths
from lumr_studio.engine.audio_boundaries import Silence

SilenceProvider = Callable[[Path], list[Silence]]


def measured_silences(video: Path) -> list[Silence]:
    """The production provider: the engine's ffmpeg silencedetect, cached per source.

    Caches under ``<LUMR_HOME>/media_cache/pipeline_state`` the way the ClipForge
    pipeline does, so the first call on a long video takes a while and later
    calls are instant.
    """
    return paths.silences_for(video)


# The level soft sound is measured at. The silences above are measured at
# ClipForge's -25 dB, where the "s" of "books" or the fading "n" of "common"
# already counts as silence. At this level only the room's own quiet does.
SOFT_NOISE_DB = -40.0


def measured_soft_silences(video: Path) -> list[Silence]:
    """The production provider of the quiet under soft sound: ``measured_silences`` at ``SOFT_NOISE_DB``, cached per source.

    Only ``word_times.with_room`` reads these, for the soft start and end of
    each word. Everything else plans on ``measured_silences``.
    """
    return paths.silences_for(video, noise_db=SOFT_NOISE_DB)


def no_silences(_video: Path) -> list[Silence]:
    """A provider that measures nothing: edges fall back to word timings alone."""
    return []
