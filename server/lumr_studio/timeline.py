"""Map between SOURCE time and EDITED time through the kept segments.

``kept`` is ``render.kept_segments`` output: sorted, non-overlapping source
spans that survive the edit. Edited time is those spans laid end to end.
"""

from __future__ import annotations

Segment = tuple[float, float]


def edited_duration(kept: list[Segment]) -> float:
    """Length of the edited video."""
    return sum(e - s for s, e in kept)


def to_edited_time(t: float, kept: list[Segment]) -> tuple[float, bool]:
    """Edited time for source time ``t``, and whether ``t`` falls inside a cut.

    A time inside a cut maps to the start of the next kept segment. A time
    after the last kept segment maps to the end of the edited video.
    """
    offset = 0.0
    for s, e in kept:
        if t < s:
            return offset, True
        if t <= e:
            return offset + (t - s), False
        offset += e - s
    return offset, True


def source_spans_for_window(kept: list[Segment], w0: float, w1: float) -> list[Segment]:
    """The source spans that play during edited time ``[w0, w1]``, in order."""
    spans: list[Segment] = []
    offset = 0.0
    for s, e in kept:
        seg_len = e - s
        lo = max(w0, offset)
        hi = min(w1, offset + seg_len)
        if hi > lo:
            spans.append((s + (lo - offset), s + (hi - offset)))
        offset += seg_len
        if offset >= w1:
            break
    return spans
