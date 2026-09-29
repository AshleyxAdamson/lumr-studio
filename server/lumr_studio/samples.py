"""Where to listen: three short samples and the clusters of an edit.

The creator can't listen to 20 minutes after every change, so the treatment
page offers three samples of about 30 seconds of edited playback, each
testing a different risk:

* ``rhythm``: the stretch where the most pauses come out, to hear whether
  the pace still sounds like the creator.
* ``big_cut``: around the largest of Claude's cuts, with enough before and
  after to tell whether the story still follows.
* ``joke``: around a laugh, to hear whether the timing survived.

Samples start and end at sentence boundaries. They are chosen once and saved
with the project (see ``treatment.py``), so after a change the creator compares like
with like. Only a new pick moves them, and it never repeats a stretch until
every choice has been used.

Clusters are the stretches with a lot of editing: the blocks on the page's
bar, and where previous and next hop. ``busy_sections`` finds them (the rule
and the numbers behind it are next to ``BUSY_WINDOW_SECONDS``) and
``clusters`` numbers and shades them (see ``cluster_level``).

Everything here is pure: plain lists in, plain dicts out.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from lumr_studio.transcript import _is_sentence_end, is_event

Span = tuple[float, float]

# A pause this long ends a sentence even without a full stop; transcripts
# sometimes miss the punctuation.
SENTENCE_GAP_SECONDS = 1.0

# About this much edited playback per sample: long enough for a joke with its
# setup, short enough to play three after every change.
SAMPLE_SECONDS = 30.0
# Around a big cut, half the sample before it and half after: the viewer has
# to know what was being said and hear that what follows still makes sense.
BIG_CUT_LEAD_SECONDS = 15.0
BIG_CUT_TAIL_SECONDS = 15.0
# Around a laugh the setup matters more than what follows it.
JOKE_LEAD_SECONDS = 20.0
JOKE_TAIL_SECONDS = 10.0

SAMPLE_KEYS = ("rhythm", "big_cut", "joke")
SAMPLE_LABELS = {"rhythm": "Rhythm", "big_cut": "A big cut", "joke": "A joke"}
# The kinds are chosen in this order: the two tied to one moment first, so the
# rhythm sample, which can sit almost anywhere, doesn't take their stretch.
_PICK_ORDER = ("big_cut", "joke", "rhythm")

# Every sentence here is read by the creator. None names a "trim" or an
# "edit": the page shows what comes out, and has no word for the thing itself.
WHY_RHYTHM = "The stretch where the most pauses come out."
WHY_BIG_CUT = "Around Claude's biggest cut, to hear whether the story still follows."
WHY_NEXT_BIG_CUT = "Around one of Claude's biggest cuts, to hear whether the story still follows."
WHY_JOKE = "Around a laugh, to hear whether the timing survived."
WHY_NO_PAUSES = "No pauses come out yet, so this is the busiest stretch."
WHY_NO_CUTS = "Claude made no cuts of its own, so this is the next busiest stretch."
WHY_NO_LAUGHS = "No laughs found, so this is the next busiest stretch."
WHY_USED_PAUSES = "Every stretch where pauses come out has had a turn, so this is the busiest one left."
WHY_USED_CUTS = "Each of Claude's cuts has had a turn, so this is the next busiest stretch."
WHY_USED_LAUGHS = "Each laugh has had a turn, so this is the next busiest stretch."
WHY_CROWDED = " The video is too short for three separate samples, so this one overlaps another."

# A sample is saved with its sentence, and samples saved before the page
# dropped the word "trim" hold the earlier wording. Reading one swaps it for
# today's, so her samples stay where they are and the page stays plain.
_EARLIER_WHYS = {
    "The stretch with the most pause trims.": WHY_RHYTHM,
    "No pause trims yet, so this is the stretch with the most edits.": WHY_NO_PAUSES,
    "Every stretch with pause trims has had a turn, so this is the busiest one left.": WHY_USED_PAUSES,
}


def plain_why(saved: str) -> str:
    """A saved sample's ``why`` in today's words. A sentence it doesn't know comes back as it is."""
    for earlier, now in _EARLIER_WHYS.items():
        if saved.startswith(earlier):
            return now + saved[len(earlier):]
    return saved


# ── Sentences and edited time ─────────────────────────────────────────────────


def sentence_spans(words: list[dict[str, Any]]) -> list[Span]:
    """Every sentence as a ``(start, end)`` source span, in order.

    A sentence ends after a word with a full stop, question mark or
    exclamation mark, or before a pause of ``SENTENCE_GAP_SECONDS``. A sound
    (a laugh, a breath) belongs to the sentence before it, so a punchline and
    its laugh stay together.
    """
    spans: list[list[float]] = []
    open_sentence = False
    last_end: float | None = None
    for w in sorted(words, key=lambda x: float(x["start"])):
        start, end = float(w["start"]), float(w["end"])
        if is_event(w):
            if spans:
                spans[-1][1] = max(spans[-1][1], end)
            else:
                spans.append([start, end])
                open_sentence = True
            last_end = max(last_end or end, end)
            continue
        gap_break = last_end is not None and start - last_end >= SENTENCE_GAP_SECONDS
        if open_sentence and not gap_break:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([start, end])
        open_sentence = not _is_sentence_end(w)
        last_end = end
    return [(round(a, 3), round(b, 3)) for a, b in spans]


class EditedClock:
    """Source time to edited time through the kept segments, by binary search.

    ``kept`` is ``render.kept_segments`` output. A source time inside a cut
    maps to where the cut sits in the edited video.
    """

    def __init__(self, kept: list[Span]) -> None:
        self._starts = [s for s, _ in kept]
        self._kept = kept
        self._offsets: list[float] = []
        total = 0.0
        for s, e in kept:
            self._offsets.append(total)
            total += e - s
        self.total = total

    def at(self, t: float) -> float:
        i = bisect.bisect_right(self._starts, t) - 1
        if i < 0:
            return 0.0
        s, e = self._kept[i]
        return self._offsets[i] + (min(t, e) - s)

    def length(self, start: float, end: float) -> float:
        """Edited seconds that play from source ``start`` to ``end``."""
        return max(0.0, self.at(end) - self.at(start))


def _overlaps(a: Span, b: Span) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _inside(span: Span, window: Span) -> bool:
    """Whether ``span`` lies mostly inside ``window``: its middle is in it."""
    return window[0] <= (span[0] + span[1]) / 2 <= window[1]


# ── Windows ───────────────────────────────────────────────────────────────────


def window_from(i: int, sentences: list[Span], clock: EditedClock, seconds: float = SAMPLE_SECONDS) -> Span:
    """The window that starts at sentence ``i`` and plays closest to ``seconds``, ending at a sentence end."""
    start = sentences[i][0]
    best = i
    for j in range(i, len(sentences)):
        best = j
        if clock.length(start, sentences[j][1]) >= seconds:
            if j > i and seconds - clock.length(start, sentences[j - 1][1]) < clock.length(start, sentences[j][1]) - seconds:
                best = j - 1
            break
    return start, sentences[best][1]


def window_around(anchor: Span, sentences: list[Span], clock: EditedClock, lead: float, tail: float) -> Span:
    """A window that plays about ``lead`` seconds before ``anchor`` and ``tail`` after it.

    Its start is a sentence start at or before the anchor, its end a sentence
    end at or after it, each the one whose edited distance is closest to the
    target. With no sentence on a side, the anchor's own edge is used.
    """
    a, b = anchor
    starts = [s for s, _ in sentences if s <= a]
    ends = [e for _, e in sentences if e >= b]
    start = min(starts, key=lambda s: (abs(clock.length(s, a) - lead), -s), default=a)
    end = min(ends, key=lambda e: (abs(clock.length(b, e) - tail), e), default=b)
    return start, end


# ── Choosing samples ──────────────────────────────────────────────────────────


@dataclass
class Edits:
    """What an edit does, as the sample and section rules read it.

    ``trims`` are ``(start, end, kind)`` automatic trims in the edit.
    ``cuts`` are Claude's cuts as ``(start, end)``, largest first, kept out
    only. ``laughs`` are sound labels of kind laugh, likely ones first.
    """

    trims: list[tuple[float, float, str]]
    cuts: list[Span]
    laughs: list[dict[str, Any]]

    def edits_in(self, window: Span) -> int:
        return sum(1 for s, e, _ in self.trims if _inside((s, e), window)) + sum(
            1 for c in self.cuts if _inside(c, window)
        )

    def pause_trims_in(self, window: Span) -> int:
        return sum(1 for s, e, k in self.trims if k == "pauses" and _inside((s, e), window))

    def removed_in(self, window: Span) -> float:
        spans = [(s, e) for s, e, _ in self.trims] + list(self.cuts)
        return sum(max(0.0, min(e, window[1]) - max(s, window[0])) for s, e in spans)


def _ranked_windows(sentences: list[Span], clock: EditedClock, edits: Edits) -> list[tuple[Span, int, int, float]]:
    """Every sentence-start window as ``(window, pause trims, edits, seconds removed)``."""
    out = []
    for i in range(len(sentences)):
        w = window_from(i, sentences, clock)
        out.append((w, edits.pause_trims_in(w), edits.edits_in(w), edits.removed_in(w)))
    return out


def _candidates(
    key: str, sentences: list[Span], clock: EditedClock, edits: Edits, windows: list[tuple[Span, int, int, float]]
) -> list[tuple[Span, str, Span | None]]:
    """``(window, why, anchor)`` for one kind, best first. Empty when the kind has nothing to show."""
    if key == "rhythm":
        ranked = sorted(windows, key=lambda r: (-r[1], -r[3], r[0][0]))
        return [(w, WHY_RHYTHM, None) for w, pauses, _, _ in ranked if pauses > 0]
    if key == "big_cut":
        return [
            (window_around(c, sentences, clock, BIG_CUT_LEAD_SECONDS, BIG_CUT_TAIL_SECONDS),
             WHY_BIG_CUT if n == 0 else WHY_NEXT_BIG_CUT, c)
            for n, c in enumerate(edits.cuts)
        ]
    anchors = [(float(lb["start"]), float(lb["end"])) for lb in edits.laughs]
    return [
        (window_around(a, sentences, clock, JOKE_LEAD_SECONDS, JOKE_TAIL_SECONDS), WHY_JOKE, a) for a in anchors
    ]


def _busiest(windows: list[tuple[Span, int, int, float]], why: str) -> list[tuple[Span, str, Span | None]]:
    ranked = sorted(windows, key=lambda r: (-r[2], -r[3], r[0][0]))
    return [(w, why, None) for w, _, _, _ in ranked]


_FALLBACK_WHY = {"rhythm": WHY_NO_PAUSES, "big_cut": WHY_NO_CUTS, "joke": WHY_NO_LAUGHS}
# When the kind has something to show but every one of them has been sampled.
_USED_WHY = {"rhythm": WHY_USED_PAUSES, "big_cut": WHY_USED_CUTS, "joke": WHY_USED_LAUGHS}


def choose_samples(
    sentences: list[Span],
    clock: EditedClock,
    edits: Edits,
    *,
    avoid: Iterable[Span] = (),
    allow_overlap: bool = False,
) -> list[dict[str, Any]] | None:
    """Three samples ``[{key, start, end, why, anchor, fallback}]`` in ``SAMPLE_KEYS`` order.

    No sample overlaps a span in ``avoid`` or another sample. When a kind has
    nothing to show (no laughs, no cuts by Claude, no pause trims), it takes
    the busiest free stretch instead and ``why`` says so. Returns None when
    some kind finds no free stretch at all; the caller then avoids less. With
    ``allow_overlap`` the samples may overlap each other and ``avoid``, as a
    last resort for a video too short for three.
    """
    if not sentences:
        return None
    taken = [] if allow_overlap else list(avoid)
    windows = _ranked_windows(sentences, clock, edits)
    chosen: dict[str, dict[str, Any]] = {}

    def free(options: list[tuple[Span, str, Span | None]]) -> tuple[Span, str, Span | None] | None:
        return next((o for o in options if not any(_overlaps(o[0], t) for t in taken)), None)

    def take(key: str, pick: tuple[Span, str, Span | None], fallback: bool) -> None:
        window, why, anchor = pick
        if allow_overlap and any(_overlaps(window, c["span"]) for c in chosen.values()):
            why += WHY_CROWDED
        chosen[key] = {"key": key, "span": window, "why": why, "anchor": anchor, "fallback": fallback}
        if not allow_overlap:
            taken.append(window)

    # Every kind with something of its own to show picks first, so a
    # stand-in never takes the stretch a real sample needed.
    had: dict[str, bool] = {}
    for key in _PICK_ORDER:
        options = _candidates(key, sentences, clock, edits, windows)
        had[key] = bool(options)
        pick = free(options)
        if pick is not None:
            take(key, pick, False)
    for key in _PICK_ORDER:
        if key in chosen:
            continue
        busiest = _busiest(windows, (_USED_WHY if had[key] else _FALLBACK_WHY)[key])
        pick = free(busiest)
        if pick is None:
            if not allow_overlap:
                return None
            pick = busiest[0]
        take(key, pick, True)
    return [
        {"key": k, "start": chosen[k]["span"][0], "end": chosen[k]["span"][1], "why": chosen[k]["why"],
         "anchor": list(chosen[k]["anchor"]) if chosen[k]["anchor"] else None, "fallback": chosen[k]["fallback"]}
        for k in SAMPLE_KEYS
    ]


# ── Busy sections: which stretches become clusters ────────────────────────────

# The rule: split the video into stretches of about BUSY_WINDOW_SECONDS, each
# from a sentence start to a sentence end. Score each stretch against the
# video's average for its length, by removed time and by number of edits,
# whichever is higher: time alone misses a run of many small trims, count
# alone misses one long cut. The busiest BUSY_SHARE of the stretches are
# busy, as long as they are busier than average. Busy neighbours merge into
# one section, up to MAX_SECTION_SECONDS.
#
# The busiest share, and not a fixed factor over the average, because the
# harder paces edit the whole video evenly. With measured word times a
# stretch at "max" holds 13 to 32 edits and almost none reaches 1.5 times the
# average, so the earlier rule (a factor of 1.5) found 3 sections at
# standard and 1 from fast on, and the bar had nowhere to hop.
#
# Measured on a 21 minute talking-head video with Claude's 16 cuts from the
# first real session, on measured word times (329 sentences, 30 stretches):
#
#   pace      edits  sections  lengths (s)  edits in each
#   natural    103      6      41 to 88     11 7 2 7 10 3
#   standard   229      7      41 to 88     8 18 4 11 15 10 16
#   fast       397      8      41 to 88     10 12 15 17 28 21 16 13
#   tight      469      8      41 to 88     11 13 17 19 35 23 23 17
#   hard       600      8      43 to 90     13 49 25 43 16 26 24 19
#   max        698      9      42 to 76     14 28 30 49 17 32 30 21 26
#
# A third of the video, in six to nine stops of under two minutes. At a
# share of 0.4 the sections covered 40% and three of them ran together; at
# 0.3 with a 135 s limit one section reached 133 s.
BUSY_WINDOW_SECONDS = 45.0
BUSY_SHARE = Fraction(1, 3)
# A stretch no busier than the video's average is not a place to stop at.
BUSY_AT_LEAST = 1.0
# The creator hops from section to section and listens; past two minutes a
# section is too long to take in at one stop.
MAX_SECTION_SECONDS = 120.0


def _chunks(sentences: list[Span], seconds: float) -> list[Span]:
    """The video in stretches of about ``seconds`` source time, each from a sentence start to a sentence end."""
    out: list[Span] = []
    i = 0
    while i < len(sentences):
        start = sentences[i][0]
        j = i
        while j + 1 < len(sentences) and sentences[j + 1][1] - start <= seconds:
            j += 1
        out.append((start, sentences[j][1]))
        i = j + 1
    return out


def _score(inside: list[Span], length: float, time_rate: float, count_rate: float) -> float:
    """How busy a stretch is against the video's average: 1.0 is average, by time or by count, whichever is higher."""
    if not inside or length <= 0:
        return 0.0
    return max(sum(e - s for s, e in inside) / (time_rate * length), len(inside) / (count_rate * length))


def busy_sections(
    sentences: list[Span],
    removed: list[tuple[float, float, str]],
    duration: float,
    *,
    window: float = BUSY_WINDOW_SECONDS,
    share: Fraction | float = BUSY_SHARE,
    longest: float = MAX_SECTION_SECONDS,
) -> list[dict[str, Any]]:
    """The busy stretches, in time order, as ``{start, end, cuts, seconds_removed}``.

    ``removed`` holds every span the edit removes as ``(start, end, source)``.
    A stretch counts an edit when the edit's middle is inside it. ``share``
    is how many of the stretches count as busy, and ``longest`` how long
    merged neighbours may get.
    """
    if not sentences or not removed or duration <= 0:
        return []
    time_rate = sum(e - s for s, e, _ in removed) / duration
    count_rate = len(removed) / duration
    chunks = _chunks(sentences, window)
    held = [[(s, e) for s, e, _ in removed if _inside((s, e), chunk)] for chunk in chunks]
    scores = [_score(inside, chunk[1] - chunk[0], time_rate, count_rate) for chunk, inside in zip(chunks, held)]
    chosen = max(1, round(len(chunks) * share))
    at_least = max(sorted(scores, reverse=True)[chosen - 1], BUSY_AT_LEAST)
    sections: list[dict[str, Any]] = []
    last_busy = -2
    for n, (chunk, inside, score) in enumerate(zip(chunks, held, scores)):
        if not inside or score < at_least:
            continue
        seconds = sum(e - s for s, e in inside)
        if n == last_busy + 1 and chunk[1] - sections[-1]["start"] <= longest:
            last = sections[-1]
            last["end"] = chunk[1]
            last["cuts"] += len(inside)
            last["seconds_removed"] += seconds
        else:
            sections.append({"start": chunk[0], "end": chunk[1], "cuts": len(inside), "seconds_removed": seconds})
        last_busy = n
    for s in sections:
        s["seconds_removed"] = round(s["seconds_removed"], 1)
    return sections


# ── Clusters: the busy stretches, numbered and shaded ─────────────────────────

# The bar on the page shades each cluster by how many edits it holds, in three
# levels, so the creator sees where the editing is dense without counting.
LEVEL_FEW, LEVEL_MANY, LEVEL_MOST = 1, 2, 3
# A cluster is shaded by where its count stands among the video's clusters:
# the third of them with the fewest edits is light, the third with the most
# is dark. By standing and not by a share of the top count, because at the
# harder paces every cluster holds nearly as many edits as the busiest one:
# against the top count, "max" shaded seven of nine clusters the middle
# shade. Measured on the 21 minute video with Claude's 16 cuts, on measured
# word times (edits per cluster, then the levels):
#
#   natural   11 7 2 7 10 3                  3 2 1 2 3 1
#   standard  8 18 4 11 15 10 16             1 3 1 2 2 1 3
#   fast      10 12 15 17 28 21 16 13        1 1 2 3 3 3 2 1
#   max       14 28 30 49 17 32 30 21 26     1 2 2 3 1 3 2 1 2
#
# The shares are exact fractions, so a count lands on the same side every time.
MANY_SHARE = Fraction(1, 3)
MOST_SHARE = Fraction(2, 3)


def cluster_level(edits: int, every_count: list[int]) -> int:
    """The shade of a cluster with ``edits`` edits, among clusters holding ``every_count``.

    The counts are put in order. A cluster with more edits than the one
    standing ``MOST_SHARE`` of the way up is ``LEVEL_MOST``, as is the
    busiest; one with more than the one standing ``MANY_SHARE`` of the way
    up is ``LEVEL_MANY``; the rest are ``LEVEL_FEW``. With nothing to compare
    against (one cluster, or every cluster holding the same count) it is
    ``LEVEL_MANY``: the middle shade claims neither "a few" nor "the most".
    """
    if len(set(every_count)) <= 1:
        return LEVEL_MANY
    ranked = sorted(every_count)
    steps = len(ranked) - 1
    if edits == ranked[-1] or edits > ranked[int(steps * MOST_SHARE)]:
        return LEVEL_MOST
    return LEVEL_MANY if edits > ranked[int(steps * MANY_SHARE)] else LEVEL_FEW


def clusters(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``busy_sections`` output as clusters: ``{number, start, end, edits, seconds_removed, level}``.

    ``number`` counts from 1 in time order. ``edits`` is the section's count
    of removed spans, Claude's cuts and automatic removals alike.
    """
    counts = [int(s["cuts"]) for s in sections]
    return [
        {"number": n, "start": s["start"], "end": s["end"], "edits": int(s["cuts"]),
         "seconds_removed": s["seconds_removed"], "level": cluster_level(int(s["cuts"]), counts)}
        for n, s in enumerate(sections, start=1)
    ]
