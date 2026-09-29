"""Where to listen: sentence spans, the three samples, and the clusters. Pure, synthetic data."""

import pytest

from lumr_studio import samples
from lumr_studio.samples import (
    LEVEL_FEW,
    LEVEL_MANY,
    LEVEL_MOST,
    SAMPLE_KEYS,
    WHY_BIG_CUT,
    WHY_CROWDED,
    WHY_JOKE,
    WHY_NO_CUTS,
    WHY_NO_LAUGHS,
    WHY_NO_PAUSES,
    WHY_RHYTHM,
    EditedClock,
    Edits,
    MAX_SECTION_SECONDS,
    busy_sections,
    choose_samples,
    cluster_level,
    clusters,
    plain_why,
    sentence_spans,
    window_around,
    window_from,
)


def w(text, start, end, kind=None):
    entry = {"word": text, "start": start, "end": end}
    if kind:
        entry["type"] = kind
    return entry


def talk(minutes: float = 4.0, sentence: float = 5.0, gap: float = 0.4):
    """Sentences of four words, ``sentence`` seconds each, ``gap`` apart, for ``minutes``."""
    words, t = [], 0.0
    while t + sentence <= minutes * 60:
        step = sentence / 4
        for i, text in enumerate(("one", "two", "three", "four.")):
            words.append(w(text, round(t + i * step, 3), round(t + (i + 1) * step - 0.05, 3)))
        t += sentence + gap
    return words


def starts_and_ends(words):
    spans = sentence_spans(words)
    return {s for s, _ in spans}, {e for _, e in spans}


# ── sentences and edited time ─────────────────────────────────────────────────


def test_sentences_end_at_a_full_stop_or_a_long_pause_and_keep_their_laugh():
    words = [
        w("So", 0.0, 0.3), w("that", 0.35, 0.6), w("happened.", 0.65, 1.0),
        w("[vocalization]", 1.1, 2.5, "event"),
        w("Anyway", 2.8, 3.1), w("moving", 3.15, 3.5), w("on", 3.55, 3.8),
        w("next", 5.0, 5.3), w("thing?", 5.35, 5.8),
    ]
    assert sentence_spans(words) == [(0.0, 2.5), (2.8, 3.8), (5.0, 5.8)]


def test_edited_clock_skips_what_is_cut():
    clock = EditedClock([(0.0, 10.0), (20.0, 30.0)])
    assert clock.at(5.0) == 5.0
    assert clock.at(15.0) == 10.0  # inside the cut: where the cut sits
    assert clock.at(25.0) == 15.0
    assert clock.length(5.0, 25.0) == 10.0
    assert clock.total == 20.0


def test_a_window_plays_about_thirty_seconds_between_sentence_boundaries():
    words = talk()
    sentences = sentence_spans(words)
    clock = EditedClock([(0.0, 240.0)])
    start, end = window_from(3, sentences, clock)
    starts, ends = starts_and_ends(words)
    assert start in starts and end in ends
    assert abs(clock.length(start, end) - 30.0) <= 3.0


def test_a_window_counts_edited_time_so_a_cut_inside_makes_it_longer():
    words = talk()
    sentences = sentence_spans(words)
    plain = window_from(0, sentences, EditedClock([(0.0, 240.0)]))
    cut = window_from(0, sentences, EditedClock([(0.0, 10.0), (20.0, 240.0)]))
    assert cut[1] - cut[0] > plain[1] - plain[0]


def test_a_window_around_a_moment_has_room_either_side():
    words = talk()
    sentences = sentence_spans(words)
    clock = EditedClock([(0.0, 240.0)])
    start, end = window_around((100.0, 104.0), sentences, clock, 15.0, 15.0)
    assert 10.0 <= 100.0 - start <= 20.0 and 10.0 <= end - 104.0 <= 20.0


# ── choosing samples ──────────────────────────────────────────────────────────


def overlaps(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


def edits_for(words, *, pauses=(), cuts=(), laughs=()):
    trims = [(s, s + 0.3, "pauses") for s in pauses]
    return Edits(trims=trims, cuts=sorted(cuts, key=lambda c: -(c[1] - c[0])), laughs=list(laughs))


def laugh(start, end):
    return {"start": start, "end": end, "kind": "laugh", "confidence": "likely"}


def test_three_samples_each_on_its_own_risk_at_sentence_boundaries():
    words = talk()
    sentences = sentence_spans(words)
    # Many pause trims around 150 s, one big cut at 60 s, a laugh at 200 s.
    pauses = [150.0 + 5.4 * i + 5.0 for i in range(6)]
    cuts = [(54.0, 64.8), (120.0, 122.0)]
    kept = [(0.0, 54.0), (64.8, 120.0), (122.0, 240.0)]
    what = edits_for(words, pauses=pauses, cuts=cuts, laughs=[laugh(205.2, 206.0)])
    samples = choose_samples(sentences, EditedClock(kept), what)
    assert [s["key"] for s in samples] == list(SAMPLE_KEYS)
    rhythm, big, joke = samples
    assert rhythm["why"] == WHY_RHYTHM and rhythm["start"] <= 160.0 <= rhythm["end"]
    assert big["why"] == WHY_BIG_CUT and big["start"] < 54.0 and big["end"] > 64.8
    assert joke["why"] == WHY_JOKE and joke["start"] < 205.2 and joke["end"] > 206.0
    starts, ends = starts_and_ends(words)
    for s in samples:
        assert s["start"] in starts and s["end"] in ends
    assert not any(overlaps(a, b) for i, a in enumerate(samples) for b in samples[i + 1:])


def test_a_kind_with_nothing_to_show_takes_the_next_busiest_stretch_and_says_so():
    words = talk()
    what = edits_for(words, pauses=[30.0, 90.0])
    samples = choose_samples(sentence_spans(words), EditedClock([(0.0, 240.0)]), what)
    by_key = {s["key"]: s for s in samples}
    assert by_key["big_cut"]["why"] == WHY_NO_CUTS and by_key["big_cut"]["fallback"]
    assert by_key["joke"]["why"] == WHY_NO_LAUGHS and by_key["joke"]["fallback"]
    assert by_key["rhythm"]["why"] == WHY_RHYTHM


def test_no_pauses_taken_out_says_so_for_rhythm():
    words = talk()
    samples = choose_samples(sentence_spans(words), EditedClock([(0.0, 240.0)]), edits_for(words))
    assert samples[0]["why"] == WHY_NO_PAUSES


def test_new_samples_avoid_every_stretch_used_before():
    words = talk(minutes=6)
    sentences, clock = sentence_spans(words), EditedClock([(0.0, 360.0)])
    what = edits_for(words, pauses=[20.0, 25.0, 200.0])
    first = choose_samples(sentences, clock, what)
    used = [(s["start"], s["end"]) for s in first]
    second = choose_samples(sentences, clock, what, avoid=used)
    assert second is not None
    for s in second:
        assert not any(s["start"] < b and a < s["end"] for a, b in used)


def test_when_every_stretch_is_used_there_is_no_pick():
    words = talk(minutes=1.5)
    sentences = sentence_spans(words)
    assert choose_samples(sentences, EditedClock([(0.0, 90.0)]), edits_for(words), avoid=[(0.0, 90.0)]) is None


def test_a_video_too_short_for_three_still_gets_three_that_say_they_overlap():
    words = talk(minutes=0.5)
    sentences = sentence_spans(words)
    clock = EditedClock([(0.0, 30.0)])
    assert choose_samples(sentences, clock, edits_for(words)) is None
    samples = choose_samples(sentences, clock, edits_for(words), allow_overlap=True)
    assert len(samples) == 3
    assert any(s["why"].endswith(WHY_CROWDED) for s in samples)


# ── busy sections ─────────────────────────────────────────────────────────────


def test_busy_sections_mark_the_dense_stretch_only():
    words = talk(minutes=6)
    sentences = sentence_spans(words)
    # One trim a minute everywhere, and eight in a row around 3:00.
    removed = [(60.0 * m + 5.0, 60.0 * m + 5.4, "auto") for m in range(6)]
    removed += [(180.0 + 5.4 * i + 5.0, 180.0 + 5.4 * i + 5.3, "auto") for i in range(8)]
    sections = busy_sections(sentences, removed, 360.0)
    assert len(sections) == 1
    (section,) = sections
    assert section["start"] <= 185.0 and section["end"] >= 215.0
    assert section["cuts"] >= 8
    starts, ends = starts_and_ends(words)
    assert section["start"] in starts and section["end"] in ends


def test_one_long_cut_makes_its_stretch_busy():
    words = talk(minutes=6)
    removed = [(60.0 * m + 5.0, 60.0 * m + 5.4, "auto") for m in range(6)] + [(100.0, 125.0, "claude")]
    sections = busy_sections(sentence_spans(words), removed, 360.0)
    assert any(s["start"] <= 100.0 and s["end"] >= 125.0 for s in sections)


def test_busy_neighbours_merge_into_one_section_up_to_two_minutes():
    words = talk(minutes=6)
    removed = [(t, t + 0.3, "auto") for t in [100.0 + 5.4 * i for i in range(20)]]
    first, second = busy_sections(sentence_spans(words), removed, 360.0)
    assert 80.0 < first["end"] - first["start"] <= MAX_SECTION_SECONDS
    assert second["start"] > first["end"] and second["end"] - second["start"] <= MAX_SECTION_SECONDS
    assert first["cuts"] + second["cuts"] == 20
    (whole,) = busy_sections(sentence_spans(words), removed, 360.0, longest=240.0)
    assert whole["end"] - whole["start"] > 120.0 and whole["cuts"] == 20


def test_a_video_edited_evenly_all_over_still_has_places_to_hop_to():
    # The harder paces cut every few seconds from start to end: no stretch is far above the average.
    words = talk(minutes=12)
    removed = [(t + 5.0, t + 5.3, "auto") for t in [5.4 * i for i in range(130)]]
    # A few stretches hold a little more: two more edits, four more, one more, three more.
    removed += [(t, t + 0.2, "auto") for t in (50.0, 55.0, 310.0, 315.0, 320.0, 325.0, 525.0, 610.0, 615.0, 620.0)]
    sections = busy_sections(sentence_spans(words), removed, 720.0)
    assert [s["cuts"] for s in sections] == [9, 11, 10], "the stretch with one more is no busier than average"
    assert all(30.0 <= s["end"] - s["start"] <= MAX_SECTION_SECONDS for s in sections)
    covered = sum(s["end"] - s["start"] for s in sections)
    assert covered <= 0.45 * 720.0, "about a third of the video, not most of it"
    assert len({c["level"] for c in clusters(sections)}) >= 2, "a spread of shades"


def test_a_stretch_no_busier_than_average_is_not_a_section():
    words = talk(minutes=6)
    removed = [(t + 5.0, t + 5.3, "auto") for t in [5.4 * i for i in range(60)]]  # the same everywhere
    sections = busy_sections(sentence_spans(words), removed, 360.0)
    assert len(sections) <= 3


@pytest.mark.parametrize("removed", [[], None])
def test_no_edits_no_sections(removed):
    words = talk()
    assert busy_sections(sentence_spans(words), removed or [], 240.0) == []


# ── clusters: numbered and shaded ─────────────────────────────────────────────


@pytest.mark.parametrize("edits, level", [
    (18, LEVEL_MOST), (12, LEVEL_MOST),   # the third of the clusters with the most edits
    (11, LEVEL_MANY), (6, LEVEL_MANY),
    (5, LEVEL_FEW), (1, LEVEL_FEW),       # the third with the fewest
])
def test_a_cluster_is_shaded_by_where_its_count_stands_among_the_others(edits, level):
    assert cluster_level(edits, [18, 12, 11, 6, 5, 1]) == level


@pytest.mark.parametrize("counts", [
    [1, 2, 3], [10, 12, 15, 17, 15, 21, 16, 13, 16], [14, 28, 30, 49, 17, 32, 32, 21, 29], [13, 49, 25, 23, 16],
    list(range(1, 34)),
])
def test_clusters_with_nearly_the_same_counts_still_get_all_three_shades(counts):
    levels = [cluster_level(c, counts) for c in counts]
    assert set(levels) == {LEVEL_FEW, LEVEL_MANY, LEVEL_MOST}
    by_count = sorted(zip(counts, levels))
    assert [level for _, level in by_count] == sorted(levels), "more edits never reads lighter"
    assert all(levels.count(level) <= len(counts) / 2 + 1 for level in set(levels)), "no shade takes over"


def test_the_same_count_always_gets_the_same_shade():
    counts = [8, 18, 4, 8, 11, 15, 10, 7]
    assert [cluster_level(c, counts) for c in counts] == [1, 3, 1, 1, 3, 3, 2, 1]


@pytest.mark.parametrize("counts", [[7], [4, 4], [1, 1, 1]])
def test_with_nothing_to_compare_the_level_is_the_middle_one(counts):
    assert [cluster_level(c, counts) for c in counts] == [LEVEL_MANY] * len(counts)


def test_the_busiest_cluster_is_always_the_darkest_when_counts_differ():
    for counts in ([1, 2], [5, 7, 4, 5, 11, 14, 8, 6], [2, 40]):
        assert cluster_level(max(counts), counts) == LEVEL_MOST


def test_the_level_goes_by_the_number_of_edits_and_never_by_the_time_removed():
    sections = [
        {"start": 0.0, "end": 45.0, "cuts": 2, "seconds_removed": 40.0},
        {"start": 90.0, "end": 135.0, "cuts": 12, "seconds_removed": 3.0},
    ]
    assert [c["level"] for c in clusters(sections)] == [LEVEL_FEW, LEVEL_MOST]


def test_clusters_are_numbered_from_one_in_time_order():
    words = talk(minutes=6)
    removed = [(20.0 + 5.4 * i, 20.3 + 5.4 * i, "auto") for i in range(6)]
    removed += [(250.0 + 5.4 * i, 250.3 + 5.4 * i, "auto") for i in range(12)]
    found = clusters(busy_sections(sentence_spans(words), removed, 360.0))
    assert [c["number"] for c in found] == list(range(1, len(found) + 1)) and len(found) >= 2
    assert [c["start"] for c in found] == sorted(c["start"] for c in found)
    assert all(set(c) == {"number", "start", "end", "edits", "seconds_removed", "level"} for c in found)
    assert sum(c["edits"] for c in found) <= len(removed)


def test_no_sections_no_clusters():
    assert clusters([]) == []


# ── the words of a sample ─────────────────────────────────────────────────────


def test_no_sample_sentence_names_a_trim_or_an_edit():
    whys = {k: v for k, v in vars(samples).items() if k.startswith("WHY_")}
    assert len(whys) >= 10
    for name, text in whys.items():
        assert "trim" not in text.lower() and "edit" not in text.lower(), name


def test_a_sample_saved_with_the_earlier_wording_reads_in_todays():
    assert plain_why("The stretch with the most pause trims.") == WHY_RHYTHM
    assert plain_why("No pause trims yet, so this is the stretch with the most edits.") == WHY_NO_PAUSES
    crowded = "The stretch with the most pause trims." + WHY_CROWDED
    assert plain_why(crowded) == WHY_RHYTHM + WHY_CROWDED
    assert plain_why(WHY_JOKE) == WHY_JOKE
    assert plain_why("A sentence written some other way.") == "A sentence written some other way."
