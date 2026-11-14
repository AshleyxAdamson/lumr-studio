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


def bake(*rows):
    """Made-up lines of a bread-baking video: ``(text, start)`` for words said back to back, ``(start, end)`` for a sound."""
    words = []
    for a, b in rows:
        if isinstance(a, str):
            t = b
            for token in a.split():
                words.append(w(token, round(t, 3), round(t + 0.3, 3)))
                t += 0.35
        else:
            words.append(w("[vocalization]", a, b, "event"))
    return words


def test_a_sound_long_after_a_sentence_stands_on_its_own():
    # The shape that went wrong: a line, a wait of half a minute with a noise in it, the next line.
    words = bake(("So you fold the dough over,", 10.0), (45.74, 53.74), ("Now it rests.", 55.1))
    line, noise, after = sentence_spans(words)
    assert line == (10.0, 12.05), "the line ends at its last word, not at the noise"
    assert noise == (45.74, 53.74), "the noise is a span of its own"
    assert after == (55.1, 56.1)
    # The same with a full stop before the wait, and with the next line right behind the noise.
    words = bake(("So you fold the dough over.", 10.0), (45.74, 53.74), ("Now it rests.", 53.9))
    assert sentence_spans(words) == [(10.0, 12.05), (45.74, 53.74), (53.9, 54.9)]


@pytest.mark.parametrize("pause, belongs", [(0.0, True), (0.5, True), (1.2, True), (1.5, True), (1.6, False), (3.0, False)])
def test_a_sound_belongs_to_the_sentence_before_it_up_to_the_limit(pause, belongs):
    assert samples.SOUND_AFTER_SENTENCE_SECONDS == 1.5
    words = bake(("It came out flat.", 0.0), (round(1.35 + pause, 3), round(3.35 + pause, 3)), ("Anyway, the oven.", 9.0))
    spans = sentence_spans(words)
    if belongs:
        assert spans == [(0.0, round(3.35 + pause, 3)), (9.0, 10.0)]
    else:
        assert spans == [(0.0, 1.35), (round(1.35 + pause, 3), round(3.35 + pause, 3)), (9.0, 10.0)]


def test_the_word_after_a_lone_sound_starts_a_new_sentence():
    # The line before the wait has no full stop, and the next word comes a tenth of a second after the noise.
    words = bake(("and then you", 0.0), (20.0, 21.0), ("shape it.", 21.1))
    assert sentence_spans(words) == [(0.0, 1.0), (20.0, 21.0), (21.1, 21.75)]


def test_a_sound_before_the_first_word_stands_on_its_own():
    words = bake((0.2, 0.9), ("Hello and welcome.", 1.0))
    assert sentence_spans(words) == [(0.2, 0.9), (1.0, 2.0)]


def test_an_um_inside_a_sentence_still_holds_it_together():
    words = bake(("Then let it rest and,", 0.0), (1.8, 2.2), ("the thing is, wait.", 2.3))
    assert sentence_spans(words) == [(0.0, 3.65)]


def test_a_laugh_in_bouts_stays_with_its_line_and_a_far_one_does_not():
    # Two bouts 1.2 s apart right after the punchline are one laugh. A third, six seconds later, is not.
    words = bake(("It came out flat.", 0.0), (1.6, 3.0), (4.2, 5.0), (11.0, 12.0), (13.2, 14.0), ("Anyway.", 20.0))
    assert sentence_spans(words) == [(0.0, 5.0), (11.0, 14.0), (20.0, 20.3)]


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


def talk_with_a_wait():
    """Two minutes of talk, a 40 s wait with an 8 s noise 30 s into it, two more minutes of talk. Returns ``(words, wait, noise)``."""
    before = talk(minutes=2)
    wait = (before[-1]["end"], before[-1]["end"] + 40.0)
    noise = (round(wait[0] + 30.0, 3), round(wait[0] + 38.0, 3))
    after = [w(x["word"], round(x["start"] + wait[1], 3), round(x["end"] + wait[1], 3)) for x in talk(minutes=2)]
    return before + [w("[vocalization]", *noise, "event")] + after, wait, noise


def test_no_sentence_runs_across_a_wait_to_a_noise_in_it():
    words, wait, noise = talk_with_a_wait()
    spans = sentence_spans(words)
    assert noise in spans
    assert not any(a < wait[0] + 1.0 and b > wait[0] + 1.0 for a, b in spans), "nothing spans the wait"


def test_a_window_that_starts_on_the_line_before_a_wait_ends_on_that_line_or_past_the_wait():
    words, wait, noise = talk_with_a_wait()
    sentences = sentence_spans(words)
    last = max(i for i, (_, b) in enumerate(sentences) if b <= wait[0] + 0.01)
    clock = EditedClock([(0.0, wait[1] + 120.0)])
    assert window_from(last, sentences, clock, seconds=4.0) == sentences[last], "one line, not the wait behind it"
    # With the wait cut, as Claude cuts it, thirty seconds of playback reach well past it.
    clock = EditedClock([(0.0, wait[0] + 0.2), (wait[1] - 0.2, wait[1] + 120.0)])
    start, end = window_from(last, sentences, clock)
    assert end > wait[1] and abs(clock.length(start, end) - 30.0) <= 3.0


def test_a_joke_sample_holds_a_laugh_that_stands_alone():
    # The laugh comes four seconds after its line: it is no part of the line, and the sample still plays it whole.
    words = talk(minutes=2)
    line_end = words[-1]["end"]
    alone = laugh(round(line_end + 4.0, 3), round(line_end + 6.0, 3))
    later = [w(x["word"], round(x["start"] + line_end + 8.0, 3), round(x["end"] + line_end + 8.0, 3)) for x in talk(minutes=2)]
    words = words + [w("[vocalization]", alone["start"], alone["end"], "event")] + later
    sentences = sentence_spans(words)
    assert (alone["start"], alone["end"]) in sentences
    picked = choose_samples(sentences, EditedClock([(0.0, 250.0)]), edits_for(words, laughs=[alone]))
    joke = picked[SAMPLE_KEYS.index("joke")]
    assert joke["why"] == WHY_JOKE and joke["start"] <= alone["start"] and joke["end"] >= alone["end"]
    starts, ends = starts_and_ends(words)
    assert joke["start"] in starts and joke["end"] in ends


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


def test_a_noise_in_a_wait_leaves_the_busy_stretches_as_a_silent_wait_does():
    words, wait, noise = talk_with_a_wait()
    duration = wait[1] + 120.0
    removed = [(wait[0] + 0.2, wait[1] - 0.2, "claude")] + [(60.0 * m + 5.0, 60.0 * m + 5.4, "auto") for m in (0, 1, 3, 4)]
    sections = busy_sections(sentence_spans(words), removed, duration)
    starts, ends = starts_and_ends(words)
    assert all(s["start"] in starts and s["end"] in ends for s in sections), "whole sentences and whole sounds"
    assert all(a["end"] < b["start"] for a, b in zip(sections, sections[1:]))
    assert not any(s["start"] < wait[0] and s["end"] > wait[0] + 1.0 for s in sections), "no line drags the wait along"
    # The same wait with nothing heard in it: the same stretches before it, and as many after, holding as much.
    quiet = busy_sections(sentence_spans([x for x in words if x.get("type") != "event"]), removed, duration)
    assert [s for s in sections if s["end"] <= wait[0]] == [s for s in quiet if s["end"] <= wait[0]]
    assert [(s["cuts"], s["seconds_removed"]) for s in sections] == [(s["cuts"], s["seconds_removed"]) for s in quiet]


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
