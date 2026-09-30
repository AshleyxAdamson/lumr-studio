"""The treatment page's logic, run with node, plus checks on what ships.

Playback is the riskiest part of the page: edited playback skips every removed
range by seeking, and a join stops at its end. That logic lives in pure
functions in the page's ``<script id="core">`` block so these tests can run
them without a browser. So does where Previous and Next land, everything the
page says about clusters, the export and the last change, which words show
struck, what a double-click on a word asks for, and the example ``?example=1``
runs from. The node tests skip when node is not installed.

The checks on what ships read the page as text: what is in the top line, what
is gone, which words never reach the screen.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from creator_words import banned_in, decimal_times_in, is_slider_length

PAGE_DIR = Path(__file__).resolve().parent.parent / "lumr_studio" / "treatment_page"
PAGE = PAGE_DIR / "index.html"
NODE = shutil.which("node")
# A node run that takes longer than this is hung, not slow.
NODE_TIMEOUT = 20
# Words the creator decided she should never have to learn. "edits" is allowed
# only as a count ("11 edits"), which the cluster tests check by example.
WORDS_GONE = re.compile(r"\btrim(?:s|med|ming)?\b|\bbusy (?:part|section)s?\b", re.I)
# The code calls the blocks on the bar clusters. She reads "More cuts here" and never this word. It is kept
# apart from WORDS_GONE because the code's own job key is the string 'cluster', which no one reads.
CLUSTER_WORD = re.compile(r"\bclusters?\b", re.I)
# The bar must be easy to grab: she found thinner ones "very hard to capture".
BAR_MIN_HEIGHT_PX = 44
# The pace control's stops, gentlest first, as the creator reads them.
PACES = ["Natural", "Standard", "Fast", "Tight", "Hard", "Max"]
# Every route the page may ask for. One more here is one more thing the server must answer.
# The samples left the page in the layout round, so it no longer asks for new ones.
ROUTES = {"api/treatment", "api/export", "api/cut", "api/cut/add", "api/cut/remove", "api/keep", "api/keep/remove",
          "api/undo", "api/usual", "api/rate"}

needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def core_source() -> str:
    """The text of the page's core script block."""
    html = PAGE.read_text(encoding="utf-8")
    match = re.search(r'<script id="core">(.*?)</script>', html, re.S)
    assert match, "index.html needs a <script id=\"core\"> block holding the pure playback logic"
    return match.group(1)


def js(expr: str) -> Any:
    """Evaluate ``expr`` with the page's ``Core`` in scope and return it, via JSON."""
    program = core_source() + f"\nprocess.stdout.write(JSON.stringify((() => {{ {expr} }})()));\n"
    proc = subprocess.run([NODE, "-"], input=program, capture_output=True, text=True, timeout=NODE_TIMEOUT)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def simulate(ranges: list[list[float]], *, start: float, duration: float, mode: str = "edited",
             stop_at: float | None = None, step: float = 0.04, seek_lands_short: float = 0.0) -> dict[str, Any]:
    """Play a fake video in node, one frame at a time, obeying ``Core.playStep``.

    Returns the seek targets, every time the fake playhead showed, and why it
    stopped. ``seek_lands_short`` makes each seek land that much early, as a
    browser can when it snaps to a frame.
    """
    return js(f"""
      const R = Core.normalizeRanges({json.dumps(ranges)}, {duration});
      let t = Core.startPoint({{ mode: {json.dumps(mode)}, ranges: R, t: {start}, duration: {duration} }});
      const seeks = [], shown = [];
      let why = 'loop limit';
      for(let i = 0; i < 100000; i++){{
        const s = Core.playStep({{ mode: {json.dumps(mode)}, ranges: R, t, stopAt: {json.dumps(stop_at)}, duration: {duration} }});
        if(s.action === 'seek'){{ seeks.push(s.to); t = s.to - {seek_lands_short}; continue; }}
        if(s.action === 'stop' || s.action === 'end'){{ why = s.action; break; }}
        shown.push(t);
        t += {step};
      }}
      return {{ seeks, shown, why, last: shown[shown.length - 1], start: shown[0] }};
    """)


def inside_any(t: float, ranges: list[list[float]]) -> bool:
    return any(s + 0.001 < t < e - 0.001 for s, e in ranges)


# ── ranges ──────────────────────────────────────────────────────────────────


@needs_node
def test_normalize_sorts_merges_touching_and_near_ranges_and_drops_empty():
    out = js("return Core.normalizeRanges([[5,6],[1,2],[2,3],[3.04,4],[7,7],[8,7],[9,10.5]], 10);")
    assert out == [[1, 4], [5, 6], [9, 10]]


@needs_node
def test_normalize_keeps_a_gap_wider_than_a_frame():
    out = js("return Core.normalizeRanges([[1,2],[2.2,3]], 10);")
    assert out == [[1, 2], [2.2, 3]]


@needs_node
def test_skip_target_inside_just_before_and_outside():
    out = js("""
      const R = [[10, 12], [20, 25]];
      return [9, 9.99, 10, 11, 11.99, 12, 15, 21, 30].map(t => Core.skipTarget(R, t));
    """)
    assert out == [None, 12, 12, 12, None, None, None, 25, None]


@needs_node
def test_next_range_start():
    out = js("const R = [[10, 12], [20, 25]]; return [0, 10.5, 12, 19.99, 26].map(t => Core.nextRangeStart(R, t));")
    assert out == [10, 20, 20, None, None]


# ── edited playback, frame by frame ─────────────────────────────────────────


@needs_node
def test_edited_playback_jumps_every_removed_range_and_never_shows_one():
    ranges = [[2.0, 3.5], [5.0, 5.3], [8.0, 12.0]]
    run = simulate(ranges, start=0, duration=15)
    assert run["seeks"] == [3.5, 5.3, 12.0]
    assert not [t for t in run["shown"] if inside_any(t, ranges)]
    assert run["why"] == "end"


@needs_node
def test_ranges_a_frame_apart_are_one_seek_not_two():
    run = simulate([[2.0, 3.0], [3.04, 4.0], [4.0, 4.5]], start=0, duration=10)
    assert run["seeks"] == [4.5]


@needs_node
def test_a_seek_that_lands_a_hair_short_does_not_loop():
    run = simulate([[2.0, 3.0], [6.0, 7.0]], start=0, duration=10, seek_lands_short=0.01)
    assert run["seeks"] == [3.0, 7.0]
    assert run["why"] == "end"


@needs_node
def test_a_range_shorter_than_a_frame_is_still_skipped():
    run = simulate([[2.01, 2.03]], start=0, duration=5)
    assert run["seeks"] == [2.03]


@needs_node
def test_play_from_inside_a_removed_range_starts_after_it():
    run = simulate([[2.0, 4.0]], start=3.0, duration=10)
    assert run["start"] == pytest.approx(4.0)


@needs_node
def test_play_from_the_end_starts_again_from_the_top():
    out = js("return Core.startPoint({ mode: 'edited', ranges: [[0, 1]], t: 10, duration: 10 });")
    assert out == 1


@needs_node
def test_a_removed_range_at_the_end_ends_playback():
    run = simulate([[8.0, 10.0]], start=7, duration=10)
    assert run["why"] == "end"
    assert run["seeks"] == []
    assert run["last"] < 8.0


@needs_node
def test_original_mode_plays_straight_through():
    ranges = [[2.0, 3.5], [8.0, 12.0]]
    run = simulate(ranges, start=0, duration=15, mode="original")
    assert run["seeks"] == []
    assert [t for t in run["shown"] if inside_any(t, ranges)]


# ── a sample or a join stops at its end ─────────────────────────────────────


@needs_node
def test_a_sample_stops_at_its_end():
    run = simulate([[102.0, 104.0]], start=100, duration=300, stop_at=130)
    assert run["why"] == "stop"
    assert run["last"] == pytest.approx(130, abs=0.05)
    assert run["seeks"] == [104.0]


@needs_node
def test_a_sample_ending_inside_a_removed_range_stops_there_rather_than_seeking_past():
    run = simulate([[128.0, 140.0]], start=100, duration=300, stop_at=130)
    assert run["why"] == "stop"
    assert run["seeks"] == []
    assert run["last"] < 128.0


@needs_node
def test_join_plan_hears_two_seconds_either_side_and_skips_the_cut_even_when_put_back():
    plan = js("return Core.joinPlan({ start: 50, end: 55 }, [[10, 11], [56, 56.5]], 300, 2, 2);")
    assert plan == {"start": 48, "stop": 57, "ranges": [[10, 11], [50, 55], [56, 56.5]]}
    run = simulate(plan["ranges"], start=plan["start"], duration=300, stop_at=plan["stop"])
    assert run["seeks"] == [55, 56.5]
    assert run["why"] == "stop"


@needs_node
def test_join_plan_near_the_start_and_end_of_the_video():
    plan = js("return Core.joinPlan({ start: 1, end: 9.5 }, [], 10, 2, 2);")
    assert plan["start"] == 0 and plan["stop"] == 10


# ── edited and source clocks ───────────────────────────────────────────────


@needs_node
def test_edited_and_source_time_round_trip():
    out = js("""
      const R = [[10, 12], [20, 25]];
      const src = [0, 5, 10, 11, 12, 19, 20, 25, 30];
      return { edited: src.map(t => Core.editedTime(R, t)), back: [0, 5, 10, 17, 18, 23].map(e => Core.sourceTime(R, e)),
               length: Core.editedLength(R, 40) };
    """)
    assert out["edited"] == [0, 5, 10, 10, 10, 17, 18, 18, 23]
    assert out["back"] == [0, 5, 12, 19, 25, 30]
    assert out["length"] == 33


# ── Previous and Next: cut to cut ───────────────────────────────────────────

# Four cuts, as the page's removed ranges: the first starts at the top of the video.
CUTS = "[[0.005, 2.657], [3.5, 9.65], [12.174, 12.745], [15.563, 15.875]]"
LEAD = 2  # JOIN_LEAD: a hop lands where Hear it cut starts


@needs_node
def test_next_goes_to_the_next_cut_and_lands_a_lead_before_it():
    out = js(f"const R = {CUTS}; return [0, 0.9, 1.5, 9, 10.174, 13.563, 20].map(t => Core.cutHop(R, t, 1, {LEAD}));")
    # at 0 the playhead already sits where the first cut lands, so Next goes to the second; from a landing, the next one
    assert out == [1, 1, 2, 2, 3, -1, -1]
    lands = js(f"const R = {CUTS}; return R.map(r => Math.max(0, r[0] - {LEAD}));")
    assert lands == [0, 1.5, 10.174, pytest.approx(13.563)]


@needs_node
def test_previous_goes_one_cut_back_and_never_lands_again_where_it_just_landed():
    out = js(f"const R = {CUTS}; return [0, 0.4, 1.5, 1.9, 2.3, 10.174, 13.563, 14.5, 30].map(t => Core.cutHop(R, t, -1, {LEAD}));")
    # just after a landing Previous goes one back; well past it, it goes back to that cut
    assert out == [-1, -1, 0, 0, 1, 1, 2, 3, 3]
    assert js("return [Core.cutHop([], 5, 1, 2), Core.cutHop([], 5, -1, 2)];") == [-1, -1]


@needs_node
def test_a_hop_goes_through_every_cut_in_order_and_back():
    out = js(f"""
      const R = {CUTS}, land = i => Math.max(0, R[i][0] - {LEAD});
      const fwd = [], back = [];
      for(let t = 0, i; (i = Core.cutHop(R, t, 1, {LEAD})) >= 0; t = land(i) - 0.01) fwd.push(i);
      for(let t = 60, i; (i = Core.cutHop(R, t, -1, {LEAD})) >= 0; t = land(i) + 0.01) back.push(i);
      return [fwd, back];
    """)
    assert out == [[1, 2, 3], [3, 2, 1, 0]], "a landing a hair off where it was sent never lands on the same cut twice"


# ── clusters: the bar ───────────────────────────────────────────────────────

CLUSTERS = """[
  { number: 1, start: 60, end: 100, clock: '1:00', clock_range: '1:00 to 1:40', edits: 4, out: '0:06 out', level: 1 },
  { number: 2, start: 450, end: 495, clock: '7:30', clock_range: '7:30 to 8:15', edits: 11, out: '0:14 out', level: 2 },
  { number: 3, start: 900, end: 960, clock: '15:00', clock_range: '15:00 to 16:00', edits: 1, out: '0:02 out', level: 3 },
]"""


@needs_node
def test_the_cluster_at_a_time():
    assert js(f"const B = {CLUSTERS}; return [59.98, 470, 495, 100, 5].map(t => Core.clusterAt(B, t));") == [0, 1, -1, -1, -1]


@needs_node
def test_a_blocks_hover_tip_is_its_name_and_its_edit_count_and_nothing_more():
    tip = js(f"const c = {CLUSTERS}[1]; return Core.clusterTip(c);")
    assert tip == {"name": "More cuts here", "edits": "11 edits"}, "two lines: what it is, then how many edits"
    said = " ".join(tip.values())
    for gone in ("to 8:15", "out", "press", "play", "pause", "Fewer", "Darker", "lighter"):
        assert gone not in said, f"the tip no longer says {gone!r}"


@needs_node
def test_the_tips_edit_count_says_one_edit_in_the_singular():
    one = js(f"const c = {CLUSTERS}[2]; return [Core.clusterTip(c).edits, Core.clusterTip(Object.assign({{}}, c, {{ edits: 17 }})).edits];")
    assert one == ["1 edit", "17 edits"]


@needs_node
def test_a_block_is_named_the_same_on_the_tip_and_for_a_screen_reader():
    out = js(f"""
      const c = {CLUSTERS}[1];
      return [Core.BLOCK_NAME, Core.clusterTip(c).name, Core.clusterName(c, false), Core.clusterName(c, true)];
    """)
    assert out[0] == out[1] == "More cuts here", "one name, held in one place"
    assert out[2] == "More cuts here, 7:30 to 8:15, 11 edits, 0:14 out. Play", "led by the name, then the numbers and what a press does"
    assert out[3] == "More cuts here, 7:30 to 8:15, 11 edits, 0:14 out. Pause"


@needs_node
def test_the_word_cluster_never_reaches_the_creator():
    """The tip, the block's name and the bar's spoken value, while playing or not, in the built-in example too."""
    said = js(f"""
      const example = Core.exampleState().clusters;
      return [...{CLUSTERS}, ...example].flatMap(c => [...Object.values(Core.clusterTip(c)),
        Core.clusterName(c, false), Core.clusterName(c, true)]);
    """)
    assert len(said) > 20 and all(said), "every line was built"
    assert not [line for line in said if CLUSTER_WORD.search(line)], "a block is never called a cluster"
    for line in said:
        assert not banned_in(line) and not WORDS_GONE.search(line), line
        assert "—" not in line and not decimal_times_in(line), line


def test_no_page_text_calls_a_block_a_cluster():
    """What the page's markup shows or says, and every sentence its scripts can build, outside comments."""
    assert not CLUSTER_WORD.search(words_on_screen()), CLUSTER_WORD.search(words_on_screen())
    said = [a or b for a, b in sentences_in_scripts() if " " in (a or b).strip()]
    assert len(said) > 100, "the scripts' sentences were found"
    assert not [s for s in said if CLUSTER_WORD.search(s)], "a script can put 'cluster' on screen"
    scripts = re.sub(r"/\*.*?\*/", " ", " ".join(re.findall(r"<script\b[^>]*>(.*?)</script>", page(), re.S)), flags=re.S)
    assert not re.search(r"in cluster|Play cluster|Pause cluster|cluster ' \+", scripts), "no spoken value or label with the word in it"


def test_the_tip_draws_a_blocks_lines_as_text_and_sits_above_the_bar():
    html = page()
    drawn = re.search(r"function showTip\(what, x\)\{(.*?)\n\}", html, re.S)
    assert drawn and "tipLine('tip-name', what.name)" in drawn.group(1) and "tipLine('tip-edits', what.edits)" in drawn.group(1)
    assert "textContent = text" in re.search(r"function tipLine\(.*?\n", html).group(0), "a line is set as text, never markup"
    assert "innerHTML" not in drawn.group(1)
    tip = re.search(r"#hovertip\{(.*?)\}", html, re.S).group(1)
    assert "bottom:calc(100% + 8px)" in tip and "top:" not in tip, "it grows upward, so a second line never covers the bar"
    assert "font-weight:600" in re.search(r"#hovertip \.tip-name\{(.*?)\}", html).group(1), "the name is bold"
    assert "showBlockTip(+b.dataset.i" in html and html.count("showBlockTip(") >= 3, "hover and focus both draw the block's tip"
    assert "hideTip" in html and "$('hovertip').hidden = true" in re.search(r"function hideTip\(\)\{(.*?)\}", html).group(0)


def test_a_block_under_the_pointer_shows_a_ring_that_reads_on_every_shade():
    html = page()
    ring = re.search(r"\.blk:hover:not\(:focus-visible\)\{([^}]*)\}", html)
    assert ring, "a hovered block draws a ring"
    assert "outline:2px solid var(--text)" in ring.group(1) and "outline-offset:1px" in ring.group(1), \
        "outside the block, on the dark track, so it reads the same on the strongest shade as on the lightest"
    assert ".blk:hover::before{" in html, "and the block lightens"
    assert "var(--text2)" not in re.search(r"\.blk:hover[^{]*\{[^}]*\}", html).group(0), "the old inside ring lost to the strongest shade"


@needs_node
def test_clusters_are_read_as_sent():
    out = js(f"return Core.clustersOf({{ clusters: {CLUSTERS}, busy: [{{ start: 1, end: 2, cuts: 9 }}] }});")
    assert [c["number"] for c in out] == [1, 2, 3]
    assert [c["level"] for c in out] == [1, 2, 3]
    assert out[1]["clock_range"] == "7:30 to 8:15" and out[1]["out"] == "0:14 out" and out[1]["edits"] == 11


@needs_node
def test_a_state_with_no_clusters_falls_back_to_busy():
    out = js("""
      return Core.clustersOf({ busy: [
        { start: 3.84, end: 47.92, clock: '0:03', cuts: 5, seconds_removed: 18.6 },
        { start: 450.2, end: 495.6, clock: '7:30', cuts: 11, seconds_removed: 14.2 },
        { start: 853.3, end: 939.3, clock: '14:13', cuts: 14, seconds_removed: 0.4 },
        { start: 1000, end: 1040, clock: '16:40', cuts: 2, seconds_removed: 3 },
      ] });
    """)
    assert [c["number"] for c in out] == [1, 2, 3, 4]
    assert [c["edits"] for c in out] == [5, 11, 14, 2]
    assert [c["level"] for c in out] == [2, 2, 2, 2], "with no level sent, every block gets the middle shade"
    assert [c["clock_range"] for c in out] == ["0:03 to 0:47", "7:30 to 8:15", "14:13 to 15:39", "16:40 to 17:20"]
    assert [c["out"] for c in out] == ["0:19 out", "0:14 out", "0:00 out", "0:03 out"]


@needs_node
def test_no_clusters_and_no_busy_is_an_empty_bar():
    assert js("return [Core.clustersOf({}), Core.clustersOf({ clusters: [] }), Core.clustersOf(null)];") == [[], [], []]


@needs_node
def test_the_shade_is_the_level_the_state_sends_and_the_page_holds_no_rule_of_its_own():
    out = js("""
      const sent = [{ start: 0, end: 10, edits: 30, level: 1 }, { start: 20, end: 30, edits: 1, level: 3 }, { start: 40, end: 50, edits: 9, level: 2 }];
      return [Core.clustersOf({ clusters: sent }).map(c => c.level), typeof Core.shadeLevels,
              Core.clustersOf({ clusters: [{ start: 0, end: 10, edits: 4, level: 7 }, { start: 20, end: 30, edits: 4 }] }).map(c => c.level)];
    """)
    assert out == [[1, 3, 2], "undefined", [2, 2]]
    html = page()
    for gone in ("MANY_SHARE", "MOST_SHARE", "shadeLevels"):
        assert gone not in html, f"{gone} is the server's rule, not the page's"


@needs_node
def test_two_clusters_that_touch_are_told_apart():
    out = js("""
      const B = [{ start: 3.6, end: 47.7 }, { start: 699.49, end: 780.77 }, { start: 780.89, end: 869.9 }, { start: 869.9, end: 900 }, { start: 903, end: 950 }];
      return [0, 1, 2, 3, 4, 5].map(i => Core.touchesBefore(B, i, 1307)).concat([Core.touchesBefore(B, 2, 0)]);
    """)
    assert out == [False, False, True, True, False, False, False]
    html = page()
    drawn = re.search(r"function renderBar\(\)\{(.*?)\n\}", html, re.S).group(1)
    assert "C.touchesBefore(CL, i, DUR) ? BLOCK_GAP_PX : 0" in drawn
    assert re.search(r"const BLOCK_GAP_PX = 2;", html), "the gap the contract asks for"


# ── photos and clips: view only ─────────────────────────────────────────────


@needs_node
def test_the_name_tag_shows_while_a_photo_or_clip_has_its_place():
    out = js("""
      const O = [{ name: 'berlin.jpg', kind: 'photo', start: 612, end: 618, status: 'shown', tag: 'Photo: berlin.jpg' },
                 { name: 'walk.mp4', kind: 'clip', start: 700, end: 703, status: 'hidden', tag: 'Clip: walk.mp4' },
                 { name: 'desk.png', kind: 'photo', start: 800, end: 810, status: 'shortened' }];
      const tag = t => { const o = Core.overlayAt(O, t); return o ? Core.tagText(o) : null; };
      return [611.9, 612, 617.99, 618, 701, 805].map(tag).concat([Core.overlayAt(undefined, 5), Core.overlayAt([], 5)]);
    """)
    assert out == [None, "Photo: berlin.jpg", "Photo: berlin.jpg", None, "Clip: walk.mp4 · hidden by a cut",
                   "Photo: desk.png", None, None]


# ── the top line ────────────────────────────────────────────────────────────


@needs_node
def test_what_the_last_change_did_in_plain_words():
    out = js("""
      const photos = [{ kind: 'photo', status: 'hidden' }], clips = [{ kind: 'clip', status: 'hidden' }];
      return [Core.changedParts({ trims: 12, seconds: -35, need_a_look: 2, overlays_hidden: 1 }, photos),
              Core.changedParts({ seconds: 4.2, overlays_hidden: 2 }, clips),
              Core.changedParts({ seconds: -0.2, overlays_hidden: 3 }, photos.concat(clips)),
              Core.changedParts({ trims: 3, seconds: 0, need_a_look: 1, overlays_hidden: 0 }, photos),
              Core.changedParts({ seconds: -31.0 }),
              Core.changedParts(null)];
    """)
    assert out == [["0:35 shorter", "1 photo now hidden"], ["0:04 longer", "2 clips now hidden"],
                   ["3 photos and clips now hidden"], [], ["0:31 shorter"], []]


@needs_node
def test_one_newly_hidden_clip_is_named_as_a_clip_beside_a_photo_hidden_earlier():
    out = js("""
      const photo = { id: 'ov1', kind: 'photo', status: 'hidden' };
      const before = [photo, { id: 'ov4', kind: 'clip', status: 'shown' }];
      const after = [photo, { id: 'ov4', kind: 'clip', status: 'hidden' }];
      return [Core.changedParts({ overlays_hidden: 1 }, after, before),
              Core.changedParts({ overlays_hidden: 1 }, after),
              Core.changedParts({ overlays_hidden: 2 }, after, [])];
    """)
    assert out == [["1 clip now hidden"], ["1 photo or clip now hidden"], ["2 photos and clips now hidden"]]


@needs_node
def test_one_needs_a_look_count_from_the_state_or_summed_from_the_groups():
    out = js("""
      const groups = [{ need_a_look: 5 }, { need_a_look: 1 }, { need_a_look: 2 }, {}];
      return [Core.needALook({ need_a_look: 8, groups: [{ need_a_look: 99 }] }), Core.needALook({ groups }),
              Core.needALook({ need_a_look: 0, groups }), Core.needALook({})];
    """)
    assert out == [8, 8, 0, 0]


@needs_node
def test_the_count_shown_is_the_top_level_one_whatever_the_last_change_counted():
    # changed.need_a_look counts every flagged join in the edit, automatic removals among
    # them, so it can be larger than the rows she can act on. The page shows need_a_look.
    out = js("""
      const state = { need_a_look: 9, groups: [{ need_a_look: 9 }], changed: { seconds: -32, need_a_look: 41, new_flags: 30, overlays_hidden: 0 } };
      return [Core.needALook(state), Core.changedParts(state.changed, [])];
    """)
    assert out == [9, ["0:32 shorter"]]
    shown = [line for line in page().splitlines() if "changed.need_a_look" in line or "changed.new_flags" in line]
    assert shown == [], "the page reads neither count from `changed`"


@needs_node
def test_the_export_button_by_state():
    out = js("""
      const folder = '/videos/my-video/exports';
      return [undefined, { state: 'idle', message: 'Nothing exported yet.', holds_earlier_edit: false },
              { state: 'running', progress: 0.42, holds_earlier_edit: true }, { state: 'running', progress: null },
              { state: 'done', progress: 1, file: 'my-video-edited.mp4', folder, message: 'Exported my-video-edited.mp4.' },
              { state: 'done', file: 'a.mp4', folder: 'C:\\\\videos\\\\' },
              { state: 'failed', message: 'The export did not finish. Press Export video to try again.' },
              { state: 'failed' }, { state: 'paused' }].map(e => Core.exportView(e));
    """)
    idle, idle_said, running, starting, done, windows, failed, failed_bare, unknown = out
    for view in (idle, idle_said, unknown):
        assert (view["label"], view["clickable"], view["note"], view["warn"]) == ("Export video", True, "", False)
    assert idle["holdsEarlier"] is None and idle_said["holdsEarlier"] is False
    assert (running["label"], running["clickable"], running["note"]) == ("Exporting 42%", False, "")
    assert running["holdsEarlier"] is True
    assert (starting["label"], starting["clickable"]) == ("Exporting…", False)
    assert (done["label"], done["clickable"], done["note"], done["warn"]) == ("Export video", True, "Exported: my-video-edited.mp4", False)
    assert done["hover"] == "/videos/my-video/exports"
    assert done["path"] == "/videos/my-video/exports/my-video-edited.mp4"
    assert windows["path"] == "C:\\videos\\a.mp4"
    assert (failed["label"], failed["clickable"], failed["warn"]) == ("Export video", True, True)
    assert failed["note"] == "The export did not finish. Press Export video to try again."
    assert failed_bare["note"] and failed_bare["warn"] is True


@needs_node
def test_a_line_from_the_server_that_holds_the_word_is_left_out():
    out = js("return ['0:04 out', 'laugh whole', '11 trims', '3 Trims skipped', 'trimmed', 'a striking trimaran', null, undefined].map(Core.shown);")
    assert out == ["0:04 out", "laugh whole", "", "", "", "a striking trimaran", "", ""]


# ── words, labels ───────────────────────────────────────────────────────────


@needs_node
def test_word_index_and_sentence_bounds():
    out = js("""
      const W = [["So", 0, .3, 0], ["this", .4, .6, 0], ["works.", .7, 1, 0], ["Next", 1.2, 1.4, 0],
                 ["one", 1.5, 1.7, 0], ["here?", 1.8, 2, 0], ["After", 5, 5.3, 0], ["a", 5.4, 5.5, 0], ["gap", 5.6, 6, 0]];
      return { at: [-1, 0.1, 0.65, 1.1, 99].map(t => Core.wordIndexAt(W, t)),
               first: Core.sentenceBounds(W, 1), second: Core.sentenceBounds(W, 4), gap: Core.sentenceBounds(W, 7),
               none: Core.sentenceBounds(W, 20) };
    """)
    assert out["at"] == [-1, 0, 1, 2, 8]
    assert out["first"] == [0, 2]
    assert out["second"] == [3, 5]
    assert out["gap"] == [6, 8]
    assert out["none"] is None


@needs_node
def test_clocks_and_lengths_are_whole_seconds():
    out = js("""
      return [Core.clock(0), Core.clock(59.9), Core.clock(479.28), Core.clock(3725),
              Core.length(0.44), Core.length(1.7), Core.length(5.88), Core.length(40.1), Core.length(59.6), Core.length(62),
              Core.outText(14.2), Core.outText(0.4), Core.outText(75.5)];
    """)
    assert out == ["0:00", "0:59", "7:59", "1:02:05", "1s", "2s", "6s", "40s", "1:00", "1:02", "0:14 out", "0:00 out", "1:16 out"]
    assert not decimal_times_in(" ".join(out))


# ── cutting and keeping words by hand ───────────────────────────────────────

# Six words, one of each kind: two that play, one the pace took out, one Claude
# took out, and a cut of the creator's own over two words.
HAND = """{
  words: [["So", 0, .3, 0, 0], ["um", .4, .6, 1, 1], ["this", .7, 1, 1, 2], ["really", 1.1, 1.5, 1, 3],
          ["truly", 1.6, 2, 1, 3], ["works.", 2.1, 2.5, 0, 0]],
  rows: [{ id: 'c0.70-1.00', by: 'claude', start: 0.68, end: 1.02, state: 'kept_out' },
         { id: 'y1.10-2.00', by: 'you', start: 1.1, end: 2.0, state: 'kept_out' }],
}"""


@needs_node
def test_a_double_click_cuts_a_word_that_plays():
    out = js(f"return Core.wordChange({HAND}, 0);")
    assert out == {"kind": "cut", "path": "api/cut/add", "body": {"start": 0, "end": 0.3}}


@needs_node
def test_a_double_click_brings_back_exactly_the_word_claude_or_the_pace_took_out():
    out = js(f"const s = {HAND}; return [Core.wordChange(s, 1), Core.wordChange(s, 2)];")
    assert out[0] == {"kind": "back", "path": "api/keep", "body": {"start": 0.4, "end": 0.6, "exact": True}}
    assert out[1] == {"kind": "back", "path": "api/keep", "body": {"start": 0.7, "end": 1, "exact": True}}


@needs_node
def test_a_double_click_on_her_own_cut_takes_that_word_out_of_it():
    out = js(f"const s = {HAND}; return [Core.wordChange(s, 3), Core.wordChange(s, 4), Core.wordChange(s, 9)];")
    assert out[0] == {"kind": "back", "path": "api/cut/remove", "body": {"id": "y1.10-2.00", "start": 1.1, "end": 1.5}}
    assert out[1] == {"kind": "back", "path": "api/cut/remove", "body": {"id": "y1.10-2.00", "start": 1.6, "end": 2}}
    assert out[2] is None, "no such word"


@needs_node
def test_cut_takes_exactly_the_words_picked_whichever_way_she_dragged():
    out = js(f"const s = {HAND}; return [Core.cutChange(s, 0, 5), Core.cutChange(s, 5, 2), Core.cutChange(s, 3, 3)];")
    assert [c["body"] for c in out] == [{"start": 0, "end": 2.5}, {"start": 0.7, "end": 2.5}, {"start": 1.1, "end": 1.5}]
    assert {c["path"] for c in out} == {"api/cut/add"} and {c["kind"] for c in out} == {"cut"}


@needs_node
def test_words_with_four_values_are_treated_as_before():
    out = js("""
      const old = [["So", 0, .3, 0], ["um", .4, .6, 1]], now = [["So", 0, .3, 0, 0], ["um", .4, .6, 1, 3]];
      return { why: old.concat(now).map(Core.whyOut), tells: [Core.tellsWho(old), Core.tellsWho(now), Core.tellsWho([]), Core.tellsWho(undefined)],
               cut: Core.wordChange({ words: old }, 0) };
    """)
    assert out["why"] == [0, 1, 0, 3], "a removed word with no fifth value counts as taken out automatically"
    assert out["tells"] == [False, True, False, False]
    assert out["cut"]["path"] == "api/cut/add"


@needs_node
def test_the_top_line_says_what_she_did_and_leaves_a_short_time_out():
    out = js("""
      const did = (kind, changed, state, was) => Core.actionParts(kind, Object.assign({ changed }, state), was);
      return [did('cut', { words_cut: 1, seconds: -0.3 }), did('cut', { words_cut: 12, seconds: -3.4 }),
              did('cut', { words_cut: 2, seconds: -0.99 }), did('cut', { words_cut: 3, seconds: -1 }),
              did('back', { words_back: 1, seconds: 0.4 }), did('back', { words_back: 4, seconds: 75.2 }),
              did('keep', { seconds: 5.2 }), did('keep', { seconds: 0 }), did('undo', { seconds: 0.3 }), did('undo', { seconds: -2 }),
              did('unkeep', { seconds: -4 }),
              did('cut', { words_cut: 0, seconds: 0 }), did('back', { words_back: 0, seconds: 0 }),
              did('cut', { words_cut: 1, seconds: -0.2, keeps_changed: 1 }, { keeps: [{}, {}] }, { keeps: [{}] }),
              did('cut', { words_cut: 1, seconds: -0.2, keeps_changed: 1 }, { keeps: [] }, { keeps: [{}] }),
              did('cut', { words_cut: 6, seconds: -2, overlays_hidden: 1 }, { overlays: [{ id: 'a', kind: 'photo', status: 'hidden' }] }, { overlays: [{ id: 'a', kind: 'photo', status: 'shown' }] })];
    """)
    assert out == [
        ["You cut 1 word."], ["You cut 12 words", "0:03 shorter"], ["You cut 2 words."], ["You cut 3 words", "0:01 shorter"],
        ["You brought back 1 word."], ["You brought back 4 words", "1:15 longer"],
        ["You kept a part", "0:05 longer"], ["You kept a part."], ["Undone."], ["Undone", "0:02 shorter"],
        ["You stopped keeping a part", "0:04 shorter"],
        ["Those words were cut already."], ["You took back a cut."],
        ["You cut 1 word", "a part you kept is shorter now"], ["You cut 1 word", "a part you kept is no longer kept"],
        ["You cut 6 words", "0:02 shorter", "1 photo now hidden"],
    ]
    said = " ".join(part for parts in out for part in parts)
    assert not decimal_times_in(said), decimal_times_in(said)
    assert not banned_in(said) and not WORDS_GONE.search(said)


@needs_node
def test_a_sentence_in_the_top_line_keeps_its_full_stop_only_when_it_comes_last():
    out = js("""
      return [Core.factsLine(['18:58 after edits', 'You cut 1 word.']),
              Core.factsLine(['18:58 after edits', 'You cut 1 word.', 'the export holds the earlier edit']),
              Core.factsLine(['18:58 after edits', 'You cut 12 words', '0:03 shorter']), Core.factsLine([]), Core.factsLine(undefined)];
    """)
    assert out == [" · 18:58 after edits · You cut 1 word.",
                   " · 18:58 after edits · You cut 1 word · the export holds the earlier edit",
                   " · 18:58 after edits · You cut 12 words · 0:03 shorter", "", ""]


@needs_node
def test_the_filler_switch_names_the_first_few_words_it_takes_out():
    out = js("""
      const all = [["you know", 12], ["like", 8], ["and", 7], ["so", 5], ["um", 4], ["i mean", 2]];
      return [Core.fewWords(all, 4), Core.fewWords(all.slice(0, 4), 4), Core.fewWords(all.slice(0, 1), 4), Core.fewWords(all.slice(4), 4),
              Core.fewWords([], 4), Core.fewWords(undefined, 4)];
    """)
    assert out == ["you know, like, and, so, and 2 more", "you know, like, and, so", "you know", "um, I mean", "", ""]


@needs_node
def test_the_cuts_heading_counts_hers_and_claudes():
    out = js("""
      const rows = [{ by: 'you' }, { by: 'claude' }, { by: 'claude' }, { by: 'you' }, { by: 'you' }];
      return [Core.cutCounts({ cut_counts: { yours: 5, claude: 17 }, rows }), Core.cutCounts({ rows, groups: [{ count: 3 }, { count: 2 }] }),
              Core.cutCounts({ rows: [{}, {}], groups: [{ count: 2 }] }), Core.cutCounts({})];
    """)
    assert out == [{"mine": 5, "claude": 17}, {"mine": 3, "claude": 2}, {"mine": 0, "claude": 2}, {"mine": 0, "claude": 0}]


# ── fine tune and filler likes ──────────────────────────────────────────────

# A state with two short sliders, as the server sends them: every place from the least to the most.
FINE = """{
  settings: { pace: 'standard', fine: { gap_length: 0.2, rhythm: 1.5 } },
  custom: null,
  paces: [{ pace: 'natural', label: 'Natural', summary: 'Only long pauses go.', seconds_saved: 46 },
          { pace: 'standard', label: 'Standard', summary: 'The usual.', seconds_saved: 104.7 }],
  fine_ranges: {
    rhythm: { min: 1, max: 2, step: 0.5, label: 'Speech kept between cuts', harder: 'max',
              positions: [{ value: 1, shown: '3 s', said: '3 seconds' }, { value: 1.5, shown: '2.5 s', said: '2.5 seconds' },
                          { value: 2, shown: '2 s', said: '2 seconds' }] },
    gap_length: { min: 0.15, max: 0.3, step: 0.05, label: 'Cut pauses longer than', harder: 'min',
                  positions: [{ value: 0.15, shown: '0.15 s', said: '0.15 seconds' }, { value: 0.2, shown: '0.2 s', said: '0.2 seconds' },
                              { value: 0.25, shown: '0.25 s', said: '0.25 seconds' }, { value: 0.3, shown: '0.3 s', said: '0.3 seconds' }] },
  },
}"""
# What a slider shows is seconds with a decimal ("0.35 s"). The second contract draws it that
# way, so these texts are the one place a decimal reaches the creator. ``creator_words`` holds
# that exception (``is_slider_length``); these two say which of its forms goes where.
SHOWN_ENDS, SAID_ENDS = " s", ("second", "seconds")


@needs_node
def test_both_sliders_cut_harder_to_the_right():
    out = js(f"return Core.fineSliders({FINE});")
    gap, rhythm = out
    assert (gap["key"], gap["label"]) == ("gap_length", "Cut pauses longer than"), "the pause slider is drawn first"
    assert gap["values"] == [0.3, 0.25, 0.2, 0.15], "the longest pause on the left, the shortest on the right"
    assert gap["texts"] == ["0.3 s", "0.25 s", "0.2 s", "0.15 s"] and gap["says"][3] == "0.15 seconds"
    assert gap["at"] == 2
    assert (rhythm["key"], rhythm["values"], rhythm["at"]) == ("rhythm", [1, 1.5, 2], 1)
    assert rhythm["texts"] == ["3 s", "2.5 s", "2 s"], "shown as speech kept, as sent, not as a rhythm number"


@needs_node
def test_a_state_from_before_fine_tune_draws_no_sliders():
    out = js(f"""
      const s = {FINE};
      return [Core.fineSliders({{ settings: {{ pace: 'standard' }}, fine_ranges: s.fine_ranges }}), Core.fineSliders({{ settings: s.settings }}),
              Core.fineSliders({{}}), Core.fineSliders(null),
              Core.fineSliders({{ settings: {{ fine: {{ gap_length: 0.2 }} }}, fine_ranges: s.fine_ranges }}).map(x => x.key)];
    """)
    assert out == [[], [], [], [], ["gap_length"]]


@needs_node
def test_the_line_under_the_track_names_a_setting_of_her_own():
    out = js(f"""
      const s = {FINE}, own = Object.assign({{}}, s, {{ custom: {{ label: 'Custom', summary: 'Your own setting, between Fast and Tight.', trims: 410, seconds_saved: 250.4 }} }});
      return [Core.paceLine(s, 'standard'), Core.paceLine(own, 'custom'), Core.paceLine(s, 'custom'), Core.CUSTOM];
    """)
    assert out[0] == {"name": "", "saves": "1:44", "summary": "The usual."}
    assert out[1] == {"name": "Custom", "saves": "4:10", "summary": "Your own setting, between Fast and Tight."}
    assert out[2] == {"name": "Custom", "saves": "", "summary": ""}, "while her move is on its way the numbers wait"
    assert out[3] == "custom"


@needs_node
def test_the_filler_likes_switch_in_both_states():
    out = js("""
      const k = { key: 'likes', label: 'Filler likes', count: 35, seconds: 5.6, picked: 41, said: 112, left_in: 5, kept_by_you: 1, ready: true };
      const but = more => Core.likesView(Object.assign({}, k, more));
      return [but({}), but({ left_in: 1 }), but({ left_in: 0 }), but({ ready: false, count: 0, picked: 0, left_in: 0 }),
              Core.likesView({ key: 'likes', count: 3 })];
    """)
    assert out[0] == {"ready": True, "line": "picked by Claude · 77 kept",
                      "more": "5 stay in because cutting them would clip the word beside them."}
    assert out[1]["more"] == "1 stays in because cutting it would clip the word beside it."
    assert out[2]["more"] == ""
    assert out[3] == {"ready": False, "line": "Ask Claude to find them.", "more": ""}
    assert out[4] == {"ready": True, "line": "picked by Claude", "more": ""}


# ── the example the page runs from with ?example=1 ──────────────────────────


def strings_in(value: Any) -> list[str]:
    """Every string inside a JSON value, however deep."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in strings_in(v)]
    if isinstance(value, list):
        return [s for v in value for s in strings_in(v)]
    return []


# The fields of a state whose text reaches the screen. Ids, keys and numbers do not.
SHOWN_FIELDS = {
    "paces": ("label", "summary"), "take_out": ("label", "words"), "groups": ("label",),
    "rows": ("clock", "reason", "flag_labels", "why", "before", "removed", "after"),
    "keeps": ("text", "note"),
    "clusters": ("clock", "clock_range", "out"), "overlays": ("clock",),
}


def creator_reads(state: dict[str, Any]) -> list[str]:
    """Every string in a state that the page can show, plus the lines the page builds from it."""
    read = [text for key, fields in SHOWN_FIELDS.items() for item in state[key] for f in fields
            for text in strings_in(item.get(f, ""))]
    read.append(state.get("word_times_note") or "")
    read += [r["label"] for r in state["fine_ranges"].values()] + [(state["custom"] or {}).get("summary", "")]
    read += js(f"""
      const s = {json.dumps({k: state[k] for k in ("clusters", "overlays", "export", "changed", "take_out", "keeps")})};
      return s.clusters.flatMap(c => [Object.values(Core.clusterTip(c)).join(' '), Core.clusterName(c, false), Core.clusterName(c, true)])
        .concat(s.overlays.map(o => Core.tagText(o).replace(o.name, '')), Core.changedParts(s.changed, s.overlays),
                ['cut', 'back', 'keep', 'unkeep', 'undo'].flatMap(kind => Core.actionParts(kind, s, s)),
                s.take_out.map(k => Core.fewWords(k.words, 4)),
                s.take_out.filter(k => k.key === 'likes').flatMap(k => [Core.likesView(k), Core.likesView(Object.assign({{}}, k, {{ ready: false }})),
                  Core.likesView(Object.assign({{}}, k, {{ left_in: 1 }}))]).flatMap(v => [v.line, v.more]),
                [Core.exportView(s.export).label]);
    """)
    return read


@needs_node
def test_the_example_has_the_round_five_shape():
    state = js("return Core.exampleState();")
    for key in ("video", "word_times", "word_times_note", "settings", "usual", "paces", "custom", "fine_ranges", "take_out",
                "need_a_look", "cut_counts", "groups", "rows", "removed", "trims", "keeps", "clusters", "overlays",
                "export", "durations", "changed", "can_undo", "words", "sounds", "updated_at"):
        assert key in state, f"the example has no {key}"
    assert set(state["clusters"][0]) == {"number", "start", "end", "clock", "clock_range", "edits", "seconds_removed", "out", "level"}
    assert set(state["overlays"][0]) == {"id", "name", "kind", "start", "end", "clock", "status", "tag"}
    assert set(state["export"]) == {"state", "progress", "file", "folder", "message", "holds_earlier_edit"}
    assert state["export"]["state"] == "idle"
    assert state["need_a_look"] == sum(g["need_a_look"] for g in state["groups"]) > 0
    assert [c["number"] for c in state["clusters"]] == list(range(1, len(state["clusters"]) + 1))
    assert {c["level"] for c in state["clusters"]} == {1, 2, 3}, "the example shows all three shades"
    assert {o["status"] for o in state["overlays"]} >= {"shown", "hidden"}
    assert len(state["keeps"]) == 1
    assert "samples" not in state, "the page shows no samples, so the example makes none"
    claudes = [r for r in state["rows"] if r["by"] == "claude"]
    yours = [r for r in state["rows"] if r["by"] == "you"]
    assert len(claudes) == 16 and len(claudes) + len(yours) == len(state["rows"]), "every row says who made the cut"
    assert len({r["id"] for r in state["rows"]}) == len(state["rows"]), "two example cuts landed on the same words"
    assert state["video"]["duration"] == state["durations"]["full"]
    # six stops, gentlest first
    assert [p["label"] for p in state["paces"]] == PACES
    assert [p["pace"] for p in state["paces"]] == [name.lower() for name in PACES]
    saved = [p["seconds_saved"] for p in state["paces"]]
    assert saved == sorted(saved) and len(set(saved)) == 6, "each stop takes out more than the one before"
    for kind in state["take_out"]:
        assert list(kind["by_pace"]) == [name.lower() for name in PACES]
    for hard in state["paces"][4:]:
        assert "pause" in hard["summary"], "Hard and Max say what the creator gives up"
    # her own cuts: a group of their own, listed first
    assert [g["key"] for g in state["groups"]] == ["yours", "repeat", "false_start", "off_topic", "other"]
    assert state["groups"][0]["label"] == "Your cuts" and state["groups"][0]["count"] == len(yours) >= 2
    assert state["groups"][0]["rows"] == [r["id"] for r in yours]
    assert all(r["group"] == "yours" and r["reason"] == "Cut by you" and r["state"] == "kept_out" for r in yours)
    assert not [r for r in claudes if r["group"] == "yours"]
    assert state["cut_counts"] == {"yours": len(yours), "claude": 16}
    assert any(" " in r["removed"] for r in yours), "one of her cuts holds more than one word, so it can be made shorter"
    # words say who took each one out
    assert {len(w) for w in state["words"]} == {5}
    assert {w[4] for w in state["words"]} == {0, 1, 2, 3, 4}
    assert all((w[4] > 0) == (w[3] == 1) for w in state["words"])
    # the filler switch is named for what it takes out, and lists it
    fillers = state["take_out"][1]
    assert (fillers["key"], fillers["label"]) == ("fillers", "Filler words")
    assert len(fillers["words"]) >= 2 and sum(n for _, n in fillers["words"]) == fillers["count"]
    assert [n for _, n in fillers["words"]] == sorted((n for _, n in fillers["words"]), reverse=True), "most first"
    assert state["can_undo"] is False and state["word_times"] == "measured"
    # fine tune: the chosen stop's own values, and the two sliders with the text for every place
    assert state["settings"]["pace"] == "standard" and state["custom"] is None
    assert state["settings"]["fine"] == {"gap_length": 0.6, "rhythm": 3}
    assert list(state["fine_ranges"]) == ["gap_length", "rhythm"]
    for slider, places in (("gap_length", 28), ("rhythm", 9)):
        r = state["fine_ranges"][slider]
        assert set(r) == {"min", "max", "step", "label", "harder", "positions"}
        assert len(r["positions"]) == places and all(set(x) == {"value", "shown", "said"} for x in r["positions"])
        assert [x["value"] for x in r["positions"]] == sorted(x["value"] for x in r["positions"]), "from the least to the most"
        assert (r["positions"][0]["value"], r["positions"][-1]["value"]) == (r["min"], r["max"])
    assert [state["fine_ranges"][k]["harder"] for k in ("gap_length", "rhythm")] == ["min", "max"]
    # filler likes: the fourth switch
    assert [k["key"] for k in state["take_out"]] == ["pauses", "fillers", "repeats", "likes"]
    likes = state["take_out"][3]
    assert set(likes) == {"key", "label", "count", "seconds", "by_pace", "words", "picked", "said", "left_in", "kept_by_you",
                          "ready", "word"}
    assert likes["label"] == "Filler likes" and likes["ready"] is True and likes["word"] == "like"
    assert likes["picked"] == likes["count"] + likes["left_in"] + likes["kept_by_you"]
    assert likes["said"] > likes["picked"] > likes["count"] > 0 and likes["left_in"] > 0
    assert state["settings"]["take_out"]["likes"] is True


@needs_node
def test_the_example_adds_up():
    state = js("return Core.exampleState();")
    removed = state["removed"]
    assert removed == sorted(removed) and all(a < b for a, b in removed)
    gone = sum(b - a for a, b in removed)
    assert state["durations"]["edited"] == pytest.approx(state["durations"]["full"] - gone, abs=0.2)
    now = next(p for p in state["paces"] if p["pace"] == state["settings"]["pace"])
    assert now["trims"] == len(state["trims"]) == sum(k["count"] for k in state["take_out"][:3]), "the three kinds the pace takes"
    assert [g["count"] for g in state["groups"]] == [sum(r["group"] == g["key"] for r in state["rows"]) for g in state["groups"]]
    for row in state["rows"]:
        inside = [w for w in state["words"] if row["start"] < (w[1] + w[2]) / 2 < row["end"]]
        assert inside and {w[4] for w in inside} == {3 if row["by"] == "you" else 2}, row["id"]
        assert " ".join(w[0] for w in inside) == row["removed"]
    picked = [w for w in state["words"] if w[4] == 4]
    assert picked and {w[0] for w in picked} == {"like,"}, "Claude picked the likes that are filler, and only those"
    assert [w for w in state["words"] if w[0] == "like" and w[4] == 0], "a like that means something plays"
    for cluster in state["clusters"]:
        assert 0 <= cluster["start"] < cluster["end"] <= state["video"]["duration"]
        assert cluster["edits"] > 0
    struck = [w for w in state["words"] if w[3]]
    assert struck and all(any(a <= (w[1] + w[2]) / 2 <= b for a, b in removed) for w in struck)
    keep = state["keeps"][0]
    assert not [r for r in removed if r[0] < keep["end"] and r[1] > keep["start"]], "nothing is taken out of a kept part"


@needs_node
def test_nothing_the_example_says_breaks_the_word_rules():
    state = js("""
      const s = Core.exampleState(), i = s.words.findIndex((w, k) => k > 40 && !w[4]);
      return [s, Core.exampleChange(s, 'api/treatment', { pace: 'fast' }), Core.exampleChange(s, 'api/treatment', { pace: 'max' }),
              Core.exampleChange(s, 'api/cut/add', Core.cutChange(s, i, i + 3).body),
              Core.exampleChange(s, 'api/treatment', { fine: { gap_length: 0.35, rhythm: 3.5 } }),
              Core.exampleChange(s, 'api/treatment', { fine: { gap_length: 1.5 } }), Core.exampleChange(s, 'api/treatment', { fine: { rhythm: 5 } })];
    """)
    for s in state:
        said = " ".join(creator_reads(s))
        assert len(said) > 2000
        assert not WORDS_GONE.search(said), WORDS_GONE.search(said)
        assert not CLUSTER_WORD.search(said), CLUSTER_WORD.search(said)
        assert not banned_in(said), banned_in(said)
        assert "—" not in said
        assert not decimal_times_in(said), decimal_times_in(said)


@needs_node
def test_changes_in_the_example_answer_like_the_server():
    out = js("""
      const s = Core.exampleState();
      const hidden = s.overlays.find(o => o.status === 'hidden');
      const row = s.rows.find(r => r.start < hidden.start && r.end > hidden.end);
      const back = Core.exampleChange(s, 'api/cut', { id: row.id, state: 'put_back' });
      const out = Core.exampleChange(back, 'api/cut', { id: row.id, state: 'kept_out' });
      const fast = Core.exampleChange(s, 'api/treatment', { pace: 'fast' });
      const noPauses = Core.exampleChange(s, 'api/treatment', { take_out: { pauses: false } });
      const kept = Core.exampleChange(s, 'api/keep', { start: row.start + 0.5, end: row.start + 1 });
      const unkept = Core.exampleChange(kept, 'api/keep/remove', { id: kept.keeps.find(k => k.start <= row.start + 0.5 && k.end >= row.start + 1).id });
      const usual = Core.exampleChange(s, 'api/usual', {});
      const refuse = (state, path, body) => { try { Core.exampleChange(state, path, body); return null; } catch(err){ return err.message; } };
      return {
        untouched: s.rows.find(r => r.id === row.id).state,
        back: { state: back.rows.find(r => r.id === row.id).state, flags: back.rows.find(r => r.id === row.id).flags, photo: back.overlays.find(o => o.id === hidden.id).status,
                changed: back.changed, said: Core.changedParts(back.changed, back.overlays), look: back.need_a_look - s.need_a_look },
        out: { photo: out.overlays.find(o => o.id === hidden.id).status, said: Core.changedParts(out.changed, out.overlays), look: out.need_a_look - s.need_a_look },
        fast: { pace: fast.settings.pace, shorter: fast.durations.edited < s.durations.edited, stutters: fast.take_out[2].count, was: s.take_out[2].count, would: s.take_out[2].by_pace.fast },
        noPauses: { count: noPauses.take_out[0].count, was: s.take_out[0].count, kinds: [...new Set(noPauses.trims.map(t => t[2]))].sort(), longer: noPauses.durations.edited > s.durations.edited },
        kept: { keeps: kept.keeps.length, row: kept.rows.find(r => r.id === row.id).state, refused: refuse(kept, 'api/cut', { id: row.id, state: 'kept_out' }) },
        unkept: unkept.keeps.length,
        usual: [s.usual, usual.usual],
      };
    """)
    assert out["untouched"] == "kept_out"
    assert out["back"]["state"] == "put_back" and out["back"]["flags"] == []
    assert out["back"]["photo"] != "hidden" and out["back"]["changed"]["overlays_hidden"] == 0
    assert out["back"]["changed"]["seconds"] > 0 and out["back"]["said"][0].endswith(" longer")
    assert out["out"]["photo"] == "hidden" and out["out"]["said"][-1] == "1 photo now hidden"
    assert out["back"]["look"] <= 0 and out["out"]["look"] == 0
    assert out["fast"]["pace"] == "fast" and out["fast"]["shorter"]
    assert out["fast"]["was"] == 0 and out["fast"]["stutters"] == out["fast"]["would"] > 0
    assert out["noPauses"]["count"] == out["noPauses"]["was"] and "pauses" not in out["noPauses"]["kinds"] and out["noPauses"]["longer"]
    assert out["kept"]["keeps"] == 2 and out["kept"]["row"] == "put_back"
    assert "flagged to keep" in out["kept"]["refused"]
    assert out["unkept"] == 1
    assert out["usual"][0] is None and out["usual"][1]["pace"] == "standard"


@needs_node
def test_cutting_and_bringing_back_words_in_the_example():
    out = js("""
      const s = Core.exampleState(), W = s.words;
      const mine = x => x.rows.filter(r => r.by === 'you').map(r => r.removed);
      const why = (x, list) => list.map(i => x.words[i][4]);
      const say = (kind, x, was) => Core.actionParts(kind, x, was).join(' · ');
      const send = (x, change) => Core.exampleChange(x, change.path, change.body);
      const refuse = (x, path, body) => { try { Core.exampleChange(x, path, body); return null; } catch(err){ return err.message; } };
      // four words that play, with words that play either side of them
      const i = W.findIndex((w, k) => k > 60 && [-1, 0, 1, 2, 3, 4].every(d => W[k + d][4] === 0 && !/[.(]/.test(W[k + d][0])));
      const one = send(s, Core.wordChange(s, i));
      const two = send(one, Core.wordChange(one, i + 1));
      const apart = send(two, Core.wordChange(two, i + 3));
      const shorter = send(two, Core.wordChange(two, i));
      const wide = send(s, Core.cutChange(s, i, i + 2));
      const middle = send(wide, Core.wordChange(wide, i + 1));
      const whole = send(one, Core.wordChange(one, i));
      const undone = send(two, { path: 'api/undo', body: {} });
      // a word Claude took out, from the middle of one of his cuts
      const row = s.rows.find(r => r.by === 'claude' && Core.wordsBetween(W, r.start, r.end).length > 4);
      const c = Core.wordsBetween(W, row.start, row.end)[2];
      const back = send(s, Core.wordChange(s, c));
      // a word the pace took out
      const a = W.findIndex(w => w[4] === 1);
      const auto = send(s, Core.wordChange(s, a));
      const atMax = Core.exampleChange(auto, 'api/treatment', { pace: 'max' });
      const cutAtMax = Core.exampleChange(two, 'api/treatment', { pace: 'max' });
      // a cut inside the part she kept
      const keep = s.keeps[0], k = Core.wordsBetween(W, keep.start, keep.end)[3];
      const inKeep = send(s, Core.wordChange(s, k));
      return {
        one: { mine: mine(one), why: why(one, [i - 1, i, i + 1]), changed: one.changed, said: say('cut', one, s), undo: [s.can_undo, one.can_undo],
               counts: [s.cut_counts, one.cut_counts], shorter: one.durations.edited < s.durations.edited },
        two: { mine: mine(two), said: say('cut', two, one), rows: two.rows.filter(r => r.by === 'you').length - s.rows.filter(r => r.by === 'you').length },
        apart: apart.rows.filter(r => r.by === 'you').length - two.rows.filter(r => r.by === 'you').length,
        shorter: { why: why(shorter, [i, i + 1]), change: Core.wordChange(two, i).path, said: say('back', shorter, two) },
        middle: { why: why(middle, [i, i + 1, i + 2]), rows: middle.rows.filter(r => r.by === 'you').length - wide.rows.filter(r => r.by === 'you').length },
        whole: { same: JSON.stringify(mine(whole)) === JSON.stringify(mine(s)), why: why(whole, [i]), changed: whole.changed.words_back },
        undone: { same: JSON.stringify(mine(undone)) === JSON.stringify(mine(one)), undo: undone.can_undo, said: say('undo', undone, two),
                  again: refuse(undone, 'api/undo', {}) },
        back: { why: why(back, [c - 1, c, c + 1]), state: back.rows.find(r => r.id === row.id).state, keep: back.keeps.find(x => x.exact),
                said: say('back', back, s), change: Core.wordChange(s, c) },
        auto: { why: why(auto, [a]), atMax: why(atMax, [a]), more: atMax.trims.length > auto.trims.length },
        cutAtMax: why(cutAtMax, [i, i + 1]),
        inKeep: { why: why(inKeep, [k - 1, k, k + 1]), keeps: inKeep.keeps.map(x => x.text), changed: inKeep.changed.keeps_changed, said: say('cut', inKeep, s),
                  kept: [k - 1, k + 1].map(j => inKeep.keeps.some(x => x.start <= W[j][1] && x.end >= W[j][2])) },
        refused: [refuse(s, 'api/cut/remove', { id: row.id }), refuse(s, 'api/cut', { id: s.rows.find(r => r.by === 'you').id, state: 'put_back' }),
                  refuse(s, 'api/cut/add', { start: 0.1, end: 0.2 }), refuse(s, 'api/undo', {})],
      };
    """)
    one, two = out["one"], out["two"]
    assert len(one["mine"]) == 4 and one["why"] == [0, 3, 0], "exactly that word, no widening"
    assert one["changed"]["words_cut"] == 1 and one["changed"]["your_cuts"] == 1 and one["said"] == "You cut 1 word."
    assert one["undo"] == [False, True] and one["shorter"]
    assert one["counts"] == [{"yours": 3, "claude": 16}, {"yours": 4, "claude": 16}]
    assert two["rows"] == 1 and two["said"] == "You cut 1 word.", "a cut that touches one of hers joins it"
    assert any(len(text.split()) == 2 for text in two["mine"])
    assert out["apart"] == 1, "one with a word between stays a cut of its own"
    assert out["shorter"] == {"why": [0, 3], "change": "api/cut/remove", "said": "You brought back 1 word."}
    assert out["middle"] == {"why": [3, 0, 3], "rows": 1}, "her cut splits in two around the word she brought back"
    assert out["whole"] == {"same": True, "why": [0], "changed": 1}
    assert out["undone"]["same"] and out["undone"]["undo"] is False and out["undone"]["said"] == "Undone."
    assert out["undone"]["again"] == "There is nothing to undo."
    back = out["back"]
    assert back["change"]["path"] == "api/keep" and back["change"]["body"]["exact"] is True
    assert back["why"] == [2, 0, 2] and back["state"] == "kept_out", "the rest of Claude's cut stays cut"
    assert back["keep"]["exact"] is True and len(back["keep"]["text"].split()) == 1 and back["said"] == "You brought back 1 word."
    assert out["auto"] == {"why": [0], "atMax": [0], "more": True}, "a word brought back stays back at every pace"
    assert out["cutAtMax"] == [3, 3], "her cuts outlive a change of pace"
    in_keep = out["inKeep"]
    assert in_keep["why"] == [0, 3, 0] and in_keep["kept"] == [True, True] and len(in_keep["keeps"]) == 2, "the kept part splits around her cut"
    assert in_keep["changed"] == 1 and in_keep["said"] == "You cut 1 word · a part you kept is shorter now"
    assert all(out["refused"]), out["refused"]
    for sentence in out["refused"]:
        assert not banned_in(sentence) and not decimal_times_in(sentence)


@needs_node
def test_fine_tune_in_the_example():
    out = js("""
      const s = Core.exampleState();
      const set = (x, fine) => Core.exampleChange(x, 'api/treatment', { fine });
      const refuse = body => { try { Core.exampleChange(s, 'api/treatment', body); return null; } catch(err){ return err.message; } };
      const gaps = s.fine_ranges.gap_length.positions.map(p => p.value).reverse(), rhythms = s.fine_ranges.rhythm.positions.map(p => p.value);
      const own = set(s, { gap_length: 0.35 }), both = set(own, { rhythm: 4.5 });
      const i = s.words.findIndex((w, k) => k > 60 && !w[4]);
      const cut = Core.exampleChange(s, 'api/cut/add', Core.cutChange(s, i, i).body), moved = set(cut, { rhythm: 5 });
      const stop = Core.exampleChange(both, 'api/treatment', { pace: 'fast' });
      const usual = Core.exampleChange(both, 'api/usual', {});
      return {
        own: { settings: own.settings, custom: own.custom, paces: own.paces.length, changed: own.changed.seconds },
        both: both.settings.fine,
        byGap: gaps.map(g => set(s, { gap_length: g, rhythm: 3 }).custom.seconds_saved),
        byRhythm: rhythms.map(r => set(s, { gap_length: 0.4, rhythm: r }).custom.seconds_saved),
        atStops: s.paces.map(p => { const at = Core.exampleChange(s, 'api/treatment', { pace: p.pace }).settings.fine; return [set(s, at).custom.seconds_saved, p.seconds_saved]; }),
        moved: { undo: [cut.can_undo, moved.can_undo], hers: moved.words[i][4], keeps: moved.keeps.length === s.keeps.length },
        stop: { settings: stop.settings, custom: stop.custom },
        usual: usual.usual,
        refused: [refuse({ fine: { gap_length: 0.37 } }), refuse({ fine: { gap_length: 2 } }), refuse({ fine: { rhythm: 5.5 } }),
                  refuse({ pace: 'fast', fine: { rhythm: 4 } })],
        shown: Object.values(s.fine_ranges).flatMap(r => r.positions.map(p => p.shown)), said: Object.values(s.fine_ranges).flatMap(r => r.positions.map(p => p.said)),
        summaries: [own, both, set(s, { gap_length: 1.5 }), set(s, { gap_length: 0.15 })].map(x => x.custom.summary),
      };
    """)
    own = out["own"]
    assert own["settings"]["pace"] == "custom" and own["settings"]["fine"] == {"gap_length": 0.35, "rhythm": 3}
    assert own["custom"]["label"] == "Custom" and own["custom"]["trims"] > 0 and own["paces"] == 6, "custom is not a seventh stop"
    assert out["both"] == {"gap_length": 0.35, "rhythm": 4.5}, "the slider she did not move keeps its value"
    assert out["byGap"] == sorted(out["byGap"]) and len(set(out["byGap"])) >= 4, "a shorter pause cuts more"
    assert out["byRhythm"] == sorted(out["byRhythm"]) and len(set(out["byRhythm"])) >= 2, "less speech kept cuts more"
    assert all(hers == stop for hers, stop in out["atStops"]), "a stop's own values give that stop's numbers"
    assert out["moved"] == {"undo": [True, True], "hers": 3, "keeps": True}, "a slider move leaves her cuts, her keeps and undo alone"
    assert out["stop"]["settings"]["pace"] == "fast" and out["stop"]["custom"] is None
    assert out["stop"]["settings"]["fine"] == {"gap_length": 0.4, "rhythm": 4}, "a stop sets the sliders back to its own values"
    assert out["usual"]["pace"] == "custom" and out["usual"]["fine"] == {"gap_length": 0.35, "rhythm": 4.5}
    assert all(out["refused"]), out["refused"]
    assert out["shown"] and all(is_slider_length(text) and text.endswith(SHOWN_ENDS) for text in out["shown"]), out["shown"]
    assert out["said"] and all(is_slider_length(text) and text.endswith(SAID_ENDS) for text in out["said"]), out["said"]
    said = " ".join(out["refused"] + out["summaries"])
    assert not banned_in(said) and not WORDS_GONE.search(said)
    assert not decimal_times_in(" ".join(out["summaries"])), "the sentence under Custom holds no numbers"


@needs_node
def test_the_filler_likes_switch_strikes_and_unstrikes_the_picked_likes():
    out = js("""
      const s = Core.exampleState();
      const likes = x => x.take_out[3], picked = x => x.words.filter(w => w[4] === 4).length;
      const sw = (x, on) => Core.exampleChange(x, 'api/treatment', { take_out: { likes: on } });
      const off = sw(s, false), on = sw(off, true);
      // one she brings back, and one Claude left that she cuts by hand
      const p = s.words.findIndex(w => w[4] === 4), left = s.words.findIndex(w => w[0] === 'like,' && w[4] === 0);
      const back = Core.exampleChange(s, Core.wordChange(s, p).path, Core.wordChange(s, p).body);
      const hand = Core.exampleChange(back, Core.wordChange(back, left).path, Core.wordChange(back, left).body);
      const handOff = sw(hand, false), handOn = sw(handOff, true);
      const fast = Core.exampleChange(s, 'api/treatment', { pace: 'max' });
      return {
        start: { picked: picked(s), count: likes(s).count, on: s.settings.take_out.likes },
        off: { picked: picked(off), plays: off.words.filter((w, i) => s.words[i][4] === 4 && w[4] === 0).length, count: likes(off).count, on: off.settings.take_out.likes,
               changed: off.changed.likes, longer: off.durations.edited > s.durations.edited },
        on: { picked: picked(on), changed: on.changed.likes, same: on.durations.edited === s.durations.edited },
        change: [Core.wordChange(s, p), Core.wordChange(back, left).path],
        back: { why: back.words[p][4], count: likes(back).count, kept: likes(back).kept_by_you, adds: likes(back).picked === likes(back).count + likes(back).left_in + likes(back).kept_by_you },
        hand: { why: hand.words[left][4], off: [handOff.words[p][4], handOff.words[left][4]], on: [handOn.words[p][4], handOn.words[left][4]] },
        atMax: picked(fast) === picked(s),
      };
    """)
    start, off = out["start"], out["off"]
    assert start["on"] is True and start["picked"] > 10
    assert off["picked"] == 0 and off["plays"] == start["picked"] and off["on"] is False, "off puts every pick back"
    assert off["count"] == start["count"], "the count still says what the switch would take"
    assert off["changed"] == -start["count"] and off["longer"]
    assert out["on"] == {"picked": start["picked"], "changed": start["count"], "same": True}
    assert out["change"][0]["path"] == "api/keep" and out["change"][0]["body"]["exact"] is True, "a double-click brings a pick back"
    assert out["change"][1] == "api/cut/add", "and cuts one Claude left in, as her own cut"
    assert out["back"] == {"why": 0, "count": start["count"] - 1, "kept": 1, "adds": True}
    assert out["hand"] == {"why": 3, "off": [0, 3], "on": [0, 3]}, "the switch leaves alone what she did by hand"
    assert out["atMax"], "the picks outlive a change of pace"


# ── what ships ──────────────────────────────────────────────────────────────


def page() -> str:
    return PAGE.read_text(encoding="utf-8")


def markup() -> str:
    """The page with its scripts and styles taken out: the tags and the words between them."""
    return re.sub(r"<script\b.*?</script>|<style\b.*?</style>", " ", page(), flags=re.S)


def part(html: str, tag: str, element_id: str) -> str:
    """The inside of ``<tag id="element_id">``, for a tag that does not hold another of its own kind."""
    match = re.search(rf'<{tag}\b[^>]*\bid="{element_id}"[^>]*>(.*?)</{tag}>', html, re.S)
    assert match, f"the page has no <{tag} id=\"{element_id}\">"
    return match.group(1)


def words_on_screen() -> str:
    """Every word the page's markup can show or say: the text, and each title and aria-label."""
    html = markup()
    told = re.findall(r'\b(?:title|aria-label|placeholder|alt)="([^"]*)"', html)
    return re.sub(r"<[^>]+>", " ", html) + " " + " ".join(told)


def sentences_in_scripts() -> list[str]:
    """The string literals in the page's scripts, comments removed: what the scripts can put on screen."""
    scripts = " ".join(re.findall(r"<script\b[^>]*>(.*?)</script>", page(), re.S))
    scripts = re.sub(r"/\*.*?\*/", " ", scripts, flags=re.S)
    return re.findall(r"'((?:[^'\\\n]|\\.)*)'|\"((?:[^\"\\\n]|\\.)*)\"", scripts)


def test_page_is_one_self_contained_file_with_no_outside_requests():
    html = page()
    assert [p.name for p in PAGE_DIR.iterdir()] == ["index.html"], "the page is one file, its example built in"
    assert not re.search(r'\b(?:src|href)\s*=\s*["\']?(?:https?:)?//', html), "no external scripts, styles or links"
    assert not re.search(r"url\(\s*['\"]?(?:https?:)?//", html)
    assert "@import" not in html
    assert not re.search(r"fetch\(\s*['\"]https?:", html)
    assert not re.search(r"https?://|\bwww\.", html), "the page holds no address outside itself"


def test_page_forces_media_muted_when_asked():
    html = page()
    assert "QS.get('muted') === '1'" in html, "the page reads ?muted=1"
    video_tags = re.findall(r"<video\b[^>]*>", html)
    assert video_tags, "the page has its video element in the markup"
    assert all(re.search(r"\bmuted\b", tag) for tag in video_tags), "the video starts muted until the page decides"
    assert not re.search(r"<audio\b|new Audio\(|AudioContext", html), "the video is the only thing that can sound"
    forced = re.search(r"function forceMuted\(\)\{(.*?)\n\}", html, re.S)
    assert forced and "m.muted = true" in forced.group(1) and "m.volume = 0" in forced.group(1)
    assert "document.addEventListener('play', forceMuted, true)" in html, "every play is muted first"
    assert "document.addEventListener('volumechange'" in html, "and it stays muted"


def test_the_top_line_has_the_status_a_small_help_button_and_one_primary_button():
    top = part(markup(), "header", "top")
    buttons = re.findall(r"<button\b[^>]*>", top)
    primary = [b for b in buttons if re.search(r'class="[^"]*\bprimary\b', b)]
    assert len(primary) == 1 and 'id="exportBtn"' in primary[0], "Export video is the one filled button up there"
    assert re.search(r'<button\b[^>]*id="exportBtn"[^>]*>Export video</button>', top)
    assert "disabled" not in primary[0], "the button works"
    help_button = re.search(r'<button\b[^>]*id="shortcutsBtn"[^>]*>([^<]*)</button>', top)
    assert help_button and help_button.group(1).strip() == "?"
    assert 'aria-label="Keyboard shortcuts"' in help_button.group(0)
    assert top.index('id="status"') < top.index('id="shortcutsBtn"') < top.index('id="exportBtn"'), "status left, then ?, then Export"
    assert 'id="copyPath"' in top


def between(html: str, start_id: str, end_id: str) -> str:
    """The markup from the element with ``start_id`` up to the element with ``end_id``."""
    start = html.index(f'id="{start_id}"')
    return html[start:html.index(f'id="{end_id}"', start)]


def test_the_video_then_its_bar_then_a_slim_transport_then_the_words():
    # Her words: "you have the video, you have the timeline just underneath the video, same width
    # as the video, and then you have the play".
    html = markup()
    stage = html[html.index('<section id="stage"'):]
    order = [stage.index(key) for key in ('class="vidwrap"', 'id="whole"', 'id="transport"', 'id="wordsHead"', 'id="words"')]
    assert order == sorted(order), "video, bar, transport, the one line of how-to, then the words"
    assert "<footer" not in html, "the bar left the foot of the page"
    transport = between(html, "transport", "wordsSec")
    order = [transport.index(f'id="{name}"') for name in ("tnow", "prevBtn", "playBtn", "nextBtn", "modeSeg")]
    assert order == sorted(order), "clock, Previous, Play, Next, Original | Edited"
    assert ">Previous<" in transport and ">Next<" in transport and ">Play<" in transport
    bar = between(html, "whole", "transport")
    assert "<button" not in bar, "the bar holds only its blocks, which the page adds"
    css = page()
    # one grid column as wide as the video holds the video, the bar, the transport and the words
    assert re.search(r"#stage\{[^}]*display:grid; grid-template-columns:minmax\(0, var\(--vid-w\)\);", css)
    assert "#stage > *{grid-column:1; min-width:0}" in css
    assert re.search(r"\.vidwrap\{[^}]*width:100%; max-width:var\(--vid-w\)", css), "the video fills the column, so the bar is exactly as wide"
    assert re.search(r"#playBtn\{min-height:32px;", css), "a slim Play"
    assert re.search(r"--transport-h:32px;", css)


def test_the_words_fill_the_height_left_and_the_video_does_not_shrink():
    css = page()
    assert re.search(r"#stage\{[^}]*grid-template-rows:auto auto auto minmax\(0,1fr\);", css), "the words' row takes what is left"
    words = re.search(r"#words\{(.*?)\}", css, re.S).group(1)
    assert "flex:1 1 auto" in words and "overflow-y:auto" in words
    assert not re.search(r"(?<![-\w])height:calc\(1\.75em \* 4", words), "no fixed four lines"
    # the video and the words split the room, and past --words-most of words the rest goes to the video:
    # at 1440 x 900 that is a 633 px video (624 before) and 11 lines of words (4 before) under the how-to line
    assert "--words-most:360px;" in css
    assert "--vid-w:max(calc(var(--room) / 2 * 16 / 9), calc((var(--room) - var(--words-most)) * 16 / 9));" in css
    room = re.search(r"--room:calc\((.*?)\);", css).group(1)
    for part in ("--top-h", "--stage-pad-t", "--stage-pad-b", "--bar-h", "--transport-h", "3 * var(--stage-gap)"):
        assert part in room, f"the room for the video and the words leaves out {part}"


def test_the_bar_is_a_heat_map_with_blocks_that_are_buttons():
    html = page()
    bar = between(markup(), "whole", "transport")
    assert "<canvas" not in html, "no drawn strip"
    assert 'role="slider"' in bar and 'id="blocks"' in bar and 'id="playhead"' in bar
    help_list = part(markup(), "div", "shortcuts")
    for word in ("a few edits", "many", "the most", "where you are"):
        assert word not in bar, "the legend left the page"
        assert word in help_list, f"the legend under the help button says {word!r}"
    drawn = re.search(r"function renderBar\(\)\{(.*?)\n\}", html, re.S)
    assert drawn and '<button type="button" class="blk"' in drawn.group(1), "each block is a button"
    assert "aria-label=\"' + esc(C.clusterName(c, false))" in drawn.group(1), "with a name a screen reader can say"
    assert 'class="num"' not in html, "no numbers under the blocks: the bar is no taller than it must be"
    assert "data-level=" in drawn.group(1)
    height = re.search(r"--bar-h:\s*(\d+)px", html)
    assert height and int(height.group(1)) >= BAR_MIN_HEIGHT_PX
    assert re.search(r"#bar\{[^}]*height:var\(--bar-h\)", html)
    assert "margin-bottom:20px" not in re.search(r"#bar\{(.*?)\}", html, re.S).group(1)
    assert re.search(r"\.blk\{[^}]*min-width:6px", html), "a short cluster can still be seen and hit"
    levels = re.findall(r"--heat[123]:rgba\((\d+),(\d+),(\d+),", html)
    assert len(levels) == 3 and len(set(levels)) == 1, "three strengths of one color"
    red = re.search(r"--red:#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})", html)
    assert tuple(int(x, 16) for x in red.groups()) != tuple(int(x) for x in levels[0]), "and that color is not the red of a cut"


def test_a_block_shows_a_play_button_under_the_pointer_or_the_keyboard():
    # Her words: "make the sections in the timeline have play buttons hovering over them so the
    # user knows that they can click them to see them go".
    html = page()
    icons = re.search(r"const BLOCK_ICONS = (.*?);\n", html, re.S).group(1)
    assert 'class="go" aria-hidden="true"' in icons and 'class="play"' in icons and 'class="pause"' in icons
    assert "+ BLOCK_ICONS +" in re.search(r"function renderBar\(\)\{(.*?)\n\}", html, re.S).group(1)
    assert ".blk:hover .go, .blk:focus-visible .go{opacity:1; transform:translate(-50%,-50%); pointer-events:auto; cursor:pointer}" in html
    assert '.blk[data-playing="true"] .go .pause{display:block}' in html, "the block that is playing offers pause"
    go = re.search(r"\.blk \.go\{(.*?)\}", html, re.S).group(1)
    assert "width:clamp(0px, calc(100% - 8px), 26px)" in go, "the circle never grows past the block, even a narrow one"


@needs_node
def test_a_block_plays_its_cluster_to_the_end_and_shows_pause_for_as_long_as_it_plays():
    out = js(f"""
      const B = {CLUSTERS};
      const idle = {{ want: false, job: 'free', key: null }};
      const press = Core.blockPress(B, 1, idle);
      const playing = {{ want: true, job: press.job, key: press.key }};
      const playedOn = {{ want: true, job: 'free', key: null }};
      return {{
        press, again: Core.blockPress(B, 1, playing), other: Core.blockPress(B, 2, playing), none: Core.blockPress(B, 7, idle),
        shows: [0, 1, 2].map(i => Core.blockPlaying(i, playing)),
        paused: Core.blockPlaying(1, {{ want: false, job: 'cluster', key: 1 }}),
        free: [0, 1, 2].map(i => Core.blockPlaying(i, playedOn)),
      }};
    """)
    assert out["press"] == {"action": "play", "job": "cluster", "key": 1, "from": 450, "stopAt": 495}, "from the cluster's start to its end"
    assert out["again"] == {"action": "pause"}, "pressing the block that plays pauses it"
    assert out["other"]["action"] == "play" and out["other"]["key"] == 2, "another block plays its own cluster instead"
    assert out["none"] is None
    assert out["shows"] == [False, True, False], "the pressed block shows pause for as long as its cluster plays"
    assert out["paused"] is False
    assert out["free"] == [False, False, False], "plain play past a cluster marks no block"
    # the stop is the one every job uses: playStep stops at stopAt
    stopped = simulate([], start=450, duration=1307, stop_at=495)
    assert stopped["why"] == "stop" and stopped["last"] < 495


def test_the_page_asks_for_nothing_it_no_longer_shows():
    html = page()
    assert "peaks" not in html, "the sound strip is gone, and so is its request"
    assert "example.json" not in html, "the example is built in"
    asked = set(re.findall(r"fetch\(\s*'([^']+)'", html)) | set(re.findall(r"send\(\s*'([^']+)'", html))
    # a change built as {path, body} and sent later is a request too
    asked |= set(re.findall(r"\bpath: '([^']+)'", html))
    assert asked == ROUTES


def test_the_taste_line_sits_under_the_usual_button_and_sends_her_to_claude_to_forget():
    html = page()
    settings = part(markup(), "aside", "settings")
    assert settings.index('id="usualBtn"') < settings.index('id="tasteLine"') < settings.index('id="h-cuts"')
    assert re.search(r'<p\b[^>]*id="tasteLine"[^>]*\bhidden\b', settings), "the line shows only when something was learned"
    assert "tasteForget" not in html and "window.confirm" not in html and "api/taste" not in html, "forgetting is not on the page"
    said = "'Learning from your changes on ' + n + (n === 1 ? ' video' : ' videos') + '. Ask Claude to forget any of it.'"
    assert said in html
    line = "Learning from your changes on 3 videos. Ask Claude to forget any of it."
    assert not banned_in(line) and not decimal_times_in(line) and "—" not in line


RATE_WORDS = ("Good cut", "Wrong cut",
              "Claude was right to cut this. Press again to clear.",
              "This should not have been cut. It goes back in. Press again to clear.",
              "Marked wrong. It is back in, so those words play now.", "Marked good.", "Rating cleared.")


def test_each_of_claudes_cuts_has_good_cut_and_wrong_cut_buttons():
    html = page()
    row = html[html.index("function cutHTML(r){"):html.index("$('groups').addEventListener('click'")]
    assert re.search(r"""data-rate="good" aria-pressed="' \+ \(r\.rating === 'good'\) \+ '" title="[^"]+">Good cut</button>""", row)
    assert re.search(r"""data-rate="bad" aria-pressed="' \+ \(r\.rating === 'bad'\) \+ '" title="[^"]+">Wrong cut</button>""", row)
    assert "role=\"group\" aria-label=\"' + esc('Rate the cut at '" in row, "the two buttons are one named group"
    assert "const rate = mine ? ''" in row, "her own cuts are not rated"
    assert "send('api/rate', { id, rating }" in html
    assert "r.rating === side ? null : side" in html, "pressing the active one again clears it"
    for said in RATE_WORDS:
        assert said in html, said
        assert not banned_in(said) and not decimal_times_in(said) and "—" not in said, said
    assert "👍" not in html and "👎" not in html, "the page uses no emoji, so the buttons are words"
    assert "'api/rate'" in re.search(r"const EDIT_ROUTES = \[[^\]]*\]", html).group(0)


@needs_node
def test_rating_a_cut_in_the_example_answers_like_the_server():
    out = js("""
      const s = Core.exampleState(), row = s.rows.find(r => r.by === 'claude');
      const good = Core.exampleChange(s, 'api/rate', { id: row.id, rating: 'good' });
      const bad = Core.exampleChange(good, 'api/rate', { id: row.id, rating: 'bad' });
      const clear = Core.exampleChange(bad, 'api/rate', { id: row.id, rating: null });
      const refuse = (state, body) => { try { Core.exampleChange(state, 'api/rate', body); return null; } catch(err){ return err.message; } };
      const at = x => x.rows.find(r => r.id === row.id);
      return {
        good: [at(good).rating, at(good).state], bad: [at(bad).rating, at(bad).state, bad.durations.edited > good.durations.edited],
        clear: [at(clear).rating, at(clear).state],
        yours: refuse(s, { id: s.rows.find(r => r.by === 'you').id, rating: 'good' }),
      };
    """)
    assert out["good"] == ["good", "kept_out"]
    assert out["bad"] == ["bad", "put_back", True]
    assert out["clear"] == [None, "put_back"], "clearing a wrong cut does not cut it again"
    assert "one of yours" in out["yours"]


def test_what_the_creator_asked_to_remove_is_gone():
    html = page()
    for gone in ("fromStart", "back5", "fwd5", "changedSec", "stripcv", "prevBusy", "nextBusy", "lookBtn", "h-keeps",
                 "samplesSec", "samples", "newSamples", "place", "barEnd"):
        assert f'id="{gone}"' not in html and f"$('{gone}')" not in html, f"{gone} is still on the page"
    screen = words_on_screen()
    for gone in ("Show all", "What changed", "Pick three new ones", "Save as my usual", "Keep no matter what",
                 "Samples, 30 seconds each", "New samples", "play a sample", "Cluster 1 of"):
        assert gone.lower() not in screen.lower(), f"{gone!r} is still on the page"
    assert "Show all" not in html
    for gone in ("playSample", "renderSamples", "markSample", "placeLine", "clusterHop", "hopCluster", "goToCluster", "WINDOW_CAP"):
        assert gone not in html, f"{gone} is still in the page"
    settings = part(markup(), "aside", "settings")
    assert "keep" not in settings.lower(), "the keep list left the left column"
    assert settings.index('id="takeList"') < settings.index('id="usualBtn"') < settings.index('id="h-cuts"'), "Make this my usual sits under the switches"
    assert re.search(r'<button\b[^>]*id="usualBtn"[^>]*\bhidden\b', settings), "and shows only when the page decides"
    look_buttons = re.findall(r'<button\b[^>]*class="look"[^>]*>', html)
    assert len(look_buttons) == 1 and 'id="lookAll"' in look_buttons[0], "one needs-a-look control, in the scripts and the markup together"
    cuts = settings[settings.index('id="h-cuts"'):]
    assert cuts.index('id="cutsCount"') < cuts.index('id="lookAll"') < cuts.index('id="groups"'), "and it sits in the Claude's cuts heading"


def test_keys_for_previous_and_next_are_listed_under_the_help_button():
    keys = re.sub(r"<[^>]+>", " ", part(markup(), "div", "shortcuts"))
    assert re.search(r"\[\s+\]\s+previous or next cut", keys)
    assert "back or forward 5 seconds" in keys and "play from the start" in keys, "their keys stay"
    assert "sample" not in keys
    html = page()
    assert "case '[': hopCut(-1); break;" in html and "case ']': hopCut(1); break;" in html
    assert "$('prevBtn').addEventListener('click', () => hopCut(-1));" in html
    assert 'title="Go to the cut before this ([)"' in html and 'title="Go to the next cut (])"' in html


@needs_node
def test_a_hop_lands_a_lead_before_the_cut_and_counts_the_cuts_it_goes_through():
    out = js(f"""
      const C = {CUTS};
      return [Core.hopTo(C, 0, 1, {LEAD}), Core.hopTo(C, 10.174, 1, {LEAD}), Core.hopTo(C, 10.174, -1, {LEAD}),
              Core.hopTo(C, 13.563, 1, {LEAD}), Core.hopTo(C, 0.2, -1, {LEAD})];
    """)
    assert out[0] == {"i": 1, "at": 1.5, "of": 4}
    assert out[1] == {"i": 3, "at": pytest.approx(13.563), "of": 4}
    assert out[2] == {"i": 1, "at": 1.5, "of": 4}
    assert out[3] is None and out[4] is None, "nothing that way: the button is dimmed and says so"
    html = page()
    hop = re.search(r"function hopCut\(dir\)\{(.*?)\n\}", html, re.S).group(1)
    assert "C.hopTo(CUTS," in hop and "JOIN_LEAD" in hop, "the page hops through the cuts that take words, landing where Hear it cut starts"


def test_an_error_never_moves_the_video_and_a_short_window_can_scroll():
    html = page()
    errbar = re.search(r"#errbar\{(.*?)\}", html, re.S).group(1)
    assert "grid-column:1 / -1" in errbar, "the error bar spans both columns, or it takes the left one and pushes the video down"
    assert "@media (min-width:821px) and (min-height:620px){ html,body{overflow:hidden} }" in html, \
        "a window shorter than the page scrolls, so the bar stays in reach"
    assert "@media (min-width:821px){ html,body{overflow:hidden} }" not in html


def test_the_words_box_shows_a_focus_ring_its_fade_cannot_hide():
    html = page()
    assert re.search(r"#wordsSec:has\(#words:focus-visible\)\{outline:2px solid var\(--gold\)", html)


def test_struck_words_stay_readable_when_picked_or_under_the_pointer():
    html = page()
    assert "#words .w.x.sel:not(.now){color:var(--red-lift)}" in html
    assert "#words .w.x:not(.now):hover{color:var(--red-lift)}" in html


def test_no_status_code_or_browser_message_reaches_the_creator():
    html = page()
    assert "server answered" not in html and "r.status +" not in html
    assert "+ err.message" not in html and "+ e.message" not in html, "a browser's own error text is not plain words"


def test_the_help_button_stays_beside_export_video_when_a_note_shows():
    top = page()
    assert top.index('id="exportNote"') < top.index('id="shortcutsBtn"') < top.index('id="exportBtn"')


def home_prefixes() -> list[str]:
    """The folders a developer's home path sits in, taken from this machine.

    Asked of the machine, not written here: the public-tree build greps every file
    for the macOS prefix, this file included. On the Mac that gives the macOS one.
    A root or container home has no shared prefix worth checking, so it adds none.
    """
    home = Path.home()
    shared = [] if home.parent == Path(home.anchor) else [str(home.parent) + "/"]
    return ["/home/", *shared]


def test_nothing_personal_or_jargon_ships():
    prefixes = home_prefixes()
    for path in PAGE_DIR.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        assert not any(prefix in text for prefix in prefixes), f"home path in {path.name}"
        assert "Downloads" not in text, f"a footage path in {path.name}"
        assert "—" not in text, f"em dash in {path.name}"
    screen = words_on_screen()
    assert not banned_in(screen), f"editor jargon on screen: {banned_in(screen)}"
    assert not decimal_times_in(screen), f"a decimal time on screen: {decimal_times_in(screen)}"


def test_the_words_she_never_has_to_learn_are_not_on_screen():
    screen = words_on_screen()
    assert not WORDS_GONE.search(screen), WORDS_GONE.search(screen)
    said = [a or b for a, b in sentences_in_scripts()]
    assert len(said) > 200, "the scripts' strings were found"
    caught = [s for s in said if WORDS_GONE.search(s)]
    assert not caught, f"a script can put these on screen: {caught}"
    sentences = [s for s in said if " " in s.strip() and "<" not in s and re.search(r"[a-z]{3}", s)]
    jargon = {s: banned_in(s) for s in sentences if banned_in(s)}
    assert not jargon, f"editor jargon a script can put on screen: {jargon}"
    # the approved words: a count ("11 edits"), the legend, the finished length in the top line, and the
    # question before she forgets what Lumr learned, which says her edits stay
    approved = (" edits", " after edits",
                "Forget what Lumr learned from your changes? Your videos and edits stay as they are.")
    elsewhere = [s for s in sentences if re.search(r"\bedits\b", s) and s not in approved]
    assert not elsewhere, f"'edits' shows only where she approved it: {elsewhere}"


# ── round 5: six stops, cutting by hand, undo ───────────────────────────────


def test_the_pace_control_draws_as_many_stops_as_it_is_sent():
    html = page()
    drawn = re.search(r"function renderPace\(\)\{(.*?)\n\}", html, re.S)
    assert drawn and "S.paces.map(p =>" in drawn.group(1), "one stop for each pace in the state, three or six"
    assert "S.paces.length" in drawn.group(1) and "gridTemplateColumns" in drawn.group(1)
    for name in PACES:
        assert f">{name}<" not in markup(), "the stops' names come from the state, not the markup"
    assert "ArrowRight" in html[html.index("$('dial').addEventListener('keydown'"):][:600], "arrow keys move one stop"
    # where six names don't fit: the first, the last and the chosen one
    assert re.search(r'#dial\[data-labels="few"\] button:not\(:first-child\):not\(:last-child\):not\(\[aria-checked="true"\]\) \.lab\{visibility:hidden\}', html)
    fit = re.search(r"function fitDial\(\)\{(.*?)\n\}", html, re.S)
    assert fit and "LABEL_GAP_PX" in fit.group(1) and "'few'" in fit.group(1)
    assert re.search(r"#dial\.many button:focus-visible \.stop\{outline:2px solid var\(--gold\)", html), "a stop shows a focus ring"


def test_the_filler_switch_shows_its_words_under_it_and_all_of_them_on_hover_or_focus():
    html = page()
    drawn = re.search(r"function renderTakeOut\(\)\{(.*?)\n\}", html, re.S).group(1)
    assert "C.fewWords(all, FILLERS_SHOWN)" in drawn and 'class="few"' in drawn
    assert 'role="tooltip"' in drawn and "aria-describedby" in drawn, "a screen reader hears the whole list"
    assert ".switch:hover + .wordtip, .switch:focus-visible + .wordtip{display:block}" in html
    assert "noneHint(k)" in drawn, "a switch with nothing to take out still says which pace would find some"
    assert "Ums and uhs" not in html


def test_the_words_offer_cut_and_keep_and_say_how():
    # Her words: "can you move the hints to above the text area, and remove the key, just the guide
    # on how to do it".
    html = markup()
    words = html[html.index('id="wordsSec"'):html.index("</section>", html.index('id="wordsSec"'))]
    head = part(words, "div", "wordsHead")
    assert words.index('id="wordsHead"') < words.index('id="words"'), "the how-to sits right above the words"
    assert re.sub(r"<[^>]+>", " ", head).split() == "Double-click a word to cut it. Drag to cut or keep a part.".split()
    assert 'id="selInfo"' in head and 'id="keptBtn"' in head, "the part she picked and what she kept share that line"
    assert 'class="wordkey"' not in words and 'id="notes"' not in html, "no key beside or under the words"
    bar = part(words, "div", "pickBar")
    assert re.findall(r"<button\b[^>]*>([^<]*)</button>", bar) == ["Cut", "Keep"]
    assert re.search(r'<div id="pickBar"[^>]*\bhidden\b', words), "they show once she has picked words"
    assert "Keep this part" not in page()
    assert words.index('id="words"') < words.index('id="pickBar"'), "the keyboard reaches Cut and Keep after the words"
    css = page()
    head_css = re.search(r"#wordsHead\{(.*?)\}", css).group(1)
    line = int(re.search(r"line-height:(\d+)px", head_css).group(1))
    gap = {"var(--s1)": 4, "var(--s2)": 8}[re.search(r"margin-bottom:([^;]+);", head_css).group(1)]
    assert line + gap < 28, "the how-to line takes the words less than one line of their own (16 px at 1.75)"


def test_the_marks_on_the_words_are_explained_under_the_help_button():
    shortcuts = part(markup(), "div", "shortcuts")
    marks = re.findall(r'<li><span class="mark ([^"]*)"[^>]*>word</span>([^<]*)</li>', shortcuts)
    assert marks == [("x auto", "taken out by the pace"), ("x", "cut by Claude"), ("x me", "cut by you"), ("kept", "kept by you")]


def test_her_cuts_are_struck_in_her_own_color_and_a_heavier_line():
    html = page()
    own = re.search(r"--user:(#[0-9a-f]{6})", html).group(1)
    red = re.search(r"--red-soft:(#[0-9a-f]{6})", html).group(1)
    assert own != red
    assert "#words .w.x.me:not(.now){color:var(--user)}" in html
    assert "#words .w.x.me{text-decoration-thickness:2px}" in html, "color is not the only sign"
    assert "#words .w.x{color:var(--red-soft)" in html, "Claude's cuts and automatic ones stay red"
    assert '.cut[data-by="you"] .said del{color:var(--user)' in html
    assert "(why === C.WHY.you ? ' me' : why === C.WHY.automatic && who ? ' auto' : '')" in html


def test_a_click_waits_for_a_double_click_so_the_video_moves_once_at_most():
    html = page()
    click = re.search(r"function clickWord\(i\)\{(.*?)\n\}", html, re.S).group(1)
    assert "setTimeout(() => { seekUser(W[i][1], true); }, CLICK_HOLD_MS)" in click, "the first click is held back, and moves the video, not the words"
    assert "clearTimeout(was.timer)" in click and "DOUBLE_CLICK_MS" in click and "toggleWord(i)" in click
    assert click.index("clearTimeout(was.timer)") < click.index("toggleWord(i)"), "the held click is dropped before the word is cut"
    hold = int(re.search(r"const CLICK_HOLD_MS = (\d+);", html).group(1))
    double = int(re.search(r"const DOUBLE_CLICK_MS = (\d+);", html).group(1))
    assert 150 <= hold <= 400 and hold <= double


def test_undo_sits_in_the_top_line_and_has_a_key():
    top = part(markup(), "header", "top")
    undo = re.search(r'<button\b[^>]*id="undoBtn"[^>]*>([^<]*)</button>', top)
    assert undo and undo.group(1) == "Undo" and re.search(r"\bhidden\b", undo.group(0)), "it shows only after she did something"
    assert top.index('id="status"') < top.index('id="undoBtn"') < top.index('id="shortcutsBtn"')
    html = page()
    assert "case 'z': case 'Z': undoLast(); break;" in html
    assert "send('api/undo', {}, { action: 'undo'" in html
    assert re.search(r"const UNDO_SHOWS_MS = \d+;", html)


def test_undo_shows_only_after_she_did_something_to_the_words():
    # The server's undo holds her cuts and keeps, never the pace, a slider or a switch.
    # So a move of one of those shows no Undo control, and hides one that was showing.
    html = page()
    applied = re.search(r"function applyState\(state, how\)\{(.*?)\n\}", html, re.S).group(1)
    assert "const did = how && how.action;" in applied
    assert "if(did && did !== 'undo' && S.can_undo) showUndo(); else hideUndo();" in applied
    moves = re.findall(r"send\('api/treatment',[^\n]*", html)
    assert len(moves) == 3, "the pace, a slider, a switch"
    assert all("action:" not in move for move in moves), "none of them is something Undo takes back"
    assert 'title="Take back your last cut or keep (Z)"' in markup()


def test_the_new_keys_are_listed_under_the_help_button():
    keys = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", part(markup(), "div", "shortcuts")))
    assert "X cut the words you picked, or the word being said" in keys
    assert "K keep the part you picked" in keys
    assert "Z undo. It covers your cuts and keeps, not the pace, the sliders or the switches." in keys, \
        "the list says what Undo covers: a slider move is not undone"
    html = page()
    assert "case 'x': case 'X': if(words.sel) cutPart(); else toggleWord(words.cur); break;" in html
    assert "case 'k': case 'K': keepPart(); break;" in html
    handler = html[html.index("document.addEventListener('keydown'"):]
    taken = re.findall(r"case '(.)':", handler[:handler.index("});")])
    assert len(taken) == len(set(taken)), f"a key does two things: {taken}"


def test_your_cuts_show_only_when_she_has_one():
    html = page()
    drawn = re.search(r"function renderGroups\(\)\{(.*?)\n\}", html, re.S).group(1)
    assert "S.groups.filter(g => g.count > 0)" in drawn, "a group with no cuts is not drawn, and that goes for hers"
    assert """setText('cutsTitle', n.mine ? 'Cuts' : "Claude's cuts")""" in drawn
    assert "' yours</span>" in drawn and "Claude’s" in drawn, "two counts once she has cut something"
    row = re.search(r"function cutHTML\(r\)\{(.*?)\n\}", html, re.S).group(1)
    assert ">Take back</button>" in row and "Hear it cut" in row
    assert "(mine ? '' : '<p class=\"reason\">'" in row, "her rows carry no reason"
    assert "data-back=" in row and "send" not in row
    assert "path: 'api/cut/remove', body: { id: r.id }" in html


def test_a_server_from_before_this_round_is_told_apart():
    html = page()
    assert "const CANT_CUT = 'This page can’t cut words yet.';" in html
    act = re.search(r"function doToWords\(change\)\{(.*?)\n\}", html, re.S).group(1)
    assert "err.status === 404" in act and "say(change.kind === 'back' ? CANT_BRING_BACK : CANT_CUT); return;" in act
    assert "showError" in act[act.index("return;"):], "any other refusal shows as an error"
    assert "err.status = r.status;" in html
    toggle = re.search(r"function toggleWord\(i\)\{(.*?)\n\}", html, re.S).group(1)
    assert "!C.tellsWho(W)" in toggle and toggle.index("!C.tellsWho(W)") < toggle.index("doToWords(change)")


# ── round 5, second wave: fine tune, filler likes, three fixes ──────────────


def test_fine_tune_is_a_line_that_opens_two_sliders_and_starts_closed():
    html = markup()
    pace = html[html.index('id="h-pace"'):html.index('id="h-take"')]
    assert pace.index('id="paceSummary"') < pace.index('id="fine"') , "under the pace summary"
    button = re.search(r'<button\b[^>]*id="fineBtn"[^>]*>', pace).group(0)
    assert 'aria-expanded="false"' in button and 'aria-controls="fineBox"' in button
    assert re.search(r'<div id="fineBox"[^>]*\bhidden\b', pace), "closed on arrival"
    assert re.search(r'<div id="fine" hidden>', pace), "and not drawn at all until the state carries fine values"
    assert ">Fine tune<" in pace
    script = page()
    drawn = re.search(r"function renderFine\(\)\{(.*?)\n\}", script, re.S).group(1)
    assert "wrap.hidden = !sliders.length;" in drawn, "a server from before Fine tune gets no Fine tune line"
    assert '<input type="range"' in drawn and '<label for="' in drawn and "<output" in drawn
    assert "aria-valuetext" in drawn and "s.says[at]" in drawn, "each slider says its value in words"
    assert "fine.held === s.key" in drawn, "an answer never moves the slider under her hand"


def test_the_page_remembers_fine_tune_open_or_closed_and_works_when_storage_is_refused():
    html = page()
    read = re.search(r"function fineOpenBefore\(\)\{(.*?)\}\n", html, re.S).group(1)
    write = re.search(r"function fineOpenKeep\(open\)\{(.*?)\}\n", html, re.S).group(1)
    for body in (read, write):
        assert body.strip().startswith("try{") and "catch(err)" in body and "localStorage" in body
    assert "return false;" in read, "with storage refused it starts closed"
    assert len(re.findall(r"localStorage", html)) == 2, "storage is touched in those two places only"
    assert "sessionStorage" not in html and "document.cookie" not in html


def test_a_drag_asks_at_most_five_times_a_second_and_once_more_on_release():
    html = page()
    every = int(re.search(r"const FINE_ASK_MS = (\d+);", html).group(1))
    assert every >= 200, "five times a second at most"
    moved = re.search(r"function fineMoved\(input, last\)\{(.*?)\n\}", html, re.S).group(1)
    assert "setText('fine-' + s.key + '-now', s.texts[at]);" in moved, "the value beside the slider changes at once"
    assert moved.index("setText(") < moved.index("askFine()"), "before anything is asked"
    assert "if(last){ clearTimeout(fine.timer); fine.timer = 0; askFine(); return; }" in moved
    assert "if(fine.timer) return;" in moved and "const wait = fine.askedAt + FINE_ASK_MS - performance.now();" in moved
    assert "if(wait <= 0) askFine();" in moved and "setTimeout(() => { fine.timer = 0; askFine(); }, wait)" in moved
    ask = re.search(r"function askFine\(\)\{(.*?)\n\}", html, re.S).group(1)
    assert "send('api/treatment', { fine: values }" in ask and "pace" not in ask, "a stop and her own values never go in one ask"
    assert "Object.assign(waiting.body.fine, values)" in ask, "an ask still waiting takes the newer values"
    assert "fine.want[k] !== fine.sent[k]" in ask, "a value that was asked for already is not asked for again"
    assert "fineMoved(e.target, false)" in html and "fineMoved(e.target, true)" in html
    assert "stopPlayback" not in moved + ask and ".pause()" not in moved + ask, "the video keeps playing while she drags"
    assert "action:" not in ask, "slider moves are left out of undo: the server's undo does not hold the pace"


def test_with_a_setting_of_her_own_no_stop_is_filled():
    html = page()
    drawn = re.search(r"function renderPace\(\)\{(.*?)\n\}", html, re.S).group(1)
    assert "const own = cur === C.CUSTOM;" in drawn
    assert "b.setAttribute('aria-checked', on ? 'true' : 'false');" in drawn and "const on = b.dataset.pace === cur;" in drawn
    assert "b.tabIndex = on || (own && !i) ? 0 : -1;" in drawn, "the keyboard can still reach the stops"
    assert "' · saves <b>'" in drawn
    assert "Custom" not in markup(), "the name comes from the state"


def test_the_filler_likes_switch_cannot_be_pressed_until_claude_has_picked():
    html = page()
    drawn = re.search(r"function renderTakeOut\(\)\{(.*?)\n\}", html, re.S).group(1)
    assert "k.key === LIKES ? C.likesView(k) : null" in drawn
    assert "const on = ready && takeOn(k.key);" in drawn, "drawn off while Claude has not picked, whatever the saved switch says"
    assert "(ready ? '' : ' aria-disabled=\"true\"')" in drawn
    assert "b.getAttribute('aria-disabled') === 'true') return;" in html
    assert "S.take_out.map(k =>" in drawn, "three switches from a server of before, four from this one"
    assert "Filler likes" not in markup() and "Filler words" not in markup(), "the labels come from the state"


def test_what_the_pace_took_out_is_struck_quietly():
    html = page()
    assert "#words .w.x.auto:not(.now){color:var(--text3)}" in html
    assert "#words .w.x.auto{text-decoration-thickness:.5px}" in html
    assert "#words .w.x.auto.sel:not(.now), #words .w.x.auto:not(.now):hover{color:var(--text2)}" in html, "and stays readable when lit"
    assert "pick: 4" in html, "Claude's picks are struck like Claude's cuts: they get no class of their own"
    assert ".pick" not in re.search(r"<style>(.*?)</style>", html, re.S).group(1)



# ── the layout round: every word of the take, and a word any cut touches ───

# Words [text, start, end, removed, who] and removed ranges, as a state sends them.
TOUCHED = """{
  words: [['recipes,', 20.183, 20.752, 0, 0], ['positivity', 23.251, 23.521, 0, 0], ['at', 25.07, 25.177, 0, 0],
          ['certain', 25.231, 25.465, 0, 0], ['times', 25.512, 25.694, 0, 0], ['(sound)', 84.08, 84.72, 0, 0],
          ['mine', 90.0, 90.4, 0, 0], ['claude', 91.0, 91.4, 0, 0], ['picked', 92.0, 92.3, 0, 0], ['gone', 93.0, 93.4, 1, 2]],
  removed: [[19.032, 20.104], [23.521, 24.89], [25.2, 25.24], [25.692, 25.7], [83.771, 84.398], [90.3, 90.6], [91.3, 91.6],
            [92.1, 92.35], [92.9, 93.5]],
  rows: [{ by: 'you', start: 90.3, end: 90.6 }, { by: 'claude', state: 'kept_out', start: 91.3, end: 91.6 },
         { by: 'claude', state: 'kept_out', start: 92.9, end: 93.5 }],
  trims: [[19.032, 20.104, 'pauses'], [23.521, 24.89, 'pauses'], [25.2, 25.24, 'pauses'], [25.692, 25.7, 'pauses'], [83.771, 84.398, 'pauses']],
}"""


@needs_node
def test_a_word_any_removal_takes_part_of_shows_struck_in_that_removals_style():
    out = js(f"""
      const s = {TOUCHED}, R = Core.normalizeRanges(s.removed, 1307);
      const shown = Core.struckWords(s.words, R, s.rows, s.trims);
      return {{ shown: shown.map(w => [w[0], w[3], w[4]]), same: shown.map((w, i) => w === s.words[i]), sent: s.words.map(w => w[3]) }};
    """)
    assert out["shown"] == [
        ["recipes,", 0, 0],        # a range that ends before the word starts takes none of it
        ["positivity", 0, 0],     # one that starts exactly at its recorded end takes none of it either
        ["at", 0, 0],
        ["certain", 1, 1],      # 0.009 s of its start: the pace's, struck quietly
        ["times", 0, 0],         # 0.002 s: the millisecond rounding of two times, which takes nothing
        ["(sound)", 1, 1],        # a laugh the pace takes the first half of
        ["mine", 1, 3],           # her own cut takes its end
        ["claude", 1, 2],         # Claude's cut
        ["picked", 1, 4],         # no row and no trim: one of Claude's picks
        ["gone", 1, 2],           # struck already by Claude's side: as sent
    ]
    assert out["same"] == [True, True, True, False, True, False, False, False, False, True], "a word that gains a strike is a copy"
    assert out["sent"] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], "the state's own words are not changed"


@needs_node
def test_a_word_the_page_struck_comes_back_with_a_double_click_and_is_never_cut_again():
    out = js(f"""
      const s = {TOUCHED}, R = Core.normalizeRanges(s.removed, 1307);
      const shown = Object.assign({{}}, s, {{ words: Core.struckWords(s.words, R, s.rows, s.trims) }});
      return [3, 5, 7].map(i => Core.wordChange(shown, i)).concat([Core.wordChange(s, 3)]);
    """)
    assert out[0] == {"kind": "back", "path": "api/keep", "body": {"start": 25.231, "end": 25.465, "exact": True}}
    assert out[1]["kind"] == "back" and out[1]["body"] == {"start": 84.08, "end": 84.72, "exact": True}
    assert out[2]["kind"] == "back" and out[2]["path"] == "api/keep", "Claude's cut splits around the word"
    assert out[3]["kind"] == "cut", "the words as sent would cut it, which is why the page sends the words it shows"
    html = page()
    assert "function shownState(){ return Object.assign({}, S, { words: W }); }" in html
    assert "C.wordChange(shownState(), i)" in html and "C.cutChange(shownState(), p.a, p.b)" in html
    assert "W = C.shownWords(S, DUR);" in html


@needs_node
def test_words_from_before_who_was_sent_are_only_marked_removed():
    out = js("""
      const W = [['a', 1, 2, 0], ['b', 2.1, 3, 0]];
      return Core.struckWords(W, [[2.5, 4]], [], undefined).concat(Core.struckWords(null, [], [], []));
    """)
    assert out == [["a", 1, 2, 0], ["b", 2.1, 3, 1]]
    assert js("return Core.struckWords([['a', 1, 2, 0, 0]], [[1.5, 3]], [], undefined)[0];") == ["a", 1, 2, 1, 1], \
        "a state that lists no trims says the pace"


def test_the_words_box_holds_every_word_and_scrolls_freely():
    # Her words: "The scrolling of the text feels limited, like I can't move further into the timeline."
    html = page()
    words = html[html.index("const words = {"):html.index("words.box.addEventListener('pointerdown'")]
    for gone in ("WINDOW_BEFORE", "WINDOW_AFTER", "WINDOW_CAP", "USER_SCROLL_MS", "userScrollAt", "w0", "w1"):
        assert gone not in words, f"{gone}: the words are no longer a window around the video"
    css = re.search(r"#words\{(.*?)\}", html, re.S).group(1)
    assert "overflow-y:auto" in css and "overflow-anchor:none" in css


@needs_node
def test_the_words_follow_the_video_until_she_scrolls_them_and_come_back_when_she_plays_or_jumps():
    out = js("""
      const box = { following: true, jumped: false, playing: true, line: 28, top: 1000, height: 356 };
      const at = (word, more) => Core.followTo(Object.assign({}, box, { word }, more || {}));
      return {
        reading: at(1000 + 150),                      // 150 px down while playing: stays
        low: at(1000 + 250),                          // past 60% while playing: moves on
        above: at(1000 - 40),                         // above the box: comes back
        away: at(1000 + 250, { following: false }),   // she scrolled away: never moves
        jump: at(1000 + 150, { jumped: true }),       // a jump: to 30% down wherever it is
        pausedLow: at(1000 + 250, { playing: false }),// paused, still in sight: stays
        pausedGone: at(1000 + 340, { playing: false }),
        hers: [Core.herScroll(1200, 1000, false), Core.herScroll(1001, 1000, false), Core.herScroll(1200, 1000, true)],
      };
    """)
    assert out["reading"] is None and out["pausedLow"] is None
    assert out["low"] == pytest.approx(1250 - 356 * 0.3) and out["above"] == pytest.approx(960 - 356 * 0.3)
    assert out["away"] is None, "it never scrolls the words out from under her"
    assert out["jump"] == pytest.approx(1150 - 356 * 0.3)
    assert out["pausedGone"] == pytest.approx(1340 - 356 * 0.3)
    assert out["hers"] == [True, False, False], "a box that changed size moved by itself, not by her hand"
    html = page()
    start = re.search(r"function startPlayback\(opts\)\{(.*?)\n\}", html, re.S).group(1)
    assert "words.rejoin(true);" in start, "pressing play brings them back"
    for event in ("wheel", "touchmove", "scroll"):
        assert f"words.box.addEventListener('{event}'" in html, f"{event} on the words is her own scrolling"


def test_a_word_already_heard_keeps_the_size_of_the_words():
    # The cut rows' quoted words are .said, and so is a word already heard. Unscoped, the rows'
    # 14 px reached the words, and the whole take shrank as it played.
    html = page()
    assert re.search(r"^\.cut \.said\{", html, re.M)
    assert not re.search(r"^\.said\{", html, re.M)
    assert "#words .w.said{color:var(--text)}" in html


def test_the_hint_column_beside_the_words_is_gone():
    html = page()
    for gone in ("fitNotes", "roomy", "--notes-w", "--notes-gap", "--one-column", 'id="notes"'):
        assert gone not in html, f"{gone}: the how-to is one line above the words at every width"


def test_the_bar_is_explained_under_the_help_button():
    shortcuts = part(markup(), "div", "shortcuts")
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", shortcuts))
    assert "Each block is a stretch where a lot changed. Press a block to play that stretch; it stops at the block's end." in text
    assert 'class="legend"' in shortcuts


@needs_node
def test_a_word_between_two_pauses_a_sliver_apart_is_not_struck():
    # At Max on the test take, "to" (40 ms) sits between two pauses the pace takes. Playback joins the
    # two, since a gap that short can't be heard, but the export keeps the word. Struck as if the joined
    # range took it, and with no row or trim of its own, it showed as one of Claude's cuts.
    out = js("""
      const s = { words: [['want', 1253.3, 1253.606, 0, 0], ['to', 1254.067, 1254.107, 0, 0], ['go', 1254.207, 1254.5, 0, 0]],
                  removed: [[1253.606, 1254.067], [1254.107, 1254.207]], rows: [],
                  trims: [[1253.606, 1254.067, 'pauses'], [1254.107, 1254.207, 'pauses']] };
      const joined = Core.normalizeRanges(s.removed, 1307);
      return { shown: Core.shownWords(s, 1307).map(w => w[3]), joined, exact: Core.normalizeRanges(s.removed, 1307, 0),
               before: Core.struckWords(s.words, joined, s.rows, s.trims)[1] };
    """)
    assert out["shown"] == [0, 0, 0], "the page strikes only what the edit removes"
    assert out["joined"] == [[1253.606, 1254.207]], "playback still skips the sliver"
    assert out["exact"] == [[1253.606, 1254.067], [1254.107, 1254.207]]
    assert out["before"] == ["to", 1254.067, 1254.107, 1, 4], "the bug: the joined range struck it as Claude's pick"


@needs_node
def test_previous_and_next_go_through_the_cuts_that_take_words_and_skip_pure_pauses():
    # At Standard on the test take 165 of 242 removed ranges are pauses, a median 3.3 s apart: stopping
    # at each made Next a nudge. Now it stops where words go: Claude's cuts, hers, picked likes, fillers.
    out = js("""
      const words = [['so', 1, 1.3, 1, 1], ['um', 5.1, 5.4, 1, 1], ['(laugh)', 9, 10, 1, 1], ['this', 12, 12.4, 0, 0],
                     ['that', 20, 20.5, 1, 2], ['like,', 30, 30.3, 1, 4]];
      const R = [[0.9, 1.35], [3, 3.8], [5.05, 5.5], [9.4, 9.9], [15, 16], [19.9, 22], [30, 30.3], [40, 41]];
      const rows = [{ by: 'claude', state: 'kept_out', start: 19.9, end: 22 }, { by: 'claude', state: 'put_back', start: 40, end: 41 },
                    { by: 'you', state: 'kept_out', start: 15, end: 16 }];
      return { cuts: Core.hopCuts(R, words, rows), pauses: Core.hopCuts([[3, 3.8], [8, 9]], [['a', 1, 2, 0, 0]], []),
               none: Core.hopCuts([], [], []) };
    """)
    assert out["cuts"] == [[0.9, 1.35], [5.05, 5.5], [15, 16], [19.9, 22], [30, 30.3]], \
        "words go there, or a cut of hers or Claude's is there; a pause, part of a laugh, or a cut put back is skipped"
    assert out["pauses"] == [[3, 3.8], [8, 9]], "with no cut that takes words, every range"
    assert out["none"] == []


# ── picking words from anywhere in the box ──────────────────────────────────
# Her words: "dragging in the empty space of the text doesnt capture the drag and select, it only works
# when click on a word itself, doing multi line edits is hard."

# Three lines of words, 28 px apart, each word box 20 px tall; the second line ends early.
LAID_OUT = """
  const rows = [[[10, 60], [70, 120], [130, 200], [210, 290]], [[10, 50], [60, 140]], [[10, 90], [100, 170], [180, 260]]];
  const boxes = [];
  rows.forEach((row, n) => row.forEach(([left, right]) => boxes.push({ left, right, top: 14 + n * 28, bottom: 34 + n * 28 })));
  const at = (x, y) => Core.nearestWord(boxes.length, i => boxes[i], x, y);
"""


@needs_node
def test_the_nearest_word_to_a_point_anywhere_in_the_box():
    out = js(LAID_OUT + """
      return {
        on: [at(30, 20), at(95, 24), at(250, 76)],
        between: [at(65, 20), at(127, 24)],          // the space between two words on a line: a tie, then nearer the second
        lineEnd: [at(280, 52), at(400, 52)],          // past the end of the short second line, near and far
        gaps: [at(30, 37), at(30, 39)],               // between lines: nearer the line above, nearer the line below
        margin: [at(2, 80), at(300, 20)],             // left and right margins
        above: at(150, 0), below: [at(40, 200), at(400, 500)],
        none: Core.nearestWord(0, () => null, 5, 5),
      };
    """)
    assert out["on"] == [0, 1, 8]
    assert out["between"] == [0, 2], "a tie goes to the word before; otherwise the nearer word"
    assert out["lineEnd"] == [5, 5], "past the end of a line: its last word"
    assert out["gaps"] == [0, 4]
    assert out["margin"] == [6, 3]
    assert out["above"] == 2 and out["below"] == [6, 8], "under the last line: the nearest word on it"
    assert out["none"] == -1


@needs_node
def test_a_drag_near_the_edge_scrolls_the_words_slowly_then_faster():
    out = js("return [100, 140, 133, 132, 116, 300, 380, 400, 700, 50, 0].map(y => Core.edgeScroll(y, 100, 400));")
    assert out[1] == 0 and out[5] == 0 and out[2] == 0, "well inside the box nothing scrolls"
    assert out[0] < 0 and out[9] < 0 and out[10] < 0, "near or past the top it scrolls up"
    assert out[6] > 0 and out[7] > 0 and out[8] > 0, "near or past the bottom it scrolls down"
    assert abs(out[4]) <= abs(out[0]) <= abs(out[9]) <= abs(out[10]) == 14, "faster the farther out, up to 14 px a frame"
    assert out[6] <= out[7] <= out[8] == 14
    assert abs(out[0]) <= 4, "slow at the edge itself, so she can stop where she means to"


@needs_node
def test_shift_click_picks_from_the_end_she_did_not_move_last():
    out = js("""
      return [Core.pickAnchor({ a: 10, b: 20, at: 20 }, 3, 7), Core.pickAnchor({ a: 10, b: 20, at: 10 }, 3, 7),
              Core.pickAnchor(null, 3, 7), Core.pickAnchor(null, -1, 7), Core.pickAnchor(null, null, -1)];
    """)
    assert out == [10, 20, 3, 7, 0], "her part's fixed end, else the word she clicked, else the word being heard"


def test_the_words_take_a_press_anywhere_and_a_finger_still_scrolls():
    html = page()
    down = html[html.index("words.box.addEventListener('pointerdown'"):html.index("words.box.addEventListener('pointermove'")]
    assert "words.nearestAt(e.clientX, e.clientY)" in down, "a press between the words starts at the nearest word"
    assert "if(onWord == null && touch) return;" in down, "a finger on the space between words scrolls, as before"
    assert "paddingRight" in down, "a press on the scrollbar's strip scrolls"
    assert "e.shiftKey" in down and "C.pickAnchor(" in down
    up = html[html.index("words.box.addEventListener('pointerup'"):html.index("function clickWord(i)")]
    assert "if(!d.onWord) return;" in up, "a click between words does what it always did: nothing"
    assert "clickWord(d.from)" in up, "a click on a word goes there; two cut it or bring it back"
    words = html[html.index("const words = {"):html.index("words.box.addEventListener('pointerdown'")]
    assert "this.leave();" in words[words.index("dragTo(){"):], "a drag is her own hand on the words: they stop following"
    assert "C.edgeScroll(" in words
    moved = html[html.index("words.box.addEventListener('scroll'"):]
    assert "words.dragTo();" in moved[:600], "the wheel in the middle of a drag carries the drag on"


def test_shift_click_is_listed_under_the_help_button():
    keys = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", part(markup(), "div", "shortcuts")))
    assert "Shift + click click a word, then Shift-click another: pick every word between" in keys
    assert "Shift-click" in part(markup(), "span", "wordsHint")



# ── the bar under the video: a scrubber ─────────────────────────────────────
# Her words: "dragging the little timeline editor doesnt work very well, it doesnt respond to clicking
# and dragging". A press on a block waited for a drag, and a press that could start a text selection or
# the browser's own drag let the browser take the pointer mid-drag (pointercancel), so the scrub died.


def test_a_press_anywhere_on_the_bar_scrubs_and_the_drag_is_captured():
    html = page()
    down = html[html.index("bar.box.addEventListener('pointerdown'"):html.index("bar.box.addEventListener('pointermove'")]
    assert "e.preventDefault();" in down, "no text selection or page drag can take the pointer away"
    assert "bar.box.setPointerCapture(e.pointerId)" in down, "the drag goes on off the bar until she lets go"
    assert "if(!blk) seekUser(barTime(e.clientX));" in down, "a press on the bar or a block's body moves the video at once"
    assert "e.target.closest('.go')" in down, "only the play button waits to see whether it is a press or a drag"
    assert "window.addEventListener('pointermove'" not in html, "the capture carries the drag, not listeners on the window"
    ended = re.search(r"function endPress\(e\)\{(.*?)\n\}", html, re.S).group(1)
    assert "e.type === 'pointerup' && !p.moving && p.block >= 0" in ended, "the play button plays only when let go without a drag"
    for event in ("pointerup", "pointercancel", "lostpointercapture"):
        assert f"'{event}'" in html[html.index("function endPress"):html.index("function endPress") + 800]
    assert "if(b && e.detail === 0) playCluster(+b.dataset.i);" in html, "Enter or Space on a block plays it"
    assert "user-select:none" in re.search(r"#whole\{(.*?)\}", html).group(1)
    assert re.search(r"#bar\{[^}]*touch-action:none", html), "a finger scrubs rather than scrolls the page"
    height = int(re.search(r"--bar-h:\s*(\d+)px", html).group(1))
    assert height >= 24


def test_a_focused_block_plays_on_space_or_enter_and_hops_on_the_arrows():
    # A reviewer found that Space on a focused block reached the document handler and toggled free
    # play instead, and that the arrow keys did nothing there: the track's own handler never saw them.
    html = page()
    assert "bar.track.addEventListener('keydown'" not in html, "one handler covers the track and its blocks"
    handler = html[html.index("bar.box.addEventListener('keydown'"):html.index("bar.box.addEventListener('keydown'") + 800]
    assert "e.target.closest('.blk')" in handler
    assert "e.key === ' ' || e.key === 'Enter'" in handler and "playCluster(+blk.dataset.i)" in handler, \
        "Space and Enter on a block play its cluster, not the document's own play or pause"
    assert "ArrowLeft: -HOP, ArrowRight: HOP" in handler, "arrows move the video by the same hop as the track"
    assert "seekUser(t + map[e.key])" in handler


def test_a_pick_from_the_space_between_words_needs_a_real_move():
    html = page()
    assert int(re.search(r"const PICK_START_PX = (\d+);", html).group(1)) >= 8, "a trackpad tap that drifts a few px picks nothing"
    assert "> PICK_START_PX;" in html[html.index("  dragTo(){"):]
    assert "if(d.onWord ? i === d.from : !far) return;" in html, \
        "a press from empty space needs the real distance, whichever word ends up nearest"


def test_cut_and_keep_never_sit_over_the_words_she_picked():
    html = page()
    placed = html[html.index("  placeBar(){"):html.index("  /** Keep the word being heard in view")]
    assert "this.el(p.a), last = this.el(p.b)" in placed, "the bar goes by the whole part, not only the end she moved"
    assert "const fitsOver = over >= box.top, fitsUnder = under + h <= box.bottom;" in placed
    assert "bar.classList.add('tight')" in placed and "$('wordsHead')" in placed, "a part that fills the box: the bar sits above the words"
