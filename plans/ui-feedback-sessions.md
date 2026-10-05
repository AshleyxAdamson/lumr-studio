# Plan: UI feedback sessions in Lumr Studio

A handoff for the agent that builds and tests this on a Mac. Read all of it
before you change anything.

## The goal

Ashley records her screen while she scrolls through an app or website and
talks through her feedback. She gives Claude the recording. Claude turns what
she said into a numbered list of feedback notes, each tied to a time in the
video and to screenshots from that moment, so Claude can see exactly what she
was talking about. Then, if she wants, Claude works through the notes in the
app's code.

The transcript with per-word times already exists in Lumr Studio
(`transcribe`, `read_transcript`, `find_words`). What's missing:

1. A tool that returns screenshots of the source video at given times, as
   images Claude sees. Call it `frames`.
2. A skill, `ui-feedback`, that walks Claude through a feedback session.

## Decisions already made

Ashley can override these. Ask her before you change them.

- **`frames` is on by default.** It goes in `EDITOR_TOOLS` in
  `offering.py`, not behind `LUMR_STUDIO_EXTRAS`. The default tool count goes
  from 11 to 12, and from 15 to 16 with every extra on.
- **The skill ships in `skills/ui-feedback/`**, beside `tight-cut`.
- **The notes file is optional.** The skill shows the notes in chat. Only
  when she says yes does it write `ui-feedback/<video stem>.md` in the project
  she's working in, linking each screenshot by its path where `frames` saved
  it. No copying of screenshots unless she asks.

## How this repo works (read these first)

- `TOOLS.md` is the contract. Its rule: **change `TOOLS.md` first, then the
  code.**
- `server/lumr_studio/server.py`: MCP wiring only. Each tool parses inputs
  and calls a plain function in `tools.py`. `look` (around line 200) is the
  model for a tool that returns an image: an `ImageContent` block plus
  `json_text(data)`, with `structured_content=data`.
- `server/lumr_studio/tools.py`: one plain function per tool, no MCP.
  Collaborators that touch the outside world are keyword parameters so tests
  can pass fakes. `tools.look` (line ~673) is the pattern: `open_project`,
  `reserve_unique_path` into a project folder, `append_receipt`, unlink the
  reserved file on failure.
- `server/lumr_studio/offering.py`: the one list of tool names. A tool in no
  list raises.
- `server/lumr_studio/project.py`: the project folder. Add a `frames_dir`
  property (`root / "frames"`) next to `looks_dir`.
- `server/lumr_studio/look.py`: `Footage` and `_run` show how this repo calls
  ffmpeg: `-nostdin`, stderr to a log file in a temp dir (never a pipe that
  can fill), a timeout, and a `StudioError` whose message says how to fix it.
  Reuse `_run` (or move it somewhere shared) instead of writing a second one.
- `server/lumr_studio/edit.py`: `clock(seconds)` gives `m:ss`. Use it for
  every time a person reads. Never format times by hand.
- Errors: raise `StudioError` with a message that says what was wrong and
  how to fix the call. `mcp_results.run` turns it into a tool error.
- Never edit `lumr_studio/engine/` or `lumr_studio/speech/`: they are
  generated copies (see `server/README.md`).
- Voice for anything a person reads (skill, README, GUIDE, PRIVACY): short
  plain sentences, contractions, no em dashes, no marketing words. Match
  `skills/tight-cut/SKILL.md`.

## Part 1: the `frames` tool

### Contract (write this into `TOOLS.md` first)

Add a `### frames` section after `### look`. Also:

- "What the plugin offers": 11 → 12 tools, add `frames` to the list; "all
  15" → "all 16".
- "Project folder": add `frames/  screenshots of the source video made by frames`.

| Input | Type | Note |
|---|---|---|
| `video_path` | string | |
| `times` | list of floats, 1 to 6 | source seconds, each from 0 to the video's length |
| `region` | list of 4 floats, optional | `[left, top, right, bottom]` as fractions of the frame (0 to 1), to zoom in on part of the screen. Applies to every time in the call. |

Read only: no (it writes files). Destructive: no.

Returns, in this order: for each time, a text block such as
`Frame 2 of 3 at 1:23 (83.40 s)` followed by that frame as an image; then one
compact JSON text block:

```json
{"frames": [{"at": 83.4, "shown": 83.367, "clock": "1:23", "path": "...", "width": 1568, "height": 980}],
 "source_size": [2880, 1800], "region": null}
```

- `at` is the time asked for. `shown` is the time of the frame actually on
  screen then (see "The frame on screen at a time" below).
- `structured_content` is that same dict.
- It works whether or not the video has a transcript or a saved edit. Times
  are always source times, never edited times.

Limits:

- At most 6 times a call. More is an error that says to split the call.
- Long edge at most 1568 px (`look.MAX_LONG_EDGE`), after the crop. Never
  upscale.
- PNG when the file is at most 1 MB (`look.MAX_FILE_BYTES`), else JPEG at
  quality 90. A UI screenshot is mostly flat colour, so PNG usually fits and
  keeps small text crisp.
- `region` must have `0 <= left < right <= 1` and `0 <= top < bottom <= 1`,
  and the crop must be at least 64 px on each side in source pixels.
  Otherwise, an error that says what to pass.
- A time below 0 or past the end is an error that names the video's length.

Files: `frames/frame-<m>m<ss.ss>s-<YYYYmmdd-HHMMSS>.png` (or `.jpg`) via
`reserve_unique_path`, so nothing is ever overwritten. Add a receipt line
(`step: "frames"`, `times`, `paths`).

### The frame on screen at a time (the tricky part)

macOS screen recordings are **variable frame rate**: QuickTime and
Cmd-Shift-5 only write a frame when the screen changes. A still screen can go
many seconds without a frame. So the obvious `ffmpeg -ss T -i video
-frames:v 1` returns the **next** frame after T, which can be after she
scrolled away. That's exactly the wrong picture. A short decode window before
T, using `-t`, fails too: it returns nothing, or lets the next frame through.

This was checked with ffmpeg on a VFR test video (screens changing at 0, 4,
10 and 13 s). Asking for 7 s with the naive command returned the 10 s frame.
The method below returned the right frame at 0, 3.9, 4.0, 7, 9.99, 10, 11
and 13 s:

1. List the frame times near T without decoding, from packet times:
   `ffprobe -v error -select_streams v:0 -read_intervals "<max(0,T-2)>%<T+0.001>" -show_entries packet=pts_time -of csv=p=0 <video>`.
   `-read_intervals` seeks to the keyframe before the start, so this
   usually reaches back far enough by itself.
2. Take the largest time at or before T. If there is none, widen the window
   back (10 s, then 60 s, then from 0) and try again.
3. Decode exactly that frame:
   `ffmpeg -nostdin -v error -ss <shown - 0.0005> -i <video> -frames:v 1 ...`.
   Crop and scale in the same command
   (`-vf crop=...,scale=...`), or in Pillow afterwards. Either is fine.

Also handle a non-zero container `start_time` (some `.mov` files have one).
ffprobe's `pts_time` is absolute, ffmpeg's `-ss` and the transcript are
relative to the start. Read `format=start_time` once and convert. Write a
test for it (`ffmpeg ... -output_ts_offset 5` makes such a file).

Run the frames in parallel with a small `ThreadPoolExecutor`, as `look.py`
does. Keep the ffmpeg and ffprobe calls behind a small class like `Footage`
so tests can fake them.

### Code changes

1. `TOOLS.md`, as above.
2. New module `server/lumr_studio/frames.py`: the pure part (validate,
   choose the frame time, crop box, scale, encode) and a `ScreenFootage`
   class for the two ffmpeg/ffprobe calls.
3. `project.py`: `frames_dir`.
4. `tools.py`: `def frames(video_path, times, region=None, *, footage=None)`.
5. `server.py`: the `frames` tool, title "Screenshots from the video",
   `read_only=False`. Description, in this repo's voice: what it returns, that
   times are source seconds, that it shows the frame on screen at that moment,
   at most 6 times, and that `region` zooms in. Content order: label text,
   image, label text, image, ..., then the JSON block. `mime_type` matches
   the file (`image/png` or `image/jpeg`).
6. Add one sentence to `_INSTRUCTIONS_START` in `server.py`: "frames shows
   the source video at any times you pass, for screen recordings with spoken
   feedback."
7. `offering.py`: add `"frames"` to `EDITOR_TOOLS`.

### Tests (`server/tests/test_frames.py`, plus count updates)

Use `synthetic_master`/`video` from `conftest.py` for plain cases, and build
a small VFR file in a fixture for the on-screen case (Pillow stills of
different solid colours, then the ffmpeg concat demuxer with `duration`
lines and `-fps_mode vfr`, the same way `plans/make_feedback_recording.py`
does it).

- On the VFR file, each asked time returns the colour that is on screen
  then: just before a change, exactly on a change, and in the middle of a
  long hold.
- `shown <= at` always, and `shown` matches the frame's packet time.
- Non-zero `start_time` file: the same times give the same colours.
- A 2880x1800 source comes back with long edge 1568 and the right aspect.
- `region=[0.5, 0.5, 1, 1]` returns the bottom-right quarter (check a pixel
  colour you drew there). A small source crop is not upscaled.
- PNG under 1 MB. A noisy frame (random pixels) falls back to JPEG.
- Errors: 0 times, 7 times, a negative time, a time past the end, a bad
  region, a relative path. Each message says how to fix it. No traceback.
- Files land in `frames/`, never beside the video, and two calls at the same
  time make two files.
- Count updates: `test_offering.py` (11 → 12, 15 → 16, and
  `test_the_skill_tests_read_the_skills_they_mean_to` now expects
  `["tight-cut", "ui-feedback"]` and `["tight-cut", "ui-feedback",
  "add-visuals", "publish-kit"]`). Check the actual order the test builds:
  default skills sorted, then extras.
- `test_mcp_smoke.py`: add `"frames": False` to `EDITOR_TOOLS`, 11 → 12,
  15 → 16. Add a smoke call to `frames` with one time and check that the
  result has a text block, then an `image` block, then the JSON block.

## Part 2: the `ui-feedback` skill

Write `skills/ui-feedback/SKILL.md`. Front matter:

```
---
name: ui-feedback
description: Use when someone has a screen recording of an app or website with their spoken feedback and wants it turned into a list of UI notes, each tied to a time in the video and a screenshot of what was on screen.
---
```

`test_offering.py` checks that a skill only names tools that are offered
(names in backticks). Everything this skill names is in the default list.

What the skill must say, step by step:

**Ground rules.**
- Lumr Studio tools for the video. Ordinary file and code tools are fine for
  the notes file and for fixes, but only after she says yes.
- Never delete a file.
- To wait on a job, `job_status` with `wait: 50`.
- Privacy, once, up front: each screenshot Claude takes is sent in the
  conversation like the transcript is. A screen recording can show emails,
  names or keys. If something on screen is private, she can say which parts
  to skip.

**0. Models up front.** The same as `tight-cut` step 0, shorter: Parakeet
writes the transcript, wav2vec 2.0 times the words, 2.85 GB once, nothing
downloads without a yes. Skip it if she's used Lumr Studio on this Mac. For
feedback the word timing model is nice to have, not needed. If she says no to
it, go on with estimated times; they're close enough to pick screenshots.

**1. Get the recording.** Ask for the path. Tell her, if she hasn't recorded
yet: Cmd-Shift-5, Options, choose a microphone. A recording with no voice has
nothing to transcribe. Call `transcribe`. Handle `needs_models`,
`downloading`, `job_id`, `aligning` and `exists` as `tight-cut` does. Ignore
everything about paces: nothing is cut here.

**2. Read everything first.** `read_transcript` from the start, following
`NEXT` until `END`. No notes until the whole thing is read: she often says
"actually, ignore what I said about the header" later.

**3. Split it into notes.** A note is one thing she wants changed, a bug, a
question, or something she likes. Rules:
- One sentence can hold two notes. Several sentences can be one note.
- Lines like "okay, now I'm scrolling down to pricing" are not notes. They
  say where she is. Use them to name the screen.
- Keep her words. Quote briefly. Never make the ask stronger or weaker than
  she said it.
- When she takes something back later, drop the note or mark it changed.
- Each note gets a start and end time from the transcript lines.

**4. Take screenshots.** For each note, call `frames`:
- at the start of the note;
- at each pointing word ("this", "here", "that", "these", "this button").
  Use `find_words` with those words to get their exact times; the cursor is
  usually on the thing as she says the word;
- if she's talking about something she just scrolled past ("that last
  section"), also about 1 to 2 seconds before the note starts;
- if the screen changes during a long note, at the middle and end too.

Up to 6 times a call, so batch neighbouring notes. Use `region` to zoom in
when the thing is small: small text, an icon, a form field. Don't take
screenshots of stretches she said to skip.

**5. Look and name it.** For each note, say what's on screen and which
element she means: its visible text, where it sits, and where the cursor is.
If it isn't clear which element she means, mark the note "unclear" and ask.
Don't guess.

**6. Show the notes in chat.** Numbered. Each one:
- the time, as the `clock` from `frames` (never formatted by hand);
- her words, quoted briefly;
- what's on screen and which element;
- the ask, in one plain line;
- the kind: change, bug, question, or keep (praise: "she likes this, don't
  change it");
- the screenshot paths.

Then: the unclear notes as questions, and one line offering to save the
notes as a file and to start on the fixes.

**7. Save the notes, only on a yes.** Write `ui-feedback/<video stem>.md` in
the project she's working in (or where she says). Each note as a section,
with screenshots linked by their paths. Never overwrite: add `-2`, `-3` if it
exists.

**8. Fix them, only on a yes.** Work in the app's code one note at a time.
Find the component from the screenshot (search the code for the visible text
first). Say which note each change is for. Skip "keep" notes, and don't touch
those parts. If she can run the app, offer to compare against the screenshot
afterwards.

**Handling tool errors.** Same as `tight-cut`: read the message, fix the
call, retry once, then tell her plainly.

## Part 3: docs and version

- `README.md`: a short "UI feedback sessions" section under "What it can do"
  with one "Things to try" line, such as `> Use ui-feedback on
  ~/Movies/app-walkthrough.mov`. In "What it runs and fetches", what leaves
  the Mac now includes the screenshots Claude takes with `frames`.
- `GUIDE.md`: line 7 (11 tools, one skill) → 12 tools, two skills. Lines 129
  and 151 say the `look` picture is the only image that goes to Claude. Add
  the `frames` screenshots.
- `PRIVACY.md`: "What Claude sees". Add the screenshots, and remove "That's
  the only time an image from your video goes to Claude". Add `frames/` to
  the project files row. Move "Last updated" to the release date.
- `.claude-plugin/plugin.json`: version `0.1.5` → `0.1.6`. Add `"screen-recording"`
  and `"ui-feedback"` to keywords. Mention feedback in the description in
  one short clause.
- Keep `plans/` out of the release: delete this folder in the last commit,
  or ask Ashley if she wants it kept.

## Part 4: testing on the Mac

### A. Unit and contract tests

```sh
cd server
uv run --locked --group dev pytest -q
```

Everything must pass. Run it once **before** you change anything, so you
know the starting point. (On Linux, without the Mac-only lock, 1365 pass and
24 fail for platform reasons, in `test_doctor.py`, `test_mcp_smoke.py` and
`test_transcribe_models.py`, and `test_aligner.py` and `test_models.py` can't
load without torch. On the Mac they should all pass.)

### B. The server over MCP

`test_mcp_smoke.py` starts the real server through `hooks/start-server.sh`.
It must list 12 tools by default and 16 with
`LUMR_STUDIO_EXTRAS=publish_kit,overlays`, and the `frames` call must return
an image block.

### C. A recording with known answers

`plans/make_feedback_recording.py` builds a 40 second fake screen recording
of a made-up app ("Acme": home, pricing, sign up, settings) at 2880x1800,
variable frame rate like a real one, with macOS `say` speaking 8 lines of
feedback at known seconds. It writes `answers.json` with each line's start
time and the screen that's up then.

```sh
python3 -m venv /tmp/fb && /tmp/fb/bin/pip install pillow
/tmp/fb/bin/python plans/make_feedback_recording.py ~/Movies/lumr-feedback-test
```

(`--tones` swaps the voice for beeps, to check the build without `say`. It
was checked that way on Linux. The voice path wasn't, because there's no
`say` there.)

Then, in Python against `tools.py` directly (not through Claude):

1. `transcribe` the file and wait on the job. The transcript should hold all
   8 lines, words close to the script.
2. Each line's first word in `read_transcript` starts within 0.5 s of its
   `answers.json` start.
3. `frames` at each line's start (two calls: 6 then 2) returns the screen
   `answers.json` names. Check by pixel: each screen has a distinct
   layout, or add a coloured corner marker per screen to the script if you
   want a one-pixel check.
4. `frames` at 19.6 s with `region=[0.3, 0.6, 0.6, 0.85]` shows the middle
   pricing card's "Buy" button large.

### D. The skill end to end, in Claude Code

Load the plugin from your checkout, in a fresh session:

```sh
claude --plugin-dir /path/to/lumr-studio
```

(If your Claude Code has no `--plugin-dir`, use `claude plugin marketplace
add /path/to/lumr-studio` and install `lumr-studio@lumr-studio`.) Run
`/mcp` and check that `lumr-studio` is connected and lists `frames`.

Prompt:

> Use ui-feedback on ~/Movies/lumr-feedback-test/feedback-demo.mp4

It passes when:

- Claude says the privacy line and the models line before it starts (models
  only if not downloaded yet).
- Exactly 5 change notes: the grey subheading, the middle pricing card,
  the Buy button, the email field, the show password toggle. "The headline is
  great" comes back as a keep note. "These three cards are fine" and
  "Settings looks good" are keep notes or left out, never change notes.
- Each note's time is within 1 s of `answers.json`, shown as `m:ss`.
- Each note names the right screen and element: the light grey subheading,
  the middle pricing card, the Buy button (all three, or asks which), the
  email field, the password field.
- The "Buy" note quotes her ask ("Start free trial") exactly.
- Screenshot paths exist under the project's `frames/` folder.
- No file is written outside `frames/` until you say yes to saving the notes.
  Saying yes writes `ui-feedback/feedback-demo.md` with working image links.

### E. A real recording

Record 1 to 2 minutes of a real site with Cmd-Shift-5 (Options, choose the
microphone), scrolling while talking, with at least one "this one here"
while pointing with the cursor. Run D's prompt on it. Check by eye that
every screenshot shows what you were talking about, especially right after a
scroll. Note anything wrong in the PR.

## Done means

- [ ] `TOOLS.md` updated first, then code.
- [ ] `frames` built, on by default, returns labelled images plus JSON.
- [ ] VFR on-screen frame test and `start_time` test pass.
- [ ] `skills/ui-feedback/SKILL.md` written.
- [ ] README, GUIDE, PRIVACY, plugin version and keywords updated.
- [ ] Full suite passes on the Mac (A, B).
- [ ] C and D pass. E done once by Ashley, or listed as to do in the PR.
- [ ] `plans/` removed, or kept on purpose.
- [ ] Committed and pushed, PR open with what was tested and what wasn't.
