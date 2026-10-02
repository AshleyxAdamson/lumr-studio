# Lumr Studio MCP server

A local MCP server for the Lumr Studio plugin. The creator's own Claude makes
the editing decisions; this server runs the machines (the speech model, ffmpeg,
the pacing and render engine) and validates what Claude proposes. The tool
contract is `../TOOLS.md`.

MCP SDK: `mcp==2.2.0`, high-level class `MCPServer` (`mcp.server.mcpserver`),
stdio transport.

## Running

The server has its own locked environment: `pyproject.toml` and `uv.lock`,
about 950 MB installed, Apple Silicon only (the lock refuses other platforms).
Nothing here needs a checkout of the repo, ClipForge or the `hammy` command.
The plugin starts it through `../.mcp.json`, which runs a small script and hands
it the plugin folder and the plugin data folder:

```sh
sh ${CLAUDE_PLUGIN_ROOT}/hooks/start-server.sh ${CLAUDE_PLUGIN_ROOT} ${CLAUDE_PLUGIN_DATA}
```

`../hooks/start-server.sh` sets `VIRTUAL_ENV=<data folder>/venv`, then runs:

```sh
uv run --active --locked --project <plugin folder>/server lumr-studio-server
```

`--active` tells uv to use that `VIRTUAL_ENV`, so
the environment is built once, outside the plugin's versioned folder, and
survives an update (an update re-points the project at the new folder and
installs nothing). `.mcp.json` has no `env` block. The script takes the two
folders as arguments because the server's process may not get
`CLAUDE_PLUGIN_ROOT` or `CLAUDE_PLUGIN_DATA` as environment variables. The first
session builds the environment. The plugin's SessionStart hook
(`../hooks/doctor.sh`) runs `uv sync --active --locked` for this folder in the
background, detached, because Claude Code stops a server that hasn't started
after 30 seconds (`MCP_TIMEOUT`) and a cold build takes longer. A build inside
the server would die with it. When the build has finished, the server's own
launch finds the environment ready and answers in seconds: Reconnect it in
`/mcp`, or start a new session. If the launch runs while the build is still
going, `uv` makes it wait for the environment's lock (`venv/.lock`) rather than
build a second copy. To run by hand, from this folder:

```sh
uv run --locked lumr-studio-server
```

That builds `.venv` here instead.

### Where the engine comes from

`lumr_studio/engine/` (the cut, render and pacing code) and `lumr_studio/speech/`
(Parakeet transcription) hold generated copies of 6 ClipForge files and 2 Hammy
files. Each says on its first line that it is generated, and from which file.
Don't edit them. Change the source, then run:

```sh
studio/tools/sync-engine.sh          # write the copies
studio/tools/sync-engine.sh --check  # exit 1 when a copy differs from its source
```

The script sits in the author's development repo, at `studio/tools/`, and the
published plugin repo leaves it out. `tests/test_engine_copy.py` runs the check,
so the suite fails until the copies are written again. `engine/paths.py` and `speech/__main__.py` are the
hand-written files in those folders.

Environment:

- `LUMR_STUDIO_PROJECTS_DIR` overrides the projects root. Otherwise it is
  `<LUMR_HOME>/studio/projects`, with `LUMR_HOME` resolved the way ClipForge
  resolves it (`engine/paths.py`), at call time.
- `LUMR_STUDIO_EXTRAS` is a comma list of extras to switch on, `publish_kit`
  and `overlays`. Empty by default: the server lists 11 tools. With both it
  lists 15. An unknown name stops the server at start. Their skills come back
  with an entry in `plugin.json` (`../GUIDE.md`, "Turning the extras on").
- `ffmpeg` and `ffprobe` must be on `PATH`. There is no `hammy` command.
- Transcribing runs `python -m lumr_studio.speech` in a process of its own,
  with `HF_HUB_OFFLINE=1`. It reads Parakeet (`mlx-community/parakeet-tdt-0.6b-v2`,
  2.47 GB) from the pinned snapshot folder in the Hugging Face cache
  (`models.SPEECH`), so nothing is looked up online. The snapshot is at
  `~/.cache/huggingface/hub/models--mlx-community--parakeet-tdt-0.6b-v2/snapshots/<commit>/`,
  or under `HF_HOME` or `HF_HUB_CACHE`.
- Measured word times need torch and torchaudio (both in the locked
  environment) and torchaudio's `WAV2VEC2_ASR_BASE_960H` file, 0.38 GB (MIT), at
  `<torch hub>/checkpoints/wav2vec2_fairseq_base_ls960_asr_ls960.pth`
  (`TORCH_HOME` moves it). MMS_FA was the aligner before; its weights are
  non-commercial and the plugin never loads them (the spike is `docs/11.03` in the author's repo).
  The generated `engine/alignment.py` still holds ClipForge's MMS_FA loader, unused by the
  plugin: a test (`test_no_plugin_code_reaches_mms_fa`) proves nothing calls it.
- Neither model is in the package. `transcribe` answers `needs_models` and
  downloads nothing. With `download_models` true, a `download_models` job
  fetches what is missing (`models.download`): each file pinned by exact size
  and sha256, written to `.part`, checked, then renamed, resumed from its last
  byte after a stop. That is the only code that fetches a model. Aligning swaps
  out torch's downloader and refuses. `TOOLS.md`, "Models", is the contract.
  Without the aligner file every tool answers `word_times: "estimated"` with a
  note that says why. The plugin's `GUIDE.md` has what that costs, in numbers.
- To change a model, edit its pin in `lumr_studio/models.py`: revision or
  file name, size and sha256. Compute the sha256 from the file itself
  (`shasum -a 256`). `tests/test_models.py` holds the aligner's file name and
  URL to what torchaudio asks for.

## Tests

```sh
cd server
uv run --locked --group dev pytest -q
```

`uv run --locked` builds `.venv` here on the first run. pytest is in the `dev`
group, and `default-groups = []` in `pyproject.toml` keeps it out of the plugin's
own environment, so the test command has to ask for it with `--group dev`. Tests use temp folders for `LUMR_STUDIO_PROJECTS_DIR` and `LUMR_HOME`,
generate a 20 second synthetic video with ffmpeg, and patch out the speech
process, the cached-silence helper and the aligner so they can never run. No
test loads a model or reaches the network: a test that aligns passes a fake
aligner or a fake model (`tests/test_aligner.py`), and a test that downloads
passes a fake opener and tiny models (`tests/tiny_models.py`). The model
caches (`HF_HOME`, `TORCH_HOME`) are temp folders too. The MCP smoke
test starts the server with the command in `.mcp.json`, through
`hooks/start-server.sh`, with both plugin folders filled in. The data folder is a
temp folder whose `venv` is a link to `.venv`, so the server still runs in `.venv`.

The drift and closure tests (`tests/test_engine_copy.py`) run
`sync-engine.sh --check` only where the repo's `clipforge/` and `hammy/` sit
beside the plugin. A copy of the plugin alone skips that one and runs the rest.

`tests/test_doctor.py` runs the first-run check (`../hooks/doctor.sh`, started
by `../hooks/hooks.json` at session start) against stub programs on a trimmed
`PATH`, so it doesn't depend on this machine's ffmpeg or uv.

## Module map

| Module | Job |
|---|---|
| `server.py` | MCP wiring only: tool names, annotations, input models, error mapping. Registers a tool only when `offering.py` offers it |
| `offering.py` | The one list of tool names: the 11 editor tools, and the extras (publish kit, overlays) behind `LUMR_STUDIO_EXTRAS` |
| `tools.py` | One plain function per tool; what the tests call |
| `project.py` | Project folder, atomic writes, receipts, new export names |
| `transcript.py` | Load and validate `words.json`; pack it into phrase lines |
| `edit.py` | Place, validate, merge and save cuts: Claude's, the creator's own, and Claude's picked filler words |
| `word_times.py` | The one place that chooses the word times: measured when the words were aligned to the sound, else estimated |
| `aligner.py` | The production aligner: wav2vec 2.0 base 960h, CTC forced alignment in 40 s windows, with the tail pad and start shift named for why |
| `models.py` | The two models' pins (host, license, size, sha256) and the one downloader: `.part`, check, rename, resume |
| `word_finder.py` | Every place a word is said, packed for Claude to pick the fillers (`find_words`) |
| `autocuts.py` | Automatic pause/filler micro-cuts via `microcut_pacing`, filed by what each takes, and kept in `plans.json` |
| `analysis.py` | The `analyze_take` report |
| `retakes.py` | Repeated-phrase (retake) candidates |
| `silences.py` | Where measured silences come from (injectable) |
| `timeline.py` | Source time to edited time, and back |
| `jobs.py` | Background jobs, status, receipts |
| `transcription.py` | The transcribe job: the speech process, then a check of the transcript. The same job measures the word times. |
| `render_jobs.py` | The render and preview jobs |
| `publish_kit.py` | Chapter rules and the `publish-kit/` files. An extra: its tools (`chapter_times`, `save_publish_kit`) list only with `publish_kit` on |
| `joins.py` | The self-check: every join read the way a viewer hears it, with a fix for Claude and a plain why for the creator |
| `sounds.py` | Likely laughs told apart from other sounds |
| `look.py` | One picture of a join, for Claude |
| `pace.py` | The six pace stops, the two sliders, and the pause each setting leaves |
| `treatment.py` | The page's state and every change the page can make |
| `samples.py` | The three samples, the clusters, and the shade rule |
| `review_server.py` | The local web server for the page: routes, token, export |
| `review.py` | The earlier decision queue, still answered, no page calls it |
| `overlays.py`, `overlay_render.py`, `overlay_tools.py`, `overlay_mcp.py` | The creator's photos and clips: checking, drawing, tools, MCP wiring. An extra: `overlay_mcp.register` runs only with `overlays` on, and the rest stays linked and dormant (`render_jobs.py` and `treatment.py` import it, and with no saved overlays it draws nothing) |
| `mcp_results.py` | How every tool answers |
| `errors.py` | `StudioError`: a failure the caller can fix |
| `engine/` | Generated copies of ClipForge's cut, render, silence, filler and alignment code, and `paths.py` (the Lumr home and the silence cache, hand-written) |
| `speech/` | Generated copies of Hammy's transcription code, and `__main__.py`, the entry the transcribe job runs |

## The treatment page

`treatment_page/index.html` is one file with no outside requests. The
creator's footage stays on the creator's machine: the server binds to
127.0.0.1 and the page plays the source video from disk.

| On the page | Where it comes from |
|---|---|
| Pace: six stops, with the time each saves | `paces[]`, `POST api/treatment {pace}` |
| Fine tune: two sliders, the pace then reads Custom | `fine_ranges`, `settings.fine`, `custom`, `POST api/treatment {fine}` |
| Four take-out switches with counts, and "Make this my usual" | `take_out[]`, `usual`, `POST api/treatment {take_out}`, `POST api/usual` |
| Filler words names the words it takes | `take_out[1].words` |
| Filler likes: Claude's picks, all on or off | `take_out[3]`, `settings.take_out.likes` |
| Cuts: the creator's own first, then Claude's in four groups, one needs-a-look count | `groups[]`, `rows[]`, `rows[].by`, `cut_counts`, `need_a_look`, `POST api/cut`, `POST api/cut/remove` |
| Play, Previous and Next between clusters, Original and Edited | `clusters[]`, `removed` |
| Three samples and "New samples" | `samples[]`, `POST api/samples` |
| Double-click a word to cut it or bring it back | `words[]` (the fifth value says who took a word out), `POST api/cut/add`, `POST api/keep {exact: true}`, `POST api/cut/remove` |
| Cut and Keep on picked words, and the "Kept: N" list | `POST api/cut/add`, `keeps[]`, `POST api/keep`, `POST api/keep/remove` |
| Undo, for the creator's cuts and keeps, one step | `can_undo`, `POST api/undo` |
| The line under the pace when the word times are estimated | `word_times`, `word_times_note` |
| The bar: one block per cluster, shaded by its number of edits | `clusters[].level` |
| A name tag on the video for a photo or clip | `overlays[]` |
| Export video, with percent, the file's name and Copy path | `export`, `POST api/export`, `GET api/export` |

The page adds `?muted=1` for tests: every media element is muted at volume 0.
`?example=1` runs it from built-in example data with no server.

The words the creator reads are checked by `tests/creator_words.py`. The word
"trim" is on that list. Text for Claude (`TOOLS.md`, tool descriptions, field
names) may say it. A time in the video is never shown as a decimal. One
thing may be: a length on a Fine tune slider, such as "0.35 s". That
exception is named in `creator_words.py` and tested in
`tests/test_creator_words.py`.

## Project files from round 5

| File | Written by | Holds |
|---|---|---|
| `words-aligned.json` | the transcribe job | the transcript's words with measured times, each with the transcript's own times beside them, the size and date of the transcript they were made from, and what the soft ends of words took (`soft_room`, with a `warning` when the room misled it). A transcript that changed makes the file stale, and the plugin falls back to estimated times until it is measured again. |
| `plans.json` | `autocuts.plan_auto_cuts` | the automatic trims planned for the six stops, and for the last 12 slider settings. A plan depends only on the words, the measured silences and the two slider values, so it is made once. |

## Contract details I had to interpret

- **Cut placement (`set_edit`).** Claude's cuts are content cuts. A word whose
  midpoint lies inside the cut is removed; every other word is kept (the
  editor's `isWordRemoved` rule). Each edge then sits in the gap next to the
  nearest kept word, keeping clipforge's natural tail (up to 0.35 s) and
  lead-in (up to 0.15 s) from `longform.pad_keep_ranges_into_gaps`, and moves
  into measured silence when the gap has some. A cut wholly inside a pause is
  kept as asked. A cut that clips words without removing any whole word is
  rejected, and the message names the nearest word boundaries. Placed cuts go
  through `render.finalize_removed_ranges` as verbatim spans. `auto_tighten`
  cuts go through it as auto spans, with full word safety. The finalized set is
  what `edit.json` stores and what `render` plays.
- **Rejections are per cut.** A bad cut is reported under `rejected` and the
  rest are still saved. But when cuts were given and every one is rejected,
  nothing is saved and the call fails with the list of rejections, so a typo
  can't wipe a good edit. Clearing is explicit: empty `cuts` with
  `auto_tighten` false clears the edit and returns `cleared: true`. A cut
  shorter than 0.1 s once placed is rejected (automatic trims are exempt). An
  end up to 0.05 s past the video's end is clamped.
- **`get_edit` stays short.** It lists Claude's cuts, as
  `{start, end, seconds, reason}`. What the creator did on the page is under
  `creator`, and Claude's picked filler words under `picks`. Automatic trims
  are summarized as
  `auto: {count, seconds_removed}`, alongside `auto_tighten` and `gap_length`.
  `include_auto=true` lists them too, capped at 50, with `not_shown` counting
  the rest. A Claude cut that absorbed automatic trims keeps its own reason
  plus "(+N automatic trims)"; `set_edit`'s `applied` uses the same form.
- **Result encoding.** For a returned dict, mcp 2.2.0 sends `structured_content`
  and also a text block made with `pydantic_core.to_json(indent=2)`. Every tool
  here returns a `CallToolResult` instead: the text block is compact JSON, and
  `structured_content` is unchanged. `read_transcript`'s text block is the
  packed transcript itself, as plain text ending in `NEXT <time>` or `END`,
  with `{text, phrase_count, next_start}` in `structured_content`. Errors still
  go through `ToolError`, so `is_error` works as before.
- **`gap_length` without `auto_tighten`** is an error, so it can't be
  silently ignored.
- **Silences.** `set_edit` and `analyze_take` measure silences with
  `engine.paths.silences_for`, which caches under `LUMR_HOME` with the file
  names ClipForge uses. The first call on a long video waits for ffmpeg.
- **`read_transcript` packing.** Hard line breaks come at pauses of 0.5 s or
  more (with a `(pause N)` marker), at events (their own line), and where the
  cut state flips (so a line is wholly `CUT` or wholly kept). Between those,
  whole sentences (ending `.`, `?` or `!`) are joined into lines of about 6 to
  12 s. Only a sentence longer than 20 s is broken: at the comma nearest its
  middle, or else at its longest gap. The budget is 8000 characters. A word
  counts as cut when its midpoint is inside a removed range. The closing
  `NEXT <time>` / `END` line is now in TOOLS.md; `next_start` is null at `END`.
- **Timing quality.** Some older transcripts have estimated word times, spread
  evenly across each sentence. `transcript.timing_quality` measures the share
  of neighbouring spoken words with identical durations (within 0.002 s) and
  counts overlapping neighbours (more than 0.05 s). A share above 0.5 means
  `approximate`: measured speech sits near 0.04 (the ClipForge fixture:
  0.038), while evenly spread sentences sit near (n-1)/n, 0.8 to 0.9. Then
  `transcribe`, `analyze_take` and `set_edit` add a `warning`. `set_edit`
  still saves.
- **`analyze_take`.** Fillers are clipforge's um/uh family (`detect_fillers`
  level 1). Retakes are the same run of four or more words said again within
  60 s. clipforge's own repeat detector only finds doubled single words and
  filters them for cutting, so it can't report retakes. Retakes are reported
  only, never cut.
- **`preview`.** The window is `pad` seconds of EDITED time either side of
  where `at` lands in the edited video. The encode seeks to the window first,
  so previews near the end of a long video stay quick.
- **`job_status`.** `progress` is `null` for transcription, where there's
  nothing to measure. Jobs live in memory; `receipts.jsonl` keeps a line for
  every finished job, `set_edit`, and `save_publish_kit`.
- **`save_publish_kit`.** The last chapter must also last 10 s before the
  edited video ends. `description.txt` ends with the chapter list, the way
  YouTube reads chapters. `kit.json` holds everything.
- **A saved edit made for a different video length** (more than 0.5 s off) is
  refused, so an edit can't slide against a replaced file.
- **Word times.** The tracer cut on Hammy's raw word timings. Since round 5
  the transcribe job aligns the words to the sound with
  `aligner.Wav2Vec2Aligner`, which reuses ClipForge's decoding and silence clamp from the copy in `engine/`,
  and every tool reads the words through `word_times.read`. On the 21 minute test take the six stops take out 1:48
  to 8:14 on measured times and 0:46 to 2:21 on estimated ones.
