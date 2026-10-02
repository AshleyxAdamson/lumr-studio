# Lumr Studio guide

The details behind the [README](README.md): the review page, what you need, the models, what runs and what's saved. `TOOLS.md` is the contract for each tool.

## The review page and tools

The plugin gives Claude 12 tools and one skill, `tight-cut`. `TOOLS.md` is the contract for each tool. None of them deletes a file of yours, and none uploads your video.

- `tight-cut` reads a whole recording, proposes a cut list with plain-language reasons, and only saves or renders once you say go.
- Every saved edit comes back with a read of each join, the way a viewer hears it, and a fix for the ones likely to sound wrong. Claude repairs those before showing you the plan.
- Likely laughs are marked in the transcript, and Claude keeps them in its own cuts. The automatic pause trims can still clip a laugh at the harder paces. On the test take, 10 places at Standard and 23 at Max. The join check flags those, and Claude tells you which stretch to keep on the page.
- Six pace stops, from Natural to Max, set how much of each pause comes out. Each leaves a pause at every join, the most between sentences. Fine tune holds two sliders for a setting between the stops.
- Claude finds every place you say "like" (`find_words`), reads each one, and picks the filler ones. "I like jazz" stays. "Into like three sections" goes. When Claude is unsure, the word stays. You get one switch for all of Claude's picks and a double-click for each.
- The review page is one page to shape the edit after Claude makes it. It plays your video from disk, and every change saves at once.

| On the page | What it does |
|---|---|
| Pace | Six stops: Natural, Standard, Fast, Tight, Hard, Max, with the time each saves. Hard and Max say what you give up. |
| Fine tune | Opens under the pace. Two sliders, "Cut pauses longer than" and "Speech kept between cuts". Both cut harder to the right. The pace then reads Custom. Pressing a stop leaves Custom. |
| Take out | A switch each for long pauses, filler words, stutters and filler likes, with a count. Filler words names the words it takes at this pace, such as "like, so, and". "Make this my usual" saves your pace and switches as the start for your next video. |
| Filler likes | The likes Claude picked, all on or off. Until Claude has picked, it reads "Ask Claude to find them." |
| Cuts | Your own cuts first, then Claude's in four groups, one row per cut. Hear it cut, leave it out, put it back, or take yours back. One count shows how many need a look, and each of those says why. |
| Words | Double-click a word to cut it. Double-click a struck word to bring it back. Drag across words to pick a part, then press Cut or Keep. Cut takes exactly the words you picked. Keep means nothing is taken out there after that, and "Kept: 1" opens the list. |
| Undo | Shows in the top line after you cut, keep, or bring back words. The Z key does the same. It takes back your last cut or keep, one step. It doesn't cover the pace, the sliders, or the switches. |
| Play, Previous, Next | Centered under the video. Previous and Next go from cut to cut. The video and the words move together. |
| Original and Edited | Switch between your recording and the edit. |
| The bar | The whole video along the bottom. Each amber block is a stretch where lots was cut close together; hover one and it says "More cuts here" and how many edits it holds. The shade shows the same: a few, many, or the most. Press the play button on a block to hear that stretch. Click or drag anywhere else on the bar to move. |
| Export video | The one filled button, upper right. It shows a percent while it runs, then the file's name with "Copy path". |

Struck words say who took them out: dim for the pace, red for Claude, teal for you. When you're done, tell Claude, and it reads what you changed.

## What you need

- A Mac with Apple Silicon (M1 or newer). Intel Macs aren't supported, and neither are Windows or Linux. The plugin's package list is locked to Apple Silicon, so it won't install anywhere else.
- Claude Code 2.1.78 or newer. That release added the plugin data folder, `${CLAUDE_PLUGIN_DATA}`, where the plugin builds its Python environment. Without it the server fails to start, or builds the environment in the wrong place.
- `uv` and `ffmpeg`, version 5 or newer. Homebrew installs both. Run `brew install uv ffmpeg`.
- About 4 GB of free disk. The Python packages take about 950 MB. The two models take 2.85 GB.
- The two models: Parakeet (2.47 GB) for the transcript and wav2vec 2.0 (0.38 GB) for word timing. They download the first time you transcribe a video, and only after you say yes. The plugin works without the word timing model and cuts less. [The models](#the-models) has the sources and licenses.

The first session builds the plugin's own Python environment from a locked package list. That's about 250 to 300 MB of downloads and a 950 MB environment. It takes a minute or two on a good connection and longer on a slow one. The build runs in the background, apart from the server, because Claude Code stops a server that hasn't started after 30 seconds, and a stopped server would take the build with it. So the first session shows lumr-studio as failed in `/mcp`, and Claude tells you the build is running. When it finishes, open `/mcp` and choose Reconnect on lumr-studio, or start a new session. Later starts take seconds. If the Mac has no Python from 3.11 to 3.13, `uv` fetches one. If the build fails, say on a dropped connection, the next session starts it again, and `build.log` in the plugin's data folder (under `~/.claude/plugins/data/`) says why.

To have the server up in the first session, start Claude Code once with `MCP_TIMEOUT=300000 claude`. That gives the server five minutes to start, and `MCP_TIMEOUT` is on [Claude Code's MCP page](https://code.claude.com/docs/en/mcp).

## First-run check

When a Claude Code session starts, a short script (`hooks/doctor.sh`) checks four things. It looks for an Apple Silicon Mac, for `uv` on the PATH Claude Code sees, for `ffmpeg` and `ffprobe`, and for ffmpeg 5 or newer with the libx264 encoder.

When all is well it prints nothing. When something's missing, Claude tells you what and gives the fix, such as `brew install ffmpeg`. The script never installs system software. `uv` and `ffmpeg` stay commands you run.

It does one thing besides look. When `uv` is there and the plugin's Python environment isn't built, it starts that build in the background from the plugin's own locked package list (`hooks/build-env.sh`), the same build the server's launch would do. It prints one line saying the build is running and that a failed server is expected until it's done. The environment, `build.log` and a short-lived `build.lock` folder in the plugin's data folder are all it writes. One build runs at a time, and a later session picks up a running build instead of starting another. If the last build failed, the line says so and points to `build.log`. Without the hook a missing `uv` would go unnoticed, because with no `uv` the server never starts and no tool could say so.

## Measured word times

The pace cuts the pauses between your words, so the plugin has to know where each word starts and ends. A transcript's own word times are estimates. Most words touch the next one, so the quiet between them is hidden inside the words, and the plugin never cuts inside a word.

Measuring pins each word to the sound. It needs the wav2vec 2.0 model. It takes under a minute for a 20 minute video (22 seconds on the 21:47 test take), once per video, inside the same wait as transcribing. With the model missing or incomplete, the plugin uses estimated times and tells you so, on the page and in chat.

What you lose without it, measured on the test take (21:47, the word "like" said 112 times):

| | Measured word times | Estimated word times |
|---|---|---|
| Natural takes out | 1:48 | 0:46 |
| Standard takes out | 3:26 | 1:25 |
| Fast takes out | 5:38 | 1:55 |
| Tight takes out | 6:30 | 2:02 |
| Hard takes out | 7:53 | 2:20 |
| Max takes out | 8:14 | 2:21 |
| Places where "like" can be cut without clipping the word beside it | 94 of 112 | 28 of 112 |

On estimated times the four harder stops sit within 26 seconds of each other, so Fast, Tight, Hard, and Max sound nearly alike. Cutting a single word by hand still works. It lands less exactly.

## The models

Lumr Studio needs two models. Neither is in the plugin. The first time you transcribe a video, Claude tells you what would download, how big it is, where it comes from and its license, and asks. Nothing downloads until you say yes.

| Model | What it does | Size | Comes from | License |
|---|---|---|---|---|
| NVIDIA Parakeet TDT 0.6B v2 | Speech recognition. Writes the transcript. | 2.47 GB | `https://huggingface.co`, `mlx-community/parakeet-tdt-0.6b-v2`, pinned to commit `8ae155301e23d820d82aa60d24817c900e69e487` | CC-BY-4.0 |
| wav2vec 2.0 base 960h | Word timing. Pins each word to the sound. | 0.38 GB | `https://download.pytorch.org/torchaudio/models/`, torchaudio's own file | MIT |

The plugin fetches them itself, in a background job that shows its progress. Some things to know.

- Every file is checked against a size and a sha256 pinned in the plugin. A download goes to a `.part` file and is renamed only when the check passes. A file that fails the check isn't installed, and Claude tells you. If the connection drops, what arrived is kept and the next try picks up where it stopped.
- Hugging Face sends its large files through its own download hosts, so your Mac may also talk to a `hf.co` host such as `us.aws.cdn.hf.co`. The name can change. Only the model comes from there, and nothing of yours goes out.
- The files land in the caches other tools share: `~/.cache/huggingface/hub` (or `HF_HOME`) and `~/.cache/torch/hub/checkpoints` (or `TORCH_HOME`). They stay after you uninstall the plugin. If the pinned Parakeet snapshot is already in the Hugging Face cache, the plugin uses it and downloads nothing.
- The plugin fetches models in one place only, after a yes. Measuring word times refuses to download anything, and the speech model runs with Hugging Face set offline.
- Say no to the word timing model and the plugin still works, on estimated times. The next section says what that costs.

Credit for both models is in `NOTICE`, and below.

## Turning the extras on

Two extras are built and tested, and they ship switched off. One is the publish kit (the `publish-kit` skill, with the tools `chapter_times` and `save_publish_kit`), which writes title options and a description with chapters, plus tags, from the video's own transcript. The other is overlays (the `add-visuals` skill, with `set_overlays` and `get_overlays`), which shows your own photos and clips over the talk.

The overlay code stays linked and dormant. With no saved overlays, nothing is drawn over your video. The publish kit's skill and tools don't load at all.

To turn them on, edit two files inside the plugin's folder.

1. In `.mcp.json`, add `"LUMR_STUDIO_EXTRAS": "publish_kit,overlays"` to `env`. Name one or both. A name that isn't an extra stops the server at start.
2. In `.claude-plugin/plugin.json`, add `"skills": ["./extras/skills/publish-kit/", "./extras/skills/add-visuals/"]`. The key adds to the default `skills/` folder.

Then restart Claude Code, or run `/reload-plugins`. A plugin update replaces those files, so you'd make the edit again.

## What runs, what it reaches, and what Claude sees

Everything runs on your Mac. The one exception is Claude, which runs wherever your Claude Code sends its requests.

| What | How it runs |
|---|---|
| The server | Claude Code starts it over stdio with `uv run --active --locked --project ${CLAUDE_PLUGIN_ROOT}/server lumr-studio-server` (from `.mcp.json`). It's the only long-running process. |
| The speech model | `python -m lumr_studio.speech <video>`, in a process of its own, once per video, reading the downloaded model from disk with Hugging Face set offline. |
| Cutting and rendering | `ffmpeg` and `ffprobe`, from your PATH. |
| The review page | A small web server on `127.0.0.1`, on a port the operating system picks, with a private token in the address. The plugin opens it in your default browser. It makes no outside requests. The Send feedback button sends a message only when you press Submit, and Help improve Lumr sends the shape of your changes only when you press Send. Your Mac's local server sends both. |
| The first-run check | `sh "${CLAUDE_PLUGIN_ROOT}/hooks/doctor.sh"`, at the start of each session. It reads your PATH and prints. |

The plugin reaches the network on first use only. It fetches these. Nothing you own goes out in any of them.

- `https://pypi.org` and `https://files.pythonhosted.org`, for the locked Python packages, through `uv`.
- `https://github.com/astral-sh/python-build-standalone`, for a Python build, only if the Mac has none from 3.11 to 3.13. This is `uv`'s default.
- `https://huggingface.co`, and the Hugging Face download hosts it redirects to (`us.aws.cdn.hf.co` when checked), for the speech model `mlx-community/parakeet-tdt-0.6b-v2` at one pinned commit. Only after you say yes. The plugin sets `HF_HUB_DISABLE_TELEMETRY=1`, so Hugging Face's usage telemetry stays off.
- `https://download.pytorch.org/torchaudio/models/`, for the word timing model, `wav2vec2_fairseq_base_ls960_asr_ls960.pth`. Only after you say yes.

The first-run check may print `https://brew.sh` in a message, when Homebrew is missing. It doesn't visit it.

What Claude sees is the conversation. The plugin adds to it in three ways.

- The transcript text, in short packed lines, as Claude reads it.
- Cut lists with their reasons, and the results of each tool.
- One small picture when Claude checks a join with `look`: a few still frames, the sound wave drawn as a line, and the words around the cut. That's the only time an image from your video goes to Claude.

The plugin never uploads your video or audio file.

## Helping improve Lumr

The review page shows a small Help improve Lumr button. It opens a list of what would be sent: how long each cut was, what kind it was, and what you did to it, such as putting it back or marking it wrong. It never sends your words or your video. Remove any line you'd rather keep, press "Show exactly what's sent" to see the data itself, and press Send only when you're happy. Nothing goes before that. After a send, you get an ID to quote if you ever want it deleted, and `shares.jsonl` in the video's project folder keeps it. `PRIVACY.md` has the detail.

## What is saved, and where

Each video gets one project folder at `~/Lumr/studio/projects/<video name>-<id>/`. The `~/Lumr` part follows `LUMR_HOME`, or the folder named in `~/.config/lumr/home.txt` if that file exists. `TOOLS.md` lists every file in the project folder. The main ones:

| File | What it holds |
|---|---|
| `edit.json` | your saved edit, with the pace and switches it was built with |
| `words-aligned.json` | your transcript's words with times measured from the sound. Made once per video. If the transcript changes, it's made again. |
| `plans.json` | the automatic removals planned for each of the six stops and for the last 12 slider settings you tried, so the page answers fast |
| `exports/` | the videos and previews you render |

Lumr learns from the cuts you put back, the cuts you mark "Good cut" or "Wrong cut" on the review page, and the words you cut or bring back. It works that out from your saved edits, on your Mac, and keeps only a note in `taste.json` of what you asked it to forget. To forget something, ask Claude: one word, one kind of cut, or everything. Your videos and edits don't change.

The transcript itself stays beside your video, as `<video>.words.json`. It's the only file the plugin puts in your video's folder. Measured silences are cached under `~/Lumr/media_cache/`. The Python environment lives in the plugin's data folder, and Claude Code deletes it when you uninstall the plugin. The models stay in the shared caches (`~/.cache/huggingface` and `~/.cache/torch`) and survive an uninstall.

## Privacy

Lumr Studio collects nothing. It has no server, no accounts and no telemetry, and the author never receives your files, your transcript or a usage count. The two exceptions are feedback you write in the review page's Send feedback form and submit, and the shape of your changes if you press Send in Help improve Lumr. `PRIVACY.md` says what each sends.

- On your Mac. Your video, audio, transcript, word times, edits and exports. The plugin reads a video only when you name it to Claude, and only on your own Mac. It never browses your folders, and it never uploads your video or audio.
- What Claude sees. The transcript text, cut lists with reasons, and the results of each tool. When Claude checks a join with `look`, a few still frames go with it. That goes wherever your Claude Code sends its requests.
- The network. Downloads only, and nothing of yours goes out in them. `huggingface.co` and `download.pytorch.org` for the two models, only after you say yes. PyPI (`pypi.org` and `files.pythonhosted.org`) through `uv` for the plugin's Python environment. `uv` also fetches a Python build from `github.com/astral-sh/python-build-standalone` if your Mac has none, and Hugging Face sends its large files through its own download hosts. Hugging Face's telemetry is switched off.

The [privacy policy](https://github.com/AshleyxAdamson/lumr-studio/blob/main/PRIVACY.md) has the rest: what's stored where, how long it stays, and how to delete it.

## Support

Bugs, questions and ideas go to [GitHub Issues](https://github.com/AshleyxAdamson/lumr-studio/issues). Say what you did, what you expected and what happened, and give the plugin version and your macOS version. If the first environment build failed, `build.log` in the plugin's data folder says why. Please don't attach a video you want to keep private.

## Security

Report a vulnerability privately through [GitHub's private vulnerability reporting](https://github.com/AshleyxAdamson/lumr-studio/security/advisories/new), not in a public issue. [SECURITY.md](https://github.com/AshleyxAdamson/lumr-studio/blob/main/SECURITY.md) says what's in scope.

## Credits

- Speech recognition: [NVIDIA Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), CC-BY-4.0. Lumr Studio downloads the MLX conversion by the mlx-community.
- Word timing: wav2vec 2.0 base 960h via torchaudio, MIT. By the wav2vec 2.0 authors at Meta AI, from [fairseq](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec) and redistributed by [torchaudio](https://docs.pytorch.org/audio/stable/generated/torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.html).

`NOTICE` holds the license links and the MIT text.

## License

Lumr Studio is under the PolyForm Noncommercial License 1.0.0. The full text is in `LICENSE`.

You can sell the videos you make with it. You can't resell the tool.

The files in `server/lumr_studio/engine/` and `server/lumr_studio/speech/` are copies of files from two other projects. The copied ClipForge and Hammy files are by the same author and included here under this plugin's license. The originals keep their own terms where they live.
