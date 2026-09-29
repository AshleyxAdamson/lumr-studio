# Lumr Studio

**Finally, an editing harness that cuts your speech without dropping you mid-word, and trims the gaps between your words so seamlessly nobody hears the edit.**

Lumr Studio is a Claude Code plugin for talking-head video. Claude reads your whole take, cuts the dead air, the filler and the stumbles, and checks every join before you hear it. It's free and runs on your Mac (Apple Silicon for now). Your video never leaves it.

## Why it's different

Silence cutters (TimeBolt, AutoCut, the silence tools in Premiere and Resolve) cut on a loudness line. They clip soft word endings and miss the small gaps inside sentences. Transcript editors (Descript, Gling, text-based editing in Premiere) cut on word times, and raw speech-to-text times are loose. On our test take they hid 427 seconds of real silence inside words.

Lumr Studio is built around the join, where one kept word lands next to the next.

- **It measures every word.** A second model pins each word to your voice, and the engine never cuts inside a word.
- **It trims gaps instead of deleting them.** Each join keeps a pause sized to where it falls, longest between sentences, with a 20 ms fade so there's no click.
- **It checks every join.** Claude flags clipped laughs, unfinished sentences and gaps that are too tight, and fixes them before you see the plan.
- **It reads before it cuts.** Claude reads the whole take first, keeps your punchlines and their setups, and judges every "like" in context. "I like jazz" stays.
- **You decide.** Every cut has a reason, nothing renders until you say go, and nothing uploads.

On a 21:47 test take, measuring took out more than twice as much at every pace (3:26 vs 1:25 at Standard) and cut 94 of 112 "likes" cleanly, against 28 without it.

## What it can do

- **Six paces**, Natural to Max, plus two sliders for anything in between.
- **Switches** for long pauses, filler words, stutters and filler likes.
- **A review page** in your browser. Double-click any word to cut it or bring it back, hear any cut, flip between the original and the edit, and export.
- **Cuts you can tell apart.** Dim for the pace, red for Claude, teal for you.
- **Extras**, off by default: a publish kit (titles, description, chapters, tags) and overlays of your own photos and clips.

## Start

You need a Mac with Apple Silicon, Claude Code 2.1.78 or newer, and `brew install uv ffmpeg`. Then:

```sh
claude plugin marketplace add AshleyxAdamson/lumr-studio
claude plugin install lumr-studio@lumr-studio
```

Start a session and tell Claude Code:

> Use tight-cut on this video. Give me a standard YouTube cut.

The first session builds the plugin's Python environment (about 950 MB), and Claude asks before it downloads the two models (2.85 GB). [GUIDE.md](GUIDE.md) has the rest: the review page, the models, what runs, what's saved, extras and support. `TOOLS.md` is the tool contract, and [PRIVACY.md](PRIVACY.md) and [SECURITY.md](SECURITY.md) cover privacy and security.

PolyForm Noncommercial 1.0.0. You can sell the videos you make with it. You can't resell the tool.

## To do

- Intel Macs, Windows and Linux.
- Bring the server up in the first session without a Reconnect.
- Keep laughs safe from the automatic trims at the harder paces.
- Undo more than one step.
- Ship the extras switched on.
