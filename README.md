# Lumr Studio

**Finally, an editing harness that cuts your speech without dropping you mid-word, and trims the gaps between your words so seamlessly nobody hears the edit.**

Lumr Studio is a Claude Code plugin for talking-head video. Claude reads your whole take, cuts the dead air, the filler and the stumbles, and checks every join before you hear it. It's free and runs on your Mac (Apple Silicon for now). Your video never leaves it.

## Why it's different

Most tools cut on a loudness line or on the transcript's word times. Loudness cutters clip soft word endings and miss the small gaps inside sentences. Transcript cutters are only as good as their word times, and raw speech-to-text times are loose: on our test take they hid 427 seconds of real silence inside words. When those tools do shorten gaps, they cut every one to the same length.

Lumr Studio is built around the join, where one kept word lands next to the next.

### What only Lumr does

- **Pauses sized to where they fall.** Most tools delete pauses or cut every gap to one length. Lumr leaves the longest pause between sentences, a shorter one at a comma and the shortest inside a clause, so the edit keeps your rhythm.
- **A reason for every cut.** Claude reads the whole take before it cuts anything, and every cut on the plan says why.
- **Your jokes stay funny.** Laughs are picked out of the audio, and Claude keeps the punchline, the setup before it and the laugh after it.
- **It knows which "like" is filler.** Claude reads each one. "I like jazz" stays, "into like three sections" goes, and when it's unsure the word stays.

### The standards, done properly

- **Never cuts inside a word.** A second model measures each word against your voice instead of trusting the transcript's guess, and any cut edge that would land in a word gets moved out.
- **Checks every join.** Each edit comes back with a read of every join, from clipped laughs to unfinished sentences to gaps that are too tight. Claude fixes the flagged ones before you see the plan, and can look at any join as a picture of the frames, the sound wave and the words.
- **No clicks.** Cuts land on the exact sample, with a 20 ms fade either side.
- **Private and yours.** Nothing uploads, nothing renders until you say go, and there's no account or telemetry.

On a 21:47 test take, measured word times took out more than twice as much at every pace (3:26 vs 1:25 at Standard) and cut 94 of 112 "likes" cleanly, against 28 without them.

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

- **Timeline export, next up.** Send the edit to Premiere, Final Cut or Resolve as a timeline (FCPXML, Premiere XML and EDL) instead of a finished file, so you can finish in your own editor. It's built from the same kept pieces the render uses, and each cut carries Claude's reason as a marker.
- Intel Macs, Windows and Linux.
- Bring the server up in the first session without a Reconnect.
- Keep laughs safe from the automatic trims at the harder paces.
- Undo more than one step.
- Ship the extras switched on.
