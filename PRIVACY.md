# Privacy

Last updated 2026-10-02.

Lumr Studio doesn't collect your data. It has no server, no accounts, no analytics and no telemetry. Nothing on your Mac is ever sent to the person who wrote it, except feedback you choose to send (see "Feedback you send") and the shape of your changes, if you choose to send it (see "What you can choose to send").

Questions about this page go to [GitHub Issues](https://github.com/AshleyxAdamson/lumr-studio/issues). The page lives at `PRIVACY.md` in [the repo](https://github.com/AshleyxAdamson/lumr-studio/blob/main/PRIVACY.md), and its history is public.

## What stays on your Mac

- Your video and audio. The plugin opens a file only when you name it to Claude, like "use tight-cut on this video". It never browses your folders. It never uploads your video or audio, and it has no tool that could.
- What it makes from them. The transcript, the measured word times, your edit and your exports. The section below says where each one lives.
- The review page. A small web server on `127.0.0.1`, on a port your operating system picks, with a private token in the address. Nothing outside your Mac can reach it, and the page makes no outside requests. The two things it can send out are feedback you write and submit, and the shape of your changes if you press Send. Your Mac's own server sends both (see "Feedback you send" and "What you can choose to send").

## What Claude sees

Lumr Studio runs inside Claude Code, so Claude is part of the conversation. The plugin adds these to it:

- The transcript text, in short packed lines.
- Cut lists with their reasons, and the results of each tool.
- When Claude checks a join with `look`, one small picture: a few still frames from your video, the sound wave drawn as a line, and the words around the cut. That's the only time an image from your video goes to Claude.

That goes wherever your Claude Code sends its requests, under your Claude plan's terms. Anthropic's policy is at https://www.anthropic.com/legal/privacy. The plugin doesn't add a second copy, and its author doesn't get one.

## What the plugin downloads

Nothing of yours goes out in these. Each host sees the request itself: your IP address and the file you asked for, as with any download. The plugin's own downloads send one fixed name as the user agent, `lumr-studio-plugin`, and no ID.

| Host | What | When |
|---|---|---|
| `pypi.org`, `files.pythonhosted.org` | The plugin's locked Python packages, through `uv` | The first session, in the background, and again when an update changes the package list |
| `github.com/astral-sh/python-build-standalone` | A Python build, only if your Mac has none from 3.11 to 3.13 | The same first session |
| `huggingface.co`, and the download hosts it redirects to, such as `us.aws.cdn.hf.co` | The speech model, `mlx-community/parakeet-tdt-0.6b-v2`, at one pinned commit | Only after you say yes |
| `download.pytorch.org` | The word timing model | Only after you say yes |

Hugging Face's usage telemetry is switched off (`HF_HUB_DISABLE_TELEMETRY=1`). Every model file is checked against a size and a sha256 pinned in the plugin before it's installed.

The first-run check may print `https://brew.sh` in a message when Homebrew is missing. It doesn't visit it.

## What's kept, and for how long

The plugin keeps files on your Mac until you delete them. The author holds nothing, so there's nothing to ask the author to delete.

| What | Where | To remove it |
|---|---|---|
| Project files: `edit.json`, `words-aligned.json`, `plans.json`, `exports/` | `~/Lumr/studio/projects/<video name>-<id>/`. The `~/Lumr` part follows `LUMR_HOME`, or the folder named in `~/.config/lumr/home.txt`. | Delete the project folder |
| What Lumr learned from your changes | Nothing extra is stored. It's worked out from the project files above each time, and `taste.json` in `~/Lumr/studio/projects/` only lists what you asked it to forget. | Ask Claude to forget it, or delete a project folder to drop what that video taught |
| The transcript | `<video>.words.json`, beside your video. It's the only file the plugin puts in your video's folder. | Delete the file |
| Measured silences | `~/Lumr/media_cache/` | Delete the folder |
| The Python environment and `build.log` | The plugin's data folder, under `~/.claude/plugins/data/` | Uninstalling the plugin removes it |
| The two models | `~/.cache/huggingface/hub` and `~/.cache/torch/hub/checkpoints`, or under `HF_HOME` and `TORCH_HOME` | Delete them by hand. They stay after an uninstall, because other tools share those caches. |

No tool in the plugin deletes a file of yours.

## Feedback you send

The review page has a Send feedback button. Nothing is sent until you press Submit in its form.

- What goes: your message, the plugin version, and a random ID for that one message. Your name and email go only if you type them, and both are optional. No video, audio, transcript or file goes with it. Please don't paste anything you'd rather keep private into the message.
- Where it goes: the Lumr Studio team's server at `feedback.lumr-studio.workers.dev`, a Cloudflare Worker that stores it in Cloudflare R2. The local server on your Mac sends it when you press Submit. The page itself never calls the internet. Setting `LUMR_SHARE_URL` to an empty value turns sending off; then the form opens a public GitHub issue in your browser instead, and nothing is sent until you submit it there. GitHub's own privacy policy covers that.
- How long it's kept: up to 6 months, then deleted. To have it deleted sooner, email ashleyxadamson@gmail.com and quote the ID you were given.

## What you can choose to send

The review page has a Help improve Lumr button. It shows unless sending is turned off with an empty `LUMR_SHARE_URL`. It opens a list of what would be sent, one line for each change, and you can remove any line. Nothing is sent until you press Send.

- What a record holds: one record for each thing you did to Claude's edit (a cut you put back, a cut you marked good or wrong, words you brought back, words you cut by hand, a part you kept, or a cut of Claude's you left in place). Each holds the pace, how long the cut was and how many words it held, what kind of cut it was and who made it, what you did to it, and the shape of up to three words on each side: what sort of word it is, how long it lasts, the pause after it, and which filler it is if it is one. It also holds whether the cut sat at the start or end of a sentence, how far a laugh was, the silence left at the join, and any warnings the join check gave. Each send carries the plugin version and a random ID.
- What it never holds: your words, your transcript, your video or audio, file names, your name or email, or a time in your video. The words beside a cut are sent as shapes, not text. The only words that can appear are a fixed list of filler words, such as "um", "uh" and "you know". The button shows the list, and "Show exactly what's sent" shows the data itself.
- Where it goes: the Lumr Studio team's server at `feedback.lumr-studio.workers.dev`, the same as feedback, which stores it in Cloudflare R2. The local server on your Mac sends it when you press Send. The page itself never calls the internet. With an empty `LUMR_SHARE_URL`, the button doesn't show. The server keeps the day it received the send, not the time. It stores and logs no IP address and no details of your Mac. It uses your IP address only to limit how fast one Mac can send, and doesn't keep it.
- How long it's kept: up to 6 months, then deleted.
- How to ask for deletion: email ashleyxadamson@gmail.com and quote the ID you were given.
- What stays on your Mac: `shares.jsonl` in the video's project folder keeps the ID of each send, the date and how many records it held, so you have the ID to quote. It holds nothing else. Delete the file and nothing else changes.

## Children

The plugin collects nothing from anyone, whatever their age. Feedback and shapes are sent only if the person presses Submit or Send. It's built for people who edit their own videos with Claude Code.

## Changes

If this page changes, the change shows in the repo's history and the date at the top moves. A new host or a new kind of data changes this page in the same release.
