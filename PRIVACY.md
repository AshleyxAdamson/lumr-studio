# Privacy

Last updated 2026-09-28.

Lumr Studio doesn't collect your data. It has no server, no accounts, no analytics and no telemetry. Nothing on your Mac is ever sent to the person who wrote it, except feedback you choose to send (see "Feedback you send").

Questions about this page go to [GitHub Issues](https://github.com/AshleyxAdamson/lumr-studio/issues). The page lives at `PRIVACY.md` in [the repo](https://github.com/AshleyxAdamson/lumr-studio/blob/main/PRIVACY.md), and its history is public.

## What stays on your Mac

- Your video and audio. The plugin opens a file only when you name it to Claude, like "use tight-cut on this video". It never browses your folders. It never uploads your video or audio, and it has no tool that could.
- What it makes from them. The transcript, the measured word times, your edit and your exports. The section below says where each one lives.
- The review page. A small web server on `127.0.0.1`, on a port your operating system picks, with a private token in the address. Nothing outside your Mac can reach it, and the page makes no outside requests. The one thing it can send out is feedback you write and submit, and your Mac's own server sends that (see "Feedback you send").

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
- Where it goes: the Lumr Studio team's server, sent by the local server on your Mac when you press Submit. The page itself never calls the internet. The server's address is `LUMR_SHARE_URL`. Until that server is set up, the form opens a public GitHub issue in your browser instead, and nothing is sent until you submit it there. GitHub's own privacy policy covers it.
- How long it's kept: until it's deleted. To ask for deletion, quote the ID you were given. TODO(owner): contact for deletion.

## Children

The plugin collects nothing from anyone, whatever their age. Feedback is sent only if the person presses Submit. It's built for people who edit their own videos with Claude Code.

## Changes

If this page changes, the change shows in the repo's history and the date at the top moves. A new host or a new kind of data changes this page in the same release.
