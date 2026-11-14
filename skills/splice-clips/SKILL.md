---
name: splice-clips
description: Use when a creator wants their own extra clips put into a video where the script or the talk calls for them, such as a clip the talk points at ("watch this"), a recording of something they describe, or b-roll that carries its own sound.
---

# Splice clips

Put the creator's extra clips into their talk at the moments the talk calls for them. Each clip plays with its own picture and sound, between two lines of the creator's, and then the talk goes on.

The creator and the script decide what goes where. The tools show you the footage, make the change, and check it. When a tool offers something the creator didn't ask for, such as a harder pace or a filler cut, leave it off.

You are editing footage you can't hear. Check your own work with the tools before the creator has to.

## Ground rules

- Use only the Lumr Studio tools. Use only clips the creator gave you, by their full path.
- Every **cue** gets a clip or a stated reason it has none. A cue is a place where the talk needs a picture: a script line such as `[Clip: the oven]`, or a spoken line that points at one ("watch this", "here's what the crust sounds like", "this is how it looked the next morning").
- Every clip the creator gave you is either used or named as unused, with why.
- Before any cut that removes words, read the `what-to-cut` skill. It says what stays, such as an aside or the last take of a line, and what goes.
- Everything the creator needs to read goes in the chat.
- To wait for a job, call `job_status` with `wait: 50`.
- When `transcribe` or `carry_edit` answers `needs_models`, nothing has downloaded. Tell the creator what would download, how big and from where, and ask. Only on a yes, call it again with `download_models` true.

## 1. Find the cues

Get the video's path, the clips' paths and the script if there is one. Call `get_edit` on the video: you need its saved cuts and, when it is a joined video, its `reel` pieces.

Call `read_transcript` with `show_cuts` true and read to `END`. List every cue with its time and the line just before it. Where the creator stopped talking and waited, the wait is the room they left for the clip: note how long it is.

Done when every bracketed cue in the script and every spoken line that points at a picture is on the list.

## 2. Watch and read every clip

For each clip: `transcribe` it, `read_transcript` to `END`, and call `frames` at six times spread across it. Do this for the whole clip, not its first seconds: the useful part is often in the middle.

Done when, for each clip, you can say what is on screen and what is said in each stretch of it.

## 3. Choose the excerpt for each cue

An **excerpt** is the stretch of a clip that delivers what the creator's line just promised. Choose it by these rules.

- It answers the line before it and sets up the line after it. After "watch this", the viewer sees the thing. Before "so the crust is twice as thick", the viewer has just seen the loaf cut open.
- It starts and ends on whole sentences of whoever speaks in it.
- It is as long as the wait the creator left, give or take a few seconds. With no wait to go by, 10 to 40 seconds. Two excerpts of one clip, back to back, are fine when the good parts are far apart.
- Setting up, small talk and anything that shows or says nothing new stay out.
- When two clips fit one cue, take the one that shows it. Name the other as unused.

A clip whose sound matters is spliced in. A photo, or a clip the creator should keep talking over, is laid over the talk: that is the `add-visuals` skill, which comes with the overlays extra. When that skill isn't in your list, the extra is off: tell the creator it exists and that the guide says how to turn it on, and carry on with the clips that are spliced.

Done when every cue has an excerpt with a start, an end and one sentence of why, and every clip is used or named.

## 4. Show the plan and wait

In chat, one table: the cue line, the clip, the excerpt's times, how long, why. Under it, the clips left out and why. Ask for a yes or changes. Nothing is joined yet.

## 5. Join, then carry the edit

A **splice point** is where a clip goes in: right after the last word of the cue line, at the start of the wait. Take the time from the transcript.

Call `assemble` with the whole sequence in play order: the creator's footage up to the first splice point, the excerpt, the footage from that point to the next, and so on. For a joined video, start from the pieces under `reel` in `get_edit`, each with its `clip`, `start` and `end`, and split the piece a splice point falls in. The transcript's times are seconds in the joined video, and a piece's `start` and `end` are seconds in its clip: second `t` of the joined video, inside a piece, is clip second `start + (t - reel_start)`. Give the new video a new name.

When the answer says a clip was recorded in HDR, tell the creator in one sentence. The joined video shows it in standard color.

Then call `carry_edit` from the old video to the new one, and wait. Read the whole report:

- **split**: a cut ran across a splice point, as it does when the creator's wait was cut. That is right.
- **shortened**, **dropped**, **not_placed**: read each one's `why`. If words the creator meant to lose are back, cut them again with `set_edit`.

Every `set_edit` answer carries `rejected`. Read it first: a cut listed there is not in the edit, so what it would remove plays. When the reason is a part the creator kept, tell them which cut it was, its time and length, and send it again with `override_keeps` only on their yes.

Done when the report's cuts add up: every cut of the old edit is carried, split, or named with a reason you can say to the creator.

## 6. Check each splice

For every splice point, both sides of it:

- `look` at the time the clip starts in the new video and at the time it ends: the `reel_start` and `reel_end` of its piece in the `assemble` answer. `look` draws the cut nearest that time, or the plain footage when no cut is within its `span`. Either way, find the splice in the stills and the words: the talk ends on a whole word and the clip starts on one.
- `preview` with `at` in the middle of the excerpt and `pad` at half the excerpt's length plus 8 seconds, so it holds the cue line, the whole excerpt and the line after. `pad` is at most 30: for an excerpt longer than 40 seconds, make two previews, one at each end of it. Keep each preview's path.
- `read_transcript` across it. The cue line, the excerpt and the next line read as one thought.

The pace tightens pauses inside the clips too. A laugh and any sound of a second or more always play whole, but a clip where someone thinks out loud can lose its air. If an excerpt reads rushed, tell the creator they can select it on the review page and press Keep.

Keep holds the whole sentences they picked from, with a laugh right after the last one, and it beats your cuts: a cut that overlaps the kept part is put back whole. So once they say they pressed it, call `get_edit` and check the kept part:

- Its entry under `creator.kept` ends at the excerpt's last word, or at a sound right after it.
- `new_duration` is longer than before by the pauses inside the excerpt and no more.

A part that runs on into the wait after the clip, or a length that jumped by about that wait, means your cut of the wait is back in the video. Tell the creator: they can remove that part from the Kept list on the page and keep the excerpt's words again. On your next `set_edit`, `rejected` names any cut the kept part still covers.

Done when every splice has a clean `look` at both ends of its clip and a preview.

## 7. Present

In chat:

- The table from step 4, now with where each clip sits in the finished video.
- One preview per splice, by path, so the creator can watch just those moments.
- What `carry_edit` split, shortened or dropped, in plain words.
- The new length beside the old one.

Then offer the review page (`review`), where each clip shows by name on the bar and in the words. Render only when the creator says go.

A spliced video is always an encoded reel, so its `quality` sets how good an export can be. When the creator says this is the final, is for upload, or asks for the best quality, read `footage` and `reel` in `get_edit` first and say what they have and what the export will be: a master export (`render` with `quality: "master"`) is about `kept_mb` megabytes. If the reel's `quality` is `standard`, say plainly that a master export of it can only be as good as that reel, and what building it again costs: `assemble` with the same name and pieces and `quality: "master"` takes about as long as the first join and makes a second reel about as big as the clips, and `carry_edit` takes the edit across. If they say before step 5 that this is for upload, pass `quality: "master"` to `assemble` there and skip the second join.

## When the creator asks for a faster video

Follow the `what-to-cut` skill's part on "Faster": it says what the ask means, whose the switches are, and what to say when pauses fall short. The call that raises the pace is `set_edit` with your cuts as `get_edit` lists them, `auto_tighten` true and a harder `pace`. The answer's `auto.take_out` lists the switches that are on: without `pauses` in it, the pace has no pause to shorten. Its `new_duration`, beside the one `get_edit` gave before, is how much shorter the talk got.
