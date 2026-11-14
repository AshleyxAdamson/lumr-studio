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
- What the creator said off script stays. It is often the best part. Take out an aside only when they ask.
- Where a scripted line was recorded more than once, the last complete take is the one that plays, the lead-in question included.
- Everything the creator needs to read goes in the chat.
- To wait for a job, call `job_status` with `wait: 50`.

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

Call `assemble` with the whole sequence in play order: the creator's footage up to the first splice point, the excerpt, the footage from that point to the next, and so on. For a joined video, use the clips and times under `reel` in `get_edit`, and split the piece a splice point falls in. Give the new video a new name.

When the answer says a clip was recorded in HDR, tell the creator in one sentence. The joined video shows it in standard color.

Then call `carry_edit` from the old video to the new one, and wait. Read the whole report:

- **split**: a cut ran across a splice point, as it does when the creator's wait was cut. That is right.
- **shortened**, **dropped**, **not_placed**: read each one's `why`. If words the creator meant to lose are back, cut them again with `set_edit`.

Done when the report's cuts add up: every cut of the old edit is carried, split, or named with a reason you can say to the creator.

## 6. Check each splice

For every splice point, both sides of it:

- `look` at the join into the clip and the join out of it. The talk ends on a whole word and the clip starts on one.
- `preview` around it, with enough `pad` to hold the cue line, the whole excerpt and the line after. Keep each preview's path.
- `read_transcript` across it. The cue line, the excerpt and the next line read as one thought.

The pace tightens pauses inside the clips too. Laughs and other sounds are safe, but a clip where someone thinks out loud can lose its air. If an excerpt reads rushed, tell the creator they can select it on the review page and press Keep.

Done when every splice has a clean `look` on both sides and a preview.

## 7. Present

In chat:

- The table from step 4, now with where each clip sits in the finished video.
- One preview per splice, by path, so the creator can watch just those moments.
- What `carry_edit` split, shortened or dropped, in plain words.
- The new length beside the old one.

Then offer the review page (`review`), where each clip shows by name on the bar and in the words. Render only when the creator says go.

## When the creator asks for a faster video

"Faster", "tighter" and "more momentum" mean shorter pauses. Raise the pace with long pauses on and the other switches as the creator left them, and say how much shorter the talk got. If that falls short of what they asked for, say so, and say that the rest would mean taking out sentences. That choice is theirs.
