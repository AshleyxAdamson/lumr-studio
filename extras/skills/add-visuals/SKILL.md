---
name: add-visuals
description: Use when a creator wants their own photos, screenshots, or video clips shown over their talking-head video while they keep talking, such as b-roll, a picture of what they're describing, or a screen recording in a corner.
---

# Add visuals

Show the creator's own photos and clips over their talk. The voice keeps playing and the picture changes, the way b-roll works. You place what the creator gives you. You never make a picture, fetch one, or guess at one.

Place a picture because the viewer wants to see it. A picture also hides the jump where a cut was made, which is a side benefit.

## Ground rules

- Use only the Lumr Studio tools. No shell commands, no scripts, no files of your own.
- Use only files the creator gave you, by their full path. If they say "the photo from Berlin", ask for the file. Never guess a path or name a file they haven't given you.
- Nothing is copied. Each overlay points at the creator's own file. If they move or rename it later, the render stops and says which one.
- Never delete a file.
- Everything the creator needs to read goes in the chat.
- To wait for a job, call `job_status` with `wait: 50`. Call it again if it's still running.

## 1. See what's there

Get the video path from the creator. Call `get_edit` to see the saved cuts, and `get_overlays` to see what's already placed. Each placed overlay has an `id`. Keep it: pass it back when you change that overlay, so the creator's list stays stable.

If there's no transcript yet, call `transcribe` and wait for it.

## 2. Find the moments

Call `read_transcript` with `show_cuts` true, from the start. Each response ends with `NEXT <time>` or `END`. On `NEXT`, call again with that time as `start`, until you see `END`. Read it all before you suggest anything.

Look for moments where a viewer would want to see something:

- The creator names a place, a person, an object, or an event. "My first apartment." "My old teacher." "The letter from my aunt."
- They describe something on a screen: an app, a message, a website, a document.
- A before and after, a list they count through, a number they want to land.
- A long stretch of one framing with several cuts in it, where a picture would rest the eye.

Skip these, even when something is named:

- Lines marked `CUT`. They're not in the video.
- A punchline, the setup right before it, and a `(laugh)` after it. The creator's face sells the joke.
- The first seconds of the video, and emotional beats such as a pause before a hard truth. The face carries those too.

## 3. Ask for the files

List the moments you found in the chat, three to six of them. For each, give the time and a short quote:

> 1. 8:11, "the bread came out perfect. We ate it warm..." A photo of the loaf would fit here.
> 2. 8:40, "the market gets busy by noon..." A clip of a street or a crowd would fit.

Then ask which ones they have a photo or clip for, and for the file's path. Dragging a file into the chat gives its path. A moment with no file is dropped. Don't press for more.

## 4. Propose placements and wait for a yes

Write a numbered list in the chat. For each overlay:

- The file's name.
- Where it starts and ends: the times and the words it covers.
- Full frame, or a box in a named corner.
- For a photo: drifting in, drifting out, or still. For a clip: muted, or its sound quietly under the voice.

How to choose:

| The creator gives you | Place it |
|---|---|
| A photo of the thing they're talking about | full frame, `motion: zoom_in` |
| A screenshot, a message, a document | a box, `size` 0.4 to 0.5, `motion: still` so text holds still |
| A clip of what they're describing | full frame, muted |
| A screen recording they talk over | a box, muted |

- Start on the word that names the thing. End at the end of that sentence, usually 4 to 8 seconds later.
- A photo taller than the video is shown whole over a blurred copy of itself. Say so if it matters to them.
- Two at once only as a box over a full frame. The box goes on a higher `layer`.
- Mix a clip's sound in only when the creator asks, and at `volume` 0.2 or less. Their voice stays on top.

Wait for the creator to say yes, or to change the list.

## 5. Save and look

Call `set_overlays` with the whole list, every overlay at once. It replaces what was saved. Include the ones already placed, with their ids, or they're dropped.

Read the result:

- `rejected`: each says why and how to fix it. Fix it or drop it.
- `notes`: what the edit does to each overlay. A cut under an overlay shortens it and is fine. `hidden` means its words were cut, so it won't show. Tell the creator, and offer another spot.
- `picture`: a still of each overlay as the viewer will see it. Look at it before you say anything is done.

In the picture, check:

- The right picture is on the right words.
- A box doesn't cover the creator's face.
- Text in a screenshot can be read.
- Nothing is sideways, stretched, or cropped in a way that loses the point of it.

If something's wrong, fix the list and call `set_overlays` again. Do this at most three times.

## 6. Show the creator

In the chat, list each overlay with its time, the file's name, and what the viewer sees. Put any note in plain words: "The photo at 8:11 now shows for 5 seconds, since the pause trims came out under it."

If they want to watch one, call `preview` at the overlay's start time and give them the clip's path. Previews and renders draw the overlays.

Render only on a clear go. Call `render`, wait on `job_status`, and give them the output path. The result's `overlays` says which were drawn and which weren't.

## 7. When the cuts change later

Overlays stay on their words when the creator changes the cuts or the pace. Afterwards, call `get_overlays` and read `notes`. Tell the creator about any overlay that's now `hidden`, and offer to move it or to keep that part of the talk.

## Handling tool errors

Read the error message. It says what was wrong. Fix the call and retry once. If it fails again, tell the creator plainly what's wrong and what you'd need from them. A missing file means asking the creator where it went. Never swap in a different file on your own.
