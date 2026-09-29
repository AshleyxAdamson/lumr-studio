---
name: publish-kit
description: Use when a creator has a finished or edited video and wants a title, description, chapters, and tags to post it, in their own words.
---

# Publish kit

Write everything from what the creator actually said. Nothing here is generic marketing copy. If a word or phrase isn't something the creator would say, it doesn't belong in the kit.

## 1. Get the transcript and the edit

Call `transcribe` (it returns at once if a transcript exists, otherwise call `job_status` with `wait: 50` until it's done). If the result carries a `warning` about estimated word timings, call it again with `force` set to true. Then call `read_transcript` from the start. Each response ends with `NEXT <time>` or `END`. On `NEXT`, call again with that time as `start`, until you see `END`.

Check `get_edit` for a saved edit. If one exists, call `read_transcript` again with `show_cuts` true and write only from words marked as kept. A line the creator cut isn't part of the video anymore, so it can't appear in the description or a chapter label, however good the line was.

## 2. Write in the creator's own words

Reuse their phrasing. If they call something "the trick" in the video, call it that in the description too, not "the technique" or "the method." Don't add hype words they didn't use themselves.

If this is the only video you have for this creator, say so: tell them voice confidence is low, since one video isn't enough to know their range. It gets better as they package more videos.

## 3. Titles

Write three options, each a different shape:

1. A plain statement of what the video is about.
2. A question the video answers.
3. A curiosity gap, naming a specific detail without giving it away.

Each under 70 characters. Each built from something the creator actually said in the video, not a template filled with generic words.

## 4. Description

Two or three short paragraphs, written as the creator, first person. Say what the video covers and why someone would want to watch it, using their own language from the transcript.

After the paragraphs, add a chapters block (see below).

If the creator gives you standing links (a newsletter, a course, a channel handle) to include, keep them exactly as given, character for character. Never invent a link. If they haven't given you any, leave that part out rather than guessing.

## 5. Chapters

Pick chapter points by finding natural topic shifts in the kept transcript. For each, note the **source** time (the time in the original recording, since that's what the transcript shows).

Once you have your source times, call `chapter_times` with all of them at once and use only the times it returns. Never write a chapter time by hand, even to round it or nudge it. If a chapter point falls inside a cut, `chapter_times` moves it to the start of the next kept segment for you.

Rules, enforced by `save_publish_kit` but worth checking yourself first:

- At least three chapters.
- The first at 0:00.
- Each chapter at least 10 seconds long.
- Labels of two to five words, in the creator's language, not generic labels like "Part 1."

## 6. Tags

Write 8 to 15 tags. Order specific before generic: the exact thing the video is about first, broader category terms after. Only use topics the video actually covers. Don't pad the list with unrelated popular terms.

## 7. Show the kit and take edits

Show the whole kit to the creator at once: titles, description with chapters, tags. Take their edits in plain language and update it. Only call `save_publish_kit` once they approve it. Give them the folder path it returns.

## 8. Remind them posting is theirs

Say plainly that this only writes the kit to a folder. They still open YouTube Studio and paste it in themselves. Nothing here posts anything on their behalf.

## Handling tool errors

Read the error message before retrying. A common one is `save_publish_kit` rejecting a chapter list (too few chapters, one under 10 seconds, or the first not at 0:00). Fix the specific field it names and retry once. If it still fails, explain the problem to the creator in plain terms rather than guessing again.
