---
name: tight-cut
description: Use when a creator has a talking-head recording and wants it tightened or edited: trimming pauses, filler, false starts, and dead weight while keeping their meaning and voice intact.
---

# Tight cut

Cut a recording for pace. The creator's meaning, humor, and rhythm stay what they were. You choose what to remove. You never change what they said.

You are editing footage you can't hear. Check your own work with the tools before the creator has to.

## Ground rules

- Use only the Lumr Studio tools. No shell commands, no scripts, no files of your own.
- Never delete a file. If the creator wants old previews cleared, tell them the folder and let them do it.
- Everything the creator needs to read goes in the chat. Never put a plan in a file.
- To wait for a job, call `job_status` with `wait: 50`. Call it again if it's still running.

## 0. Say up front what it needs

Lumr Studio needs two models on this Mac, and the first time they have to download. Don't let that surprise the creator after they've picked a video and are ready to go. In your first reply, before you ask for the video or anything else, tell them in a few short lines:

- **Parakeet** (NVIDIA Parakeet TDT 0.6B v2, 2.47 GB from Hugging Face) writes the transcript: the words they said, with punctuation.
- **wav2vec 2.0** (0.38 GB from PyTorch's download site) finds exactly where each word starts and ends in the sound. That's what lets the cuts trim the gaps between words without ever cutting into one.
- Together that's 2.85 GB, downloaded once and kept for every video after. Nothing downloads until they say yes, and you'll ask before it does.

Skip this if the creator says they've used Lumr Studio on this Mac before, or if a `transcribe` answer in this conversation already came back without `needs_models`.

## 1. Confirm the video

Get the video path from the creator. Call `transcribe` with it. It transcribes and measures the word times in one job. If the models aren't on this Mac yet, it answers `needs_models` and downloads nothing.

| The answer | Do |
|---|---|
| `status: "needs_models"` | Nothing has downloaded, and nothing does until the creator says yes. Ask first. See below. |
| `status: "downloading"` with a `job_id` | The creator said yes and the download started. Wait on `job_status`, and tell the creator how far along it is. When it's done, call `transcribe` again with the same path. |
| `job_id` | Wait on `job_status`. Tell the creator it transcribes and then measures where each word sits, about a minute more for a 20 minute video. |
| `status: "aligning"` with a `job_id` | The transcript is there. The job measures the word times. Wait on it. |
| `status: "exists"` | Go on. |

On `needs_models`, ask the creator before you do anything else. In a few short lines say what would download, from `models`: each name, what it does, its size in MB, where it comes from (`source_host`) and its license, then the total. What each does, in plain words:

- The speech recognition model (Parakeet) writes the transcript. Without it there's nothing to edit.
- The word timing model (wav2vec 2.0) pins each word to the sound, so cuts land in the real gaps between words. Without it the edit still works, on the transcript's rougher times, and the harder paces take out much less.

Say that nothing has downloaded yet and that it happens only if they say yes. If you already told them about the models in step 0, keep this short and just ask. Then stop and wait for the answer.

- **Yes:** call `transcribe` with the same path and `download_models` set to true. Wait on `job_status` with `wait: 50`, and call it again while it runs. Say the progress when you have it. If the job fails, tell the creator the error in one sentence and offer to try again. A stopped download picks up where it left off, and a file that fails its checksum was not installed.
- **No:** if the answer has a `word_count`, the transcript is already there. Go on with the edit on estimated word times and tell the creator once, in one sentence, that the harder paces will take out much less. If it has no `word_count`, there is no transcript and nothing to edit. Say so, and say the speech model is what's needed. Don't download and don't look for another way.

That is the only wait before you read. Then look at `word_times` in the result:

- `measured`: every pace works as it should.
- `estimated`: this machine can't measure word times (`word_times_note` says what is missing). Go on with the edit. Tell the creator once, in one sentence: the pauses between words are hidden, so the harder paces will take out much less than they could. Don't retry and don't try to install anything.

If the result carries `word_times_warning`, the room's own noise misled the measure of where words fade out. Tell the creator once, in one sentence, what it says, and go on with the edit.

If the finished job carries `edit`, a saved edit was placed again on the measured times. Tell the creator what it removed before and what it removes now.

If the result carries a `warning` about the transcript's own word timings, tell the creator in one sentence, then call `transcribe` again with `force` set to true and wait for it.

## 2. Read the whole video before you propose anything

Call `analyze_take` for the compact report.

Call `get_edit`. If it carries `creator`, the creator has tuned this video on the page before. Every cut they put back, every part they kept and every word they brought back is something they want left in. Don't propose it again. Every cut they made by hand is settled: it stays whatever you send. Use the pace and switches they chose.

If `get_edit` carries `taste`, read its `lessons`. They come from this creator's changes on their other videos. Let them shape your proposal: propose fewer cuts of a kind they usually put back, pick more of a filler word they cut by hand, never cut a word they keep bringing back, and start from the pace they keep choosing if they have no usual saved. This video's own `creator` changes and the spine rules always win over taste. When you apply a lesson, tell the creator in one short line which one. For example: "You usually put back restated points, so I've left most of them in."

Then call `read_transcript` from the start. Each response ends with `NEXT <time>` or `END`. On `NEXT`, call again with that time as `start`. Keep going until you see `END`. Don't propose a cut until you've read all of it. A cut made on partial context can break a payoff you haven't seen yet.

While you read, note three things:

- **Laughs.** A line `(laugh 3.5s)` means the line before it was a punchline that landed. `(laugh? 1.6s)` is a possible one. Treat both as jokes.
- **How this creator is funny.** Dry asides, jokes at their own expense, deadpan, whatever it is. Their humor is part of their voice, including the jokes you wouldn't make.
- **Setups.** A joke needs the line that sets it up, even when that line restates something. A line that looks redundant right before a punchline is usually the setup.

## 3. Ask at most two questions

Only ask if the answer would change the edit, and only if the creator hasn't already said:

- Target length, or how tight they want it.
- Whether there's a script or outline to keep to.

If both are clear, go straight to deciding cuts.

## 4. Decide the cuts

Think of the video as having a spine: the parts the story can't lose.

**Spine, never cut.**

- The hook, the core argument, the payoff.
- Setup a later point depends on. If cutting a moment leaves a later moment unexplained, that moment is spine.
- Every punchline, the setup before it, the laugh after it, and the pauses between them.
- Emotional beats, such as a pause before a hard truth.
- Asides that sound like the creator. These can look like dead weight in a transcript and are the opposite.

**Optional.** Examples, side stories, and elaborations that support the spine. Trim these first when the creator wants it tighter.

**Safe to cut, once clearly identified.**

- False starts and restarts. Keep the clean version.
- Runs of filler: several "um" or "you know" in a row. One filler word inside a natural sentence is never a cut of its own. A single "like" is a pick (step 5).
- Dead air well beyond a natural breath.
- Tangents that don't serve the spine and aren't funny or personal.
- Repeated takes. Keep the last complete take unless an earlier one is clearly stronger.

**Where a cut starts and ends.**

- Cut whole sentences. Use the start and end times printed on the transcript lines. Those are sentence boundaries.
- Cut inside a sentence only to fix a false start or a retake, where the words after the cut finish the sentence the words before it began.
- Read the join aloud in your head: the last kept words, then the first kept words after. If it doesn't read as one person talking, move the cut.
- Removing 20 seconds or more can lose the thread. Check that what follows still makes sense to someone who never heard the removed part.

**Give every cut a kind.** The creator sees your cuts grouped by it, so they can check one kind at a time. Pick the one that fits what you meant:

| `kind` | When |
|---|---|
| `repeat` | the creator says again what they already said: a restated point, a second take |
| `false_start` | a sentence started and dropped, then started again |
| `off_topic` | a tangent, a plug, an aside that doesn't serve the video |
| `other` | anything else: dead air, a cough, a throat clear |

Choose it from what you meant by the cut, not from words in your reason. If a line looks like a repeat but sets up a joke, it's spine, not a cut.

**Write each reason for the creator.** They read it on the page, next to the words you cut, and decide from it whether to put the cut back. One short sentence, in plain words, about what they said and why it can go. No editor or marketing shorthand: no "CTA", "pitch tail", "plug", "back-catalog", "b-roll".

| Good | Not this |
|---|---|
| You make the budget point again, a minute after the first time. | Redundant restatement of the budget point. |
| This asks viewers to watch your older videos, and it breaks up the story. | Back-catalog plug, pitch tail before the CTA. |

**Say it the way the page does.** Your choices are cuts, and so are the ones the creator makes by hand. The page has no name for the small automatic ones: it says what goes, such as long pauses, filler words, and stutters. Say those in chat too. The tools call them trims; keep that word out of what the creator reads.

**Automatic trims.** `auto_tighten` shortens pauses and drops stray fillers across the whole video. `pace` sets how hard. `analyze_take` lists what each pace would save on this take.

| The creator asked for | Use |
|---|---|
| didn't say | `auto_tighten` on, `pace` left out: their usual, or `standard` when they have none |
| a light trim | `pace: natural` |
| tight, or "a YouTube cut" | `pace: standard` |
| fast, punchy, short-form energy | `pace: fast` |
| harder than fast, "cut it harder", "really tighten it" | `pace: tight`, then `hard` if they want more |
| as hard as it goes | `pace: max` |

There are six stops: natural, standard, fast, tight, hard, max. Leave `pace` out unless the creator asked for something. Left out, it keeps what they set on the page or saved as their usual. Their own choice beats your guess.

Every pace leaves a pause at each join: the most between sentences, less at a comma, the least inside a clause. From `hard` on that pause is very short. The automatic pause trims steer around a marked laugh but can still clip one, most at the harder paces: on the test take, 10 places at Standard and 23 at Max. Each shows up in `joins` as a `removes_laugh` or `clips_beat` row on an automatic trim. Read those rows in step 6 and pass on their fix, which asks the creator to keep that stretch on the page.

Say what the hard stops cost before you use one. `hard` leaves almost no pause between sentences. `max` cuts every pause it can reach, several each second of talk in places, and some joins will sound choppy. Both take more filler words, such as "and", "so" and "you know". No pace takes a "like": that word is yours to pick (step 5). Offer the page so the creator hears it before they export: Next steps through each cut that takes words; plain pauses are skipped.

When the creator says it feels rushed or choppy, go down a level. When they say it drags, go up one. Tell them which level you used and what it saved.

You can only name one of the six stops. When the creator wants something between two of them, point them to Fine tune on the page: two sliders under the pace, one for the shortest pause that gets cut and one for how much speech stays between two cuts.

With `word_times: "estimated"` the stops from `fast` up remove little more than `standard`. Say so if the creator asks for a harder cut and it barely changes.

## 5. Pick the filler likes

No rule can tell "I like jazz" from "divide this into like three sections", so the pace never takes a "like". You can tell, by reading, and your picks are the only way one goes without the creator's hand. Do this after you have read the whole transcript and before you present the plan.

Call `find_words` with `["like"]`. It lists every place the word is said, with six words either side. Read each line and decide.

**The test.** Take the word out and read the sentence again. If it means the same, the word was filler.

| Kind | Example | Verdict |
|---|---|---|
| Verb | "I like jazz" | keep |
| Comparison | "it does look like a very old" | keep |
| Example, such as | "like a cup of tea" | keep |
| Quoting | "I was like, no way" | keep, it stands in for "said" |
| Pause word | "in the first week, like, or even" | cut |
| Hedge before a number or a thing | "divide this into like three sections" | cut |
| Sentence starter | "Like, how am I" | cut |

When unsure, keep. A wrong cut changes what the creator said. A missed filler costs a fifth of a second.

Skip a line that says `OUT:claude` or `OUT:creator`: it is gone already. A line that says `OUT:pick` you picked before; pick it again if it is filler, since a new list replaces the old one. A "like" never says `OUT:pace`; another word you look up can, and you can still pick it. A line that says `NOT CLEAN` you can pick too; the server leaves it in, since cutting it would clip the word beside it, and counts it for the creator.

Hand your picks to `set_edit` as `picks`, in the same call as your cuts (step 6): each one the `id` from `find_words` and a few words of reason, such as "pause word" or "hedge before a number". Picks are not cuts. Never put a single filler word in `cuts`.

If the creator cut likes by hand that you had kept (see `creator.cut_words` and `creator.cuts` in `get_edit`), that is a lesson. Look at which kind they cut, and pick that kind this time.

If `word_times` is `estimated`, most picks will be left in as not clean. Pick anyway, and tell the creator why few come out.

## 6. Save a draft and check it yourself

Call `set_edit` with your cuts and your `picks`. This saves a draft. Nothing is rendered and the source video is never changed.

Read the result:

- `rejected`: fix each one using the reason given, or drop it.
- `adjusted`: the server moved an edge to a word boundary. Fine.
- `joins`: the self-check. Read every flagged row. Each shows the words either side of the join and what would fix it.
- `picks`: how many of your picked likes are out, which were left in and why, which the creator kept, and which ids were refused. For a refused id, call `find_words` again and use its ids. When you call `set_edit` again to fix cuts, leave `picks` out: the saved ones stay.

For each flagged join, decide: fix it, drop the cut, or keep it because it's right. A `splice` on a retake is right. A `removes_laugh` almost never is.

Then call `set_edit` again with the corrected list. Do this at most three times. If joins are still flagged after that, leave them and tell the creator about them in step 7.

For the two or three cuts you're least sure of, call `look` with the cut's start time. Check the picture for a jump between the two large frames, and for sound running through the join line.

## 7. Present the plan in chat

Write a numbered cut list in the chat. For each cut, or each group of small trims on one line:

- Start and end time in the source video.
- Length removed.
- The words removed, quoted briefly.
- A plain reason.

Then:

- The current length and the new length.
- Any join still flagged, in plain words: "Cut 7 ends mid-sentence. I kept it since the next line restarts the thought. Listen to that one."
- Jokes you noticed and kept, in one line, so the creator knows you saw them.
- The likes, in one or two sentences: how many times the word is said, how many you picked as filler, and how many stay in. Say that the page has a switch for all of them (Filler likes) and that a double-click brings any one back or cuts one you kept. Such as: "You say 'like' 112 times. I picked 41 that are filler; the other 71 mean something, so they stay. On the page, Filler likes switches all 41 on or off, and you can double-click any single one."

Use the `clock` value on each cut for the times you show. Never work out minutes and seconds yourself.

Then offer the page, and wait for a yes before you open it. A browser window the creator didn't ask for is a bad surprise. The offer is the one line the creator must not miss, so it goes last, in this box, exactly as drawn and inside a code block so it keeps its shape:

```
  .-------------------------------------------------.
  |                                                 |
  |   >>>  REVIEW YOUR EDIT  <<<                    |
  |                                                 |
  |   Open the page to hear every cut, put any      |
  |   back, and knock out words yourself.           |
  |                                                 |
  |   Say "yes" and it opens in your browser.       |
  |                                                 |
  '-------------------------------------------------'
```

Under the box, one line: "It runs on your machine only. Or answer here, such as "put back 9"." Draw the box once per offer. Don't put anything else in a box.

On a yes, call `review` and give them the address with these steps:

1. Press Next to go to each cut in turn, then Play to hear it. The bar under the video shows where the most changed; press a block on it to play that stretch.
2. If it feels rushed or drags, move the pace. For a setting between two stops, open Fine tune under it and move the two sliders. Under Take out, turn off anything you want left in. Filler likes switches off the likes I picked.
3. Open my cuts by kind. Put back any you want to keep.
4. A word you want gone? Double-click it. Double-click a struck word to bring it back. Drag across words to cut or keep a part.
5. Found a moment that must stay? Select it and press "Keep".
6. Every change saves by itself. When it sounds right, press Export video, or come back and say you're done.

If they'd sooner answer in chat, skip the page.

## 8. Take the creator's answer

They may answer in chat, on the page, or both.

- **In chat.** "Put back 4" drops that cut. "Tighten the intro" means look at the opening again and cut harder there. Call `set_edit` with the new list, check `joins` again, and show what changed.
- **On the page.** When they say they're done, call `get_edit` and read `creator`. Tell them what you saw in plain words: the pace they chose, the switches they turned off, the cuts they put back, the parts they kept, the words they brought back, and the cuts they made themselves. Then say what you take from it in one sentence, such as "You put back both repeated lines before jokes, so I'll leave setups alone."

Their changes are already in the saved edit. If you call `set_edit` again, leave `pace` out so their pace stays, and don't send a cut they put back.

**Their own pace.** When `creator.fine` is there, they set the pace with the two sliders and it reads `custom`. The two values are theirs. Leave `pace` and `gap_length` out of every `set_edit`, or you replace their setting with a stop. Tell them what you saw in plain words, such as "You set it to cut pauses from 0.35 seconds, with 0.85 seconds of speech kept between cuts." Name a pace only when they ask you for one.

**Their switch.** `creator.take_out.likes: false` means they switched your picked likes off. Leave it. You can still hand in a better pick list; it is saved for when they switch it on.

A cut the creator put back, and any part they kept, stays that way. `set_edit` rejects a cut there. A cut of yours over a word they brought back splits around that word. Pass `override_keeps` only when the creator tells you to cut it after all.

**The creator's own cuts.** `creator.cuts` lists what they cut by hand and `creator.cut_words` counts what those cuts say.

- They are settled. Don't send them in `cuts`: they stay in the edit by themselves. Don't offer to put one back, and don't count their time as your saving.
- If the join check flags one, that is for you to know. Mention it only if they ask how a spot sounds. Never change their cut to clear a flag.
- Learn from them. Many cuts of one filler word, such as eleven "like"s, says they want fewer of them. Tell them what you noticed and offer what takes more. For "like", offer to read the ones still in again and pick more of them. For other filler words, the next pace up takes more (the page lists which, with counts). They can knock out the rest by double-click. Cuts of whole sentences say what kind of line they don't want; look for more lines like it and propose those as cuts of your own.
- A word they brought back says the opposite: leave that word, and ones used the same way, alone.

If they want to watch a stretch as a video file, call `preview` with the time. The page is faster for checking cuts one by one.

## 9. Render only on a clear go

The creator can export from the page by themselves. After they used the page, look for `export` in `get_edit`. If it's there, wait on `job_status` with its `job_id` and give them the output path. Render again only when `edit_changed_since` is true or they ask for it.

Otherwise, once the creator says to proceed, call `render`, wait on `job_status`, and give them the output path. If `render` answers `already_running`, an export is going: wait on the `job_id` it gave you.

## Handling tool errors

Read the error message. It says what was wrong. Fix the call and retry once. If it fails again, tell the creator plainly what's wrong and what you'd need from them. Don't retry silently more than once, and don't guess at a fix you can't explain.
