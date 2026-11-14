---
name: assemble-takes
description: Use when a creator has several clips of one video, or one clip with many retakes, and wants them put together following a script: joining the clips, keeping the best take of each line, and cutting the rest.
---

# Assemble takes

Put one video together from clips that hold retakes. The creator said each line more than once, on purpose, so they could pick the best one. You pick it for them, cut the rest, and leave them a draft to fine-tune. Their meaning, humor, and rhythm stay what they were. You never change what they said.

You are editing footage you can't hear. Check your own work with the tools before the creator has to.

## Ground rules

- Use only the Lumr Studio tools. No shell commands, no scripts, no files of your own. Don't open the clips or the script file yourself. The script comes in the chat.
- Never delete a file. The joined video and its folder stay. If the creator wants old files cleared, tell them the folder and let them do it.
- Everything the creator needs to read goes in the chat. The take map, the plan, the lists: never in a file.
- To wait for a job, call `job_status` with `wait: 50`. Call it again if it's still running.

## 1. Get the clips and the script

Ask for two things, in one message:

- **The clips.** Absolute paths, in the order the finished video should play. That is usually the order they were recorded. If the creator recorded the intro last, it goes first in the list. One clip is fine when all the retakes are in it. If a clip has a stretch they want left out, such as a long dead start, they can give a start and end in seconds inside that clip.
- **The script**, pasted into the chat, if there is one. No script is fine. Then the order they spoke in is the script, and you judge each take by what was said around it.

Pick a short name for the video, such as the script's title or what the creator calls it. Plain letters (A to Z), digits, spaces, `-` and `_` only. A name like "café" is refused, so leave accents out.

## 2. Join the clips and transcribe

Call `assemble` with the name and the clips as `pieces`, in play order. It joins them into one video file, the reel, and keeps a small file beside it that records which reel seconds came from which clip.

| The answer | Do |
|---|---|
| `status: "exists"` | A reel for exactly this list of clips is already there. Go on. |
| `job_id` | Wait on `job_status`. Tell the creator the clips are being joined. The finished job's `result` has the same fields as `exists`, with `status: "built"`. |

Read the answer:

- `reel` is the path of the joined video. **Use it as the `video_path` for every tool from here on.** The clips never go to another tool. Calling `assemble` again with the same clips gives back the same reel, so it is safe to call twice.
- `how` is `copy` or `encode`. `copy` means the clips matched, so they were joined without re-encoding and nothing was lost. `encode` means a clip had a `start` or `end`, or the clips differed in size, frame rate, or sound, so Studio cut and matched them to the first clip in one pass. It takes longer and the picture is encoded one extra time, then once more at export. Tell the creator which, in one sentence. You know whether you gave a `start` or `end`. If you gave neither and `how` is `encode`, the clips differed in some detail of their format, or the plain join came out wrong and was built again. The answer doesn't say which, so don't name a cause.
- `pieces` gives each clip's `reel_start` and `reel_end`. That is how you turn a reel time into "clip 2".
- A piece with `converted_from` (`HDR` or `wide color`) is a clip the camera, usually a phone, recorded in HDR. The reel shows it in standard color so it matches the other clips, and `how` is `encode` because of it. `note` has the sentence. Tell the creator in one sentence, such as "the phone clip was recorded in HDR; the joined video shows it in standard color". Their clip is not changed.
- `duration` is the length of the reel, the clips added up.

Studio never changes the clips. The reel is a second full copy of the footage in the Lumr folder (`studio/reels`), it takes as much disk space as the clips, and Studio never deletes it. Tell the creator where it is and that clearing it is up to them.

If `assemble` fails, the message says what was wrong, such as a clip that is missing, has no sound, or a start and end that don't fit. Fix the call and retry once.

Before you call `transcribe`, say up front what it needs. Lumr Studio needs two models on this Mac, and the first time they have to download. Tell the creator in a few short lines, while the reel builds if it is still running:

- **Parakeet** (NVIDIA Parakeet TDT 0.6B v2, 2.47 GB from Hugging Face) writes the transcript: the words they said, with punctuation.
- **wav2vec 2.0** (0.38 GB from PyTorch's download site) finds exactly where each word starts and ends in the sound. That's what lets the cuts land between words and never inside one.
- Together that's 2.85 GB, downloaded once and kept for every video after. Nothing downloads until they say yes, and you'll ask before it does.

Skip this if the creator says they've used Lumr Studio on this Mac before, or if a `transcribe` answer in this conversation already came back without `needs_models`.

Then call `transcribe` with the reel path. It transcribes and measures the word times in one job, and it takes longer the longer the reel is.

| The answer | Do |
|---|---|
| `status: "needs_models"` | Nothing has downloaded, and nothing does until the creator says yes. Ask first. See below. |
| `status: "downloading"` with a `job_id` | The creator said yes and the download started. Wait on `job_status`, and tell the creator how far along it is. When it's done, call `transcribe` again with the reel path. |
| `job_id` | Wait on `job_status`. Tell the creator it transcribes and then measures where each word sits. |
| `status: "aligning"` with a `job_id` | The transcript is there. The job measures the word times. Wait on it. |
| `status: "finding_sounds"` with a `job_id` | The transcript and its word times are there. The job finds the laughs and other sounds between the words, which an older transcript was never checked for. Nothing is transcribed again. Wait on it. |
| `status: "exists"` | Go on. |

On `needs_models`, ask the creator before you do anything else. In a few short lines say what would download, from `models`: each name, what it does, its size in MB, where it comes from (`source_host`) and its license, then the total. If you already told them in the lines above, keep this short and just ask. Say that nothing has downloaded yet and that it happens only if they say yes. Then stop and wait.

- **Yes:** call `transcribe` with the same path and `download_models` set to true. Wait on `job_status` with `wait: 50`, and call it again while it runs. If the job fails, tell the creator the error in one sentence and offer to try again. A stopped download picks up where it left off.
- **No:** if the answer has a `word_count`, the transcript is already there. Go on on estimated word times and tell the creator once, in one sentence, that the harder paces will take out much less. If it has no `word_count`, there is no transcript and nothing to work from. Say so, and say the speech model is what's needed. Don't download and don't look for another way.

Then look at `word_times` in the result:

- `measured`: every pace works as it should.
- `estimated`: this machine can't measure word times (`word_times_note` says what is missing). Go on. Tell the creator once, in one sentence, that the pauses between words are hidden, so the harder paces will take out much less. Don't retry and don't try to install anything.

If the result carries `word_times_warning`, tell the creator once, in one sentence, what it says, and go on. If it carries a `warning` about the transcript's own word timings, tell the creator in one sentence, then call `transcribe` again with `force` set to true and wait for it.

## 3. Read all of it

Call `get_edit` with the reel path. The `reel` field lists the clips and where each sits. If it also carries `creator`, the creator has tuned this reel on the page before. Every cut they put back, every part they kept, and every word they brought back is something they want left in. Their own cuts are settled. Don't propose any of it again.

Call `analyze_take`. Its `retakes` list shows phrases said again within a minute. Treat it as a hint. It misses retakes that are worded differently, and it lists repeats that are only a figure of speech.

Then call `read_transcript` from the start. Each response ends with `NEXT <time>` or `END`. On `NEXT`, call again with that time as `start`. Keep going until you see `END`. Don't choose a take until you've read all of it. A take chosen on partial context can drop a line you haven't seen yet.

Where a clip begins, the transcript has a marker line, such as:

```
== clip 2 of 4: MVI_4628.MP4 starts 378.28 ==
```

Note each one. The times in the transcript are reel times, and the markers tell you which clip you are in. While you read, note:

- **Slates and talk to the camera.** "Take two." "Okay, again." "Let me start over." The sound of the creator reaching for the camera at the end of a clip.
- **Laughs.** A line `(laugh 3.5s)` means the line before it was a punchline that landed. `(laugh? 1.6s)` is a possible one. `transcribe` finds the laughs, giggles and other wordless sounds between the words, and each has a line: `(sound 1.4s)` is one that is no clear laugh. A laugh the speech model wrote down as a word has no line, and neither has any sound while an answer carries `sounds_note` (call `transcribe` again first, and wait on its job). So also list the places the creator laughs on purpose: stage directions in the script such as "(laughs)", the stretch after a joke, and a long "pause" near a punchline or right at the start of a clip. A long "pause" whose picture from `look` shows a loud sound wave is most likely a laugh. You need this list in step 6.
- **Asides.** Things said that are not in the script. Some are whole thoughts. Some are the creator talking to themselves.
- **Long silences.** Stretches where the creator stopped talking but the recording kept running, often to leave room for b-roll.

## 4. Build the take map

For each script line, or each beat if there's no script, list every take of it with its reel times. Mark each take:

| Mark | Meaning |
|---|---|
| complete | The whole line, said through to the end, with no stumble. |
| broken off | The creator stops partway and starts again. |
| flubbed | Said through to the end, but a word is wrong, dropped, or tripped over. |

A worked example, in the seconds the transcript prints:

| Script line | Takes found | Each take |
|---|---|---|
| 1. "Last spring I started baking my own bread." | 12.4 to 17.0, 24.8 to 26.5, 31.0 to 36.2 | 1: complete. 2: broken off after "Last spring I started". 3: complete. |
| 2. "The first thing you learn is to wait for the dough." | 38.0 to 42.5, 47.3 to 51.9 | 1: flubbed ("the dou... dough"). 2: complete. |
| 3. "My first loaf took two days." | 55.0 to 57.8 | 1: complete. |

Also list two more things:

- **Said, but not in the script.** For each, the reel time and what it is: a slate ("okay, again" at 18.0), a start-over line, a joke, a whole thought, or talk to themselves.
- **In the script, but never said.** Every line you found no take of. Check twice before you call a line missing. The creator may have said it in other words.

Match takes by reading. `analyze_take`'s list is only a hint, and creators often reword the script, so a take is a take of a line when it covers the same beat, even in other words. A slate such as "redoing this" or "let me start over" is the best sign that the take before it is dropped.

Keep this map in your head and in the chat when you present it. Don't write it to a file.

## 5. Choose the takes

Read the `what-to-cut` skill before you choose, and again whenever the creator pushes back on a cut. It holds the judgment for what stays and what goes; this step holds the mechanics for takes.

Keep one take of each line.

**Default to the last complete take.** Creators retake because the earlier one did not land. The later one is usually the keeper. A question or lead-in that belongs to a line comes from the same take as the line.

Go to an earlier take only when the last one:

- is broken off, or
- is flubbed, or
- drops something the script needs.

Then take the latest complete take before it. If no take is complete, keep the cleanest one and say so in step 7. If two complete takes are close and you can't tell which is better, keep the last and tell the creator the other exists, so they can ask for it.

**Don't cut a laugh the creator got.** A cut from take 1 to take 3 removes everything between them, laugh included. If a `(laugh)` line follows the take you would drop, or that take has an ad-lib the later take lacks, keep that take, or keep the last one and show the creator the choice in step 7. Cutting a laugh away is the one edit they can't hear coming.

**Takes sit back to back.** A retake comes within a minute or so of the take it replaces, with slates and false starts between. The same line said minutes apart, such as a callback, or a line said in the intro and again at the end, is not a retake. Keep both and list it for the creator. Cut only what a retake replaces.

**Cut every other take, whole.**

- Start the cut at the first word of the dropped take. End it at the first word of the kept take. Use the start times printed on the transcript lines. Then the join lands on a sentence start, and the viewer hears one clean line. A line can hold more than one sentence, so a take may start inside it with no time printed. To get that time, call `read_transcript` with `start` set just before the take's first word and `end` a few seconds on: the window opens on the first word that starts after `start`, so the first line shows its time. Or call `find_words` on the take's first word: the id carries its start time.
- Everything between the two goes with it: a broken-off try, a slate, "let's start over." One cut covers it all.
- When you keep a take that is not the last, the cut runs from the first word of the take after it to the first word of whatever comes next.
- A cut that crosses a clip boundary is fine. The clips are one video now.
- A laugh or reaction that belongs to the line you keep can sit just inside the dropped take's side of the cut, right after the kept line's last word. Place the start of the cut after it. Look at the words and the sound labels either side of the start, and at the picture with `look` once the draft is saved.

In the example above, line 1 keeps take 3 (31.0 to 36.2). The cut runs from 12.4, the first word of take 1, to 31.0. That takes take 1, the broken-off take 2, and the "okay, again" at 18.0 in one go. Line 2 keeps take 2. The cut runs from 38.0 to 47.3.

**Give every cut a kind.**

| `kind` | When |
|---|---|
| `repeat` | a whole take said again and dropped |
| `false_start` | a take that broke off and was started again |
| `off_topic` | a tangent the creator doesn't want in the video |
| `other` | slates at the start of a clip, the reach for the camera at the end, dead air |

When one cut covers a whole take and a broken-off try, use `repeat`.

**Write each reason for the creator.** They read it on the page, next to the words you cut. One short sentence, in plain words: what they said, and why the other take stays. No editor shorthand.

| Good | Not this |
|---|---|
| You say this line a second time at 31 seconds, all the way through. I kept the second. | Superseded by later complete take. |

**Slates and "let's start over" lines go.** They were for you and the camera, not the viewer.

**Asides that are whole thoughts stay.** A joke, a story beat, a thing only this creator would say. If one sits between two takes, cut around it: two cuts, not one. List every aside you kept, so the creator knows you saw it. An aside that is only a half sentence goes with the take it sits in. One you can't tell from a slate stays, and goes on that list.

**Long silences go, and you list their places.** When the creator stops talking for several seconds and the recording keeps running, they usually left room for b-roll. Before you cut one, call `look` at its start and read the sound wave and the pictures. A silence can be a creator showing something to the camera, a laugh, or a sound you can't hear in the transcript. Keep any stretch where the words around it point at something shown ("look at this", "see"), or where the wave is loud with no word on it, and mention it to the creator. Cut the rest as `other`. Do this before you save the cut that removes it: with no cut near, `look` shows the plain footage, and once the silence is cut it shows the join and not what was inside. The creator needs to know where it was, so for each one keep a note of its reel time. After you save the draft in step 6, find where it falls in the edited video: call `look` with the silence's start time. The answer's `edited_at` is that place.

**Keep the order.** The reel plays in the order the clips were joined. Cuts take takes out. They don't move anything. If the takes you kept come out of script order, tell the creator. Offer to join the clips again in another order, or with a start and end on a clip. That makes a new reel. Call `carry_edit` from the old reel to the new one and the transcript and the cuts go with it; read its report for any cut it split or dropped.

## 6. Then pace

Now the video has one take of each line. The rest is the same job as a tight cut, so go to the tight-cut skill and follow it from its step 4.

- **Step 4, decide the cuts.** The retakes are done. Don't cut them twice. Look for what is left: filler runs, dead air, padding. Set `auto_tighten` and the `pace` from what the creator asked for. If they didn't say how tight they want it, ask for that and nothing else. `set_edit` takes a `pace` only together with `auto_tighten` on. With it off, a `pace` is an error, and there are no automatic trims at all. Read "Before you turn automatic trimming on" below first.
- **Step 5, pick the ums and the filler likes.** Call `find_words` with `["um"]` and read each line by that step's rules: an um that starts a sentence, or has about half a second of quiet or more on at least one side, goes. An um tucked between words with a short gap on both sides stays, and so does any you are unsure of. Leave some in. Your retake cuts are saved by now. A line that reads `OPENS A SENTENCE AFTER A CUT` is an um at the head of what a cut keeps: it goes. Then, only if the creator asked for likes to go, call `find_words` with `["like"]` and read each line. Check each pick against the script before you save it. A "like" that is part of a scripted line, such as "Like I could not just express myself", is not filler. Leave it in. If they didn't ask, leave every "like" in and say so.
- **Step 6, save a draft and check it.** Call `set_edit`. Read `rejected`, `adjusted`, `joins`, and `picks`. Fix what is flagged, up to three times.

**Before you turn automatic trimming on, check it against the laughs.** No automatic trim takes any part of a laugh or of a sound of a second or more that the transcript has a line for. A laugh with no line is a pause to it, and it can cut one the creator got on purpose. Do this with the list of laughs you made in step 3:

1. Save the draft with `auto_tighten` on and the pace the creator asked for. Call `get_edit` with `include_auto` set to true and compare the listed trims with your laugh list. It lists only the first 50 trims, and the answer's `auto.not_shown` counts the rest. When there are more, also call `look` for each laugh, at the middle of the pause that holds it, and compare the answer's `join` with the start and end of that pause. A trim on either end of the laugh shows there. Calling `look` at the laugh's own time is not enough, because `look` draws the cut nearest that time, and a trim at the far end of the laugh can lose to a nearer one.
2. Read the `joins` for `removes_laugh` and `clips_beat` too. They catch only the laughs the transcript labeled, and each carries a fix: ask the creator to keep that stretch on the page. Pass that fix on. Keep automatic trimming on, as the tight-cut skill does. Don't turn it off to protect a laugh.
3. For every laugh on your list that a trim sits on, label or no label, tell the creator in step 7 to press Keep on those spots on the review page, with the laughs' times, before they pick a pace there. Keep protects the stretch from the page's automatic trims. Picking a pace on the page without it can cut the laugh.

**Expect a few flags on your take cuts, and don't take all the fixes.**

- A flag of `long_jump` is expected on every retake cut. Twenty seconds or more are gone, because that's the point of the cut. The join check's suggested fix for it, such as "End at X to keep the last cut sentence as a bridge", would put a slate or a dropped take back in. Don't follow it. Do check the other flags on those joins.
- `set_edit` may move a cut's edge to a word boundary or into the quiet next to a word. It reports this under `adjusted`, with the time you asked for and the time it used. The server does this so cuts never land inside a word. The new edge can still swallow a laugh or a small sound next to it. After you save, re-read around each take cut with `read_transcript` (with `show_cuts` true) and `look`, and check that the moved edge didn't take a laugh or a sound that belongs to the line you kept. If it did, move your edge to leave it, and save again.
- Overlapping cuts are merged. A take cut that touches an automatic trim becomes one cut, and the pause left at its join can shrink. A `tight` flag on a join of your own means that happened. Give the cut an edge that leaves about 0.4 seconds of quiet.

**Look at the start and end of every clip and every kept take.** Creators lean into the lens to start and stop the camera, so the first and last second or two of a clip is often a blurred face, a hand, or handling noise. A retake often starts the same way, and so does a kept take. You can't hear any of this, so use `look`. After the draft is saved, call it at each clip's `reel_start` and `reel_end` from `pieces`, and at the join before and after each kept take. The join picture shows the last kept frame before the cut and the first after it. If a frame is a blurred face, a hand across the lens, or the sound wave shows a bump with no word on it, move the edge of the cut past it (or add a cut of your own, kind `other`). Leave a little quiet before the first word.

**Send the take cuts and the pace cuts in one `set_edit` call.** `set_edit` replaces the saved list of your cuts each time. If you save the take cuts first and then call `set_edit` with only the pace, the take cuts are gone. Every call carries the whole list: the take cuts, the silences, and anything else of yours. Their own cuts stay by themselves.

On joins that sit on a clip boundary, call `look` once. The room sound or the light can change from one clip to the next.

## 7. Present the plan in chat

Show these, in this order:

1. **The take map** as a table, one row per script line, with these columns: the line, the takes found, the take kept, and why. Name a take by its clip and its first words, such as "clip 2, 'Last spring I started'". Use the `clock` value from the cuts for any time you show. Never work out minutes and seconds yourself.
2. **Every other cut that removes words**, numbered and shown the way `what-to-cut` shows a change.
3. **The lengths.** The reel's length and the new length.
4. **The asides you kept**, each in a few words with its time.
5. **Script lines that were never said**, if any.
6. **Where the silences were**, if you cut any. For each, the reel time and its place in the edited video, so the creator knows where to put b-roll.
7. **Any join still flagged**, in plain words, and a `long_jump` on a retake cut only as "expected": "The cut at 4:12 ends mid-sentence. I kept it since the next take restarts the thought. Listen to that one."
8. **The ums and the likes**, in a sentence or two each. For the ums: how many you found, how many go, and that the rest are the ones inside sentences, left on purpose. For the likes: that every one is left in, or, if they asked for likes to go, how many times the word is said, how many you picked as filler, and how many stay in.
9. **Automatic trimming and the laughs**, if a trim sat over a laugh: the laughs' times, and that they should press Keep on those spots on the page before they pick a pace there.

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

On a yes, call `review` with the reel path and give them the address with these steps:

1. Press Next to go to each cut in turn, then Play to hear it. The bar under the video shows where the most changed; press a block on it to play that stretch.
2. If it feels rushed or drags, move the pace. For a setting between two stops, open Fine tune under it and move the two sliders. Under Take out, turn off anything you want left in. Ums and likes switches off the ums and likes I picked.
3. Open my cuts by kind. The retakes I cut are under Said twice and False starts. Put one back only if you want both takes in the video.
4. A word you want gone? Double-click it. Double-click a struck word to bring it back. Drag across words to cut or keep a part.
5. Found a moment that must stay? Select it and press "Keep".
6. Every change saves by itself. When it sounds right, press Export video, or come back and say you're done.

Say once that if they'd sooner use a different take of a line, they can tell you here and you'll change the cuts.

If they'd sooner answer in chat, skip the page.

## 8. Take the creator's answer, then render

Follow the tight-cut skill's steps 8 and 9. Two things differ here.

- **In chat.** "Use the first take of line 3" means you change the cuts: drop the cut that removed it, cut the take you had kept, and call `set_edit` with the whole new list. Check `joins` again and show what changed.
- **Export.** The finished video lands in the reel's own project folder, under `exports/`. The clips are never touched.

If the creator wants a different order of clips, or another clip added, that is a new `assemble` call. It makes a new reel, and a new reel is a new project. Call `carry_edit` from the old reel to the new one: the cuts, the picks, the creator's own changes and the transcript go with it, and its report names every cut it split, shortened or dropped. To put the creator's other clips in where the talk calls for them, follow the `splice-clips` skill.

## Handling tool errors

Read the error message. It says what was wrong. Fix the call and retry once. If it fails again, tell the creator plainly what's wrong and what you'd need from them. Don't retry silently more than once, and don't guess at a fix you can't explain.

For `assemble`, the usual causes are a clip path that is wrong or has moved, a file that isn't a video, a clip with no sound, and a start and end that fall outside the clip. Ask the creator for the right path or numbers.
