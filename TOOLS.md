# Lumr Studio tool contract

The agreement between the MCP server and the skills. The server implements
these tools. The skills tell Claude how to use them. Change this file first,
then the code.

Scope: one video, from recording to export. The plugin offers the editor
slice by default (see "What the plugin offers"); the publish kit and the
overlays are extras, switched off.

Round 2 adds ways for Claude to check its own cuts (`joins` on every saved
edit, `look`) and one place for the creator to review them (`review`).

Round 3 turns that place into the treatment page: the creator sets the pace
and what to take out, puts back or leaves out each of Claude's cuts, flags
parts to keep no matter what, and listens to three samples. Each of Claude's
cuts now carries a `kind`, and `get_edit` reports what the creator changed.
Round 3 also adds overlays: the creator's own photos and clips shown over the
talk (`set_overlays`, `get_overlays`), drawn by `render` and `preview`.

Round 4 makes the page simpler and lets the creator export from it. Export
video on the page starts the same full render `render` starts, as the same
kind of job, so `job_status` reports on it. `get_edit` names that job under
`export`. One full render runs per video at a time.

Round 5 makes the pace cut as hard as the Mac app and lets the creator cut by
hand. Word times are measured from the sound (aligned), in the same job as
transcribing. The pace has six stops. On the page the creator knocks out any
word and brings any removed word back. Her cuts are her own, kept apart from
Claude's, and `get_edit` reports them.

The second wave of round 5 adds two things. Fine tune: two sliders under the
pace, for the shortest pause cut and the rhythm; a setting made with them is
the pace `custom`. Filler likes: `find_words` lists every place a word is
said, Claude picks the filler ones by reading and hands them to `set_edit` as
`picks`, and the creator has one switch for all of them.

Words. A **cut** is a choice with a reason, Claude's or the creator's own: the
page and the chat both say "cut". A **trim** is one of the small automatic
ones: a long pause, a filler word, a stutter. The tools and this file say
"trim"; the page no longer does, and neither should anything the creator
reads. To the creator, name what goes: long pauses, filler words, stutters.
**Edits** means all of them.

## What the plugin offers

By default the server lists 11 tools: `transcribe`, `read_transcript`,
`analyze_take`, `find_words`, `get_edit`, `set_edit`, `preview`, `render`,
`look`, `review` and `job_status`. Four more are extras, off unless the
environment variable `LUMR_STUDIO_EXTRAS` names them, as a comma list:

| Extra | Adds | Skill that goes with it |
|---|---|---|
| `publish_kit` | `chapter_times`, `save_publish_kit` | `publish-kit` |
| `overlays` | `set_overlays`, `get_overlays` | `add-visuals` |

With `LUMR_STUDIO_EXTRAS=publish_kit,overlays` the server lists all 15. A name
that is not an extra stops the server at start. An extra's skill folder sits
in `extras/skills/`, outside the folder Claude Code scans, and comes back with
an entry in `plugin.json` (`README.md`, "Turning the extras on"). The overlay
code stays linked and dormant: with no saved overlays `render`, `preview` and
the page draw nothing over the talk. `server/lumr_studio/offering.py` holds
the list, and a test holds every offered skill to it: a skill in view never
names a tool that is off. The tool sections below are the contract for the
extras too, for when they are on.

## Rules for every tool

- Every tool takes `video_path`, an absolute path to the source video, except
  `job_status`.
- The source video is never modified or overwritten.
- Results are short. A tool returns a summary and file paths. It never returns
  a whole transcript.
- Failures return an error that says what was wrong and how to fix it, so
  Claude can correct the call and retry.
- Times are seconds in the SOURCE video, as floats, unless a field name says
  otherwise.
- Long work returns a `job_id` at once. `job_status` reports on it.
- Every tool carries `title`, `readOnlyHint`, and `destructiveHint`.
- No tool deletes a file of the creator's. The server removes only its own
  files: a half-written render that failed, and the in-between encode in
  `work/` once the overlays are drawn.

## Project folder

One per video, created on first use:

```
<LUMR_HOME>/studio/projects/<video stem>-<first 12 hex of sha1 of the resolved path>/
  edit.json          the saved edit, with the pace and switches it was built with, and Claude's picks
  words-aligned.json the transcript's words with times measured from the sound, each word
                     with room for its sound
  plans.json         the automatic trims planned for each pace, kept so the page opens fast:
                     the six stops, and the last 12 settings tried with the sliders
  treatment.json     the page's three samples and the stretches used before
  overlays.json      the creator's photos and clips over the talk
  review.json        the creator's decisions from the round 2 review page, if any
  sounds.json        which sounds in the transcript are likely laughs
  receipts.jsonl     one line per completed step
  exports/           rendered videos and previews
  work/              the edit's encode while overlays are drawn over it; emptied after
  looks/             pictures of joins made by `look`, and the stills `set_overlays` draws
  review/            the page's cached sound envelope, and stills from its frame route
  publish-kit/       title, description, chapters, tags
```

`overlays.json` sits beside `edit.json` and is independent of it. `set_edit`
never touches it, and `set_overlays` never touches the cuts.

The projects root also holds `usual.json`, the creator's usual pace and
switches, one file for every video.

`LUMR_HOME` resolves the way ClipForge resolves it. The environment variable
`LUMR_STUDIO_PROJECTS_DIR` overrides the projects root. Tests must set it.

The transcript stays where the speech model writes it, beside the video, as
`<video>.words.json`. The measured word times are the plugin's own and stay
in the project folder.

## Word times

Every tool that reads words gets them from one place, so the automatic trims,
the join check, Claude's cuts, the creator's cuts, keeps, samples, the packed
transcript and the words on the page always agree.

| `word_times` | The words' times come from | What it means |
|---|---|---|
| `measured` | aligning each word to the sound | the pauses between words are real, and every pace reaches what it should |
| `estimated` | the transcript as the speech model wrote it | most words touch their neighbours, so the quiet between them is hidden inside the words. The engine never cuts inside a word, so the harder paces find far less. |

On the 21 minute test take, `max` removes 402 s on measured times and 141 s on
estimated ones.

A measured word has room for its sound. The aligner's own times are tight:
it ends a word at the first measured silence after its start, and the closed
mouth before the "t" of "take" is such a silence. On the test take 29 of
3,231 words came out under 5 ms long, and 244 s of sound sat under no word.
The engine never cuts inside a word, but a word 1 ms long protects 1 ms. So
before the times are saved each word is given the sound that belongs to it
(`word_times.with_room`):

- Sound runs on. A word keeps the sound that follows its end and the sound
  that leads into its start, through any quiet shorter than 0.15 s, the
  shortest pause any pace cuts.
- A word the aligner could not place keeps the room the transcript gave it,
  from its first sound to its last. That is a word shorter than one frame of
  the aligner for each letter, 20 ms a letter: "take" under 80 ms. The
  aligner gives every letter a frame, so a shorter word was cut short after.
- No word grows past its neighbours' measured edges or into a sound of the
  transcript, so a laugh stays whole.

On the test take 2,448 of 3,231 words were given room. 429 had been cut
short, and 6 still are: they sit against a neighbour on both sides. The
words cover 802 s, up from 516. A word that was given room keeps the
aligner's times beside its own (`aligned_start`, `aligned_end`).

Aligning needs torch, torchaudio and one model file of about 380 MB:
torchaudio's `WAV2VEC2_ASR_BASE_960H` (wav2vec 2.0 base 960h, MIT), kept at
`<torch hub>/checkpoints/wav2vec2_fairseq_base_ls960_asr_ls960.pth`. It
replaced `MMS_FA`, whose weights are non-commercial, and the plugin never
loads `MMS_FA`. Aligning never downloads the file. `transcribe` fetches it,
once the creator says yes (see `transcribe` below). On a machine without it
the plugin runs on estimated times and says so: every answer that reads words
carries `word_times`, and `word_times_note` says why when it is `estimated`.

Aligning takes about a minute for a 20 minute video. The text of the
transcript never changes, only the times. A sound (a laugh, a breath) keeps
the time the transcript gave it, less any part a measured word turns out to
sit in; a sound that was words all along is dropped.

When the word times change under a saved edit (aligned for the first time, or
the video transcribed again), the edit is placed again on the new times, once.
Going from estimated to measured, every cut and kept part moves with the words
it holds, so it holds the same words as before. Row ids come from placed
edges, so ids of cuts change: read `get_edit` again before changing a cut.

Measured times saved before words had room need no measuring again. The
first tool that reads them gives each word its room and saves the file. An
edit placed on the earlier times moves with its words the same way: Claude's
cuts, the creator's cuts, the words she brought back and Claude's picks hold
the words they held. The automatic trims are planned again. On the test
take the edit got 0:51 longer at `fast`: the sound of words that had gone
with the pauses, and the "like"s the pace no longer takes.

## Tools

### transcribe

Read only: no. Destructive: no. Long job.

| Input | Type | Note |
|---|---|---|
| `video_path` | string | |
| `force` | bool, default false | re-transcribe even if a transcript exists |
| `download_models` | bool, default false | fetch the models the last answer said were missing. Pass true only after the creator said yes. |

One job transcribes and measures the word times, so Claude waits once.

| Answer | When |
|---|---|
| `{status: "needs_models", models, total_mb, note, ...}` | a model this call needs is not on the machine. Nothing has downloaded and nothing started. See "Models". |
| `{status: "downloading", job_id, models}` | `download_models` was true and a model was missing. The job fetches it. See "Models". |
| `{job_id}` | there is no transcript yet, or `force` is true. The job does both. |
| `{status: "aligning", job_id, ...}` | the transcript is there and its word times are not measured yet. The job measures them. |
| `{status: "exists", words_path, word_count, duration, word_times, ...}` | nothing to do. With `word_times: "estimated"` this machine can't measure them, and `word_times_note` says what is missing. |

The finished job's result carries `word_times`, and with `measured` also
`aligning: {seconds, words_moved, saved_to, sounds, sounds_that_were_words}`. When a saved edit was placed
again on the measured times it carries `edit: {before, now, note}` with the
cut count and time removed before and now. Tell the creator when they differ:
the measured times show pauses the estimates hid, so the same pace removes
more.

A job that could not measure the times still ends `done`, on estimated times,
with `word_times_note` saying why. Go on with the edit and tell the creator
in one sentence that the harder paces will do less.

#### Models

Two models are needed, and neither ships with the plugin.

| Model | Needed when | From | Size | License |
|---|---|---|---|---|
| NVIDIA Parakeet TDT 0.6B v2, the speech recognition | there is no transcript, or `force` is true | `huggingface.co`, `mlx-community/parakeet-tdt-0.6b-v2` at one pinned commit | 2.47 GB | CC-BY-4.0 |
| wav2vec 2.0 base 960h, the word timing | the word times are not measured yet | `download.pytorch.org`, through torchaudio's own file name | 0.38 GB | MIT |

Nothing downloads until the creator says yes. The flow, in one tool:

1. `transcribe` finds a model missing and answers `needs_models`:
   `models` is a list of `{name, size_mb, source_host, license, folder}`,
   `total_mb` adds up the sizes as listed, and `note` tells Claude what to do.
   When the transcript is already there and only the word timing model is
   missing, the answer also carries the transcript's fields
   (`word_count`, `duration`, `word_times: "estimated"`, and so on) so Claude
   can go on with the edit if the creator says no.
2. Claude tells the creator what would download, how big, from where and under
   which license, and asks. Claude asks before it calls anything again.
3. On a yes, Claude calls `transcribe` with `download_models: true`. The answer
   is `{status: "downloading", job_id, models}`, and the job's kind is
   `download_models`. `job_status` reports it like any long job, with
   `progress` from 0 to 1 over the bytes of every file still to fetch.
4. The finished job's result carries `models_downloaded` (names) and `next`.
   Claude calls `transcribe` again with the same `video_path`. It goes on as
   if the models had been there.

A second call with `download_models: true` while the job runs answers with the
same `job_id`. A call with it true when nothing is missing downloads nothing
and goes on. Downloading is the only thing in the plugin that fetches a model:
aligning refuses to, and so does the speech model, which reads its files from
the pinned folder with Hugging Face set offline.

How a file arrives. Each file is pinned by exact size and sha256. It is
written to `<name>.part`, checked, then renamed, so a file at its real name is
whole and was checked. A `.part` left by a stopped download is resumed from
its last byte (or started again when the server won't resume). A file that
fails its checksum is deleted and the job fails with the reason. Nothing is
installed, and Claude can offer to try once more. A stopped connection fails
the job and keeps what arrived. Not enough free disk fails it before the first
byte. Files go to the caches other tools share and survive an uninstall:
Hugging Face's (`~/.cache/huggingface/hub`, or `HF_HOME`) and torch's
(`~/.cache/torch/hub/checkpoints`, or `TORCH_HOME`).

#### How a word keeps its sound

- A word keeps its soft start and end. Silences are measured at -25 dB, then
  once more at -40 dB. Soft sound runs on up to 0.3 s after a word and
  0.15 s before it.
- Where the transcript heard a sound inside the word, the word runs on
  through quiet under 0.25 s too, when quiet plus sound is 0.5 s or less.
- A word that runs into the next ends where the next starts. This covers
  ClipForge's 40 s alignment seams, where the later word is right.
- A word's soft end also runs into a sound heard starting inside it.

The result's `aligning` carries `words_given_soft_room`, `soft_room_seconds`
and `soft_room_share`. `word_times_warning` appears when the soft room misled
the measure. Tell the creator once.

### read_transcript

Read only: yes.

| Input | Type | Note |
|---|---|---|
| `video_path` | string | |
| `start` | float, optional | window start |
| `end` | float, optional | window end |
| `show_cuts` | bool, default false | mark words the saved edit removes |

Returns packed text, one phrase per line, split at pauses:

```
[12.40-15.85] so the first thing i want to say is
  (pause 0.9)
[16.75-19.20] um you don't have to have it figured out
```

Sounds that aren't speech appear on their own line, labeled by kind:

```
[478.10-482.60] okay, the first rule of sourdough is to never trust the timer.
[482.70-486.22] (laugh 3.5s)
[92.04-92.28] (sound 0.2s)
```

`(laugh 3.5s)` is a likely laugh. `(laugh? 1.6s)` is a possible one. `(sound
0.2s)` is anything else: a breath, a hesitation, a bump. The labels come from
the sound's length, its place in the sentence, and its measured shape. They
are estimates. A laugh marks the line before it as a punchline.

With `show_cuts`, removed phrases are prefixed with `CUT`.

Windows are contiguous and never overlap. Every response ends with one of two
lines, so the reader always knows whether more remains:

- `NEXT <time>` when the size budget cut the window short. Pass that time as
  `start` to continue.
- `END` when the response reached the last word of the transcript.

### find_words

Read only: yes.

| Input | Type | Note |
|---|---|---|
| `video_path` | string | |
| `words` | list of string | one to five words or short phrases, such as `["like"]` or `["you know", "so"]`. A phrase is at most four words. |
| `start` | float, optional | go on from a `NEXT <time>` line |

Returns every place each word is said, as packed text, one place a line:

```
like: said 112 times, 94 clean to cut, 18 not, 11 already out.
Each line: id, clock, the words before, [the word], the words after, q<quiet before>/<quiet after> ...
w34.336-34.458 0:34 wherever you are in the process, [like,] or even like you're just curious, q.76/.32 OUT:claude
w245.263-245.483 4:05 it would be [like] if you q.12/.02 NOT CLEAN: cutting it would clip the next word
END
```

| Part of a line | Meaning |
|---|---|
| `w34.336-34.458` | the id: pass it to `set_edit` in `picks` |
| `0:34` | the clock time, ready to show |
| six words, `[the word]`, six words | enough to tell a verb from a filler |
| `q.76/.32` | seconds of quiet before and after the word |
| `OUT:<who>` | the saved edit removes it already: `pace`, `claude` (one of Claude's cuts), `creator`, or `pick` (picked before). The pace never takes "like", so a "like" never reads `OUT:pace`. |
| `NOT CLEAN: <why>` | a cut there would clip the word beside it. `set_edit` leaves such a pick in. |

The result also carries `words: [{word, said, clean, already_out}]`,
`listed`, `next_start` and `word_times`. 112 places take about 13,000
characters, so they fit in one answer. A longer list ends with
`NEXT <time>`.

How `clean` is judged. A cut is clean when each side has quiet between the
word and its neighbour: at least 0.03 s between the two as the aligner
placed them, or a measured silence at that edge. Closer than that the two
words run into each other, and a cut takes the end or the start of the
neighbour with it. A "word" longer than a second is an aligning fault and is
never clean. On the test take 94 of 112 "like"s are clean on measured word
times, and 28 of 112 on estimated ones.

The quiet on each line (`q`) is what lies between the word's room and its
neighbour's, so it is often `.00` where the sound runs from one word into
the next. `clean` reads the aligner's own edges, which say whether it could
tell the two words apart. The cut takes the word's whole room.

The tool takes any word. This round the page's switch and the skill cover
"like".

### analyze_take

Read only: yes.

| Input | Type | Note |
|---|---|---|
| `video_path` | string | |

Returns a compact report: duration, word count, speaking rate, the longest
pauses, filler words with times, repeated phrases that look like retakes,
`word_times`, and `paces`: for each of the six pace levels, how many trims it
would make on this take and the time they would save.

### get_edit

Read only: yes.

| Input | Type | Note |
|---|---|---|
| `video_path` | string | |
| `include_auto` | bool, default false | also list automatic trims one by one (first 50) |
| `check` | bool, default false | also return the `joins` self-check |

Returns Claude's saved cuts with reasons and kinds, a count of automatic
trims (`auto`), the pace, the total removed time, the resulting duration, and
`word_times`. Each cut carries `clock`, its source times as `m:ss-m:ss`, ready
to show the creator. The creator's own cuts are never in `cuts`; they are
under `creator`.

When the creator changed something on the page, the result carries `creator`:

```
{pace: {claude: "standard", creator: "fast"},
 take_out: {fillers: false},
 put_back: [{clock, start, end, reason}],
 kept: [{clock, start, end, text, note}],
 brought_back: [{clock, start, end, text}],
 cuts: [{clock, start, end, text}],
 cut_words: [["like", 11], ["so", 3], ["you know", 2]],
 fine: {gap_length: 0.35, rhythm: 3.5, speech_kept_between_cuts: 0.85, note}}
```

Each key is there only when it applies.

| Key | What the creator did |
|---|---|
| `pace`, `take_out` | moved away from the pace or switches Claude asked for. `take_out.likes: false` means they switched Claude's picked filler words off. |
| `fine` | set the pace with the two sliders; `pace` reads `custom`. Listed for as long as the pace is custom. The values are hers: leave `pace` and `gap_length` out of `set_edit` to keep them. |
| `put_back` | put back one of Claude's cuts. Carries Claude's reason. |
| `kept` | flagged a part "keep no matter what". It widened to whole sentences. Carries the words and their note. |
| `brought_back` | brought back exactly these words, one by one, from a cut of Claude's or an automatic trim |
| `cuts` | cut these words by hand |
| `cut_words` | what her cuts say, counted, most first. A cut of one or two words counts as what it says; a longer cut counts each word. |

Read it before proposing anything new: every entry is the creator saying what
they want. Once Claude calls `set_edit` with their pace and switches, those
stop showing as changes. Everything else stays listed: it lasts.

Her cuts are settled. Never send one as a cut of Claude's, never offer to
undo one, and never count its time as Claude's saving. Learn from
`cut_words`: eleven "like"s cut by hand says she wants fewer of them, so say
what you noticed and offer what would take more of them.

When Claude has picked filler words, the result carries `picks`:

```
{switch_on: true, picked: 41,
 picks: [{id, clock, text, reason}]}
```

`switch_on` is the creator's switch on the page. With it off no pick is
taken out, and the list stays saved.

When the creator used the round 2 review page on this video, the result
carries `review`: `{accepted, restored: [...], changed: [...],
waiting_for_apply}`. That page is gone; new projects never have it.

When a full render of this video has run since the server started, the result
carries `export`:

```
{job_id: "render-3f9c2a71b0de", status: "running" | "done" | "failed",
 edit_changed_since: false}
```

It is the latest full render, whoever started it: `render`, or the creator
pressing Export video on the page. Pass `job_id` to `job_status` for the
progress, the output path, or the error. `edit_changed_since` is true when the
saved edit now removes something else than that render holds, so its file is
behind the edit. Jobs live in memory: after a server restart `export` is gone,
and the file is still in `exports/`.

### set_edit

Read only: no. Destructive: no. Replaces the saved edit for this video.

| Input | Type | Note |
|---|---|---|
| `video_path` | string | |
| `cuts` | list of `{start, end, reason, kind}` | every cut needs a reason, one plain sentence the creator reads on the page; `kind` is optional |
| `auto_tighten` | bool, default false | also make automatic trims across the whole video: long pauses, filler words, stutters. False makes none. |
| `pace` | string, optional | how hard `auto_tighten` trims: `natural`, `standard`, `fast`, `tight`, `hard` or `max`. Left out: the pace already set for this video, else the creator's usual, else `standard`. |
| `gap_length` | float, optional | overrides the pace's shortest trimmed pause, 0.15 to 1.5. Usually left out. |
| `override_keeps` | bool, default false | allow cuts in spans the creator kept (a cut put back, or a part flagged to keep). Only when the creator asks. |
| `picks` | list of `{id, reason}`, optional | single filler words Claude picked by reading, by their ids from `find_words`. Left out: the picks saved before stay. A list replaces them. An empty list clears them. |

Each cut's `kind` tells the creator what sort of cut it is. The page groups
Claude's cuts by it:

| `kind` | Use for |
|---|---|
| `repeat` | the creator says again what was already said: a restated point, a second take |
| `false_start` | a sentence started, abandoned, and restarted |
| `off_topic` | a tangent, a plug, an aside that doesn't serve the video |
| `other` | anything else: dead air, a cough, a throat clear |

A cut sent without `kind` is `other`. A kind not on the list rejects that
cut. The kind comes from Claude only; the server never guesses it from the
reason.

The switches the creator set on the page stay when Claude calls `set_edit`
with `auto_tighten`: if the creator turned off filler words, they stay off. A
video with nothing saved starts from the creator's usual.

What the creator did by hand stays too, whatever `cuts` holds: the cuts she
made, the parts she kept, and the words she brought back.

A pace the creator set with the sliders stays when `pace` is left out.
`pace: "custom"` is refused: her setting can't be picked by name. So is
`gap_length` while her setting stands. Passing one of the six stops
replaces her setting; do that only when she asks for a pace.

#### Picks

`picks` are kept apart from `cuts` on purpose. A cut is a row on the page,
with a reason the creator reads and a put back of its own. A picked filler
word is none of that: there can be dozens, so the creator has one switch for
all of them (Filler likes) and a double-click for each.

- Each pick is placed the way the creator's own word cuts are: the word's
  own span, widened into the quiet beside it when under 0.1 s, never into
  the word beside it. No pick is refused for being short.
- A pick that is not clean is not cut. It stays in the list, and the answer
  names it under `left_in` with why.
- A pick on a word the creator brought back, or inside a part she kept, is
  not cut: what she did by hand wins. The answer names it under
  `kept_by_the_creator`.
- The pace never takes a word that is judged by reading, which today is
  "like" (`autocuts.JUDGED_BY_READING`). So a picked "like" is out while
  Filler likes is on and plays while it is off, at every pace. A pick
  beside a pause the pace removes is one removal with it, counted as the
  pick.
- An id that names no word as the transcript is timed now is refused, with
  the reason, and the rest are saved. When every pick is refused nothing is
  saved. Ids change when the word times do: call `find_words` again after
  `transcribe` measured them.
- The switch is the creator's. No call turns it on or off.

With `picks` the result carries:

```
picks: {picked: 41, switch_on: true, out: 35,
        left_in: [{id, clock, text, why}],
        kept_by_the_creator: [{id, clock, text}],
        rejected: [{index, pick, why}]}
```

`out` is counted with the switch on, whether it is on or off.

The server validates before saving:

- start is before end, both inside the video
- overlapping cuts are merged
- cut edges are moved clear of spoken words
- a cut that can't be made safe is rejected, with the reason
- a cut inside a span the creator kept (a cut they put back, or a part they
  flagged to keep) is rejected, unless `override_keeps` is true
- a cut over a word the creator brought back splits around that word: the
  rest is cut, the word plays. With every word of it brought back the cut is
  rejected like one in a kept span. `override_keeps` cuts the word after all.
- automatic trims steer around a marked punchline and the laugh after it, but
  can still clip one, most at the harder paces (10 places at Standard and 23 at
  Max on the test take). Those come back in `joins` as `removes_laugh` or
  `clips_beat` rows on an automatic trim, with a fix that asks the creator to
  keep that stretch on the page

Returns `{applied: [...], adjusted: [...], rejected: [...],
removed_seconds, new_duration, joins}`. Each adjusted entry shows the
requested and final times. With `auto_tighten` the result also carries
`auto`: the pace, `fine` (the two values behind it), the kinds taken out, and how many trims were planned,
shortened, dropped, or skipped to protect a joke or a kept span. `cleared`
is true when an empty list cleared the edit. `word_times` says what the cuts
were placed on. `creator_cuts_kept` counts the creator's own cuts that stayed
in the edit. An automatic trim shorter than 0.1 s is dropped, and
`auto.edges_moved_out_of_words` counts the cut edges moved out of a word so
no automatic cut starts or ends inside one. `warning` appears when the transcript itself was written with
made-up word times (every word in a sentence the same length); that is rarer
and worse than `word_times: "estimated"`, and the fix is `transcribe` with
`force`.

#### Pace

One level sets which pauses get trimmed, how far apart cuts must be, and the
pause left at the join. Seconds:

| Pace | Trims pauses longer than | Speech kept between cuts | Left between sentences | Left at a comma | Left inside a clause |
|---|---|---|---|---|---|
| `natural` | 1.0 | 2.0 | 0.6 | 0.4 | 0.3 |
| `standard` | 0.6 | 1.2 | 0.4 | 0.25 | 0.18 |
| `fast` | 0.4 | 0.5 | 0.25 | 0.15 | 0.1 |
| `tight` | 0.3 | 0.25 | 0.18 | 0.1 | 0.06 |
| `hard` | 0.2 | none | 0.12 | 0.06 | 0.03 |
| `max` | 0.15 | none | 0.1 | 0.05 | 0.03 |

`natural` reads as one unbroken take. `standard` is the usual talking-head
cut, and where a video starts when the creator has no usual saved: the first
real test went well at standard. `fast` is quick and punchy. `tight` takes
the small pauses inside sentences too. `hard` leaves almost no pause between
sentences, and cuts can land a few words apart. `max` cuts every pause the
engine can reach and still leaves a very short one at each join; some joins
will sound choppy.

#### The creator's own setting

Under the stops the page has two sliders, the two the Mac app has:

| Slider | Range | Step | Sets |
|---|---|---|---|
| Cut pauses longer than | 0.15 to 1.5 s | 0.05 | the shortest pause that gets cut |
| Speech kept between cuts | rhythm 1 to 5, shown as 3 s down to 0 s | half a rhythm | how far apart two cuts must be |

A setting made with them is the pace `custom`. It is no seventh stop:
`paces` in every answer stays six.

- The pause left at each join lies on a straight line between the two stops
  whose pause lengths sit either side of hers. At 0.35 s, between `fast`
  (0.4) and `tight` (0.3), it is 0.215 s between sentences, 0.125 at a
  comma, 0.08 inside a clause. At or past either end it is that end's.
- The filler list is the nearest by pause length, the way ClipForge picks
  it: 0.15 to 0.25 the longest list, 0.3 to 0.4 the middle one, 0.45 to 0.75
  the short one, from 0.8 up none.
- "Make this my usual" saves it, and a video edited later starts from it.

On the test take: 0.5 s with 0.85 s kept gives 13 cuts a minute and 3:29
out; 0.35 s with 0.5 s kept gives 19 and 4:56; 0.25 s with none kept gives
24 and 5:51.

On the 21 minute test take, on measured word times, cuts each minute and
time removed: natural 3 and 1:23, standard 9 and 2:48, fast 18 and 4:41,
tight 22 and 5:31, hard 27 and 6:24, max 32 and 6:42.

Automatic trims come in three kinds, each a switch on the page: long pauses
(`pauses`), filler words (`fillers`), and stutters (`repeats`, a word said
twice in a row). The result's `auto.take_out` lists the kinds that were on.

The filler switch is named for what it removes. An "um" is rare; "and",
"so", "you know" are what it mostly takes, and only where a pause sits
beside the word, or the word is doubled. `natural` has no filler list, so
the switch does nothing there. The page lists the words the switch takes at
the current pace, with counts.

It never takes "like". No rule tells a filler "like" from a real one: on the
test take the pace took 10 at `standard`, and "not feeling like I know
the way" and "ask what it was like." lost theirs. So "like" is
judged by reading (`autocuts.JUDGED_BY_READING`): Claude reads each one and
picks the fillers, and the creator has the Filler likes switch for those. A
rule and a judgment never share a word. ClipForge still lists "like" as a
filler and plans removals for it; the plugin shortens each one around the
word.

What a trim may take (`autocuts.trims_taken`):

- A word that no switch owns is never taken. A trim that lands on one is
  shortened around it. A word said quietly counts as silence to the
  detector, so a trim planned for a pause can land on a word.
- A filler belongs to the filler switch, whatever its trim was named for.
  With the switch on, a long pause takes the "and" that sits in it. With
  the switch off, the trim is shortened around the word, and the pause
  either side of it still goes. A filler is a word on the list ClipForge
  reads at that pace, so `standard` takes "and" and "um" and never "so".
- The first of a word said twice belongs to the stutter switch the same
  way, from `fast` on.
- A word judged by reading belongs to no switch of the pace, said once or
  twice. A trim that lands on a "like" is shortened around it at every
  pace, and the pause beside it still goes, down to what the pace leaves.
  On the test take "like" is said twice in a row in two places, and one of
  them is the end of a sentence and the start of the next.

So the filler switch's count and list hold every filler the pace takes. The
count is by the word. The time beside each switch adds up with no second
counted twice: a filler inside a long pause gives its own length to the
filler switch and the rest to long pauses. On the test take the switch holds
9 words at `standard`, 44 at `fast` and 85 at `max`, and with it off no
filler goes at any pace.

A trim is filed by what it takes. One that ClipForge named for a filler and
that takes no filler, because it landed on the pause beside the word, is a
long pause and answers to that switch.

A trim is shortened until the pause it leaves meets the level. Pauses are
measured from the audio as well as the word timings. The result's `auto`
says how many trims were shortened or dropped for this.

The numbers come from editing tools' defaults, editors' guidance, and
measurements on real footage. See `docs/14.07 Cutting rhythm.md`.

#### The `joins` self-check

A join is the point where the video skips a cut: the last kept word before
it now sits next to the first kept word after it. `joins` reads every join
the way a viewer hears it and lists the ones likely to sound wrong.

```
{joins: 166, flagged: 7, clean: 159, by_flag: {...}, not_listed: 0,
 text: "#12 412.30-431.80 -19.5s claude [mid_sentence_in, long_jump]
   ...and that's when I knew. | The first thing is...
   fix: <what to change, with times>"}
```

| Flag | Meaning |
|---|---|
| `mid_sentence_out` | the viewer hears a sentence that never finishes |
| `mid_sentence_in` | the viewer hears a sentence with no beginning |
| `splice` | both sides are mid-sentence. Right for a false start, wrong otherwise. |
| `removes_laugh` | the cut takes a laugh, its punchline, or the pause between them |
| `clips_beat` | a trim shortens the pause around a punchline or laugh |
| `re_entry` | what follows opens with a word pointing back at something removed |
| `long_jump` | 20 seconds or more removed. The viewer may lose the thread. |
| `fragment` | under 1 second or 3 words kept between two cuts |
| `tight` | under 0.10 seconds of silence left between the joined words |

A flag is a reason to look again. Some flagged joins are right as they are.

Each join's `source` says who made the cut: `claude`, `you` (the creator, by
hand), `pick` (a filler word Claude picked) or `auto`. A pick is judged as
lightly as an automatic trim. The creator's cuts are checked the way Claude's are, so one
can be flagged. Tell her what the check found if she asks; the cut is hers
to keep. Fixing a flag on one of her cuts is never a reason to change it.

Each flagged join is written twice, for two readers. The `fix:` line in
`text` (the row's `note`) is for Claude: it names source seconds to pass back
to `set_edit`. The treatment page shows the creator a plain `why` instead,
such as "The sentence before this cut never finishes.", with no times, since
all she can do there is put the cut back or leave it out. A `why` never says
"trim": for an automatic one it says "An automatic cut". A cut of hers that
takes a word out of the middle of a sentence reads "This takes words out of
the middle of a sentence. Hear it once to be sure it still sounds whole."

### preview

Read only: no. Destructive: no. Long job.

| Input | Type | Note |
|---|---|---|
| `video_path` | string | |
| `at` | float | source time to preview around |
| `pad` | float, default 6.0 | seconds shown before and after |

Renders a short clip of the edited result around one point and returns
`{job_id}`. The finished job gives the clip's path. Saved overlays that show
in the window are drawn; see "render and preview, with overlays".

### render

Read only: no. Destructive: no. Long job.

| Input | Type |
|---|---|
| `video_path` | string |

Renders the full edited video into `exports/` and returns `{job_id}`. A new
file name each time. Nothing is overwritten. Saved overlays are drawn.

One full render runs per video at a time. The creator can start one from the
page with Export video. While one is running, `render` starts nothing and
returns the running one:

```
{job_id, already_running: true, note}
```

Wait on that `job_id` with `job_status`. It holds the edit as saved when it
started. If the edit changed since, call `render` again once it finishes.

#### How render and preview cut

Both cut the edit in one ffmpeg pass that reads the source once, so the time
doesn't grow with the number of cuts. On a 22 minute take at 1280x720, 423
kept pieces export in about a minute and a half.

- The sound is cut on the sample the edit names, with a 20 ms fade either
  side of each join.
- The picture follows the sound. Only frames recorded inside a kept piece are
  shown, each within half a frame of its sound.
- The file is as long as the edit says: `duration` matches
  `expected_duration` to the frame.
- The encode is the same as before: H.264 High, CRF 19 for a render and 23
  for a preview, AAC at 192k, the source's size and frame rate.

The pass needs ffmpeg's `asegment` filter, which came with ffmpeg 5. With an
older ffmpeg both fall back to ClipForge's encode, whose time grows fast with
the number of cuts, and the result carries `note` saying so. Tell the creator
that a newer ffmpeg makes exports quick.

#### render and preview, with overlays

With no overlays saved, each is the one pass above.

With overlays saved, each runs in two passes: the cut of the edit into
`work/`, then one ffmpeg pass that draws every overlay over that file into
`exports/`. The result gains:

```
overlays: {drawn: [ids], clip_sound_mixed, not_shown: [ids], notes,
           edit_pass_seconds, overlay_pass_seconds}
```

A preview draws only the overlays that show in its window, picking up a
drift or a clip where it would be at that moment. The job's progress covers
both passes, each with the share of the time it takes: at 50% about half
the wait is over.

### look

Read only: no. Destructive: no. Writes one picture into `looks/`.

| Input | Type | Note |
|---|---|---|
| `video_path` | string | |
| `at` | float | source time. The join nearest to it is shown. |
| `span` | float, default 3.0 | seconds of edited video shown either side, at most 6 |

Returns a picture of one join as the viewer gets it, as an image Claude can
see, plus `{path, join, edited_at}`. Top to bottom:

1. the last kept frame before the cut beside the first kept frame after
2. a strip of stills across the window, in edited order
3. the sound wave of the edited audio
4. the kept words along the same time axis, with laughs marked
5. the removed words and the length removed

One strong vertical line marks the join in bands 2 to 4. Look for a jump in
the picture at the seam pair, and for the sound wave running through the
line, which means the cut lands mid-word.

With no cut near `at`, the picture shows the plain source around that time.

### set_overlays

Extra: `overlays`. Off unless `LUMR_STUDIO_EXTRAS` names it.

Read only: no. Destructive: no. Replaces the saved overlays for this video.

| Input | Type | Note |
|---|---|---|
| `video_path` | string | |
| `overlays` | list of overlay objects | the whole list. Fields read back from `get_overlays` are accepted and ignored. |
| `picture` | bool, default true | draw stills of the overlays as the viewer will see them |

Each overlay is checked on its own: the file exists and ffprobe reads it as a
photo or a video, every field is in range, and no two share a layer while
overlapping. A rejected overlay says why and how to fix it, and the rest are
saved. When overlays were given and every one is rejected, nothing is saved.
An empty list clears the overlays. At most 30.

Returns `{saved: [{id, kind, file, clock, place, layer, on_screen: {at,
seconds}, status}], rejected: [{index, overlay, why}], notes, picture:
{path, shows, left_out}}`. `on_screen` is in EDITED seconds under the saved
edit. `picture` is one image with a still of up to six overlays, each at the
middle of its time on screen with everything on screen then drawn. The MCP
result carries it as an image block, so Claude sees it.

One overlay:

```json
{
  "id": "ov1",
  "file": "/path/to/berlin.jpg",
  "kind": "image",
  "start": 491.88,
  "end": 497.9,
  "place": "full",
  "size": null,
  "fit": "fit",
  "motion": "zoom_in",
  "clip_in": null,
  "clip_out": null,
  "volume": null,
  "layer": 1,
  "fade": true,
  "note": "the harvest",
  "media": {"width": 1600, "height": 1200, "duration": null, "has_sound": false},
  "starts_on": {"word": "when", "start": 491.88},
  "ends_on": {"word": "we", "start": 497.68}
}
```

| Field | Set by | Note |
|---|---|---|
| `id` | server, or kept from input | `ov<n>`, never reused for this video. Pass it back to update that overlay. |
| `file` | Claude | absolute path to the creator's photo or clip |
| `kind` | server | `image` or `video`, from probing the file, never from its name |
| `start`, `end` | Claude | SOURCE seconds. At least 1 second apart, inside the video. |
| `place` | Claude | `full` (default), or a box at `top_left`, `top_right`, `bottom_left`, `bottom_right`, `center` |
| `size` | Claude | boxes only: the share of the frame's width and height the box takes, 0.2 to 0.6, default 0.4 |
| `fit` | Claude or server | full only: `fill` crops to cover the frame, `fit` shows the whole picture over a blurred copy. Default `fill` when the file's shape is within 20% of the video's, else `fit`. |
| `motion` | Claude | photos only: `zoom_in` (default), `zoom_out`, or `still`. A drift is 8% over the overlay's time on screen. |
| `clip_in`, `clip_out` | Claude | clips only: seconds into the clip. Default the whole clip. |
| `volume` | Claude | clips only: 0 (default) is muted, at most 0.5. 0.2 sits quietly under the voice. |
| `layer` | Claude | 1 (default) to 9. Higher shows on top where two overlap. |
| `fade` | Claude | a 0.3 second fade in and out, default true |
| `note` | Claude | why it goes there, at most 200 characters |
| `media` | server | the file's size as displayed, its length, whether it has sound |
| `starts_on`, `ends_on` | server | the first and last words under the span |

Sizes are shares of the frame and places are names, never pixels, so a
placement works at any video size. A box has a thin white edge.

#### Anchors and the hard cases

An overlay is anchored to SOURCE time, which never changes, with the words
under it recorded. It is placed on the edited video when it is read or
rendered, through the edit saved at that moment. So an overlay stays on its
words however often the cuts or the pace change.

| Case | Rule | What Claude reads |
|---|---|---|
| The start falls inside a cut | it appears when the talk resumes | note: "its start 8:02 is inside a cut, so it appears when the talk resumes at 8:03 on 'today'" |
| A cut falls inside the span | it covers the same words for less time, and hides the jump in the picture at each join | status `shortened`, note with the new length |
| Under 1 second left on screen | not shown, not rendered | status `hidden`, note: move it, or ask whether to keep that part |
| A clip shorter than its time on screen | it ends with the clip and the talk shows again | status `shortened`, note with the source time to end it at |
| The end is past the video's end | rejected when saved | the video's length |
| Two overlap on one layer | the later one in the list is rejected | put the top one on a higher layer, or move one |
| The transcript changed since placing | still placed by source time | note: read that stretch again |

No file of the creator's is copied. Each overlay keeps the absolute path to
their file. Every render checks each file again and stops, naming the
overlay, when one is missing or changed.

### get_overlays

Extra: `overlays`. Off unless `LUMR_STUDIO_EXTRAS` names it.

Read only: yes.

| Input | Type |
|---|---|
| `video_path` | string |

Returns `{overlays: [...], shown, not_shown: [ids], notes, edited_duration}`.
Each overlay is the saved object plus `clock`, `on_screen`, `status`
(`shown`, `shortened` or `hidden`), `notes`, and `over`, a short quote of the
kept words it covers.

### review

Read only: no. Destructive: no. Opens the treatment page.

| Input | Type | Note |
|---|---|---|
| `video_path` | string | |
| `open_browser` | bool, default true | open the page in the creator's browser |

Starts a small web server on this machine only and returns `{url,
opened_in_browser, claude_cuts, creator_cuts, trims, need_a_look,
word_times}`. On the page the creator:

- sets the pace with a dial of six stops, or opens Fine tune under it and
  sets the two sliders; the pace then reads Custom
- turns each kind of automatic trim on or off, each with a count and the
  time it saves. The filler switch lists the words it takes. A switch with
  nothing to take out at this pace says which pace would find some.
- turns Filler likes on or off: the filler words Claude picked, all at once.
  Until Claude has picked, the switch can't be pressed and says "Ask Claude
  to find them."
- sees her own cuts first (Your cuts), then Claude's grouped by kind (Said
  twice, False starts, Off topic, Other cuts), one row each with Claude's
  reason, and puts any of Claude's back or leaves it out. A cut whose join
  was flagged says why in plain words.
- double-clicks a word to cut it, and a struck word to bring it back.
  Selecting words offers Cut and Keep. A cut takes exactly the picked words.
  One step of undo, for her cuts and keeps. It does not cover the pace, the
  sliders or the switches.
- flags any stretch "keep no matter what": it widens to whole sentences, and
  nothing of Claude's or the pace's cuts inside it
- listens to three samples of about 30 seconds (the stretch where the most
  pauses come out, around Claude's biggest cut, around a laugh), or picks
  three new ones
- sees the whole video as a bar with an amber block for each stretch where lots
  was cut close together, shaded by how many edits it holds. Hovering a block
  says "More cuts here" and how many edits it holds, and its play button plays
  that stretch. Previous and Next go from cut to cut.
- sees a name tag on the video while one of their photos or clips is on
  screen. The page only shows them; changes go through `set_overlays`.
- saves the pace and switches as their usual for the next video
- exports the finished video with Export video

Every change saves to the edit at once. The page plays the source video
straight from disk. Nothing is uploaded. The address works until the Claude
session ends.

Export video starts the same job `render` starts, in the same registry. The
page shows its progress, then the file's name and folder. The creator may
export without telling Claude, so after they used the page, read `export` in
`get_edit` before rendering. When an export fails, the page tells the creator
to try again or ask Claude; `job_status` with that `job_id` has the error.

Ask the creator before calling this. It opens a window in their browser.

After the creator says they're done, call `get_edit` and read `creator` and
`export`.

#### What the creator does by hand

| She | The edit |
|---|---|
| cuts words | removes exactly those words. A word under 0.1 s takes the quiet beside it, never a piece of the word beside it. A cut next to one of hers joins it. |
| cuts inside a part she kept | the newer action wins: the kept part shrinks or splits around the words |
| brings a word back from a cut of Claude's | the cut splits around the word; the rest stays cut |
| brings a word back from an automatic trim | the trim is shortened around the word. The word stays back at every pace. |
| brings back a filler word Claude picked | the word plays, and stays back when Filler likes goes off and on again |
| cuts a word Claude left in | her own cut, a row under Your cuts. A lesson for Claude's next picks. |
| switches Filler likes off | every pick of Claude's plays again. Her own cuts and the words she brought back stay as they are. |
| moves a slider | the pace reads custom. Her cuts, keeps and Claude's picks stay. Not a step of undo. |
| keeps a part (not one word) | as before. A cut of hers inside the words she picked is taken back; one the widening reaches stays. |
| takes back a cut of hers | the words play again, unless Claude or the pace also removes them |

Her cuts and keeps outlive every pace change, every switch and every
`set_edit`. A word she cut beside a trimmed pause still leaves the pause the
pace leaves.

### job_status

Read only: yes.

| Input | Type | Note |
|---|---|---|
| `job_id` | string | from `transcribe`, `preview` or `render`, or from `export` in `get_edit` |
| `wait` | float, default 0 | seconds to wait for the job to finish before answering, at most 50 |

Returns `{status: "running" | "done" | "failed", progress, result, error}`.

An export the creator started from the page is a render job like any other.
Its `job_id` is in `get_edit` under `export`. Its result carries
`output_path`, the file to point the creator to.

Pass `wait` to hold the call open while the job runs, so one call replaces
many. With `wait: 0` it answers at once.

### chapter_times

Extra: `publish_kit`. Off unless `LUMR_STUDIO_EXTRAS` names it.

Read only: yes.

| Input | Type | Note |
|---|---|---|
| `video_path` | string | |
| `source_times` | list of float | times in the source video |

Returns each time converted to the edited video, using the saved edit. A time
that falls inside a cut maps to the start of the next kept segment.

### save_publish_kit

Extra: `publish_kit`. Off unless `LUMR_STUDIO_EXTRAS` names it.

Read only: no. Destructive: no.

| Input | Type | Note |
|---|---|---|
| `video_path` | string | |
| `titles` | list of string | two to five options |
| `description` | string | |
| `chapters` | list of `{time, label}` | `time` in EDITED seconds |
| `tags` | list of string | |

Validates YouTube's chapter rules: the first chapter starts at 0, there are at
least three, and each is at least 10 seconds long. Writes `publish-kit/` and
returns the file paths.
