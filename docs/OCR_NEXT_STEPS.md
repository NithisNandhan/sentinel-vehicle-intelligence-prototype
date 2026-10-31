# Plate reading: what changed on 31 October 2026 and what comes next

Branch `work/plate-pipeline`. Two opt-in settings, both **off by default**. Neither lowers the
3-frame rule or any acceptance threshold (mean character evidence 0.85, weakest character
0.75, support 0.6 in two frames, log-score margin 0.5). **No accuracy gain is claimed:**
there are no human-checked labels, and the AI-drafted sample has 2 readable crops (1 plate).

## What was already tried (do not repeat)

From `OCR_FINETUNING.md`, `OCR_IMPROVEMENTS.md`, `OCR_MODEL_SELECTION.md`,
`INDIAN_OCR_CORPUS.md` and `DEEP_OCR_ADAPTATION.md`:

1. Model choice: on one NIVU vehicle in 21 degraded copies, CCT-S and PARSeq accepted 0/21;
   Indian-fine-tuned Awiros (CTC) accepted 11/21, none wrong. Awiros is the current reader.
2. Awiros limits on that passage: accepted at plate widths of 54 px and above, failed at 41 px, lost the
   truth below 28 px; motion blur kernel 5+ and darkness 0.5 or less abstained.
3. Quality-weighted frame selection: 11 to 12/21 (one borderline case crossed 0.75).
4. ECC reconstruction + NAFNet deblurring + deskew: no extra accepted readings (12/21).
5. Automatic plate crops: 13/21. Every one of these is one vehicle, not 21 independent tests.
6. CTC-head fine-tune on 48 Bapatla registrations (CC BY 4.0): held-out 9/10 before and after,
   character errors 2 to 1; multi-frame 13/21 unchanged.
7. Corpus built: 30,110 crops (1,066 photographs, 18,000 generated, 11,044 mixed origin).
   Publisher labels are not human-verified.
8. Deeper fine-tune (head + last SVTR block + one encoder conv; 802 photos + 25% generated):
   still test 22/31 to 28/31 exact, legacy 9/10 unchanged, multi-frame 12 to 13/21. That is the
   current `indian-corpus-encoder` checkpoint. Labels were AI-reviewed, not human.
9. Live shadow on CAM01/CAM02: 0 plate crops reached OCR; later runs could not connect.
10. This branch already added: non-plate filter (false reads on NO_PLATE crops 5/14 to 1/14,
    tuned on the same sample); two-line splitting (failed twice, keep off); crop enlarging (no gain).

## Change 1: Indian plate format validator (`SENTINEL_PLATE_FORMAT_VALIDATOR`)

`edge/sentinel_edge/plate_format.py`. Before fusion scores a candidate, it must be:

- **State series:** one of 40 state/UT codes (current, plus legacy DN, OR, UA and TS; TG
  added for Telangana since 2024), district 01-99 (Delhi also 1-9), 0-3 series letters,
  number 1-4 digits and not 0000.
- **Bharat series:** year 21 or later, `BH`, four digits, one or two letters.

A reading that already fits is never changed. One that does not fit may have look-alikes
swapped (O/0, I/1, B/8, S/5, Z/2, G/6), but only when exactly one valid plate needs the
fewest swaps. Two equally small fixes (`ambiguous`) or no fix at all (`invalid`) means the
reading is dropped, not guessed. A fixed candidate is then **rescored against the frames'
pixels**. If the frames do not show the swapped character, the unchanged gate rejects it.
The reconstruction veto keeps the broad format, so it stays exactly as strict as before.

Not covered (marked invalid): diplomatic, military, temporary and trade plates. A
valid format can still be the wrong vehicle; this is not a registry.

**Marked "format-adjusted, unverified".** Any candidate whose text came from a look-alike
swap, rather than being read as-is by some frame, carries `format_adjusted=True`. That covers
corrected readings and readings the validator created where the default showed none. The mark
appears in four places:

- the frame call-out: `... - FORMAT-ADJUSTED, UNVERIFIED`
- the dashboard Inspector's gate reasons: "Leading reading is format-adjusted, unverified: ..."
- the track diagnostics (`format_adjusted` on each candidate)
- the `tools/replay_plate_tracks.py` outputs (`*_note` columns in `crop_details.csv` and
  `track_details.csv`)

A marked reading is never accepted unless it passes the unchanged frame gate. The central
API does not carry the mark, because that would change its contract. With the validator off,
nothing is ever marked.

Tests: `edge/tests/test_plate_format.py`, 27 tests (25 sub-cases). They cover valid plates
with look-alikes in correct positions left unchanged, each of the six confusions in letter and
digit positions, impossible readings, the ambiguous case, and the reader and pipeline wiring.
Six of them cover the format-adjusted mark, including a validator-created reading from
three frames that the gate rejects (weakest character below 0.75). They also check that the
3-frame rule still applies to it, and that the mark reaches the gate reasons, the track
display and the frame call-out.

## Change 2: per-position confidence-weighted vote (`SENTINEL_POSITION_VOTE`)

`position_vote` in `edge/sentinel_edge/ctc_fusion.py`. Each original frame's greedy
reading votes per character position. A vote weighs `frame quality x character peak
probability`. Only frames that read the most common length vote, because CTC positions
only line up when lengths agree. The vote returns nothing when fewer than two frames
agree on length, or when a length or position ties.

The voted string is added as **one more candidate**, rescored by exact CTC likelihood
against every frame, and must pass the same gate. So the vote can surface a reading that
no single frame's beam held (each frame wrong in a different place), but it cannot accept
anything on its own.
Ranking stays by CTC likelihood. A mean-of-peaks vote does not penalise a dropped
character, so letting it rank or accept would need new acceptance evidence.

Tests: `edge/tests/test_position_vote.py`, 19 tests. They cover majority voting, confidence
outweighing count, frame weights, the length rule, the no-guess cases, and the vote producing a
plate no frame read. They also show the 3-frame rule and the evidence gate still holding, the
vote following the format setting, and the pipeline wiring. Seven cover `vote_report()`, the
diagnostic that says why the vote did or did not add a candidate.

## Measured results

**Sample size: unproven.** There are **0 human-checked labels**. The only labels are 50
AI-drafted crops from the Mumbai clip (`new video.mp4`, 4K, 15 samples at 500 ms): **2 readable
crops of 1 plate**, 14 `NO_PLATE`, 34 `UNREADABLE`. That is far below the 30
readable plates needed, so exact match and character error rate cannot show a gain or a loss.
The Hyderabad pilot labels (3 plates, also AI-drafted) could not be used: the video is not on
this machine.

Run with `tools/replay_plate_tracks.py` in the analysis container, real Awiros OCR on CPU
(`indian-corpus-encoder` checkpoint), on the saved probe output (165 crops, 33 tracks).
Scored with `tools/eval_plates.py`. Outputs are local only
(`data/plate-review/mumbai/replay-2026-10-31/`, ignored by Git).

**Single crops** (50 labelled; `raw` is the greedy text, `format` the one-crop reader's top
candidate):

| Reading | Exact (n=2) | Char. errors (of 20) | Readings on NO_PLATE (n=14) | Readings on UNREADABLE (n=34) |
| --- | --- | --- | --- | --- |
| raw, validator off | 2 | 0 | 14 | 34 |
| raw, validator on | 2 | 0 | **0** | **11** |
| format, validator off | 2 | 0 | 5 | 29 |
| format, validator on | 2 | 0 | **2** | **25** |

**Tracks** (each probe track replayed through fusion in PTS order; 22 tracks have a label: 2
readable [the same plate on 2 tracks], 6 `NO_PLATE`, 14 `UNREADABLE`):

| Setting | Accepted correct (n=2) | Accepted wrong | Accepted on NO_PLATE / UNREADABLE | Top candidate shown on NO_PLATE (n=6) | Top shown on UNREADABLE (n=14) |
| --- | --- | --- | --- | --- | --- |
| both off (default) | 1 | 0 | 0 / 0 | 3 | 9 |
| validator | 1 | 0 | 0 / 0 | **2** | 9 |
| vote | 1 | 0 | 0 / 0 | 3 | 9 |
| validator + vote | 1 | 0 | 0 / 0 | **2** | 9 |

Across all 33 tracks, every setting accepted exactly one reading, on track 25, and it
matches its label. The other readable track (161) had only one crop, so the 3-frame
rule left it unread under every setting.

What the numbers do and do not show:

- **Validator:** on single crops it removed every greedy reading of a non-plate (14 to 0) and
  most readings of unreadable plates (34 to 11), while keeping both readable crops. Example:
  the goods-carrier sign (crop c49) that the non-plate filter could not stop is now rejected,
  because it reads as district 00. In fusion it changed what the overlay would show, not what was accepted.
  On one non-plate track (33) it removed the shown candidate. **One side effect:** on
  `UNREADABLE` crop c43 the validated reader now shows a 7-character reading where the
  default showed nothing. A corrected look-alike can surface a candidate that was not shown
  before; the gate did not accept it. Such readings are now marked format-adjusted.
  (The results above were produced before the mark was added.) Labels are drafts, so these are counts on one clip, not rates.
- **Vote:** **no effect at all on this sample**. The top candidate and the accepted reading were
  identical with and without it on every track. See the next section for why.
- Neither setting produced a wrong accepted reading. With 1 readable plate, that says nothing
  about accuracy.
- The replay does not apply the non-plate filter, the plate-centre reset or reconstruction,
  and it reuses the probe's tracks. Running all four settings took about 52 minutes on one
  core; per-setting time was not measured.

## Why the vote changed nothing (diagnosed 31 October 2026)

`tools/explain_position_vote.py` replays the same 33 tracks with the vote on and logs, per
vehicle, why the vote did or did not add a candidate (`CTCPlateReader.vote_report()`).
Per-track rows are local only (`replay-2026-10-31/vote_per_track.csv`, plate text).

**Verdict: not a bug.** The vote runs and is passed to fusion wherever fusion runs. It changed
nothing for two reasons. First, half the vehicles never collect enough readings. Second, where
votes exist, they almost never offer a plate that fusion was not already considering.

| Readings per vehicle (most in one window) | 1 | 2 | 3 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Tracks (of 33) | 9 | 6 | 5 | 3 | 1 | 1 | 8 |

- **Too few readings:** 15 of 33 tracks never reach 3 readings, so fusion and the vote never
  run for them.
- **106 steps where fusion runs** (3 or more readings) on the other 18 tracks:

| What the vote did | Steps |
| --- | --- |
| No vote: readings of different lengths, none in the majority (length tie) | 27 |
| Voted, but the result is not a plate format (short partial readings such as 4-6 characters) | 40 |
| Voted a plate that some frame's beam already held, so it was already a candidate | 38 |
| Voted a plate no beam held: a new candidate | **1** |

- At 22 steps the voted plate differed from the top reading. In 21 of them it was already a
  candidate that the CTC likelihood ranked lower. By design the vote does not rank
  candidates, so it could not change the top.
- The single new candidate (track 86) was not accepted. At the end of that track the top
  reading was the same with and without the vote.
- Readings within one vehicle disagree on length a lot. For example, one track's 8 readings
  had 6 different lengths. This is why so few votes form, and why those that do are often
  short fragments.

So more readings per vehicle (larger plates, more frames) is the precondition. Even then,
the vote can only change results where no single frame's beam already holds the plate.

## What remains

- **No real accuracy measurement exists.** Every figure above rests on AI-drafted labels
  with 1 readable plate. It cannot show a gain or a loss.
- Neither setting is turned on, in `compose.yaml` or locally. On this sample the validator
  only reduces what is shown on non-plates, and the vote does nothing measurable. Decide
  after the labelled test below.
- Rare formats (diplomatic, military, temporary, trade) are rejected when the validator is on.
- The vote only helps when frames agree on length. Readings that drop or add a character
  do not vote. On the Mumbai clip it added 1 new candidate in 106 fusion steps (see above).
- The validator's fixes pass through the gate only when the pixels support them. On small
  plates they rarely will.
- Fusion is 76% of analysis time (Phase 7). The validator adds a cached format check per
  beam text, and the vote adds one candidate. Timing was not re-measured.
- The plate detector, not OCR, is the first loss on this footage: 104 of 271 vehicle boxes
  gave no plate crop (Phase 2 probe).

## What you need to provide

1. **Footage with readable plates.** Permitted recordings (with `tools/register_recording.py`
   receipts) where plates are about **80 px wide or more** in the native frame and each
   plate is visible in at least 3 samples 500 ms apart. Measured boundaries: 54 px and above
   accepted, 41 px failed. In the Mumbai clip the median plate crop is 62 px, only 23 of 165
   reach 80 px, and the only two readable crops are 85 and 90 px wide.
2. **At least 30 hand-checked labels** of distinct plates, made by a person from the pixels
   (not from pipeline output), in the `tools/eval_plates.py` format
   (`crop_id,label` with a plate, `NO_PLATE` or `UNREADABLE`). The current 50 drafts in
   `data/plate-review/mumbai/labels_draft.csv` are a starting point to check, not truth.
   Keep the test labels apart from anything used to tune.
3. **A decision on fine-tuning,** with the licence of any dataset used. Candidates already
   audited: Bapatla (CC BY 4.0, used), umar1103 Final Licence (CC BY 4.0, mostly generated
   artwork), NIVU (CC BY 4.0, evaluation only). Any new set needs its licence, source and
   permission recorded before use. Training on the test labels is not allowed.

With those, rerun:

```text
python /workspace/tools/run_ocr_research.py /workspace/tools/replay_plate_tracks.py \
    <probe dir> <crops.csv> <labels.csv> <out dir>
python tools/eval_plates.py --labels <out>/track_labels.csv --predictions <out>/track_predictions.csv
```

Turn a setting on only if it improves exact matches on the held-out labels without adding
wrong accepted readings.
