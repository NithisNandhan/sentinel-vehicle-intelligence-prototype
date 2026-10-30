# Work log: plate pipeline (branch `work/plate-pipeline`)

Unattended session started 2026-10-31. Nothing is pushed. Every behaviour change is
behind a setting that is off by default. The 3-frame fusion rule and every acceptance
threshold are unchanged. No accuracy claim is made beyond the labelled sample, and the
labels are AI-drafted, not verified ground truth.

## Setup decisions

- **Branch base.** `work/plate-pipeline` starts from `tracker/keep-unconfirmed-vehicles`
  (commit `de00f38`), which already holds Phase 5 (bike tracking, off by default,
  12 tests). It was finished and measured earlier today, so it is not redone.
- **Earlier frame-label change kept apart.** The compact on-frame labels approved earlier
  are not behind a setting, so they were committed on their own branch
  `ui/compact-frame-labels` (`5509db7`, from `main`) and are not part of this branch.
  The running analysis service still has them loaded; it is not restarted here.
- **Crops and labels stay out of Git.** Plate crops and draft labels contain full plate
  text from a stock clip whose page URL was not recorded, and the project keeps full
  plate text out of central storage. They live in `data/plate-review/` (ignored by
  `/data/*`). Only tools, tests and aggregate numbers are committed.
- **Backend tests.** The API image has no pytest, so backend tests run in a throwaway
  container from that image with `backend/` mounted read-only and the dev requirements
  installed. No volume is used.

## Baseline tests (before any change)

| Suite | Result |
| --- | --- |
| Edge (`edge/tests`, in the analysis container) | 397 passed, 2 skipped |
| Tools (`tools/test_handoff.py`) | 14 passed |
| Backend (`backend/tests`) | 103 passed |
| Frontend (`vitest`) | 31 passed |

## Phase 1: draft labels (done)

- 50 of the 165 probe crops from the Mumbai clip (`new video.mp4`, 15 samples at 500 ms,
  general detector profile), chosen evenly by file order, not by content.
- Labelled by Claude from enlarged contact sheets that show only an ID and size, never the
  pipeline's OCR. `UNREADABLE` was used whenever any character was uncertain.
- Result: **2 readable** (c36, c39; the same plate), **14 NO_PLATE**, **34 UNREADABLE**.
  5 crops are two-line plates (c02, c08, c09, c33, c45); all 5 are UNREADABLE.
- Files (local only, ignored by Git): `data/plate-review/mumbai/labels_draft.csv`,
  `crops/`, `sheets/`, `crops.csv`, `README.md`.
- **10 least sure** (a person should check these first): c22, c26 (looked nearly
  readable, kept UNREADABLE), c33, c45, c02, c08 (two-line, bottom line ambiguous),
  c36 (readable, medium confidence), c41, c44 (partly legible), c12 (unsure it is a plate).
- Caveat: earlier the same day a few pipeline readings for some of these vehicles were
  seen. Those crops were kept UNREADABLE unless the pixels alone were clear.
- **Consequence for later phases:** exact-match and character error rate rest on n=2
  readable crops and cannot show an improvement or a regression with any confidence. The
  NO_PLATE false-read rate (n=14) is the most informative metric available.

## Phase 2: evaluation script and baseline (done)

- `tools/eval_plates.py`: exact-match rate, character error rate (edit distance / label
  characters; an empty reading costs every character) and false-read rate on NO_PLATE
  crops. Every rate carries its denominator and is `null` at n=0. 8 unit tests in
  `tools/test_eval_plates.py`.
- `tools/run_plate_variants.py` reads crops with the real OCR. Two readings per variant:
  `raw` (greedy CTC text) and `format` (best Indian-format candidate from a one-crop
  fusion reader, which is what the pipeline could display).
- `tools/plate_crop_probe.py` re-runs the pipeline's detection steps on a clip and saves
  each crop with its plate-detector score. To get that score, `PlateDetector.box` and
  `TargetedPlateSearch` now keep `last_confidence`; which box is chosen is unchanged
  (4 tests in `edge/tests/test_plate_confidence.py`).
- The re-run reproduced the earlier probe exactly (271 vehicle boxes, 165 crops, 104 with
  no plate) and all 50 labelled crops are byte-identical, so scores attach reliably.

Baseline on the 50 draft-labelled crops:

| Reading | Exact match (n=2) | CER | False reads on NO_PLATE (n=14) | Readings on UNREADABLE (n=34) |
| --- | --- | --- | --- | --- |
| raw | 2/2 | 0/20 | 14/14 | 34/34 |
| format | 2/2 | 0/20 | 5/14 | 29/34 |

- The OCR agrees with both image-only labels, but n=2 cannot show an accuracy level.
- 5 of 14 non-plates yield a well-formed Indian plate string: invented readings that can
  reach fusion. Phase 3 targets these.
- Host note: `tools/test_handoff.py` has 4 Windows file-locking errors when run on the
  host; it passes 14/14 in the Linux container, which is where tools tests are run.

## Phase 3: reject non-plates (done)

- `edge/sentinel_edge/plate_filter.py`, setting `SENTINEL_REJECT_NON_PLATES` (off by
  default). A proposal is dropped before it can become fusion evidence when: its
  width/height is outside 0.8-8, its plate-detector score is below 0.35 (an unknown score
  is never held against it), or the OCR has no character with peak probability >= 0.6
  (the same per-character peak that acceptance already requires in two frames).
- The screen runs right after the size check, so a rejected proposal cannot move the
  plate centre or reset the reader that holds a real plate's evidence.
- Fusion safety: tests show a low-confidence invented reading is never accepted, with
  the filter off (fusion's own per-character gate) or on (it never becomes evidence).
  The 3-frame rule and the 0.85 acceptance threshold are untouched.
- Limits were chosen on the same 50 draft labels they are measured on, so the numbers
  below are optimistic (overfit) and need a held-out check.

| On 50 draft-labelled crops | Before | After |
| --- | --- | --- |
| False reads on NO_PLATE, raw (n=14) | 14 | **1** |
| False reads on NO_PLATE, format-valid (n=14) | 5 | **1** |
| Readable plates lost (n=2) | - | **0** |
| UNREADABLE crops dropped (n=34) | - | 13 |

- The remaining false read is c49, a goods-carrier sign: real, confident text that no
  shape or confidence rule separates from a plate.
- The 13 UNREADABLE crops dropped include c02, a real two-line plate whose OCR had no
  confident character. In the pipeline each dropped crop is one fewer observation, so
  some tracks may now take longer to reach 3 observations. Not measured on a full run.
- Tests: 7 in `edge/tests/test_plate_filter.py`.

## Phase 4: two-line plates (attempt 1 failed)

- `edge/sentinel_edge/two_line.py`, setting `SENTINEL_TWO_LINE_PLATES` (off by default):
  split when height/width >= 0.45 and an ink gap separates two inked rows, read each row,
  join top then bottom with a blank step (5 tests).
- Attempt 1 made readings worse. The single-line read of the two-line crops was already
  consistent with the labeller's partial notes, and splitting turned it into noise. The
  rule also split 7 single-line or non-plate crops whose padding makes them look tall.
  Format-valid readings on two-line crops went from 5 of 5 to 0 of 5. The OCR model
  appears to read stacked rows already (its training corpus included two-row plates).
- Attempt 2: keep the split only when the OCR is more confident about it (see below).
- Attempt 2 failed too. Reading both ways and keeping the split only when it has more
  confident characters left all 5 two-line crops at their single-line reading (no
  change) and split exactly one crop, c43, a single-line plate cut off at the edge.
  Exact match 2/2 and CER 0/20 unchanged; NO_PLATE false reads unchanged (format 5/14).
  No benefit on this sample and one wrong split. Logged under **Blocked**; the code stays
  off by default and should not be enabled.

## Blocked

- **Phase 4, two-line plates** (2 attempts). The OCR model already produces plausible
  single-line reads of the two-line plates here, and splitting either adds noise
  (attempt 1) or changes nothing useful (attempt 2). A real test needs readable,
  human-labelled two-line plates; all 5 in this sample are UNREADABLE.

## Phase 5: bike tracking (done earlier, re-measured on this branch)

- Code from `tracker/keep-unconfirmed-vehicles` (`de00f38`): setting
  `SENTINEL_KEEP_UNCONFIRMED_VEHICLES` (off by default) keeps detections ByteTrack hides
  until confirmed and links them by plausible motion relative to box size (the same
  discontinuity rule the OCR stage uses); 12 tests, including two touching bikes that
  must not merge or swap and a car that must not get a second ID.
- Re-measured on this branch: identical to the earlier numbers, off and on.

Mumbai clip, 15 samples, motorcycles (heuristic counts; no ground truth):

| | Off | On |
| --- | --- | --- |
| Boxes shown (detector found 92) | 11 | 92 |
| Distinct IDs | 7 | 25 |
| One-frame IDs (first/last sample excluded) | 1 | 6 |
| Likely ID switches | 0 | 0 |
| Same-frame duplicate-ID pairs, all classes | 15 | 15 |

- Trucks: likely switches 0 -> 3. On the 720p `video-project-2.mp4` bikes are shown
  (13 -> 122 boxes) but mostly not linked: 98 IDs, 80 one-frame. The switch heuristic
  uses the same motion rule as the linker, so it cannot see breaks beyond that rule.

## Phase 6: comparisons (report only, no default changed)

**Plate detectors** on the Mumbai clip (271 vehicle boxes, 15 samples):

| | general (parent, default) | plate_adapted |
| --- | --- | --- |
| Plate crops | 165 | 177 |
| Vehicle boxes with no plate found | 104 | 92 |
| Found by this detector only | 6 | 18 |

- The 18 adapted-only crops were inspected (draft judgement): about 4 look like plates and
  about 14 are windows, vehicle fronts, bars, a bus panel, a grille, an LED sign or blur.
  Its extra finds are mostly false proposals. Detector scores are on different scales.

**Enlarging crops before OCR** (cubic), on the 50 draft-labelled crops:

| | Baseline | 2x | 3x |
| --- | --- | --- | --- |
| Exact match (n=2) | 2/2 | 2/2 | 2/2 |
| CER | 0/20 | 0/20 | 0/20 |
| Format-valid false reads on NO_PLATE (n=14) | 5 | 5 | 4 |
| Format-valid readings on UNREADABLE (n=34) | 29 | 25 | 24 |
| Raw readings changed vs baseline | - | 35 | 38 |

- Every change falls on UNREADABLE or NO_PLATE crops, where nothing can be verified.
  Enlarging adds no detail. No measurable benefit; no default change.

## Phase 7: speed (code done; timing NOT measured)

- **Fusion only on a new reading:** already the case. `CTCPlateReader.candidates()` caches
  its result and the cache is cleared only by `add()` (a new reading) or `propose()`
  (`edge/sentinel_edge/ctc_fusion.py`). No code change was made, to avoid a redundant
  second cache.
- **More CPU threads:** `SENTINEL_VEHICLE_THREADS` (vehicle detector, torch) and
  `SENTINEL_PLATE_THREADS` (plate detector, ONNX). The default stays 2, the long-standing
  hard-coded value; invalid values fail loudly (3 tests). OCR threads already follow the
  existing `SENTINEL_CPU_THREADS` (default 4 in compose).
- `tools/time_tracked_pipeline.py` times the same tracked pipeline as the service per
  stage and fingerprints every output (tracks, plate boxes, candidates with scores,
  acceptances) so two runs can be checked for identical results.
- **Not measured:** the timing job (run A, defaults, then run B, 6/4/8 threads) was
  stopped by Claude Code because the computer was critically low on memory. Its leftover
  process inside the container was stopped too (container memory 2.7 GiB -> 1.4 GiB).
  It was not restarted. So there are **no before/after times and no identical-output
  check** for the thread settings; leave them at the default until measured.
- Earlier observation (service, not this tool): the analysis used about one CPU core of 16.
