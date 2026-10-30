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
