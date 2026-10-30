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
