# Plate pipeline work: results (branch `work/plate-pipeline`, not pushed)

Unattended session, 2026-10-31. Full detail and every decision: `docs/WORK_LOG.md`.

**Read this first.** Every number below comes from one 7.5 s Mumbai clip
(`new video.mp4`, 4K, 15 samples at 500 ms). The plate labels are **drafts written by
Claude (an AI assistant) by looking at the crops. They are not verified ground truth.**
Only 2 of the 50 labelled crops are readable, so exact match and character error rate
cannot show an accuracy level, an improvement or a regression. No accuracy claim is made
beyond this labelled sample. Every behaviour change is behind a setting that is **off by
default**. The 3-frame fusion rule and every acceptance threshold are unchanged.

## Opt-in settings added (all off)

| Setting | Component | Effect when `true` | Recommendation |
| --- | --- | --- | --- |
| `SENTINEL_REJECT_NON_PLATES` | edge | Drop non-plate proposals before fusion | Promising; check on held-out labels first |
| `SENTINEL_TWO_LINE_PLATES` | edge | Read two-line plates row by row | **Do not enable** (blocked) |
| `SENTINEL_KEEP_UNCONFIRMED_VEHICLES` | edge | Show and plate-search unconfirmed vehicles | Helps on 4K; noisy on 720p |
| `SENTINEL_VEHICLE_THREADS`, `SENTINEL_PLATE_THREADS` | edge | CPU threads (default 2) | See Phase 7 |
| `SENTINEL_STRICT_PLAYBACK` | edge | Refuse pause/resume on a finished run (409) | Safe to enable |
| `RECOGNITION_IDLE_EVIDENCE_404` | API | 404, not 503, for evidence after Stop | Safe to enable |
| `VITE_NEUTRAL_IDLE_LABEL` | dashboard | Idle panel not called "live" | Safe to enable (needs dashboard rebuild) |

## Phase by phase

### 1. Draft labels
50 of 165 crops, chosen evenly by file order. **2 readable** (same plate), **14
NO_PLATE**, **34 UNREADABLE** (5 of them two-line). The 10 least sure: c22, c26, c33, c45,
c02, c08, c36, c41, c44, c12. Labels and crops stay local in `data/plate-review/mumbai/`
(not in Git).

### 2. Evaluation tool and baseline
`tools/eval_plates.py` (8 tests). Baseline on the 50 crops:

| Reading | Exact (n=2) | CER (n=20 chars) | False reads on NO_PLATE (n=14) |
| --- | --- | --- | --- |
| raw OCR | 2/2 | 0 | 14 |
| format-valid candidate | 2/2 | 0 | 5 |

### 3. Reject non-plates
| On 50 crops | Before | After |
| --- | --- | --- |
| False reads on NO_PLATE, format-valid (n=14) | 5 | **1** |
| False reads on NO_PLATE, raw (n=14) | 14 | **1** |
| Readable plates lost (n=2) | - | 0 |
| UNREADABLE crops dropped (n=34) | - | 13 |

The remaining false read is a goods-carrier sign (real text). Tests show fusion cannot
accept a low-confidence invented reading with the filter on or off. The limits were set
on the same labels they are scored on, so this result is optimistic.

### 4. Two-line plates: **blocked after 2 attempts**
Attempt 1 turned 5 of 5 format-valid two-line readings into noise and split 7 crops
that were not two-line. Attempt 2 (keep the split only when more confident) changed
nothing useful and split one single-line crop. The OCR model already reads stacked rows
plausibly here.

### 5. Bike tracking (Mumbai clip, motorcycles; heuristic counts)
| | Off | On |
| --- | --- | --- |
| Boxes shown (detector found 92) | 11 | 92 |
| Distinct IDs | 7 | 25 |
| One-frame IDs | 1 | 6 |
| Likely ID switches | 0 | 0 |
| Same-frame duplicate-ID pairs (all classes) | 15 | 15 |

Trucks: likely switches 0 -> 3. On the 720p clip bikes are shown but mostly not linked
(98 IDs for 122 boxes, 80 one-frame). 12 tests, including touching bikes that must not
merge and a car that must not get a second ID.

### 6. Comparisons (report only)
- **Plate detectors:** `plate_adapted` returns 177 crops vs 165, but of its 18
  adapted-only crops about 14 look like non-plates (draft judgement). Default unchanged.
- **Enlarging crops 2x / 3x:** readable crops unchanged (2/2); format-valid false reads
  on NO_PLATE 5 -> 5 / 4; all other changes fall on crops nobody could verify. No benefit
  shown; default unchanged.

### 7. Speed: **not measured**
- Recomputing fusion only on a new reading is already how the reader works (it caches
  candidates until `add()` or `propose()`), so no code change was made.
- Thread counts for the vehicle and plate detectors are now settings (default 2,
  unchanged), with a timing tool that fingerprints all outputs.
- The timing run was stopped by Claude Code because the computer was critically low on
  memory, and it was not restarted. **There are no before/after times and no
  identical-output check.** Keep the defaults until this is measured.

### 8. Small fixes (each behind its own off-by-default setting)
- Idle "SHARED LIVE PROCESSING" heading -> "SHARED PROCESSING · NO SOURCE SELECTED"
  (`VITE_NEUTRAL_IDLE_LABEL`), 1 frontend test.
- Evidence after Stop: 404 "No recording is being analysed" instead of 503
  (`RECOGNITION_IDLE_EVIDENCE_404`); the dashboard already shows a 404 as empty.
  2 backend tests.
- Pause/resume on a finished run: 409 instead of a silent no-op (`SENTINEL_STRICT_PLAYBACK`),
  which the API already reports as "Playback control not available". 3 edge tests.

## Tests

| Suite | Before | After |
| --- | --- | --- |
| Edge | 397 passed, 2 skipped | 420 passed, 2 skipped |
| Tools | 14 passed | 22 passed |
| Backend | 103 passed | 105 passed |
| Frontend + type check | 31 passed | 32 passed, clean |

## Still unproven

- Any accuracy level: 2 readable labelled crops, AI-drafted, one clip.
- That the Phase 3 limits generalise: chosen and scored on the same 50 crops.
- The effect of Phase 3 on full runs (fewer observations may delay the 3-frame rule).
- Whether bike linking helps plates get read: not measured end to end.
- Two-line reading: no readable two-line plate in the sample.

## Needs human

1. Check the 10 least-sure draft labels (c22, c26, c33, c45, c02, c08, c36, c41, c44,
   c12), then all 50, in `data/plate-review/mumbai/labels_draft.csv` with the sheets in
   `sheets/`.
2. Better footage: plates facing the camera and large in frame (ideally 100 px wide or
   more), including readable two-line plates, with a person labelling a held-out set.
3. Decide whether to push `work/plate-pipeline` (and `tracker/keep-unconfirmed-vehicles`,
   `ui/compact-frame-labels`). Nothing has been pushed.
4. Decide which opt-in settings, if any, to enable after a held-out check.
5. Re-run the Phase 7 timing when the computer has free memory (close other programs):
   `tools/time_tracked_pipeline.py` with default threads, then with more, and compare
   `outputs_sha256`.
