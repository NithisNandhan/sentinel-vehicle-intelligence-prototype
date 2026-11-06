# Feature decisions

Audit of the plate-pipeline work merged from `work/plate-pipeline`, `tracker/keep-unconfirmed-vehicles`,
`ui/compact-frame-labels` and the OCR experiment branch, plus the decisions taken in this work. "Pipeline A" is the
`worker.py`/`vision.py` path; "Pipeline B" is the tracked CTC path (see `SENTINEL_BASELINE.md` for this assumption).
No Pipeline A code was removed.

| Feature | Where | Evidence for / against | Decision |
| --- | --- | --- | --- |
| Pipeline A (`worker.py`, `vision.py`, FastALPR, `temporal-v2`/`hybrid-v1`) | edge | production worker; zero accepted real plates recorded | **Retained, frozen** (no investment, not removed; rollback target) |
| Strict format validator `plate_format.py` | edge | 21 tests; on 50 AI-drafted labels cut greedy readings on non-plates 14→0 | **Modified & integrated**: look-alike swaps now only *flagged* (`strict_status`) for confirmation; the swap still exists behind its opt-in flag and is marked `format_adjusted`, which can never be CONFIRMED |
| Per-position vote (`position_vote`) | `ctc_fusion.py` | diagnosed as not a bug; changed nothing on the one readable plate (n=1) | **Retained opt-in**, not in profiles (unproven) |
| Non-plate filter | `plate_filter.py` | 14→1 false reads on 14 non-plates; limits tuned on the same 50 draft labels (overfit) | **Retained opt-in**, not in profiles until held-out labels exist |
| Unconfirmed-vehicle tracking | `vision.py` | helps 4K bikes (11→92 boxes), noisy at 720p; no ground truth | **Retained opt-in** (4K-only option); not in profiles |
| Plate recovery (`--plate-recovery`: 250 ms sampling + full-image fallback) | `crop_collector.py`, viewer | exists; blocking mode for recordings | **Integrated into candidate** as *adaptive* sampling: 250 ms only while a visible vehicle lacks 3 distinct observations; `--plate-recovery` kept as is |
| Crop generation (`automatic_crops.py`, tiles) | edge | native tiling experiments, GPU-oriented | **Integrated into `candidate-gpu`** via `ProposalPool` (budgeted); not enabled on CPU |
| Text boundaries (`text_boundaries.py`, PP-OCRv5 det) | edge | needs Paddle + pinned assets; proposals include signs | **Integrated into `candidate-gpu`** as lowest-ranked source with a plate-likeness filter; loaded only if `SENTINEL_TEXT_DETECTOR_ASSETS` is set; never downloads |
| OCR input review (`ocr_input_review.py`, HTML tool) | edge/tools | local developer viewer using full text | **Retained as developer tool**; production review is the access-controlled Review page |
| Candidate ranking (`CTCPlateReader.candidates`) | `ctc_fusion.py` | exact CTC rescoring across frames | **Retained unchanged**; gate untouched |
| Two-line row-by-row OCR (`two_line.py`) | edge | two failed attempts (5/5 → 0/5 valid) | **Retained disabled**; `layout.py` only classifies and blocks on conflict. No claim that two-row improved. |
| ONNX export, SVTRv2-B | edge experiments | SVTRv2-B wrong on 36/137 of Awiros' correct reads | **Veto-only, disabled** (`secondary_veto.py`); needs development-set evaluation before enabling |
| Result states | new | closes a gate hole and keeps uncertain vehicles | **Added** (candidate profiles) |
| Partial watchlist matching | new | needs snapshot on edge (security) | **Added, off by default**; review-only |
| `layout` conflict blocking | new | a plate read as both one- and two-row is not trustworthy | **Added** |
| Plate-level association across tracker IDs | new | unambiguous-match-only rule | **Added** (candidate) |
| Near-duplicate crop guard | new | stalled vehicle must not self-confirm | **Added** (candidate) |

## Decisions taken without asking (reversible)

1. **Strict format policy in the acceptance path:** an accepted reading that fails strict state-code/district rules is
   *not* CONFIRMED; it becomes a PARTIAL for review. Rare legitimate formats (military, diplomatic, temporary, trade) are
   therefore never auto-confirmed. Safer; costs coverage on those plates, which this system does not claim to support.
2. **Length agreement as a blocker:** the existing gate can confirm a nine-character reading when every frame read ten.
   Candidate profile requires unanimous frame lengths. Baseline keeps the old behaviour so rollback is exact.
3. **Device revocation as policy, not isolation** (see `SECURITY_AND_PRIVACY.md`).
4. **Named profiles, not flag combinations.** Only `baseline`, `candidate`, `candidate-gpu` are tested.
5. **Backend schema:** separate `recognition_results` table; the fingerprint-required `sightings` table is untouched.
6. **Reviewer identity:** SHA-256-digest tokens in configuration, chosen over adding user accounts (no existing user model).
7. **Dead-lettering permanent 4xx rejections** (400/404/409/413/422) instead of retrying forever (an existing limitation
   listed in `CURRENT_STATUS.md`). 401/403/5xx stay queued.
8. **Result policy thresholds are unvalidated starting values** (`result-policy-1`), copied from the existing gate where one existed.

## Removed or consolidated

Nothing was deleted. Consolidation of Pipeline A is **deferred** until the candidate profile passes a sealed evaluation,
shadow comparison and a rollback test on real footage (none can be run in this environment).
