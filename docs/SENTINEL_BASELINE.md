# Sentinel baseline (Phase 0)

Recorded before any behaviour change. Evidence labels: **Measured here** = run in this session;
**Cited** = taken from an existing repository document and not re-measured.

## Repository state at start

| Item | Value |
| --- | --- |
| Working branch | `claude/vibrant-shannon-v8c0p4` (the dedicated feature branch; never `main`) |
| Start commit | `2e5fa7d` (unify local dashboard) |
| After fast-forward to teammate snapshot | `8b8ff91` (`origin/ccr-c5a6ab25-2b62mi`: plate-pipeline, tracker, UI, recording receipts, OCR experiments) |
| Working tree | clean at start; nothing of the user's was modified, reset or deleted |
| Pushed | **No.** Nothing is pushed by this work |

`main` does not contain the plate-pipeline work (`work/plate-pipeline`, `tracker/keep-unconfirmed-vehicles`,
`ui/compact-frame-labels`). The snapshot branch merges them; this work builds on it.

## "Pipeline A" and "Pipeline B" in this repository

No file or document in the repository uses these names. The assumption used here, documented in
`FEATURE_DECISIONS.md`:

- **Pipeline A** = the original per-frame worker path `worker.py` → `vision.py` (`VisionPipeline`, FastALPR,
  `PLATE_READER=temporal-v2|hybrid-v1`). It is the production worker and is **left untouched and still working**.
- **Pipeline B** = the tracked, multi-frame CTC path `live_detection_viewer.py` → `tracked_live_viewer.py` →
  `tracked_ocr.py` → `ctc_fusion.py` → `shared_delivery.py` (adapted Awiros CTC reader). All new work is here.

## Hardware (this build container) — Measured here

| Item | Value |
| --- | --- |
| CPU | Intel Xeon @ 2.10 GHz, 4 cores, 1 thread/core |
| RAM | 15 GB, no swap |
| GPU | **None** (`nvidia-smi` absent) |
| Model weights | **None present** (no `.pt`, `.safetensors`; no torch, no Paddle) |
| Docker | client only; no running daemon |

Consequences: the real-model recorded-video evaluation, the 200-image and 37-crop regressions, GPU throughput
and ONNX parity **could not be run here** (see `IMPLEMENTATION_STATUS.md`, blockers).

## Existing tests — Measured here (before changes)

| Suite | Result |
| --- | --- |
| Edge `edge/tests` | 505 passed, 1 skipped, **2 failed** |
| Backend `backend/tests` | 105 passed |
| Frontend `vitest` + `tsc --noEmit` | 36 passed, type-check clean |
| Tools `tools/test_*.py` | 26 passed |

The 2 pre-existing edge failures, unrelated to this work and left unfixed:

1. `test_svtrv2_experiment…normalisation` – needs `torchvision`, absent from the edge test extras (environment).
2. `test_engineering_page…heartbeat` – the old engineering-console page is now a redirect (HTTP 307/410 in
   `live_detection_viewer.py`); the test still expects the previous status text. Stale test of retired UI.

## Existing evaluation metrics — Cited, not re-measured

| Metric | Value | Source and caveat |
| --- | --- | --- |
| Recorded pilot (Hyderabad, 4 passages, AI-drafted labels) | exact reads 0/3, accepted 0, plate localisation 2/3, speed 0.134× real time on a Ryzen CPU | `data/evaluation/hyderabad-pilot-001.json`; labels by Claude, not independent humans |
| Mumbai clip, 50 AI-drafted crop labels | 2 readable, 14 non-plate, 34 unreadable | `docs/WORK_LOG.md`; exact match rests on n=2 |
| Mumbai 4K, 15 samples, CPU | 60.8 s per analysed frame mean; fusion = 76 % of time; 1.65 GB peak | `docs/WORK_LOG.md` Phase 7 |
| External Indian OCR set (200 frozen refs) | adapted Awiros 137/200 (68.5 %) | `docs/EXTERNAL_OCR_EVALUATION_2026-11-01.md`; single-crop OCR, **not** passage coverage |
| Awiros vs SVTRv2-B on that set | Awiros alone correct on 36 of its 137 correct reads | same document |
| 37 development crops | adapted Awiros 29/37, SVTRv2-B 20/37 | `docs/OCR_ERROR_ANALYSIS_2026-11-01.md` |
| Accepted precision, readable-passage coverage, partial/unreadable behaviour | **never measured on independent human labels** | no sealed set existed |

Not available in the repository: vehicle-detection coverage on human-labelled boxes (0 labelled vehicle boxes in
the pilot), character error rate on full passages, any partial/unreadable measurement (those states did not exist).

## Current pipeline configuration (Pipeline B, recorded service)

`compose.yaml` service `recorded-analysis`: CPU, detector profile `general` (YOLO26s vehicle + plate detector),
adapted Awiros CTC checkpoint `indian-corpus-encoder.safetensors` (SHA-256 prefix `c2ba896d…`), ByteTrack,
acceptance threshold 0.85, ≥3 distinct frames, per-character peak ≥0.75 minimum and ≥2 frames at peak ≥0.6.
Plate options (`reject_non_plates`, `format_validator`, `position_vote`, `two_line`, unconfirmed vehicles) are off
by default. Delivery sends **accepted sightings only**.

## Per-frame processing budget (defined here)

| Profile | Sampling | Budget per analysed sample | Basis |
| --- | --- | --- | --- |
| GPU (`candidate-gpu`) | 500 ms, 250 ms while evidence is short | mean ≤ 500 ms, p95 ≤ 1000 ms end to end; `ProposalPool` hard cap 150 ms and 24 calls per frame for extra proposal sources | must keep up with the sampling interval; **not yet measured** (no GPU) |
| CPU recorded replay (`candidate`) | unpaced, never drops a sample | no real-time budget; report speed vs real time; extra sources (tiles, text detector) **off** until a CPU benchmark shows they fit | prior measurement: 0.13× real time (pilot) and ~60 s per 4K sample |
| CPU live | not supported by measurement | any CPU live use needs its own benchmark first | — |

## Baseline failures and gaps (not hidden)

- Zero accepted plates on every real recording evaluated to date; no independent-human sealed evaluation exists.
- Fusion dominates CPU time (76 %), so extra proposal sources cannot be afforded on CPU without optimising fusion.
- The existing `accepted()` gate can confirm a candidate that is **one character shorter** than every frame read
  (CTC deletion path). Reproduced by a regression test in `edge/tests/test_passages.py`; the candidate profile closes it,
  the baseline profile deliberately keeps it for rollback fidelity.
