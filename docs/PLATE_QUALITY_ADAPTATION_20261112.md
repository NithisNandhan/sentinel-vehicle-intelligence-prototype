# Improving the existing plate detector — 12 November 2026

The original YOLOv9-tiny-512 weights remain the default. A new optional
`general_quality` profile searches bounded native views while protecting the
original detections. It improved the selected Mumbai diagnostic from 10 to 11
reference detections and from 7 to 8 exact readings. This is an input-search
improvement, not a new OCR model or a successful weight replacement.

## Measured results

Same 149 frozen automatic vehicle regions, 14 timestamps, 18 development
references, unchanged Indian-adapted Awiros OCR and unrestricted decoder:

| Method | References found / 18 | Exact readings / 18 | Median detection per vehicle ROI |
|---|---:|---:|---:|
| Original tiny detector, single view | 10 | 7 | 25.2 ms GPU |
| New fine-tuned tiny weights | 8 | 5 | 14.5 ms GPU, raw ONNX |
| Original weights, protected native views | 11 | 8 | 96.6 ms GPU |

The runtime difference for the fine-tuned weights includes a different raw-ONNX
adapter, so it is not evidence of faster neural-network arithmetic. Quality
search used 863 detector calls for 149 vehicle regions, with at most six per
region. All 60 original crop hashes and their OCR readings were checked to
survive unchanged. Reference 12 is the additional exact reading.

Reserved publisher-labelled test: 77 original images, 83 annotated plates per
condition. The three conditions are repeated views of the same images, not
249 independent vehicles. Results below use fixed confidence .25 and one-to-one
IoU .5 matching:

| Condition | Original matched / 83 | New weights matched / 83 | Protected views matched / 83 |
|---|---:|---:|---:|
| Native | 76 | 75 | 79 |
| Quarter resolution | 53 | 59 | 53 |
| Motion blur | 54 | 52 | 60 |

Extra search has a precision cost. On native images, predictions unmatched to
publisher labels increase 13 -> 17 and annotation precision falls .8539 ->
.8229. On blurred images, unmatched predictions rise 8 -> 15 and annotation
precision falls .8710 -> .8000. Annotation omissions may contribute; these are
not independently adjudicated false-positive counts. The new profile therefore
remains optional and is suitable for reviewing additional candidates, rather
than a silent global replacement. Recognition acceptance, watchlist rules and
temporal evidence requirements are unchanged.

These are selected single-frame diagnostics. They must not be compared directly
with the older 12/18 reconstructed found-at-least-once score over a whole clip.
Full-video recognition, two-feed throughput and independent camera
generalization have not been established by this experiment.

## What changed

- `plate_scale_search.py`: full native ROI, then at most five additional native
  views. Large regions use spatially spread 512/640-pixel windows; smaller
  regions use overlapping corner/lower views. No source pixels are invented.
- Original full-view detections are protected. Extra detections cannot replace
  overlapping originals just because confidence is higher. Artificial tile-edge
  cuts, very small extra detections and implausible extra shapes are excluded.
  There is no new filtering or NMS of original full-view predictions; the
  bounded output cap is 32 proposals per vehicle region.
- `general_quality` uses the existing YOLO26s vehicle model and the exact same
  parent plate checkpoint. CPU/GPU selection stays explicit. Existing profiles
  and the default `general` profile are unchanged.
- The existing targeted-search wrapper reports the actual inner model calls and
  avoids recursively tiling a region that the quality profile already searched.
- Alternative crops from one frame remain one temporal observation. Neither
  plate acceptance thresholds nor automatic alert eligibility were loosened.

The first native-view experiment was rejected: size filtering accidentally
discarded small original detections, and confidence-based deduplication could
replace a good original crop with a worse tile. Both mechanisms are covered by
regression tests. That run remains separate in local artifacts.

## Fine-tuning experiment and rejection

Training used the existing grouped Gujarat/Make ML publisher-labelled export,
with **360 original training images plus 352 context-crop alternatives**.
Validation had 130 originals plus 129 context alternatives. The 77 test originals
were unchanged. Crop alternatives inherit source groups and partitions; hashes,
label bounds and cross-partition exact duplicates are checked. Partial cuts of
another annotated plate trigger a full-image fallback. No Mumbai frames,
manual plate boundaries or plate strings were used for training.

The current publisher-provided trainable YOLOv9-tiny counterpart was trained for
12 epochs on the RTX 3050 6 GB: AdamW 0.00005, box loss gain 9, mosaic .05,
first ten backbone layers frozen, batch four, FP32, seed 42. The run took
376.35 seconds. Checkpoint selection used validation fitness only. It improved
some reduced-resolution box metrics but regressed native, blur and Mumbai
recognition, so it **was not promoted**.

The export passed zero-input CPU parity and **48 real-image CPU/CUDA parity
checks**, with CUDA TF32 disabled. Rejected export SHA256:
`2497adf041acbbca0a9f966b3cc6de2aacd595a1aac2dcf3d486ff8c088e3fad`.
Training artifacts are retained in the existing research volume at
`/training/runs/plate-context-20261112`, not loaded by the quality profile.
Source pretraining overlap and camera identity are unknown; grouped partitions
do not establish an untouched camera holdout.

## Use and review

Local before/after viewer: `http://127.0.0.1:7810/`. Select reference **12** for
the recovered plate. `baseline_scale` means the protected native-view method;
`yolov9t512` means the original single-view detector. Images/readings remain
in ignored local benchmark directories.

To explicitly run the recorded application with the optional profile:

```powershell
.\scripts\start-recorded-analysis.ps1 -Device gpu -DetectorProfile general_quality -Unpaced
```

This builds the updated application runtime and still requires selecting a
permitted recording and starting analysis in the dashboard. It uses already
cached parent weights; no new model download is needed. Omit `-Unpaced` to pace
playback. Use `-DetectorProfile general` to return to the default search.
Only the model preflight and a real-image adapter smoke test were run here;
the application was not restarted and a whole-clip quality-profile session was
not claimed complete.

Reproduction uses `tools/prepare_plate_roi_adaptation.py`, the bounded options
in `tools/run_detector_finetune.py`, real-image validation in
`tools/export_plate_detector.py`, `tools/compare_plate_corners.py` and
`python -m sentinel_edge.detector_adaptation_eval --scale-search`. Use fresh
output/run names. Original training defaults are preserved.

Local artifacts: `data/benchmarks/plate-context-adaptation-20261112/`.
Aggregate-only public evidence:
`docs/evidence/plate-quality-adaptation-20261112.json`.
No images, plate transcriptions or credentials are included in that evidence.

Verification: 123 focused tests passed (the localhost HTTP test needed an
unsandboxed rerun), real-model GPU profile smoke test passed, and the shared
vehicle/plate/OCR GPU preflight passed. Browser inspection verified all 18
references, both methods, loaded images and the new reference-12 reading.
