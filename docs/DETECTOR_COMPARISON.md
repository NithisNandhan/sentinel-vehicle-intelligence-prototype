# Detector comparison and isolated fine-tuning — 27 September 2026

## Initial selection

**Vehicle candidate: YOLO26s at 640. Plate parent: YOLOv9-tiny at 512.**
This is a bounded comparison of the eight previously discussed models, not
every detector published worldwide. Newer or larger did not always win.

Initial inputs: 13 permitted Bapatla/NIVU pictures, representing 11 distinct
target vehicles. Each picture has an approximate AI-reviewed target vehicle
and plate box. Only the target is labelled, not every background object. Native,
half-resolution, quarter-resolution, Gaussian blur and horizontal motion blur
produce 65 diagnostic cases. These are **not 65 independent vehicles** and are
not a held-out, independently human-labelled accuracy dataset.

### Vehicle results

Hits mean target box overlap IoU >= 0.5. Warmed whole-adapter median latency,
not end-to-end pipeline latency or guaranteed frames per second:

| Model | Input | CPU hits / 65 | CPU median ms | GPU hits / 65 | GPU median ms |
|---|---:|---:|---:|---:|---:|
| YOLOv8n | 640 | 56 | 30.50 | 56 | 11.07 |
| YOLOv8n | 960 | 51 | 74.07 | 51 | 15.16 |
| YOLO26n | 640 | 60 | 33.33 | 60 | 12.96 |
| YOLO26n | 960 | 55 | 93.05 | 55 | 13.22 |
| YOLO26s | 640 | 63 | 72.41 | 64 | 11.79 |
| YOLO26s | 960 | 61 | 156.26 | 61 | 20.87 |

YOLO26s-640 has the highest target coverage in this small sample. Its GPU
median happens to be below nano's in this run; do not infer that small always
runs faster than nano. Production services were not stopped for these tests.
One CPU/GPU hit differs: numerical/threshold variation exists; exact parity
is not claimed. Bigger input size did not improve this particular sample.

### Plate results

| Model | Released input | Hits / 65 | Hits IoU >= .75 | CPU median ms | GPU median ms |
|---|---:|---:|---:|---:|---:|
| YOLOv9-tiny | 512 | 61 | 51 | 43.28 | 19.96 |
| YOLOv9-small | 608 | 60 | 53 | 147.74 | 31.46 |
| Plate-trained YOLO11-medium | 640 | 58 | 46 | 420.23 | 37.90 |
| Plate-trained YOLO11-large | 640 | 53 | 36 | 616.39 | 44.45 |
| Plate-trained YOLO11-extra-large | 640 | 55 | 42 | 1183.07 | 63.08 |

CPU and successful GPU plate runs have the same coverage counts. Small gives
slightly more tight alignments, but tiny has better overall coverage and lower
latency. Medium helps one quarter-resolution case but loses native/motion cases.
YOLO11 medium missed native HP2 and JH1 entirely. This provides no justification
for a blanket larger-plate-model replacement.

## Measurement and integrity

- RTX 3050 6 GB Laptop GPU; Python 3.11; Torch 2.14.0+cu130;
  Ultralytics 8.4.162; ONNX Runtime GPU 1.30.0; OpenCV 4.14.0.94.
- Confidence .25; NMS IoU .45 where applicable; three warm-ups excluded;
  CPU library threads capped at two; GPU results synchronized before timing.
- YOLO26 uses the released Ultralytics end-to-end prediction behavior.
  Plate models retain their released resolutions; this compares model-plus-input
  configurations, not architectures at an identical input size.
- Actual ONNX GPU sessions list CUDAExecutionProvider. This does not establish
  that every operator runs on GPU or validate combined Paddle OCR/GPU execution.
- The first plate GPU attempt failed because the OCR launcher selected CUDA 12
  libraries while ONNX Runtime required CUDA 13. A separate process using the
  already installed CUDA 13 libraries completed all five plate GPU tests.
- Public originals and frozen manifest checksums are checked before inference.
  HF YOLO11 ONNX files are revision-pinned and publisher LFS SHA256-verified.
  Official YOLO26 and publisher YOLOv9 assets have local SHA256 receipts.
- The YOLO11 adapter uses standard RGB FP32 letterboxing and strict single-class
  output-shape checking. Tests cover preprocessing and native coordinate restore.
- Unmatched boxes in the initial sample are **not verified false positives**:
  background objects are not exhaustively labelled.

No OCR string accuracy, live-camera accuracy, tracking recall, throughput at
80,000 cameras, or complete detection of every traffic participant is measured
by the initial comparison.

Local detailed reports (ignored by Git):
`data/benchmarks/reports/detector-comparison-vehicle.json`,
`detector-comparison-plate.json`, and `detector-comparison-plate-cuda.json`.

## Completed plate adaptation

The selected plate model's publisher supplies a trainable PyTorch counterpart.
Its pickle globals were inspected without executing pickle. The pinned original
YOLOv9 source was used only in a network-disabled research container with no
credentials mounted. Model/source licensing notices remain separate from this
application's licence; original YOLOv9 training code is GPL-3.0.

Training: 20 epochs, 360 images, validation 130 images, reserved test 77 images;
512 input, batch four, AdamW .0002, first ten backbone layers frozen, FP32,
seed 42, mild augmentation and limited mosaic. Training completed in 369.37 s.
Checkpoint selected by validation fitness, not by test performance. Test has
34 Gujarat and 43 broader Make ML images, with 83 publisher plate annotations.
Near-image/filename/time grouping and exact duplicate/conflicting-box checks
precede splitting. This is not a verified camera- or registration-identity split.

| Reserved test condition | Parent matched / 83 | Adapted matched / 83 | Parent annotation precision | Adapted annotation precision |
|---|---:|---:|---:|---:|
| Native | 76 | 75 | .8539 | .8824 |
| Quarter resolution | 53 | 72 | .7361 | .8471 |
| Motion blur (9 px) | 54 | 69 | .8710 | .8846 |

At stricter IoU .75, native aligned plates improve from 36 to 46; quarter from
17 to 36; motion from 30 to 42. At IoU .5, aggregate annotation recall across
231 stress cases rises from .7349 to .8675. Aggregate cases remain only 77
original images, not 231 independent vehicles. Whole-adapter GPU medians are
19.74 ms parent and 21.10 ms adapted. Parent uses released ONNX; candidate uses
original Torch, so timings include runtime differences.

**Recommendation: keep adapted YOLOv9-tiny as an experimental low-quality
candidate, with the original retained.** It improves stressed inputs, but loses
one native match. Do not silently make it the production default or claim 90%
plate-string recognition. Publisher annotations are not newly verified human
truth; pretrained data overlap and camera generalisation remain unknown.

Research compatibility fixes: use checkpoint architecture instead of upstream
placeholder `yolo.yaml`; disable decorative font/AMP-reference downloads in the
offline trainer; alias removed NumPy `trapz` to equivalent `trapezoid`. Earlier
failed-run directories are preserved. No metric thresholds or ground truth were
changed to make training succeed.

Local adaptation report: `data/benchmarks/reports/plate-detector-adaptation-test.json`.
Training artifacts remain in Docker volume `sentinel_detector_training`, under
`/training/runs/plate-r3`; receipt/checkpoint hashes are recorded there.

### Indian-only plate-location breakdown

The Gujarat reserved subset has 34 images / 34 plate annotations, separate from
the 43 Make ML images / 49 annotations. At IoU .5:

| Gujarat condition | Parent matched / 34 | Adapted matched / 34 |
|---|---:|---:|
| Native | 31 | 33 |
| Quarter resolution | 18 | 31 |
| Motion blur | 15 | 29 |

Tight IoU .75 matches improve 16 -> 22 native, 6 -> 16 quarter and 6 -> 24
motion. This is a small publisher-labelled regional plate-location test, **not
plate-text recognition or a calibrated confidence claim**. The mixed-country
native regression comes from Make ML (45 -> 42 matches), not this Gujarat fold.
The separate Bapatla/NIVU tight-box regression still prevents a global switch.

### Export and diagnostic regression check

The candidate was exported to plain ONNX `[1,5,5376]` (xywh plus probability,
external NMS). It is **not interchangeable** with the original end-to-end ONNX
backend. SHA256 is `d0416773dccb3ec8c852390e03a8b6f177850e0f693c2a529fc49c7961525611`.
Forty-eight real-image checks (eight images, three conditions, CPU and GPU)
pass raw-output parity with Torch at rtol .002 / atol .05. Initial GPU parity
failed with ONNX's default TF32; explicitly disabling `use_tf32` resolves it.
Torch TF32 is also disabled in this check. GPU provider was verified.
See the [ONNX Runtime CUDA options](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#use_tf32).

On the initial 65-case diagnostic, adapted ONNX finds 61 targets at IoU .5,
equal to the parent's 61, but tight IoU .75 matches drop **51 to 21**. Native
finds 13/13, quarter 12/13, motion 10/13. Median adapter time is 12.72 ms.
These approximate target annotations are not independently verified accuracy,
but the tight-box regression is a reason **not to promote it globally**. Wider
box predictions may admit unwanted background to OCR. Improvements on the
publisher-held-out test do not erase this cross-source regression.

Export/regression evidence: `plate-detector-export.json`,
`plate-export-parity.json`, `plate-export-regression.json` in local reports.

## Completed small vehicle adaptation pilot

YOLO26s: 20 epochs, first ten layers frozen, .0002 AdamW, 640 input, batch four,
FP32; 117 training images (including source-provided variants), 11 validation,
six reserved test frames from one Madrid camera. Source-frame variants stay
together. The ZIP has numeric vehicle categories but no names file, so neutral
names are used; evaluation pools vehicle boxes without claiming class semantics.

On those six test frames, 247 publisher vehicle boxes:

| Condition | Parent matched | Adapted matched |
|---|---:|---:|
| Native | 94 | 209 |
| Quarter resolution | 46 | 153 |
| Motion blur | 58 | 77 |

Native annotation precision is .9038 parent / .9048 candidate. Motion-blur
precision falls .7342 -> .6638, and tight IoU .75 matches fall 36 -> 8 under
motion. This is a significant regression boundary despite improved IoU .5
coverage. Six frames from one camera cannot establish Indian CCTV readiness.
The original is retained. The separate primary-source Indian CCTV experiment
is complete; its results and regression boundaries are documented below.

## Indian CCTV adaptation protocol

The BMD-45 subset is acquired and quality-gated: 999 train / 148 validation /
150 reserved test images, with 13 actual publisher vehicle subclasses. One
dimension-mismatched image and two cross-split similar images are excluded;
originals are preserved, explicit file lists prevent re-admission. Validation
selects the checkpoint; fixed test labels never enter training. The source is
CC BY 4.0; no Sentinel recordings or account credentials are used.

Training uses YOLO26s-640, batch four, AdamW .0002, 20 completed epochs,
patience six, first ten layers frozen, FP32, seed 42 and the same mild
augmentation protocol as the pilot. Source consensus labels, not new human truth.
Camera identities and pretraining overlap are not independently verified.

The initial Indian run hit `CUDA_ERROR_UNKNOWN` after ten saved epochs. A fresh
container passed CUDA availability and a GPU matrix-operation smoke test, and
the last checkpoint resumed the original 20-epoch protocol. Services were not
restarted and saved checkpoints were not discarded. Recovery uses model,
optimizer and scheduler resume; restarted augmentation RNG is not claimed
bitwise identical to an uninterrupted run. A driver failure is not evidence of
a model accuracy problem. Recovery-session wall time was 2,290.76 s; the first
ten completed epochs took 853.061 s according to their CSV. Combined recorded
training work is about 3,143.82 s, excluding the failed unsaved work and downtime.
The actual receipt's `elapsed_seconds` records the recovery session only; future
resume runs explicitly separate session and combined accounting.

### Reserved Indian CCTV results

150 original reserved images contain 1,592 labelled vehicle instances per
condition. Native, quarter-resolution and motion blur are repeated views of
these same images, not 4,776 distinct independent vehicles. Box matches ignore
subclass labels and use IoU .5 with confidence .25; YOLO26 retains its released
end-to-end prediction behavior, rather than traditional NMS.

| Condition | Parent matched / 1592 | Indian adapted matched / 1592 | Parent coverage | Adapted coverage |
|---|---:|---:|---:|---:|
| Native | 787 | 1263 | 49.43% | 79.33% |
| Quarter resolution | 788 | 1239 | 49.50% | 77.83% |
| Motion blur | 629 | 1194 | 39.51% | 75.00% |

Tight IoU .75 matches improve 596 -> 1,104 native, 584 -> 1,105 quarter,
461 -> 1,018 motion. Aggregate box coverage is .4615 parent / .7739 adapted;
annotation precision is .7411 / .7293. Native annotation precision is .7348 /
.7305. Increased coverage is not a claim of zero false positives or accurate
vehicle subclass classification. Source consensus may omit some objects.

Native localisation by ground-truth subtype reveals the trade-off:
two-wheelers improve **242 -> 601 / 777**, three-wheelers **122 -> 232 / 270**;
hatchbacks fall 114 -> 106 / 127, sedans 75 -> 67 / 84, SUVs 54 -> 52 / 61,
MUVs 27 -> 23 / 32, vans 15 -> 12 / 18. Rare classes remain weakly sampled.
The comparison is class-agnostic localisation, not a predicted-type confusion
matrix. Overall gains must not hide the car-category regressions.

Best validation class-aware mAP50 was .48083 and mAP50-95 .37347; these are
validation metrics, not the reserved fixed-threshold box coverage above.
Checkpoint SHA256:
`a6d7cb9b39d77ac5f2b494ac4e264454d3d5bc13e6b78d92700671c658a36a5a`.
Full evidence: `vehicle-detector-adaptation-bmd-test.json` in local reports,
and `/training/runs/vehicle-bmd/training-receipt.json` in the research volume.

### Cross-source regression and recommendation

On the earlier 13-image / 65-stress-case development diagnostic, parent finds
**64/65**, Indian-adapted **49/65**. Native target coverage falls 13/13 -> 10/13;
tight matches fall 64/65 -> 45/65 overall. This is a material domain-regression
boundary. Retain **YOLO26s-640 as the general vehicle candidate** and the Indian
adaptation as a separately identified **CCTV specialist experiment**, not a
universal replacement. Additional integration must preserve car coverage and
map the 13-class output correctly before tracking/OCR.

Recorded adapter medians on BMD are 56.50 ms parent / 47.49 ms adapted. These
runs had other host applications and prototype services active, with observed
memory/GPU contention. They are not isolated throughput claims and are not
comparable directly to the initial small-image 11.79 ms run. Cross-source test
logged an NMS time-limit warning during warm-up/unmeasured work: all timed
case latencies were <=327.70 ms parent / <=115.56 ms adapted, below the reported
2,050 ms limit. No timeout occurred in the recorded timed-case intervals.
Cross-source evidence: `vehicle-bmd-cross-source-regression.json`.

The baseline test includes COCO bicycle in addition to car, motorcycle, bus and
truck, since BMD labels bicycles. BMD also includes a dedicated three-wheeler
category absent from COCO. Paired metrics pool vehicle boxes; they are not a
claim of equal-taxonomy subclass accuracy. A subclass checkpoint must **not** be
copied over the production COCO model: the current pipeline's numeric class
filter would interpret these IDs incorrectly. Mapping, tracker consistency and
live validation would be separate integration gates.

## Reproduction entry points

- `tools/cache_detector_comparison.py`: isolated model cache and provenance.
- `tools/prepare_detector_comparison.py`: freeze small target diagnostics.
- `python -m sentinel_edge.detector_comparison`: CPU/GPU comparisons.
- `tools/acquire_detector_datasets.py`: specifically audited dataset acquisition.
- `python -m sentinel_edge.detector_training_data`: grouped native training export.
- `edge/Dockerfile.detector-training`: separate research dependencies, no live replacement.
- `tools/run_detector_finetune.py`: bounded training with preserved prior runs.
- `python -m sentinel_edge.detector_adaptation_eval`: paired reserved-image tests.
- `tools/acquire_bmd_subset.py`, `tools/guard_bmd_splits.py`: bounded Indian CCTV
  acquisition and pre-training transitive near-image quarantine.
- `tools/export_plate_detector.py`, `tools/validate_plate_export.py`: strict raw
  ONNX export, real-image FP32 parity and previous-diagnostic regression checks.
- `tools/validate_vehicle_adaptation.py`: previous-source vehicle regression
  check; `tools/render_detector_summary.py`: static viewer of actual JSON reports.

Containers use `sentinel-detector-training:local`, built from
`edge/Dockerfile.detector-training`; production image remains unchanged. Mount
only `edge` at `/workspace/edge` (read-only), `tools` at `/workspace/tools`
(read-only), permitted benchmark data at `/workspace/data/benchmarks`, the
existing model volume at `/models` and `sentinel_detector_training` at
`/training`. For evaluation set working
directory `/workspace/edge` and `PYTHONPATH=/workspace/edge`; loading `/app`'s
installed copy would hide newly added benchmark modules. Training/evaluation
containers have network disabled; training mounts never include `.env` or stream
credentials. Initial diagnostic/export validation used a broader repository
bind mount (the harness does not read `.env`); subsequent evaluation uses the
scoped mounts above to exclude the credential file structurally.
For ONNX CUDA evaluation use the installed CUDA 13 / cuDNN library paths;
the Paddle CUDA 12 research launcher is not suitable for these ONNX tests.

Selected run entry points (inside the isolated container):

```text
python /workspace/tools/run_detector_finetune.py --kind plate --name plate-r3 --epochs 20
python /workspace/tools/run_detector_finetune.py --kind vehicle --dataset vehicle-bmd --name vehicle-bmd --epochs 20
python -m sentinel_edge.detector_adaptation_eval --kind plate --root /training/plate --checkpoint /training/runs/plate-r3/weights/best.pt --report /workspace/data/benchmarks/reports/plate-detector-adaptation-test.json
python -m sentinel_edge.detector_adaptation_eval --kind vehicle --root /training/vehicle-bmd --checkpoint /training/runs/vehicle-bmd/weights/best.pt --report /workspace/data/benchmarks/reports/vehicle-detector-adaptation-bmd-test.json
```

Use a new explicit run name for reruns; prepared datasets and checkpoints are
preserved instead of silently overwritten. YOLOv9 training requires the pinned
original source under `/models/experiments/yolov9-training`. Current results use
the hashes recorded in each local training receipt, not a newly downloaded
latest source or regenerated manifest. An export is likewise never overwritten.

For an interrupted (not completed) vehicle run only, add `--resume`. The harness
checks the existing training settings and rejects a changed data path, epoch
budget, batch size, input size, freeze/seed/precision or learning rate. It
preserves saved weights and resumes optimiser state; it does not silently start
a new experiment. Completed runs require a new explicit name.

Local human-readable results: `data/benchmarks/reports/detector-summary.html`,
served by the existing localhost-only benchmark server at
<http://127.0.0.1:7777/detector-summary.html>. It displays actual JSON results,
not a new live feed or synthetic accuracy demonstration.

Model selection never changes `.env`, monitored cameras, live defaults or OCR.
The full edge test suite passed in both the research runtime and a fresh Python
environment installed only from declared test dependencies: 219 tests and five
subtests, 24.77 s for the clean run. Missing Pillow/PyAV test dependencies were
fixed before this successful rerun. All 25 experiment tool files parsed
successfully. The local viewer was checked in the browser, including regression
warnings; these software checks are not additional live-camera validation.
Datasets, weights, training-source checkout and image review sheets remain local,
not committed to GitHub. See [DETECTOR_DATASETS.md](DETECTOR_DATASETS.md) for sources.
