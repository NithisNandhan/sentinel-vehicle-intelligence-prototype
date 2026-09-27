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
The original is retained. A separate primary-source Indian CCTV experiment is
being prepared; its results must be documented before any further recommendation.

## Indian CCTV adaptation protocol

The BMD-45 subset is acquired and quality-gated: 999 train / 148 validation /
150 reserved test images, with 13 actual publisher vehicle subclasses. One
dimension-mismatched image and two cross-split similar images are excluded;
originals are preserved, explicit file lists prevent re-admission. Validation
selects the checkpoint; fixed test labels never enter training. The source is
CC BY 4.0; no Sentinel recordings or account credentials are used.

Training uses YOLO26s-640, batch four, AdamW .0002, 20 requested epochs,
patience six, first ten layers frozen, FP32, seed 42 and the same mild
augmentation protocol as the pilot. Results remain pending until the run and
paired reserved tests complete. Source consensus labels, not new human truth.
Camera identities and pretraining overlap are not independently verified.

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

Containers use `sentinel-detector-training:local`, built from
`edge/Dockerfile.detector-training`; production image remains unchanged. Mount
the repository at `/workspace`, the existing model volume at `/models` and
`sentinel_detector_training` at `/training`. For evaluation set working
directory `/workspace/edge` and `PYTHONPATH=/workspace/edge`; loading `/app`'s
installed copy would hide newly added benchmark modules. Training/evaluation
containers have network disabled and do not receive `.env` or stream credentials.
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

Model selection never changes `.env`, monitored cameras, live defaults or OCR.
Datasets, weights, training-source checkout and image review sheets remain local,
not committed to GitHub. See [DETECTOR_DATASETS.md](DETECTOR_DATASETS.md) for sources.
