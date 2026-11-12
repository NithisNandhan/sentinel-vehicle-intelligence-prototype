# Plate detector comparison — 12 November 2026

The current YOLOv9-tiny-512 remains the baseline. None of the tested replacements
improved the total exact readings on this small Mumbai development diagnostic.
No production detector or OCR weights were changed.

| Detector | Reference plates found / 18 | Exact native readings / 18 | Exact rectified readings / 18 | Median detection per vehicle ROI |
|---|---:|---:|---:|---|
| YOLOv9-tiny-512 | 10 | 7 | Not produced | 25.2 ms, GPU |
| YOLO11-medium plate | 9 | 7 | Not produced | 30.0 ms, GPU |
| YOLO11-large plate | 9 | 5 | Not produced | 37.8 ms, GPU |
| YOLO11-extra-large plate | 8 | 5 | Not produced | 59.6 ms, GPU |
| IWPOD-NET | 8 | 6 | 7 | 150.9 ms, CPU |

IWPOD rectification recovered the angled reference 2 that its native crop and
the baseline failed to read. It lost other references, so it is a useful
candidate for selective rectification, not a justified wholesale replacement.
The baseline's seven correct references and IWPOD's seven rectified correct
references overlap on six: their union of eight is an **oracle diagnostic
ceiling**, not the output of an implemented automatic selector.

## What was actually tested

- 149 unchanged, automatically detected vehicle regions at 14 frozen timestamps
  in the permitted 4K Mumbai recording. Source SHA256:
  `6b7064605b3d4d12c6c1da4566cbcec1cd2e3e3bd7b77814865fe9df22045f65`.
- All model inputs came from the same saved capture. They did not receive manual
  plate boundaries or correct strings. Existing reviewed vehicle exclusions are
  inherited from that capture; this is not an untouched whole-scene run.
- The same 18 existing reviewed references were used after inference. The best
  geometric match at the exact timestamp was selected using IoU >= 0.30, with
  detector confidence breaking ties. OCR answers never selected a match.
- Coverage of 95% of a reference box is a geometric proxy, not proof that every
  character is present. Only 16/18 reference boxes had >=95% coverage by a saved
  automatic vehicle ROI. Upstream vehicle regions limit every detector here.
- OCR was the frozen Indian-adapted Awiros checkpoint, SHA256
  `c2ba896d994b174071c95ace0f5ffc8a61cec3fc6eecce8f0bc5feb7dec4456d`.
  It ran on GPU with the unrestricted CTC-prefix top decoder, no registration
  repair, no training, and no threshold tuning against these answers.
- YOLO models used their native input configurations and confidence 0.25.
  IWPOD used the author's `fullimage` preset (maximum width 480), confidence
  0.35, and published 240x80 rectification. These are preset comparisons, not
  an equal-resolution or exhaustive threshold-sweep study.
- YOLO detection and all OCR ran on the RTX 3050 6 GB GPU. IWPOD detection used
  isolated TensorFlow CPU 2.15.1 for legacy Keras compatibility. Timings include
  preprocessing and postprocessing, exclude warmup, crop-file writes and OCR.
  CPU/GPU times are observed local costs, not an architecture speed ranking.
  ONNX Runtime GPU 1.30.0, Torch 2.14.0, OpenCV 5.0.0.93 were recorded.
- Other predicted objects are unlabelled. Full-scene false-positive precision,
  exhaustive recall, and generalization accuracy **cannot be computed** here.
- This is single-frame detector/OCR diagnosis. The earlier 12/18 whole-clip
  reconstructed found-at-least-once score is a different experiment.

## Review and reproduce

Local review: `http://127.0.0.1:7809/`, reference selector 1–18, manual reference,
native crops, IWPOD rectifications and actual/read text side by side.
Local output: `data/benchmarks/detector-corners-20261112/results/`.
The aggregate-only tracked result is
`docs/evidence/plate-detector-comparison-20261112.json`. Images and transcriptions
stay in ignored local benchmark directories.

The comparison uses an isolated container named
`sentinel-detector-corners-20261112`, with the project at `/workspace` and existing
model-cache volume at `/models`. In that container:

```sh
python tools/compare_plate_corners.py prepare
python tools/iwpod_adapter.py prepare --source data/benchmarks/detector-corners-20261112/iwpod-source
# TensorFlow installation is isolated from the application runtime.
python -m pip install --target /models/experiments/iwpod-deps tensorflow-cpu==2.15.1
PYTHONPATH=/models/experiments/iwpod-deps CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 python tools/compare_plate_corners.py infer --model iwpod --device cpu
# Run infer followed by ocr for each model; use a fresh --output on replays.
python tools/compare_plate_corners.py infer --model yolo11m_plate --device cuda
python tools/compare_plate_corners.py ocr --model yolo11m_plate --device cuda
python tools/compare_plate_corners.py score
```

Model names: `yolov9t512`, `yolo11m_plate`, `yolo11l_plate`,
`yolo11x_plate`, `iwpod`. Run `ocr` for IWPOD too. The baseline checkpoint and
frozen Awiros assets must already exist in the model cache. Inference never
downloads assets. IWPOD's explicit preparation command validates fixed hashes
and preserves the upstream MIT license. YOLO downloads are pinned to the
publisher revision and SHA256, with AGPL-3.0 provenance in the local receipt.

The YOLO11 publisher discloses training/test contamination; its advertised
metrics were not used for choosing a winner:
[model card](https://huggingface.co/morsetechlab/yolov11-license-plate-detection).
IWPOD source and weights:
[author repository](https://github.com/claudiojung/iwpod-net), pinned commit
`8cb667e54baef7ccb14bee33e293624627b2838a`.

## Decision

Keep the existing detector. First compare native versus geometrically normalized
inputs and preserve strong original evidence during multiframe aggregation.
IWPOD can be evaluated as a bounded fallback on angled failures; these results
do not justify replacing the working detector or claiming improved full-video
accuracy. Twenty focused geometry, decoder and asset-integrity tests passed.
