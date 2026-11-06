# Automatic crops approaching manual crops

This investigation leaves character recognition unchanged. It compares image
geometry on the original, permitted Mumbai 4K recording and the existing 21
manually selected targets. Results are development evidence, not unseen-video
accuracy, exhaustive vehicle recall, or production readiness.

## What worked

| Method | Targets with a matching native crop at IoU >= 0.50 |
| --- | ---: |
| Current vehicle-first search at the labelled instants | 3/21 |
| Direct overlapping plate-detector tiles, then local re-detection | 10/21 |
| Lightweight text-boundary detector | 13/21 |
| At least one matching proposal from the two tiled detectors | 15/21 |

Manual references have 17 corner polygons and four contextual rectangles.
Fourteen of those 17 corner-labelled targets also have a matching automatic text
quadrilateral at IoU >= 0.50. Native rectangle and quadrilateral measurements
describe different output geometries and should not be interchanged.

These counts associate a reference with an already-produced proposal **after
inference**. They show that useful crop candidates exist; they do not show that
the software automatically selected the right plate or rejected other text.
Background text and unlabelled plates have not been exhaustively annotated, so
proposal precision is unknown. The historical 7/19 crop pairing used different
association criteria and is not comparable with this geometry test.

The main findings are:

- A missed vehicle must not prevent a separate plate search. Native overlapping
  tiles preserve small plates that disappear when the whole 4K frame is fitted
  into a 512-pixel detector input.
- Keep every usable plate proposal, rather than only the first plate in a scene.
- A local second pass around each plate and a small safe margin reduce clipping.
- Some registrations are painted onto the vehicle or use unusual lettering.
  A text-boundary detector can find and straighten these when a detector trained
  to recognise conventional physical plates misses them.
- Neither path solves every example. Target 04 is only partly detected, target
  16 remains missed, and the text detector misses the steep two-row target 06
  that the plate detector finds. Targets 08 and 14 were human-unreadable in the
  prior reading exercise; that does not exclude them from geometry scoring.

## Recommended crop architecture

1. Run the existing vehicle/plate path for inexpensive frequent observations.
2. Periodically run a direct plate scan on overlapping native-resolution tiles.
3. Use text detection as an additional proposal source for missed or poorly
   bounded registrations, including painted registrations. Associate proposals
   with vehicle context without requiring the original vehicle detector to
   approve every proposal.
4. Filter plate text versus branding/signs, and group separated or two-row text.
   This classifier/grouping stage remains to be implemented and evaluated.
5. Retain the native crop, source box and timestamp. Keep the four-corner warp
   separately as an interpolated working image. It is not recovered detail.
6. Track the plate region across frames and retain multiple good views, with
   bounded per-frame work. Validate this on the full recording before enabling
   it in the dashboard processing pipeline.

The new modules are isolated research utilities. They have **not** been enabled
in the video session, sighting delivery, or central alert workflow.

## Models and cost

The existing YOLO26s vehicle detector and YOLOv9-t-512 plate detector are unchanged.
The additional model is the official
[PP-OCRv5 mobile text detector](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det/tree/0d63e78e2b680928f6b1747d76a08db6e645efb7),
Apache-2.0, pinned revision `0d63e78e2b680928f6b1747d76a08db6e645efb7`.
Only its text-localisation network is loaded. No character recognizer is loaded,
and no cloud inference is used. Downloaded model/config/notices total 4,939,860
bytes; no heavier OCR models were restored.

On this RTX 3050 Laptop GPU, the selected plate scan plus refinement averaged
765 ms per tested frame; text detection averaged 822 ms separately. These exclude
video decoding, OCR, delivery and dashboard rendering. Running both at every
video frame is impractical on this laptop. Test a periodic fallback schedule.
Smaller plate tiles with a lower threshold took 1,772 ms per tested frame and
still matched 10/21, so that denser configuration is rejected.

The initial detector-only attempt used the OCR research launcher and hit a
cuDNN version conflict. Running the detector with its own container environment
resolved it. The first text-detector run hit CUDA error 999 at the last sampled
frame; a retry completed all 14 selected frames. This is not an endurance test.

## Review and reproduction

Local review pages, served on loopback port 7777:

- `/automatic-crops.html`: manual, current vehicle-first, tiled, refined, and
  optional contour-warp images.
- `/text-boundaries.html`: manual native, automatic text native, and automatic
  four-corner working view. No plate readings or recognition scores.

Images, detailed reports and acquisition receipts remain under ignored
`data/benchmarks/`. Public aggregate evidence is
`docs/evidence/automatic-crop-geometry-20261106.json`; it contains no plate text.
The recording remains a separately restored permitted asset, not a Git source
file. Matching its hash and existing manual manifests is required to reproduce
the same measurements.

Download the small detector with:

```powershell
python tools/cache_text_localizer.py --output data/benchmarks/text-localizer-20261106
```

For plate-only inference, use the existing edge-worker container with GPU access,
`--workdir /workspace`, `PYTHONPATH=/workspace/edge`, the model volume mounted
read-only at `/models`, source mounts at `/workspace/edge` and `/workspace/tools`,
and the local data folder at `/workspace/data`. Invoke:

```text
python /workspace/tools/compare_automatic_crops.py
  --review /workspace/data/benchmarks/recorded-footage-20261104/manual-review
  --recording /workspace/data/recordings/footage.mp4
  --output /workspace/data/benchmarks/reports/CHOOSE-A-FRESH-NAME
  --device gpu
```

Run text detection in a separate container through the existing isolated Paddle
launcher, with the same mounts and GPU access, without a custom PYTHONPATH:

```text
python /workspace/tools/run_ocr_research.py
  /workspace/tools/compare_text_boundary_crops.py
  --review /workspace/data/benchmarks/recorded-footage-20261104/manual-review
  --recording /workspace/data/recordings/footage.mp4
  --assets /workspace/data/benchmarks/text-localizer-20261106
  --output /workspace/data/benchmarks/reports/ANOTHER-FRESH-NAME
  --device gpu
```

The launcher's name is historical; this command constructs only the text
detector. Model downloads are explicit and inference has no network access.

## Verification and next acceptance gate

22 crop and existing collector tests passed. Both review pages are checked for
21 cards, valid images, navigation, and mobile overflow. Before runtime adoption,
add disjoint footage, exhaustive plate/non-plate region annotations, false
proposal counts, multi-row/grouped registration examples, temporal association,
and a full-video interruption/endurance test. Keep OCR tuning separate until
crop coverage and boundaries are reliable.
