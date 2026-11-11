# Native plate crop recovery and before/after review

The new crop-recovery paths remain opt-in. They improve the full-plate crop of
the reported PT0111 example, but **do not improve the eighteen-label exact-reading
coverage: previous 12/18, new 12/18 (66.7%)**. No previously correct labelled
reading is lost. This development result does not establish full-video precision
or performance on unseen footage. Operator defaults and central alerts are unchanged.

## Implemented methods

- Inspect related enclosing text proposals rather than accepting only the first
  high-confidence detector fragment. A focused retry uses surrounding native
  pixels with the existing text model, bounded to two retries per search frame.
- Look for closed quadrilateral plate boundaries around a fragment. Refinement
  must retain all original seed pixels; rectification is blocked when it would
  cut detected seed glyphs. A separate positive plate in the extension vetoes growth.
- Prefer crop completeness before sharpness when retaining observations, while
  restoring the original image-quality weights for CTC/reconstruction. Alternatives
  in one frame never become additional independent temporal observations.
- Add an optional character-row extent refinement. It follows character-sized
  image components along the seed's estimated text baseline and extends the
  native crop to include connected components. Any original confident OCR character
  is required; plate answers and manual labels are not used. Another detected
  plate vetoes growth. Enclosed character holes cannot outvote whole glyphs.

These are adaptations of coarse-to-fine and boundary-refinement ideas, not a
reproduction of a published learned detector or a new model fine-tune. Relevant
sources are [vehicle–plate relation mining](https://arxiv.org/abs/2010.14266) and
[edge-based plate-box refinement](https://arxiv.org/abs/1601.05610).

## What was measured

The source is the receipt-bound, user-supplied Pexels Mumbai recording, not a
genuine Sentinel stream. All model weights remain fixed. Plate labels and their
reference images are loaded only by separate scoring/viewing tools after inference.

The full detection/recovery trial decoded all 928 source frames, used the
original 124 search timestamps, and completed in 489.25 seconds after model
loading. It produced 302 evidence histories and used 578 original-input OCR
calls. Its found-anywhere score remained 12/18. A GPU attempt aborted with an
unknown CUDA error; the successful retry kept plate detection and OCR on GPU,
with text boundaries on CPU. The retry exposed a Paddle CPU setup bug: the legacy
memory-optimisation pass is now enabled only for GPU predictors. Real CPU text
inference was verified after that fix. Device/budget changes limit causal
comparison of the full detection trial.

The subsequent controlled crop-only replay isolates the stroke-boundary change:
all 928 source frames were decoded against the frozen crop inventory, with
existing tracks, sample timestamps and fusion weights preserved. Three of 493
retained native originals changed, and 490 original OCR posterior sets were reused.
All 301 unchanged crop groups retained exactly their previous raw readings.
Two of the three changed crop histories are outside the eighteen human labels;
their new readings remain unverified and are not added to the denominator.
No new plate detections or frame votes were introduced. Processing took 285.42
seconds after OCR loading, including reconstruction and parent-evidence replay;
this is not a fresh detection throughput measurement.

For numbered reference 5 / PT0111, the automatic crop expands from **180×116 to
369×137 pixels**, recovering the missing prefix. The resulting top reading still
gets two letters wrong. This history has only one retained original, so its
reconstruction branch uses the available evidence; it is not evidence of a
successful multi-frame reconstruction for this particular vehicle. Crop recovery
is visible, but complete OCR correctness remains unresolved.

The user's score counts a label if its full number appears as any active
reconstructed top output anywhere in the clip, once per reference. Timestamp,
confirmation threshold and crop position do not gate this coverage score, and
extra incorrect guesses do not reduce it. The original exact-timestamp scores
are retained separately. Neither experiment gains or loses a labelled exact
reading under the requested metric.

## Review artifacts and reproduction

The local comparison is `data/benchmarks/crop-comparison-20261111/index.html`.
It contains a dropdown numbered 1–18, the human reference, old/new automatic
crops, old/new reconstructed readings and the actual label. Three PNG contact
sheets cover all eighteen references. Missing or ambiguous automatic associations
stay empty; geometry correspondence is explicitly illustrative. Actual plate
strings, crop images and generated HTML remain local and are excluded from Git.
`tools/export_changed_crops.py` additionally exports all changed crop histories
from a provenance-bound replay, including the two unlabelled ones. Their presence
in the image does not establish additional correct readings.

Using the existing research environment and read-only cached model assets:

```text
# GPU OCR; permitted source directory and model cache use the usual container mounts.
python tools/run_ocr_research.py tools/replay_completed_crops.py --capture data/benchmarks/plate-validated-final-20261110 --output data/benchmarks/<fresh-replay>
python tools/run_owned_multiframe.py --phase score --output data/benchmarks/<fresh-replay> --review data/benchmarks/recorded-footage-20261104/manual-review
python tools/render_crop_comparison.py --baseline data/benchmarks/plate-validated-final-20261110 --current data/benchmarks/<fresh-replay> --review data/benchmarks/recorded-footage-20261104/manual-review --output data/benchmarks/<fresh-comparison>
```

For a new detection run, add `--complete-crops --refine-characters` to the
existing `--plate-first --plate-flow --vehicle-lifecycle --visual-identity
--validate-plates` profile. The combined character-refinement profile has not
yet been measured as a fresh detector/tracker rerun; its shared refinement
function is validated by the controlled replay and unit tests. The full trial
above measured `--complete-crops` without the later character-row refinement.

Aggregate evidence is in `docs/evidence/crop-recovery-20261111.json` and
`docs/evidence/crop-recovery-full-trial-20261111.json`. **155 focused tests pass**
across the model container and native viewer environment. Tests cover source
and label hashes, native-pixel preservation, separate-plate vetoes, CPU predictor
configuration, frame-vote independence, the found-anywhere metric and safe
comparison rendering. The viewer was built by one bounded helper agent; main
integration and model experiments were run by the primary agent.

Next work should examine recognition on the now-complete tilted/fancy-font
plate, and evaluate crop completeness on separately labelled footage. These
changes should not be presented as an OCR accuracy improvement or deployed
solely on the strength of the selected examples.
