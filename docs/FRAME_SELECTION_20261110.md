# Safer evidence continuity and frame-selection experiments

## Result and deployment decision

The known mixed-vehicle failure is fixed in the recorded worker. OCR histories
now restart before admitting a plate after an implausible gap, scale change or
global position jump. Old confirmation, reconstruction, native crop bank and
diagnostic evidence are cleared together; finalization retains the earlier
passage's own sample/search/detection counts. Screened duplicate frames can
maintain geometric liveness but never become extra independent OCR votes.

The tested recorded worker was restarted, reached its normal paused state and
resolved the exact source hash used in validation. OCR weights, detector models,
sampling cadence and acceptance thresholds are unchanged.

The stricter image-only selector is **experimental**, not deployed into the
alert/confirmation path. It did not improve exact readings over the existing
best-quality-frame rule on the available labels.

## Frozen-input comparison

All six policies reuse the same native images and original OCR probabilities
from the earlier complete run. Features are extracted before scoring and use no
plate transcriptions or OCR confidence. Each reference follows its geometry-
matched anchor into its own continuity group; the answer never chooses a group.

| Rule | Exact on the six matched references | Exact on the two continuity-clean references with at least three observations |
|---|---:|---:|
| Earlier reconstruction pipeline | 2/6 | 0/2 |
| Earlier best-quality single frame | 3/6 | 0/2 |
| Continuity-checked original-frame fusion | 3/6 | 0/2 |
| New image-quality-selected single frame | 3/6 | 0/2 |
| New best-three-frame fusion | 3/6 | 0/2 |
| Best frame with gated multi-frame assistance | 3/6 | 0/2 |

The recovered reference is 19, whose history previously contained two different
vehicles. None of the earlier exact readings on the matched subset were lost.
There is no measured selection-policy improvement beyond continuity repair.
No policy generated a model-confirmed reading on these matched references.

Eighteen previously reviewed targets were considered, excluding unreadable IDs
08 and 14 and disputed ID 21. Twelve lack a safe association at their exact
reference timestamp; they are not proven full-passage detector misses. Conservative
all-target exact counts are 2/18 for the earlier pipeline and 3/18 for the other
rules. These are selected development diagnostics, not whole-video accuracy.

## Image selection implemented for evaluation

`frame_selection.py` computes bounded scores from native focus, contrast,
exposure and pixel dimensions. Native images are never resized or enhanced by
this selector. Fixed screening limits reject widths below 48 pixels, heights
below 12 pixels, Laplacian sharpness below 40, contrast below 12, or an extreme
bright/dark fraction above 0.90. They were not searched or tuned against labels.

Tilt and possible edge contact receive soft penalties. A plate border can also
touch an edge, and a white plate can be bright without being overexposed. These
heuristics cannot certify that every character is present or readable. Every
assessment explicitly states that completeness is unverified.

The selector retains the highest-ranked eligible distinct crops, with at least
150 ms separation, and restores chronological order for fusion. An exact duplicate
does not add a vote. The tested three-frame assistance policy uses a multi-frame
result only if it passes the existing acceptance gate; otherwise the best-frame
reading remains an unverified hypothesis. It never confirms a plate from one image.

Of 171 distinct native image hashes in 172 admitted observations, 158 passed the
fixed image screen and 13 were rejected for insufficient native dimensions.
This does not establish that the remaining images contain readable, complete plates.

## Complete GPU validation of the deployed guard

The new run decoded all **928** source frames, analysed **62** sampled frames and
recorded **174** admitted OCR observations in **126** reader segments. It reached
EOF successfully in 147.8 seconds including diagnostic evidence collection.
The original and validation runs have small crop-count differences because the
existing proposal search uses timing budgets; the frozen-input comparison above
isolates selection using identical inputs. Their wall times are not a controlled
throughput comparison.

The two original native pixel hashes from 28.0 and 30.5 seconds now occur in
separate reader segments (`ocr-0115` and `ocr-0126`) even though the tracker ID
remains 117. The former correct reading is preserved. Repeating the same reference
association/scoring protocol on the new capture gives **3/6 exact** for single-frame,
original multi-frame and reconstruction approaches.

The guard created 103 continuity resets: 69 gaps, 27 relative jumps and seven
scale changes. Eligible fusion windows fell from 42 in the earlier run to 29;
this conservatism can discard useful evidence too. Three unique readings were
accepted by the model elsewhere in the clip, but these labels do not verify them.
There is no claim that geometric continuity establishes vehicle identity or
eliminates every possible mix.

Default continuity limits are a maximum gap of 1,000 ms, width/height scale ratios
within 2.5 in either direction, and a centre displacement within 1.5 times the
larger plate-box diagonal. One missed 500 ms sample is allowed; larger gaps split.
Only global plate geometry and video timestamps enter this check.

## Verification and artifacts

Fifty continuity/tracked-OCR/passage/reconstruction tests and 29 selector/CTC/
image-quality/native-reconstruction tests passed in the existing cached image.
The production source path and hash were verified after the worker restarted.
A separate QA agent checked leakage, grouping and lifecycle behaviour. The saved
Graphify graph was reused; its installed launcher still prevents incremental refresh.

Local artifacts:

- `data/benchmarks/frame-selection-20261110/`: frozen features, detailed policy
  results and `index.html` with native views, scores, reasons and chosen timestamps.
- `data/benchmarks/frame-guard-validation-20261110/`: full new trace, scored
  results, `guard-validation.json` and preserved runtime-source hashes.
- Existing human labels and earlier benchmark reports remain unchanged.

Example reproduction of the feature phase, inside the cached OpenCV image with
`edge`, `tools` and `data` mounted under `/workspace`:

```text
python /workspace/tools/compare_frame_selection.py --phase features
  --trace /workspace/data/benchmarks/temporal-approaches-20261110
  --output /workspace/data/benchmarks/<fresh-selection-folder>
```

Then score locally with the same paths under `data/benchmarks`:

```powershell
python tools/compare_frame_selection.py --phase score `
  --trace data/benchmarks/temporal-approaches-20261110 `
  --output data/benchmarks/<fresh-selection-folder>
```

The exact capture/scoring inputs are ignored local evidence. GitHub receives
source, tests, this protocol and aggregates, not raw footage, native crops,
plate transcriptions or credentials. No downloads or new model training were needed.

## Remaining work

Collect more human-labelled continuous vehicle passages and measure all earlier
wrong acceptances, not just final hypotheses. Improve plate/text completeness checks
and retest frame selection on those fixed labels. Keep richer filtering experimental
until it improves exact readings without losing correct outputs on a larger set.

The earlier full-video labelling export groups cards by tracker ID. Since this
investigation established ID reuse, regenerate its inventory around passage
boundaries before treating it as a final full-video accuracy report; its pending
labels have not been changed or finalized by this work.
