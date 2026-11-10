# Multi-frame OCR with explicit vehicle ownership

## Measured result

One bounded GPU run completed all 928 decoded frames, 124 plate-search samples
and 315 original OCR calls in 280.66 seconds. Vehicle detections/tracks were
reused from the frozen full-frame tracking run. The router retained 399 owned
observations in 179 plate groups, rejected 88 ambiguous ownership proposals and
deduplicated 16 proposals. No group was lost to crop-bank eviction. Reviewed
unsafe history V027 was excluded. The input review snapshot hash was verified.

| Method | Exact on 12 matched references | Exact on seven matched references with at least three originals | All eighteen reference targets |
|---|---:|---:|---:|
| Highest-image-quality single frame | 5/12 | 4/7 | 5/18 |
| Owned multi-frame OCR | **7/12** | **6/7** | **7/18** |
| Owned multi-frame plus reconstruction | **7/12** | **6/7** | **7/18** |

Multi-frame recognition recovered reference targets 03 and 15 and lost none
that the single-frame baseline got exactly right. Reconstruction added no exact
reading beyond ordinary multi-frame OCR in this comparison. Six references
lacked a safe association at their labelled timestamp; this is not proof their
vehicles were never detected during the whole passage.

Three model-accepted readings matched the available human labels; no accepted
reading disagreed within that matched labelled subset. Seven groups passed the
model gate overall, leaving four acceptances unverified by these labels. These
groups are not assumed to be distinct physical vehicles. No alerts were sent.

Native-image inspection confirms repeated views of the same scooter for recovery
03. Recovery 15 contains consistent plate views but also a retained headlight
false crop; the ownership gate prevents another vehicle's contribution but cannot
prove every crop is a plate. This remaining detector/completeness defect is
visible in the viewer and requires follow-up. The comparison is against the
implemented image-quality selector, not an oracle that picks a correct frame.

There are 147 single-frame, 155 multi-frame and 156 reconstruction groups with
nonempty hypotheses across the run. These are unverified group outputs, not
147/155/156 correct or unique plates. Many groups have fewer than three views.
The text proposal search also recorded 1,162 skipped extra searches under its
existing budget, so candidate-search coverage is bounded.

Twenty-five ownership, CTC fusion and reconstruction tests pass. The GPU worker
was restored after the run. This change is connected to the engineering benchmark
pipeline; the central operational alert path is not automatically switched.

This experiment connects the dense 960-input vehicle tracking replay to the
existing GPU plate detector and adapted Awiros OCR. It does not train, download
models, lower confirmation gates or send watchlist events. Recognition is a
bounded recorded benchmark, not live inference inside the operator dashboard.

## Evidence rules

Every source frame is decoded and matched to its exact frozen vehicle timestamp.
Plate search uses a 250 ms source-time bucket; tracking remains at every frame.
Only qualified vehicles are searched. All visible vehicle boxes, including
tentative and reviewed unsafe boxes, participate in ownership checks.

A plate requires at least 90% area coverage by exactly one vehicle. Overlapping
ownership is rejected rather than choosing the smallest or highest-confidence
vehicle. Reviewed non-vehicles, mixed histories and uncertain histories are
excluded from fusion. Duplicate plate proposals are merged at IoU >=0.50 before
assignment; separate simultaneous plate proposals for one owner are treated as
ambiguous. These are conservative geometric heuristics, not physical-identity proof.

Evidence keys include the owner passage ID and a plate-continuity generation.
Existing PTS, position and scale guards restart plate evidence after suspicious
changes. Different vehicle IDs cannot share an OCR history. Native images are
not resized for storage; any geometric text correction is a separate working view.

At most six distinct, time-separated native views are retained per plate group
inside a 128 MiB bounded crop bank. Selection uses the existing image-quality
heuristic, never truth or OCR confidence. Reconstructed images do not count as
independent observations. Missing or evicted evidence is recorded explicitly.

## Paired comparison

The same original OCR probability arrays support three methods:

1. Existing candidate decoding on the highest-image-quality native observation.
2. Existing CTC sequence fusion on the retained independent observations.
3. The same fusion with checked ECC reconstruction suggestions, rescored on
   original probabilities; disagreement can veto model acceptance.

Unverified top hypotheses and up to twenty candidates remain separate from
model-gate acceptance. Acceptance still needs three originals, mean character
score >=0.85, minimum character score >=0.75, candidate margin and repeated
character support. These scores are not calibrated correctness probabilities.

Human truth is loaded only after inference. The previous eighteen usable Mumbai
references are matched by time (20 ms tolerance) and geometry (IoU >=0.30 using
existing manually outlined text/plate bounds). Unreadable IDs 08 and 14 and
disputed ID 21 remain excluded. Ambiguous reference associations and histories
linked to conflicting truths are not assigned a convenient answer. Unmatched
references stay in the all-eighteen denominator; matched and >=3-original subsets
are reported separately. These are development references, not an exhaustive
vehicle inventory or unseen-camera accuracy.

## Artifacts and reproduction

The new `OwnedPlateRouter` has tests for ownership, simultaneous vehicles,
tentative/excluded neighbours, duplicate proposals, plate ambiguity, timestamps
and generation changes. Fusion and reconstruction dependency tests are also run.

Use the existing cached GPU Compose service and isolated `run_ocr_research.py`
launcher to run `tools/run_owned_multiframe.py --phase run --output <fresh-folder>`,
with the data volume mounted under `/workspace/data`. The source tracking run
defaults to `vehicle-tracking-960-20261110`. Complete the benchmark before scoring:

```powershell
python tools/run_owned_multiframe.py --phase score `
  --output data/benchmarks/owned-multiframe-20261110
```

The ignored local folder contains native PNGs, frozen original posteriors,
predictions, the input vehicle-review snapshot, an accuracy diagnostic and a
side-by-side `index.html`. Source, tests and aggregate measurements are published;
raw footage, crops, plaintext plate readings and user review labels remain local.
No model/threshold search or repeated tuning loop is part of this run.
