# Native plate context and independent plate histories

## Measured result

| Path | Single native frame | Multi-frame | Reconstruction + multi-frame |
| --- | --- | --- | --- |
| Preserved vehicle-owned baseline | 5/18 | 7/18 | 7/18 |
| Padded search, appearance/OCR association | 7/18 | 7/18 | 7/18 |
| Padded search, checked motion association | 6/18 | **8/18** | **8/18** |

These are the same eighteen selected development targets, not full-video
accuracy. The final multi-frame path gained references 06 and 10 and lost 03
relative to the original multi-frame baseline. Ten references had a unique
matched crop; eight were exact. Five matched references passed the unchanged
text acceptance gate, all five agreeing with their labels. Four additional
accepted histories have no matched human reference and remain unverified.
No general false-positive rate can be inferred from this small selected set.

The final run decoded all 928 frames, sampled 124 timestamps, performed 481
original-input OCR calls and produced 245 evidence histories, not 245 unique
vehicles. It took 317.17 seconds after model loading, compared with the
baseline's 280.66 seconds. This offline diagnostic is not real-time processing.
The proposal pool includes a wall-time budget for optional searches, so optional
calls can vary between runs; the counts describe these recorded executions.

The user-reported plate previously distributed across V011/V006 now stays in
one PT history over eight timestamps from 1.000 through 2.767 seconds. Its early
0.767-second observation remains separate; uncertain vehicle ownership stays
unassigned. Its fused OCR hypothesis is still wrong and unaccepted. This is an
improvement in continuity, not proof of correct recognition or physical owner.

At 3.267 seconds the padded automatic search captures the full visible plate
that previously appeared as V031-P001 and V011-P003 fragments. OCR still reads
it incorrectly. The separate human-directed probe also failed to read the native
union correctly; its detector redetection selected a different region. Neither
that manual union nor its OCR was fed into the automatic benchmark. No nearby
fragment pair triggered the optional recovery search in the complete automatic
run; that branch is covered by synthetic tests, not successful field evidence.

The implementation passes **59 focused tests**, including native crop bounds,
fragment replacement, conflicting text, competing associations, partial rival
coverage, vehicle-ID changes, forward/backward motion, failed reseeding,
timestamps, capacity bounds and temporal/reconstruction safeguards. The normal
recorded worker was restored after GPU benchmarking. The new path remains an
opt-in engineering experiment; it has not replaced operational alert delivery.
The remaining work is crop/reading quality selection, the missed reference 03,
and a larger independent identity/recognition review before promotion.

Aggregate measurements and source fingerprints are in
`docs/evidence/plate-context-20261110.json`. The local comparison viewer is
`http://127.0.0.1:7800/`; the original baseline remains on port 7799.

This experiment addresses the reported cross-vehicle plate leaks and the
same-frame left/right fragments. It leaves the deployed OCR checkpoint,
recognition thresholds, frozen dense vehicle detections and original benchmark
unchanged. It does not replace the operator alert path.

## Implementation

`plate_context.py` expands each eligible vehicle search by 20 percent, with
48–256 native-pixel padding per axis. All boxes are clipped to the source image;
the search receives an original-resolution slice. The detector and text-region
pool search this context, rather than using a vehicle box as a hard plate border.

Nearby horizontal fragments can trigger at most two extra searches per sampled
frame. A new detection must span the centres of both fragments and at least
85 percent of their union's width to replace them. Proximity alone never joins
OCR strings. The recovered crop is taken from the original frame. Two pieces
from one timestamp still count as one observation.

`PlateIdentityTracker` maintains PT identifiers independently of vehicle IDs.
Association requires a bounded video-time gap, predicted position, compatible
scale, and either normalized crop appearance similarity or complete strong OCR
agreement. Three conflicting high-scoring characters veto a continuation.
Close competing assignments create separate histories instead of forcing a
match. The tracker stores two observations and a strongest-reading anchor per
active identity; it expires histories after 1,000 ms and caps active identities
at 512. This is a conservative heuristic, not physical identity ground truth.

The optional `--plate-flow` path follows plate-local image features on every
decoded frame, using a grayscale tracking copy up to 1,280 pixels wide. It never
resizes saved native crops. Features need forward/backward agreement within one
tracking pixel, at least eight surviving points, 65 percent survival and
75 percent affine-fit support. A prediction needs two decoded-frame updates and
IoU >=0.30 with the next detection to support association. Strong OCR conflicts,
geometry limits and competing-assignment abstention remain in force. Motion
state expires after one second without a detection and is capped at 64 groups.
Flow provides no independent text evidence; texture loss deliberately splits a
history rather than inventing a continuation.

Vehicle ownership is a separate decision. One eligible vehicle must cover at
least 90 percent of the plate, with no competing vehicle covering 20 percent or
more. Tentative and excluded neighbours can veto ownership. An owner requires
support at two independent timestamps; ambiguous frames expose no stale owner
lock. Unassigned plate hypotheses remain visible for engineering review.

The benchmark runs identity OCR before association, caches its original
posteriors, and retains at most six native observations per group in a 128 MiB
crop bank. Single-view, CTC multi-frame and checked reconstruction comparisons
reuse those original posteriors. Reconstruction suggestions do not add votes.
Text model acceptance is separate from confirmed vehicle ownership. No central
event is sent by this experiment.

## Evaluation boundaries

Inference uses the full permitted 928-frame source with 250 ms plate sampling.
Human transcriptions are loaded only by the separate scoring phase. The eighteen
previously reviewed development targets use the existing timestamp and IoU
matching rule; unmatched and conflicting histories count as misses. They are not
an exhaustive plate inventory or a held-out generalization test. Comparison
must use the common all-eighteen denominator, because matched subsets change.

`probe_plate_context.py` separately tests human-specified crop boxes from a local
JSON spec. Its manual union and redetection are diagnostic only and cannot enter
the automatic score. The spec binds recording identity, SHA256 and exact PTS.

## Reproduction and local artifacts

Use the installed recorded GPU Compose profile with the isolated
`run_ocr_research.py` launcher and mount `data` at `/workspace/data`. Run
`tools/run_owned_multiframe.py --phase run --plate-first --output <fresh-folder>`
from `/workspace`, then score that folder with `--phase score` and the existing
local review directory. Add `--plate-flow` for the motion-supported trial. Omit
`--plate-first` to retain the previous routing path.
Do not overwrite a prior run.

`data/benchmarks/plate-first-20261110/` holds the local predictions, native PNGs,
posteriors, frozen input review, accuracy report and static viewer. The paired
motion run is `plate-flow-checked-20261110/`; the baseline remains in
`owned-multiframe-20261110/`. Footage, crops, transcriptions
and human labels remain local; published evidence contains aggregates and hashes.

`plate-flow-20261110/` is superseded and must not be used for acceptance evidence:
code review found that failed feature reseeding retained an earlier prediction.
The corrected implementation clears that state even when no new points can be
selected; a displaced, textureless reseed is covered by a regression test.
