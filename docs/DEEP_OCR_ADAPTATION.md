# Bounded deeper Indian-plate OCR adaptation

This is an experimental checkpoint, not a production deployment or a universal
plate-reading claim. Dataset acquisition and the original live worker are unchanged.

## Environment and safeguards

GPU PaddlePaddle 3.2.2 was installed from the publisher's CUDA 12.6 index,
using the [official installation route](https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/linux-pip_en.html).
The GPU package lives in `/models/experiments/awiros/gpu-deps`; its declared
CUDA libraries live separately in `gpu-runtime`. Production Torch/CUDA were not
replaced. Pinned versions are in `edge/requirements-awiros-gpu-research.txt`.
`tools/run_ocr_research.py` selects these libraries only for research commands.

The real GPU operation and Paddle's installation check passed on the RTX 3050
6 GB. Full OCR CPU/GPU parity passed on three licensed crops: identical greedy
readings, maximum probability difference 0.000064552 versus a 0.0001 bound.
Initial default-precision parity failed; disabling TF32 fixed it without relaxing
the bound. These three checks establish compatibility, not recognition accuracy.

## Label review and preparation

The 89 photographic holdout crops were displayed on eight label-hidden sheets.
Codex independently proposed 68 visual transcriptions before consulting publisher
text; 21 were uncertain, cropped or unreadable. Of the proposed transcriptions,
48 agreed with publisher text and 20 disagreed. Disagreements were not silently
corrected, promoted to ground truth or returned to training.

This is **AI visual review, not independent human verification**. Agreement can
still contain shared errors. The 48 retained crops represent 44 registration
strings. Originals and the merged corpus manifest were preserved; the new review
and adaptation selection are separate ignored local files.

Of 951 photographic training candidates, 149 failed conservative native size or
aspect-ratio gates. This is a crop-quality filter, not proof that all remaining
labels are correct. Training uses 802 photographic crops (many publisher-silver
labels), plus a pool of 300 generated plates. Each full four-sample batch contains
three photographs and one generated image, so synthetic share is at most 25%.
The origin-unverified pool was not used.

Splits are registration-disjoint before augmentation:

| Split | Native crops | Distinct registration strings |
|---|---:|---:|
| Training | 1,102 | 1,025 |
| Validation | 37 | 33 |
| Test | 31 | 31 |

Validation and test each include 10 preserved legacy visual-review samples.
Fresh diagnostic samples use explicit `AI_blind_visual_publisher_agreement_not_human_truth`
labels. Test images were not consulted during checkpoint selection. Pretraining
overlap remains unknown. Duplicate views of a registration are not independent
vehicles. These still images were never presented as multiple real video frames.

`tools/stage_ocr_corpus.py` exported only the selected 1,170 native crops and
manifest, a 108,595,200-byte hash-checked archive, to the Linux research volume.
This avoids repeated small-file access over the Windows/WSL boundary.

## Actual training

The original Awiros weights were preserved. The experiment trained 133,864
parameters: the CTC output head, the final SVTR recognition block, and one
late depthwise image-encoder convolution. All other parameters and batch-normalisation
buffers stayed byte-identical. The publisher's training-only feature detach was
disabled so the selected image layer actually received gradients; inference
forward values and architecture are unchanged.

Adam learning rate 0.00002, batch size at most 4, global gradient clipping 1.0,
seed 20260928, 12 epochs maximum, mild training-only image augmentation. Validation
exact matches and then character-error rate chose epoch 8. Four unimproved epochs
ended the run. Full checkpoint reload/inference parity passed.

Checkpoint: `/models/experiments/awiros/adaptations/indian-corpus-encoder.safetensors`.
A checksum-matched Windows backup is preserved under ignored
`data/benchmarks/indian-corpus/checkpoints/indian-corpus-encoder.safetensors`.
SHA256: `c2ba896d994b174071c95ace0f5ffc8a61cec3fc6eecce8f0bc5feb7dec4456d`.
Parent SHA256: `f9f1264e0c115a239ca6d6700a74ceffab17168b75dafc6c169c232012c68e47`.
Manifest SHA256: `89fe372395b5ceb6c8aabd1ed24bca9274137485a6ecce1c190cc0a0c8ee37d0`.

## Measured still diagnostics

| Metric | Untouched parent | Selected adapted model |
|---|---:|---:|
| Validation exact plates | 24/37 | 29/37 |
| Validation character errors | 60/360 | 42/360 |
| Test exact plates | 22/31 | 28/31 |
| Test character errors | 23/295 | 13/295 |
| New AI-reviewed test subset | 13/21 | 19/21 |
| Legacy reviewed test subset | 9/10 | 9/10 |

28/31 is 90.3% **on this particular still diagnostic**, not 90% reliability on
blurry cameras, calibrated confidence, a large human-verified test, or live accuracy.
The legacy subset did not improve. Training labels remain partly noisy, and
image-selection/review biases limit generalisation claims.

The local before/after viewer is
http://127.0.0.1:7777/indian-corpus-adaptation.html (62 cards). Each shows the native
licensed crop, raw reading and ranked candidates; a single still cannot be accepted
as a multi-frame sighting. Scores remain uncalibrated.

## Multi-frame reconstruction and failure boundaries

The same existing NIVU passage, three distinct video-derived stills, was run through
21 degradation conditions using quality-weighted native-frame fusion, optional
deskew and ECC reconstruction diagnostics. This is one development-visible vehicle,
with assigned replay spacing rather than verified real video PTS. It is a regression
test, not a fresh multi-vehicle accuracy test.

| Outcome | Untouched parent | Selected adapted model |
|---|---:|---:|
| Correct accepted identifiers | 12 | 13 |
| Wrong accepted identifiers | 0 | 0 |
| Uncertain / abstained | 9 | 8 |

The changed case was darkness factor 0.25. The resolution boundary did not improve:
native median plate width 54 px succeeded, 41 px retained the true candidate but
failed acceptance, and 28 px or less lost the true candidate from the beam.
Motion kernels 5, 9 and 15 still failed acceptance; kernel 3 succeeded. Gaussian
blur kernels 3 through 9 succeeded on this one passage. These are tested settings,
not universal image-quality thresholds. Reconstruction did not create missing
characters or supply extra independent observations.

Viewers: `/indian-corpus-multiframe-parent.html` and
`/indian-corpus-multiframe-adapted.html` on the same local viewer server. Uncertain
cases retain ranked possibilities when supported by OCR evidence; candidates are
not guaranteed to include the truth.

## Live shadow validation

The read-only probe in `sentinel_edge.awiros_live_shadow` compares parent and
adapted OCR on the same native plate crops from CAM01/CAM02. It uses the existing
local relays, continuous decoding, timestamp-based sampling and separate per-camera
state. Its conservative plate-box IoU association is diagnostic: it does not
replace or validate the production YOLO/ByteTrack vehicle pipeline.

Only aggregate counts and timings leave memory. Genuine frames, full plate text,
probability matrices and ephemeral keyed identifiers are not serialized. The
probe has no central writes and never promotes a checkpoint. Genuine live accuracy
cannot be calculated without independently labelled live passages.

The first full-frame localisation run decoded both feeds for 90 seconds each:

| Camera | Decoded frames | Timestamp-sampled frames | Detected plate crops | Accepted identifiers |
|---|---:|---:|---:|---:|
| CAM01 | 1,377 | 173 | 0 | 0 |
| CAM02 | 2,049 | 137 | 0 | 0 |

There were no missing PTS or OCR processing errors. Crucially, **no plate crops
reached either OCR model**, so this is not a successful live OCR test and does
not measure whether the adapted model can read these scenes. Full-frame plate
localisation is the observed bottleneck in that run.

A follow-up added native YOLO vehicle crops before plate localisation. During
that run, neither relay delivered frames within its bounded startup window.
The aggregate failure report was preserved; this is a connectivity failure,
not evidence of worse OCR. A subsequent direct RTSP/HLS test also decoded zero
frames on both cameras within its bounded startup windows. Both failure reports
are retained separately. An additional catalogue request returned the existing
adapter's explicit Sentinel sign-in failure. This does not establish that the
password is wrong; account access, session policy and connectivity need checking.
Vehicle-crop localisation and adapted OCR on live plates remain unvalidated;
the original production reader was not replaced.

Final edge regression suite: 179 tests passed, including registration leakage,
modified crops, explicit label tiers, allowed trainable scope, native vehicle-crop
pixel preservation and aggregate-only shadow output. These tests do not certify
live accuracy. The consolidated local page is
http://127.0.0.1:7777/indian-corpus-summary.html.

## Remaining acceptance limits

The experiment improves a bounded still diagnostic and one dark regression case.
It does not establish correctness from unreadable live plates. Independent human
label verification, larger real temporal test sets, calibrated confidence, difficult
two-row/angled-plate coverage and repeatable live read success remain needed before
production promotion. No reconstruction or database lookup can guarantee recovery
of characters absent from the recorded pixels.
