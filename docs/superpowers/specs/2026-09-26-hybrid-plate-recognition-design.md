# Hybrid multi-frame plate recognition

Status: approved for implementation by the user's "execute the plan" request;
experimental implementation and measured pilot results are in docs/HYBRID_RECOGNITION.md.
Date: 2026-09-26.

## Goal and evidence boundary

Extend the existing edge recognition flow with pixel-preserving crop collection,
geometric alignment, bounded restoration and sequence-aware OCR fusion. Keep
the deployed temporal-v2 reader as the baseline and rollback path. Do not promise
90% accuracy or a recognised plate from every passage. A highest-ranked candidate
is not automatically an accepted reading.

Inspection confirms original decoded-frame slicing already exists in
`edge/sentinel_edge/vision.py`. Missing capabilities are retained crop windows,
geometric alignment, restoration, variable-length sequence fusion and explicit
double-row processing. The existing three-frame, one-vehicle licensed pilot is
development evidence only; it cannot establish general accuracy.

## Selected architecture

Continue YOLOv8n/ByteTrack vehicle association and FastALPR plate localisation.
Add an opt-in hybrid reader behind the same PlateObservation return interface.
Do not change central APIs, the dashboard, watchlist matching or privacy rules.
Use `PLATE_READER=temporal-v2|hybrid-v1`; default remains temporal-v2 until the
hybrid evaluation is reviewed. Stamp emitted records with the actual variant
and OCR/restoration identities, using the existing model-version field.

Pipeline: native video -> tracked native plate crops -> quality selection ->
checked geometric alignment -> multi-frame reconstruction -> optional learned
deblurring -> original/restored OCR -> sequence-aware evidence fusion -> accepted,
uncertain or NO-READ -> existing privacy-protected delivery.

The reconstruction is a new processed image derived from captured observations,
not a generated replacement plate. Never render a predicted string into an image
and feed that rendering back as recognition evidence.

### 1. Native crops and bounded windows

Clamp crop bounds in decoded-frame coordinates. Detection preprocessing must
not become the source of the retained crop. Copy native pixels before the frame
buffer is reused; no JPEG encoding or resizing in this collection stage.
Retain PTS, native dimensions, vehicle/plate boxes and numeric quality features.
Keep at most eight crops per track, 128 active tracks, and 64 MiB of crop data
per camera. Reject a single crop exceeding the budget; otherwise evict oldest
observations/tracks when needed. Retain the existing 40-attempt limit, spacing,
deduplication and discontinuity resets. Clear images with the evidence window.

### 2. Alignment and multi-frame reconstruction

Use the sharpest usable native observation as reference. First establish a
bounded affine registration with OpenCV ECC; treat non-convergence, implausible
warps and poor overlap as rejection, not permission to combine unrelated pixels.
Add pretrained torchvision RAFT-small C_T_V2 as the learned alignment variant,
using forward/backward consistency masks. Use its documented preprocessing and
minimum input size, keeping model-input enlargement separate from native pixel
provenance. Compare ECC-only and learned alignment on identical passages.

Require at least three distinct source observations. Warp neighbours onto the
reference grid with validity masks; reject a neighbour with less than 70% valid
plate overlap or median forward/backward error over 1.5 pixels in reference
coordinates. Fuse only valid samples with bounded quality-weighted averaging.
These initial engineering gates are not accuracy claims. Native-grid fusion is
multi-frame denoising/reconstruction, not evidence of recovered resolution.
Do not label interpolation as super-resolution.

Each reconstructed result carries in-memory source-observation IDs, reference
PTS, valid-pixel/support masks, alignment diagnostics and processing identity.
Changing the reference starts a fresh reconstruction window. Overlapping windows
do not create new independent evidence from reused source observations. Require
at least three distinct observations surviving alignment; otherwise keep original
OCR available and report reconstruction as unavailable. Preserve originals and
never overwrite them with rectified, enlarged or restored versions.

### 3. Learned restoration as a separate experiment

Evaluate the author's NAFNet-GoPro-width32 deblurring model on the aligned
reconstruction and compare with no restoration. It is a general deblurring
model, not a plate-trained super-resolution model. Retain the original path;
restoration cannot independently authorise a reading. Do not add generative
upscaling, diffusion, adversarial sharpening or an LLM to this build.

Obtain source/weights from official releases only. Record source revision,
checkpoint SHA-256, architecture and applicable notices in an asset manifest
before loading. Check checkpoint provenance and licence before use. Keep model
assets ignored and dependencies isolated; never replace production Torch to
satisfy a research model. If an asset is inaccessible or incompatible, report
that stage as unavailable, rather than silently substituting another model.

### 4. Recognition and sequence fusion

Compare current CCT-S with the existing PARSeq adapter. PARSeq is the selected
variable-length hybrid reader with its fixed 12-character capacity plus EOS;
unsupported lengths abstain. Keep complete distributions, unknown-token mass
and uncertain EOS evidence. Original and restored outputs from the same source
window count as correlated alternatives, not additional independent frames.

Implement bounded insertion/deletion-aware sequence alignment before combining
character evidence. Missing aligned positions contribute no confirming vote.
Evaluate one-row and two-row hypotheses separately; for a two-row hypothesis,
use a stable horizontal separation visible across at least three crops, recognise
rows individually and concatenate top then bottom. If row segmentation is
unstable, hypotheses disagree or capacity is exceeded, abstain.

Use existing broad Indian-format constraints to rank at most 200 candidates;
never renormalise away invalid/unknown mass. Preserve current minimum evidence,
mean-score, weakest-character, repeated-support and margin gates initially.
Require original-frame support at each accepted character position; one restored
window cannot manufacture repeat support. No registry or confidence boost is
assumed. Scores remain explicitly uncalibrated until held-out calibration exists.

### 5. Reconstruction visualisation

Build an offline local HTML benchmark viewer, separate from the central dashboard
and live camera preview. Display original native crop, multi-frame reconstruction
and optional deblurred reconstruction side by side at matched display scale. Show
native dimensions, distinct source count, alignment rejection count and processing
time. Label outputs "Processed reconstruction - not original footage" and
"Learned deblurring - may alter character details". Keep recognition output in a
separate panel; never draw predicted characters into the plate image. Display
UNCERTAIN and NO-READ honestly rather than making an uncertain result look verified.

The recognition panel shows every retained unique candidate, up to the existing
200-candidate beam limit, sorted by descending evidence score. This is a bounded
candidate list, not every mathematically possible registration and not a guarantee
that the true plate is retained. Show rank, candidate text, uncalibrated model
score, weakest-character score and distinct source-observation support. Do not
present scores as percentage accuracy or renormalise the displayed list into an
exhaustive probability distribution. Highlight positions where candidates differ.

Show the accepted plate separately when the normal evidence gates pass. Otherwise
label the ranked list UNCERTAIN, including its rejection reason. If no candidate
survives, show NO-READ, an empty list and any supported partial text with unknown
positions marked '?'; do not manufacture complete registrations just to fill the
list. Candidate output never creates events or watchlist alerts. Ground-truth
labels may be used only by the evaluator after inference, never by candidate
generation or reconstruction.

Full benchmark candidate strings are permitted only in the local visual report
for licensed inputs, under the ignored report directory. Keep them out of numeric
aggregate JSON, ordinary logs, SQLite outbox, central APIs and the live dashboard.
Escape candidate text when rendering HTML. The viewer runs without external
requests and must not depend on available Sentinel access.

Viewer image assets and reports may be written only for provenance-validated,
permitted non-Sentinel benchmark inputs under ignored `data/benchmarks/reports`.
Include dataset attribution, licence and modification notices. No external image
hosting, upload, CDN scripts or cloud inference. Do not bundle images or labels
into Git. Existing numeric report JSON remains free of plate strings and images.
Reject live/Sentinel input for persisted visual reports. Genuine live-image
visualisation or storage requires a separate explicit privacy decision.

## Runtime safety and rollout

Run heavy GPU research variants sequentially, not alongside a live AI worker.
Restore the worker in a finally block after any benchmark interruption. Do not
stop central services or camera relays. Bound restoration attempts to one per
window update and four per track. GPU OOM or rejected alignment discards the
experimental result and reports a numeric failure; no fabricated observation.
Benchmark opt-in hybrid integration before considering a default switch.

For integrated hybrid operation, use one shared heavy-model executor per worker,
batch size one, and a bounded pending queue of eight track windows. Never create
one RAFT/NAFNet model instance per vehicle. Bypass optional restoration at six
pending windows; resume below four. Under overload or model failure preserve
original OCR and report the bypass reason, rather than waiting indefinitely.
These bounds are initial resource controls, not a guarantee of throughput/uptime.

Genuine Sentinel frames, crops, candidates and plaintext plates stay in RAM.
No cloud calls, retained live footage or image/text logging. Only existing
masked/HMAC metadata enters durable storage. No uncertain candidate automatically
becomes a watchlist alert. Diagnostic reasons remain numeric/internal.

## Checkpoints, tests and acceptance

Send a user update before and after each checkpoint, and at least every minute
during active work. Each update states changes, tests run, measured findings and
remaining limitations. Distinguish planned, implemented and validated work.

1. Crop preservation: assert pixel equality with native slices, boundary mapping,
   copy independence, memory eviction and discontinuity cleanup.
2. Alignment: synthetic known translations/affine changes, identity transforms,
   occlusion masks, unrelated plates, failed ECC, incorrect flow and no valid
   overlap. Report error in native pixels; synthetic tests are not OCR accuracy.
3. Restoration: verify asset metadata, load/warm-up, output dimensions/finite
   values, GPU memory and failure isolation. Measure exact recognition, not just
   image sharpness. Test that a restored copy adds no extra temporal support.
4. Fusion: dropped/inserted characters, ambiguous endings, unknown mass, stable
   and unstable row splits, correlated copies and wrong-but-plausible strings.
5. Benchmark: compare temporal-v2, crop/sequence-only, alignment/fusion and
   alignment/restoration/fusion on the same permitted passages and degradation
   controls. Keep labels outside inference. Report correct, wrong, abstained,
   coverage, candidate inclusion, stage failures, latency and peak GPU memory.
   Verify viewer source/output association, truthful labels, attribution, absence
   of externally loaded content, and refusal to persist live/Sentinel images.
   Test uncertain candidate display, stable ranking/deduplication, the 200-entry
   cap, empty/partial NO-READ output, HTML escaping and unchanged alert emission.
6. Integration: run edge/backend regression tests, privacy checks, offline smoke
   workflow and bounded live shadow observation without saving media. Promote
   only after correct-reading improvement is demonstrated with no additional
   wrong accepted readings on the evaluated passages. Small pilots remain pilots;
   broader claims require independent held-out vehicles and cameras.

No improvement is a valid benchmark outcome, not grounds for lowering acceptance
thresholds. If the baseline has no successful recognition, state that a meaningful
success-to-failure curve has not yet been established. Do not present the current
single-vehicle dataset as the finished-product accuracy study.

## Sources and alternatives

- Torchvision RAFT-small, weights and input requirements:
  https://docs.pytorch.org/vision/main/models/generated/torchvision.models.optical_flow.raft_small.html
- NAFNet author implementation and model list:
  https://github.com/megvii-research/NAFNet
  https://github.com/megvii-research/NAFNet/blob/main/LICENSE
- Evidence-preserving multi-frame restoration research (not our implementation):
  https://arxiv.org/abs/2508.14797

The selected hybrid reuses existing detection and OCR and makes failures
inspectable. OCR-only fusion remains the baseline. A jointly trained temporal
recognition network remains future research, requiring suitable labelled
sequences and a separate training/evaluation proposal.
