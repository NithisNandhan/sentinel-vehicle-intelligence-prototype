# Experimental hybrid recognition and local viewer

## What is implemented

The unchanged default is `PLATE_READER=temporal-v2`. The opt-in `hybrid-v1`
shares cached PARSeq resources between camera workers and retains copied native
plate pixels in RAM: eight crops per track, 128 tracks and 64 MiB per camera.
ByteTrack, plate localisation, timestamps, discontinuity guards, forty-attempt
limit and privacy-protected outbox delivery remain in place.

ECC affine registration or pretrained RAFT-small aligns neighbours to a native
reference grid. Masks reject bad overlap, inconsistent flow and unsupported
pixels. Three distinct observations must survive reconstruction. Quality-weighted
fusion is reconstruction/denoising, **not proven super-resolution**. Optional
NAFNet-GoPro-width32 deblurs a working copy; inputs exceeding 256 pixels in either
dimension bypass this experimental restorer. Originals are never overwritten.

PARSeq produces character distributions per original observation. Edit alignment
combines variable-length readings, retaining unknown mass and uncertain endings.
Stable two-row hypotheses are recognised row by row and joined top then bottom;
unsupported lengths abstain. The sequence reducer is deterministic, **not a newly
trained temporal neural network**. Deep learning supplies OCR, optional flow and
deblurring. Training and calibrated correctness probabilities remain future work.

Original-frame acceptance remains mandatory. Reconstructed/deblurred OCR supplies
separate diagnostic candidates and a disagreement veto; it cannot boost acceptance
by counting reused frames twice. Candidate scores are uncalibrated. The local HTML
viewer combines retained unique original and processed candidates, capped at 200,
with evidence origin, weak characters, partial text and explicit UNCERTAIN/NO-READ
states. It cannot enumerate every possible plate or guarantee the truth is present.

## Measured pilot, 26 September 2026

Inputs: one licensed NIVU vehicle passage, three distinct video-derived crop
exports, manually verified annotations and assigned replay timestamps. These are
not verified original video PTS. Repeated degradation cases are not independent
vehicles, and this is development data, not a held-out accuracy study.

ECC plus NAFNet: 21 degradation conditions (resolution, Gaussian blur, directional
motion blur, noise and darkness), two readers and two stages = **84 evaluations**.
Results: **0 correct accepted, 0 wrong accepted, 84 abstentions**. Reconstruction
and deblurring ran in 34 evaluations. Warmup was excluded. Torch peak allocated
GPU memory was approximately 224.6 MiB; this excludes ONNX, driver and other
allocations and is not total GPU usage.

For manually annotated plate crops, the hybrid original-frame candidate beam
retained the correct identifier in 16/21 conditions versus CCT's 9/21. However,
the truth was top-ranked in only 1/21 hybrid conditions versus 4/21 for CCT.
Candidate inclusion is not recognition accuracy. Neither reader passed acceptance.
The hybrid retained the truth at median plate widths 55 and 41 pixels, where CCT
lost it; neither retained it at 27, 20 or 14 pixels in this pilot. These are local
observations, **not universal camera-resolution boundaries**.

RAFT plus NAFNet resolution pilot: 28 evaluations, all abstentions, no wrong
accepted identifiers. Reconstruction succeeded in three manually cropped
resolution conditions versus four with ECC. That initial comparison included
cold-stage timing and does not establish a latency advantage. ECC remains the
default experimental alignment. No acceptance threshold was lowered.

The no-deblurring ablation also returned 0 correct, 0 wrong and 84 abstentions,
with the same original-beam inclusion/top-ranking counts. It is saved separately.
Since original-frame distributions
determine acceptance and the original candidate beam, beam inclusion improvements
cannot be attributed to NAFNet or reconstruction. The OCR/sequence change is
confounded; a larger, separately controlled OCR-only study is still required.

## Reproduce and inspect

Cache YOLO/FastALPR as before, plus the pinned PARSeq source/checkpoint and isolated
research dependencies described in [DEGRADATION_BENCHMARK.md](DEGRADATION_BENCHMARK.md).
The hybrid loader requires these cached assets and verifies the checkpoint's
published hash prefix before loading local source. It does not fetch PARSeq during
hybrid inference. Do not replace production Torch or CUDA.

Explicitly download optional alignment/restoration assets:

```text
python -m sentinel_edge prepare-hybrid-models
```

Run against permitted, independently annotated input, with `MODEL_CACHE_DIR`
pointing to the populated cache and the ignored report directory writable:

```text
python -m sentinel_edge benchmark-hybrid --manifest data/benchmarks/nivu/manifest.json --degradation-family all --hybrid-restoration --report data/benchmarks/reports/hybrid-ecc-naf-all.json --viewer data/benchmarks/reports/hybrid-ecc-naf-all.html
```

In Compose, mount this repository at `/workspace`, use `/workspace/edge` as the
working directory, and use absolute `/workspace/data/...` arguments. Without that
working-directory override, the installed container package can shadow new source.
Use a one-off container and temporarily pause only the live AI worker, restoring
it in a `finally` block. Database and preview services can stay running.

Omit `--hybrid-restoration` for the no-deblur ablation. Add
`--hybrid-alignment raft` for learned flow. Inference initialisation uses the
experiment-only native-JIT workaround and isolated cached dependency directory.

Serve only the report directory on loopback to view the generated page. The page
contains no scripts, CDN assets, external requests or alert submission. Licensed
full plate strings and image data remain ignored local benchmark artifacts, never
central API/database/outbox/log output. Sentinel/corp8 sources, live source modes,
unapproved provenance and output outside `data/benchmarks/reports` are rejected.

## Sources and asset boundaries

- [Torchvision RAFT-small](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.optical_flow.raft_small.html): C_T_V2 checkpoint, documented transforms, published hash prefix.
- [NAFNet](https://github.com/megvii-research/NAFNet): GoPro width32 checkpoint;
  source revision `2b4af71ebe098a92a75910c233a3965a3e93ede4`; MIT notice included.
  The inference-only architecture uses equivalent LayerNorm operations instead of
  BasicSR custom autograd and global pooling for bounded small crops. Checkpoint
  keys load strictly. Full numerical parity against the author's local-pool
  implementation has not yet been established.
- [PARSeq](https://github.com/baudm/parseq): source revision
  `1902db043c029a7e03a3818c616c06600af574be`, author checkpoint `parseq-bb5792a6.pt`.

Optional assets carry source URL, revision, licence URL and local SHA-256 in the
ignored cache manifest. NAFNet's downloaded hash checks local integrity, not an
independent publisher signature. A failed model stage stays unavailable; it is
not replaced silently. All assets and licensed report inputs stay out of Git.

## Unmet gates

Correct-reading improvement has **not** been demonstrated. This experiment is not
promoted to the running live worker. A success-to-failure curve cannot be claimed
when the clear baseline itself never accepts the plate. More independent clear
passages, original timestamps, double-row labels, held-out calibration, model
parity checks, two-feed shadow performance and real accepted reads remain required.
No registry, training, face recognition or generated replacement plates were used.
