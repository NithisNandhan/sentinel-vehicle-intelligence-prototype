# Character-guided and low-contrast plate discovery

This opt-in experiment extends the existing plate detector. It does not change
model weights, OCR settings, acceptance thresholds or the central alert path.
Generated images, plate readings and reference labels stay in ignored local
benchmark folders.

## Changes

- `LowContrastPlateSearch` wraps the current search, including optional native
  crop completion. Existing results retain priority.
- Character-guided discovery uses the cached PP-OCRv5 text detector even without
  an initial plate box. Text outlines must contain several aligned glyph-sized
  components. Nearby compatible text fragments may generate an additional
  hypothesis; surrounding body and plate colours need not differ.
- Contrast discovery applies CLAHE to the luminance channel of a working copy,
  then calls the existing plate detector with its original thresholds. It does
  not resize the region or change its coordinates.
- Both methods can retain additional proposals in the native vehicle context,
  instead of losing them because another crop ranked first. Proposal provenance
  follows retained originals into the saved results.
- Same-frame overlapping proposals are deduplicated. Character groups are
  unverified text proposals, not calibrated plate detections. Existing ownership,
  evidence triage and frame-vote protections still apply.
- Cropping, OCR and reconstruction use the original frame pixels. Enhanced
  images never supply extra temporal observations.

The extra work defaults to 24 additional model calls per sampled frame and
four retained alternatives per vehicle search. `--discovery-call-budget` sets
the extra-call limit explicitly; the CPU trial uses six. Vehicle search order
rotates across samples. These are development defaults,
not tuned optima. The character-component checks are heuristics; vehicle branding,
windows and signs can still produce false proposals.

## Running the experiment

Use the existing pinned model cache, permitted recording receipts and frozen
tracking inventory. Configure `SENTINEL_TEXT_DETECTOR_ASSETS` and
`SENTINEL_TEXT_DETECTOR_DEVICE` for the cached text model. The recorded-video
research runner must have a writable local benchmark output and model assets
mounted read-only. No Sentinel credentials or delivery are needed; the common
configuration requires nonempty local dummy EDGE_API_KEY and WATCHLIST_HMAC_KEY.

```text
python tools/run_ocr_research.py tools/run_owned_multiframe.py --phase run --output data/benchmarks/<fresh-capture> --plate-first --plate-flow --vehicle-lifecycle --visual-identity --validate-plates --complete-crops --refine-characters --character-guided --contrast-detection --device gpu
python tools/run_owned_multiframe.py --phase score --output data/benchmarks/<fresh-capture> --review data/benchmarks/recorded-footage-20261104/manual-review
python tools/render_low_contrast_comparison.py --baseline data/benchmarks/crop-extent-replay-20261111 --current data/benchmarks/<fresh-capture> --review data/benchmarks/recorded-footage-20261104/manual-review --output data/benchmarks/<fresh-comparison>
```

For an explicit CPU fallback use `--device cpu --discovery-call-budget 6`, `SENTINEL_OCR_DEVICE=cpu` and
`SENTINEL_TEXT_DETECTOR_DEVICE=cpu`. A failed GPU request is not silently retried
on CPU. The results viewer permits runtime-device differences, displays them,
and still rejects changed model hashes/checkpoints/profiles.

The experimental flags are off by default. The main dashboard's operational
profile is not automatically replaced by this research result.

## Local comparison

The viewer preserves the existing numbered reference dropdown and shows human,
previous and new crops/readings. It adds source attribution for the displayed
original, a contrast illustration, detector counters and a paginated inspection
queue of histories supported by either new method.

The contrast illustration enhances the selected crop. The detector enhances the
larger search region, so CLAHE context can differ; the illustration is explicitly
not the exact detector input. The native crop remains the OCR evidence.

The selected-reference score counts an exact active reconstructed top output
anywhere in the clip, once per reference. Extra wrong guesses are not penalised.
It is coverage, not precision, full-video accuracy or evidence of the identity
of an unlabelled candidate. Source attribution does not prove a method was
necessary: an overlapping original proposal may also have found the plate.

## Verification boundaries

Matching vehicle/plate colours are a plausible failure hypothesis, not a proven
diagnosis. This experiment remains vehicle-context based; it does not add
independent full-frame tiled search. Missing or excluded vehicle regions,
insufficient pixels, unreadable text and later retention/association losses can
still prevent a correct output.

The combined experiment is not an independent ablation of each method. Frozen
weights do not remove differences caused by CPU/GPU numerical behaviour,
wall-clock proposal budgets or altered crop selection. Separate method-only runs
on the same device and a held-out video are required for causal/generalisation
claims.

## Completed Mumbai trial, 11 November 2026

The completed capture is `data/benchmarks/low-contrast-bounded-cpu-20261111`;
the local comparison is `data/benchmarks/low-contrast-comparison-20261111`.
Aggregate evidence without plate strings/images is published in
`docs/evidence/low-contrast-20261111.json`.

- All 928 frames were decoded; 124 search timestamps were processed.
- The existing plate and OCR checkpoint hashes remained unchanged. Detection,
  OCR and the text helper ran on CPU after NVIDIA container preflight failed.
- The run took 3,276.54 seconds (54.6 minutes) after model loading. This CPU
  research profile is not real-time processing.
- Contrast detection generated 113 proposals; character-group discovery
  generated 253. After merging, routing and retention, 185 histories had support
  from at least one new method. These are not verified additional plates or
  unique vehicles. Histories can have support from both methods.
- The requested found-anywhere score stayed **12/18 (66.7%)**, with no labelled
  readings gained or lost compared with the previous crop-replay capture.
- The full capture contains 404 histories versus 304 previously. Additional
  histories must be reviewed for false proposals and fragmentation.
- Sixty focused detector/evidence/tracking tests and fifteen comparison-viewer
  tests pass. Browser checks verified all eighteen references, image loading,
  source filtering and candidate pagination.

The result does not yet demonstrate a reading improvement or establish that
matching plate/body colours caused the original misses. Keep both paths opt-in
while reviewing the candidate queue. A missing displayed correspondence is not
proof of a missed physical plate.
