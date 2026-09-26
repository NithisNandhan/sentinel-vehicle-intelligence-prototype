# Indian OCR model selection: measured development pilot

## Decision and scope

Awiros ANPR OCR is the preferred **next experimental recognition model** after
testing its publicly released Indian-fine-tuned checkpoint locally. It is not
yet the live-worker default or a model further trained by this project. There
is no claim that all models were benchmarked, that this is universally best,
or that a 90% confidence score represents 90% correctness.

The live default remains `temporal-v2`. No Sentinel frames, crops or plaintext
plates were retained for this experiment. No registry, restoration, generated
replacement image or ground-truth label was used during inference.

## Pinned assets and isolated runtime

Model: https://huggingface.co/Awiros/anpr-ocr . Model revision:
`eadbc5ae4faa8284c8a547a60142fd555c3059ca`.
Published weight SHA-256:
`f9f1264e0c115a239ca6d6700a74ceffab17168b75dafc6c169c232012c68e47`.
Model weights are 149,448,448 bytes; the release supplies an Apache-2.0 notice.

Framework: https://github.com/PaddlePaddle/PaddleOCR . Pinned v3.4.0 source
revision: `fc82f229112a03aff46cc53db6166a98ea65a8fa`.
Only the `ppocr` source tree and licence are extracted. The author's inference
script is not executed: its architecture configuration is parsed as literal
data. Automatic cloning of current framework source is disabled. All checkpoint
keys and tensor shapes must match; missing or extra parameters are fatal.

Preparation is an explicit download operation. Inference is cached-only and
verifies model/source integrity against the preparation manifest. Locally
computed source hashes are integrity checks, not independent signatures.
The author-published full weight checksum is verified separately.

CPU dependencies are isolated under `/models/experiments/awiros/deps` using
`edge/requirements-awiros-research.txt`. Production Torch/CUDA and site-packages
are not replaced. A separate container has no Sentinel credentials and mounts
benchmark reports writable, but project inputs read-only. CPU inference is
requested explicitly; GPU requests fail if Paddle lacks CUDA support.

## Recognition and multi-frame evidence

Native plate crops are copied using independently annotated coordinates; they
are not resized or reconstructed before being retained as the original view.
The model's working input follows the author's BGR/aspect-preserving height-48,
width-at-most-320 resize, right padding and [-1, 1] normalization.

The complete CTC posterior is preserved. CTC time positions are not character
positions and cannot be averaged like fixed-slot CCT outputs. A raw-token
prefix beam decodes each original frame. Case folding and space removal happen
after collapse. Repeated symbols, including distinct uppercase/lowercase tokens,
retain proper CTC semantics.

Up to 200 candidate identifiers from frame beams are rescored against **every**
frame using exact CTC forward likelihood. The temporal rank is the mean of these
per-frame sequence log-likelihoods. It is not a calibrated probability or an
independence-based posterior. Repeated source IDs, timestamps closer than 150 ms
and more than eight retained observations are rejected or bounded. Processed
copies cannot count as extra original observations.

Acceptance requires three original observations, mean character evidence >= .85,
weakest-character evidence >= .75, support >= .60 in at least two observations
per character, and a >= .5 log-score margin over the next retained candidate.
For CTC, character evidence is the peak on the best alignment path, not a
marginal character probability. It is not interchangeable with CCT's fixed-slot
scores. Identical numerical thresholds do **not** establish identical calibration
or accepted precision between model families. No calibrated live acceptance
claim is made. Grammar remains a broad format hint, not a vehicle registry.

Prefix-beam numerical rescaling keeps the accumulated log scale. It changes
neither model probabilities nor evidence rules. Tests compare both prefix mass
and full-sequence scoring with exhaustive raw CTC paths, including repeat/case/
space cases. The final full replay checks that numerical optimisation preserves
the observed pilot outcomes.

## Measured results, 26 September 2026

Source: NIVU Indian License Plate Dataset, Roboflow Universe, April 2022,
https://universe.roboflow.com/nivu/indian-license-plate-knte7 . CC BY 4.0:
https://creativecommons.org/licenses/by/4.0/ . Cropping and deterministic
degradation are modifications. Images, labels and reports remain ignored local
files, not redistributed in Git.

One independently read development vehicle, three distinct video-derived crops,
assigned replay timestamps 0/400/800 ms. Original video PTS are unknown. All
21 degradation conditions use that same vehicle. These are not independent test
vehicles, not held-out accuracy, and not automatic plate localisation results.

| Reader | Correct accepted conditions | Incorrect accepted | Abstentions |
| --- | ---: | ---: | ---: |
| Existing CCT-S v2 + temporal reader | 0 | 0 | 21 |
| Existing pretrained PARSeq + sequence reader | 0 | 0 | 21 |
| Indian-fine-tuned Awiros + CTC temporal reader | 11 | 0 | 10 |

Every raw native-frame reading in Awiros's clear baseline matched the verified
label, and its multi-frame top candidate was accepted. This demonstrates recovery
of **one distinct plate**, not eleven plates and not a production accuracy rate.

Observed Awiros boundaries on this passage:

- Resolution: accepted at median plate widths 109, 81 and 54 pixels. At 41
  pixels the truth remained top-ranked but character evidence was insufficient.
  At 28, 20 and 13 pixels the true identifier left the retained beam.
- Gaussian blur: accepted at all tested kernels 3, 5, 7 and 9 pixels.
- Horizontal motion blur: accepted at kernel 3, uncertain at 5, truth absent
  from the beam at 9 and 15.
- Noise: accepted at sigma 5 and 15; uncertain at 30.
- Brightness: accepted at multiplier .75; uncertain at .5 and .25.

These are passage-specific engineering observations, not universal quality
cutoffs. Relative blur and brightness transforms do not reproduce every physical
camera condition. Compression and night/headlight cases remain untested.

First CPU run: median network inference approximately 2,116 ms per crop; full
three-frame conditions took approximately 6.5--9.4 seconds, including decoding
and rescoring. Docker was capped at two CPUs and 4 GiB RAM. The CPU limits and
background services affect timing. The model is **not live-speed validated**.
Author RTX 3090 latency/accuracy claims are not substituted for these measurements.

Final replay after decoder optimisation: the same 11 correct / 0 wrong / 10
abstention outcomes and the same true-candidate ranks in all 21 conditions.
Median network inference was approximately 2,090 ms per crop; this remains a
CPU-only measurement. The optimisation does not resolve the network runtime
bottleneck or establish real-time GPU performance.

Reports: `data/benchmarks/reports/ocr-selection-{cct,parseq,awiros}.json`.
The final numerical-parity replay uses `ocr-selection-awiros-validated.json`.
Matching HTML filenames show original crops, frame-by-frame raw OCR, evaluation
label (after inference only), retained candidates and explicit uncertainty.

## Reproduce

In a separate research container, keep `/workspace/edge` as the working directory
to avoid importing an old installed source package. Mount the model cache at
`/models`, the repository read-only at `/workspace`, and only
`/workspace/data/benchmarks/reports` writable. Do not pass the live `.env`.

```text
python -m pip install --target /models/experiments/awiros/deps --no-cache-dir -r /workspace/edge/requirements-awiros-research.txt
python -m sentinel_edge.awiros_experiment --cache /models/experiments/awiros
```

For Awiros, set `PYTHONPATH=/models/experiments/awiros/deps:/workspace/edge`
before Python starts so its NumPy/OpenCV dependencies are actually isolated.
Set `OMP_NUM_THREADS=2` and `OPENBLAS_NUM_THREADS=2` to reproduce this CPU run.

```text
python -m sentinel_edge.ocr_selection_benchmark --manifest /workspace/data/benchmarks/nivu/manifest.json --cache /models --backend awiros --family all --report /workspace/data/benchmarks/reports/ocr-selection-awiros-validated.json --viewer /workspace/data/benchmarks/reports/ocr-selection-awiros-validated.html
```

Run `--backend cct` in the existing base image with its cached FastPlateOCR
assets, and `--backend parseq` with the already cached PARSeq research packages.
The latter must use `TORCH_DISABLE_NATIVE_JIT=1`. Record the actual device per
report; do not infer GPU comparisons from these CPU pilot results.

Serve only `data/benchmarks/reports` on loopback. The HTML page uses no scripts,
remote images, CDN resources or alert API requests. Aggregate JSON contains no
plate text or media. The live worker remains unchanged.

## Further fine-tuning: gates and methodology

**No additional training was performed.** Three frames of one vehicle cannot
support meaningful training, model selection and independent validation.
Training on the failing example and claiming improvement on it would be leakage.

Before further fine-tuning, obtain a diverse, permitted corpus with checked
transcriptions and original provenance. A candidate corpus is
https://huggingface.co/datasets/thundarstrom/indian-anpr-ocr-corpus ; its card
declares CC BY 4.0 and 18,537 crops, but underlying source provenance and duplicate
identities remain to be established. Its strict common-format filter excludes
some formats and is not an exhaustive legal Indian plate specification. It has
not been downloaded or accepted as training data. A published licence tag alone
does not establish the intended corpus's completeness or labelling accuracy.

The approved training direction remains:

1. Validate permissions, transcriptions, unreadable labels and exact/near duplicate
   images. Split by vehicle/registration/source passage before augmentation.
   Preserve the current pilot solely as an already-seen development regression.
2. Use Awiros as the primary starting checkpoint, retaining its known dictionary
   and strict model shape contract. Train a PARSeq challenger on identical
   training vehicles and evaluate identical untouched test vehicles.
3. Begin with recogniser-head adaptation and conservative learning rates, then
   selectively unfreeze the image encoder if development results improve. Profile
   GPU memory first; small batches, accumulation and mixed precision are proposed
   options, not validated 6 GiB training claims.
4. Balance single/two-row layouts, states, fonts, angle, size and lighting. Include
   realistic motion blur, compression and noise without inventing readable
   labels for genuinely unreadable samples. Synthetic plates supplement real
   data; they do not establish real-camera accuracy.
5. Use early stopping on separate validation vehicles. Measure whole-plate
   precision/coverage, character errors and candidate recall, not only training
   loss or frame accuracy. Calibrate acceptance on development data and check
   calibration on untouched test vehicles. Preserve out-of-format outputs as
   uncertainty instead of silently forcing a common registration pattern.
6. Promote a checkpoint only after independent gains, original/processed ablations,
   model-export parity, GPU profiling, two-feed shadow processing and privacy checks.

Remaining limitations: one seen vehicle, approximate crop coordinates, synthetic
degradation, uncalibrated CTC evidence, no GPU model/export test, no automatic
localisation comparison for the new model, no independent two-row/night test,
and no new fine-tuned checkpoint. Zero wrong readings in this tiny pilot is not
proof of zero false positives.
