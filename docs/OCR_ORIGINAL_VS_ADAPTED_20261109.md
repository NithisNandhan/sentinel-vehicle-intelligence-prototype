# Original Awiros versus our deployed OCR — 9 November 2026

## Question and decision

Did our earlier fine-tuning improve the original Awiros model, or merely fit its
training examples? This is a fixed-checkpoint comparison, with no new training,
threshold search, checkpoint selection or automatic model replacement.

The original [Awiros release](https://huggingface.co/Awiros/anpr-ocr) is already
an Indian-plate specialist adapted from PP-OCRv5. We compare that release with
our subsequent adaptation, not a generic OCR model with an Indian-trained one.

## Checkpoints and protocol

- Original publisher SHA256:
  `f9f1264e0c115a239ca6d6700a74ceffab17168b75dafc6c169c232012c68e47`.
- Deployed adaptation SHA256:
  `c2ba896d994b174071c95ace0f5ffc8a61cec3fc6eecce8f0bc5feb7dec4456d`.
- Publisher revision: `eadbc5ae4faa8284c8a547a60142fd555c3059ca`.
- Our adaptation used 802 publisher-labelled training photographs and 300
  synthetic examples. It updated a bounded encoder tail and CTC head; epoch 8
  was selected using validation during a 12-epoch run. That validation set is
  therefore not untouched test evidence.
- Both checkpoints use the same architecture, image crops, shared resizing,
  padding, normalisation, character dictionary and greedy CTC decoding. The
  publisher's inference source uses the same preparation approach.
- Three warm-up inferences precede timing. Model order alternates per crop.
  Timing covers OCR input preparation and synchronised inference, excluding
  decoding, detection, candidate fusion and alert delivery.
- All adaptation training registrations and crop hashes, including synthetic
  examples, are checked against evaluation slices. The exact original training
  manifest must match the hash stored with our deployed checkpoint.
- Crop bytes and source-manifest bindings are verified. Mumbai ID 21 remains
  excluded because its transcription was disputed. Mumbai also receives the
  same existing single-frame candidate decoder for both models, reported
  separately from raw greedy OCR.

Training photographs are an explicitly **in-sample control**, not accuracy
evidence. Corpus labels include publisher/AI-agreement and legacy review tiers;
they are not all independent human truth. Mumbai has user-reviewed labels.
RoadText uses publisher transcriptions and selected native crops from moving
camera footage. Repeated RoadText crops are not independent vehicles.

The evaluation sets were kept out of our adaptation's optimizer batches, but
have since been examined for development. The original publisher's pretraining
membership is unknown. This comparison measures relative fixed-crop behaviour;
it cannot establish unseen-camera or full-video CCTV accuracy.

## Reproduction

Use the cached `sentinel-prototype-edge-worker:latest` image, `--gpus all`,
`--ipc=host`, `--network none`, and the isolated launcher. Mount `edge` and
`tools` read-only under `/workspace`, `data` under `/workspace/data`, and
`sentinel-prototype_model_cache` read-only under `/models`. The command is:

```text
python /workspace/tools/run_ocr_research.py /workspace/tools/compare_awiros_checkpoints.py
  --output /workspace/data/benchmarks/reports/awiros-original-vs-deployed-20261109
```

Line breaks above are for readability. A fresh output directory is required.
Pause the recorded worker during the controlled GPU measurement, then restore
it with its unchanged deployed checkpoint. No assets are downloaded by this
comparison. Detailed readings remain in ignored `data/benchmarks/reports/`;
public reporting includes aggregates only.

## Measured results

**Our earlier adaptation improved exact readings on every evaluated slice. Keep it active.** This differs from the later RoadText continuation, which was rejected for cross-domain regressions.

All rows below use raw greedy CTC with identical frozen crops:

| Slice | Original exact | Our adaptation exact | Exact-read gain | Original character errors | Adapted character errors |
|---|---:|---:|---:|---:|---:|
| Corpus validation (used for original selection) | 24/37 | 29/37 | +13.51 percentage points | 60 | 42 |
| Corpus test | 22/31 | 28/31 | +19.35 percentage points | 23 | 13 |
| External regression | 121/200 | 137/200 | +8.00 percentage points | 173 | 135 |
| Mumbai manual crops | 11/18 | 14/18 | +16.67 percentage points | 17 | 13 |
| RoadText held-out crops | 42/106 | 43/106 | +0.94 percentage points | 230 | 206 |
| Training photographs - in-sample control | 483/802 | 621/802 | +17.21 percentage points | 807 | 470 |

Paired outcomes show what changed, rather than just comparing aggregate scores:

| Slice | Previously wrong crops recovered | Previously correct crops lost |
|---|---:|---:|
| Corpus test | 6 | 0 |
| External regression | 17 | 1 |
| Mumbai manual crops | 3 | 0 |
| RoadText held-out crops | 2 | 1 |

With the **same existing single-frame candidate decoder**, Mumbai improves from **13/18 original** to **15/18 adapted**. Two crops were recovered and none of the original exact readings were lost. This is a separate decoding measurement, not a multi-frame result.

Training photographs improve from 483/802 to 621/802. That is evidence of fitting the training set; it is not included as generalisation evidence. Improvements on Mumbai, the corpus test and external regression show that the benefit is not confined to the training images. These remain previously examined crop sets, with the label-quality and pretraining-overlap limitations above.

RoadText gives a useful boundary: only one additional exact crop, across 106 crops representing 13 registrations. The earlier adaptation does not solve the moving-camera domain. Its character error count still falls from 230 to 206. No broad claim about all CCTV feeds follows.

OCR-only median on the 200 external crops: **13.44 ms original versus 13.51 ms adapted**. The architecture and its cost are essentially unchanged in this run. Small timing differences should not be treated as a speed improvement.

Nine focused comparison/checkpoint/metric tests passed. Both full checkpoints loaded strictly; our exact original training manifest matched the deployed metadata, and evaluation registration/hash separation passed against all 1,102 adaptation training examples. No fitting, threshold tuning or replacement occurred.

The dashboard worker was restored with the same deployed adaptation. Aggregate evidence: [machine-readable comparison](evidence/awiros-original-vs-adapted-20261109.json). Detailed local report: `data/benchmarks/reports/awiros-original-vs-deployed-20261109/report.json`.
