# Indian OCR adaptation pilot — 27 September 2026

Actual training and matched benchmark reruns are complete. This is a small CTC
output-layer adaptation, not a fully retrained encoder or production upgrade.
The live worker and its acceptance gates remain unchanged.

## Measured result

| Evaluation | Original Awiros | Adapted Awiros |
|---|---:|---:|
| Separate test registrations: raw whole-plate exact | 9 / 10 | 9 / 10 |
| Character errors / test characters | 2 / 92 | 1 / 92 |
| Character error rate | 2.17% | 1.09% |
| True plate in retained top 10 candidates | 10 / 10 | 10 / 10 |
| Multi-frame conditions: accepted correct | 13 / 21 | 13 / 21 |
| Multi-frame conditions: wrong accepted | 0 / 21 | 0 / 21 |
| Multi-frame conditions: abstained | 8 / 21 | 8 / 21 |

The remaining misread got one character closer. Whole-plate recognition and the
temporal acceptance boundary did not improve. Do not present 9/10 clear stills
as 90% general CCTV accuracy. The checkpoint remains experimental.

The temporal test uses the existing NIVU passage: one already-seen vehicle,
three distinct source images and 21 degradations, not 21 independent vehicles.
Replay timestamps are assigned, not verified camera PTS. Both runs use identical
detector, quality weights, deskew, decoder and evidence gates. Concurrent two-core
CPU runs are not a controlled speed comparison.

## Cleaning and separation

Source: [Bapatla dataset, DOI 10.5281/zenodo.13954136](https://zenodo.org/records/13954136),
publisher-declared CC BY 4.0. Attribution: Bhuma Chandra Mohan; CH V M S N Pavan
Kumar; Katakam; Bethala Venkata Varma; Vatada Sai; Kagitha Sambaiah; Bapatla
Engineering College, 2024. Earlier Awiros pretraining overlap and acquisition
provenance of every photograph remain unknown.

- Not a 1,700-image training run. Original workbook unchanged. Excluded mixed
  filename groups and unmatched/partial labels; format matching is not verification.
- Fetched 48 regional proposals, reusing cached images: 6,742,627 bytes via
  bounded HTTP ranges. ZIP member CRCs verified; whole archive MD5 not verified.
- Combined proposals with eight prior verified samples. Automatically selected
  the highest-confidence plate region; labels never selected a box.
- Reviewed 53 native crop/label pairs visually, retaining 48 registrations.
  Excluded AP2, BR3, HR1, ML2 and NL4 for mismatch/ambiguity or insufficient
  clarity. Previous AN4/AP12 quarantine remains. No OCR pseudo-labels used.
- Integer-sliced lossless PNG crops preserve original pixels. Only network input
  and review display copies resize. These are individual stills, not sequences.
- SHA256 order of `sentinel-20260927-` plus registration gives 28 train, 10
  validation, 10 test registrations, before augmentation. Guards reject cross-split
  registrations, exact original/crop duplicates, unreviewed/altered files and
  escaping paths. Full-corpus perceptual audit is not claimed.
- NIVU was not used for training/validation. Test labels were queried after
  checkpoint selection. Held out from this adaptation only, not guaranteed absent
  from publisher pretraining. No genuine Sentinel footage used or retained.

Ignored manifest: `data/benchmarks/bapatla/training_manifest.json`; SHA256:
`1b7fd4f43d662efc7f724df1c842e457d8a1e6537e7c6dc97515b8e4214bf559`.
Original images, native crops, review sheets and workbook remain local.

Reproduce preparation using the earlier inventory/workbook and cached detector:

```text
python tools/prepare_indian_ocr_pilot.py --root data/benchmarks/bapatla
python -m sentinel_edge.indian_ocr_finetune --prepare-crops /workspace/data/benchmarks/bapatla
python tools/review_indian_ocr_pilot.py --root data/benchmarks/bapatla --verified AN1 AN2 AN5 AN6 AN7 AN10 AP1 AP10 AR2 AR4 AS1 AS2 BR4 CG1 CG4 CH2 CH3 DL1 DL2 DN1 DN2 GA1 GA2 GJ1 GJ2 HP1 HP2 HR2 JH1 JH2 JK1 JK2 KA1 KA2 KL3 KL4 MH1 MH2 ML1 MN2 MN3 MP1 MP2 NL2 OD3 OD4 OD6 OD11
```

That list records the actual visual-review decisions. New images require new
review, not automatic label acceptance.

## Optimisation and checkpoint

Parent SHA256:
`f9f1264e0c115a239ca6d6700a74ceffab17168b75dafc6c169c232012c68e47`.
Only `head.ctc_head.fc.weight` and `.bias` trained: **7,744 parameters**.
Encoder, sequence encoder, NRTR branch and batch-normalisation buffers unchanged,
checked byte-for-byte. Cached frozen features feed the trainable linear CTC head.
This is real weight optimisation, but not learning new encoder image features.

CPU Paddle 3.2.2; seed 20260927; Adam rate 0.0002; batch 8; gradient norm clipped
at 1.0; 60 epochs. Four training views: original, mild blur, 0.8 resolution and
slight darkness/noise. Correlated augmentations never count as temporal votes.
Raw logits enter [Paddle CTCLoss, which includes softmax internally](https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddle/nn/CTCLoss_en.html).

Validation orders exact count, character errors, then CTC loss, with patience 12.
Validation was already 10/10 exact before tuning; loss fell 0.03289 → 0.01367.
Epoch **55** selected. Test loss fell 0.67641 → 0.40529. Lower loss alone is not
a deployment recommendation.

Adapted SHA256:
`1f0455587b1a5fb8437798f41f950a0017dd9b7d56be9b8e0195bcf332bdbb30`.
Docker model volume: `/models/experiments/awiros/adaptations/bapatla-head.safetensors`
and companion JSON. Last epoch saved separately. Original weights untouched;
weights local only, not committed/uploaded.

Explicit loading checks parent/dictionary/file hashes, tensor keys/shapes, actual
frozen-state equality and changed head tensors. Strict reload matched cached
predictions within 1e-5; separate full-model test inference reproduced metrics.

## Re-run and inspect

Use the [isolated Awiros setup](OCR_MODEL_SELECTION.md): repo read-only `/workspace`,
reports separately writable, model cache `/models`, workdir `/workspace/edge`,
`PYTHONPATH=/models/experiments/awiros/deps:/workspace/edge`,
`OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`. No live `.env`.

```text
python -m sentinel_edge.indian_ocr_finetune --manifest /workspace/data/benchmarks/bapatla/training_manifest.json --cache /models/experiments/awiros --output /models/experiments/awiros/adaptations/bapatla-head.safetensors --report /workspace/data/benchmarks/reports/awiros-finetune.json --viewer /workspace/data/benchmarks/reports/awiros-finetune-test.html --epochs 60
```

Add `--evaluate-only` and report `awiros-finetune-test.json` to evaluate existing
weights without training. Run `sentinel_edge.ocr_selection_benchmark` on NIVU with
`--quality-weighted --automatic-crops --deskew --family all`, once without a
checkpoint (`awiros-parent-rerun-all` outputs), once with
`--checkpoint /models/experiments/awiros/adaptations/bapatla-head.safetensors`
(`awiros-finetuned-all` outputs). Generate summary:

```text
python -m sentinel_edge.ocr_adaptation_report --reports data/benchmarks/reports --manifest data/benchmarks/bapatla/training_manifest.json
```

[Local viewer](http://127.0.0.1:7777/awiros-finetune-summary.html) links all test
crops/readings/candidates and both temporal curves. Single-still cards say RAW
MATCH / RAW ERROR, not accepted alerts. JSON contains no plate strings/media;
licensed-data HTML stays ignored and loopback-only, with no external requests.

**161 edge tests passed.** GPU/export parity, live shadow validation, diverse
difficult/two-row/night inputs and confidence calibration remain gates. Further
tuning needs more diverse difficult examples; lowering evidence gates is not a
substitute. Production remains unchanged.
