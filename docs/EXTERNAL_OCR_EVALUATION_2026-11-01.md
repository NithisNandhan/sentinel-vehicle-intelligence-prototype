# Separate-source Indian OCR evaluation — 1 November 2026

The current adapted Awiros checkpoint remains the best of the three tested readers
on this frozen external diagnostic. Its result is **137/200 (68.5%)**, not a general
90.3% CCTV-accuracy claim. No model, confidence threshold or application default was
changed, and no training used this evaluation source.

## Dataset acquisition and actual contents

The academic [Indian vehicle license plate dataset](https://www.kaggle.com/datasets/saisirishan/indian-vehicle-dataset)
was released by Sai Sirisha Nadiminti, Pranav Kant Gaur and Abhilash Bhardwaj; see
their [paper](https://arxiv.org/abs/2207.06657). Its original release has XML plate
texts/boxes and web-photograph and mobile-video-derived acquisition groups.

Downloaded official version 1: 186,679,675 bytes, archive SHA256
`bd7ea8e07f60fdd6fe35218184d75f3259f2a5005caa3dba6b6bfeba1d740b3c`.
The actual archive contains 1,698 image files and 1,697 XML annotations. It was not
part of the four-source corpus used for our adaptation. Nevertheless, it overlaps
with existing material: a different dataset name did not guarantee new examples.

The source declares CC BY-NC-ND 4.0. This run is private noncommercial evaluation
only. Source images/crops and their image-containing viewer remain local and are
not redistributed through GitHub; source, permission and annotation receipts are
retained. The app's licence does not replace the dataset's terms.

Other candidates were checked against actual files:

- DataCluster's current public sample has 47 images/XML files; only 20 XML files
  contain a text attribute (25 nonempty object texts), despite broader card claims.
  Several initially checked XMLs contained only boxes. It was not scored here.
- The advertised thundarstrom OCR corpus currently contains only metadata/README,
  not the claimed image corpus.
- Bharat Scene Text requires gated access; no access form was submitted.
- IURS/NPDS is detection-oriented with pre-augmented images and no established OCR
  transcriptions in its description. It was not treated as labelled recognition data.

## Freeze and leakage guards

The 200-reference test manifest was frozen before predictions. It has one image
per registration, balanced by acquisition group using deterministic hash ordering:
67 state-wise OLX images, 67 Google images and 66 video-derived stills. No model
score or crop-quality score selected the easiest samples.

Preparation accounted for:

| Outcome | Count |
|---|---:|
| Plate objects processed after image/dimension checks | 1,692 |
| Previously seen registration observations excluded | 724 |
| Perceptual similarity to a prior training crop excluded | 3 |
| Unsupported reference text excluded | 1 |
| Repeated-registration observations excluded | 586 |
| Frozen scored registrations | 200 |
| Eligible registrations reserved and unscored | 178 |

Additionally four annotations had dimension mismatches and one lacked its matching
image. All exclusions are retained locally; invalid source records were not silently
replaced after model results were examined.

Exact registration, original-image-byte and decoded-crop-pixel checks cover the
previous 30,110-crop corpus. Perceptual comparison uses 64-bit DCT pHash, Hamming
distance <=6, against the 1,102 actual adaptation training crops, plus near-duplicate
checks within the candidate pool. Such a guard is conservative and not exhaustive
vehicle identity verification. Crop RGB identities use the existing corpus convention.

A post-freeze audit verified 200 distinct registrations, zero retained overlap with
the prior corpus by registration/original bytes/crop pixels, and all crop hashes.
The manifest SHA256 is
`048703fed7286d96de50abe02d312d4648e477dbe794dc4df8abcff1520fc1ba`.

The frozen reference-normalisation policy uppercases and removes whitespace/hyphens,
accepting alphanumeric strings of length 5–15 without OCR correction or a guessed
legal-format filter. None of the selected 200 references lacks letters or digits.
The preparation recipe preserves that original policy for reproducibility; a
normalisation unit test was corrected rather than changing the frozen selection.

References are the authors' XML transcriptions, not a new independent human review.
Some may contain annotation errors or ambiguous/unreadable plates. Public pretraining
overlap for either base model is unknown. Thus these are newly admitted project-test
examples, not a certified dataset unseen by every pretrained model.

## Measured fixed-checkpoint results

All candidates used the same frozen crops, publisher preprocessing, FP32, batch one,
and a separate RTX 3050 6GB test container. No parameter search or further training
occurred. The earlier NIVU development sequence was deliberately omitted.

| Model | Whole plates exact | Character errors / 1,894 | Median OCR ms | P95 OCR ms |
|---|---:|---:|---:|---:|
| Original Awiros | 121/200 (60.5%) | 173 | 20.373 | 34.388 |
| Adapted Awiros | **137/200 (68.5%)** | **135** | 21.510 | 33.206 |
| SVTRv2-B | 117/200 (58.5%) | 177 | **11.728** | **25.530** |

OCR timing includes preprocessing/transfers/output materialisation, not plate
detection, tracking, temporal fusion, network delivery or UI. It is one warmed
diagnostic pass, not a complete-video throughput or endurance benchmark. Input
dimensions and runtime conditions differ from the earlier 31-image experiment;
do not interpret its different timings as a new optimisation.

Adapted Awiros gained 17 correct reads over the parent and lost one: net +16/200.
Against SVTRv2-B, adapted Awiros alone was correct on 36; SVTRv2-B alone was correct
on 16; both were wrong on 47. Their reference-aware union would be 153/200, but this
is an oracle upper bound, not an achieved ensemble or a rule for choosing outputs.

| Source group | N | Parent Awiros exact | Adapted exact | SVTRv2-B exact |
|---|---:|---:|---:|---:|
| State-wise OLX | 67 | 33 | 37 | 32 |
| Google images | 67 | 47 | 49 | 46 |
| Video-derived stills | 66 | 41 | 51 | 39 |

Native-size breakdown:

| Native crop width | N | Parent exact | Adapted exact | SVTRv2 exact |
|---|---:|---:|---:|---:|
| 32–63 px | 34 | 12 | 12 | 12 |
| 64–127 px | 68 | 44 | 49 | 44 |
| 128+ px | 98 | 65 | 76 | 61 |

There are no <32px cases in this selected sample. Among the 39 crops with native
width/height <2.5, adapted Awiros reads 17 and SVTRv2 reads 18. Aspect ratio is a
proxy, not a manually verified row-layout label. These counts motivate inspection
of small inputs, layout and errors; they do not prove blur alone is the cause.

## Decision and next experiments

Keep adapted Awiros as the current reader. Its prior 28/31 diagnostic was too narrow
to represent general CCTV performance; this source-separated result gives a more
challenging measure while retaining important reference/pretraining limitations.

Useful next work, not implemented in this experiment:

1. Human-review the apparent failure cases and ambiguous XML references without
   turning the tested examples into further training data.
2. Test native frame selection, crop padding/alignment and two-line layout handling
   on a separate development set; measure changes against the unchanged control.
3. Evaluate a second-reader disagreement policy because SVTRv2 recovered 16 misses.
   Scores from different readers must not be directly treated as comparable calibrated
   probabilities, and the oracle union is not a deployable selector.
4. If training is needed, acquire representative training data with appropriate
   permission, keeping this test source excluded. Preserve the 178 unscored reserved
   registrations for a later frozen evaluation, with a source/identity audit.
5. Validate the resulting full multi-frame workflow on separately labelled CCTV
   passages. These still results cannot establish vehicle recall, accepted-alert
   precision or resilience of complete video processing.

## Reproduction and artefacts

- `tools/acquire_external_ocr.py`: official source/version/permission and bounded
  resumable acquisition.
- `tools/prepare_external_ocr_test.py`: native crops, training-overlap checks,
  source balancing and immutable test freeze; refuses to overwrite a frozen manifest.
- `tools/compare_svtrv2.py`: shared scorer now supports still-only tests up to 250
  samples without attaching an unrelated historical temporal sequence.
- `tools/render_svtrv2_comparison.py`: local private-evaluation viewer flag, source
  permissions, measured-input hash check and plate-free aggregate output.

Acquire on Windows:

```powershell
& backend/.venv/Scripts/python.exe -X utf8 tools/acquire_external_ocr.py --root data/benchmarks/fresh-indian-20261101/academic
```

Use the existing cached NVIDIA image with `edge` and `tools` read-only under
`/workspace`, old benchmark data read-only at `/benchmarks`, model volume read-only
at `/models`, SVTR assets read-only at `/research`, and the new evaluation directory
writable at `/fresh`. Within that isolated container:

```text
PYTHONPATH=/workspace/edge python /workspace/tools/prepare_external_ocr_test.py --root /fresh --previous /benchmarks/indian-corpus --maximum 200
PYTHONPATH=/workspace/edge python /workspace/tools/compare_svtrv2.py --backend svtrv2 --manifest /fresh/adaptation_manifest.json --output /fresh/results/svtrv2-test.json
python /workspace/tools/run_ocr_research.py /workspace/tools/compare_svtrv2.py --backend awiros-parent --manifest /fresh/adaptation_manifest.json --output /fresh/results/awiros-parent-test.json
python /workspace/tools/run_ocr_research.py /workspace/tools/compare_svtrv2.py --backend awiros-adapted --manifest /fresh/adaptation_manifest.json --output /fresh/results/awiros-adapted-test.json
```

Full references, native crops, prediction files, exclusion records, reserve and
permission notices remain ignored locally under
`data/benchmarks/fresh-indian-20261101/academic/`. The private local viewer is
`data/benchmarks/reports/external-ocr-20261101.html`. Only aggregate findings,
protocol, code and tests are published to GitHub.

## Verification

21 focused tests passed, covering reference normalisation without character
correction, compatible RGB image identities, perceptual hash behaviour, frozen
input checks, local-viewer permission restrictions and the existing CTC/selection
benchmarks. Python compilation passed. The real-model GPU runs all completed on
the same frozen 200-case input. Viewer verification checked every source crop and
the 200 result cards; the long provenance captions were given mobile wrapping.
