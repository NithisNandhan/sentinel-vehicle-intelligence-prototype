# OCR failure review and preprocessing experiment — 1 November 2026

**Decision: retain native adapted Awiros.** None of the seven fixed development
policies improved exact plate recognition. No weights, serving configuration,
acceptance gates, dashboard or teammate video-job code changed. This work adds a
reproducible diagnostic experiment and private review viewer, not a new accuracy
claim or production ensemble.

## Failure review

Reviewed all 63 differences in the already-scored external 200-image diagnostic
using native crops enlarged with nearest-neighbour interpolation for inspection.
This was AI visual review, **not independent human adjudication**. The original
references and reported 137/200 score remain unchanged.

Objective dimensions: 22 of the 63 crops are narrower than 64 pixels; 16 have
native height below 16 pixels. These groups overlap.

Each case was assigned one primary review hypothesis. Multiple causes may coexist;
the categories are not verified causal attribution:

| Review hypothesis | Cases |
|---|---:|
| Small or blurred glyphs | 25 |
| Reference/transcription review needed | 12 |
| Perspective or rotation | 8 |
| Inserted, repeated or omitted characters | 7 |
| Two-line layout | 6 |
| Light lettering on red/special layout | 3 |
| Occluded or clipped characters | 2 |

Entry 4 visibly supports the model reading rather than its XML reference. Other
flagged references have possible confusable characters, ambiguous text or source
problems. **Twelve review flags do not establish twelve annotation errors.** A human
should review the native image and original scene before accepting corrections.
No OCR output was substituted for ground truth. One crop also resembles a foreign
registration and is truncated, illustrating that this source is not a certified
all-Indian, fully-readable recognition test.

## Development experiment

Used all 37 existing indian-corpus validation crops, with the cached adapted
Awiros checkpoint and unchanged publisher preprocessing. This validation data was
already used during adaptation/model selection; it is development evidence, not
a new independent test. The 178 reserved external registrations were not scored
or visually inspected. The previously scored 200 images were not rerun under any
new policy because no development candidate justified promotion.

| Fixed policy | Exact /37 | Character errors | Recovered | Regressed |
|---|---:|---:|---:|---:|
| Native greedy baseline | **29** | **42** | — | — |
| Native CTC beam, width 32 | 29 | 45 | 0 | 0 |
| Replicate-edge padding | 26 | 52 | 0 | 3 |
| Modest luminance CLAHE | 28 | 48 | 0 | 1 |
| Existing conservative affine deskew | 27 | 46 | 0 | 2 |
| Image-selected two-row greedy | 28 | 51 | 0 | 1 |
| Image-selected two-row beam | 28 | 52 | 0 | 1 |

The row detector fired on three development crops; none was recovered, and one
previously correct reading regressed. A central whitespace gap is insufficient
to establish a reliable reading-order/layout decision. Padding, contrast and
deskew changed the input distribution without recovering a complete plate here.
Beam decoding did not fix the inserted/missing-character cluster on this set.

The initial run was preserved. A final run after replacing a deprecated GPU-sync
API and adding execution hashes reproduced every exact/error/recovery/regression
count. No policies or thresholds were tuned between runs. Timings vary with runtime
conditions. Final native median was 18.648 ms; extra-view policies include the
baseline plus additional inference and processing. These are **not full-pipeline
FPS or recorded-video endurance measurements**.

## Second reader: useful alternatives, insufficient confirmation

Ran the fixed SVTRv2-B checkpoint on the same 37 development crops: 20/37 exact,
compared with adapted Awiros 29/37. Then joined existing predictions by verified
sample ID and reference. External figures below reuse the earlier scored outputs;
they are descriptive analysis, not a newly blind ensemble test.

| Cohort | Readers agreed | Agreed correct | Agreed wrong | Disagreed |
|---|---:|---:|---:|---:|
| Existing development, 37 | 19 | 19 | 0 | 18 |
| Already-scored external, 200 | 107 | 101 | 6 | 93 |

SVTR alone recovered one Awiros miss in development, while Awiros alone recovered
ten SVTR misses. The external complement remains 16 SVTR-only versus 36 Awiros-only
correct readings. An agreement-only flag on the external set would cover 107/200
inputs, including six reference-wrong readings. Its apparent 101/107 precision is
not calibrated correctness, and correlated readers may share mistakes. No automatic
confirmation, score comparison or watchlist alert rule was added.

## What should happen next

1. Human-adjudicate flagged references in the local viewer, preserving the original
   label, adjudicated label, readability and reviewer identity separately. Do not
   train on this evaluation source or quietly rewrite its historical score.
2. Label a small representative set of **actual vehicle passages** from permitted
   recorded CCTV: plate text, readable/unreadable state, native plate bounds,
   one-/two-line layout, track identity and timestamps. Split by vehicle/source
   recording before experimenting. Transformed copies are not distinct frames.
3. Compare the current pipeline with quality-selected distinct-frame OCR and a
   dedicated row-layout hypothesis on those development passages. The current
   generic whitespace splitter should not become a default.
4. Use SVTR disagreement as review evidence initially. Any selector or confirmation
   policy needs separate development tuning followed by a new frozen evaluation.
5. Only then commission targeted training examples for verified recurring layout
   or character errors, rather than training on guessed references or adding
   unrestricted sharpening/restoration.

## Reproduction and artefacts

- `tools/audit_ocr_failures.py`: checks frozen crop/report hashes and creates seven
  private contact sheets for the existing failures.
- `edge/sentinel_edge/ocr_crop_experiment.py`: image-copy transforms, guarded row
  hypotheses and existing CTC beam decoding; no serving imports.
- `tools/experiment_ocr_crops.py`: bounded, hash-checked development runs; refuses
  overwriting outputs. Test replay is disabled; a successful candidate would need
  a separate frozen evaluation protocol before testing.
- `tools/summarize_ocr_crops.py`: paired-reader joins, review coverage validation,
  plate-free aggregate and explicitly private local viewer.
- `docs/evidence/ocr-failure-audit-20261101.json`: review categories and case numbers,
  without plate strings or source images.
- `docs/evidence/ocr-crop-ablation-20261101.json`: numeric measurements and hashes.

Reuse the cached isolated GPU container and read-only models/old corpus mounts
described in the separate-source evaluation report. Commands inside that container:

```text
python /workspace/tools/audit_ocr_failures.py --root /fresh
python /workspace/tools/run_ocr_research.py /workspace/tools/experiment_ocr_crops.py --manifest /benchmarks/indian-corpus/adaptation_manifest.json --output /fresh/crop-experiments/development-final.json
PYTHONPATH=/workspace/edge python /workspace/tools/compare_svtrv2.py --backend svtrv2 --manifest /benchmarks/indian-corpus/adaptation_manifest.json --split validation --output /fresh/crop-experiments/svtrv2-development.json
```

Existing result paths are immutable for the experiment runner; use a new output
path for an intentional reproduction and point the summarizer at that run after
reviewing its hashes. On Windows, to rebuild the existing private review viewer:

```powershell
& backend/.venv/Scripts/python.exe tools/summarize_ocr_crops.py --root data/benchmarks/fresh-indian-20261101/academic --corpus data/benchmarks/indian-corpus --review docs/evidence/ocr-failure-audit-20261101.json --output data/benchmarks/reports/ocr-error-analysis-20261101.html --aggregate docs/evidence/ocr-crop-ablation-20261101.json --allow-private-evaluation
```

The private viewer contains CC BY-NC-ND 4.0 evaluation images and stays ignored
locally. GitHub receives code, review hypotheses and plate-free aggregates only.
No dataset images, crops, screenshots of private plates, new weights or credentials
are uploaded.

## Verification

- 29 focused tests passed: native input/padding contract, immutable crops, layout
  safeguards, CTC repeated-character semantics, report/index source binding,
  paired-reader identity joins, private viewer path, and existing OCR adapters.
- All four new Python implementation files compiled successfully.
- Headless browser check: 63 cards, 63 loaded images, no page errors and no horizontal
  page overflow at 390px. The viewer responds locally on port 7777.
- Reserved-manifest SHA256 remained
  `409d1d7f459425493bd36721f2c5bfd63220c24e3254736068d7544439847064`.
- The measured experiment module matches its saved execution hash. The original
  source-separated test manifest and references were not modified.
- One read-only specialist reviewed preprocessing contracts and experiment integrity;
  findings about private paths and report-to-crop bindings were fixed and tested.
