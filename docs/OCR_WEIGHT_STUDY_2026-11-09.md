# OCR reading diagnosis and weight study — 9 November 2026

## Experiment contract

Improve exact plate readings from fixed crops. Vehicle/plate detection, crop
coordinates and alert acceptance thresholds remain unchanged. Preserve the
current checkpoint and never select weights by their confidence scores alone.

Baseline: the current adapted Awiros checkpoint, SHA256
`c2ba896d994b174071c95ace0f5ffc8a61cec3fc6eecce8f0bc5feb7dec4456d`.

Data: 802 existing photographic training crops; 37 corpus-validation crops for
selection; 31 corpus-test crops, 18 readable human-labelled Mumbai crops and
200 separate-source evaluation-only crops for regression checks. These sets
have been examined in previous development. They are not a fresh blind test;
publisher labels and pretraining overlap retain the existing caveats. Disputed
Mumbai label 21 is excluded. Training excludes all evaluation registrations and
exact crop hashes. No live footage or new model downloads are required.

## Order and methods

1. Verify training-forward versus serving-forward posterior parity.
2. On the same human-labelled crops, compare original native, previously
   rectified, padded and contrast-adjusted inputs with frozen weights. Measure
   exact strings, missing/extra/substituted characters, beam-search truth
   inclusion and direct reference likelihood. Alternative views are not
   independent frames.
3. Test six changes to the actual CTC blank-class bias weight. A cached-logit
   sweep is mathematically equivalent to changing that output-layer bias;
   selected weights are validated by an actual saved-checkpoint reload.
4. Train four continuations from the same baseline: output head at learning
   rates 1e-5/5e-5, and the previously supported encoder tail plus head at
   2e-6/1e-5. Use two epochs, training-only photographs, CTC loss, gradient
   clipping, fixed seed and frozen batch-normalisation state.
5. Test four baseline/candidate weight blends for each continuation. Choose
   using validation exact matches, then character edit count; keep baseline
   when tied. Freeze the selection before external/manual regression evaluation.
6. Check the selected checkpoint on all regression sets and run a real
   distinct-frame sequence replay. Promotion requires validation improvement
   without fewer exact readings or more character errors on any guardrail set.
   Regressing candidates remain experimental and never replace runtime weights.

The initial phase trains only the small supported head/tail subset. Permission to change any
weight does not make random changes to all 37 million parameters a useful
search. If this experiment shows no improvement, report that result directly.

## Expanded experiment and results

The initial 22 settings produced no validation improvement. A second experiment
therefore used stronger head/tail learning rates (2e-4 and 5e-5) and unfroze the
full image-encoder/CTC path at 1e-5 for four epochs each. The auxiliary NRTR
decoder stayed frozen. In total: seven training continuations, 20 epochs,
4,020 optimizer steps, and 34 distinct bias/blend settings (40 evaluations,
including the six repeated bias controls). The expanded run explicitly checked
that selected weights changed; its full-path final epoch changed 2,647,159
stored values. This does not mean every unfrozen parameter changed numerically.

The validation-selected candidate was the full-encoder second epoch, without
baseline blending. Strict saved-checkpoint reload and serving-forward parity
passed. It **failed promotion**:

| Fixed crop set | Current exact plates | Candidate exact plates | Current character errors | Candidate character errors |
| --- | ---: | ---: | ---: | ---: |
| Corpus validation | 29/37 | 29/37 | 42 | 36 |
| Mumbai human-labelled crops | 14/18 | 12/18 | 13 | 15 |
| Corpus test regression | 28/31 | 28/31 | 13 | 12 |
| Separate-source regression | 137/200 | 135/200 | 135 | 123 |

These numbers measure raw greedy OCR strings on fixed crops, not exhaustive
video recall or accepted alerts. The current CTC candidate decoder already
finds the correct top candidate in 15/18 Mumbai crops; that additional recovery
is existing decoding behaviour, not a gain from the new weights.

### Failure isolation

- Shared tensor preprocessing and training-forward/serving-forward parity
  passed. There is no demonstrated normalization mismatch.
- Previous native contextual crops read 5/18 correctly; the prepared rectified
  crops read 14/18. This is a manual geometry diagnostic, not an accuracy claim
  for the newly integrated automatic crop policy.
- Applying padding to every rectified crop reduced exact reads to 8/18; applying
  contrast adjustment to every crop gave 13/18. Neither becomes a global default.
- Of four remaining raw failures, reference IDs 01, 04 and 16 had no correct
  string in beam@100. Correct-text likelihood gaps were 8.08, 6.70 and 4.76 log
  units respectively. The missed/small characters are weak in the model output;
  simply showing another candidate is insufficient. Visual inspection also
  showed strong character-size differences within the painted registrations.
- Reference 20 was a close substitution (log gap 0.18); its correct string was
  present in the beam and ranked first by the existing plate candidate decoder.
- The candidate reduced missing-character counts but introduced substitutions.
  In Mumbai, missing characters fell 7 to 4 while substitutions rose 5 to 10.
  On the separate source, missing characters fell 33 to 18, but whole-plate
  accuracy still fell. Higher apparent character completeness is not success.
- On one existing three-frame NIVU development passage, baseline and candidate
  both read 3/3 frames and fused correctly, including the existing acceptance
  gate. Original capture timing is unknown; 400 ms replay spacing was assigned.
- A separate replay associated one retained Mumbai track to a human reference
  using exact PTS and strong native-pixel template agreement, without matching
  predicted strings. Neither model read a correct individual frame or fused
  plate on that track. Six tracks could not be associated safely and were
  excluded. This small check does not show fusion losing an already correct
  frame, and does not establish video-wide fusion correctness.

### Decision and next work

The current model remains active. No OCR weights, detection/crop settings,
acceptance thresholds or watchlist behaviour were changed in deployment.
The rejected candidate is retained locally at
`data/benchmarks/reports/ocr-weights-20261109-extended/candidate.safetensors`
with its hashed metadata; it is not included in the code push.

The next useful training data are labelled **actual CCTV crops** covering the
remaining missing/small characters and mixed character scales, with all frames
of a vehicle kept in one split. Reserve new vehicles/footage before training.
Reusing more epochs on these same 802 still photographs did not deliver a
better replacement. Separately, test image-only scale/layout normalization and
selective retry policies; do not turn blanket padding or contrast on.

Five study safeguards, four checkpoint/fine-tune tests, eight fusion tests,
five corpus/shadow tests and four integrity tests passed (26 tests). The
corpus/shadow tests were rerun in the actual OCR container after the Windows
interpreter lacked optional OpenCV. All recorded failures and final aggregate
results are preserved; no test expectation was weakened.
After restoring recorded analysis, the baseline OCR, vehicle detector, plate
detector and text localizer also passed the real four-model GPU preflight.

Public plate-free measurements:
`docs/evidence/ocr-weight-study-20261109.json`. Detailed local outputs are under
`data/benchmarks/reports/ocr-weights-20261109*` and remain excluded from Git.

## Reproduction

Use the existing GPU edge image and isolated Paddle launcher, with model cache
read-only at `/models`, source read-only at `/workspace/edge` and
`/workspace/tools`, and local `data` at `/workspace/data`. Stop recorded analysis
temporarily to free the GPU. No network is needed for inference or training.

```text
python /workspace/tools/run_ocr_research.py /workspace/tools/study_ocr_weights.py
  --corpus /workspace/data/benchmarks/indian-corpus
  --external /workspace/data/benchmarks/fresh-indian-20261101/academic
  --review /workspace/data/benchmarks/recorded-footage-20261104/manual-review
  --output /workspace/data/benchmarks/reports/ocr-weights-CHOOSE-FRESH-NAME
  --epochs 2
```

Detailed readings and candidate checkpoint(s) stay in ignored local data. Only
plate-free aggregate results and reproducible source belong in the code commit.
The current checkpoint remains the rollback target throughout.

For the expanded run add `--extended --epochs 4` and optionally
`--reuse-diagnosis /workspace/data/benchmarks/reports/PRIOR-RUN/report.json`.
Diagnosis reuse requires identical baseline and input manifest hashes.
`tools/replay_ocr_weight_study.py` replays the licensed NIVU passage;
`tools/diagnose_runtime_ocr_tracks.py` performs the conservative exact-PTS pixel
association replay. Their detailed outputs also remain local.
