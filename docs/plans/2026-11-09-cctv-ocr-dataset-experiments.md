# CCTV OCR dataset experiments

Approved direction: 9 November 2026. Keep the crop pipeline fixed and improve
recognition using representative labelled traffic-video data.

## Execution order

1. Fine-tune the current adapted Awiros model on a verified Indian plate subset
   of **RoadText-3K**, replacing the unavailable StreetOCRCorrect corpus.
2. If the Indian evaluation is insufficient, fine-tune on UFPR-SR-Plates alone.
3. Fine-tune on RodoSol-ALPR alone.
4. Fine-tune on UFPR-SR-Plates and RodoSol together, using balanced sampling.
5. Compare all completed candidates with the unchanged current model and use
   the best candidate that passes Indian CCTV evaluation and runtime checks.

**User review gate:** after fine-tuning on each dataset, run regression tests,
report the benchmarks, and stop for the user's review before starting the next
dataset. Do not automatically continue to UFPR, RodoSol or their combination.

Independent dataset experiments start from the same preserved current baseline
so an earlier failed run does not contaminate the next. Additional sequential
adaptation is a separate experiment, not silently mixed into this comparison.
The existing baseline SHA256 is
`c2ba896d994b174071c95ace0f5ffc8a61cec3fc6eecce8f0bc5feb7dec4456d`.

## Access audit

Checked the publishers' repositories and author project page on 9 November 2026.
The StreetOCRCorrect public figures are demonstrations, not a training set.
The new first stage acquires one Indian RoadText archive and its matching public
annotations. Record actual source coverage in the experiment receipt rather
than describing a bounded subset as all 1,000 Indian videos.

### RoadText-3K — active first stage

- Publisher: https://datasets.cvc.uab.es/roadtext3k/
- Paper: https://marcalr.github.io/pdfs/DAS22.pdf
- Source includes 1,000 Indian ten-second, 1280x720 vehicle-mounted-camera clips.
  It is not fixed CCTV. Source text annotations include a dedicated plate enum,
  line transcriptions and track IDs. Some registrations occupy multiple lines.
- The public JSON has missing/null transcriptions for some clips. Check plate
  text coverage before downloading the corresponding large video archive.
- Only readable, nonempty publisher plate transcriptions can supervise OCR.
  Keep complete-registration-shaped source lines for this first experiment;
  quarantine fragments, conflicting track labels and unsupported formats.
  Never generate a missing registration from our model or a format rule.
- Indian membership comes from an official `I_*.tar.gz` member path. Do not
  infer country solely from numeric annotation IDs.
- If official split IDs are unavailable, declare custom ordered-video blocks
  and quarantine registrations crossing blocks. Preserve an untouched RoadText test cohort and the
  existing Mumbai, corpus and external-source regression sets.
- The paper expressly provides training partitions for academic model research.
  A separate RoadText-3K licence was not found. Do not assume permission to
  redistribute source media or claim commercial-use rights.
- Proposed bounded training comparison: warm-start the unchanged current model;
  compare an encoder-tail/CTC update with a lower-rate full encoder/CTC update.
  Select the checkpoint on RoadText validation only, then run all regressions.

### StreetOCRCorrect / OCR-On-the-go

- Public framework: https://github.com/rohitsaluja22/StreetOCRCorrect
- Related recognition project: https://github.com/rohitsaluja22/OCR-On-the-go
- Author's project page: https://rohitsaluja22.github.io/project.html
- The current author page says the OCR-On-the-go dataset can be requested at
  **rohitsaluja22@gmail.com**. Confirm that the supplied material includes the
  human-corrected license-plate corpus described in StreetOCRCorrect, rather
  than only street-sign data or automatic OCR suggestions.
- Required contents: source frames or native crops, corrected plate text,
  vehicle/track identities, timestamps if available, original splits, and terms
  permitting local fine-tuning and model-weight sharing for this project.
- Public source-video links and model suggestions are not substitutes for
  human-corrected training labels. Paper illustrations are not a training set.
- No email has been sent and no access agreement has been signed.

### UFPR-SR-Plates

- Publisher: https://github.com/valfride/UFPR-SR-Plates
- Access agreement:
  https://github.com/valfride/UFPR-SR-Plates/blob/main/media/license_agreement.pdf
- The publisher requests a completed agreement sent from a valid university
  email to **menotti@inf.ufpr.br** for noncommercial academic access.
- Stated response window is generally 1–5 business days, not guaranteed.
- Brazilian/Mercosur registrations; 10,000 tracks, each with five low-resolution
  and five higher-resolution samples. All frames of a plate stay in one split.

### RodoSol-ALPR

- Publisher: https://github.com/raysonlaroca/rodosol-alpr-dataset
- Access agreement:
  https://github.com/raysonlaroca/rodosol-alpr-dataset/blob/main/pdfs/license-agreement.pdf
- The publisher requests a completed agreement sent from a valid university
  email to **rblsantos@inf.ufpr.br** for noncommercial academic access.
- Stated response window is generally 1–5 business days, not guaranteed.
- Brazilian/Mercosur registrations from fixed toll cameras. Preserve the supplied
  training/validation/test partitions and audit duplicates against our data.

## Common measurement rules

- Freeze a new human-labelled Indian CCTV evaluation set before fitting. The
  existing 18 Mumbai crops and 200 separate-source samples remain useful
  regression checks, but have been examined repeatedly and are not fresh tests.
- Group all frames of the same vehicle/plate in one split. Compare unique
  vehicles as well as images; many near-identical frames are not many vehicles.
- Select hyperparameters and checkpoints using validation only. Do not repeatedly
  use the final test set to choose weights or select the winner. Rank on a common
  Indian validation set, then confirm the chosen model on the untouched test set.
- Primary metric: complete exact plate readings. Also count missing, extra and
  substituted characters, wrongly accepted strings, coverage and inference cost.
- Keep separate crop-only OCR, tracked multi-frame reading and end-to-end video
  metrics so missed detections are not silently treated as OCR mistakes.
- Train Brazilian labels as written. Do not force them into Indian registration
  patterns. Country-specific runtime formatting must not distort dataset scoring.
- Use comparable seeds, augmentation policy and training budgets. Bound frames
  per track and balance sources in the combined experiment.
- Preserve checkpoints, hashes, dataset manifests and measured outcomes for each
  run. The current checkpoint stays active until a replacement is validated.
- Public redistribution of source images or trained checkpoints depends on the
  actual dataset terms; acquiring data does not automatically grant that right.

## Immediate blocker and useful next action

RoadText-3K replaces the StreetOCRCorrect access dependency. UFPR-SR-Plates and
RodoSol still require their publisher access processes before those later stages.
This stage must finish with a RoadText benchmark/regression report and a user
review gate. No automatic deployment or subsequent-dataset run is authorised by
completion of the first fine-tuning experiment alone.

These experiments address an important remaining gap. They do not demonstrate
that all possible OCR improvements have been exhausted, and no target accuracy
is promised before the new data and held-out measurements exist.
