# OCR improvement stages: measured progress and stopping point

Checked 27 September 2026 (India local date). This is a bounded extension to
the existing benchmark, not a new production reader or a completed live test.
The approved five-step direction was used; no architecture or privacy scope
was expanded. The running live-worker default remains unchanged.

## Stages 1–3: implementation and comparisons

1. **Image-quality selection and weighting.** Native, unresized crops are scored
   using focus, contrast, exposure and pixel dimensions, without OCR outputs or
   evaluation labels. A bounded pool of at most 40 observations selects up to
   eight distinct, temporally separated frames, then restores timestamp order.
   CTC sequence likelihood and character evidence use relative quality weights.
   Existing thresholds, three-original-observation requirement and two-frame
   character-support gate remain. Equal weights are still the default. The local
   passage contains only three frames: larger-window selection gains were not
   measured. The score is a heuristic; noise can imitate sharpness.
2. **Reconstruction and learned deblurring with Awiros.** Checked ECC alignment
   and weighted multi-frame reconstruction reuse the existing implementation.
   The cached NAFNet checkpoint supplies an optional deblurred working view.
   Unsupported alignment yields no reconstruction. OCR reads each processed
   image separately; its candidates are diagnostic and cannot create additional
   original observations. A strong conflicting processed reading can veto an
   accepted original reading. It cannot independently create an alert.
3. **Automatic plate crops and optional angle correction.** The cached YOLOv9-t
   plate detector selects the highest-confidence region without the evaluation
   box, adds 3% padding and copies native pixels. Ground-truth boxes are used
   only afterwards for overlap scoring. A conservative edge-agreement deskew
   produces an expanded, rotated working canvas; it preserves the original
   and abstains when edge support is weak. This is affine deskew, not a learned
   corner detector or full perspective correction. Corrected-view OCR is
   diagnostic. Vehicle detection/tracking and multi-vehicle association were
   **not** evaluated in this crop comparison.

All comparisons used the same permitted NIVU passage, three distinct source
images, assigned replay timestamps and 21 deterministic conditions. These are
**one vehicle and one plate**, not 21 independent vehicles. The manifest SHA-256
was `91d3b4fa479d88ee03c3bdc8aedeb3b4a2dd64e88975906f7cc11ee1ae5720c7`.
The model and dataset licence/pins are in [OCR_MODEL_SELECTION.md](OCR_MODEL_SELECTION.md).

| Variant | Correct accepted | Wrong accepted | Abstained |
| --- | ---: | ---: | ---: |
| Equal-weight original-crop baseline, rerun | 11 | 0 | 10 |
| Quality-weighted original crops | 12 | 0 | 9 |
| Quality weights + ECC + NAFNet + optional deskew | 12 | 0 | 9 |
| Automatic crops + quality weights + optional deskew | 13 | 0 | 8 |

The first gain is `darkness_0.5`: the correct candidate was already ranked first.
Weighting moved the weakest-character evidence from 0.7464 to 0.7566, just above
the unchanged 0.75 gate. This is a borderline evidence-weighting change, not
newly learned character recognition or a calibrated accuracy gain.

The automatic-crop gain is `noise_30`. Its median crop width changed from 109 to
113 native pixels. The three detected boxes had IoUs of 0.8992, 0.7645 and 0.9257
against the approximate annotations. Across all conditions, the detector found
a region in 63/63 frames; 62/63 overlapped the target at IoU >= 0.5. These reused
frames do not establish real-camera detector recall or correct association.

ECC produced reconstructions in 18/21 conditions. Diagnostic greedy OCR matched
the label on 16/18 reconstructed views and 15/21 deblurred views. Optional deskew
ran in 12 cases and its greedy read matched in 10. These are correlated working
views, **not additional accepted plates or independent recognition trials**.
At `resolution_0.375`, reconstruction read the correct text but NAFNet did not;
deblurring is therefore not assumed to help. Neither processing stage increased
final accepted readings beyond quality weighting. Severe motion blur and the
smallest crops still fail. No thresholds were lowered and no registry was used.

CPU inference medians were roughly 2.25–2.63 seconds per original crop. Network
warm-up is excluded; two-CPU/4-GiB containers ran concurrently on one laptop.
Timings are diagnostic, not controlled comparative throughput or live latency.
Optional processing adds work; GPU acceleration is not yet verified.

## Local results and reproduction

Ignored reports/viewers under `data/benchmarks/reports`:

- `awiros-baseline-check-all.json` / `.html`
- `awiros-quality-all.json` / `.html`
- `awiros-reconstruction-all.json` / `.html`
- `awiros-automatic-all.json` / `.html`

Use the existing isolated Awiros container setup documented in
[OCR_MODEL_SELECTION.md](OCR_MODEL_SELECTION.md), with the repo mounted at
`/workspace`, working directory `/workspace/edge`, model cache `/models`, and
only the report directory writable. Do not pass the live `.env`. Automatic
localisation also requires the cached FastALPR volume at `/root/.cache`.
No new download was required for these comparisons.

Run the existing benchmark module with the same manifest and `--family all`:

```text
python -m sentinel_edge.ocr_selection_benchmark --manifest /workspace/data/benchmarks/nivu/manifest.json --cache /models --report /workspace/data/benchmarks/reports/result.json --viewer /workspace/data/benchmarks/reports/result.html
```

Variant switches:

- Baseline: no additional switches.
- Quality weighting/selection: `--quality-weighted`.
- Reconstruction comparison: `--quality-weighted --reconstruction --restoration --deskew`.
- Automatic-crop comparison: `--quality-weighted --automatic-crops --deskew`.

Aggregate reports contain no plate strings or media. The local licensed-data
viewer displays originals, processed images and candidates. All four generated
pages have 21 sections, no script elements or external image requests. Genuine
Sentinel footage is excluded. **154 edge tests passed**, including native pixel
preservation, quality bounds, duplicate/timestamp separation, equal-weight
decoder parity, no-detection abstention, processed-view non-voting, privacy
rejection and escaped/bounded diagnostic output.

## Stage 4 blocker: no diverse training files

The proposed [Indian OCR corpus](https://huggingface.co/datasets/thundarstrom/indian-anpr-ocr-corpus)
advertises 18,537 crops, but its live file API at revision
`f25fb3ddd41c2e436ccbb5fe81a3c76be9f4ab4c` returned only `.gitattributes` and
`README.md`. The dataset page also states that it is empty. There are no image,
label or split files to download from that repository. A lookup of the same
maintainer name on the public GitHub user API returned HTTP 404.

The only locally verified labelled temporal input remains the three images of
one already-seen NIVU vehicle. Other probe images have no verified permission
and transcription manifest. Training and validating on augmented copies of
the same vehicle would not fulfil the approved diverse, vehicle-separated
evaluation. **No additional fine-tuning or new trained checkpoint was produced.**

To unblock this step, supply an actual downloadable or local permitted Indian
plate corpus with checked transcriptions, original source/permission notices
and enough distinct registrations/passages for separate train, validation and
untouched test groups. Include two-row, small, blurred and low-light examples.
The current pilot stays a development regression and cannot become held-out
evidence. Missing files, not model tuning complexity, caused this stop.

Per the user's requested stopping rule, work stopped at this blocking input
error. **Stage 5 GPU/export parity and subsequent live CAM01/CAM02 shadow tests
were not run in this stage.** Live authentication was not rechecked here; older
access results must not be presented as current. The improved experimental
reader has not been promoted to the live worker. Resume with valid training
inputs, then verify GPU model-output parity and privacy-preserving live shadow
processing before changing the production default.

### Subsequent dataset acquisition

The follow-up search found a real CC BY 4.0 Zenodo archive with 1,700 images and
a transcription workbook. The workbook and ten images were acquired; eight
image/label pairs were visually confirmed and two quarantined. This resolves
the lack of any additional verified images, but not full-corpus cleaning,
vehicle-separated training or held-out validation. See
[dataset audit and remaining gates](INDIAN_DATASET_AUDIT.md). The historical
empty-corpus finding above remains correct for that specific Hugging Face repo.

### Subsequent fine-tuning pilot

The approved follow-up expanded visual review to 48 distinct registrations and
trained the Awiros CTC linear output layer. Separate test results were 9/10 exact
before and after, with character errors falling from two to one. Both matched
multi-frame runs accepted 13/21 conditions correctly, abstained on eight, and
accepted none incorrectly. This is not a demonstrated improvement in the blurry
plate acceptance boundary. See [training, evidence and reproduction](OCR_FINETUNING.md).
The historical blocker above describes the earlier stage, not the current pilot.
Production remains unchanged; GPU/live validation and larger diverse training
remain unfinished.
