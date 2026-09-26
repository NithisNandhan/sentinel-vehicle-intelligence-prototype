# Indian plate dataset acquisition and label audit

Checked 27 September 2026, India local date. Dataset search succeeded: real,
openly licensed Indian plate images and an independently checked sample of
transcriptions are now available locally. No new training or live promotion
was performed during acquisition.

## Source and permission

[Number Plate Number Identification Dataset, Zenodo record 13954136](https://zenodo.org/records/13954136),
DOI `10.5281/zenodo.13954136`, published 19 October 2024 by Bapatla Engineering
College contributors Bhuma Chandra Mohan, CH V M S N Pavan Kumar, Katakam,
Bethala Venkata Varma, Vatada Sai and Kagitha Sambaiah.

The [record API](https://zenodo.org/api/records/13954136) declares open access and
**CC BY 4.0**. The licence allows attributed reuse, including this research work.
The record describes 1,700 vehicle images and an Excel transcription table.
Its actual ZIP inventory contains 1,700 images and `number_plate.xlsx`.
This is an available dataset, unlike the empty Hugging Face corpus previously
proposed. The creators' licence declaration is preserved; this audit does not
independently establish the acquisition history of every photograph.

The archive is 1,468,393,969 bytes. Publisher checksum:
`md5:e0a3d1e121f52806abb65248c8b86149`.
Only precise HTTP range reads were used for this audit, transferring about
1.745 MB rather than downloading the complete 1.5 GB archive. ZIP CRC validation
passed for each retrieved member. **The complete archive MD5 was not verified**
because the complete archive has not been downloaded.

## What is locally acquired and verified

Under the ignored `data/benchmarks/bapatla` directory:

- Original, unmodified transcription workbook: `number_plate.xlsx`.
- Ten original images under `images`.
- Complete ZIP member inventory and checksums: `inventory.json`.
- Image/label audit counts: `audit_summary.json`.
- Eight visually confirmed image/transcription pairs, with image SHA-256,
  attribution and explicit review decisions: `verified_sample.json`.

Each of the ten downloaded images was viewed and compared with its publisher
transcription. Eight pairs were sufficiently clear and matched: AN1, AN2, AN5,
AN6, AN7, AN10, AP1 and AP10. AN4 was quarantined because its plate is partly
obscured and its label is incomplete. AP12 was quarantined because an O/D
character could not be independently confirmed. No OCR output was used to
manufacture verification labels. Original labels were not silently corrected.

These are still images of different vehicles, **not a multi-frame passage**.
The reviewed sample is labelled development/visual-review data, not an untouched
test set. Eight samples are useful regression inputs, not sufficient evidence
for general accuracy or a robust further-fine-tuned checkpoint.

## Corpus issues requiring cleaning before full training

The spreadsheet skill was used for read-only coverage/duplicate analysis. It
helped preserve the original workbook while separating publisher labels from
the independently verified subset.

- 1,700 populated image/label rows; no missing transcription cells.
- 1,358 distinct publisher label strings, with 342 repeated-label rows. Split
  by registration/vehicle before augmentation so repeats cannot cross splits.
- Only 595 filenames match archive basenames exactly. Removing spaces and
  parentheses from stems and comparing case-insensitively maps 1,696 rows
  uniquely. This is a documented mapping, not a workbook rewrite.
- Four rows remain unmapped: C270, C407, DL36 and WB55. Unmapped archive members
  are C (271), C (408), DL38 and WB5. Do not pair these by row order or guess
  that these are typographical corrections without checking the actual images.
- 1,446 strings match a common Indian registration format. **Format matching is
  not label verification**, and the pattern does not cover every Indian format.
  The table also contains foreign-looking, partial and non-registration strings.
  This source must not be treated as 1,700 clean, verified Indian transcriptions.
- Only the ten downloaded samples have been visually audited. Full-image exact
  and perceptual duplicate checks, the remaining transcriptions and uncertain
  labels have not been audited. Pretraining overlap with the existing checkpoint
  is unknown.

Recommended preparation: retain source notices; fetch additional or full source
files with checksum verification; map filenames explicitly; quarantine unmapped,
partial and ambiguous entries; check labels against images; deduplicate; group
by registration before train/validation/test splitting. Use native automatic
plate crops, exclude unsafe crop failures, and keep labels out of inference.
Only then run the already approved fine-tuning and untouched evaluation.

## GPU/live status

The GPU preflight still sees an NVIDIA RTX 3050 6 GB through the container, but
there is no cached GPU Paddle environment for the Awiros reader. This proves
device visibility only, not Awiros GPU inference, parity or throughput.

The user requested dataset acquisition first, with GPU/live testing as the
fallback if no suitable images could be found. The acquisition branch succeeded
with eight independently checked image/label pairs. Full training, Awiros GPU
model validation and subsequent CAM01/CAM02 shadow processing remain unfinished;
no live stream was opened and no credentials or production settings were changed
during this dataset audit. Existing services were left running.

## Follow-up: completed bounded adaptation

The initial acquisition findings above are historical. The next approved stage
expanded proposals to 53 automatically cropped image/label pairs, visually
confirmed 48 distinct registrations, quarantined five additional uncertain pairs,
and split 28 train / 10 validation / 10 test registrations before augmentation.
A CPU recognition-head adaptation and matched benchmark reruns are complete.
See [OCR_FINETUNING.md](OCR_FINETUNING.md) for actual metrics, hashes and commands.
The complete 1,700-image archive still has not been downloaded or cleaned; GPU
model validation, live shadow tests and production promotion were not performed.
