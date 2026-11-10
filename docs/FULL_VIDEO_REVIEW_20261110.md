# Full Mumbai video review — 10 November 2026

The complete permitted `footage.mp4` recording has been processed with the
unchanged deployed OCR and general detector profile. Human labels are pending;
there is no measured plate-recognition accuracy yet.

Open the local review page at **http://127.0.0.1:7795/**. The server runs on this
computer and stores labels on disk. It does not require Docker for labelling
after the frozen run has been created. Python 3.11 or newer is needed; install
Pillow (`python -m pip install Pillow`) to add native missed-vehicle images.

## What the run actually completed

| Item | Measured result |
|---|---:|
| Original video duration | 30.933 seconds |
| Original resolution | 3840 × 2160 |
| Decoded source frames | 928 |
| Analysed frames at the existing sampling cadence | 62 |
| Decoder reached end-of-file | Yes |
| Tracker entries, including possible fragments/duplicates | 51 |
| Entries with at least one automatic plate crop | 34 |
| Entries with a nonempty model hypothesis | 31 |
| Entries with a model acceptance (not human verified) | 3 |
| Native plate views retained for human review | 125 |
| Wall time including evidence collection/export | 211.78 seconds |

These are processing counts, not accuracy. A tracker entry is not necessarily a
distinct vehicle. The observer kept every encountered track; no dashboard-list
cap was applied. Up to six image-quality-selected native plate views per entry
are retained for review; this is a display/evidence limit, not a cap on the
normal pipeline's inference. Model confidence did not choose the human truth.

Video sampling and acceptance rules were not relaxed to improve these counts.
This offline run does not demonstrate real-time throughput. The recording has
already been used in development, so this is a full-clip evaluation, not a fresh
blind test or a general CCTV accuracy claim.

## How to label it

1. Enter the reviewer's name. OCR predictions stay hidden while you label.
2. For each entry, inspect its native crops and use **View in original video**
   to check the full 4K recording. Enlargement changes display size only.
3. Choose **Readable** and type the complete plate, or mark it **Unreadable**,
   **No visible plate**, or **Not a vehicle** as appropriate. Do not guess an
   uncertain character.
4. If two entries represent the same physical vehicle passage, mark the later
   entry as a duplicate and select its main card. Duplicate cycles are rejected.
5. Watch the complete recording. Use **Add missed vehicle**, pause at a useful
   frame and select its rectangle. Numeric coordinates are available as an
   alternative to dragging. Enter its truth like any other card. It remains a
   pipeline miss; manually cropping it does not create a new model prediction.
6. Check the complete-inventory declaration only after reviewing the whole clip
   and adding missing vehicles. Otherwise detection coverage stays unmeasured
   and the report explicitly describes a reviewed-track diagnostic.
7. **Finish review and calculate report** freezes the labels and reveals the
   comparison. Download the report JSON or print it for the final submission.

Drafts save automatically to disk with revision conflict checks. Incomplete text
also stays in the browser backup for this exact run. **Download draft** is always
available; completing all cards is required only for finalisation. After
finalisation the review is read-only, preserving the measured result.

## Metrics and denominators

- **Best-guess exact coverage:** readable vehicle passages whose final frozen
  hypothesis is exactly correct, divided by all labelled readable passages,
  including manually added misses.
- **OCR exact on cropped readable passages:** exact readings divided by readable
  passages that had an automatic plate crop. Missing OCR output still fails.
- **Plate-crop coverage:** readable passages with a plate crop divided by all
  readable passages.
- **Accepted exact coverage / accepted precision:** actual unchanged-gate
  acceptances are separate from speculative top guesses. An earlier wrong
  acceptance remains an error even if later corrected. Acceptances on human-
  unreadable plates are reported as unverifiable, not silently counted correct.
- **Top-5 / top-20 candidate recall:** whether truth occurs in the final ranking,
  without selecting a favourable earlier frame using the answer.
- **Vehicle-detection coverage:** computed only with a human-declared complete
  vehicle inventory. Duplicate fragments count as one human-labelled passage.

Undefined ratios have a null value and their denominator, rather than an
invented zero or perfect score. Human-unreadable and no-visible-plate cases are
reported separately from readable OCR failures. OCR scores are uncalibrated.

## Files, provenance and restart

The private local run folder is
`data/benchmarks/full-video-review-verified-20261110/`:

- `manifest.json`: full-run metadata, model identities, settings and review cards.
- `predictions.json`: frozen pipeline output, hidden by the labelling API.
- `assets/`: original native plate/vehicle crops and reviewer-added evidence.
- `labels.json`: current draft; `labels.previous.json` retains the previous save.
- After finalisation: `labels.final.json`, `accuracy-report.json` and the atomic
  `finalized.json` bundle.

The source SHA256 is
`6b7064605b3d4d12c6c1da4566cbcec1cd2e3e3bd7b77814865fe9df22045f65`.
The frozen review-manifest SHA256 is
`fba2dadd4438402a20a50553ca939f9f2b2626edc9bfd5bbfd1c1b16b8ff1beb`.
The recorded pipeline's base Git revision is `1b44f05`; the manifest separately
pins the observer script by SHA256 because it was new during this run.

Restart only the labelling service, without rerunning AI:

```powershell
.\scripts\start-video-review.ps1
```

This restores saved labels and uses the same frozen predictions. For a new
evaluation, `tools/run_video_review.py` requires a fresh output folder; it never
overwrites this one. It runs through the same `observe_tracked` path as the
dashboard and verifies the recording's existing permission receipt. The normal
recorded dashboard worker was restarted after this evaluation.

Raw footage, crops, detailed predictions and human labels stay local under
ignored `data/benchmarks/`. GitHub receives code and aggregate evidence only.
The source clip retains its own Pexels terms and is not republished as a raw
video in the repository.

## Validation boundary

All 18 controlled checks passed: 11 scoring tests, five local-server tests and
two observer integration tests. These cover denominator handling, missing predictions, duplicate
links, stale revisions, durable finalisation, read-only final reports, evidence
binding, hidden predictions, native missed-vehicle PNGs and video byte ranges.
Browser checks on a separate synthetic fixture verified video loading, draft
saving, partial-text restoration after reload, and adding a missed-vehicle card.
The browser finalisation test was blocked by automatic approval review and did
not execute. No synthetic labels were submitted to the real run, whose labels
remain for the human reviewer.

An initial observer run omitted the acceptance callback. That run is superseded
by the verified rerun above. The corrected observer connects to the acceptance
gate without a central delivery client and rejects completion if its distinct
accepted strings disagree with the pipeline summary. Synthetic integration tests
exercise both callback capture and rejection of an incomplete capture.

Implementation used the brainstorming and accessibility guidance and reused the
existing Graphify graph. Scoring and review-page work were delegated separately;
the primary integrator owns the run, server, integration and verification.
