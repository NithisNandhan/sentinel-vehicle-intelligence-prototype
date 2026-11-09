# RoadText OCR adaptation — 9 November 2026

This is the first dataset stage approved in the
[experiment plan](plans/2026-11-09-cctv-ocr-dataset-experiments.md).
RoadText-3K replaces StreetOCRCorrect. The next dataset must wait for the user's
review of this stage's benchmarks. No experimental checkpoint is automatically
activated in the dashboard.

## Source and scope

The [publisher's directory](https://datasets.cvc.uab.es/roadtext3k/) supplies
`I_400.tar.gz` and `roadtext-annotation-fixed.json`. The
[author's paper](https://marcalr.github.io/pdfs/DAS22.pdf) describes the benchmark
and its human annotation procedure. It is moving-camera road imagery, not fixed
CCTV. This experiment uses one bounded Indian archive, not the whole dataset.

Only explicit plate-category labels with nonempty, nonconflicting, complete
registration-shaped text enter this experiment. Separators are removed and text
is uppercased; missing characters and ambiguous substitutions are never supplied
by our model. Fragmentary labels and unsupported formats are excluded. These
are publisher transcriptions, not a claim that every label was independently
re-annotated by this project.

Videos are ordered by source ID and assigned to custom 50:20:30
training/validation/test blocks. Every registration appearing across blocks is
excluded from all blocks. Remaining frames of a video and registration stay in
one partition. These are **not the official split IDs**, and ordered video IDs
do not establish independent recording sessions.
Existing benchmark registrations are excluded before crop selection. Unknown
membership of the publisher's original pretraining set remains a limitation.

The downloader records HTTP source metadata, local hashes and annotation byte
ranges. The public publisher's HTTP endpoint was used after local HTTPS
certificate verification failed; TLS verification was not disabled. A local
hash establishes repeatability, not publisher-signed authenticity.

Raw videos, source annotations, crops, detailed plate-level reports and new
checkpoints stay in ignored `data/`. A separate RoadText-3K redistribution or
commercial-use licence was not established. Only code and aggregate evidence
are included in the source push.

## Frozen cohort

The archive contains 100 Indian videos, IDs 2401–2500. Actual decoded frames are
1920×1080. Native admitted crops have median dimensions 102×29 pixels. Of 1,367
retained crops, 818 from 14 cross-block registrations are quarantined, leaving:

| Partition | Crops | Annotated tracks | Distinct registrations | Videos |
|---|---:|---:|---:|---:|
| Training | 352 | 51 | 27 | 33 |
| Validation | 91 | 19 | 17 | 12 |
| Test | 106 | 19 | 13 | 11 |

These counts differ: a registration may occur in several tracks/videos within
its partition. Do not describe 549 crops as 549 independent vehicles.

Before fitting, an attempted connected-component split was rejected: repeated
registrations linked 1,201 crops into one component, leaving only 11 training
registrations and two validation registrations. Source-block quarantine was
chosen before any model scores were inspected. The importer also explicitly
counts terminal annotation entries beyond the available video; four requested
crop observations were unavailable and never used. The publisher's bundled
`400.json` is skipped in favour of the separately pinned fixed annotations.

The final frozen manifest is `cohort-v4/manifest.json`, SHA256
`fc7d7de79d0698acff3925f200034c68562bba8aaf8c77280a694013d598c897`.
Failed preparation folders are not training cohorts. Verified source archive
SHA256 is `9523533f423aa49210fbe95869b1a3c79ba97706b21359056a19c3d3b7a3c7ca`.
The 2.17 GB temporary download fragments were removed after final checksum
verification; the verified archive remains available locally.

## Reproduction

The installed edge-worker image and cached isolated Paddle runtime are required.
Run the acquisition on the host:

```powershell
python tools/acquire_roadtext3k.py --output data/benchmarks/roadtext3k-20261109 --workers 8
```

The native-crop importer runs inside the existing container environment with
these mounts: `edge` and `tools` read-only under `/workspace`, `data` writable
under `/workspace/data`, and `sentinel-prototype_model_cache` read-only under
`/models`. Use image `sentinel-prototype-edge-worker:latest`, working directory
`/workspace`, and `--network none`. Its command is:

```text
python /workspace/tools/run_ocr_research.py /workspace/tools/prepare_roadtext3k.py
  --archive /workspace/data/benchmarks/roadtext3k-20261109/I_400.tar.gz
  --annotations /workspace/data/benchmarks/roadtext3k-20261109/annotations-2390-2509.json
  --output /workspace/data/benchmarks/roadtext3k-20261109/cohort-v4
  --exclude /workspace/data/benchmarks/indian-corpus/adaptation_manifest.json
  --exclude /workspace/data/benchmarks/fresh-indian-20261101/academic/adaptation_manifest.json
  --exclude /workspace/data/benchmarks/recorded-footage-20261104/manual-review/ocr-development-manifest.json
  --exclude /workspace/data/benchmarks/nivu/manifest.json
```

Line breaks above are for readability. Fresh output directories are required;
do not overwrite an existing frozen crop set or training run.

The importer decodes each source once, uses annotation-aligned native pixels
without resizing, and retains at most eight distinct crops per track. Image
quality selects crops without consulting predictions or matching model outputs
to the expected answer. Hashes bind source media, labels and crop pixels.

Temporarily stop `recorded-analysis` before training to free the GPU. Use the
same mounts, add `--gpus all --ipc=host`, and run:

```text
python /workspace/tools/run_ocr_research.py /workspace/tools/finetune_roadtext3k.py
  --manifest /workspace/data/benchmarks/roadtext3k-20261109/cohort-v4/manifest.json
  --output /workspace/data/benchmarks/reports/roadtext-finetune-20261109
  --epochs 4
```

The run warm-starts the preserved `c2ba896d…` checkpoint. It compares encoder-tail
updates at learning rate `3e-5` with full image-encoder/CTC updates at `5e-6`,
four epochs each, batch size four, fixed seed 20261109 and bounded exposure
augmentation. Auxiliary recognition heads and running statistics stay frozen.
Training and serving preprocessing/forward outputs must agree.

Checkpoint selection uses RoadText validation exact matches, then character
edits. The selected checkpoint is saved **before** test and legacy regressions
are evaluated. Safetensors reload parity must pass. Restart the recorded worker
with its unchanged production checkpoint after the experiment.

## Reading the measurements

Raw greedy exact reads isolate changes to OCR weights. Report character error
counts alongside them. Adjacent crops are correlated observations, so frame
counts must not be described as independent vehicles.

The RoadText temporal check replays retained annotated crops with their source
timestamps through the existing quality-weighted CTC fusion and acceptance
gates. It counts top-ranked exact tracks and final-window accepted tracks, plus
tracks that produced any correct or incorrect acceptance at an earlier retained
observation. Earlier wrong acceptances are preserved even if later corrected.
It is **not** an end-to-end detector/tracker video benchmark
and does not test missed plates.

Mumbai, old corpus and external-source checks are regression diagnostics. They
have already been examined and are not a fresh blind CCTV accuracy estimate.
Model score is not a calibrated probability of a correct plate. A replacement
requires validation improvement without regressions and an explicit subsequent
deployment decision.

## Measured results

**Decision: reject the RoadText candidate for production.** The small source-domain gain did not generalise. The baseline remains active; no subsequent dataset has started.

Raw greedy OCR results (exact complete strings; no model-score threshold used to decide correctness):

| Evaluation | Baseline exact | Candidate exact | Baseline character errors | Candidate character errors |
|---|---:|---:|---:|---:|
| RoadText validation | 39/91 | 41/91 | 113 | 106 |
| RoadText held-out test | 43/106 | 46/106 | 206 | 195 |
| Old corpus validation | 29/37 | 25/37 | 42 | 52 |
| Old corpus test | 28/31 | 25/31 | 13 | 16 |
| External regression | 137/200 | 132/200 | 135 | 141 |
| Mumbai manual crops | 14/18 | 10/18 | 13 | 17 |

The selected full-encoder candidate is epoch 4 at learning rate `5e-6`. Both configurations together ran eight epochs and 704 optimizer steps on the NVIDIA RTX 3050 Laptop GPU. Summed epoch compute/validation time was 62.97 seconds; this excludes data acquisition, preparation, model startup and final replay/regressions.

Timestamped multi-frame replay used 19 annotated tracks representing 13 registrations:

| Track result | Baseline | Candidate |
|---|---:|---:|
| Final top-ranked exact tracks | 9 | 10 |
| Final correctly accepted tracks | 4 | 4 |
| Final wrongly accepted tracks | 2 | 3 |
| Tracks with any correct acceptance | 4 | 5 |
| Tracks with any wrong acceptance | 3 | 4 |
| Tracks whose first acceptance was wrong | 3 | 4 |

OCR-only median on held-out RoadText crops: **15.89 ms baseline**, **15.01 ms candidate**. These timings exclude the much slower CPU candidate-fusion replay and do not establish end-to-end throughput.

The baseline also produced wrong acceptances in this replay. This finding argues for an independent acceptance/calibration audit; do not describe either model as 90%-reliable CCTV recognition.

Saved-checkpoint strict reload and serving parity passed. The 57 focused data-admission/OCR/fusion tests passed. Aggregate evidence is in [the machine-readable report](evidence/roadtext-finetune-20261109.json). Detailed per-plate reports, crops and the experimental checkpoint remain local under `data/benchmarks/`; source redistribution rights were not established.

Candidate SHA256: `ea35604d59a13ea83397cfc683d81293382dbead47fa045b8635b37e945bbe24`.

After restoration, the four-model GPU preflight passed for OCR, vehicle,
plate and text-outline inference with the unchanged `c2ba896d…` baseline.

The next stage waits for user review. More independent Indian registrations and balanced rehearsal of the old training domain are reasonable follow-up experiments; this run used only 27 training registrations and does not show that all of RoadText or all OCR improvements have been exhausted. Do not select a new configuration using these already-opened test results.
