# Complete teammate handoff — 7 October 2026

This release accompanies the latest source on `main`. Read
`docs/TEAMMATE_HANDOFF.md` in the repository before restoring.

The attached multipart archives contain:

- `model-cache`: cached vehicle/plate detectors, adapted OCR checkpoints and the isolated OCR runtime.
- `alpr-cache`: FastALPR/Open Image Models/CTT OCR weights.
- `detector-training`: detector training runs, all best/last checkpoints, labels and permitted training data.
- `benchmark-data`: licensed OCR datasets, benchmark reports, receipts and host-side training outputs.
- `saved-events`: central metadata and operator audit snapshot, SQLite outbox backups, and engineering session metadata.
- `presentation-assets`: revised HLD, architecture, PowerPoint, caption files and demonstration videos.

Every archive requires its `.manifest.json` and **all matching `.partNNN`
files**. The included restore tool checks SHA256 hashes before extraction and
refuses to overwrite a nonempty destination. Large files are release assets,
not Git LFS pointers. Model hashes are also recorded in the manifests.

Central snapshot: 6 cameras, 16 sightings, 15 alerts, 22 operator audit entries,
16,625 health reports and 1 synthetic watchlist entry. All exported delivery
queues were empty at snapshot time. These are historical records, not a claim
of successful live OCR.

Credentials, original HMAC secrets, active camera URLs, local assistant state
and an unverified-permission dataset probe are excluded. Preserve dataset and
model licensing/attribution requirements. The fine-tuned OCR is diagnostic;
small benchmark improvements do not establish live accuracy.

The next stage is adapting the pipeline to permitted **downloaded CCTV videos**.
The synthetic offline demo already runs without Sentinel access. Recorded-video
input for the integrated dashboards is not implemented by this handoff.
