# Teammate handoff: source, models, saved state and next work

## What to download

Repository: <https://github.com/NithisNandhan/sentinel-vehicle-intelligence-prototype>

Release: <https://github.com/NithisNandhan/sentinel-vehicle-intelligence-prototype/releases/tag/handoff-2026-10-07>

The repository is private. The owner must grant the teammate repository access
in GitHub Settings > Collaborators. Downloading only the repository ZIP is not
sufficient: Git ignores models and local data deliberately. They are in the
release, not missing or silently regenerated. Docker images are rebuilt from
the checked-in Dockerfiles; images themselves are not included.

| Location | Contents |
| --- | --- |
| Repository | Current backend, edge pipeline, both dashboards, tests, Compose, tools, docs and small submission artifacts |
| `model-cache` release archive | Vehicle/plate detector variants, adapted Awiros OCR checkpoints, PARSeq checkpoint and isolated OCR dependencies |
| `alpr-cache` release archive | FastALPR plate detector and CCT global OCR weights |
| `detector-training` release archive | All detector training runs, best/last weights, datasets, labels and training results |
| `benchmark-data` release archive | Host OCR datasets, adapted checkpoint copy, receipts, licensed crops and HTML/JSON reports |
| `saved-events` release archive | Consistent PostgreSQL metadata snapshot, SQLite queue backups and engineering session JSON |
| `presentation-assets` release archive | Revised Word HLD, architecture, PPTX, captions, demo poster and both demo videos |

Download every part for each archive you need and its manifest into one folder.
Parts are 256 MiB except the last part. Do not extract a part individually;
the restore tool reads them together. Each manifest records part sizes/hashes
and fine-tuned checkpoint hashes. Private release downloads require GitHub login.
With GitHub CLI installed and signed in:

```powershell
git clone https://github.com/NithisNandhan/sentinel-vehicle-intelligence-prototype.git
cd sentinel-vehicle-intelligence-prototype
gh release download handoff-2026-10-07 --repo NithisNandhan/sentinel-vehicle-intelligence-prototype --dir data/handoff-assets
```

Allow substantial free disk space: approximately 19 GB of uncompressed assets,
plus downloads, Docker images and test scratch space. 60 GB free is a practical
starting allowance, not a measured hardware requirement. The GPU engineering
path needs NVIDIA-compatible Docker/WSL2; offline central services do not.

## Fresh-install restore on Windows

Start Docker Desktop. Use a fresh checkout/installation, not the original
laptop's populated volumes. Never delete existing volumes just to make restore
pass: the tools intentionally refuse to replace existing records/files.

```powershell
.\scripts\check-prereqs.ps1
.\scripts\init-env.ps1
.\scripts\restore-teammate-handoff.ps1 -AssetsPath .\data\handoff-assets
```

The restore script builds central/offline service images, verifies and extracts
archives, recreates the original model volume names, migrates an empty central
database, imports events/audits/health, restores the three service outboxes, and
starts only the API, dashboard and local media gateway. It does not start live
Sentinel cameras. It refuses central import while the API is running, and SQL
import is transactional and refuses a populated database. If a restore fails,
inspect the failed destination rather than deleting unrelated data.

`-SkipTraining`, `-SkipBenchmarks`, `-SkipModels` and `-SkipCentral` allow selective
restoration; skipped archives need not be downloaded. `saved-events` and
`presentation-assets` are always extracted. With `-SkipCentral`, metadata is
available as files but is not imported. Standalone extraction is also available:

```powershell
python .\tools\handoff_archives.py restore --manifest .\data\handoff-assets\benchmark-data.manifest.json --destination .\data\restored-benchmarks
```

Use Python 3.11 with current security updates or newer for the archive extraction
filter. Linux model/runtime archives should be restored through Docker, not
directly into Windows: native libraries and symlink semantics differ.

Supplemental files land under `data/handoff-restored`. Benchmark reports are
under `data/handoff-restored/benchmark-data/benchmarks/reports`; saved events are
under `data/handoff-restored/saved-events`. Source tools that default to
`data/benchmarks` can instead be passed the restored paths. Serve the reports:

```powershell
python -m http.server 7777 --bind 127.0.0.1 --directory data/handoff-restored/benchmark-data/benchmarks/reports
```

Open <http://127.0.0.1:7777/indian-corpus-summary.html>. The saved metadata snapshot
contains 6 cameras, 16 sightings, 15 alerts, 22 audit entries, 16,625 health
reports and 1 watchlist entry. The exported SQLite queues were empty. Engineering
session JSON is historical metadata, not a way to replay original video.

The operator dashboard is <http://127.0.0.1:5173/>. Unlock using the **new**
`OPERATOR_API_KEY` in your local `.env`. To run the synthetic demonstration:

```powershell
docker compose --profile offline up -d demo-simulator
```

Historical HMAC identifiers are preserved, but the original HMAC secret is not
exported. New events use your freshly generated secret. Consequently a plaintext
plate search with the new key cannot correlate old saved identifiers; archived
alerts and audit history remain viewable. Synthetic startup seed data may add
new-key demo records. Camera preview URLs were removed from exported events.

## Which model paths to use

Inside the restored model volume, the shared engineering OCR checkpoint is:

`/models/experiments/awiros/adaptations/indian-corpus-encoder.safetensors`

SHA256: `c2ba896d994b174071c95ace0f5ffc8a61cec3fc6eecce8f0bc5feb7dec4456d`.

The archive also preserves parent OCR weights and Bapatla adaptation variants,
not just the selected checkpoint. Detector training weights are:

- `/training/runs/vehicle-bmd/weights/best.pt`: adapted YOLO26s vehicle model.
- `/training/runs/vehicle/weights/best.pt`: earlier vehicle adaptation.
- `/training/runs/plate-r3/weights/best.pt`: adapted plate detector, with last/stripped variants retained.
- `/models/experiments/detectors/yolov9t512-indian-adapted.onnx`: adapted plate ONNX used by experimental profiles.

See `docs/DETECTOR_COMPARISON.md`, `docs/DETECTOR_DATASETS.md`,
`docs/DEEP_OCR_ADAPTATION.md`, `docs/LIVE_DETECTION_VIEWER.md` and
`docs/DUAL_INTERFACES.md`. Model/data licenses remain separate from the application
license. Preserve source receipts, dataset licenses and attribution; private
storage is not permission to redistribute otherwise restricted data. The
unverified `dats_probe` dataset is not included.

The independent production edge worker still uses YOLOv8n/FastALPR. The opt-in
engineering path supports YOLO26 and adapted profiles. Do not confuse their
defaults or promote a diagnostic checkpoint to production without testing.
Build the engineering image when needed:

```powershell
docker compose --profile online build edge-worker
```

Building this image does not start a live camera connection. The existing
`start-live-detection-viewer.ps1` and `start-dual-interfaces.ps1` launchers are
still **live-camera launchers**, not downloaded-video launchers.

## Next stage: downloaded CCTV footage

The owner has changed the intended input to permitted, high-quality downloaded
videos. No Sentinel credentials are needed for that direction. This handoff
does not claim that integrated recorded-video ingestion already works.

1. Select footage with permission to download/use, visible Indian plates and
   original resolution. Keep source/license receipts. Do not merely upscale
   unreadable pixels and call them ground truth.
2. Add a local-video PyAV source adapter to the shared recognition session.
   Retain video PTS, reset tracking at EOF/loop cuts, support start/pause/stop,
   and distinguish paced demonstration playback from unpaced benchmarks.
3. Feed decoded native-resolution frames into the existing vehicle tracking,
   plate cropping, adapted OCR, ECC alignment and multi-frame evidence ranking.
   Wire the same session to the engineering dashboard and accepted-event path.
4. Explicitly label recorded-video inputs throughout API/UI/reporting. Current
   source contracts distinguish `live` and `synthetic`; extend them with a
   recorded source instead of labelling downloaded footage as genuine live feeds.
5. Keep best-effort OCR candidates in the engineering console; accepted plate
   identifiers must still pass the central gate. Do not silently force uncertain
   candidates into operational alerts.
6. Manually transcribe a held-out set separated by vehicle/video from training.
   Report detection/crop coverage, correct/wrong accepted full plates, abstentions,
   candidate recall, and processing latency. Never equate OCR scores with measured
   accuracy or guarantee every vehicle will be identified.
7. Verify local video processing, persistent events, watchlist matching, operator
   actions and restart recovery together; then record a labelled demonstration.

No accepted genuine live plate identifier was established by earlier testing.
The adapted OCR improved a small licensed diagnostic (28/31 exact reads), not
validated live camera accuracy. Reconstruction is geometrical evidence fusion,
not a trained neural temporal model or a way to recover absent image information.
The aim of better footage is measurable end-to-end recognition, not a hidden
change to the acceptance threshold.

## Verification and publication tools

Before this handoff: edge regression suite 287 passed, 1 optional test skipped;
central API 53 passed; frontend 15 passed and production build succeeded;
handoff tool tests 13 passed. The handoff tests cover multipart restoration,
byte-identical resumption, corruption, non-overwrite, safe
extraction and SQLite snapshot redaction. See release metadata for asset hashes.

The publishing tools never print GitHub credentials, publish only to the existing
private repository, verify upload hashes, and do not force-push. `.env`, assistant
state, Docker registry credentials and original authenticated URLs are excluded.
Files/media from genuine Sentinel feeds were not retained and cannot be supplied.

### Interrupted release upload recovery (owner only)

`tools/finish_handoff_upload.py` resumes the existing private draft, skips
checksum-verified assets, and retries transport failures up to five times.
It removes only abandoned, digest-free `starter` uploads for eligible local assets in that
unpublished draft; completed assets are never overwritten. It stops on integrity
errors. All six bundle manifests must exist before it can publish.

The owner runs it after starting the task-owned benchmark exporter, supplying
the two staging directories, pushed commit SHA, existing release ID and an
ignored `data/` status-file path. It is not needed by the receiving teammate.
Run only one uploader at a time. Inspect its status JSON and stdout/stderr logs;
`published` means publication finished, not merely that the process started.
The laptop must remain awake and connected, and Docker must stay running until
the benchmark export completes. This helper does not change power settings,
stop the application's services or include credentials in the handoff.
