# Sentinel Vehicle Intelligence Project Status and Completion Handoff

Status date 8 October 2026

## Current position

We have built a local vehicle-intelligence prototype with a central service, an operator dashboard, a separate engineering console, durable event delivery, experimental detector and OCR adaptations, multi-frame recognition, checked image reconstruction, and an offline synthetic demonstration. The project and its supporting assets are backed up on private GitHub.

The software infrastructure is substantially implemented. Reliable plate recognition on the supplied Sentinel cameras is not established, and an integrated downloaded-video input is not yet implemented. The next teammate should restore the project, connect permitted high-quality recorded footage to the shared pipeline, and measure recognition against independently reviewed labels before calling the product complete.

The latest direction is downloaded CCTV footage rather than reliance on Sentinel connectivity. This changes the input source, not the intended workflow of vehicle tracking, plate recognition, watchlist matching and operator investigation. No deadline or accuracy guarantee is attached to this handoff.

## How to interpret the status

| Area | Current status | Important boundary |
| --- | --- | --- |
| Central API and operator workflow | Implemented and tested | Synthetic workflow success is not live OCR success. |
| Both interfaces | Implemented and browser reviewed | The engineering console is a diagnostic interface, not proof of correct readings. |
| Streaming and AI execution | Implemented with bounded runtime checks | Sustained two-camera AI acceptance is not established. |
| Fine-tuning and reconstruction | Experimental work completed | Not every experimental model is a production default. |
| Live plate recognition | Unproven | No independently verified genuine live registration was established. |
| Downloaded-video dashboard input | To be built | Existing live launchers are not recorded-video launchers. |
| GitHub recovery backup | Published and verified | Download the repository and the release assets. |

## Scope retained and excluded

The scope remains vehicle detection, plate localisation and reading, synthetic watchlist alerts, health monitoring, routes, reports and human review. Face recognition, OpenDVR integration, vehicle appearance identification, unrestricted registry access and an 80,000-camera deployment are not delivered features. Regional scaling is an architectural direction, not a measured laptop capacity.

<!-- pagebreak -->

# Architecture and Current Workflow

## Central workflow

Accepted sightings carry a unique event ID, UTC timestamp, camera ID, masked plate, keyed plate identifier and processing metadata. A SQLite outbox retains unsent events. FastAPI ingests them, checks the synthetic watchlist, prevents duplicate event insertion and stores sightings and alerts in PostgreSQL. WebSockets notify the operator dashboard. Operators acknowledge, investigate, resolve or mark false positives; decisions are audited.

## Recognition workflow

The production worker and the selected engineering session are separate paths. The production worker retains its YOLOv8n and FastALPR defaults. The shared session observed by both interfaces uses the general YOLO26s vehicle profile, ByteTrack, the parent YOLOv9-tiny plate detector and the adapted Awiros CTC OCR reader unless an experimental profile is explicitly chosen.

The shared pipeline continuously decodes video, samples using video presentation timestamps, detects and tracks vehicles, searches vehicle regions for plates, and extracts original-resolution crops. OCR produces character alternatives for distinct observations of the same tracked vehicle. Multi-frame evidence ranks candidate strings. Checked reconstruction can propose additional strings, but those strings are scored against original-frame evidence and do not count as extra independent observations.

An acceptance gate separates operational delivery from uncertain diagnostic output. Best-effort text, partial strings and alternatives can be inspected in the engineering console; they are not automatically confirmed watchlist matches.

## Technology in the repository

| Layer | Technology | Role |
| --- | --- | --- |
| Deployment | Windows, WSL2, Docker Desktop, Compose | Local services and isolated research environments |
| Media | FFmpeg, MediaMTX, PyAV, OpenCV | Streaming, decoding, previews and image operations |
| Detection and tracking | YOLOv8n, experimental YOLO26 and plate models, ByteTrack | Vehicle and plate localisation with track identity |
| Recognition | FastALPR, CCT OCR, adapted Awiros CTC, PARSeq experiments | Baseline and experimental plate readers |
| ML runtime | PyTorch, ONNX Runtime, isolated Paddle GPU runtime | Model inference and adaptation |
| Central API | Python, FastAPI, Pydantic, WebSockets | Contracts, ingestion, alerts and operator actions |
| Storage | PostgreSQL, PostGIS, SQLAlchemy, Alembic, SQLite | Central data, migrations and durable edge queues |
| Frontend | React, TypeScript, Vite, MapLibre, HLS.js | Operator interface, map and stream playback |
| Verification | Python tests, pytest, Vitest, browser checks and benchmark tools | Regression and measured diagnostics |

The shared viewer previously validated CPU detector execution and GPU OCR. GPU visibility or provider availability alone does not prove every stage runs on GPU. H.264 passthrough and conditional incompatible-codec conversion belong to the relay design; NVIDIA encoding is not a blanket measured performance claim.

<!-- pagebreak -->

# Completed Streaming Backend and Delivery Work

## Camera access and media handling

- Camera discovery and authentication adapters were implemented. The newer public-grid catalogue is cameras.json; earlier api/ingest documentation refers to an older or different endpoint.
- Credentials are read locally. Portal sign-in and cookies stay server-side, with retry cooldowns, host restrictions and redacted errors.
- RTSP over TCP, authenticated HLS fallback, reconnect backoff and stream discontinuity handling are implemented.
- FFmpeg relays and MediaMTX provide local previews without recording genuine footage. H.264 can pass through without unnecessary encoding.
- Timing uses video PTS rather than advertised FPS or arrival time. Track and OCR state reset on relevant timestamp gaps, cuts and camera changes.
- Catalogue search is separate from monitored AI cameras. Opening a preview does not add a camera to the AI workload.
- On-demand previews have protected start, status, heartbeat and stop operations, a two-preview cap and idle cleanup.
- Catalogue failures remain explicit. A configured-pair fallback is labelled as such and is not presented as a full catalogue.

## Central services and operator records

- Versioned endpoints cover cameras, sightings, health, watchlist snapshots, alerts, routes, hourly reports, incident reports, catalogue previews and recognition metadata.
- PostgreSQL tables and Alembic migrations persist cameras, sightings, alerts, health reports, watchlist entries and alert audit actions. PostGIS is enabled; authoritative coordinates and production spatial-query validation remain separate work.
- Watchlist matching uses keyed identifiers and a synthetic list. No government vehicle-registry integration is implemented.
- Event IDs make retries idempotent. Accepted shared-session readings use a separate durable outbox with bounded camera and identifier deduplication.
- Alert updates and operator decisions survive API restart. Incident reporting includes the relevant event and audit history.
- Edge liveness and hourly summaries support the health workflow. Hourly report endpoints and synthetic reporting have been exercised.
- Offline simulation produces explicitly labelled synthetic sightings, health changes, routes and suspicious events through the same central workflow.

## Privacy boundaries

Genuine frames, full plate text, OCR posteriors and diagnostic images remain in bounded edge memory during processing. The central recognition projection excludes images, source URLs and full hypotheses. Native diagnostic crops are not routed through the central API. Locking, switching, idle expiry, stale data and shutdown clear relevant evidence. Synthetic evidence remains distinguishable from genuine inputs.

These are implemented protections and test coverage, not an independent security certification. Production authentication, retention governance and broader security review remain unfinished.

<!-- pagebreak -->

# Completed Interfaces and Diagnostic Tools

## Operator dashboard

- Camera health and central connectivity states, including an explicit stale-data warning when the API is unreachable.
- Search by authorised camera ID, name or location text, with ranked matches and pagination.
- Preview-only camera access, live-player states and unavailable or offline feedback.
- Map display using labelled demo coordinates, not authoritative surveyed locations.
- Alert queue and incident detail, with synthetic source labels.
- Acknowledge, resolve and false-positive actions with audit history.
- Ordered route search from sightings and printable incident reporting.
- Hourly summaries and a guarded synthetic alert injector for demo mode.
- Shared recognition controls to start, switch or pause the selected camera and inspect masked states and processing counts.

## Engineering console

- Original scene and detection overlays, including vehicle boxes and track identity.
- Native lossless PNG plate crops. Scene JPEG display and enlarged previews are distinguished from recognition evidence.
- Per-track histories of distinct crops, per-frame OCR text, character alternatives and up to 200 retained candidates.
- Best-effort highest-ranked output when evidence supports a candidate, while retaining uncertainty, partial text and reasons for withholding acceptance.
- Acceptance-gate reasons, sampling PTS, stage timings, collection counters, dropped-packet and freshness information.
- A reconstructed derived-image card, OCR proposals, contributing original count and reconstruction timing.
- Explicit labels that scores are not calibrated probabilities and derived images add zero independent observations.
- Shared start, switch and pause behaviour with the operator panel. The two interfaces observe one selected session, not two duplicate runs of it.
- Operator-key protection, localhost restrictions and bounded RAM cleanup rather than a recording or unrestricted full-text export endpoint.

The shared recognition path now schedules all eligible vehicle regions by default instead of the former small rotating budget. It also includes missing-track-ID association fallback. This removes an artificial processing limit; it does not guarantee the vehicle detector finds every vehicle. Detailed crop retention is still bounded to protect memory.

## Offline viewers and submission artifacts

HTML benchmark viewers compare detector profiles, native crops, OCR adaptations, reconstruction and degradation experiments. JSON reports retain counts, hashes, timings and source limitations. Corpus review sheets and data-audit tools support label inspection and duplicate analysis.

The project also preserves the revised HLD, architecture diagram, solution presentation, captions and demonstration videos. These explain or demonstrate the prototype; their existence is not a new accuracy test. Full catalogue access and preview availability depend on current authorisation and connectivity.

<!-- pagebreak -->

# Models Datasets and Fine Tuning Completed

## Detector comparison and model selection

A bounded comparison tested vehicle variants including YOLOv8n, YOLO26n and YOLO26s, and plate variants including YOLOv9-tiny, YOLOv9-small and plate-trained YOLO11 medium, large and extra-large exports. It was not a survey of every model available worldwide.

On 65 diagnostic conditions derived from 13 images and 11 target vehicles, general YOLO26s at 640 had 63 CPU target hits and 64 GPU hits. YOLOv9-tiny had 61 plate hits; the small variant had 60 with somewhat tighter boxes in some cases. Larger models did not consistently improve this small diagnostic. These are target-overlap checks on transformed images, not vehicle recall across a road network.

Plate and vehicle adaptations were trained and exported, with parent weights, best and last checkpoints, training receipts and evaluation tools preserved. Adapted models had cross-source regressions, so the general vehicle detector and parent plate detector remain the shared viewer defaults. Production defaults were not silently replaced.

## OCR corpus and adaptation

The merged OCR corpus contains 30,110 unique crops: 1,066 publisher-declared photographs, 18,000 known generated plates and 11,044 mixed-origin crops. These are not 30,110 independently verified real vehicles. Exact duplicates were removed, provenance and licence references retained, and origin-unverified data kept separate from the default photographic training pool.

The deeper Awiros adaptation used 802 quality-filtered photographic crops plus a pool of 300 generated plates. Registration-disjoint splits contained 1,102 training crops, 37 validation crops and 31 test crops. Only selected head, recognition-block and encoder parameters were trained; epoch 8 was chosen using validation results. Some labels are publisher-silver or AI-reviewed, not independently human-verified truth.

| Still diagnostic | Parent OCR | Adapted OCR |
| --- | --- | --- |
| Exact validation strings | 24 of 37 | 29 of 37 |
| Exact test strings | 22 of 31 | 28 of 31 |
| Test character errors | 23 of 295 | 13 of 295 |

The 28 of 31 result is 90.3% on this particular small still-image diagnostic. It is not a 90% confidence guarantee, validated CCTV accuracy or proof that unreadable pixels can be recovered. Unknown pretraining overlap and label-selection bias limit generalisation.

## Preserved model locations

Selected engineering OCR: /models/experiments/awiros/adaptations/indian-corpus-encoder.safetensors. Its SHA256 is c2ba896d994b174071c95ace0f5ffc8a61cec3fc6eecce8f0bc5feb7dec4456d.

Detector assets include /training/runs/vehicle-bmd/weights/best.pt, /training/runs/vehicle/weights/best.pt, /training/runs/plate-r3/weights/best.pt and /models/experiments/detectors/yolov9t512-indian-adapted.onnx. Parent models, alternative adaptations and research runtimes are backed up too. Preserve each model and dataset licence separately from the application licence.

<!-- pagebreak -->

# Multi Frame Recognition and Reconstruction Completed

## What the implemented method does

Plate extraction copies a native-resolution pixel region without first enlarging or reducing the stored crop. The OCR model still performs its normal fixed-size input preprocessing. Preserving a crop does not mean that the model has recovered characters missing from the original image.

The system collects distinct observations for the same vehicle track. It suppresses duplicate images and too-close observations, resets inconsistent histories, and keeps track evidence separate. CTC probabilities describe character alternatives and possible alignments. Quality-weighted multi-frame scoring ranks bounded plate candidates rather than simply concatenating the strongest character seen anywhere.

After enough crops exist, checked ECC registration aligns neighbouring crops to a high-quality reference. Fusion combines supported native-grid pixels; unsupported borders retain reference pixels. In the shared implementation, at least three aligned sources and sufficient overlap are required. OCR runs on the derived image, and its suggestions are rescored using original-frame character likelihood and support. A contradictory derived reading can withhold delivery.

The engineering console keeps original and derived evidence separate. Alignment or OCR failure falls back to the original path. Memory, attempts and candidate counts are bounded, and resets clear reconstructed evidence. This is geometric reconstruction plus a fine-tuned deep OCR model, not a newly trained neural temporal reconstruction network.

## What the experiments established

One clear Indian development passage, represented by three permitted NIVU video-derived images, was correctly read by legacy, general and plate-adapted detector profiles. The India-adapted vehicle profile found only one frame and abstained. This is one recognised registration, not three independent successes. Replay PTS were assigned, and training overlap is unknown.

On 21 degraded conditions of that same development passage, parent OCR accepted the correct string in 12 conditions and the adapted reader in 13, with no wrong accepted string in that limited test. A subsequent four-method crop-collection comparison produced 13 correct acceptances and eight abstentions for each method. Checked reconstruction did not increase that acceptance count.

In those experiments, a roughly 54 to 59 pixel native plate could succeed in some conditions; around 41 pixels the correct candidate could remain first-ranked while acceptance failed, and stronger downsampling could remove it from the candidate list. Motion blur also caused failure. These are boundaries of one test passage, not universal camera thresholds.

## What is not established

Reconstruction is not guaranteed to add readable information. OCR alternatives are not guaranteed to contain the true plate. A model score is not a calibrated probability. Reusing generated or reconstructed pixels cannot create independent temporal votes. Learned restoration experiments are separate; learned restoration remains inactive in the shared live session and must not be advertised as deployed super-resolution.

<!-- pagebreak -->

# Verification Results and Acceptance Gaps

## Recorded regression and recovery checks

| Check | Recorded result | Boundary |
| --- | --- | --- |
| Edge regression suite | 287 passed and 1 optional skipped | Regression coverage, not recognition accuracy |
| Backend suite | 53 passed | API behaviour; deprecation warnings remain |
| Frontend suite and build | 15 passed and build passed | Large JavaScript bundle warning remains |
| Handoff tools | 14 passed | Archive, corruption, non-overwrite and SQLite recovery checks |
| Container synthetic WebSocket delivery | One sample 228.87 ms | Excludes OCR and browser rendering; not a percentile |
| Browser-visible synthetic alert | One sample 966 ms | Includes automation overhead; not a distribution |
| Container API outage | About 39 minutes; 156 queued events at one snapshot, then drained | Synthetic recovery, not sustained live recognition |
| API restart | Resolved alert and audit history persisted | Not a complete database-volume disaster drill |
| Representative GitHub restore | Remote archives restored and database counts matched | Existing cached runtime images; not a full fresh OS install |

These results were recorded during development and the October handoff. This document does not claim that all suites were rerun today.

## Live recognition evidence

Authorised catalogue, RTSP and HLS decoding worked in bounded checks. Later shared-session browser checks showed vehicle boxes, native crops, per-frame OCR and camera switching. Short reconstruction probes ran on CAM17 and produced derived windows, but no accepted identifier. Other camera windows had no usable plate crops. Access and relay failures were also observed.

No independently verified genuine Sentinel registration was established. In the recorded shared-session and follow-up live tests, model-accepted readings remained zero. Counter increases, GPU model loading and successful preview playback must not be presented as correctly identified plates.

## Original acceptance conditions still requiring closure

- Two cameras processed concurrently for 15 minutes with measured AI freshness, delivery and reconnection, not merely decoded or tested sequentially.
- A correctly read, labelled real vehicle passage delivered through the complete recognition and operator workflow.
- Broad recognition measurement on unseen, human-reviewed Indian footage, including wrong acceptances and abstentions.
- Repeated end-to-end alert latency and resource measurements rather than one synthetic sample.
- Full clean-machine recovery and repeatable offline operation after rebuilding images from the repository.
- Recorded-video workflow acceptance under the new input direction, including playback controls and explicit source labelling.

The offline synthetic operator demonstration is available. It remains a valid demonstration of alert handling, but it is not a substitute for recognition validation.

<!-- pagebreak -->

# Known Limitations and Failure Diagnosis

## Plate localisation and native information

Small, angled, blurred, distant or night-time plates often supply insufficient character information. Detection can also select an incorrect but geometrically plausible plate region. More vehicle proposals do not necessarily produce more usable plate crops. Image dimensions such as 1920 by 1080 describe the whole scene, not the number of pixels available for each character.

## Missed vehicles and bounded processing

Removing a small ROI budget avoids intentionally skipping eligible vehicle boxes, but detection misses, occlusion, class mapping, unstable tracking, sampling gaps and bounded evidence retention remain. The system cannot honestly promise identification of every vehicle. Large workloads can increase lag and reduce the number of useful repeated crops per passage.

## Reconstruction and OCR uncertainty

Misalignment can damage character edges. Derived images can appear clearer without containing more correct information. A reconstructed hypothesis must remain supported by original observations. The current method is not trained end-to-end on a large Indian video dataset, and no registry lookup is available to validate guesses.

## Dataset and evaluation quality

Some transcriptions are noisy, some origins are uncertain, and the still-image test is small. Development fixtures have been reused for ablations, so they cannot be treated as fresh independent holdout evidence. Missing exhaustive vehicle and plate annotations prevents a defensible real-world recall estimate. Pretrained-model overlap is unknown.

## Operational limitations

Sentinel access has been intermittent. A past relay used outdated local credentials; repairing it established connectivity, not better OCR. Catalogue search must report unavailable rather than scrape portal cards. Approximate map coordinates, shared prototype keys, browser bundle size and framework deprecations all need attention before real deployment.

## Security and restoration limitations

The prototype is local, not a hardened public service with full user accounts and role-based access. Authentic credentials, original HMAC secrets and raw genuine footage are excluded from the handoff. Historical keyed identifiers remain viewable, but new keys cannot reproduce old plaintext-plate searches. Representative restoration passed; a complete new-machine rebuild is still required.

## Correct response to the failures

Diagnose the stages separately: source freshness, vehicle coverage, plate localisation, usable crop yield, OCR, track association and acceptance. Keep frozen labels and thresholds while comparing alternatives. Do not hide failure by accepting every top-ranked string, fitting models to the test sequence, relabelling downloaded video as live, or calling an OCR score measured confidence.

<!-- pagebreak -->

# Immediate Completion Work for the Teammate

## Priority 0 Restore and establish a reproducible baseline

Suggested owner: integrator. Restore a clean checkout plus release assets into isolated destinations, rebuild the services and run the synthetic demo. Verify archived alerts, operator audits, queues, benchmark viewers and checkpoint hashes. Record operating-system, driver, package and model versions. Completion means a clean-machine run succeeds without using the original laptop's Docker images or populated volumes.

## Priority 0 Obtain permitted recorded footage and labels

Suggested owner: data and QA contributor. Select high-quality original-resolution CCTV videos with readable Indian plates and explicit permission to download and use them. Preserve source receipts. Independently transcribe plates, mark unreadable passages and annotate vehicles and plate boxes. Separate training and test data by vehicle, video and scene, not by randomly splitting neighbouring frames. Completion means an auditable held-out set exists before further tuning.

## Priority 0 Implement recorded input in the shared pipeline

Suggested owner: streaming and AI contributor. Add a local-video PyAV source to the shared recognition session. Support select, start, pause, resume and stop; retain original video PTS; clear histories at EOF, looping, seeking and scene changes. Distinguish real-time-paced playback for demonstration from unpaced benchmark processing. Feed native frames into the existing track, crop, OCR and reconstruction code. Existing live launchers must not be presented as this adapter.

## Priority 0 Extend the contracts and both interfaces

Suggested owner: API and frontend contributors. Add an explicit recorded source label through schemas, storage, events, reports and UI. Show recording name, video time and processing state. Keep human decisions and operational alerts separate from diagnostic candidates. Maintain original privacy boundaries and avoid sending full OCR hypotheses through the central recognition proxy. Completion means both interfaces observe one recorded session and source labels remain correct everywhere.

## Priority 0 Prove the full operator workflow with a real labelled passage

Suggested owner: integrator and QA. Run the unchanged recognition acceptance gate on held-out footage. Demonstrate a correctly read vehicle, a known test watchlist match, persistent event delivery and acknowledge or resolve actions. Label an injected watchlist or alert as synthetic where appropriate; do not replace an OCR failure with an undisclosed injected plate. Completion means a reviewer can trace the output to genuine recorded observations and labels.

## Priority 0 Publish measurements and record the new demonstration

Suggested owner: QA and presenter. Report unique vehicle coverage, correct plate localisation, usable crop yield, complete exact reads, wrong acceptances, abstentions, candidate recall and latency. Include failure examples, dropped frames, memory and GPU usage. Select only measured claims for the demonstration and state which footage is recorded. A functioning dashboard alone does not close this task.

<!-- pagebreak -->

# Subsequent Improvement Work

## Priority 1 Improve the bottleneck identified by the baseline

Compare native whole-vehicle crops, tiled plate search, direct-image recovery, detector resolution, sampling intervals and track association on the same labelled passages. Keep compute cost and false detections alongside usable crop coverage. Promote a change only when unseen scenes improve without an unacceptable precision or latency regression. The opt-in recovery path already exists; it is not yet proven to improve live results.

## Priority 1 Evaluate OCR and reconstruction fairly

Compare parent and adapted OCR with original-only evidence, checked reconstruction and deskew under identical crops and labels. Measure complete plate correctness, not just visually attractive images. If the correct candidate is absent, investigate localisation, OCR domain fit and decoding before threshold tuning. If the candidate is present but withheld, inspect evidence support and calibration rather than automatically treating it as true.

## Priority 1 Tune with broader verified Indian data

Review photographic labels, retain source licences, include two-line plates and actual CCTV conditions, and exclude origin-unverified data unless reviewed. Train only on the training partition. Select checkpoints on validation data and evaluate once on untouched vehicle and scene groups. A specialised plate detector or OCR update should remain experimental until its exported runtime has parity and regression checks.

## Priority 1 Calibrate decisions and close resilience gaps

Choose precision, coverage and latency targets before scoring the final test. Calibrate confidence on a separate validation set and preserve an uncertain review state. Test EOF, restart, seek, scene cuts, absent PTS, stale frames, outbox saturation, API outages, model-load failure and interrupted extraction. Confirm accepted events survive replay without duplicates and operator decisions remain persistent.

## Priority 1 Harden the local product

Reduce frontend bundle size, address deprecated framework lifecycle APIs, document reproducible dependency versions, and check current licences. Replace shared prototype keys with authenticated users and role-based actions if deploying beyond a trusted local machine. Add retention policies, access auditing, security review and authoritative camera coordinates where required. Measure two concurrent recorded streams and longer runs before claiming throughput.

## Priority 2 Investigate registry support and regional scaling

Authorised registry access could narrow ambiguous plate candidates using permitted metadata, but it is not implemented and does not establish certainty by itself. Candidate generation must first retain the true plate; wrong or incomplete registry entries can mislead selection. Treat registry matching as a separate approved integration with privacy controls and measured error rates.

Partitioned regional workers, central capacity planning, model deployment, fleet monitoring and security operations require new load and failure evidence. The prototype does not validate an 80,000-camera network. Face recognition and OpenDVR remain outside the present completion scope rather than prerequisites for a plate-recognition product.

<!-- pagebreak -->

# GitHub Backup and Restoration

## What is backed up

Repository: https://github.com/NithisNandhan/sentinel-vehicle-intelligence-prototype

Release: https://github.com/NithisNandhan/sentinel-vehicle-intelligence-prototype/releases/tag/handoff-2026-10-07

The release is published with 68 assets totalling 14,638,600,042 bytes, approximately 14.64 GB. Every asset size and SHA256 matched the local inventory during final verification. Source commit b3b05e186c2449c8cd6b33677b8b596542ce72c0 includes the finalized read-only SQLite snapshot restore fix. Later documentation changes may advance main; the release source tag preserves this recovery baseline.

| Bundle | Parts | Contents |
| --- | --- | --- |
| model-cache | 22 | Detectors, adapted OCR, parent checkpoints and isolated research runtime |
| alpr-cache | 1 | FastALPR detector and OCR weights |
| detector-training | 16 | Training datasets, labels, runs and checkpoints |
| benchmark-data | 18 | OCR corpus, diagnostic reports, receipts and host outputs |
| saved-events | 1 | Central snapshot, SQLite backups and session metadata |
| presentation-assets | 1 | HLD, architecture, presentation, captions and demo videos |

The 59 archive parts accompany six manifests and three validation receipts. Each bundle needs its manifest and all matching parts. The restore tool verifies hashes and refuses nonempty destinations. These are actual files, not Git LFS placeholders.

## Restore sequence

Keep GitHub login and private-repository access. Install Git, current Python 3.11 or newer, Docker Desktop and WSL2. GPU research also needs compatible NVIDIA drivers and container access. Allow roughly 60 GB free as a practical starting allowance for downloads, extraction and images.

```powershell
git clone https://github.com/NithisNandhan/sentinel-vehicle-intelligence-prototype.git
cd sentinel-vehicle-intelligence-prototype
gh release download handoff-2026-10-07 --repo NithisNandhan/sentinel-vehicle-intelligence-prototype --dir data/handoff-assets
.\scripts\check-prereqs.ps1
.\scripts\init-env.ps1
.\scripts\restore-teammate-handoff.ps1 -AssetsPath .\data\handoff-assets
```

The gh command requires GitHub CLI installed and authenticated; browser downloads are also possible. Docker images are rebuilt, not downloaded from this release. The restore starts the central API, dashboard and media gateway, not live cameras. Follow docs/TEAMMATE_HANDOFF.md for selective restore, models, benchmark viewing and offline simulation.

## Preserved state and safety

The snapshot contains six cameras, 16 sightings, 15 alerts, 22 audit entries, 16,625 health reports and one synthetic watchlist entry. Delivery queues were empty at capture. This is a point-in-time snapshot, not automatic backup of future changes.

Credentials, original HMAC secrets, source URLs, raw genuine footage and assistant-local state are excluded. Representative remote-download and database restores passed using cached runtime images. Test a complete clean-machine rebuild before deleting every local copy. A factory reset does not delete GitHub assets, but access to the private repository is still essential.

<!-- pagebreak -->

# Completion Checklist and Source Index

## Conditions for a credible completed recorded video prototype

| Condition | Present position | Evidence needed to close it |
| --- | --- | --- |
| Clean-machine restoration | Representative restore passed | Full rebuild and demo on a clean host |
| Recorded-video input and controls | Not integrated | Start, pause, seek, EOF and restart tests |
| Shared operator and engineering session | Implemented for camera input | Recorded-source integration and source labels |
| Readable Indian plates recognised correctly | Small still and development evidence | Independent held-out multi-vehicle video evaluation |
| Reliable acceptance decisions | Rules implemented, scores uncalibrated | Wrong acceptance and abstention measurement plus calibration |
| End-to-end watchlist and incident workflow | Synthetic flow verified | Correct recorded reading through durable delivery and operator actions |
| Endurance and failure recovery | Synthetic outage and short probes | Two-stream endurance, recorded replay and interruption tests |
| Submission claims | Documentation and demo artifacts exist | Fresh measured recorded-video demonstration with limitations |

## Where to continue in the code

- edge/sentinel_edge/stream.py, cameras.py and relay.py: source access, timestamp-aware decoding and relay handling.
- edge/sentinel_edge/tracked_live_viewer.py and live_detection_viewer.py: selected analysis session and viewer service.
- edge/sentinel_edge/tracked_ocr.py, crop_collector.py and plate_search.py: track evidence, crop collection and localisation.
- edge/sentinel_edge/ctc_fusion.py, sequence_fusion.py, reconstruction.py and live_reconstruction.py: candidate scoring and checked reconstruction.
- edge/sentinel_edge/experimental_detectors.py, corpus_ocr_adaptation.py and tools/run_ocr_research.py: model profiles, adaptation and isolated runtime.
- edge/sentinel_edge/outbox.py, shared_delivery.py and backend/app/main.py: accepted-event delivery, watchlist matching and operator actions.
- backend/app/recognition.py, frontend/src/RecognitionPanel.tsx and engineering_page.py: shared controls and separate metadata or diagnostic projections.
- scripts/restore-teammate-handoff.ps1 and tools/handoff_archives.py, handoff_events.py: archive, database and queue restoration.

## Evidence and further reading

The detailed project records are VALIDATION.md, docs/DUAL_INTERFACES.md, docs/DETECTOR_COMPARISON.md, docs/DETECTOR_OCR_INTEGRATION.md, docs/DEEP_OCR_ADAPTATION.md, docs/INDIAN_OCR_CORPUS.md, docs/LIVE_RECONSTRUCTION.md, docs/CROP_COLLECTION_VALIDATION_2026-09-28.md, docs/PLATE_RECOVERY_VALIDATION.md and docs/TEAMMATE_HANDOFF.md. Older camera choices and test counts in historical entries do not supersede later updates.

The release receipts environment-validation-final.json and factory-reset-recovery-validation.json record the regression and representative recovery scope. Benchmark HTML and JSON are in the benchmark-data archive. Training and dataset permissions remain attached to their original source records; private GitHub storage does not remove those obligations.

## Recommended next action

First complete a clean restore and a labelled recorded-video baseline. Then improve whichever measured stage actually fails. The present product needs better evidence and a new input adapter; it does not need an unsupported promise that larger models, more frames or reconstruction can read every plate.
