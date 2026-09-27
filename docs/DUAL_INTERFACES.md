# Operator dashboard and engineering console

## Start and use

Run `scripts/start-dual-interfaces.ps1` from PowerShell in the project folder.
Docker Desktop must be running; the existing cached detector/OCR assets and
authorised local `.env` are required. Use `-SkipBuild` only when the latest
API/dashboard images are already built. This does not install or retrain models.

- Operator dashboard: http://127.0.0.1:5173/
- Engineering console: http://127.0.0.1:7780/

Unlock each using the existing local operator key, without sharing it. The
engineering key lives only in page memory; the existing operator dashboard key
remains in session storage. Select a camera and explicitly start analysis in
either interface. Both observe and control the same single selected session.
Pause affects both; closing/locking one view does not stop the other. A visible
interface sends a 20-second heartbeat; 60 seconds without heartbeat pauses and
clears the capture. The service itself has an eight-hour maximum lifetime when
started through the dual-interface launcher.

If an earlier viewer container is already running, the launcher does not replace
it silently. Stop `sentinel-live-detection-viewer` explicitly and relaunch to use
updated code. Only that viewer should be stopped; the production services remain.

## What is shared, and what is not

The shared selected pipeline uses cached general YOLO26s vehicle detection,
ByteTrack, the parent YOLOv9-tiny plate detector and adapted Awiros CTC OCR.
Distinct observations are fused per vehicle. Detector execution is CPU and OCR
is GPU in the previously validated isolated runtime, not an all-GPU claim.

The existing two-camera production worker is still separate and unchanged.
Opening the two new interfaces does not create two copies of the selected AI
session; it does not eliminate the separate existing production worker either.
Camera search/ordinary HLS playback remain preview-only and do not start AI.

The API exposes four operator-key-protected metadata endpoints:

- `GET /v1/recognition/status`
- `GET /v1/recognition/cameras`
- `POST /v1/recognition/session` with explicit `camera_id` string or null
- `POST /v1/recognition/heartbeat` with the current camera ID

Heartbeats cannot select another camera or restart a paused capture. Internal
edge routes use the edge key; browser evidence routes require the operator key,
process token and local Host/Origin controls. Full candidate text, images, OCR
posteriors and stream URLs never enter the central recognition proxy responses.

When current catalogue access fails, the opt-in launcher may offer only the
previously approved/configured local CAM01/CAM02 relays. This scope is explicitly
labelled `configured_relays`, not a complete catalogue. Full-catalogue search
keeps its independent unavailable state; no portal scraping is introduced.

## Operator view

The existing command-centre layout retains catalogue search, camera preview,
alerts, map, routes, reports and audited actions. Its shared recognition panel
adds camera Start/Switch/Pause, fresh processing counts and masked per-track
reading states. It retries camera discovery after startup failures. Stale track
rows are suppressed, and requests do not overlap. Images and full hypotheses
are not received by this panel.

## Engineering view

Shows the same session's aligned overlays or original scene display, native
lossless PNG crops, selected-track histories (up to eight distinct frames),
per-frame greedy OCR, CTC character alternatives, up to 200 retained candidates,
acceptance gate reasons, video PTS and measured stage timings. Scene display is
JPEG; only native crops are lossless. Display enlargement adds no information.

These are observable model inputs/outputs, not access to internal thoughts.
Character scores, quality scores and ranking margins are not calibrated
probabilities. Reconstruction is explicitly `not_run` in this session; no
reconstruction output or accuracy benefit is claimed. Existing offline hybrid
experiments are separate.

Images, full hypotheses and posterior histories stay in bounded RAM. Crop bank
budget is 32 MiB/32 tracks/eight frames per track; detailed snapshots include at
most 16 recent tracks. OCR evidence expires after a 2.5-second PTS gap even when
a vehicle stays visible but no plate is detected. Stale HTTP snapshots omit
sensitive evidence. Lock, hidden-page state, errors, camera switches, idle
expiry and shutdown clear the relevant evidence. No media download, recording
or full-text logging endpoint exists.

## Delivery and truthfulness

The dual launcher enables durable delivery of strictly model-accepted readings
through a separate SQLite outbox volume. Only masked plate/HMAC/metadata is
stored. Uncertain/partial candidates are never delivered. Camera+identifier
deduplication is 60 seconds and bounded; central event IDs prevent retry
duplicates. Model version explicitly labels these readings unverified.

A model-accepted reading is NOT an independently verified registration. No new
OCR accuracy result is established by these UI changes. Existing demo alerts are
synthetic and remain labelled; an operator can demonstrate the alert workflow
without pretending they were recognised from Sentinel imagery.

## Recorded verification on 27 September 2026

Both services were rebuilt and browser-reviewed. The engineering console started
CAM17 and the operator dashboard observed the same active camera, seven current
vehicle boxes and two cumulative plate crops. The operator then switched the
shared session to CAM01; the engineering console observed the same switch and
displayed native crops, partial OCR, character alternatives and gate reasons.

A deployed metadata-only `/v1/recognition/status` check on CAM01 reported 1,795
decoded frames, 166 analysed frames, seven plate crops, frame age 0.1 s and zero
model-accepted readings. These are one-session observations, not vehicle recall,
OCR accuracy, sustained capacity or a 15-minute/two-feed acceptance result.
Current catalogue/sign-in access initially failed, then recovered on retry and
returned 30 authorised live cameras. Availability can change; startup errors are
shown explicitly.

The backend suite passed 53 tests and frontend suite passed 15 tests plus a
production build. The final edge suite result is recorded in `VALIDATION.md`.
Executable engineering-page behaviour tests also ran locally with Node; Node is
optional and this one test skips in the Python-only edge test container.
