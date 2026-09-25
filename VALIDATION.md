# Prototype validation record

This file separates measured results from acceptance gates that still need Sentinel access or labelled image data. All events used below were synthetic.

## Measured on 2026-09-23

Configuration: Windows laptop, NVIDIA GeForce RTX 3050 Laptop GPU (6 GB), local FastAPI with SQLite, Vite dashboard, and the offline edge simulator. Docker/WSL and live CV inference were not part of these runs. The local browser run used test-only keys and `backend/ui-test.db`, not Sentinel credentials.

| Check | Result | Evidence / command |
| --- | --- | --- |
| Edge unit tests | 11 passed | `PYTHONPATH=edge python -m unittest discover -s edge/tests -q` |
| Backend API tests | 14 passed | `backend/.venv/Scripts/python.exe -m pytest -q` from `backend` |
| Dashboard contract tests | 5 passed | `npm test` from `frontend` |
| Dashboard production build | Passed after final UI fixes | `npm run build` from `frontend`; JS 1,927.30 kB, gzip 556.84 kB, with a bundle-size warning |
| Synthetic API workflow | Passed | `scripts/smoke-test.py`: watchlist match, event-id deduplication, alert queue, acknowledge, resolve, route, incident and hourly endpoints; one measured ingest request 31.5 ms |
| Local alert delivery | Passed | `scripts/measure-alert-latency.mjs`: one synthetic alert reached the dashboard WebSocket in 30.55 ms, below the three-second target. This is not browser-render or OCR latency. |
| Simulator replay | Passed | Two suspicious route sightings became two alerts; replay with the same deterministic event IDs left the alert count unchanged and the outbox empty. |
| Five-minute local API outage | Passed locally | API down from before 17:43:18 UTC until after 17:48:39 UTC. Five synthetic events stayed in the SQLite outbox, then drained on recovery. The central database held three sightings and two alerts for that edge; replay left those counts unchanged. Dashboard changed from Reconnecting to Live updates and showed the recovered alerts. This does not yet test Docker container/volume restart. |
| Browser walkthrough | Passed for synthetic flow | Local dashboard displayed the alert, labelled it simulated, allowed acknowledge and resolve, showed audited incident details and route search. Map canvas layout was corrected during this check. |
| Operator decisions across API restart | Passed locally | The resolved alert and its two audit entries remained available after stopping and restarting the SQLite-backed API. PostgreSQL/Docker persistence remains untested. |
| Browser-visible synthetic alert | One sample passed | From clicking **Inject simulated alert** to an accessibility snapshot containing the new labelled alert took 966 ms, including browser automation overhead. This is one local observation, not a latency distribution. |

The backend test run reports deprecation warnings for FastAPI startup events and Starlette's TestClient/httpx integration. No test failed. The dashboard build reports a large MapLibre-containing JavaScript chunk; code-splitting is a performance follow-up, not a functional failure.

## Docker/WSL validation on 2026-09-24 (local time)

Configuration: Docker Desktop Linux engine 29.8.0 under WSL2, PostgreSQL/PostGIS, containerized FastAPI, React dashboard, MediaMTX, and the offline simulator. Sentinel credentials were not configured; no live camera feed or model was exercised.

| Check | Result | Evidence / command |
| --- | --- | --- |
| Offline Compose startup | Passed | `docker compose --profile offline up --build -d`; database reported healthy and all five offline services started. The initial pull required adding Docker Desktop's existing credential-helper directory to this PowerShell process's PATH. |
| PostgreSQL migration | Passed | API log reported Alembic upgrade `0001_initial` and successful startup. |
| Container GPU visibility | Passed | `docker run --rm --gpus all --entrypoint nvidia-smi sentinel-prototype-api:latest` saw the NVIDIA GeForce RTX 3050 Laptop GPU with 6144 MiB. This checks device exposure, not ONNX or OCR throughput. |
| Synthetic API workflow | Passed | `scripts/smoke-test.py` against the containerized API: watchlist, event-id idempotency, queue, acknowledge, resolve, route, incident, and hourly report. One ingest request took 146.73 ms. |
| Local WebSocket delivery | Passed | `scripts/measure-alert-latency.mjs` against the containerized dashboard/API: one synthetic alert arrived in 228.87 ms, below the three-second target. This excludes browser rendering and OCR. |
| Operator decision after API container restart | Passed | Incident 3 remained `resolved`, `synthetic`, with two audit entries after `docker compose --profile offline restart api`. |
| API outage and durable outbox | Passed | API stopped at 2026-09-23 18:30:42 UTC and restarted at 19:10:05 UTC (about 39 minutes). While down, the simulator's SQLite outbox reached 156 queued events; after recovery it drained to zero. Simulator sighting/alert counts stayed at 3/2, while health-report count rose from 12 to 172. No Docker volume was deleted. |
| Updated edge package and catalogue tests | Passed | Rebuilt the offline simulator image after the official-catalogue adapter change, then ran `python -m unittest discover -s /tests -v` inside that image with tests mounted read-only: 13 passed. The API health endpoint returned `ok` and dashboard returned HTTP 200 after Docker Desktop was restarted. This does not validate the live AI image or Sentinel access. |
| Follow-up offline workflow | Passed | With the five-service offline stack still running, `scripts/smoke-test.py` passed watchlist, idempotency, alert actions, route, incident and hourly checks (one ingest request: 52.36 ms). A separate synthetic WebSocket event arrived in 26.5 ms. Backend 14 tests and dashboard 5 tests passed; the dashboard production build passed with the existing large-bundle warning. |

The Docker outage exceeded the planned five-minute interruption. Its reported queue depth is one snapshot during the outage, not the maximum depth. No percentile latency distribution or sustained load-capacity claim has been made.

## Not yet verified

- Sentinel-camera model accuracy and sustained live throughput remain unverified; a later section records model loading and local RTSP fixture benchmarks.
- The [official Sentinel resource guide](https://sentinel.gujarat.gov.in/resource) identifies `/api/ingest` as the catalogue and says its camera IDs and supplied RTSP/HLS URLs are the contract. The authenticated JSON shape, Sentinel RTSP/HLS connectivity, preview playback, and 15-minute live stability remain unverified. A later authenticated request returned HTTP 403.
- The user selected the first two visible portal cameras: CAM01 (Chiman bhai Bridge) and CAM02 (Janpath). Their portal preview paths used `/cam01/` and `/cam02/`, but local `CAMERA_IDS` is currently blank and the mapping to authorised catalogue IDs is not confirmed. Live startup now requires two explicit IDs.
- The live AI image was subsequently built successfully and loaded the three models; that does not validate access to a Sentinel feed.
- A full database-volume stop/start was not exercised. API container restart and queued-event recovery with persistent Docker volumes were verified.
- Repeated browser-render latency distribution, OCR accuracy on a permitted labelled plate set, and load capacity. One browser-visible synthetic alert was observed within three seconds, but no distribution or live-OCR timing has been measured.

The local map uses deliberately approximate demo coordinates for cam01 and cam04. They are not authoritative camera locations. PostGIS is enabled in the PostgreSQL migration; this first build displays the supplied latitude/longitude columns and does not yet rely on spatial indexes or GIS queries.

## Home-page catalogue search validation on 2026-09-25/26 (local time)

The new catalogue gateway is separate from the two-camera AI worker. It searches cached authorised camera metadata, withholds source URLs from operator responses, and starts at most two temporary preview relays. The offline profile does not start this gateway.

| Check | Result | Evidence / boundary |
| --- | --- | --- |
| Edge catalogue and preview tests | Passed | 24 edge unit tests, including an 8,000-camera fixture, exact-ID/prefix/name/location ranking, pagination, URL exclusion, HTML sign-in rejection, offline and URL-less cameras, relay failure, heartbeat, idle cleanup, and the two-preview cap. |
| Central API tests | Passed | 17 backend tests, including operator-key rejection and response URL stripping. |
| Dashboard tests and build | Passed | 7 frontend tests; production build passed with the existing bundle-size warning. |
| Compose configuration and images | Passed | Online and native Compose configuration checks passed; the catalogue gateway, API, and dashboard images built successfully. |
| Containerized catalogue prerequisite | Expected unavailable | The gateway started, but the current Sentinel request returned sign-in HTML rather than authenticated `/api/ingest` JSON. The protected central search endpoint returned HTTP 503 without presenting the monitored-camera list as a full catalogue. |
| Existing offline workflow after refresh | Passed | The offline API/dashboard were recreated from updated images; the synthetic end-to-end smoke test passed. |

An authorised camera outside CAM01/CAM02 has **not** been opened, and actual on-demand HLS playback has **not** been verified. Those acceptance checks require the authorised JSON catalogue and a reachable third live stream. No claim of live full-catalogue readiness follows from the unit tests or the expected 503 response.

## Live-model preparation and local-stream checks on 2026-09-26 (local time)

Configuration: the built live worker image on the same RTX 3050 laptop, persistent `model_cache` and `alpr_cache` volumes, and two short-lived FFmpeg publishers of Ultralytics' bundled `bus.jpg` image into local MediaMTX. The publishers were stopped after testing. No Sentinel frame was saved or analysed in these checks, and no OCR text was printed or stored.

| Check | Measured result | Boundary |
| --- | --- | --- |
| Model load and cache | YOLOv8n, FastALPR plate detector and OCR model initialized. A second preparation run reused both plate-model files without downloading. | This proves model initialization, not recognition accuracy. |
| Execution providers | YOLO used `cuda:0`; plate detector and OCR sessions reported `CUDAExecutionProvider` with CPU fallback. | ONNX Runtime provider availability and active session configuration were checked, not overall GPU utilization. |
| Bundled-image model benchmark | 20 timed runs after three warm-ups: YOLOv8n/ByteTrack median 31.40 ms, p95 56.86 ms; FastALPR full-image median 37.97 ms, p95 50.64 ms. One vehicle box and zero plates were detected in the last run. | One bundled still image, not a labelled Sentinel dataset or end-to-end camera latency. |
| One paced local RTSP feed | Over a 20-second probe: 94 frames decoded and 94 inference calls; warmed vision-call median 56.09 ms, p95 113.85 ms. Model warm-up was 9.99 s and stream startup 6.03 s. | Local repeated-image feed at about 5 fps, not Sentinel. |
| Two concurrent paced local RTSP feeds | Over 20 seconds: 98/98 decoded/inferred frames on one feed and 94/94 on the other. Vision-call medians were 126.77/127.86 ms; p95s were 180.19/184.98 ms. | Two independent vision pipelines in one worker container, but the scene was the same repeated image on each feed. |
| Authenticated Sentinel catalogue | HTTP 403 from `/api/ingest` with the current local configuration. The authenticated JSON, exact CAM01/CAM02 API IDs, and supplied stream URLs remain unavailable. | No authorised Sentinel camera could be decoded; no live-feed or watchlist claim follows. |
| Direct RTSP access rule | The provided access note says RTSP/WebRTC use the public gateway rather than the HLS CDN, and RTSP requires registered-user credentials in the URL. The adapter now percent-encodes credentials and adds them only to the configured direct-stream host; focused tests pass. TCP port 8554 on the stated gateway was reachable from this laptop. | A reachable port is not a successful camera connection. Without an authorised catalogue path, no Sentinel stream could be decoded. |
| Regression and deployment | 29 edge, 17 backend and 7 frontend tests passed; the online Compose configuration and live worker image built. The offline end-to-end smoke test passed with 154.4 ms synthetic ingest latency. A model preparation reused cached FastALPR files. ByteTrack's `lap` dependency is now in the image. | Synthetic and local fixture checks remain separate from real footage acceptance. |

The local fixture establishes that the current models can load and execute on two concurrent 5-fps test feeds. It does **not** establish vehicle/plate recall, false-positive rate, or a need for tuning on Sentinel imagery. A permitted labelled image set or authorised live feeds with ground truth are required to decide that. No tuning was performed.
