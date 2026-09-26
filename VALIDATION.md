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
- The newer public-grid resource guide identifies `/cameras.json`, not the older/generic `/api/ingest`, as the public catalogue. Updated credentials now validate authenticated JSON and short RTSP/HLS decoding checks (see the latest section); browser preview playback and 15-minute AI stability remain unverified.
- The user selected CAM01 (Chiman bhai Bridge) and CAM02 (Janpath). The newer guide identifies their IDs as `cam01` and `cam02`, and local `CAMERA_IDS` is now pinned to that pair. The documented-pattern fallback does not prove either feed is currently authorised or live.
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
| Containerized catalogue prerequisite | Expected unavailable | At this historical check, the gateway reached sign-in HTML rather than authenticated `/api/ingest` JSON. The protected central search endpoint returned HTTP 503 without presenting the monitored-camera list as a full catalogue. The later public-grid guide supersedes the endpoint choice; see the current access section below. |
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
| Earlier catalogue attempt | HTTP 403 from the older/generic `/api/ingest` endpoint. The newer public-grid guide subsequently named `/cameras.json`; see the current access section below. | This old endpoint result is not evidence that the newer catalogue works. |
| Direct RTSP access rule | The provided access note says RTSP/WebRTC use the public gateway rather than the HLS CDN, and RTSP requires registered-user credentials in the URL. The adapter now percent-encodes credentials and adds them only to the configured direct-stream host; focused tests pass. TCP port 8554 on the stated gateway was reachable from this laptop. | A reachable port is not a successful camera connection. Without an authorised catalogue path, no Sentinel stream could be decoded. |
| Regression and deployment | 29 edge, 17 backend and 7 frontend tests passed; the online Compose configuration and live worker image built. The offline end-to-end smoke test passed with 154.4 ms synthetic ingest latency. A model preparation reused cached FastALPR files. ByteTrack's `lap` dependency is now in the image. | Synthetic and local fixture checks remain separate from real footage acceptance. |

The local fixture establishes that the current models can load and execute on two concurrent 5-fps test feeds. It does **not** establish vehicle/plate recall, false-positive rate, or a need for tuning on Sentinel imagery. A permitted labelled image set or authorised live feeds with ground truth are required to decide that. No tuning was performed.

## Historical public-grid access failures on 2026-09-26 (local time)

The team supplied a more specific public-grid guide: `/cameras.json` is the camera catalogue; direct RTSP uses `103.250.160.189:8554/stream/<id>` with URL-encoded registered-user credentials; HLS uses `https://cctv.corp8.cloud/<id>/index.m3u8`. Another pasted page still refers to `/api/ingest` and generic host URLs, so it is treated as an older or different deployment until the provider clarifies it. The common guidance about RTSP/TCP, PTS timing, mixed codecs, backoff, and loop discontinuities still applies.

| Check | Result | Boundary |
| --- | --- | --- |
| Public catalogue without credentials | HTTP 302 to `/auth/login` from `/cameras.json`. | Not authenticated JSON. |
| Public catalogue with configured Basic credentials | HTTP 302 to `/auth/login` from `/cameras.json`. | The local email/password alone do not establish a catalogue session. |
| Direct RTSP gateway | TCP port 8554 reachable from this laptop; a metadata-only `ffprobe` of the documented CAM01 path returned unauthorized. | Network reachability is proven; stream authentication and decoding are not. No footage was saved. |
| HLS fallback endpoint | A bounded request to the documented CAM01 playlist with the configured Basic credentials returned HTTP 403. | HLS access is also unverified; no playlist or media content was printed or saved. |
| Adapter and fallback | `/cameras.json` is tried first. JSON camera IDs without URLs receive the documented public patterns. Only explicitly configured `cam01,cam02` may use those patterns when catalogue access fails. The full-catalogue search gateway never uses the pair fallback. | No dashboard result is presented as a full catalogue until authorised JSON is received. |
| Full-catalogue fail-closed check | A rebuilt gateway queried with the current local access returned status 503 and no search results. | The dashboard cannot search all cameras until authorised `/cameras.json` JSON is available. |
| Current regression | 35 edge, 17 backend, and 7 frontend tests passed. The online image and Compose configuration built; the offline end-to-end smoke test passed with 78.7 ms synthetic ingest latency. An ephemeral live worker selected `cam01,cam02` with `documented_fallback=True` without opening media. | Local/synthetic behavior is verified separately from Sentinel media access. |

These access failures may mean the account lacks the required approved-list permission, uses a different access password, or needs a separate login session. The tests do not identify which. No live Sentinel frames or plate reads have been measured.

## Updated credentials and actual live decoding on 2026-09-26 (local time)

The user updated the local credentials. Normal portal form sign-in now succeeds, and the adapter performs that sign-in automatically when catalogue requests return a login page. Cookies remain in memory and are not included in operator responses. Failed authentication has a retry cooldown; cross-origin portal redirects are rejected. HLS media uses domain-scoped FFmpeg cookies rather than a raw Cookie header forwarded to arbitrary media hosts.

| Check | Result | Boundary |
| --- | --- | --- |
| Authenticated catalogue | 30 JSON camera records; CAM01/CAM02 selected without fallback. | This is the current authorised catalogue, not an 8,000-camera deployment. |
| HLS playlist access | Both CAM01 and CAM02 returned valid playlists after form login. | Separately followed by actual decoding below. |
| Concurrent RTSP/TCP decoding | Native Windows PyAV 16.1.0: CAM01 328 frames in 20.016 seconds, startup 4.843 seconds; CAM02 481 frames in 20.125 seconds, startup 4.422 seconds. Both H.264, 1920x1080; no PTS discontinuities in these samples. | Decode only. No frames saved, AI calls, plate reads, central writes, or 15-minute stability test. |
| Concurrent authenticated HLS decoding | CAM01 1,569 frames in 20.063 seconds, startup 2.484 seconds; CAM02 2,421 frames in 20.000 seconds, startup 3.313 seconds. Both H.264, 1920x1080. | Buffered content advanced PTS by about 104.6/80.7 seconds respectively. These counts are not real-time delivered fps or AI throughput, and do not establish low latency. |
| Full-catalogue gateway search | Native `CatalogueState.search('cam03', 0, 25)` returned one authorised CAM03 result, with no source URL fields. | HTTP/dashboard playback and a third-camera preview remain unverified because Docker was unavailable. |
| Authentication and regressions | 41 edge tests, 17 backend tests and 7 frontend tests passed. Dashboard production build passed with the existing large-bundle warning. | Tests include a local HTTP login fixture, cookie reuse/expiry, parallel sign-in, rejection cooldown, cross-origin protection and media cookie scoping. |
| Updated live images | Stream adapter, catalogue gateway and worker rebuilt using cached dependency layers. | An image build does not prove runtime AI processing. |
| Local Docker runtime | A normal restart briefly restored the engine and allowed rebuilding. Starting online services failed with HTTP 502 and WSL bootstrap exit `0xc00000fd`; subsequent Docker status was stopped. API requests timed out. | No factory reset, volume deletion, Windows restart, driver change or WSL reinstallation was performed. Root cause is not established; sustained AI/central/dashboard checks remain blocked. |

Reproduce bounded decode checks with `backend/.venv/Scripts/python.exe scripts/check-live-access.py --seconds 20` and `--protocol hls` for the fallback. The script reads local `.env`, does not print source URLs or native error messages, and retains no media. Its optional 900-second duration is a decoder check, not the complete AI acceptance test.

The brainstorming step kept this change scoped to the previously approved automatic sign-in workflow. No model tuning was performed or justified by these decoder-only results. Actual live YOLO/FastALPR performance, OCR ground-truth accuracy, a genuine central sighting and the 15-minute two-camera AI acceptance gate still require verification.
