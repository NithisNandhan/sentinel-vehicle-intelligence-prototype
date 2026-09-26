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

## Live YOLOv8n / FastALPR test on 2026-09-26 (local time)

After the user recovered Docker, the cached models loaded on the RTX 3050. YOLO used `cuda:0`; both FastALPR ONNX sessions used CUDA with CPU fallback. RTSP/TCP relays carried CAM01/CAM02 to local MediaMTX. The synthetic simulator stayed stopped. Test-only in-memory instrumentation counted model calls and results without saving frames or printing plate text; no confidence or consensus thresholds were changed.

| Three-minute concurrent live test | CAM01 | CAM02 |
| --- | --- | --- |
| Decoded frames | 2,689 | 5,336 |
| Sampled inference frames | 896 | 890 |
| Approximate sampled calls per wall-clock second | 4.98 | 4.94 |
| Complete vision-call median / p95 | 25.47 / 56.72 ms | 25.30 / 61.60 ms |
| YOLO/ByteTrack median / p95 | 23.91 / 48.41 ms | 23.10 / 45.35 ms |
| Vehicle boxes across sampled frames | 1,593 | 1,407 |
| FastALPR vehicle-crop calls | 168 | 240 |
| FastALPR crop-call median / p95 | 20.36 / 30.46 ms | 19.54 / 33.39 ms |
| Candidate plate regions | 31 | 28 |
| Non-empty OCR results | 1 | 0 |
| OCR results at or above 0.85 confidence | 0 | 0 |
| Confirmed plate sightings | 0 | 0 |

Vehicle boxes and plate regions are repeated detections, not unique vehicle counts or verified correct detections. Timing includes Python/model calls but excludes camera-network-to-dashboard latency. These samples show vehicle detection/tracking and the plate models executing at the configured sampling rate, but do not establish detection recall or OCR accuracy.

The normal live worker also delivered central health reports: two initial reports per camera totalled 914 decoded / 305 inferred frames on CAM01 and 1,852 decoded / 309 inferred frames on CAM02. Both latest states were ONLINE with zero outbox depth; the database contained no live sightings. This verifies live health delivery, not a genuine plate-sighting acceptance gate.

A separate bundled-image smoke benchmark (three warm-ups, 20 repeats, before live tests) measured YOLO/ByteTrack median 12.41 ms / p95 16.07 ms and FastALPR full-image median 15.45 ms / p95 30.38 ms. This remains a non-Sentinel fixture benchmark, not accuracy evidence.

A follow-up diagnostic encountered a temporary MediaMTX 404. PyAV's native `FFmpegError` did not follow the assumed `OSError` inheritance on this image and escaped the existing retry loop. The decoder and relay probe now normalize that error into a redacted `OSError`, preserving the already-approved protocol fallback/backoff flow. Two focused tests cover redaction and successful retry after a native 404. All 43 edge tests and 17 backend tests passed; live images were rebuilt and deployed. The brainstorming scope remained limited to this resilience correction, with no recognition-model or threshold change.

The repeated 45-second crop diagnostic sampled about one frame per second, independently of production crop-quality/dedup gates. CAM01: 37 sampled frames, 51 vehicle crops, 14 candidate plates; candidate widths min/median/max 9/33/34 pixels and heights 5/22.5/24 pixels. CAM02: 36 sampled frames, 54 vehicle crops, 15 candidates; widths 14/28/104 pixels and heights 6/13/28 pixels. Normal OCR returned 0/2 non-empty readings for CAM01/CAM02; adding roughly 5% crop padding and 2x cubic scaling returned 0/3. No reading in either variant met 0.85 confidence. No production model, crop rule, threshold or consensus requirement was changed; the normal live worker was restored after testing.

Interpretation: candidate crops are often small, and simple padding/upscaling did not solve confident recognition in this sample. Small size is a plausible contributor, not a proven root cause: false detections, viewing angle, blur and model/domain mismatch have not been separated by labelled evaluation. YOLO fine-tuning is not justified by these tests. OCR improvement is needed before claiming usable number-plate recognition, but deciding between better input/plate cropping, another detector/OCR model and fine-tuning requires permitted labelled plate examples. Lowering thresholds simply to create sightings would not validate recognition. The 15-minute AI acceptance run and genuine confirmed central sighting remain incomplete.

Final checks: a real missing local RTSP path produced redacted `OSError` results from both the deployed decoder and relay probe, with no source URL in the message. Fresh central reports after restoring the worker showed CAM01 ONLINE with 462 decoded / 154 inferred frames, CAM02 ONLINE with 857 decoded / 143 inferred frames, and zero queued events on both. API, database, dashboard, MediaMTX, gateway, stream adapter and live worker were running. All 43 edge, 17 backend and 7 dashboard tests passed. The live worker remains running; these snapshots are not a 15-minute uninterrupted acceptance run.

## Multi-frame deep-learning OCR upgrade on 2026-09-26

Implemented per-vehicle character-probability fusion on top of pretrained neural detection/OCR. Production defaults now use `yolo-v9-t-512-license-plate-end2end` and `cct-s-v2-global-model`; YOLOv8n and ByteTrack remain. See [the exact methodology and limits](docs/MULTIFRAME_RECOGNITION.md). No custom temporal network was trained, no super-resolution was added, and no registry was queried.

An initial comparison had a requested 180-second window but only about 86–87 seconds between first and last decoded frames because startup consumed much of the window. All methods accepted zero plates. The benchmark was corrected to start each camera's observation timer at its first decoded frame, and rerun. The following is the corrected run, starting at **2026-09-26 05:09:47 UTC** (10:39:47 India time), with models initialized and warmed first. Both feeds used local RTSP/TCP relays; production inference was stopped during comparison to avoid another competing worker. The simulator remained stopped. No footage, crops, OCR strings or central sightings were saved by the benchmark.

| Corrected concurrent live comparison | CAM01 | CAM02 |
| --- | --- | --- |
| Actual first-to-last-frame observation | 179.60 s | 179.38 s |
| Stream startup | 0.80 s | 4.35 s |
| Decoded frames | 2,366 | 3,010 |
| Sampled frames, shared by all three methods | 450 | 194 |
| Vehicle boxes across sampled frames | 1,138 | 249 |
| Tracker IDs observed | 96 | 33 |
| Legacy: plate regions / non-empty OCR / accepted unique identifiers | 49 / 0 / **0** | 8 / 1 / **0** |
| Larger models with legacy consensus: regions / non-empty OCR / accepted identifiers | 6 / 0 / **0** | 1 / 0 / **0** |
| Temporal pipeline: regions / probability observations / accepted identifiers | 11 / 9 / **0** | 1 / 1 / **0** |
| Temporal tracks reaching three observations | 1 | 0 |
| Legacy vision-call median / p95 | 20.29 / 64.23 ms | 27.54 / 82.44 ms |
| Larger-model consensus vision-call median / p95 | 20.25 / 64.89 ms | 31.06 / 89.85 ms |
| Temporal vision-call median / p95 | 32.52 / 80.42 ms | 29.64 / 84.21 ms |
| Processing errors, every method | 0 | 0 |

Counts are repeated model detections, not labelled correct plates or unique verified vehicles. All variants used the same sampled frames, but temporal processing also widened crop gates and attempt budgets: this is not a fusion-only ablation. The six comparison pipelines share one GPU, so sampled rates and timing are not production-throughput or camera-to-dashboard latency claims. First/last-frame observation does not prove uninterrupted delivery. Larger models did not demonstrate an improvement; fewer detected plate regions cannot be interpreted as better precision without labels.

The real-model probability adapter passed a parity check on an **in-memory synthetic rendered image**: its decoded slots and per-slot maxima matched the public OCR API, maximum score difference 0.0; maximum probability-row-sum error approximately 9.92e-8. Both detector/OCR sessions used CUDA with CPU fallback. The OCR model exposes ten character slots. Runtime versions on this laptop were Torch 2.14.0+cu130 and ONNX Runtime 1.30.0. This check establishes adapter correctness, not correct recognition of the synthetic label or real-camera accuracy.

Verification: **60 edge tests, 17 backend tests and 7 frontend tests passed**. Edge tests include complementary character errors recovered at the unchanged production acceptance threshold, non-inflation of scores from repeated identical inputs, abstention on weak or tied characters, timestamp spacing, bounded buffers/attempts, scene/spatial resets and duplicate suppression. These are synthetic probability fixtures and mocked-model pipeline tests, not labelled neural-model accuracy tests. Backend tests retained existing deprecation warnings; frontend tests required running outside the restricted filesystem sandbox because its build tool scans parent directories.

**Measured outcome: zero accepted live plate identifiers across both cameras.** The upgrade is implemented, but useful recognition is still unproven. There are no ground-truth labels, so whole-plate accuracy, accepted precision, candidate recall and confidence calibration remain unknown. Zero accepted identifiers does not prove recognition is impossible; nor does successful fusion in a synthetic test show that this footage contains sufficient character evidence. The genuine-sighting and fifteen-minute AI acceptance gates remain unmet. No threshold was lowered to generate alerts, and no tuning was performed.

Deployment handoff: the rebuilt temporal worker is running. Post-test source stalls required restarting the stream adapter and worker; their cause was not established. At **05:18:55–56 UTC**, fresh central health reports showed CAM01 ONLINE with 431 decoded / 144 inferred frames, CAM02 ONLINE with 872 decoded / 147 inferred frames, and zero outbox depth on both. The worker's checked startup logs had no tracebacks or AI-processing failures and reported CUDA providers. This verifies restored processing and health delivery at that snapshot, not uninterrupted stability or a genuine recognised plate.

## Licensed multi-frame degradation pilot on 2026-09-26

The final runs generated reports at approximately 13:44–13:45 UTC. A manually
labelled development passage used three distinct, vehicle-centred video-derived
NIVU images published under CC BY 4.0. Original video PTS were unavailable;
400 ms replay spacing was explicitly assigned. This is one vehicle, not an
independent test set or a live timing validation. See
[method, attribution and reproduction](docs/DEGRADATION_BENCHMARK.md).

Both CCT-S and the pinned official PARSeq model were evaluated with the same
temporal v2 safeguards, frames, thresholds and declared replay timing. Each ran
21 quality conditions in two stages: automatic detection/tracking/OCR and a
human-marked-crop OCR diagnostic. Across **84 condition-stage evaluations of
the same passage**, there were **zero correct accepted readings, zero wrong
accepted readings and 84 abstentions**. Accepted precision is undefined at zero
coverage. This is not evidence of general zero accuracy or a universal limit.

| Resolution scale | Median annotated plate width | CCT end-to-end diagnostic |
|---|---|---|
| 1.0 | 109 px | Correct final top candidate, but weakest character score 0.259; abstained |
| 0.75 | 82 px | Candidate ranking ambiguity; abstained |
| 0.5 | 55 px | Candidate ranking ambiguity; abstained |
| 0.375 | 41 px | True label absent from final retained beam; abstained |
| 0.25 | 27 px | Target plate localization failed; abstained |
| 0.1875 | 20 px | Target plate localization failed; abstained |
| 0.125 | 14 px | Target plate localization failed; abstained |

At original quality the CCT end-to-end best mean score was 0.8584, but its
weakest score was 0.2588. The marked-crop stage also abstained: best mean 0.8261,
weakest 0.3800, and the correct identifier was not top-ranked. PARSeq's original
marked-crop result was similarly insufficient: best mean 0.8706, weakest 0.1912,
with incorrect top ranking. None of these scores is calibrated correctness.
All Gaussian blur, horizontal motion, noise and brightness conditions also
abstained. Recognition already fails at original quality, so this pilot does
not establish the onset of a successful-to-failing recognition curve.

Inference-stage timing is not dashboard latency. On the original three-frame
passage, CCT took 858.47 ms end-to-end including the first model call and 23.22 ms
in the marked-crop diagnostic; PARSeq took 794.29 and 84.07 ms respectively.
These single-passage timings are not representative throughput benchmarks.
Final aggregate JSON reports remain local under ignored
`data/benchmarks/reports/final-*.json`; they include runtime versions, model
revision, thresholds and numeric diagnostics without images or plaintext labels.

The benchmark warm-up was corrected to explicitly select ByteTrack. Variable
export dimensions are now padded rather than stretched. Target-only evidence,
vehicle-box matching, decoded annotation bounds, explicit timing provenance and
plate-level split isolation prevent several misleading scoring outcomes.

Production temporal v2 adds grammar-aware beam pruning, trailing-length evidence
checks, geometric/class/time-gap fusion resets and an internal bounded NO-READ
reason. The changed image was rebuilt and the CCT live worker restored. PARSeq
remains an opt-in research adapter; no production model, threshold or watchlist
was switched. Its initial missing-C-compiler error was resolved with the
installed Torch's experiment-only `TORCH_DISABLE_NATIVE_JIT=1` setting.

At **13:48:32 UTC**, both CAM01 and CAM02 had fresh central ONLINE health
timestamps after deployment. Checked startup logs showed cached models loading
with CUDA/CPU providers and no traceback. This verifies restored health reporting
at that snapshot, not continuous stability or recognised live plates.

Verification: **102 edge tests and 17 backend tests passed**; backend retained
three existing deprecation warnings. This does not demonstrate recognition gain.
The synthetic end-to-end smoke test also passed watchlist matching, idempotency,
alert queue, acknowledgement, resolution, route, incident and hourly-report
checks. It created labelled synthetic alert 12; ingest latency was 182.35 ms,
not camera-to-dashboard latency or an OCR benchmark.
No training, P2 detector, learned temporal restoration, two-row segmentation,
appearance matching or confidence calibration was performed. Further work needs
more permitted original passages and independently verified labels; lowering
thresholds or treating the one-vehicle development pilot as accuracy would be
misleading. Live recognition and the fifteen-minute acceptance gate remain unmet.
