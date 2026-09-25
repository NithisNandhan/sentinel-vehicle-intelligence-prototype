# Sentinel Vehicle Intelligence Prototype

This repository contains a two-camera demonstration of a regional vehicle-intelligence system. It has an **online** mode for authorised Sentinel feeds and an **offline** mode that sends clearly labelled synthetic events through the same central alert workflow. The 80,000-camera figure in the source brief is an architectural target, not measured capacity of this prototype.

The system does not perform face recognition. Genuine camera frames, crops, clips, and plaintext number plates are not persisted. Offline demonstration imagery is synthetic.

## Before starting

- Windows with WSL2, Docker Desktop, hardware virtualization, and enough free memory for the database, dashboard, media gateway, and AI worker. The online worker needs a working NVIDIA GPU container or the native Windows fallback.
- The NVIDIA Windows driver. Do **not** install a separate NVIDIA Linux display driver in WSL2.
- For online mode, a Sentinel account authorised for the live feeds. Enter credentials only in the local `.env` file. Never commit that file or paste passwords into issues or chat.
- For measured OCR accuracy, a permitted labelled set of plate crops. The synthetic demonstration is not an accuracy benchmark.

WSL2 and Docker Desktop are now installed on the development laptop. For a different machine, see the official [WSL installation guide](https://learn.microsoft.com/windows/wsl/install) and [Docker Desktop installation guide](https://docs.docker.com/desktop/setup/install/windows-install/). Installation may require administrator privileges and a restart.

## Local setup

From PowerShell in this folder:

```powershell
.\scripts\check-prereqs.ps1
.\scripts\init-env.ps1
docker compose --profile offline up --build
```

Run these commands in the same PowerShell window. On a per-user Docker Desktop installation, `check-prereqs.ps1` can add the existing Docker command and credential-helper folder to that window's PATH without changing Windows settings. Start Docker Desktop first if the engine is not running.

Open <http://localhost:5173>. The operator key is in the uncommitted `.env` file as `OPERATOR_API_KEY`; enter it in the dashboard unlock field. The demo injector and operator actions require that key. API documentation is at <http://localhost:8000/docs>.

Online mode requires authorised `/api/ingest` JSON access as well as `SENTINEL_EMAIL` and `SENTINEL_PASSWORD` in `.env`; use `SENTINEL_COOKIE` only if the provider approves a session-based API connection. YOLOv8n and FastALPR weights are downloaded automatically into persistent Docker volumes. After the catalogue confirms two exact camera IDs, set `CAMERA_IDS` and start the two-feed system:

For direct RTSP access, the adapter uses the catalogue-supplied port and stream path and adds the registered email/password to URLs on `SENTINEL_RTSP_HOST` (default `103.250.160.189`), percent-encoding characters such as `@`. It never adds local credentials to a different host. HLS remains on the supplied CDN URL with authenticated request headers. The RTSP host setting does not invent a stream path or bypass the catalogue prerequisite.

```powershell
docker compose --profile online run --no-deps --rm edge-worker python -m sentinel_edge prepare-models
docker compose --profile online up --build
```

If Docker cannot expose the GPU to the AI container, start the central services and CPU media relay with `docker compose --profile native up --build`, then run the edge package on Windows with the same `.env` keys and `CENTRAL_API_URL=http://localhost:8000`, `MEDIA_RELAY_BASE_URL=rtsp://localhost:8554`. The native worker setup is documented in `edge/README.md`.

Only two cameras are selected for processing. The worker reads the official `/api/ingest` catalogue and uses the URLs it supplies; it does not construct stream URLs from IDs. `CAMERA_IDS` must contain two exact IDs from that catalogue before live or relay startup, preventing an arbitrary first-two selection. Online mode does not start without Sentinel credentials. Avoid starting both profiles together, since that mixes synthetic and live activity.

The selected portal cards are CAM01 (Chiman bhai Bridge) and CAM02 (Janpath), but the local `CAMERA_IDS` setting is currently blank. Their portal preview paths use `/cam01/` and `/cam02/`; the authenticated catalogue must confirm the exact API IDs before starting the live profile. Do not let a blank setting silently select a different pair. The offline simulator has its own synthetic route and does not use `CAMERA_IDS`.

Stop the current profile with `docker compose --profile offline down` or `docker compose --profile online down`. The database and outbox use persistent Docker volumes; `down` does not erase them.

## Services

| Service | Responsibility |
| --- | --- |
| `database` | PostgreSQL/PostGIS camera, sighting, alert, and report data |
| `api` | FastAPI ingestion, watchlist matching, operator API, WebSockets |
| `dashboard` | React operator console at localhost:5173 |
| `mediamtx` | On-demand live media gateway; recording disabled |
| `stream-adapter` | Online FFmpeg relay and codec adaptation |
| `stream-adapter-cpu` | Relay for the native Windows AI-worker fallback |
| `catalogue-gateway` | Authenticated full-catalogue search and temporary previews in online mode |
| `catalogue-gateway-cpu` | The same search/preview service for the native-worker fallback |
| `edge-worker` | Online vehicle detection, tracking, plate OCR, durable outbox |
| `demo-simulator` | Offline synthetic sightings and health events |

The worker writes sightings to its SQLite outbox before sending them. The API deduplicates on event UUID. Operator actions have an audit trail. All timestamps are stored in UTC and displayed in India time.

## Find any authorised camera

After unlocking the dashboard, the home page can search the authorised Sentinel catalogue by ID, camera name, or place name. Results are paged and include offline cameras, which cannot be previewed. Opening a live result starts a temporary MediaMTX preview; it does not change the two AI-monitored feeds. At most two temporary previews run at once, and idle previews stop after one minute. Source URLs and Sentinel credentials stay in the internal gateway, not in catalogue search responses.

The catalogue gateway is available only in the online or native profile. The official `/api/ingest` endpoint must return authenticated JSON. If it redirects to an HTML sign-in page, the dashboard shows a catalogue-unavailable message rather than scraping portal cards or showing the two monitored cameras as a false full-catalogue result. The offline demonstration remains usable without catalogue search.

## Privacy and evidence

Live processing only stores masked plate text, a keyed HMAC identifier, confidence, camera, time, model version, and health metadata. The HMAC key must match between edge and API; `scripts/init-env.ps1` generates it once in `.env`. Route search accepts a plate in the operator interface, hashes it for the query, and discards the plaintext. The demonstration watchlist uses synthetic plates only.

Synthetic events and any demonstration image are marked `synthetic` in the API and dashboard. A synthetic alert proves delivery and operator handling, not OCR accuracy. Performance and accuracy claims require recorded measurements.

## Acceptance gates

1. Offline: a synthetic match appears in the alert queue, can be acknowledged and resolved, and appears in the incident report.
2. Durability: after interrupting the API, the outbox retains events and drains after recovery without duplicates.
3. Online: two live feeds remain connected and processed for fifteen minutes, with health visible in the dashboard.
4. Privacy: no genuine frame or plaintext plate is written to the database, outbox, logs, or browser bundle.
5. Measurement: OCR accuracy, event latency, and load results are reported from actual test output only.

The online gates cannot be claimed until Sentinel credentials, camera availability, Docker/WSL2, and GPU access are validated on the presentation machine.
Measured results and remaining gates are recorded in [VALIDATION.md](VALIDATION.md).

## Local verification

With the offline profile running and Python 3.11+ available on the host, execute the synthetic API workflow check:

```powershell
python .\scripts\smoke-test.py
```

It creates a new, labelled synthetic alert, repeats its event ID to verify deduplication, acknowledges and resolves the alert, and checks route and report endpoints. To measure delivery to the dashboard's WebSocket, run `node .\scripts\measure-alert-latency.mjs`; this is not browser-render or live-OCR latency. A short local ingestion benchmark is available with `python .\scripts\load-test.py --seconds 10 --rate 5`; only run longer tests against disposable demo data. None of these checks establishes live OCR accuracy.

