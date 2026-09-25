# Sentinel edge worker

The edge worker processes two cameras. Real frames and plaintext plates stay in memory. SQLite stores only masked plates, keyed plate hashes, and metadata.

## Commands

From this directory after installing `pip install -e '.[live]'`:

- `python -m sentinel_edge prepare-models` caches YOLOv8n and FastALPR models under `MODEL_CACHE_DIR` before an offline run.
- `python -m sentinel_edge relay` discovers two cameras, tries authenticated RTSP and then HLS, and publishes H264 streams to MediaMTX. H264 is copied; H265 uses NVENC if available, then libx264.
- `python -m sentinel_edge live` runs two inference workers. Set `MEDIA_RELAY_BASE_URL=rtsp://mediamtx:8554` when the relay is running, so Sentinel sees one ingest connection per camera. Omit it for native/direct access.
- `python -m sentinel_edge simulate` sends deterministic synthetic sightings and continues to send health updates. `--once` sends once and exits.

Docker build context is the repository root. Build `docker build --target offline -f edge/Dockerfile .` for the simulator without CV dependencies, or `docker build --target live -f edge/Dockerfile .` for relay and AI. Mount persistent volumes at `/data` for the outbox and `/models` for model downloads. FFmpeg never records footage.

`python -m sentinel_edge gateway` serves an internal, edge-key-protected catalogue search and temporary preview controller on port 8765. It caches the machine-readable Sentinel catalogue for five minutes, returns only ID/name/location/codec/live metadata, and relays selected feeds to MediaMTX under temporary paths. It is not exposed on a host port. HTML sign-in responses fail closed; valid authorised JSON access is required for full-catalogue search.

Copy `.env.example` to a local untracked secret file and fill `EDGE_API_KEY`, `WATCHLIST_HMAC_KEY`, and for live/relay `SENTINEL_EMAIL` and `SENTINEL_PASSWORD`. If the catalogue or HLS endpoint requires the portal session, supply `SENTINEL_COOKIE` locally as well. Never put Sentinel credentials into Compose YAML or committed files. `CAMERA_IDS` can hold two exact IDs from the catalogue, including numeric IDs. `PREVIEW_BASE_URL` is a credential-free browser-reachable MediaMTX HLS origin. A live worker must have credentials even when reading the local relay, because it verifies the Sentinel catalogue on startup.

The simulator defaults to a route through `cam01` and `cam04` with suspicious plate `GJ01AB1234`. Its HMAC key and demo plate must match the central watchlist seed. It contains no video data.

Health records are sent every 30 seconds, including during feed failure. Frame and reconnect counts describe the preceding interval; the central API sums these into hourly reports.

For the native Windows fallback, leave the API, database, dashboard, MediaMTX and stream relay in Docker. Install Python 3.11, FFmpeg and the `live` extra on Windows, copy the same local secrets into the worker process environment, set `CENTRAL_API_URL=http://localhost:8000`, `MEDIA_RELAY_BASE_URL=rtsp://localhost:8554`, `EDGE_DB_PATH` to a writable local file, and `MODEL_CACHE_DIR` to a persistent local model directory. Start only `python -m sentinel_edge live` on Windows; do not also start the containerized edge worker. `SENTINEL_BASE_URL` is configurable; the authenticated catalogue response and stream access still require live verification.

Run dependency-light checks with `python -m unittest discover -s tests`. Live AI and Sentinel connectivity require the optional `live` dependencies, model assets, valid account access, and a reachable stream.

The [official Sentinel resource guide](https://sentinel.gujarat.gov.in/resource) identifies `/api/ingest` as the camera catalogue and says its stream URLs are authoritative. The actual authenticated JSON shape and live decoding remain integration checks. The adapter accepts numeric or named IDs and top-level or nested `urls` fields, but fails closed if it cannot find an RTSP or HLS URL. Without `CAMERA_IDS`, it chooses the first two reported live entries. Set `CAMERA_IDS` after checking feed quality to lock suitable demo cameras.
