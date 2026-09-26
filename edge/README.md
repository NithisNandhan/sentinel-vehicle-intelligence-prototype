# Sentinel edge worker

The edge worker processes two cameras. Real frames and plaintext plates stay in memory. SQLite stores only masked plates, keyed plate hashes, and metadata.

## Commands

From this directory after installing `pip install -e '.[live]'`:

- `python -m sentinel_edge prepare-models` caches YOLOv8n under `MODEL_CACHE_DIR` and FastALPR's detector/OCR files under the persistent `alpr_cache` Docker volume before a live run. The container fetches missing files automatically; no manual model download is needed.
- `python -m sentinel_edge relay` discovers two cameras, tries authenticated RTSP and then HLS, and publishes H264 streams to MediaMTX. H264 is copied; H265 uses NVENC if available, then libx264.
- `python -m sentinel_edge live` runs two inference workers. Set `MEDIA_RELAY_BASE_URL=rtsp://mediamtx:8554` when the relay is running, so Sentinel sees one ingest connection per camera. Omit it for native/direct access.
- `python -m sentinel_edge simulate` sends deterministic synthetic sightings and continues to send health updates. `--once` sends once and exits.
- `python -m sentinel_edge benchmark-models` measures warmed YOLOv8n/ByteTrack and FastALPR calls on Ultralytics' bundled traffic photo without saving frames or printing plate text.
- `python -m sentinel_edge benchmark-live --probe-seconds 180` compares the legacy pipeline, larger models with legacy consensus, and temporal fusion on the same sampled CAM01/CAM02 frames. It reports only aggregate counts and timings, not images or plate strings, and makes no central writes. Stop the normal worker during this comparison to avoid GPU contention, then restart it even if the benchmark fails.
- `python -m sentinel_edge probe-stream --probe-seconds 20` and `probe-two-streams` measure one or two local MediaMTX RTSP fixtures using the same timestamp sampling as the worker. These are local integration checks, not Sentinel accuracy tests.

Docker build context is the repository root. Build `docker build --target offline -f edge/Dockerfile .` for the simulator without CV dependencies, or `docker build --target live -f edge/Dockerfile .` for relay and AI. Mount persistent volumes at `/data` for the outbox, `/models` for YOLOv8n, and `/root/.cache` for FastALPR's own model hubs. FFmpeg never records footage.

`python -m sentinel_edge gateway` serves an internal, edge-key-protected catalogue search and temporary preview controller on port 8765. It caches the machine-readable Sentinel catalogue for five minutes, returns only ID/name/location/codec/live metadata, and relays selected feeds to MediaMTX under temporary paths. It is not exposed on a host port. HTML sign-in responses trigger bounded form authentication, then fail closed if authorised JSON remains unavailable.

Copy `.env.example` to a local untracked secret file and fill `EDGE_API_KEY`, `WATCHLIST_HMAC_KEY`, and for live/relay `SENTINEL_EMAIL` and `SENTINEL_PASSWORD`. Sign-in redirects trigger normal `/auth/login` form authentication; session cookies stay in memory, refresh after expiry and are scoped to the portal host. Normally leave the optional `SENTINEL_COOKIE` override blank. Never put Sentinel credentials into Compose YAML or committed files. `CAMERA_IDS=cam01,cam02` selects the two approved prototype cameras. `PREVIEW_BASE_URL` is a credential-free browser-reachable MediaMTX HLS origin. A live worker still needs Sentinel credentials when reading the local relay because it checks the catalogue or the explicit fallback at startup.

Direct RTSP uses the catalogue-supplied URL when present and adds percent-encoded registered-user credentials only when its hostname matches `SENTINEL_RTSP_HOST` (default `103.250.160.189`). Set that variable to an approved non-CDN streaming hostname if the provider uses one. Existing URL userinfo is preserved. For IDs in authorised JSON with no URL, the adapter uses the newer public guide's RTSP/HLS patterns. If the catalogue cannot be read, only an explicit `cam01,cam02` selection may use those patterns; full-catalogue search never uses this fallback.

The simulator defaults to a route through `cam01` and `cam04` with suspicious plate `GJ01AB1234`. Its HMAC key and demo plate must match the central watchlist seed. It contains no video data.

Health records are sent every 30 seconds, including during feed failure. Frame and reconnect counts describe the preceding interval; the central API sums these into hourly reports.

For the native Windows fallback, leave the API, database, dashboard, MediaMTX and stream relay in Docker. Install Python 3.11, FFmpeg, the `live` extra, and `lap>=0.5.12,<1` for ByteTrack on Windows; the Docker live image includes `lap` already. Copy the same local secrets into the worker process environment, set `CENTRAL_API_URL=http://localhost:8000`, `MEDIA_RELAY_BASE_URL=rtsp://localhost:8554`, `EDGE_DB_PATH` to a writable local file, and `MODEL_CACHE_DIR` to a persistent local model directory. Start only `python -m sentinel_edge live` on Windows; do not also start the containerized edge worker. `SENTINEL_BASE_URL` is configurable; the authenticated catalogue response and stream access still require live verification.

Run dependency-light checks with `python -m unittest discover -s tests`. Live AI and Sentinel connectivity require the optional `live` dependencies, model assets, valid account access, and a reachable stream.

## Multi-frame recognition

Live mode now defaults to the 512-pixel pretrained plate detector and CCT-S v2 pretrained deep-learning OCR model. The worker fuses full per-character probability distributions across a bounded window for each ByteTrack vehicle. This is neural OCR plus temporal evidence aggregation, not a newly trained video model or super-resolution system. The larger models are cached automatically; `prepare-models` prepares the same defaults used by production.

Read [the recognition methodology and limitations](../docs/MULTIFRAME_RECOGNITION.md) before interpreting scores. The deployed OCR model has ten character slots. Low-quality results and unresolved alternatives never leave the worker as accepted sightings. No registry lookup is implemented.

The newer public-grid guide supplied by the team identifies `/cameras.json` as the catalogue and documents the direct public-IP RTSP and CDN HLS patterns. An older/generic guide names `/api/ingest`; do not mix its endpoint patterns with the public grid. The authenticated JSON shape and live decoding remain integration checks. The adapter accepts numeric or named IDs and top-level or nested `urls` fields. Live/relay require two explicit IDs; only `cam01,cam02` have a documented fallback when catalogue access fails.
