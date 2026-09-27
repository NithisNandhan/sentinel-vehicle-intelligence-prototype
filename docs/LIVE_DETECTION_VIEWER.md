# Local live detection viewer

This opt-in viewer makes the existing live shadow diagnostic visible. It does
not modify the production OCR model, alert API, watchlist or outbox.

The launcher now defaults to the experimental **general YOLO26s + parent plate
detector** profile, with independent ByteTrack OCR histories. See
[integration results](DETECTOR_OCR_INTEGRATION.md): one licensed development
registration was recognised automatically, but the new live CAM14/CAM17 probes
accepted none. `-DetectorProfile legacy`, `plate_adapted` and `indian_adapted`
select alternate cached detectors without changing production defaults. Calling
the module without `--detector-profile` still retains the earlier diagnostic
algorithm for compatibility.

## Start

With the existing live Compose services running, cached research models present
and credentials configured in the local `.env`:

```powershell
.\scripts\start-live-detection-viewer.ps1
```

Open **http://127.0.0.1:7780/** and select a camera from the authorised live JSON
catalogue. CAM01/CAM02 remain quick-access difficult baselines. Only one selected
camera is decoded at a time. New diagnostic cameras use direct RTSP-over-TCP/HLS;
the original production pair uses its existing MediaMTX relay, with direct
RTSP-over-TCP/HLS reader as fallback if the relay supplies no first frame within
30 seconds. Each initial source wait is bounded; unavailable access is displayed
explicitly and retried with a delay. Pause closes that capture;
changing cameras clears old media and temporal state. Closing/hiding the browser
stops its heartbeat and expires the capture within 60 seconds (plus bounded
decoder/inference shutdown). The whole diagnostic stops after 30 minutes; run the
script again for a new session. Ctrl+C stops the foreground launcher sooner.
Authorised temporary previews now retry a clean relay EOF as well as decode
errors, with bounded backoff. The first frame after a reconnect invalidates old
tracking/OCR evidence. A permanently unavailable preview is shown as unavailable
and retried by the viewer, rather than reusing stale frames as fresh evidence.

## What the screen means

- Green boxes: vehicle regions assigned ByteTrack IDs by the selected detector
  at .25 confidence and 640 inference setting. There is no three-box display
  cap. General/legacy profiles include bicycles, cars, motorcycles, buses and
  trucks; the BMD profile explicitly maps its 13 classes to coarse families.
  This does not mean every visible vehicle is detected.
- Yellow boxes: the highest-confidence plate proposal in each searched native
  vehicle region. At most three vehicle ROIs are searched per sampled frame,
  rotating across eligible tracks. Each track has its own OCR history. Proposals
  may be false positives and are not human-verified plate labels. The older
  no-profile diagnostic instead searches full-frame first and selects one plate.
- The full image and boxes belong to the **same analysed frame**. Detection is
  sampled using PTS with a minimum 500 ms interval. Detection/OCR speed can
  reduce actual update rate; this is not full-frame-rate preview.
- The last native plate crop remains visible for up to ten seconds of active
  processing with an age label. The model receives original crop pixels before
  drawing or display encoding. The crop is lossless PNG; CSS enlargement adds
  no detail. The full-frame browser image is JPEG for transport, not OCR input.
- The isolated deeper-adapted Awiros checkpoint runs on GPU. Distinct original
  crops contribute to CTC sequence fusion. Three observations are a minimum,
  not a guarantee of acceptance. Masked leading hypotheses appear after three
  observations. No full registration text leaves the inference process.
- Association now uses ByteTrack vehicle IDs plus independent geometry/class,
  plate-position and stale-OCR guards. Gaps over 2.5 seconds or scene/PTS resets
  clear histories. Continuous decoding keeps the latest frame and drops backlog;
  this can still miss observations under load. The older no-profile diagnostic
  retains plate-box IoU association. Neither mode runs reconstruction or learned
  deblurring, and neither demonstrates verified live accuracy by itself.
- Accepted means the model's evidence gates passed, **not verified correctness
  or calibrated confidence**. No accuracy can be inferred without ground truth.
- A stale-frame warning means the image is no longer a current live view.

## Privacy and scope

The viewer accepts only live camera IDs returned by authorised catalogue JSON;
new IDs are never guessed if access fails. `/cameras` exposes only IDs and names,
never source URLs. Production `.env` camera IDs and Compose services are unchanged.
Genuine images exist only in server/browser
memory; nothing writes them to files, reports, databases or central APIs.
Snapshot responses carry in-memory images and masked text with `Cache-Control:
no-store`; there is no download/recording endpoint. Users can still capture their
own screen; no web viewer can prevent that. Request logging is disabled.

The launch script publishes Docker's port only to `127.0.0.1`. A per-process
viewer token protects snapshots/session requests, and Host/Origin checks reject
cross-site mutations and DNS rebinding. Tokens and stream credentials are never
included in logs or committed source. The standalone module defaults to binding
localhost; `--container-bind` is only for a loopback-published Docker port.

Fourteen viewer tests cover native pixels, ROI coordinate alignment, masked OCR,
old-session exclusion, camera restrictions, idle expiry, headers and HTTP
token/origin protection, bounded startup, direct fallback and OCR-first GPU
initialization/warmup before YOLO loads the production Torch libraries. The full
edge suite passes 199 tests, including sequential screening/report privacy and
protected-gateway fallback/temporary-preview cleanup tests. Browser/live
validation is separate from those automated checks.

An actual GPU smoke check also loaded the fine-tuned reader, ran its warmup,
ran CPU YOLO on synthetic pixels, then ran GPU OCR again successfully. This
validates library coexistence, not live OCR accuracy. OCR-first initialization
avoids the CUDA/cuDNN library conflict found when Torch loaded first.

On 2026-09-27 the page, camera controls and paused state were exercised in the
local browser. Initial access rechecks found an unavailable catalogue and no
CAM02 relay/direct frames. After correcting GPU initialization and restarting
the diagnostic, **CAM02 live display succeeded**: the browser showed 1,336
decoded frames, 93 analysed frames, two possible plate crops and a recent
64 x 26-pixel native crop with one distinct OCR observation. A subsequent visual
check showed aligned green vehicle boxes on the Janpath feed (1,592 decoded,
110 analysed, two plate crops). No accepted plate reading or live OCR accuracy
was established. CAM01 live display in this viewer remains unverified. These are
point-in-time checks, not feed-stability guarantees. No genuine images were saved
and no synthetic substitutions were used to conceal the earlier access blocker.

## Sequential suitability screening

`sentinel_edge.camera_suitability` reuses these same models and gates, validates
each requested ID against current authorised JSON, and opens candidate feeds one
at a time. Its ignored JSON report contains only frame dimensions, counts,
recent median native crop widths/quality, repeat-observation spans, candidate
windows and HMAC-counted unverified accepted identifiers. It retains no images,
masked/full OCR strings, source URLs or credentials. Default screening is 45
seconds after first frame, with a separately bounded startup/whole-attempt timer.
The quality medians describe the most recent 256 samples, not calibrated OCR
confidence. Start/report-update timestamps are local diagnostic UTC times, not
camera video timestamps.

These measurements separate vehicle proposals, plate localisation and temporal
OCR. Frame instances are **not unique vehicles**; crop counts are **not verified
plates**. Without independently labelled visible vehicles/plates there is no
detection recall or OCR accuracy percentage. Scene speed is inspected visually;
no km/h estimate is inferred from uncalibrated camera geometry. Preference for a
clearer test scene does not improve or conceal the CAM01/CAM02 baseline results.

If a new direct catalogue login fails, the diagnostic can reuse the existing
operator-key-protected internal catalogue gateway. It receives only authorised
live IDs/names, then opens an existing temporary preview for the selected camera.
The gateway fallback searches documented `cam`-prefixed grid IDs; a differently
named future grid would require an explicit catalogue integration update.
That preview is heartbeated every 20 seconds, closed in a finally block, and
subject to the gateway's two-preview cap and 60-second idle expiry. A gateway
authorisation/catalogue failure also fails closed. This does not add cameras to
the production AI worker or expose Sentinel URLs to the browser.

### Measured initial screen — 2026-09-27

Each camera ran sequentially for approximately 45 seconds after its first frame.
These are aggregate live measurements, not a labelled accuracy benchmark.

| Camera | Frame size | Analysed frames | Vehicle box instances | Plate crops | Median crop width | Maximum observations | Candidate windows | Accepted, unverified |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CAM12 | 1280×720 | 68 | 3 | 0 | — | 0 | 0 | 0 |
| CAM14 | 1920×1080 | 57 | 320 | 14 | 63.5 px | 3 | 1 | 0 |
| CAM17 | 1920×1080 | 65 | 461 | 32 | 41 px | 8 | 15 | 0 |
| CAM19 | 1280×720 | 61 | 0 | 0 | — | 0 | 0 | 0 |

CAM17 is a provisional repeat-observation test scene, not a proven readable-plate
feed: the browser showed daytime parked motorcycles and passing traffic, with
small, angled, often two-line plates. Its longest associated observation span
was 12.481 seconds. Catalogue names are labels, not verified scene locations.
CAM14's inspected scene was a night intersection with headlight glare and visible
compression damage; its higher nominal resolution did not establish clearer
characters. Sparse detections on CAM12/CAM19 do not prove detector misses without
ground truth. No full plate was accepted in this initial screen.

An additional sequential screen requested 30 seconds each. Startup delays and
the bounded whole-attempt timer left partial analysed windows; these are not
four completed 30-second benchmarks:

| Camera | Resolution | Observed window | Analysed frames | Plate crops | Median crop width | Maximum observations |
|---|---|---:|---:|---:|---:|---:|
| CAM07 | 1920×1080 | 25.08 s | 27 | 0 | — | 0 |
| CAM10 | 854×480 | 9.29 s | 12 | 3 | 32 px | 1 |
| CAM18 | 1920×1080 | 5.39 s | 7 | 0 | — | 0 |
| CAM30 | 1920×1080 | 27.63 s | 12 | 5 | 92 px | 1 |

No additional camera produced a candidate fusion window or an accepted reading.
CAM30's larger crops justify visual inspection, not an accuracy claim. Camera
comparisons use differing scene/time windows and cannot establish detector recall.
New screening reports flag whether the requested observation window completed.
