# Local live detection viewer

This opt-in viewer makes the existing live shadow diagnostic visible. It does
not modify the production OCR model, alert API, watchlist or outbox.

## Start

With the existing live Compose services running, cached research models present
and credentials configured in the local `.env`:

```powershell
.\scripts\start-live-detection-viewer.ps1
```

Open **http://127.0.0.1:7780/** and choose CAM01 or CAM02. One selected camera is
decoded at a time through the existing MediaMTX relay, with the existing direct
RTSP-over-TCP/HLS reader as fallback if the relay supplies no first frame within
30 seconds. Each initial source wait is bounded; unavailable access is displayed
explicitly and retried with a delay. Pause closes that capture;
changing cameras clears old media and temporal state. Closing/hiding the browser
stops its heartbeat and expires the capture within 60 seconds (plus bounded
decoder/inference shutdown). The whole diagnostic stops after 30 minutes; run the
script again for a new session. Ctrl+C stops the foreground launcher sooner.

## What the screen means

- Green boxes: the three highest-confidence YOLOv8n vehicle regions, if present.
- Yellow box: the highest-confidence full-frame plate proposal, or the first
  detected plate in one of those native vehicle regions. These can be false
  positives; they are not human-verified plate labels.
- The full image and boxes belong to the **same analysed frame**. Detection is
  sampled using PTS with a minimum 500 ms interval. CPU detection/OCR speed can
  reduce actual update rate; this is not full-frame-rate preview.
- The last native plate crop remains visible for up to ten seconds of active
  processing with an age label. The model receives original crop pixels before
  drawing or display encoding. The crop is lossless PNG; CSS enlargement adds
  no detail. The full-frame browser image is JPEG for transport, not OCR input.
- The isolated deeper-adapted Awiros checkpoint runs on GPU. Distinct original
  crops contribute to CTC sequence fusion. Three observations are a minimum,
  not a guarantee of acceptance. Masked leading hypotheses appear after three
  observations. No full registration text leaves the inference process.
- Association is conservative plate-box IoU plus a 1.5-second gap limit, **not
  the complete production ByteTrack vehicle association**. This diagnostic
  intentionally exposes insufficient repeat detections instead of claiming
  a live-recognition result. It does not run reconstruction or learned deblurring.
- Accepted means the model's evidence gates passed, **not verified correctness
  or calibrated confidence**. No accuracy can be inferred without ground truth.
- A stale-frame warning means the image is no longer a current live view.

## Privacy and scope

Only CAM01 and CAM02 are accepted. Genuine images exist only in server/browser
memory; nothing writes them to files, reports, databases or central APIs.
Snapshot responses carry in-memory images and masked text with `Cache-Control:
no-store`; there is no download/recording endpoint. Users can still capture their
own screen; no web viewer can prevent that. Request logging is disabled.

The launch script publishes Docker's port only to `127.0.0.1`. A per-process
viewer token protects snapshots/session requests, and Host/Origin checks reject
cross-site mutations and DNS rebinding. Tokens and stream credentials are never
included in logs or committed source. The standalone module defaults to binding
localhost; `--container-bind` is only for a loopback-published Docker port.

Eleven focused tests cover native pixels, ROI coordinate alignment, masked OCR,
old-session exclusion, camera restrictions, idle expiry, headers and HTTP
token/origin protection, bounded startup, direct fallback and OCR-first GPU
initialization/warmup before YOLO loads the production Torch libraries. The full
edge suite passes 190 tests. Browser/live
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
