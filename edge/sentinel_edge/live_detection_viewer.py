"""Opt-in local diagnostic: aligned live overlays, no recording or central writes.

Frames and plate text remain in process/browser memory. HTTP responses are
non-cacheable. Masked OCR is unverified; plate boxes may be false positives.
One selected camera is decoded; browser heartbeat expires after 60 seconds.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import secrets
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .privacy import mask_plate


class LiveState:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.RLock()
        self.camera_id = None
        self.capture_stop = threading.Event()
        self.last_heartbeat = 0.
        self.value = {'status': 'paused', 'camera_id': None}

    def select(self, camera_id):
        if camera_id not in (None, 'cam01', 'cam02'):
            raise ValueError('Only CAM01 and CAM02 are authorised for this diagnostic')
        with self.lock:
            self.last_heartbeat = self.clock()
            if camera_id == self.camera_id:
                return
            self.capture_stop.set()
            self.capture_stop = threading.Event()
            self.camera_id = camera_id
            self.value = {'status': 'connecting' if camera_id else 'paused', 'camera_id': camera_id}

    def expire(self):
        with self.lock:
            if self.camera_id and self.clock() - self.last_heartbeat > 60:
                self.select(None)

    def publish(self, value, capture_stop=None):
        with self.lock:
            if (self.camera_id == value.get('camera_id') and
                    (capture_stop is None or capture_stop is self.capture_stop)):
                self.value = value

    def snapshot(self):
        with self.lock:
            result = dict(self.value)
            if result.get('published_at'):
                result['frame_age_seconds'] = round(self.clock() - result['published_at'], 1)
            return result


def safe_request(host, origin, mutation, port=7780):
    if host not in (f'localhost:{port}', f'127.0.0.1:{port}'):
        return False
    return not mutation or origin == f'http://{host}'


def clipped_box(image, box):
    x1, y1, x2, y2 = map(int, box)
    return max(0, x1), max(0, y1), min(image.shape[1], x2), min(image.shape[0], y2)


def crop_box(image, box):
    x1, y1, x2, y2 = clipped_box(image, box)
    return image[y1:y2, x1:x2].copy() if x2 > x1 and y2 > y1 else None


def detect_regions(image, vehicle_model, detector):
    result = vehicle_model.predict(image, imgsz=640, classes=[2, 3, 5, 7], conf=.25,
                                   device='cpu', verbose=False)[0]
    ranked = sorted(zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()), key=lambda v: -v[1])[:3]
    vehicles = [clipped_box(image, box) for box, _ in ranked]
    plate = detector.box(image)
    if plate is None:
        for box in vehicles:
            region = crop_box(image, box)
            if region is None or not region.size:
                continue
            local = detector.box(region)
            if local is not None:
                plate = (local[0] + box[0], local[1] + box[1], local[2] + box[0], local[3] + box[1])
                break
    return vehicles, clipped_box(image, plate) if plate is not None else None


def encoded(image, extension):
    import cv2
    options = [cv2.IMWRITE_JPEG_QUALITY, 80] if extension == '.jpg' else []
    ok, data = cv2.imencode(extension, image, options)
    if not ok:
        raise ValueError('Display encoding failed')
    return base64.b64encode(data).decode('ascii')


class StartupStop:
    """Propagate user cancellation and bound the wait for the first frame."""
    def __init__(self, parent, timeout=30, clock=time.monotonic):
        self.parent, self.clock = parent, clock
        self.deadline = clock() + timeout
        self.started = False

    def is_set(self):
        return self.parent.is_set() or (not self.started and self.clock() >= self.deadline)

    def wait(self, seconds):
        if not self.started:
            seconds = max(0, min(seconds, self.deadline-self.clock()))
        return self.parent.wait(seconds) or self.is_set()


def viewer_frames(config, camera, stop):
    from .stream import read_camera
    options = [config]
    if config.media_relay_base_url:
        options.append(replace(config, media_relay_base_url=''))
    for candidate in options:
        guard = StartupStop(stop)
        received = False
        for transport, frame in read_camera(candidate, camera, stop=guard):
            if stop.is_set():
                return
            received = guard.started = True
            yield transport, frame
        if stop.is_set():
            return
        if received:
            return
    raise TimeoutError('No live frames received; no media retained')


def observe(config, camera, model, detector, vehicle_model, state, stop):
    import cv2
    from .ctc_fusion import CTCPlateReader
    from .ocr_selection_benchmark import box_iou
    from .plate_quality import crop_quality
    reader = CTCPlateReader(model.tokens)
    previous_box = previous_stamp = last_pts = None
    decoded = sampled = plate_crops = 0
    last_crop = None
    for _, frame in viewer_frames(config, camera, stop):
        if stop.is_set():
            return
        decoded += 1
        if frame.discontinuity:
            reader = CTCPlateReader(model.tokens)
            previous_box = previous_stamp = last_pts = last_crop = None
        if frame.pts_ms is None or (last_pts is not None and frame.pts_ms - last_pts < 500):
            continue
        last_pts = frame.pts_ms
        sampled += 1
        started = time.monotonic()
        vehicles, plate = detect_regions(frame.image, vehicle_model, detector)
        view = frame.image.copy()
        status = 'no plate in this frame'
        masked = None
        for box in vehicles:
            cv2.rectangle(view, box[:2], box[2:], (90, 220, 90), 2)
            cv2.putText(view, 'vehicle', (box[0], max(18, box[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, .55, (90, 220, 90), 2)
        crop = crop_box(frame.image, plate) if plate is not None else None
        if crop is not None and crop.shape[1] >= 24 and crop.shape[0] >= 8:
            plate_crops += 1
            cv2.rectangle(view, plate[:2], plate[2:], (0, 210, 255), 2)
            cv2.putText(view, 'possible plate', (plate[0], max(18, plate[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 210, 255), 2)
            if previous_box is None or box_iou(previous_box, plate) < .5 or frame.pts_ms - previous_stamp > 1500:
                reader = CTCPlateReader(model.tokens)
            previous_box, previous_stamp = plate, frame.pts_ms
            reader.add(model.probabilities(crop), timestamp_ms=frame.pts_ms,
                       source_id=hashlib.sha256(crop.tobytes()).hexdigest(), quality=crop_quality(crop))
            status = 'collecting distinct frames'
            if reader.observation_count >= 3:
                candidates = reader.candidates()
                accepted = reader.accepted()
                status = 'accepted by model — NOT verified' if accepted else 'uncertain' if candidates else 'no complete candidate'
                if candidates:
                    masked = mask_plate(candidates[0].plate)
            last_crop = {'image': encoded(crop, '.png'), 'width': crop.shape[1], 'height': crop.shape[0],
                         'seen_at': time.monotonic(), 'observations': reader.observation_count,
                         'ocr_status': status, 'masked_plate': masked}
        now = time.monotonic()
        crop_display = dict(last_crop) if last_crop and now - last_crop['seen_at'] <= 10 else None
        if crop_display:
            crop_display['age_seconds'] = round(now - crop_display.pop('seen_at'), 1)
        state.publish({'camera_id': camera.camera_id, 'status': 'live', 'published_at': now,
                       'pts_ms': frame.pts_ms, 'frame': encoded(view, '.jpg'), 'crop': crop_display,
                       'vehicle_boxes': len(vehicles), 'plate_in_frame': bool(crop is not None and crop.shape[1] >= 24 and crop.shape[0] >= 8),
                       'decoded_frames': decoded, 'analysed_frames': sampled, 'plate_crops': plate_crops,
                       'processing_ms': round((now-started)*1000), 'ocr_status': status}, stop)


PAGE = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sentinel · Live detection lab</title><style>
*{box-sizing:border-box}body{margin:0;background:#09131e;color:#e1edf6;font:16px system-ui;padding:28px}main{max-width:1450px;margin:auto}
h1{margin:6px 0 10px;font-size:30px}.eyebrow{color:#63dbc0;font-size:12px;letter-spacing:2px}.muted{color:#9fb5c7}button{background:#163346;color:#e4f5ff;border:1px solid #3f6b82;border-radius:8px;padding:12px 18px;cursor:pointer;margin:8px 8px 8px 0}button.active{background:#215d51;border-color:#70e1c1}
.grid{display:grid;grid-template-columns:minmax(0,3fr) minmax(260px,1fr);gap:20px}.panel{background:#102230;border:1px solid #294354;border-radius:12px;padding:18px}.video{position:relative;min-height:350px;background:#030a10;display:grid;place-items:center}.video img{width:100%;max-height:70vh;object-fit:contain}.video p{position:absolute;background:#102230dd;padding:12px;border-radius:8px}.crop{min-height:120px;display:grid;place-items:center;background:#060e17;overflow:auto}.crop img{image-rendering:pixelated;max-width:100%}dl{display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:14px}dt{color:#9fb5c7}dd{margin:0;text-align:right}.legend{font-size:14px;margin:14px 0}.green{color:#71dd78}.yellow{color:#ffda5a}.badge{font-size:13px;background:#183448;padding:5px 9px;border-radius:6px}#ocr{line-height:1.6}.note{line-height:1.6;font-size:13px}#message{color:#ffd67d;margin-bottom:12px}@media(max-width:850px){.grid{grid-template-columns:1fr}body{padding:16px}}
</style><main><span class="eyebrow">LOCAL · READ-ONLY DIAGNOSTIC</span><h1>Live detection lab</h1>
<p class="muted">See exactly which frame the models analysed. This is sampled inference, not full-frame-rate playback.</p>
<div><button data-camera="cam01">CAM01 · Chiman bhai Bridge</button><button data-camera="cam02">CAM02 · Janpath</button><button id="pause">Pause</button><span id="state" class="badge">Ready</span></div>
<div class="legend"><span class="green">Green: detected vehicle</span> · <span class="yellow">Yellow: possible plate</span> · Boxes are model predictions, not verified labels.</div>
<div id="message">Choose a camera to begin. Nothing is recorded.</div><div class="grid"><section class="panel video"><img id="frame" alt="Selected live camera with aligned detection boxes" hidden><p id="empty">Waiting for a selected camera</p></section>
<aside class="panel"><h2>Last native plate crop</h2><div class="crop"><img id="crop" alt="Last detected plate crop, native pixels" hidden><span id="nocrop" class="muted">No plate crop yet</span></div>
<p id="cropinfo" class="muted"></p><p id="ocr">No OCR observations yet.</p><dl><dt>Frames decoded</dt><dd id="decoded">0</dd><dt>Frames analysed</dt><dd id="analysed">0</dd><dt>Plate crops</dt><dd id="plates">0</dd><dt>Vehicle boxes now</dt><dd id="vehicles">0</dd><dt>Processing</dt><dd id="timing">—</dd><dt>Frame age</dt><dd id="age">—</dd></dl>
<p class="note muted">Fine-tuned OCR + distinct-frame CTC fusion. Plate association here uses conservative box overlap, not the full production tracker. A model-accepted reading is not proof of correctness. OCR output is masked. The crop is lossless PNG; display enlargement adds no information. Full-frame display uses JPEG.</p></aside></div>
<p class="note muted">No recording, downloads, saved plate text or central alerts. Images stay in server/browser memory. One camera at a time. Pauses after 60 seconds without a browser heartbeat; lab shuts down after 30 minutes.</p></main>
<script>
const token=__TOKEN__, $=id=>document.getElementById(id);let camera=null,busy=false,sessionError='';
async function select(id){const response=await fetch('/session',{method:'POST',headers:{'Content-Type':'application/json','X-Viewer-Token':token},body:JSON.stringify({camera_id:id}),cache:'no-store'});if(!response.ok)throw Error('Session unavailable');camera=id;document.querySelectorAll('[data-camera]').forEach(b=>b.classList.toggle('active',b.dataset.camera===id));$('frame').hidden=true;$('crop').hidden=true;$('empty').hidden=false;$('empty').textContent=id?'Connecting…':'Paused';}
document.querySelectorAll('[data-camera]').forEach(b=>b.onclick=()=>{sessionError='';select(b.dataset.camera).catch(()=>{sessionError='Cannot start the local session. Refresh this page and retry.';$('message').textContent=sessionError;});});$('pause').onclick=()=>select(null).catch(()=>{sessionError='Cannot pause the local session.';});
async function refresh(){if(busy)return;busy=true;try{const r=await fetch('/snapshot',{headers:{'X-Viewer-Token':token},cache:'no-store'});if(!r.ok)throw Error();const s=await r.json();$('state').textContent=s.status;$('message').textContent=s.status==='live'?(s.frame_age_seconds>5?'Feed stale/reconnecting — the displayed frame is not current.':s.ocr_status):s.status==='unavailable'?'Live access or models unavailable. No footage was saved.':s.status==='connecting'?'Connecting/loading models…':'Choose a camera to begin.';
if(sessionError)$('message').textContent=sessionError;if(s.camera_id!==camera){if(s.status==='paused'){camera=null;$('frame').hidden=true;$('crop').hidden=true;$('empty').hidden=false;$('empty').textContent='Paused — select a camera to restart';document.querySelectorAll('[data-camera]').forEach(b=>b.classList.remove('active'));}return;}if(s.frame){$('frame').src='data:image/jpeg;base64,'+s.frame;$('frame').hidden=false;$('empty').hidden=true;}else if(s.status==='unavailable'){$('frame').hidden=true;$('empty').hidden=false;$('empty').textContent='No live frames available — retrying safely';}$('decoded').textContent=s.decoded_frames||0;$('analysed').textContent=s.analysed_frames||0;$('plates').textContent=s.plate_crops||0;$('vehicles').textContent=s.vehicle_boxes||0;$('timing').textContent=s.processing_ms==null?'—':s.processing_ms+' ms';$('age').textContent=s.frame_age_seconds==null?'—':s.frame_age_seconds+' s';
if(s.crop){$('crop').src='data:image/png;base64,'+s.crop.image;$('crop').style.width=s.crop.width*3+'px';$('crop').hidden=false;$('nocrop').hidden=true;$('cropinfo').textContent=s.crop.width+' × '+s.crop.height+' native pixels · last detection '+(s.crop.age_seconds+s.frame_age_seconds).toFixed(1)+' s ago';$('ocr').textContent=s.crop.ocr_status+' · '+s.crop.observations+' distinct observations'+(s.crop.masked_plate?' · '+s.crop.masked_plate:'');}else{$('crop').hidden=true;$('nocrop').hidden=false;$('cropinfo').textContent='';$('ocr').textContent='No recent plate crop.';}
}catch{$('message').textContent='Viewer unavailable or its 30-minute session ended.';}finally{busy=false;}}setInterval(refresh,750);setInterval(()=>{if(camera&&document.visibilityState==='visible')fetch('/session',{method:'POST',headers:{'Content-Type':'application/json','X-Viewer-Token':token},body:JSON.stringify({camera_id:camera}),cache:'no-store'}).catch(()=>{});},20000);refresh();
</script></html>'''


def serve(config, cache, checkpoint, port=7780, lifetime=1800, bind_host='127.0.0.1'):
    state, shutdown = LiveState(), threading.Event()
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # Never log live URLs, media or OCR text.

        def reply(self, code, body, content_type='application/json'):
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store, max-age=0')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def allowed(self, mutation=False):
            valid = safe_request(self.headers.get('Host'), self.headers.get('Origin'), mutation, port)
            if self.path != '/' or mutation:
                valid = valid and secrets.compare_digest(self.headers.get('X-Viewer-Token', ''), token)
            if not valid:
                self.reply(403, b'{"error":"Local viewer access rejected"}')
            return valid

        def do_GET(self):
            if not self.allowed():
                return
            if self.path == '/':
                self.reply(200, PAGE.replace('__TOKEN__', json.dumps(token)).encode(), 'text/html; charset=utf-8')
            elif self.path == '/snapshot':
                self.reply(200, json.dumps(state.snapshot()).encode())
            else:
                self.reply(404, b'{}')

        def do_POST(self):
            if not self.allowed(True):
                return
            if self.path != '/session':
                self.reply(404, b'{}'); return
            try:
                self.connection.settimeout(5)
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 256:
                    raise ValueError()
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict) or set(data) != {'camera_id'}:
                    raise ValueError()
                state.select(data['camera_id'])
                self.reply(200, b'{"ok":true}')
            except (ValueError, TypeError, KeyError):
                self.reply(400, b'{"error":"Select cam01, cam02 or null"}')

    def work():
        model = detector = vehicle = cameras = None
        while not shutdown.is_set():
            with state.lock:
                selected, stop = state.camera_id, state.capture_stop
            if not selected:
                shutdown.wait(.25); continue
            try:
                if model is None:
                    import av
                    av.logging.set_level(av.logging.PANIC)
                    from ultralytics import YOLO
                    from .awiros_experiment import AwirosExperiment
                    from .ocr_selection_benchmark import PlateDetectorExperiment
                    from .cameras import discover_live_pair
                    cameras, _ = discover_live_pair(config)
                    cameras = {c.camera_id: c for c in cameras}
                    detector = PlateDetectorExperiment()
                    vehicle = YOLO(str(config.model_cache_dir/'yolov8n.pt'))
                    model = AwirosExperiment(cache, device='gpu', checkpoint=checkpoint)
                observe(config, cameras[selected], model, detector, vehicle, state, stop)
            except Exception:
                state.publish({'camera_id': selected, 'status': 'unavailable'}, stop)
                stop.wait(5)  # Redact third-party exceptions; retry, never tight-loop.

    server = ThreadingHTTPServer((bind_host, port), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=work, daemon=True)
    thread.start()

    def lifecycle():
        deadline = time.monotonic() + lifetime
        while not shutdown.wait(1):
            state.expire()
            if time.monotonic() >= deadline:
                server.shutdown(); return
    threading.Thread(target=lifecycle, daemon=True).start()
    print(f'Local live detection viewer ready on port {port}; no recording; lifetime {lifetime}s', flush=True)
    try:
        server.serve_forever()
    finally:
        shutdown.set(); state.select(None); server.server_close(); thread.join(timeout=10)


if __name__ == '__main__':
    from .config import Config
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--container-bind', action='store_true', help='Use only with a Docker loopback-only port publication')
    args = parser.parse_args()
    serve(Config.from_env(live=True), args.cache, args.checkpoint,
          bind_host='0.0.0.0' if args.container_bind else '127.0.0.1')
