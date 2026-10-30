"""Opt-in local diagnostic: aligned live overlays, no recording or central writes.

Frames and plate text remain in process/browser memory. HTTP responses are
non-cacheable. Masked OCR is unverified; plate boxes may be false positives.
One selected camera is decoded; browser heartbeat expires after 60 seconds.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import statistics
import secrets
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from collections import deque
from urllib.parse import parse_qs, urlsplit

from .privacy import mask_plate


LIVE_STALE_SECONDS = 5
# A recording is analysed as fast as this machine allows, which on a CPU can be
# several seconds per sample. Its evidence is the current position, not a feed.
RECORDED_STALE_SECONDS = 30


class LiveState:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.RLock()
        self.camera_id = None
        self.capture_stop = threading.Event()
        self.last_heartbeat = 0.
        self.value = {'status': 'paused', 'camera_id': None}
        self.allowed_ids = {'cam01', 'cam02'}
        # Permitted local recordings, selectable alongside cameras but always
        # labelled `recorded`. Playback state belongs to the one shared session.
        self.recordings = {}
        self.paused = threading.Event()
        self.start_ms = 0
        self.session_id = None

    def _selectable(self, camera_id):
        return camera_id in self.allowed_ids or camera_id in self.recordings

    def authorise(self, camera_ids):
        with self.lock:
            self.allowed_ids = set(camera_ids)
            if self.camera_id and not self._selectable(self.camera_id):
                self.select(None)

    def authorise_recordings(self, recordings):
        with self.lock:
            self.recordings = dict(recordings)
            if self.camera_id and not self._selectable(self.camera_id):
                self.select(None)

    def _begin(self, camera_id, start_ms=0):
        """Start a new capture generation; the old one stops and its evidence goes."""
        self.capture_stop.set()
        self.capture_stop = threading.Event()
        self.camera_id = camera_id
        self.start_ms = start_ms
        self.paused.clear()
        self.session_id = None
        if camera_id in self.recordings:
            from .recorded_source import new_session_id
            self.session_id = new_session_id()
        self.value = {'status': 'connecting' if camera_id else 'paused', 'camera_id': camera_id}

    def select(self, camera_id):
        with self.lock:
            if camera_id is not None and not self._selectable(camera_id):
                raise ValueError('Select a live camera from the authorised catalogue')
            self.last_heartbeat = self.clock()
            if camera_id == self.camera_id:
                return
            self._begin(camera_id)

    def control(self, action, position_ms=None):
        """Playback control for the selected recording only; cameras have none."""
        with self.lock:
            recording = self.recordings.get(self.camera_id)
            if recording is None:
                raise ValueError('Playback controls need a selected recording')
            if action in ('pause', 'resume', 'restart'):
                if position_ms is not None:
                    raise ValueError('Only seek takes a position')
            elif action == 'seek':
                if (isinstance(position_ms, bool) or not isinstance(position_ms, int) or position_ms < 0
                        or (recording.duration_ms is not None and position_ms > recording.duration_ms)):
                    raise ValueError('Seek position is outside the recording')
            else:
                raise ValueError('Unknown playback action')
            self.last_heartbeat = self.clock()
            if action == 'pause':
                if self.value.get('status') != 'completed':
                    self.paused.set()
            elif action == 'resume':
                self.paused.clear()
            else:
                # Restart and seek are new runs: tracking, temporal evidence and
                # reconstruction restart clean under a new session ID.
                self._begin(self.camera_id, position_ms if action == 'seek' else 0)

    def complete(self, capture_stop):
        """Mark end of file for the generation that reached it, keeping its last evidence."""
        with self.lock:
            if capture_stop is self.capture_stop and self.camera_id in self.recordings:
                self.paused.clear()
                self.value = dict(self.value, status='completed', camera_id=self.camera_id)

    def run_parameters(self):
        with self.lock:
            return (self.camera_id, self.capture_stop, self.recordings.get(self.camera_id),
                    self.session_id, self.start_ms)

    def expire(self):
        with self.lock:
            if self.camera_id and self.clock() - self.last_heartbeat > 60:
                self.select(None)

    def publish(self, value, capture_stop=None):
        with self.lock:
            if (self.camera_id == value.get('camera_id') and
                    (capture_stop is None or capture_stop is self.capture_stop)):
                self.value = value

    def heartbeat(self, camera_id):
        with self.lock:
            if camera_id is None or camera_id != self.camera_id:
                raise ValueError('The shared camera session changed')
            self.last_heartbeat = self.clock()

    def snapshot(self, *, engineering=False):
        with self.lock:
            result = dict(self.value)
            recording = self.recordings.get(self.camera_id)
            limit = RECORDED_STALE_SECONDS if recording else LIVE_STALE_SECONDS
            if not engineering:
                result.pop('debug', None)
            # A held recording has not moved on: its last frame is still the
            # current position. Idle expiry, not age, clears it.
            held = recording is not None and (self.paused.is_set() or result.get('status') == 'completed')
            if result.get('published_at'):
                result['frame_age_seconds'] = 0. if held else round(self.clock() - result['published_at'], 1)
                if result['frame_age_seconds'] > limit:
                    # Do not transmit old sensitive evidence even if a caller
                    # bypasses the browser's own freshness checks.
                    result.pop('frame',None); result.pop('crop',None); result.pop('debug',None)
                    if result.get('status') == 'live':
                        result['status'] = 'unavailable'
            result['source_mode'] = 'recorded' if recording else 'live'
            result['stale_after_seconds'] = limit
            if recording:
                result.update(session_id=self.session_id, recording_name=recording.name,
                              video_duration_ms=recording.duration_ms,
                              time_basis=recording.public()['time_basis'])
                if self.paused.is_set() and result.get('status') in ('processing', 'connecting'):
                    result['status'] = 'paused'
            return result

    def operator_snapshot(self):
        from .recognition_contract import operator_snapshot
        return operator_snapshot(self.snapshot(engineering=True))


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


def roi_schedule(count, offset, budget=None):
    """Choose eligible vehicle regions; None means search every region this frame."""
    if not count:
        return []
    limit = count if budget is None else min(count, budget)
    return [(offset + i) % count for i in range(limit)]


def detect_regions(image, vehicle_model, detector, *, roi_offset=0):
    result = vehicle_model.predict(image, imgsz=640, classes=[2, 3, 5, 7], conf=.25,
                                   device='cpu', verbose=False)[0]
    ranked = sorted(zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()), key=lambda v: -v[1])
    vehicles = [clipped_box(image, box) for box, _ in ranked]
    plate = detector.box(image)
    if plate is None:
        for index in roi_schedule(len(vehicles), roi_offset):
            box = vehicles[index]
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
    if camera is not None and not camera.rtsp_url and not camera.hls_url:
        from .diagnostic_gateway import frames
        guard = StartupStop(stop)
        for transport, frame in frames(config, camera, guard):
            guard.started = True
            yield transport, frame
        return
    # Reuse the production pair's relay to avoid extra source copies. After
    # bounded relay failures, try the authorised direct RTSP/HLS sources.
    if camera is not None and camera.camera_id not in getattr(config, 'camera_ids', ('cam01', 'cam02')):
        config = replace(config, media_relay_base_url='')
    options = ([config,replace(config,media_relay_base_url='')]
               if config.media_relay_base_url else [config])
    for candidate in options:
        guard = StartupStop(stop)
        received = False
        for transport, frame in read_camera(candidate, camera, stop=guard,
                retry_limit=3 if candidate is config and len(options)>1 else None):
            if stop.is_set():
                return
            received = guard.started = True
            yield transport, frame
        if stop.is_set():
            return
        if received:
            return
    if camera is not None:
        # A separately authorised preview can reuse the gateway's established
        # session when a new direct decoder session receives no first frame.
        from .diagnostic_gateway import frames
        guard = StartupStop(stop)
        for transport, frame in frames(config, camera, guard):
            guard.started = True
            yield transport, frame
        return
    raise TimeoutError('No live frames received; no media retained')


def load_models(config, cache, checkpoint, *, detector_profile=None, training_root=None, device='gpu'):
    # Paddle's isolated CUDA 12 libraries must initialize before importing
    # Torch/YOLO (the production image bundles a different CUDA/cuDNN version).
    import numpy as np
    from .awiros_experiment import AwirosExperiment
    # The OCR device is an explicit choice. A GPU request on a machine without
    # one fails visibly in AwirosExperiment; it is never downgraded to CPU.
    if device not in ('cpu', 'gpu'):
        raise ValueError('Explicit CPU/GPU device required')
    reader = AwirosExperiment(cache, device=device, checkpoint=checkpoint)
    reader.probabilities(np.zeros((32, 120, 3), dtype=np.uint8))
    if detector_profile is not None:
        from .experimental_detectors import PROFILES, PlateDetector, VehicleTracker
        if detector_profile not in PROFILES:
            raise ValueError('Unknown experimental detector profile')
        vehicle, plate = PROFILES[detector_profile]
        # Torch preloads its matching CUDA libraries before ONNX constructs a
        # CUDA session. Paddle's isolated runtime must still warm up first.
        tracker = VehicleTracker(vehicle, config.model_cache_dir,
                                 training_root or Path('/training'), device=device)
        return reader, PlateDetector(plate, config.model_cache_dir, device=device), tracker
    from ultralytics import YOLO
    from .ocr_selection_benchmark import PlateDetectorExperiment
    return reader, PlateDetectorExperiment(), YOLO(str(config.model_cache_dir/'yolov8n.pt'))


def dashboard_view(snapshot, width=1280, rows=30, *, mode='overlays'):
    """Analysed frame (shrunk for transport) and vehicles seen, for a recording only."""
    import cv2
    import numpy as np
    if snapshot.get('source_mode') != 'recorded':
        return None
    if mode not in ('original', 'overlays'):
        raise ValueError('Unknown frame display mode')
    frame = snapshot.get('debug', {}).get('original_frame') if mode == 'original' else snapshot.get('frame')
    if frame:
        image = cv2.imdecode(np.frombuffer(base64.b64decode(frame), np.uint8), cv2.IMREAD_COLOR)
        if image is not None and image.shape[1] > width:
            image = cv2.resize(image, (width, round(image.shape[0]*width/image.shape[1])), interpolation=cv2.INTER_AREA)
        ok, data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 78]) if image is not None else (False, None)
        frame = base64.b64encode(data).decode('ascii') if ok else None
    return {'source_mode': 'recorded', 'session_id': snapshot.get('session_id'), 'status': snapshot.get('status'),
            'video_position_ms': snapshot.get('video_position_ms'), 'frame': frame or None,
            'vehicles': snapshot.get('debug', {}).get('passed', [])[:rows]}


def model_identity(reader, detector_profile=None, *, detector=None, vehicle_model=None):
    """What actually loaded, for diagnostics. Names and hashes only, no paths."""
    from .experimental_detectors import HASHES, PROFILES
    vehicle, plate = PROFILES.get(detector_profile, ('yolov8n', 'fast-alpr'))
    adaptation = getattr(reader, 'adaptation', None) or {}
    # Older unprofiled diagnostics still run their detectors on CPU. For tracked
    # profiles, report the adapters actually loaded, not the requested setting.
    def actual_device(adapter):
        # Unprofiled YOLO exposes a torch.device, but its legacy inference call
        # explicitly uses CPU. Metadata must remain JSON-safe strings.
        if not detector_profile:
            return 'cpu'
        value = getattr(adapter, 'device', None)
        return value if isinstance(value, str) and value in ('cpu', 'gpu') else 'unknown'
    plate_device = actual_device(detector)
    vehicle_device = actual_device(vehicle_model)
    return {'ocr_model': 'awiros-ctc-adapted' if adaptation else 'awiros-ctc-parent',
            'ocr_checkpoint_sha256': adaptation.get('weights_sha256'),
            'ocr_device': getattr(reader, 'device', None),
            'detector_profile': detector_profile or 'legacy',
            'vehicle_model': vehicle, 'vehicle_model_sha256': HASHES.get(vehicle),
            'plate_model': plate, 'plate_model_sha256': HASHES.get(plate),
            'plate_device': plate_device, 'vehicle_device': vehicle_device,
            'detector_device': plate_device if plate_device == vehicle_device else 'mixed'}


def observe(config, camera, model, detector, vehicle_model, state, stop, duration_seconds=None):
    import cv2
    from .ctc_fusion import CTCPlateReader
    from .ocr_selection_benchmark import box_iou
    from .plate_quality import crop_quality
    reader = CTCPlateReader(model.tokens)
    previous_box = previous_stamp = last_pts = None
    decoded = sampled = plate_crops = 0
    vehicle_frames = peak_vehicles = total_boxes = eligible = candidate_windows = max_observations = max_span = 0
    first_at = sequence_start = None
    widths, qualities, vehicle_widths, timings = (deque(maxlen=256) for _ in range(4))
    accepted_ids, secret = set(), secrets.token_bytes(32)
    last_crop = None
    for _, frame in viewer_frames(config, camera, stop):
        if stop.is_set():
            return
        if first_at is None:
            first_at = time.monotonic()
        if duration_seconds is not None and time.monotonic()-first_at >= duration_seconds:
            return
        decoded += 1
        if frame.discontinuity:
            reader = CTCPlateReader(model.tokens)
            previous_box = previous_stamp = last_pts = last_crop = None
            sequence_start = None
        if frame.pts_ms is None or (last_pts is not None and frame.pts_ms - last_pts < 500):
            continue
        last_pts = frame.pts_ms
        sampled += 1
        started = time.monotonic()
        vehicles, plate = detect_regions(frame.image, vehicle_model, detector, roi_offset=(sampled-1)*3)
        vehicle_frames += bool(vehicles)
        total_boxes += len(vehicles)
        peak_vehicles = max(peak_vehicles, len(vehicles))
        vehicle_widths.extend(box[2]-box[0] for box in vehicles)
        view = frame.image.copy()
        status = 'no plate in this frame'
        masked = None
        for box in vehicles:
            cv2.rectangle(view, box[:2], box[2:], (90, 220, 90), 2)
            cv2.putText(view, 'vehicle', (box[0], max(18, box[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, .55, (90, 220, 90), 2)
        crop = crop_box(frame.image, plate) if plate is not None else None
        if crop is not None and crop.shape[1] >= 24 and crop.shape[0] >= 8:
            plate_crops += 1
            widths.append(crop.shape[1]); qualities.append(crop_quality(crop))
            cv2.rectangle(view, plate[:2], plate[2:], (0, 210, 255), 2)
            cv2.putText(view, 'possible plate', (plate[0], max(18, plate[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 210, 255), 2)
            if previous_box is None or box_iou(previous_box, plate) < .5 or frame.pts_ms - previous_stamp > 1500:
                reader = CTCPlateReader(model.tokens)
                sequence_start = frame.pts_ms
            previous_box, previous_stamp = plate, frame.pts_ms
            reader.add(model.probabilities(crop), timestamp_ms=frame.pts_ms,
                       source_id=hashlib.sha256(crop.tobytes()).hexdigest(), quality=crop_quality(crop))
            status = 'collecting distinct frames'
            max_observations = max(max_observations, reader.observation_count)
            max_span = max(max_span, frame.pts_ms-sequence_start)
            if reader.observation_count >= 3:
                eligible += 1
                candidates = reader.candidates()
                accepted = reader.accepted()
                candidate_windows += bool(candidates)
                if accepted:
                    accepted_ids.add(hmac.digest(secret, accepted.plate.encode(), 'sha256'))
                status = 'accepted by model — NOT verified' if accepted else 'uncertain' if candidates else 'no complete candidate'
                if candidates:
                    masked = mask_plate(candidates[0].plate)
            last_crop = {'image': encoded(crop, '.png'), 'width': crop.shape[1], 'height': crop.shape[0],
                         'seen_at': time.monotonic(), 'observations': reader.observation_count,
                         'ocr_status': status, 'masked_plate': masked}
        now = time.monotonic()
        timings.append((now-started)*1000)
        crop_display = dict(last_crop) if last_crop and now - last_crop['seen_at'] <= 10 else None
        if crop_display:
            crop_display['age_seconds'] = round(now - crop_display.pop('seen_at'), 1)
        state.publish({'camera_id': camera.camera_id, 'status': 'live', 'published_at': now,
                       'pts_ms': frame.pts_ms, 'frame': encoded(view, '.jpg'), 'crop': crop_display,
                       'vehicle_boxes': len(vehicles), 'plate_in_frame': bool(crop is not None and crop.shape[1] >= 24 and crop.shape[0] >= 8),
                       'decoded_frames': decoded, 'analysed_frames': sampled, 'plate_crops': plate_crops,
                       'resolution': [frame.image.shape[1], frame.image.shape[0]],
                       'observed_seconds': round(now-first_at, 2), 'vehicle_frames': vehicle_frames,
                       'vehicle_box_instances': total_boxes, 'peak_vehicle_boxes': peak_vehicles,
                       'median_vehicle_width': statistics.median(vehicle_widths) if vehicle_widths else None,
                       'median_plate_width': statistics.median(widths) if widths else None,
                       'median_crop_quality': round(statistics.median(qualities), 3) if qualities else None,
                       'maximum_distinct_observations': max_observations, 'maximum_plate_span_ms': max_span,
                       'eligible_fusion_windows': eligible, 'candidate_windows': candidate_windows,
                       'unique_model_accepted_unverified': len(accepted_ids),
                       'median_processing_ms': round(statistics.median(timings)),
                       'processing_ms': round((now-started)*1000), 'ocr_status': status}, stop)


METRIC_FIELDS = ('camera_id', 'status', 'resolution', 'decoded_frames', 'analysed_frames', 'plate_crops',
                 'observed_seconds', 'vehicle_frames', 'vehicle_box_instances', 'peak_vehicle_boxes',
                 'median_vehicle_width', 'median_plate_width', 'median_crop_quality',
                 'maximum_distinct_observations', 'maximum_plate_span_ms', 'eligible_fusion_windows',
                 'candidate_windows', 'unique_model_accepted_unverified', 'median_processing_ms')


def safe_metrics(snapshot):
    return {field: snapshot[field] for field in METRIC_FIELDS if field in snapshot}


PAGE = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sentinel · Live detection lab</title><style>
*{box-sizing:border-box}body{margin:0;background:#09131e;color:#e1edf6;font:16px system-ui;padding:28px}main{max-width:1450px;margin:auto}
h1{margin:6px 0 10px;font-size:30px}.eyebrow{color:#63dbc0;font-size:12px;letter-spacing:2px}.muted{color:#9fb5c7}button{background:#163346;color:#e4f5ff;border:1px solid #3f6b82;border-radius:8px;padding:12px 18px;cursor:pointer;margin:8px 8px 8px 0}button.active{background:#215d51;border-color:#70e1c1}
.grid{display:grid;grid-template-columns:minmax(0,3fr) minmax(260px,1fr);gap:20px;align-items:start}.panel{background:#102230;border:1px solid #294354;border-radius:12px;padding:18px}.video{position:relative;min-height:220px;aspect-ratio:16/9;background:#030a10;display:grid;place-items:center}.video img{width:100%;height:100%;max-height:70vh;object-fit:contain}.video p{position:absolute;background:#102230dd;padding:12px;border-radius:8px}.crop{min-height:120px;display:grid;place-items:center;background:#060e17;overflow:auto}.crop img{image-rendering:pixelated;max-width:100%}dl{display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:14px}dt{color:#9fb5c7}dd{margin:0;text-align:right}.legend{font-size:14px;margin:14px 0}.green{color:#71dd78}.yellow{color:#ffda5a}.badge{font-size:13px;background:#183448;padding:5px 9px;border-radius:6px}#ocr{line-height:1.6}.note{line-height:1.6;font-size:13px}#message{color:#ffd67d;margin-bottom:12px}@media(max-width:850px){.grid{grid-template-columns:1fr}body{padding:16px}}
</style><main><span class="eyebrow">LOCAL · READ-ONLY DIAGNOSTIC</span><h1>Live detection lab</h1>
<p class="muted">See exactly which frame the models analysed. This is sampled inference, not full-frame-rate playback.</p>
<div><button data-camera="cam01">CAM01 · difficult baseline</button><button data-camera="cam02">CAM02 · difficult baseline</button><button id="pause">Pause</button><span id="state" class="badge">Ready</span></div>
<div><label for="camera-select">Authorised test camera </label><select id="camera-select" style="background:#163346;color:#e4f5ff;padding:10px;max-width:100%" disabled><option>Loading catalogue…</option></select><button id="open-camera" disabled>Open selected camera</button><span id="catalogue-status" class="muted"></span></div>
<div class="legend"><span class="green">Green: detected vehicle</span> · <span class="yellow">Yellow: possible plate</span> · Boxes are model predictions, not verified labels.</div>
<div id="message">Choose a camera to begin. Nothing is recorded.</div><div class="grid"><section class="panel video"><img id="frame" alt="Selected live camera with aligned detection boxes" hidden><p id="empty">Waiting for a selected camera</p></section>
<aside class="panel"><h2>Last native plate crop</h2><div class="crop"><img id="crop" alt="Last detected plate crop, native pixels" hidden><span id="nocrop" class="muted">No plate crop yet</span></div>
<p id="cropinfo" class="muted"></p><p id="ocr">No OCR observations yet.</p><dl><dt>Native resolution</dt><dd id="resolution">—</dd><dt>Frames decoded</dt><dd id="decoded">0</dd><dt>Frames analysed</dt><dd id="analysed">0</dd><dt>Plate crops</dt><dd id="plates">0</dd><dt>Vehicle boxes now (all)</dt><dd id="vehicles">0</dd><dt>Peak vehicle boxes</dt><dd id="peak">0</dd><dt>Max distinct plate frames</dt><dd id="observations">0</dd><dt>Candidate windows</dt><dd id="candidates">0</dd><dt>Accepted identifiers (unverified)</dt><dd id="accepted">0</dd><dt>Processing</dt><dd id="timing">—</dd><dt>Frame age</dt><dd id="age">—</dd></dl>
<p class="note muted">All vehicle boxes returned by the detector are drawn. This older no-profile mode searches the full frame, then each detected vehicle crop until the first plate proposal; at most one plate is passed to its OCR history per sample. The tracked profile uses ByteTrack vehicle IDs, searches every eligible tracked vehicle crop and keeps independent OCR histories. In either mode, missed vehicle detections and low-quality plates remain possible. Fine-tuned OCR + distinct-frame CTC fusion. A model-accepted reading is not proof of correctness. OCR output is masked. The crop is lossless PNG; display enlargement adds no information. Full-frame display uses JPEG.</p></aside></div>
<p class="note muted">No recording, downloads, saved plate text or central alerts. Images stay in server/browser memory. One camera at a time. Pauses after 60 seconds without a browser heartbeat; lab shuts down after 30 minutes.</p></main>
<script>
const token=__TOKEN__, $=id=>document.getElementById(id);let camera=null,busy=false,sessionError='';
async function select(id){const response=await fetch('/session',{method:'POST',headers:{'Content-Type':'application/json','X-Viewer-Token':token},body:JSON.stringify({camera_id:id}),cache:'no-store'});if(!response.ok)throw Error('Session unavailable');camera=id;document.querySelectorAll('[data-camera]').forEach(b=>b.classList.toggle('active',b.dataset.camera===id));$('frame').hidden=true;$('crop').hidden=true;$('empty').hidden=false;$('empty').textContent=id?'Connecting…':'Paused';}
document.querySelectorAll('[data-camera]').forEach(b=>b.onclick=()=>{sessionError='';select(b.dataset.camera).catch(()=>{sessionError='Cannot start the local session. Refresh this page and retry.';$('message').textContent=sessionError;});});$('pause').onclick=()=>select(null).catch(()=>{sessionError='Cannot pause the local session.';});
async function catalogue(){try{const r=await fetch('/cameras',{headers:{'X-Viewer-Token':token},cache:'no-store'});if(!r.ok)throw Error();const d=await r.json();$('camera-select').replaceChildren();for(const c of d.cameras){const o=document.createElement('option');o.value=c.id;o.textContent=c.id.toUpperCase()+' · '+c.name;$('camera-select').append(o);}if(d.cameras.some(c=>c.id==='cam12'))$('camera-select').value='cam12';$('camera-select').disabled=!d.cameras.length;$('open-camera').disabled=!d.cameras.length;$('catalogue-status').textContent=d.available?d.cameras.length+' live catalogue entries':'Catalogue unavailable · original pair only';if(!d.available)setTimeout(catalogue,15000);}catch{$('catalogue-status').textContent='Catalogue unavailable';setTimeout(catalogue,15000);}}$('open-camera').onclick=()=>{sessionError='';select($('camera-select').value).catch(()=>{sessionError='Cannot open this authorised camera.';});};catalogue();
async function refresh(){if(busy)return;busy=true;try{const r=await fetch('/snapshot',{headers:{'X-Viewer-Token':token},cache:'no-store'});if(!r.ok)throw Error();const s=await r.json();$('state').textContent=(s.camera_id?s.camera_id.toUpperCase()+' · ':'')+s.status;$('message').textContent=s.status==='live'?(s.frame_age_seconds>5?'Feed stale/reconnecting — the displayed frame is not current.':s.ocr_status):s.status==='unavailable'?'Live access or models unavailable. No footage was saved.':s.status==='connecting'?'Connecting/loading models…':'Choose a camera to begin.';
if(sessionError)$('message').textContent=sessionError;if(s.camera_id!==camera){if(s.status==='paused'){camera=null;$('frame').hidden=true;$('crop').hidden=true;$('empty').hidden=false;$('empty').textContent='Paused — select a camera to restart';document.querySelectorAll('[data-camera]').forEach(b=>b.classList.remove('active'));}return;}if(s.frame){$('frame').src='data:image/jpeg;base64,'+s.frame;$('frame').hidden=false;$('empty').hidden=true;}else if(s.status==='unavailable'){$('frame').hidden=true;$('empty').hidden=false;$('empty').textContent='No live frames available — retrying safely';}$('decoded').textContent=s.decoded_frames||0;$('analysed').textContent=s.analysed_frames||0;$('plates').textContent=s.plate_crops||0;$('vehicles').textContent=s.vehicle_boxes||0;$('timing').textContent=s.processing_ms==null?'—':s.processing_ms+' ms';$('age').textContent=s.frame_age_seconds==null?'—':s.frame_age_seconds+' s';
$('resolution').textContent=s.resolution?s.resolution.join(' × '):'—';$('peak').textContent=s.peak_vehicle_boxes||0;$('observations').textContent=s.maximum_distinct_observations||0;$('candidates').textContent=s.candidate_windows||0;$('accepted').textContent=s.unique_model_accepted_unverified||0;
if(s.crop){$('crop').src='data:image/png;base64,'+s.crop.image;$('crop').style.width=s.crop.width*3+'px';$('crop').hidden=false;$('nocrop').hidden=true;$('cropinfo').textContent=s.crop.width+' × '+s.crop.height+' native pixels · last detection '+(s.crop.age_seconds+s.frame_age_seconds).toFixed(1)+' s ago';$('ocr').textContent=s.crop.ocr_status+' · '+s.crop.observations+' distinct observations'+(s.crop.masked_plate?' · '+s.crop.masked_plate:'');}else{$('crop').hidden=true;$('nocrop').hidden=false;$('cropinfo').textContent='';$('ocr').textContent='No recent plate crop.';}
}catch{$('message').textContent='Viewer unavailable or its 30-minute session ended.';}finally{busy=false;}}setInterval(refresh,750);setInterval(()=>{if(camera&&document.visibilityState==='visible')fetch('/session',{method:'POST',headers:{'Content-Type':'application/json','X-Viewer-Token':token},body:JSON.stringify({camera_id:camera}),cache:'no-store'}).catch(()=>{});},20000);refresh();
</script></html>'''


def viewer_page(token,detector_profile=None):
    import html
    page = PAGE.replace('__TOKEN__',json.dumps(token))
    if detector_profile:
        page = page.replace('<h1>Live detection lab</h1>',
            '<h1>Live detection lab</h1><p class="muted">Experimental tracked detector profile: '+
            html.escape(detector_profile)+' · plate numbers remain unverified.</p>')
        page = page.replace('with at most one plate passed to OCR.',
            'with independent OCR evidence for each searched vehicle.')
        page = page.replace('Plate association here uses conservative box overlap, not the full production tracker.',
            'Association uses ByteTrack vehicle IDs plus geometry, class and stale-evidence guards.')
    return page


def configured_relay_cameras(config):
    from urllib.parse import urlsplit
    from .cameras import Camera, APPROVED_MANUAL_PAIRS
    base = getattr(config, 'media_relay_base_url', '')
    if (frozenset(getattr(config, 'camera_ids', ())) not in APPROVED_MANUAL_PAIRS or
            len(config.camera_ids) != 2 or not base):
        return {}
    parsed = urlsplit(base)
    if (parsed.scheme != 'rtsp' or parsed.hostname not in {'mediamtx','localhost','127.0.0.1'}
            or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ('','/')):
        return {}
    names = {'cam01': 'Chiman bhai Bridge · configured relay', 'cam02': 'Janpath · configured relay',
             'cam19':'CAM19 · configured relay', 'cam26':'CAM26 · configured relay', 'cam17':'CAM17 · configured relay'}
    return {key: Camera(key,names[key],rtsp_url=base.rstrip('/')+'/'+key) for key in config.camera_ids}


def serve(config, cache, checkpoint, port=7780, lifetime=1800, bind_host='127.0.0.1',
          detector_profile=None, training_root=None, operator_key=None, deliver=False,
          configured_relays=False, plate_recovery=False, device='gpu',
          recordings_dir=None, live_catalogue=True, paced=True, dashboard_evidence=False):
    if device not in ('cpu', 'gpu'):
        raise ValueError('Explicit CPU/GPU device required')
    if recordings_dir is not None and not detector_profile:
        raise ValueError('Recorded analysis needs a tracked detector profile')
    state, shutdown = LiveState(), threading.Event()
    identity = {}
    library = None
    recordings_view = {'recordings': [], 'rejected': []}
    if recordings_dir is not None:
        from .recorded_source import RecordingLibrary
        library = RecordingLibrary(recordings_dir)
    if not live_catalogue:
        state.authorise(())  # Recordings only: no camera is selectable.
    # Browser UI is provided only by the unified dashboard. Retain the
    # operator_key argument for launch/library compatibility, never publish it.
    catalogue = {'available': False, 'scope': 'unavailable', 'cameras': []}
    delivery = None
    if deliver:
        if config is None or not detector_profile:
            raise ValueError('Delivery requires a configured tracked detector profile')
        from .shared_delivery import SharedDelivery
        delivery = SharedDelivery(config, detector_profile)
        delivery.start(shutdown)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # Never log live URLs, media or OCR text.

        def reply(self, code, body, content_type='application/json', *, headers=None):
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store, max-age=0')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'self'; frame-ancestors 'none'")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def allowed(self, mutation=False):
            if self.path.startswith('/internal/'):
                expected = getattr(config, 'edge_api_key', '')
                valid = bool(expected) and secrets.compare_digest(self.headers.get('X-Edge-Key', ''), expected)
                if not valid:
                    self.reply(403, b'{"error":"Shared recognition access rejected"}')
                return valid
            self.reply(410, b'{"error":"Use the unified dashboard"}')
            return False

        def do_GET(self):
            parsed = urlsplit(self.path)
            path = parsed.path
            if path in ('/', '/engineering'):
                if not safe_request(self.headers.get('Host'), self.headers.get('Origin'), False, port):
                    self.reply(403, b'{"error":"Local viewer access rejected"}'); return
                self.reply(307, b'', headers={'Location': 'http://127.0.0.1:5173/'})
                return
            if not self.allowed():
                return
            if path == '/internal/diagnostics':
                from .unified_diagnostics import DiagnosticUnavailable, diagnostic_view
                query = parse_qs(parsed.query, keep_blank_values=True)
                if set(query) != {'session_id', 'track_id'} or any(len(values) != 1 for values in query.values()):
                    self.reply(400, b'{"error":"Invalid diagnostic identifiers"}'); return
                try:
                    body = diagnostic_view(state.snapshot(engineering=True), dict(identity),
                                           query['session_id'][0], query['track_id'][0], enabled=dashboard_evidence)
                except DiagnosticUnavailable as error:
                    self.reply(error.status, json.dumps({'error': error.detail}).encode()); return
                self.reply(200, json.dumps(body, allow_nan=False).encode())
            elif path == '/internal/evidence':
                # Opt-in, and for permitted recordings only: the analysed frame and
                # full unverified readings for the operator dashboard. A live
                # camera's imagery and plate text never leave this process.
                query = parse_qs(parsed.query, keep_blank_values=True)
                if set(query) - {'mode'} or len(query.get('mode', ['overlays'])) != 1 or query.get('mode', ['overlays'])[0] not in ('original', 'overlays'):
                    self.reply(400, b'{"error":"Invalid frame display mode"}'); return
                body = dashboard_view(state.snapshot(engineering=True), mode=query.get('mode', ['overlays'])[0]) if dashboard_evidence else None
                if body is None:
                    self.reply(404, b'{}')
                else:
                    self.reply(200, json.dumps(body).encode())
            elif path == '/internal/status':
                snapshot = state.operator_snapshot()
                snapshot['model_profile'] = detector_profile or 'legacy'
                snapshot['inference_device'] = identity.get('ocr_device')
                snapshot['delivery'] = {'enabled': delivery is not None, 'outbox_depth': delivery.depth() if delivery else 0}
                self.reply(200, json.dumps(snapshot).encode())
            elif path == '/internal/cameras':
                body = dict(catalogue, recordings=recordings_view['recordings'])
                self.reply(200, json.dumps(body).encode())
            else:
                self.reply(404, b'{}')

        def do_POST(self):
            if not self.allowed(True):
                return
            if self.path not in ('/session', '/heartbeat', '/internal/session', '/internal/heartbeat',
                                 '/control', '/internal/control'):
                self.reply(404, b'{}'); return
            try:
                self.connection.settimeout(5)
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 256:
                    raise ValueError()
                data = json.loads(self.rfile.read(length))
                if self.path.endswith('/control'):
                    if (not isinstance(data, dict) or not {'action'} <= set(data) <= {'action', 'position_ms'}
                            or not isinstance(data['action'], str)):
                        raise ValueError()
                    try:
                        state.control(data['action'], data.get('position_ms'))
                    except ValueError:
                        self.reply(409, b'{"error":"Playback control rejected"}'); return
                    if self.path.startswith('/internal/'):
                        snapshot = state.operator_snapshot()
                        snapshot['model_profile'] = detector_profile or 'legacy'
                        snapshot['inference_device'] = identity.get('ocr_device')
                        snapshot['delivery'] = {'enabled': delivery is not None, 'outbox_depth': delivery.depth() if delivery else 0}
                        self.reply(200, json.dumps(snapshot).encode())
                    else:
                        self.reply(200, b'{"ok":true}')
                    return
                if not isinstance(data, dict) or set(data) != {'camera_id'}:
                    raise ValueError()
                if not (data['camera_id'] is None or isinstance(data['camera_id'], str)):
                    raise ValueError()
                if self.path.endswith('/heartbeat'):
                    try:
                        state.heartbeat(data['camera_id'])
                    except ValueError:
                        self.reply(409, b'{"error":"The shared camera session changed"}'); return
                else:
                    state.select(data['camera_id'])
                if self.path.startswith('/internal/'):
                    snapshot = state.operator_snapshot()
                    snapshot['model_profile'] = detector_profile or 'legacy'
                    snapshot['inference_device'] = identity.get('ocr_device')
                    snapshot['delivery'] = {'enabled': delivery is not None, 'outbox_depth': delivery.depth() if delivery else 0}
                    self.reply(200, json.dumps(snapshot).encode())
                else:
                    self.reply(200, b'{"ok":true}')
            except (ValueError, TypeError, KeyError):
                self.reply(400, b'{"error":"Select an authorised source ID or null, or a valid playback action"}')

    def work():
        nonlocal catalogue
        model = detector = vehicle = cameras = None
        next_scan = next_discovery = 0.
        if not live_catalogue:
            cameras = {}
        while not shutdown.is_set():
            if library is not None and time.monotonic() >= next_scan:
                # Rescan so a newly registered file appears without a restart.
                next_scan = time.monotonic()+15
                found, rejected = library.scan()
                state.authorise_recordings(found)
                recordings_view['recordings'] = [item.public() for item in found.values()]
                recordings_view['rejected'] = rejected
            if cameras is None and config is not None and time.monotonic() >= next_discovery:
                try:
                    from .diagnostic_gateway import discover
                    cameras = {c.camera_id: c for c in discover(config) if c.live}
                    state.authorise(cameras)
                    catalogue = {'available': True, 'scope': 'authorised_catalogue',
                                 'cameras': [{'id': c.camera_id, 'name': c.name} for c in sorted(cameras.values(),
                                     key=lambda c:(c.camera_id not in config.camera_ids,
                                                   config.camera_ids.index(c.camera_id) if c.camera_id in config.camera_ids else 0,
                                                   c.camera_id))]}
                except Exception:
                    # Only opt-in use of the already approved/configured local
                    # pair. This is not a cached or fabricated full catalogue.
                    if configured_relays:
                        cameras = configured_relay_cameras(config) or None
                    if cameras:
                        state.authorise(cameras)
                        catalogue = {'available': True, 'scope': 'configured_relays',
                                     'cameras': [{'id': c.camera_id, 'name': c.name} for c in cameras.values()]}
                    else:
                        # Retry later without holding up a selected recording.
                        next_discovery = time.monotonic()+15
            selected, stop, recording, session_id, start_ms = state.run_parameters()
            if not selected:
                shutdown.wait(.25); continue
            if recording is None and (cameras is None or selected not in cameras):
                state.publish({'camera_id': selected, 'status': 'unavailable'}, stop)
                shutdown.wait(1); continue
            try:
                if model is None:
                    import av
                    av.logging.set_level(av.logging.PANIC)
                    model, detector, vehicle = load_models(config, cache, checkpoint,
                        detector_profile=detector_profile, training_root=training_root, device=device)
                    identity.update(model_identity(model, detector_profile,
                                                   detector=detector, vehicle_model=vehicle))
                if recording is not None:
                    from .recorded_source import RecordedRun
                    from .tracked_live_viewer import observe_tracked
                    run = RecordedRun(recording, session_id, start_ms, paced, state.paused)
                    observe_tracked(config, recording, model, detector, vehicle, state, stop,
                                    debug=True, delivery=delivery, detector_profile=detector_profile,
                                    plate_recovery=plate_recovery, recorded=run)
                    state.complete(stop)
                    # Hold the finished or stopped run until the operator acts.
                    while not stop.wait(1) and not shutdown.is_set():
                        pass
                elif detector_profile:
                    from .tracked_live_viewer import observe_tracked
                    observe_tracked(config, cameras[selected], model, detector, vehicle, state, stop,
                                    debug=True, delivery=delivery, detector_profile=detector_profile,
                                    plate_recovery=plate_recovery)
                else:
                    observe(config, cameras[selected], model, detector, vehicle, state, stop)
            except Exception as exc:
                # Class names are safe diagnostics; third-party messages may
                # contain authenticated stream URLs and must never leave RAM.
                code = getattr(exc,'failure_code',type(exc).__name__)
                state.publish({'camera_id': selected, 'status': 'unavailable',
                               'failure_code': code if isinstance(code,str) and code.isidentifier() else 'CaptureFailure'}, stop)
                if recording is not None:
                    # A bad file will not fix itself: wait for a new selection
                    # or restart instead of re-reading it in a loop.
                    while not stop.wait(1) and not shutdown.is_set():
                        pass
                else:
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
    parser = argparse.ArgumentParser(description='Shared engineering recognition session')
    parser.add_argument('--recordings-dir', type=Path,
                        help='Folder of permitted local videos, each with a .receipt.json beside it')
    parser.add_argument('--recordings-only', action='store_true',
                        help='Analyse recordings without any Sentinel credentials or camera catalogue')
    parser.add_argument('--dashboard-evidence', action='store_true',
                        default=os.getenv('SENTINEL_DASHBOARD_EVIDENCE','').lower() == 'true',
                        help='Recorded sources only: let the operator dashboard show the analysed frame and full unverified readings')
    parser.add_argument('--unpaced', action='store_true',
                        default=os.getenv('SENTINEL_RECORDED_UNPACED','').lower() == 'true',
                        help='Process recordings as fast as possible instead of at most real time; frames are never dropped either way')
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--container-bind', action='store_true', help='Use only with a Docker loopback-only port publication')
    parser.add_argument('--detector-profile', choices=('legacy','general','plate_adapted','indian_adapted'))
    parser.add_argument('--training-root', type=Path, default=Path('/training'))
    parser.add_argument('--deliver', action='store_true', help='Durably deliver strictly model-accepted sightings, never candidates or imagery')
    parser.add_argument('--plate-recovery', action='store_true', help='Opt-in native full-image fallback and 250 ms sampling near tracked vehicles')
    parser.add_argument('--lifetime', type=int, default=1800)
    parser.add_argument('--device', choices=('cpu','gpu'), default=os.getenv('SENTINEL_OCR_DEVICE','gpu'),
                        help='OCR and tracked detector device. Explicit; GPU never falls back to CPU')
    parser.add_argument('--configured-relays', action='store_true', help='When discovery fails, use only the previously approved CAM01/CAM02 local relays; not full catalogue access')
    args = parser.parse_args()
    if args.recordings_only and args.recordings_dir is None:
        parser.error('--recordings-only needs --recordings-dir')
    if args.recordings_dir is not None and args.detector_profile is None:
        args.detector_profile = 'general'
    serve(Config.from_env(live=not args.recordings_only), args.cache, args.checkpoint,
          recordings_dir=args.recordings_dir, live_catalogue=not args.recordings_only, paced=not args.unpaced,
          bind_host='0.0.0.0' if args.container_bind else '127.0.0.1',
          detector_profile=args.detector_profile, training_root=args.training_root, deliver=args.deliver,
          lifetime=max(60,min(28800,args.lifetime)), configured_relays=args.configured_relays,
          plate_recovery=args.plate_recovery, device=args.device, dashboard_evidence=args.dashboard_evidence)
