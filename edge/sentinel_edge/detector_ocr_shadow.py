"""Paired live detector -> tracked OCR diagnostic; no central writes or recordings."""
import argparse
from collections import deque
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import threading
import time

from .config import Config
from .experimental_detectors import PROFILES, HASHES, PlateDetector, VehicleTracker
from .live_detection_viewer import viewer_frames
from .tracked_ocr import TrackedOCR


class _DecodeStop:
    """Closing a decode tap must not cancel the operator's camera session."""
    def __init__(self,parent):
        self.parent,self.local = parent,threading.Event()

    def is_set(self):
        return self.parent.is_set() or self.local.is_set()

    def set(self):
        self.local.set()

    def wait(self,seconds):
        deadline = time.monotonic()+seconds
        while not self.is_set():
            remaining = deadline-time.monotonic()
            if remaining <= 0:return False
            self.local.wait(min(.1,remaining))
        return True


class LatestFrames:
    """Continuously decode, retain one native frame and never build a video backlog."""
    def __init__(self, frames, stop):
        self.stop = _DecodeStop(stop)
        # A callable receives local cancellation, so decoder retries stop on
        # close without marking the outer operator session cancelled.
        self.frames = frames(self.stop) if callable(frames) else frames
        self.condition = threading.Condition()
        self.latest = None
        self.sequence = 0
        self.finished = self.failed = False
        self.pending_cut = False
        self.decoded = self.overwritten = 0
        self.consumed = 0
        self.thread = threading.Thread(target=self._read,daemon=True)

    def _read(self):
        try:
            for _,frame in self.frames:
                if self.stop.is_set(): break
                with self.condition:
                    self.decoded += 1
                    self.overwritten += self.sequence > self.consumed
                    self.pending_cut = self.pending_cut or frame.discontinuity
                    self.sequence += 1
                    self.latest = frame
                    self.condition.notify_all()
        except Exception as exc:
            self.failed = True  # Never serialize source URL or third-party message.
            self.failure_code = type(exc).__name__
        finally:
            with self.condition:
                self.finished = True
                self.condition.notify_all()

    def __enter__(self):
        self.thread.start()
        return self

    def take(self, previous):
        with self.condition:
            self.condition.wait_for(lambda:self.sequence > previous or self.finished or self.stop.is_set(),timeout=1)
            if self.sequence <= previous or self.stop.is_set(): return None
            frame = replace(self.latest,discontinuity=self.pending_cut)
            self.pending_cut = False
            self.consumed = self.sequence
            return self.sequence,frame

    def __exit__(self,*_):
        self.stop.set()
        with self.condition:self.condition.notify_all()
        self.thread.join(timeout=10)
        self.latest = None


def load_ocr(cache,checkpoint):
    import numpy as np
    from .awiros_experiment import AwirosExperiment
    # CUDA 12 Paddle initializes first; Torch and ONNX detectors explicitly CPU.
    model = AwirosExperiment(cache,device='gpu',checkpoint=checkpoint)
    model.probabilities(np.zeros((32,120,3),np.uint8))
    return model


def load_profiles(ocr,model_cache,training_root,names,roi_budget=2):
    vehicles,plates,pipelines = {},{},{}
    for name in names:
        vehicle,plate = PROFILES[name]
        if vehicle not in vehicles:vehicles[vehicle] = VehicleTracker(vehicle,model_cache,training_root)
        if plate not in plates:plates[plate] = PlateDetector(plate,model_cache)
        pipelines[name] = TrackedOCR(ocr,plates[plate],roi_budget=roi_budget)
    return vehicles,pipelines


def probe(config,camera,ocr,vehicles,pipelines,seconds):
    for model in vehicles.values():model.reset()
    # New camera-local histories and HMAC keys, while sharing already loaded weights.
    pipelines = {name:TrackedOCR(ocr,p.detector,roi_budget=p.roi_budget) for name,p in pipelines.items()}
    stop = threading.Event()
    timer = threading.Timer(seconds+45,stop.set); timer.start()
    first = last_sample = last_pts = None
    sequence = sampled = missing_pts = 0
    timings = {name:deque(maxlen=512) for name in pipelines}
    vehicle_times = {name:deque(maxlen=512) for name in vehicles}
    resolution = None
    last_progress = time.monotonic()
    try:
        with LatestFrames(lambda token:viewer_frames(config,camera,token),stop) as tap:
            while not stop.is_set():
                item = tap.take(sequence)
                if item is None:
                    if tap.finished:break
                    continue
                sequence,frame = item
                now = time.monotonic()
                if first is None:
                    first = now; timer.cancel()
                    timer = threading.Timer(seconds,stop.set); timer.start()
                if frame.discontinuity or (last_pts is not None and frame.pts_ms is not None and
                                           (frame.pts_ms < last_pts or frame.pts_ms-last_pts > 2500)):
                    last_sample = None
                    for model in vehicles.values():model.reset()
                    for pipeline in pipelines.values():pipeline.reset()
                if frame.pts_ms is None:
                    missing_pts += 1; continue
                last_pts = frame.pts_ms
                if last_sample is not None and frame.pts_ms-last_sample < 500:continue
                last_sample = frame.pts_ms
                sampled += 1
                resolution = [frame.image.shape[1],frame.image.shape[0]]
                tracks = {}
                for name,model in vehicles.items():
                    started = time.perf_counter()
                    tracks[name] = model.track(frame.image)
                    vehicle_times[name].append((time.perf_counter()-started)*1000)
                # Alternate order to avoid always giving one profile the warm cache.
                order = list(pipelines)
                if sampled % 2:order.reverse()
                cache = {}
                for name in order:
                    started = time.perf_counter()
                    pipelines[name].process(frame.image,tracks[PROFILES[name][0]],frame.pts_ms,posterior_cache=cache)
                    timings[name].append((time.perf_counter()-started)*1000)
                if now-last_progress >= 30:
                    print(json.dumps({'camera_id':camera.camera_id,'sampled_frames':sampled,
                        'elapsed_seconds':round(now-first)}),flush=True)
                    last_progress = now
            observed = time.monotonic()-first if first is not None else 0
            result = {'camera_id':camera.camera_id,'resolution':resolution,'decoded_frames':tap.decoded,
                'overwritten_unanalysed_frames':tap.overwritten,'sampled_frames':sampled,
                'missing_pts':missing_pts,'observed_seconds':round(observed,2),'decoder_failed':tap.failed,
                'requested_window_completed':observed >= seconds-1,'methods':{}}
            for name,pipeline in pipelines.items():
                result['methods'][name] = dict(pipeline.summary(),
                    median_vehicle_tracking_ms=round(statistics.median(vehicle_times[PROFILES[name][0]]),2) if vehicle_times[PROFILES[name][0]] else None,
                    median_plate_ocr_fusion_ms=round(statistics.median(timings[name]),2) if timings[name] else None)
            return result
    finally:
        stop.set(); timer.cancel()


def main(args):
    if not 15 <= args.seconds <= 300 or not 1 <= len(args.cameras) <= 2 or len(set(args.cameras)) != len(args.cameras):
        raise ValueError('One or two authorised cameras, 15–300 seconds required')
    if len(set(args.profiles)) != len(args.profiles) or any(p not in PROFILES for p in args.profiles):
        raise ValueError('Known distinct detector profiles required')
    root = Path('/workspace/data/benchmarks/reports').resolve()
    if not args.report.resolve().is_relative_to(root) or args.report.suffix != '.json' or args.report.exists():
        raise ValueError('New aggregate-only report in ignored benchmark directory required')
    import av
    av.logging.set_level(av.logging.PANIC)
    from .diagnostic_gateway import discover
    config = replace(Config.from_env(live=True),model_cache_dir=Path('/models'),media_relay_base_url='')
    catalogue = {c.camera_id:c for c in discover(config) if c.live}
    if set(args.cameras)-set(catalogue):raise ValueError('Requested camera absent from current authorised catalogue')
    ocr = load_ocr(args.cache,args.checkpoint)
    vehicles,pipelines = load_profiles(ocr,config.model_cache_dir,args.training_root,args.profiles,args.roi_budget)
    result = {'started_at_utc':datetime.now(timezone.utc).isoformat(),'source_mode':'live',
        'recordings_saved':False,'plaintext_plates_saved':False,'central_writes':False,
        'ground_truth_available':False,'verified_plate_readings':None,'accuracy':None,'production_promoted':False,
        'detector_device':'cpu','ocr_device':'gpu','model_hashes':HASHES,
        'profiles':{name:PROFILES[name] for name in args.profiles},'roi_budget_per_profile':args.roi_budget,
        'requested_seconds_per_camera':args.seconds,
        'caveat':'Same sampled native frames per profile, ByteTrack histories separately guarded. Latest-frame dropping avoids backlog. Counts are model proposals, not accuracy; paired processing is not standalone throughput. No reconstruction ablation or registry used.',
        'cameras':[]}
    for camera_id in args.cameras:
        print(json.dumps({'testing_camera':camera_id}),flush=True)
        row = probe(config,catalogue[camera_id],ocr,vehicles,pipelines,args.seconds)
        result['cameras'].append(row)
        args.report.write_text(json.dumps(result,indent=2))
        print(json.dumps(row),flush=True)
    if any(not c['sampled_frames'] or c['decoder_failed'] or not c['requested_window_completed'] for c in result['cameras']):
        raise RuntimeError('Incomplete live diagnostic; aggregate evidence preserved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,default=Path('/models/experiments/awiros'))
    parser.add_argument('--checkpoint',type=Path,default=Path('/models/experiments/awiros/adaptations/indian-corpus-encoder.safetensors'))
    parser.add_argument('--training-root',type=Path,default=Path('/training'))
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--cameras',type=lambda s:s.split(','),default=['cam14','cam17'])
    parser.add_argument('--profiles',type=lambda s:s.split(','),default=list(PROFILES))
    parser.add_argument('--seconds',type=int,default=90)
    parser.add_argument('--roi-budget',type=int,default=2)
    args = parser.parse_args()
    try:main(args)
    except Exception as exc:
        print(json.dumps({'diagnostic_failed':type(exc).__name__,'details_redacted':True}),flush=True)
        raise SystemExit(1) from None
