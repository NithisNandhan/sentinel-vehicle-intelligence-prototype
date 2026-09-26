"""Read-only, bounded plate-crop OCR comparison; no genuine media/text saved.

Plate-box IoU association is diagnostic, NOT the production YOLO/ByteTrack
vehicle pipeline. Counts cannot establish correctness without ground truth.
"""
import argparse
import hashlib
import hmac
import json
import os
import statistics
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .awiros_experiment import AwirosExperiment
from .cameras import discover_live_pair
from .config import Config
from .ctc_fusion import CTCPlateReader
from .ocr_selection_benchmark import PlateDetectorExperiment, box_iou
from .plate_quality import crop_quality
from .stream import read_camera


def probe(config,camera,models,detector,seconds,vehicle_model=None):
    stop=threading.Event(); timer=threading.Timer(seconds+45,stop.set); timer.start()
    first=None; last_pts=None; previous_box=None; previous_stamp=None
    readers={name:CTCPlateReader(model.tokens) for name,model in models.items()}
    ids={name:set() for name in models}; secret=os.urandom(32)
    count={'decoded_frames':0,'sampled_frames':0,'native_plate_crops':0,'vehicle_regions':0,'missing_pts':0,'resets':0,'errors':0}
    candidates={name:0 for name in models}; latencies={name:[] for name in models}
    widths=[]; last_update=time.monotonic()
    try:
        for _,frame in read_camera(config,camera,stop=stop):
            if stop.is_set(): break
            now=time.monotonic()
            if first is None:
                first=now; timer.cancel(); timer=threading.Timer(seconds,stop.set); timer.start()
            count['decoded_frames']+=1
            if frame.discontinuity:
                last_pts=previous_box=previous_stamp=None
                readers={name:CTCPlateReader(model.tokens) for name,model in models.items()}
                count['resets']+=1
            if frame.pts_ms is None:
                count['missing_pts']+=1; continue
            if last_pts is not None and frame.pts_ms-last_pts<500: continue
            last_pts=frame.pts_ms; count['sampled_frames']+=1
            if now-last_update>=30:
                print(json.dumps({'camera_id':camera.camera_id,'elapsed_seconds':round(now-first),**count}),flush=True)
                last_update=now
            box=detector.box(frame.image)
            if box is None and vehicle_model is not None:
                prediction=vehicle_model.predict(frame.image,imgsz=640,classes=[2,3,5,7],conf=.25,device='cpu',verbose=False)[0]
                vehicle_boxes=sorted(zip(prediction.boxes.xyxy.tolist(),prediction.boxes.conf.tolist()),key=lambda v:-v[1])[:3]
                for vehicle_box,_ in vehicle_boxes:
                    vx1,vy1,vx2,vy2=map(int,vehicle_box)
                    vx1=max(0,vx1); vy1=max(0,vy1); vx2=min(frame.image.shape[1],vx2); vy2=min(frame.image.shape[0],vy2)
                    region=frame.image[vy1:vy2,vx1:vx2]
                    if not region.size: continue
                    count['vehicle_regions']+=1
                    local=detector.box(region)
                    if local is not None:
                        box=(local[0]+vx1,local[1]+vy1,local[2]+vx1,local[3]+vy1)
                        break
            if box is None: continue
            x1,y1,x2,y2=map(int,box)
            crop=frame.image[y1:y2,x1:x2].copy()
            if crop.size==0 or crop.shape[1]<24 or crop.shape[0]<8: continue
            count['native_plate_crops']+=1; widths.append(crop.shape[1])
            if previous_box is None or box_iou(previous_box,box)<.5 or frame.pts_ms-previous_stamp>1500:
                readers={name:CTCPlateReader(model.tokens) for name,model in models.items()}
                count['resets']+=1
            previous_box,previous_stamp=box,frame.pts_ms
            source_id=hashlib.sha256(crop.tobytes()).hexdigest()
            order=list(models.items())
            if count['sampled_frames']%2: order.reverse()
            for name,model in order:
                started=time.perf_counter()
                try:
                    readers[name].add(model.probabilities(crop),timestamp_ms=frame.pts_ms,source_id=source_id,quality=crop_quality(crop))
                    if readers[name].observation_count>=3:
                        retained=readers[name].candidates()
                        candidates[name]+=bool(retained)
                        accepted=readers[name].accepted()
                        if accepted: ids[name].add(hmac.digest(secret,accepted.plate.encode(),'sha256'))
                except Exception:
                    count['errors']+=1 # Never serialize third-party errors, URLs or OCR text.
                latencies[name].append((time.perf_counter()-started)*1000)
    finally:
        stop.set(); timer.cancel()
    return {'camera_id':camera.camera_id,**count,'observed_seconds':round(time.monotonic()-first,2) if first else 0,
            'median_native_crop_width':statistics.median(widths) if widths else None,
            'methods':{name:{'unique_accepted_unverified_identifiers':len(ids[name]),
                             'windows_with_retained_candidates':candidates[name],
                             'median_ocr_and_fusion_ms':statistics.median(latencies[name]) if latencies[name] else None}
                       for name in models}}


def main(cache,checkpoint,report,seconds,vehicle_crops=False):
    import av
    av.logging.set_level(av.logging.PANIC)
    config=Config.from_env(live=True)
    cameras,fallback=discover_live_pair(config)
    if {c.camera_id for c in cameras}!={'cam01','cam02'} or not 15<=seconds<=300:
        raise ValueError('Shadow test restricted to the approved pair and bounded duration')
    models={'parent':AwirosExperiment(cache,device='gpu'),
            'deeper_adapted':AwirosExperiment(cache,device='gpu',checkpoint=checkpoint)}
    detector=PlateDetectorExperiment()
    vehicle_model=None
    if vehicle_crops:
        from ultralytics import YOLO
        vehicle_model=YOLO(str(config.model_cache_dir/'yolov8n.pt'))
    result={'started_at_utc':datetime.now(timezone.utc).isoformat(),'source_mode':'live',
            'recordings_saved':False,'plaintext_plates_saved':False,'central_writes':False,
            'ground_truth_available':False,'accuracy':None,'production_promoted':False,
            'association':'conservative plate-box IoU diagnostic; production vehicle tracking not evaluated',
            'vehicle_crop_fallback':vehicle_crops,'vehicle_detector_device':'cpu' if vehicle_crops else None,
            'catalogue_fallback_used':fallback,'requested_seconds_per_camera':seconds,
            'cameras':[probe(config,camera,models,detector,seconds,vehicle_model) for camera in cameras]}
    report.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    if any(not r['sampled_frames'] or r['errors'] for r in result['cameras']):
        raise RuntimeError('Live shadow decode or OCR validation failed; aggregate report preserved')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True); parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--seconds',type=int,default=90); parser.add_argument('--vehicle-crops',action='store_true'); args=parser.parse_args()
    main(args.cache,args.checkpoint,args.report,args.seconds,args.vehicle_crops)
