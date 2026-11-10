"""Tracking-only recording benchmark. No OCR, plate model or central delivery.

One frozen GPU detection pass is replayed through sparse and dense ByteTrack.
Playback shows actual observations; smooth playback is not a processing-FPS claim.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'edge'))
from sentinel_edge.recorded_source import RecordingLibrary, read_recording
from sentinel_edge.vehicle_track_policy import VehicleTrackPolicy

VEHICLE_CLASSES = (2, 3, 5, 7)
MODEL_SHA = '646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes((json.dumps(value, allow_nan=False, indent=2)+'\n').encode('utf-8'))
    temporary.replace(path)


def capture(args):
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO
    if args.output.exists():
        raise ValueError('Use a fresh immutable benchmark directory')
    args.output.mkdir(parents=True)
    (args.output/'capture-script.py').write_bytes(Path(__file__).read_bytes())
    write_json(args.output/'progress.json', dict(stage='loading_vehicle_model'))
    recordings, _ = RecordingLibrary(args.recordings_dir).scan()
    source = recordings.get(args.recording_id)
    if source is None:
        raise ValueError('A permitted recording and valid receipt are required')
    model_path = args.models/'experiments/detectors/yolo26s.pt'
    if sha(model_path) != MODEL_SHA or not torch.cuda.is_available():
        raise ValueError('Verified cached model and working CUDA required')
    torch.set_num_threads(2)
    model = YOLO(str(model_path)).to('cuda:0')
    for _ in range(3):
        model.predict(np.zeros((args.image_size,args.image_size,3), np.uint8), device=0, imgsz=args.image_size,
                      classes=list(VEHICLE_CLASSES), conf=.25, iou=.45, verbose=False, save=False)
    frames, timings = [], []
    began = time.monotonic()
    for _, frame in read_recording(source, threading.Event(), paced=False):
        if frame.pts_ms is None:
            raise ValueError('PTS is required for every frame')
        torch.cuda.synchronize()
        started = time.monotonic()
        result = model.predict(frame.image, device=0, imgsz=args.image_size, classes=list(VEHICLE_CLASSES),
                               conf=.25, iou=.45, verbose=False, save=False)[0]
        torch.cuda.synchronize()
        timings.append((time.monotonic()-started)*1000)
        boxes = result.boxes.data.cpu().numpy().tolist() if result.boxes is not None else []
        frames.append(dict(position_ms=frame.pts_ms-source.start_pts_ms,
                           discontinuity=frame.discontinuity, detections=boxes))
        if len(frames)%100 == 0:
            progress = dict(stage='detecting', frames=len(frames), position_ms=frames[-1]['position_ms'])
            write_json(args.output/'progress.json', progress)
            print(json.dumps(progress), flush=True)
    if not frames:
        raise ValueError('No decoded frames')
    document = dict(schema_version=1, source_sha256=source.sha256, recording_id=source.recording_id,
                    source_name=source.path.name, duration_ms=source.duration_ms,
                    width=int(result.orig_shape[1]), height=int(result.orig_shape[0]), reached_eof=True,
                    decoded_frames=len(frames), vehicle_inference_frames=len(frames),
                    model='yolo26s', model_sha256=MODEL_SHA, classes=list(VEHICLE_CLASSES),
                    confidence=.25, nms_iou=.45, inference_size=args.image_size, actual_device=str(model.device),
                    ultralytics_version=ultralytics.__version__,
                    capture_script_sha256=sha(args.output/'capture-script.py'),
                    wall_seconds=round(time.monotonic()-began,3),
                    median_detector_ms=round(float(np.median(timings)),3),
                    p95_detector_ms=round(float(np.percentile(timings,95)),3),
                    detector_seconds=sum(timings)/1000,
                    ocr_loaded=False, plate_model_loaded=False, central_delivery=False,
                    frames=frames)
    with gzip.open(args.output/'detections.json.gz','wt',encoding='utf-8') as target:
        json.dump(document,target,allow_nan=False)
    write_json(args.output/'progress.json', dict(stage='detections_frozen', frames=len(frames), reached_eof=True))
    print(json.dumps(dict(stage='detections_frozen', frames=len(frames),
                         wall_seconds=document['wall_seconds'], median_detector_ms=document['median_detector_ms'])),flush=True)


def read_detections(root):
    with gzip.open(root/'detections.json.gz','rt',encoding='utf-8') as source:
        document = json.load(source)
    if not document['reached_eof'] or document['decoded_frames'] != len(document['frames']):
        raise ValueError('A complete detection pass is required')
    return document


def pack(track):
    return [track['passage_id'], *[round(v,2) for v in track['box']],
            track['vehicle_class'], round(track['confidence'],4), int(track['status']=='qualified')]


def ownership(box, tracks):
    """Diagnostic candidates only; never choose a convenient owner in overlap."""
    area = (box[2]-box[0])*(box[3]-box[1])
    matches = []
    for track in tracks:
        b = track['box']
        intersection = max(0,min(b[2],box[2])-max(b[0],box[0]))*max(0,min(b[3],box[3])-max(b[1],box[1]))
        if area > 0 and intersection/area >= .9:
            matches.append(track['passage_id'])
    return dict(candidate_owners=matches, status='unique' if len(matches)==1 else 'ambiguous' if matches else 'unassigned')


def replay(args):
    import numpy as np
    import ultralytics
    import yaml
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.byte_tracker import BYTETracker
    if (args.output/'report.json').exists():
        raise ValueError('Preserve the completed replay')
    capture_root = args.capture_dir or args.output
    document = read_detections(capture_root)
    args.output.mkdir(parents=True,exist_ok=True)
    if ultralytics.__version__ != document['ultralytics_version']:
        raise ValueError('Tracker runtime differs from capture')
    config_path = Path(ultralytics.__file__).parent/'cfg/trackers/bytetrack.yaml'
    config = yaml.safe_load(config_path.read_text())
    # Sequential replays avoid cross-instance changes to ByteTrack's class-level ID counter.
    results, mode_frames = {}, {}
    for mode, interval in [('sparse',500), ('dense',0), ('screened',0)]:
        effective_config = {**config, **({'new_track_thresh':.4,
            **({'fuse_score':False} if args.motion_association else {})} if mode=='screened' else {})}
        tracker = BYTETracker(SimpleNamespace(**effective_config))
        policy, rows, latest, last_update = VehicleTrackPolicy(), [], [], None
        raw_count = qualified_count = updates = 0
        native_ids = set()
        started = time.monotonic()
        for frame in document['frames']:
            pts = frame['position_ms']
            if last_update is None or pts-last_update >= interval:
                if frame['discontinuity']:
                    tracker.reset()
                array = np.asarray(frame['detections'],dtype=np.float32).reshape(-1,6)
                boxes = Boxes(array,orig_shape=(document['height'],document['width']))
                tracked = tracker.update(boxes)
                detections = [dict(native_id=str(int(t[4])), box=[float(v) for v in t[:4]],
                                   confidence=float(t[5]),vehicle_class=int(t[6])) for t in tracked]
                latest = policy.update(detections,pts)
                native_ids.update(t['native_id'] for t in detections)
                raw_count += len(detections)
                qualified_count += sum(t['status']=='qualified' for t in latest)
                last_update = pts
                updates += 1
            rows.append(dict(t=pts, tracks=[pack(t) for t in latest], held_ms=pts-last_update))
        records = policy.finish(document['frames'][-1]['position_ms'])
        for record in records:
            record['preview_paths'] = []
        results[mode] = dict(updates=updates, tracker_config=effective_config, native_track_ids=len(native_ids),
            passage_records=len(records), qualified_passages=sum(r['status']=='qualified' for r in records),
            tentative_only_passages=sum(r['status']!='qualified' for r in records),
            tracked_observation_instances=raw_count, qualified_observation_instances=qualified_count,
            qualification_excluded_instances=raw_count-qualified_count,
            cpu_replay_seconds=round(time.monotonic()-started,3), policy_counters=policy.counters,
            records=records)
        mode_frames[mode] = rows
    # Existing visually reviewed plate positions are used only for ownership inspection,
    # never to train, tune or choose vehicle detections or tracker output.
    anchors = [(9500,[1805,1711,1916,1795]),(10000,[1886,1769,2028,1814]),
               (11000,[1693,1767,1834,1817]),(28000,[1946,1690,2150,1775]),
               (30500,[1336,1567,1469,1632])]
    anchor_rows = []
    for pts, box in anchors:
        row = dict(position_ms=pts, plate_box=box)
        for mode in results:
            frame = min(mode_frames[mode],key=lambda f:abs(f['t']-pts))
            tracks = [dict(passage_id=t[0],box=t[1:5]) for t in frame['tracks']]
            row[mode] = ownership(box,tracks)
        anchor_rows.append(row)
    timeline = [dict(t=frame['position_ms'], d=mode_frames['dense'][i]['tracks'],
                     s=mode_frames['sparse'][i]['tracks'], sa=mode_frames['sparse'][i]['held_ms'],
                     q=mode_frames['screened'][i]['tracks'], r=frame['detections']) for i,frame in enumerate(document['frames'])]
    write_json(args.output/'timeline.json',dict(width=document['width'],height=document['height'],frames=timeline))
    report = {k:v for k,v in document.items() if k!='frames'}
    report.update(detections_sha256=sha(capture_root/'detections.json.gz'),
                  tracker_config=config, tracker_config_sha256=sha(config_path),
                  policy_source_sha256=sha(Path(__file__).resolve().parents[1]/'edge/sentinel_edge/vehicle_track_policy.py'),
                  replay_script_sha256=sha(__file__), strategies=results, anchors=anchor_rows,
                  identity_accuracy=None, detection_precision=None, recall=None,
                  limitations=['No human vehicle inventory or track identities supplied; no IDF1, precision, recall or identity-switch accuracy claim.',
                               'Qualified means repeated model evidence, not a verified vehicle. Persistent false positives may qualify.',
                               'Sparse boxes are held between 500ms updates, explicitly labelled in the viewer; dense boxes use actual per-frame tracker outputs.',
                               'ByteTrack buffer counts updates: same 30-update setting represents different durations in sparse versus dense modes. The outer passage policy independently uses PTS.',
                               'Only motorized COCO classes 2,3,5,7 are included; bicycles are excluded from this plate-workload demonstration.'])
    write_json(args.output/'report.json',report)
    (args.output/'replay-script.py').write_bytes(Path(__file__).read_bytes())
    write_json(args.output/'progress.json',dict(stage='tracking_replayed',frames=len(timeline)))
    print(json.dumps(dict(stage='tracking_replayed',summary={m:{k:v for k,v in d.items() if k!='records'} for m,d in results.items()},anchors=anchor_rows)),flush=True)


def thumbnails(args):
    import cv2
    report_path = args.output/'report.json'
    report = json.loads(report_path.read_bytes())
    timeline = json.loads((args.output/'timeline.json').read_bytes())['frames']
    if (args.output/'assets').exists():
        raise ValueError('Preserve existing thumbnails')
    (args.output/'assets').mkdir()
    sightings = {}
    for frame in timeline:
        for track in frame['q']:
            sightings.setdefault(track[0],[]).append((frame['t'],track[1:5]))
    requested = {}
    for pid, rows in sightings.items():
        selected = [rows[0], max(rows,key=lambda r:(r[1][2]-r[1][0])*(r[1][3]-r[1][1])), rows[-1]]
        for index,(pts,box) in enumerate(dict.fromkeys((pts,tuple(box)) for pts,box in selected)):
            requested.setdefault(pts,[]).append((pid,index,box))
    recordings, _ = RecordingLibrary(args.recordings_dir).scan()
    source = recordings.get(args.recording_id)
    if source is None or source.sha256!=report['source_sha256']:
        raise ValueError('The permitted source recording changed')
    records = {r['passage_id']:r for r in report['strategies']['screened']['records']}
    for _, frame in read_recording(source,threading.Event(),paced=False):
        pts = frame.pts_ms-source.start_pts_ms
        for pid,index,box in requested.pop(pts,[]):
            x1,y1,x2,y2 = [int(v) for v in box]
            crop = frame.image[max(0,y1):min(frame.image.shape[0],y2),max(0,x1):min(frame.image.shape[1],x2)]
            if not crop.size:
                continue
            if crop.shape[1]>440:
                crop = cv2.resize(crop,(440,max(1,round(crop.shape[0]*440/crop.shape[1]))))
            relative = f'assets/{pid}-{index}.jpg'
            if not cv2.imwrite(str(args.output/relative),crop,[cv2.IMWRITE_JPEG_QUALITY,90]):
                raise ValueError('Thumbnail write failed')
            records[pid]['preview_paths'].append(dict(path=relative,position_ms=pts,box=list(box),resized_for_review=True))
    if requested:
        raise ValueError('Source ended before requested review frames')
    write_json(report_path,report)
    write_json(args.output/'progress.json',dict(stage='ready',review_passages=len(records),
               thumbnails=sum(len(r['preview_paths']) for r in records.values())))
    print(json.dumps(dict(stage='ready',review_passages=len(records))),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('capture','replay','thumbnails'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recordings-dir', type=Path, default=Path('/recordings'))
    parser.add_argument('--recording-id', default='rec-6b7064605b3d')
    parser.add_argument('--models', type=Path, default=Path('/models'))
    parser.add_argument('--image-size',type=int,choices=(640,960,1280),default=640)
    parser.add_argument('--capture-dir',type=Path)
    parser.add_argument('--motion-association',action='store_true',help='Candidate: associate screened tracks by overlap without multiplying detector score')
    args = parser.parse_args()
    {'capture':capture,'replay':replay,'thumbnails':thumbnails}[args.phase](args)


if __name__ == '__main__':
    main()
