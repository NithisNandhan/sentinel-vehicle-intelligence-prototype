"""Compare what the vehicle detector sees with what the tracker passes on.

The shared session only searches for plates on *tracked* vehicles. This samples
a recording exactly as the session does (every 500 ms of video time) and, on
each sample, counts the detector's boxes and the boxes that survive tracking.
A large gap means vehicles are being detected but never reach plate search.
Counts only; nothing is saved from the footage.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'edge'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--recordings-dir', type=Path, required=True)
    parser.add_argument('--name', required=True, help='Part of the recording title')
    parser.add_argument('--interval-ms', type=int, default=500)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cache', type=Path, default=Path('/models/experiments/awiros'))
    parser.add_argument('--checkpoint', type=Path,
                        default=Path('/models/experiments/awiros/adaptations/indian-corpus-encoder.safetensors'))
    parser.add_argument('--detector-profile', default='general')
    parser.add_argument('--device', choices=('cpu', 'gpu'), default=os.getenv('SENTINEL_OCR_DEVICE', 'gpu'))
    args = parser.parse_args()

    import threading
    from sentinel_edge.config import Config
    from sentinel_edge.experimental_detectors import COCO_CLASSES
    from sentinel_edge.live_detection_viewer import clipped_box, load_models
    from sentinel_edge.recorded_source import RecordingLibrary, read_recording

    recordings, _ = RecordingLibrary(args.recordings_dir).scan()
    recording = next(r for r in recordings.values() if args.name.lower() in r.name.lower())
    _, _, vehicle = load_models(Config.from_env(live=False), args.cache, args.checkpoint,
                                detector_profile=args.detector_profile, device=args.device)
    vehicle.reset()
    rows, last, identities = [], None, set()
    for _, frame in read_recording(recording, threading.Event(), paced=False):
        if last is not None and frame.pts_ms-last < args.interval_ms:
            continue
        last = frame.pts_ms
        detected = vehicle.model.predict(frame.image, imgsz=640, classes=list(COCO_CLASSES), conf=.25,
                                         device='cpu', verbose=False)[0]
        boxes = [clipped_box(frame.image, b) for b in detected.boxes.xyxy.tolist()]
        eligible = sum(b[2]-b[0] >= 64 and b[3]-b[1] >= 32 for b in boxes)
        tracks = vehicle.track(frame.image)
        tracked = sum(t.box[2]-t.box[0] >= 64 and t.box[3]-t.box[1] >= 32 for t in tracks)
        identities.update(t.track_id for t in tracks)
        rows.append({'position_ms': frame.pts_ms-recording.start_pts_ms, 'detected': eligible, 'tracked': tracked})
    detected_total, tracked_total = sum(r['detected'] for r in rows), sum(r['tracked'] for r in rows)
    report = {
        'recording': recording.name, 'interval_ms': args.interval_ms, 'samples': len(rows),
        'vehicle_boxes_detected': detected_total, 'vehicle_boxes_tracked': tracked_total,
        'share_of_detected_boxes_that_reach_plate_search': round(tracked_total/detected_total, 3) if detected_total else None,
        'median_detected_per_sample': statistics.median(r['detected'] for r in rows),
        'median_tracked_per_sample': statistics.median(r['tracked'] for r in rows),
        'samples_with_vehicles_detected_but_none_tracked': sum(r['detected'] > 0 and r['tracked'] == 0 for r in rows),
        'distinct_track_ids': len(identities),
        'notice': 'Boxes at least 64 by 32 pixels, the same eligibility rule as the session. '
                  'Detector and tracker use the same model, classes and 0.25 threshold.',
        'per_sample': rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'per_sample'}, indent=2))
