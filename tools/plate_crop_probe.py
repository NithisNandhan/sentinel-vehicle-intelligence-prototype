"""Plate proposals the pipeline would read on one clip, with their crops and detector scores.

Run inside the recorded-analysis container (same models and runtime):

    python /workspace/tools/run_ocr_research.py /workspace/tools/plate_crop_probe.py \
        "/recordings/<video>" /tmp/probe --profile general

Samples every 500 ms like the recorded pipeline and applies its size gates (vehicle
64x32, plate 24x8). Diagnostic only: no setting changes and no OCR. Detector scores
are uncalibrated. Crops contain plate pixels: keep the output out of Git.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, '/workspace/edge')
from sentinel_edge.experimental_detectors import PROFILES, PlateDetector, VehicleTracker  # noqa: E402
from sentinel_edge.live_detection_viewer import clipped_box, crop_box  # noqa: E402
from sentinel_edge.plate_search import TargetedPlateSearch  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('video')
    parser.add_argument('output', type=Path)
    parser.add_argument('--profile', choices=('general', 'plate_adapted'), default='general')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    vehicle_name, plate_name = PROFILES[args.profile]
    tracker = VehicleTracker(vehicle_name, '/models', '/training')
    search = TargetedPlateSearch(PlateDetector(plate_name, '/models'))

    counts, rows = collections.Counter(), []
    capture, last, sample = cv2.VideoCapture(args.video), None, 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        pts = round(capture.get(cv2.CAP_PROP_POS_MSEC))
        if last is not None and pts - last < 500:
            continue
        last, sample = pts, sample + 1
        for track in tracker.track(image):
            box = clipped_box(image, track.box)
            counts['vehicle boxes'] += 1
            if box[2] - box[0] < 64 or box[3] - box[1] < 32:
                counts['vehicle below 64x32'] += 1
                continue
            region = crop_box(image, box)
            local = search.box(region)
            if local is None:
                counts['no plate found'] += 1
                continue
            local = clipped_box(region, local)
            w, h = local[2] - local[0], local[3] - local[1]
            if w < 24 or h < 8:
                counts['plate below 24x8'] += 1
                continue
            counts['plate crops'] += 1
            name = f'read-{sample:02d}-{track.track_id}-{w}x{h}.png'
            cv2.imwrite(str(args.output / name), crop_box(region, local))
            rows.append({'file': name, 'sample': sample, 'pts_ms': pts, 'track': track.track_id,
                         'vehicle_class': track.vehicle_class, 'plate_px': [w, h],
                         'detector_confidence': search.last_confidence, 'fallback_search': search.last_fallback})
    report = {'video': Path(args.video).name, 'profile': args.profile, 'plate_model': plate_name,
              'samples': sample, 'counts': dict(counts), 'crops': rows}
    (args.output / 'report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps({'profile': args.profile, 'samples': sample, 'counts': dict(counts)}, indent=1))


if __name__ == '__main__':
    main()
