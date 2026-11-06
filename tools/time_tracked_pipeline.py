"""Time the shared tracked pipeline on one clip and fingerprint everything it outputs.

Run inside the recorded-analysis container, with nothing else busy, e.g.:

    SENTINEL_VEHICLE_THREADS=2 python /workspace/tools/run_ocr_research.py \
        /workspace/tools/time_tracked_pipeline.py "/recordings/<video>" /tmp/timing.json

Same steps as the service (general profile, 500 ms samples, reconstruction on,
plate options from the environment). Outputs per sample: tracks, plate boxes, OCR
status, observation counts, every candidate with its score, and accepted readings.
Two runs with the same `outputs_sha256` produced identical results. Plate text stays
in the local output file only; keep it out of Git.
"""
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, '/workspace/edge')
from sentinel_edge.awiros_experiment import AwirosExperiment  # noqa: E402
from sentinel_edge.experimental_detectors import PROFILES, PlateDetector, VehicleTracker  # noqa: E402
from sentinel_edge.plate_options import PlateOptions  # noqa: E402
from sentinel_edge.plate_search import TargetedPlateSearch  # noqa: E402
from sentinel_edge.tracked_ocr import TrackedOCR  # noqa: E402

CACHE = Path('/models/experiments/awiros')


def main():
    video, output = sys.argv[1], Path(sys.argv[2])
    loaded = time.monotonic()
    vehicle_name, plate_name = PROFILES['general']
    tracker = VehicleTracker(vehicle_name, '/models', '/training')
    reader = AwirosExperiment(CACHE, device='cpu', checkpoint=CACHE / 'adaptations/indian-corpus-encoder.safetensors')
    pipeline = TrackedOCR(reader, TargetedPlateSearch(PlateDetector(plate_name, '/models')), threshold=.85,
                          reconstruction=True, options=PlateOptions.from_env())
    load_seconds = time.monotonic() - loaded

    samples, outputs = [], []
    capture, last = cv2.VideoCapture(video), None
    while True:
        ok, image = capture.read()
        if not ok:
            break
        pts = round(capture.get(cv2.CAP_PROP_POS_MSEC))
        if last is not None and pts - last < 500:
            continue
        last = pts
        started = time.monotonic()
        tracks = tracker.track(image)
        vehicle_ms = (time.monotonic() - started) * 1000
        displays = pipeline.process(image, tracks, pts)
        total_ms = (time.monotonic() - started) * 1000
        samples.append({'pts_ms': pts, 'total_ms': round(total_ms, 1), 'vehicle_ms': round(vehicle_ms, 1),
                        **{f'{k}_ms': round(v, 1) for k, v in pipeline.stage_ms.items()}})
        state = []
        for key, history in sorted(pipeline.histories.items()):
            accepted = history.reader.accepted(.85) if history.reader.observation_count >= 3 else None
            state.append([key, list(history.box), history.reader.observation_count,
                          [[c.plate, round(c.log_score, 6), round(c.score, 6)] for c in history.reader.candidates()][:5],
                          accepted.plate if accepted else None])
        outputs.append({'pts_ms': pts, 'tracks': [[t.track_id, list(t.box), t.vehicle_class] for t in tracks],
                        'displays': [[d['track_id'], list(d['box']), d['ocr_status'], d['observations']] for d in displays],
                        'state': state})
    digest = hashlib.sha256(json.dumps(outputs, sort_keys=True).encode()).hexdigest()
    totals = [s['total_ms'] for s in samples]
    summary = {'samples': len(samples), 'load_seconds': round(load_seconds, 1),
               'analysis_seconds': round(sum(totals) / 1000, 1), 'median_sample_ms': round(statistics.median(totals)),
               'stage_seconds': {key: round(sum(s.get(key, 0) for s in samples) / 1000, 1)
                                 for key in ('vehicle_ms', 'plate_ms', 'ocr_ms', 'reconstruction_ms', 'fusion_ms')},
               'outputs_sha256': digest, 'metrics': dict(pipeline.metrics)}
    output.write_text(json.dumps({'summary': summary, 'samples': samples, 'outputs': outputs}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
