"""Bounded CPU/GPU inference timing on a permission-receipted recording.

Images, crops and OCR strings stay in memory. Reports contain only timing,
aggregate counts and model identity. This is NOT an accuracy benchmark or a
full-dashboard throughput measurement; decoding, queueing and rendering are
excluded from the per-sample times.
"""
import argparse
import json
from pathlib import Path
import platform
import statistics
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'edge'))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--device', choices=('cpu', 'gpu'), required=True)
    parser.add_argument('--recordings-dir', type=Path, default=Path('/recordings'))
    parser.add_argument('--recording-id', required=True)
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--start-ms', type=int, default=0)
    parser.add_argument('--profile', choices=('general', 'legacy', 'plate_adapted'), default='general')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.samples <= 32 or args.start_ms < 0:
        parser.error('Use 1-32 samples and a nonnegative start position')

    from sentinel_edge.config import Config
    from sentinel_edge.recorded_source import RecordingLibrary, read_recording
    from sentinel_edge.live_detection_viewer import load_models, model_identity
    from sentinel_edge.plate_search import TargetedPlateSearch
    from sentinel_edge.tracked_ocr import TrackedOCR
    import numpy as np

    recordings, _ = RecordingLibrary(args.recordings_dir).scan()
    recording = recordings.get(args.recording_id)
    if recording is None:
        raise ValueError('Requested recording has no valid permission receipt')
    frames = []
    next_ms = args.start_ms
    source = read_recording(recording, threading.Event(), paced=False, start_ms=args.start_ms)
    try:
        for _, frame in source:
            if frame.pts_ms >= next_ms:
                frames.append(frame)
                next_ms = frame.pts_ms + 500
                if len(frames) == args.samples:
                    break
    finally:
        source.close()
    if len(frames) != args.samples:
        raise ValueError('Recording is too short for the requested sample window')

    config = Config.from_env(live=False)
    cache = Path('/models/experiments/awiros')
    started = time.perf_counter()
    reader, detector, vehicle = load_models(config, cache,
        cache / 'adaptations/indian-corpus-encoder.safetensors',
        detector_profile=args.profile, device=args.device)
    import torch
    def synchronize():
        if args.device == 'gpu':
            torch.cuda.synchronize()
            reader.paddle.device.synchronize()
    # Exclude first-use model setup from steady-state timings in both modes.
    blank = np.zeros(frames[0].image.shape, dtype=np.uint8)
    vehicle.track(blank)
    detector.boxes(blank)
    reader.probabilities(np.zeros((32, 120, 3), dtype=np.uint8))
    synchronize()
    vehicle.reset()
    load_seconds = time.perf_counter() - started
    pipeline = TrackedOCR(reader, TargetedPlateSearch(detector),
                          threshold=config.ocr_min_confidence, reconstruction=True)
    timings = []
    stages = []
    for frame in frames:
        synchronize()
        started = time.perf_counter()
        tracks = vehicle.track(frame.image)
        synchronize()
        detection_ms = (time.perf_counter() - started) * 1000
        pipeline.process(frame.image, tracks, frame.pts_ms)
        synchronize()
        timings.append((time.perf_counter() - started) * 1000)
        stages.append(dict(vehicle=round(detection_ms, 3),
                           **{k: round(v, 3) for k, v in pipeline.stage_ms.items()}))
    report = {
        'benchmark': 'bounded_sample_inference_not_end_to_end_or_accuracy',
        'recording_sha256': recording.sha256,
        'sample_pts_ms': [frame.pts_ms for frame in frames],
        'resolution': [frames[0].image.shape[1], frames[0].image.shape[0]],
        'device': args.device, 'profile': args.profile,
        'models': model_identity(reader, args.profile, detector=detector, vehicle_model=vehicle),
        'runtime': {'python': platform.python_version(), 'torch': torch.__version__},
        'load_and_warmup_seconds': round(load_seconds, 3),
        'sample_total_ms': [round(value, 3) for value in timings],
        'median_sample_ms': round(statistics.median(timings), 3),
        'stages_ms': stages,
        'counts': {key: int(pipeline.metrics[key]) for key in
                   ('analysed_frames', 'vehicle_box_instances', 'plate_crops', 'ocr_calls')},
        'notice': 'No plate strings, crops or images exported. Counts are not recognition accuracy.'}
    encoded = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(encoded + '\n')
    print(encoded, flush=True)


if __name__ == '__main__':
    main()
