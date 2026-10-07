"""Sequential authorised camera screening. Aggregate metadata only, never media.

Counts are frame instances and unverified model proposals, not recall/accuracy.
Ground truth is deliberately not inferred from OCR guesses.
"""
import argparse
import json
import threading
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from .diagnostic_gateway import discover
from .config import Config
from .live_detection_viewer import LiveState, load_models, observe, safe_metrics


def run(config, cache, checkpoint, camera_ids, seconds, report):
    if not 15 <= seconds <= 180 or not 1 <= len(camera_ids) <= 8 or len(set(camera_ids)) != len(camera_ids):
        raise ValueError('Use one to eight distinct cameras, 15–180 seconds each')
    started_at = datetime.now(timezone.utc).isoformat()
    # A camera not returned as live by current authorised JSON cannot be probed.
    cameras = {c.camera_id: c for c in discover(config) if c.live}
    if set(camera_ids)-set(cameras):
        raise ValueError('Requested screening camera is absent from the live catalogue')
    root = Path('/workspace/data/benchmarks/reports').resolve()
    if not report.resolve().is_relative_to(root) or report.suffix != '.json':
        raise ValueError('Aggregate reports must remain in ignored benchmark reports')
    import av
    av.logging.set_level(av.logging.PANIC)
    model, detector, vehicle = load_models(config, cache, checkpoint)
    results = []
    config = replace(config, media_relay_base_url='')
    for camera_id in camera_ids:
        state = LiveState(); state.authorise(camera_ids); state.select(camera_id)
        stop = state.capture_stop
        timer = threading.Timer(seconds+40, stop.set); timer.start()
        print(json.dumps({'screening_camera': camera_id, 'requested_seconds': seconds}), flush=True)
        failed = False
        try:
            observe(config, cameras[camera_id], model, detector, vehicle, state, stop, duration_seconds=seconds)
        except Exception:
            failed = True  # Never print raw third-party messages or source URLs.
        finally:
            stop.set(); timer.cancel()
        result = safe_metrics(state.snapshot())
        result['completed_without_processing_error'] = not failed
        result['requested_window_completed'] = result.get('observed_seconds', 0) >= seconds - 1
        if not result.get('analysed_frames'):
            result['status'] = 'unavailable'
        results.append(result)
        document = {'started_at_utc': started_at, 'report_updated_at_utc': datetime.now(timezone.utc).isoformat(), 'source_mode': 'live',
                    'sequential': True, 'requested_seconds_per_camera': seconds,
                    'recordings_saved': False, 'plaintext_plates_saved': False, 'central_writes': False,
                    'ground_truth_available': False, 'accuracy': None,
                    'caveat': 'Legacy path returns at most one plate proposal per frame; tracked viewer searches every eligible tracked vehicle ROI. Counts are not unique verified vehicles or plates.',
                    'cameras': results}
        report.write_text(json.dumps(document, indent=2))
        print(json.dumps(result), flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cameras', required=True)
    parser.add_argument('--seconds', type=int, default=45)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    run(Config.from_env(live=True), args.cache, args.checkpoint, args.cameras.split(','), args.seconds, args.report)
