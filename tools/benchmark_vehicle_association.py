"""Diagnostic association replay: frozen boxes, native source frames, no OCR.

Each strategy directory is directly consumable as --tracking by
run_owned_multiframe.py. Counts describe tracker output, not identity accuracy.
"""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import threading
import time
from types import SimpleNamespace

from benchmark_vehicle_tracking import (MODEL_SHA, VehicleTrackPolicy, pack,
                                        read_detections, sha, write_json)
from sentinel_edge.recorded_source import RecordingLibrary, read_recording


def config_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_capture(document):
    if not document['frames'] or document['vehicle_inference_frames'] != document['decoded_frames']:
        raise ValueError('A nonempty complete per-frame detection capture is required')
    if document['model_sha256'] != MODEL_SHA:
        raise ValueError('Unexpected frozen detector model identity')
    previous = -1
    for frame in document['frames']:
        pts = frame['position_ms']
        if isinstance(pts, bool) or not isinstance(pts, (int, float)) or not math.isfinite(pts) or pts < 0 or pts <= previous:
            raise ValueError('Frozen source timestamps must be finite and strictly increasing')
        previous = pts
        for row in frame['detections']:
            if (len(row) != 6 or not all(math.isfinite(v) for v in row)
                    or row[2] <= row[0] or row[3] <= row[1]
                    or not 0 <= row[4] <= 1 or row[5] not in (2, 3, 5, 7)):
                raise ValueError('Invalid frozen vehicle detection')


def validate_alignment(frame, frozen, source, document):
    if (frame.pts_ms is None or frame.pts_ms-source.start_pts_ms != frozen['position_ms']
            or bool(frame.discontinuity) != bool(frozen['discontinuity'])
            or tuple(frame.image.shape[:2]) != (document['height'], document['width'])):
        raise ValueError('Native image does not exactly align with frozen detection PTS/shape/discontinuity')


class CountedEncoder:
    """Reject missing/invalid embeddings, rather than silently falling back to geometry."""
    def __init__(self, encoder):
        self.encoder = encoder
        self.calls = self.features = 0
        self.dimension = None

    def __call__(self, image, boxes):
        import numpy as np
        features = self.encoder(image, boxes)
        if len(features) != len(boxes):
            raise ValueError('Appearance encoder omitted candidates')
        for feature in features:
            array = np.asarray(feature)
            if array.ndim != 1 or not array.size or not np.isfinite(array).all() or np.linalg.norm(array) <= 1e-12:
                raise ValueError('Appearance encoder returned an invalid/empty feature')
            if self.dimension is not None and self.dimension != array.size:
                raise ValueError('Appearance feature dimensions changed')
            self.dimension = int(array.size)
        self.calls += 1
        self.features += len(features)
        return features


def encoder_identity(encoder, path):
    from sentinel_edge.vehicle_reid import VehicleAppearanceEncoder
    if isinstance(encoder,VehicleAppearanceEncoder):
        return dict(path=str(path),sha256=sha(path),kind='VehicleAppearanceEncoder',backend='ONNXRuntime',device='cuda:0',
                    providers=encoder.session.get_providers(),input_size=208,preprocessing='Original RGB float32 0..255, no external normalization',
                    source='https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/vehicle-reid-0001')
    backend = encoder.model
    device = str(getattr(backend, 'device', 'unknown'))
    session = getattr(backend, 'session', None)
    providers = session.get_providers() if session is not None else None
    if not str(encoder.device).startswith('cuda') or not device.startswith('cuda'):
        raise ValueError('Learned appearance encoder must actually use CUDA')
    if providers is not None and (not providers or providers[0] not in ('CUDAExecutionProvider', 'TensorrtExecutionProvider')):
        raise ValueError('ONNX appearance encoder did not select a GPU execution provider')
    return dict(path=str(path), sha256=sha(path), kind=type(encoder).__name__,
                backend=type(backend).__name__, device=device, providers=providers,
                is_pt=encoder.is_pt, input_size=encoder.imgsz,
                note='Learned appearance embedding model; not a second vehicle detection pass.')


def run(args):
    import numpy as np
    import torch
    import ultralytics
    import yaml
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.trackers.utils import reid

    if args.output.exists():
        raise ValueError('Use a fresh immutable output directory')
    document = read_detections(args.capture_dir)
    validate_capture(document)
    if ultralytics.__version__ != document['ultralytics_version']:
        raise ValueError('Tracker runtime differs from frozen capture')
    if not torch.cuda.is_available():
        raise ValueError('A working CUDA device is required for learned appearance replay')
    encoder_path = args.encoder.resolve()
    if not encoder_path.is_relative_to(args.models.resolve()) or not encoder_path.is_file():
        raise ValueError('An explicit existing encoder file inside the model cache is required')
    recordings, _ = RecordingLibrary(args.recordings_dir).scan()
    source = recordings.get(document['recording_id'])
    if source is None or source.sha256 != document['source_sha256'] or source.path.name != document['source_name']:
        raise ValueError('Permitted native source differs from frozen capture')
    torch.set_num_threads(2)
    cfg_root = Path(ultralytics.__file__).parent/'cfg/trackers'
    configs = {name: yaml.safe_load((cfg_root/f'{name}.yaml').read_text()) for name in ('bytetrack', 'botsort')}
    args.output.mkdir(parents=True)
    (args.output/'comparison-script.py').write_bytes(Path(__file__).read_bytes())
    common = {k: v for k, v in document.items() if k != 'frames'}
    common.update(detections_sha256=sha(args.capture_dir/'detections.json.gz'),
                  source='recorded', source_start_pts_ms=source.start_pts_ms,
                  replay_script_sha256=sha(__file__), policy_source_sha256=sha(inspect.getfile(VehicleTrackPolicy)),
                  tracker_source_sha256={c.__name__: sha(inspect.getfile(c)) for c in (BYTETracker, BOTSORT)},
                  encoder_source_sha256=sha(inspect.getfile(reid)),
                  identity_accuracy=None, detection_precision=None, recall=None,
                  limitations=['Diagnostic output counts and review timelines are not MOT or physical-identity accuracy.',
                               'No vehicle identity labels are used to tune or select tracker parameters.',
                               'Both BoT-SORT variants disable camera-motion compensation for this fixed camera.',
                               'All strategies use exactly the same frozen detections and native source PTS.',
                               'Appearance runtime includes native crop encoding; detector inference is excluded.',
                               'No OCR, plate detector, central delivery or new vehicle detection pass is performed.'])
    results = {}
    for mode in args.modes:
        with_reid = mode == 'botsort_reid'
        config_name = 'bytetrack' if mode == 'bytetrack_screened' else 'botsort'
        config = {**configs[config_name], 'new_track_thresh': .4}
        if config_name == 'botsort':
            config.update(gmc_method='none', with_reid=with_reid,
                          model=str(encoder_path) if with_reid else 'auto', device='cuda:0')
        torch.cuda.synchronize()
        load_started = time.monotonic()
        vehicle_encoder=with_reid and args.encoder_kind=='vehicle'
        initialization={**config,'with_reid':False} if vehicle_encoder else config
        tracker = (BYTETracker if config_name == 'bytetrack' else BOTSORT)(SimpleNamespace(**initialization))
        if vehicle_encoder:
            from sentinel_edge.vehicle_reid import VehicleAppearanceEncoder
            tracker.args.with_reid=True;tracker.encoder=VehicleAppearanceEncoder(encoder_path)
        torch.cuda.synchronize()
        loading_seconds = time.monotonic()-load_started
        appearance, counter = None, None
        if with_reid:
            if not vehicle_encoder and not isinstance(tracker.encoder, reid.ReID):
                raise ValueError('Learned encoder was not loaded; native-feature fallback is forbidden')
            appearance = encoder_identity(tracker.encoder, encoder_path)
            appearance['adapter_source_sha256']=sha(inspect.getfile(type(tracker.encoder)))
            counter = CountedEncoder(tracker.encoder)
            tracker.encoder = counter
        policy, rows, timings, native_ids = VehicleTrackPolicy(), [], [], set()
        raw_count = qualified_count = segment = processed = 0
        began = time.monotonic()
        for _, frame in read_recording(source, threading.Event(), paced=False):
            if processed >= len(document['frames']):
                raise ValueError('Native source has additional frames beyond frozen capture')
            frozen = document['frames'][processed]
            validate_alignment(frame, frozen, source, document)
            pts = frozen['position_ms']
            if frame.discontinuity:
                tracker.reset()
                segment += 1
            boxes = Boxes(np.asarray(frozen['detections'], dtype=np.float32).reshape(-1, 6),
                          orig_shape=(document['height'], document['width']))
            torch.cuda.synchronize()
            started = time.monotonic()
            tracked = tracker.update(boxes, frame.image)
            torch.cuda.synchronize()
            timings.append((time.monotonic()-started)*1000)
            observations = [dict(native_id=f'{segment}:{int(t[4])}', box=[float(v) for v in t[:4]],
                                 confidence=float(t[5]), vehicle_class=int(t[6])) for t in tracked]
            visible = policy.update(observations, pts)
            native_ids.update(t['native_id'] for t in observations)
            raw_count += len(observations)
            qualified_count += sum(t['status'] == 'qualified' for t in visible)
            rows.append(dict(t=pts, q=[pack(t) for t in visible]))
            processed += 1
            if processed % 100 == 0:
                progress = dict(stage='replaying', strategy=mode, frames=processed)
                write_json(args.output/'progress.json', progress)
                print(json.dumps(progress), flush=True)
        if processed != document['decoded_frames']:
            raise ValueError('Native source ended before frozen capture')
        if counter is not None and not counter.features:
            raise ValueError('ReID strategy never produced learned appearance features')
        records = policy.finish(rows[-1]['t'])
        for record in records:
            record['preview_paths'] = []
        result = dict(updates=processed, tracker_config=config, tracker_config_sha256=config_sha(config),
                      base_config_sha256=sha(cfg_root/f'{config_name}.yaml'),
                      native_track_ids=len(native_ids), passage_records=len(records),
                      qualified_passages=sum(r['status'] == 'qualified' for r in records),
                      tracked_observation_instances=raw_count, qualified_observation_instances=qualified_count,
                      qualification_excluded_instances=raw_count-qualified_count,
                      replay_wall_seconds=round(time.monotonic()-began, 3),
                      model_load_seconds=round(loading_seconds, 3),
                      tracker_seconds=round(sum(timings)/1000, 3),
                      median_tracker_ms=round(float(np.median(timings)), 3),
                      p95_tracker_ms=round(float(np.percentile(timings, 95)), 3),
                      appearance=appearance, encoder_calls=counter.calls if counter else 0,
                      encoded_features=counter.features if counter else 0,
                      feature_dimension=counter.dimension if counter else None,
                      policy_counters=policy.counters, records=records)
        results[mode] = result
        target = args.output/mode
        target.mkdir()
        write_json(target/'timeline.json', dict(width=document['width'], height=document['height'], frames=rows))
        write_json(target/'report.json', {**common, 'selected_strategy': mode, 'strategies': {mode: result}})
        del tracker, counter
        torch.cuda.empty_cache()
    write_json(args.output/'report.json', {**common, 'strategies': results})
    write_json(args.output/'progress.json', dict(stage='complete', strategies=list(results), frames=document['decoded_frames']))
    print(json.dumps(dict(stage='complete', summaries={m: {k: v for k, v in r.items() if k != 'records'}
                                                        for m, r in results.items()})), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', required=True, type=Path)
    parser.add_argument('--recordings-dir', type=Path, default=Path('/recordings'))
    parser.add_argument('--models', type=Path, default=Path('/models'))
    parser.add_argument('--encoder', required=True, type=Path)
    parser.add_argument('--encoder-kind',choices=('ultralytics','vehicle'),default='ultralytics')
    parser.add_argument('--modes',nargs='+',choices=('bytetrack_screened','botsort_motion','botsort_reid'),default=['bytetrack_screened','botsort_motion','botsort_reid'])
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
