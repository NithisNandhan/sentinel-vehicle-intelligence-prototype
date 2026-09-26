"""Native-crop multi-frame reader comparison, isolated from live delivery.

Only independently labelled permitted benchmark inputs are allowed. Aggregate
JSON never contains plate text; detailed readings appear only in the ignored
local HTML viewer. No detector, registry, restoration or training is involved.
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from .benchmark_viewer import render_viewer, validate_viewer_scope
from .ctc_fusion import CTCPlateReader, greedy_text
from .degradation_benchmark import (_frames, _scaled_box, degrade, degradation_levels,
                                    score_passage, validate_annotation_bounds,
                                    validate_manifest, validate_sequence)
from .reconstruction import native_crop


class CCTExperiment:
    def __init__(self):
        from fast_plate_ocr import LicensePlateRecognizer
        self.model = LicensePlateRecognizer('cct-s-v2-global-model', device='cpu',
                                            providers=['CPUExecutionProvider'])
        self.config = self.model.config
        self.device = 'cpu'

    def probabilities(self, image):
        import cv2
        from fast_plate_ocr.inference.plate_recognizer import _load_image_from_source, preprocess_image
        cfg = self.config
        if cfg.image_color_mode == 'grayscale':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif cfg.image_color_mode == 'rgb':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        batch = preprocess_image(_load_image_from_source(image, cfg))
        output = self.model.model.run([self.model.plate_output_name], {'input': batch})
        return output[0].reshape((-1, cfg.max_plate_slots, len(cfg.alphabet)))[0].tolist()


def run_selection_benchmark(manifest_path: Path, cache: Path, *, backend='awiros', family='all',
                            report_path: Path, viewer_path: Path, device='cpu'):
    import numpy as np
    document = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    root = manifest_path.parent.resolve()
    passages = validate_manifest(document, root)
    validate_viewer_scope(viewer_path, document['dataset'])
    if report_path.suffix.lower() != '.json':
        raise ValueError('Aggregate OCR report must be JSON')
    validate_viewer_scope(report_path.with_suffix('.html'), document['dataset'])
    if len(passages) > 20:
        raise ValueError('Short diagnostic reader comparison only; at most twenty passages')
    specs = [spec for spec in degradation_levels() if family == 'all' or spec['family'] == family]
    if not specs or len(specs) * len(passages) > 100:
        raise ValueError('Invalid or oversized comparison')
    loaded = time.perf_counter()
    if backend == 'awiros':
        from .awiros_experiment import AwirosExperiment, MODEL_REVISION
        ocr = AwirosExperiment(cache / 'experiments' / 'awiros', device=device)
        revision = MODEL_REVISION
    elif backend == 'parseq':
        import sys
        sys.path.insert(0, str(cache / 'experiments' / 'deps'))
        os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
        from .parseq_experiment import PARSeqExperiment, REVISION
        ocr = PARSeqExperiment(cache / 'experiments', cached_only=True)
        revision = REVISION
    elif backend == 'cct':
        ocr = CCTExperiment()
        revision = 'cct-s-v2-global-model'
    else:
        raise ValueError('Unknown OCR comparison backend')
    loading_ms = (time.perf_counter() - loaded) * 1000
    ocr.probabilities(np.zeros((32, 128, 3), dtype=np.uint8))
    report_rows, visuals, inference_times = [], [], []
    for passage_index, passage in enumerate(passages):
        material, total, previous = [], 0, None
        for image, pts, annotation in _frames(root, passage):
            if previous is not None and pts - previous < 300:
                continue
            previous = pts
            validate_annotation_bounds(annotation, image.shape)
            if not annotation.get('plate_box'):
                raise ValueError('Independent plate boxes required for OCR diagnostic')
            total += image.nbytes
            if total > 384 * 1024 * 1024 or len(material) >= 40:
                raise ValueError('Short bounded OCR diagnostic passage required')
            material.append((image, pts, annotation))
        validate_sequence([pts for _, pts, _ in material],
                          [hashlib.sha256(image.tobytes()).hexdigest() for image, _, _ in material])
        for spec in specs:
            if backend == 'awiros':
                reader = CTCPlateReader(ocr.tokens)
            else:
                from .sequence_fusion import SequencePlateReader
                from .temporal import TemporalPlateReader
                cfg = ocr.config
                reader = (SequencePlateReader(cfg.alphabet, cfg.pad_char, slots=cfg.max_plate_slots)
                          if backend == 'parseq' else TemporalPlateReader(cfg.alphabet, cfg.pad_char))
            started = time.perf_counter()
            frame_views, frame_numeric, widths = [], [], []
            displayed = None
            for index, (original, pts, annotation) in enumerate(material):
                image = degrade(original, spec, seed=passage_index * 100000 + index)
                box = _scaled_box(annotation['plate_box'], original.shape, image.shape)
                crop = native_crop(image, box)
                displayed = crop
                widths.append(crop.shape[1])
                tick = time.perf_counter()
                matrix = ocr.probabilities(crop)
                elapsed = (time.perf_counter() - tick) * 1000
                inference_times.append(elapsed)
                if backend == 'awiros':
                    text = greedy_text(matrix, ocr.tokens)
                    reader.add(matrix, timestamp_ms=pts,
                               source_id=hashlib.sha256(original.tobytes()).hexdigest())
                else:
                    cfg = ocr.config
                    text = ''.join(cfg.alphabet[max(range(len(row)), key=row.__getitem__)]
                                   for row in matrix).split(cfg.pad_char, 1)[0]
                    reader.add(matrix, timestamp_ms=pts)
                frame_numeric.append({'index': index, 'pts_ms': pts, 'native_width': crop.shape[1],
                                      'native_height': crop.shape[0], 'ocr_ms': round(elapsed, 2),
                                      'greedy_exact': text == passage['ground_truth']})
                frame_views.append({'image': crop, 'index': index, 'reading': text or 'empty',
                                    'pts_ms': pts, 'ocr_ms': round(elapsed, 2)})
            accepted = reader.accepted(.85)
            candidates = reader.candidates()
            duration = (time.perf_counter() - started) * 1000
            # Scoring only happens here: labels never enter a model or decoder.
            scored = score_passage({passage['ground_truth']}, {accepted.plate} if accepted else set())
            truth_rank = next((rank for rank, candidate in enumerate(candidates, 1)
                               if candidate.plate == passage['ground_truth']), None)
            reason = ('wrong_accepted_identifier' if scored['wrong'] else
                      'insufficient_temporal_evidence' if reader.observation_count < 3 else
                      'true_identifier_absent_from_candidate_beam' if truth_rank is None else
                      'candidate_ranking_ambiguity' if truth_rank != 1 else
                      'acceptance_evidence_below_threshold') if not scored['correct'] else None
            row = {'passage_index': passage_index, 'split': passage['split'],
                   'condition': spec['name'], **scored, 'truth_rank': truth_rank,
                   'failure_reason': reason, 'observations': reader.observation_count,
                   'native_width_median': statistics.median(widths),
                   'processing_ms': round(duration, 2), 'frame_diagnostics': frame_numeric,
                   'best_character_evidence': candidates[0].score if candidates else None,
                   'weakest_character_evidence': candidates[0].minimum_character_score if candidates else None}
            report_rows.append(row)
            visuals.append({'source_mode': 'benchmark',
                            'title': f'Native OCR diagnostic · {backend} · Passage {passage_index + 1} · {spec["name"]}',
                            'original': displayed, 'candidates': candidates,
                            'accepted': accepted.plate if accepted else None, 'reason': reason or 'original_frame_evidence_passed',
                            'dimensions': list(displayed.shape[:2][::-1]) if displayed is not None else None,
                            'duration_ms': round(duration, 2), 'outcome': scored['outcome'],
                            'expected': passage['ground_truth'], 'frame_diagnostics': frame_views,
                            'score_note': ('CTC ranks by mean per-frame sequence log-likelihood. Character evidence uses best-path peaks; '
                                           'it is not calibrated confidence and is not equivalent to fixed-slot model scores.'
                                           if backend == 'awiros' else 'Existing reader evidence gates; model scores remain uncalibrated.')})
            print(json.dumps({'backend': backend, 'condition': spec['name'], **scored,
                              'truth_rank': truth_rank, 'processing_ms': round(duration, 2)}), flush=True)
    report = {'evaluation': 'native-crop multi-frame OCR selection diagnostic',
              'generated_at': datetime.now(timezone.utc).isoformat(),
              'backend': backend, 'model_revision': revision, 'device': ocr.device,
              'further_fine_tuned_here': False, 'loading_ms': round(loading_ms, 2),
              'warmup_excluded': True, 'stage': 'oracle_plate_ocr',
              'automatic_detection_evaluated': False, 'live_timing_validated': False,
              'confidence_is_accuracy': False, 'held_out_accuracy_claimed': False,
              'unique_vehicles': len({p['vehicle_id'] for p in passages}),
              'timing_modes': sorted({p.get('timing_mode', 'verified_pts') for p in passages}),
              'dataset_license': document['dataset']['license'], 'family': family,
              'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              'ocr_median_ms': statistics.median(inference_times),
              'correct': sum(row['correct'] for row in report_rows),
              'wrong': sum(row['wrong'] for row in report_rows),
              'abstain': sum(row['abstain'] for row in report_rows), 'rows': report_rows}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    render_viewer(viewer_path, document['dataset'], visuals)
    print(json.dumps({key: report[key] for key in ('backend', 'device', 'correct', 'wrong', 'abstain', 'ocr_median_ms')}), flush=True)
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Permitted local native-crop multi-frame reader comparison')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--backend', choices=('awiros', 'parseq', 'cct'), default='awiros')
    parser.add_argument('--family', choices=('resolution', 'blur', 'motion', 'noise', 'darkness', 'all'), default='all')
    parser.add_argument('--device', choices=('cpu', 'gpu'), default='cpu')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--viewer', type=Path, required=True)
    args = parser.parse_args()
    run_selection_benchmark(args.manifest, args.cache, backend=args.backend, family=args.family,
                            report_path=args.report, viewer_path=args.viewer, device=args.device)
