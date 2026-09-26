"""Matched multi-frame hybrid benchmark; labels never enter inference."""
from __future__ import annotations

import hashlib
import json
import statistics
import time
from pathlib import Path

from .benchmark_viewer import render_viewer, validate_viewer_scope
from .degradation_benchmark import (validate_manifest, validate_sequence, validate_annotation_bounds,
                                    _frames, degrade, degradation_levels, _scaled_box, _target_box,
                                    score_passage, summarize_outcomes)
from .hybrid import HybridEngine, HybridResources
from .reconstruction import native_crop
from .temporal import TemporalPlateReader
from .vision import VisionPipeline


def run_hybrid_benchmark(config, manifest_path: Path, *, family='resolution', alignment='ecc',
                         restoration=False, output_path=None, viewer_path=None):
    import cv2
    import numpy as np
    import torch
    document = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    root = manifest_path.parent.resolve()
    passages = validate_manifest(document, root)
    if viewer_path:
        validate_viewer_scope(viewer_path, document['dataset'])
    specs = [s for s in degradation_levels() if family == 'all' or s['family'] == family]
    if not specs:
        raise ValueError('Unknown degradation family')
    resources = HybridResources(config.model_cache_dir / 'experiments', alignment=alignment, restoration=restoration)
    baseline = VisionPipeline(config.model_cache_dir / 'yolov8n.pt')
    hybrid = VisionPipeline(config.model_cache_dir / 'yolov8n.pt', hybrid_resources=resources)
    # Warm-up is separate from timed passages and is not a readable plate test.
    resources.probabilities(np.zeros((32, 128, 3), np.uint8))
    for pipeline in (baseline, hybrid):
        pipeline.process(np.zeros((640, 640, 3), np.uint8), pts_ms=0)
        pipeline._probabilities(np.zeros((32, 128, 3), np.uint8))
        pipeline.reset()
    if resources.aligner:
        warm = np.random.default_rng(0).integers(0, 256, (32, 128, 3), dtype=np.uint8)
        resources.aligner(warm, warm.copy())
    if resources.restorer:
        resources.restorer(np.zeros((32, 128, 3), np.uint8))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    cfg = baseline.alpr.ocr.ocr_model.config
    rows, visuals = [], []
    for passage_index, passage in enumerate(passages):
        material, bytes_used, previous = [], 0, None
        for image, pts, annotation in _frames(root, passage):
            if previous is not None and pts - previous < 1000 / config.inference_fps:
                continue
            previous = pts
            validate_annotation_bounds(annotation, image.shape)
            bytes_used += image.nbytes
            if bytes_used > 384 * 1024 * 1024 or len(material) >= 1800:
                raise ValueError('Passage exceeds bounded benchmark memory')
            material.append((image, pts, annotation))
        validate_sequence([p for _, p, _ in material], [hashlib.sha256(i.tobytes()).hexdigest() for i, _, _ in material])
        height, width = max(i.shape[0] for i, _, _ in material), max(i.shape[1] for i, _, _ in material)
        material = [(cv2.copyMakeBorder(i, 0, height - i.shape[0], 0, width - i.shape[1], cv2.BORDER_CONSTANT), p, a)
                    for i, p, a in material]
        if any(not a.get('vehicle_box') or not a.get('plate_box') for _, _, a in material):
            raise ValueError('Matched comparison needs independently annotated vehicle/plate boxes')
        for spec in specs:
            for stage in ('oracle_plate_ocr', 'end_to_end'):
                for mode in ('temporal-v2', 'hybrid-v1'):
                    pipeline = baseline if mode == 'temporal-v2' else hybrid
                    pipeline.reset()
                    pipeline.last_plate_seen.clear()
                    pipeline.metrics.clear()
                    pipeline.hybrid and pipeline.hybrid.counters.clear()
                    reader = TemporalPlateReader(cfg.alphabet, cfg.pad_char)
                    engine = HybridEngine(resources, threshold=config.ocr_min_confidence, retain_visuals=bool(viewer_path))
                    predicted, candidates, target_tracks = set(), [], {}
                    last_ocr, last_original = None, None
                    widths = []
                    started = time.perf_counter()
                    for frame_index, (original, pts, annotation) in enumerate(material):
                        if last_ocr is not None and pts - last_ocr < 300:
                            continue
                        last_ocr = pts
                        image = degrade(original, spec, seed=passage_index * 100000 + frame_index)
                        scaled = {k: _scaled_box(annotation[k], original.shape, image.shape)
                                  for k in ('plate_box', 'vehicle_box')}
                        box = scaled['plate_box']
                        widths.append(box[2] - box[0])
                        if stage == 'end_to_end':
                            for observation in pipeline.process(image, pts_ms=pts):
                                if _target_box(observation.vehicle_bbox, scaled):
                                    predicted.add(observation.plate)
                            for track_id, track in pipeline.tracks.items():
                                if track.last_bbox_pts == pts and _target_box(track.vehicle_bbox, scaled):
                                    target_tracks[track_id] = track
                            if len(target_tracks) > 128:
                                del target_tracks[next(iter(target_tracks))]
                            continue
                        if box[2] - box[0] < 20 or box[3] - box[1] < 6:
                            continue
                        crop = native_crop(image, box)
                        last_original = crop
                        if mode == 'hybrid-v1':
                            accepted = engine.process('target', crop, pts)
                        else:
                            reader.add(pipeline._probabilities(crop), timestamp_ms=pts)
                            accepted = reader.accepted(config.ocr_min_confidence)
                        if accepted:
                            predicted.add(accepted.plate)
                    duration = (time.perf_counter() - started) * 1000
                    target_readers = ([engine.tracks.get('target')] if mode == 'hybrid-v1' else [reader]) if stage == 'oracle_plate_ocr' else [t.reader for t in target_tracks.values()]
                    target_readers = [r for r in target_readers if r is not None]
                    candidates = sorted((c for r in target_readers for c in r.candidates()), key=lambda c: -c.log_score)[:200]
                    best = candidates[0] if candidates else None
                    state = engine.tracks.get('target') if stage == 'oracle_plate_ocr' and mode == 'hybrid-v1' else None
                    counters = dict(engine.counters if stage == 'oracle_plate_ocr' else pipeline.metrics)
                    if stage == 'end_to_end' and pipeline.hybrid:
                        counters.update(pipeline.hybrid.counters)
                    reconstructed = counters.get('reconstructed_windows', 0) > 0
                    deblurred = counters.get('restored_windows', 0) > 0
                    evidence = max((r.observation_count for r in target_readers), default=0)
                    scored = score_passage({passage['ground_truth']}, predicted)
                    in_beam = any(c.plate == passage['ground_truth'] for c in candidates)
                    reason = None if scored['correct'] else ('wrong_accepted_identifier' if scored['wrong'] else
                             'insufficient_temporal_evidence' if evidence < 3 else
                             'true_identifier_absent_from_candidate_beam' if not in_beam else
                             'acceptance_evidence_below_threshold')
                    rows.append({'passage_index': passage_index, 'stage': stage,
                                 'variant': mode + ':' + spec['name'], 'split': passage['split'],
                                 **scored, 'failure_reason': reason, 'duration_ms': round(duration, 2),
                                 'probability_observations': evidence,
                                 'ground_truth_in_final_candidate_beam': in_beam,
                                 'ground_truth_is_final_top_candidate': bool(best and best.plate == passage['ground_truth']),
                                 'best_candidate_score': best.score if best else None,
                                 'plate_width_median_px': statistics.median(widths) if widths else None,
                                 'reconstructed': reconstructed, 'deblurred': deblurred,
                                 'counters': counters})
                    if viewer_path and stage == 'oracle_plate_ocr':
                        result = state.reconstruction if state else None
                        displayed_original = result.reference_image if result else last_original
                        visuals.append({'source_mode': 'benchmark', 'title': f'Manually annotated plate crops · Passage {passage_index + 1} · {mode} · {spec["name"]}',
                                        'original': result.reference_image if result else last_original,
                                        'reconstruction': result.image if result else None,
                                        'restored': state.restored if state else None,
                                        'candidates': candidates,
                                        'processed_candidates': state.processed_candidates if state else [],
                                        'accepted': next(iter(predicted)) if len(predicted) == 1 else None,
                                        'partial': state.partial() if state else '', 'reason': state.reason if state else reason,
                                        'source_count': len(result.source_ids) if result else 0,
                                        'rejected': result.rejected if result else 0,
                                        'dimensions': list(displayed_original.shape[:2][::-1]) if displayed_original is not None else None,
                                        'duration_ms': round(duration, 2), 'outcome': scored['outcome']})
                        if len(visuals) > 100:
                            raise ValueError('Too many visual cases; shorten benchmark')
                    print(json.dumps({'passage_index': passage_index, 'stage': stage, 'variant': mode,
                                      'condition': spec['name'], 'outcome': scored['outcome'],
                                      'reconstructed': reconstructed, 'deblurred': deblurred}), flush=True)
    report = {'evaluation': 'matched multi-frame hybrid comparison', 'family': family,
              'alignment': alignment, 'restoration_requested': restoration,
              'hybrid_model_version': resources.model_version, 'passages': len(passages),
              'live_timing_validated': False, 'oracle_stage_is_not_product_performance': True,
              'warmup_excluded': True, 'memory_excludes_onnx_driver_and_non_torch_allocations': True,
              'dataset_license': document['dataset']['license'],
              'peak_torch_gpu_memory_mb': torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else None,
              'rows': rows, **summarize_outcomes(rows)}
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if viewer_path:
        render_viewer(viewer_path, document['dataset'], visuals)
    print(json.dumps({'conditions': len(rows), 'correct': sum(r['correct'] for r in rows),
                      'wrong': sum(r['wrong'] for r in rows), 'abstain': sum(r['abstain'] for r in rows),
                      'reconstructed': sum(r['reconstructed'] for r in rows),
                      'deblurred': sum(r['deblurred'] for r in rows)}), flush=True)
    return report
