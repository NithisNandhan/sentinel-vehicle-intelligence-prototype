"""Paired development-only crop/OCR/reconstruction failure curve. No live media."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from datetime import datetime, timezone

from sentinel_edge.benchmark_viewer import render_viewer
from sentinel_edge.ctc_fusion import CTCPlateReader, greedy_text
from sentinel_edge.degradation_benchmark import validate_manifest, _frames, _scaled_box, degrade, degradation_levels
from sentinel_edge.detector_ocr_shadow import load_ocr
from sentinel_edge.experimental_detectors import HASHES, PlateDetector, VehicleTracker
from sentinel_edge.live_reconstruction import ReconstructionState, update, disagreement
from sentinel_edge.plate_quality import crop_quality
from sentinel_edge.plate_search import TargetedPlateSearch
from sentinel_edge.reconstruction import CropBank, native_crop
from sentinel_edge.tracked_ocr import TrackedOCR


def evaluate(reader, state, truth):
    candidates = reader.candidates() if reader.observation_count >= 3 else []
    accepted = reader.accepted() if candidates else None
    if disagreement(state,accepted,.85):
        accepted = None
    return candidates, accepted, {
        'correct': bool(accepted and accepted.plate == truth),
        'wrong': bool(accepted and accepted.plate != truth),
        'abstain': accepted is None,
        'truth_rank': next((i+1 for i,c in enumerate(candidates) if c.plate == truth),None),
        'observations': reader.observation_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',default='crop-collection-20260928')
    args = parser.parse_args()
    if not args.name.replace('-','').isalnum():
        raise ValueError('Simple unique report name required')
    root = Path('/workspace/data/benchmarks/nivu')
    report = root.parent/'reports'/f'{args.name}.json'
    viewer = report.with_suffix('.html')
    if report.exists() or viewer.exists():
        raise ValueError('Existing experiment preserved; choose a new name')
    manifest = root/'manifest.json'
    document = json.loads(manifest.read_text())
    passages = validate_manifest(document,root)
    if len(passages)!=1 or len(passages[0]['frames'])!=3:
        raise ValueError('Frozen development sequence required')
    passage = passages[0]
    material = list(_frames(root,passage))
    ocr = load_ocr(Path('/models/experiments/awiros'),Path('/models/experiments/awiros/adaptations/indian-corpus-encoder.safetensors'))
    vehicle = VehicleTracker('yolo26s',Path('/models'),Path('/training'))
    detector = PlateDetector('parent',Path('/models'))
    rows, visuals = [], []
    for spec in degradation_levels():
        vehicle.reset()
        pipelines = {
            'automatic_original': TrackedOCR(ocr,detector,roi_budget=8,debug=True),
            'automatic_targeted_reconstruction': TrackedOCR(ocr,TargetedPlateSearch(detector),roi_budget=8,debug=True,reconstruction=True),
        }
        readers = {name:CTCPlateReader(ocr.tokens) for name in ('oracle_original','oracle_reconstruction')}
        bank = CropBank(retain_best=True,max_age_ms=2500)
        recon = ReconstructionState()
        crops = {name:[] for name in (*pipelines,*readers)}
        localisation = {name:[] for name in pipelines}
        durations = {name:[] for name in crops}
        started = time.perf_counter()
        for index,(original,pts,annotation) in enumerate(material):
            image = degrade(original,spec,index)
            expected = _scaled_box(annotation['plate_box'],original.shape,image.shape)
            tracks = vehicle.track(image)
            cache = {}
            for name,pipeline in pipelines.items():
                tick = time.perf_counter()
                displays = pipeline.process(image,tracks,pts,posterior_cache=cache)
                durations[name].append((time.perf_counter()-tick)*1000)
                # Target box is used only for localisation scoring after inference.
                from sentinel_edge.ocr_selection_benchmark import box_iou
                localisation[name].append(max((box_iou(d['box'],expected) for d in displays),default=0.))
                crops[name].extend(d['crop'] for d in displays)
            crop = native_crop(image,expected)
            quality = crop_quality(crop)
            source = hashlib.sha256(crop.tobytes()).hexdigest()
            tick = time.perf_counter()
            probabilities = ocr.probabilities(crop)
            duration = (time.perf_counter()-tick)*1000
            bank.add('oracle',crop,pts,quality=quality)
            for name,reader in readers.items():
                reader.add(probabilities,timestamp_ms=pts,source_id=source,quality=quality)
                crops[name].append(crop)
                durations[name].append(duration)
            update(recon,readers['oracle_reconstruction'],bank.get('oracle'),ocr,pts)
        for name in crops:
            state = None
            frame_views = []
            if name in pipelines:
                pipeline = pipelines[name]
                states = list(pipeline.histories.values())
                scored = [(*evaluate(h.reader,h.reconstruction,passage['ground_truth']),h) for h in states]
                # Aggregate every acceptance: a good target cannot hide a wrong one.
                correct = any(item[2]['correct'] for item in scored)
                wrong = sum(item[2]['wrong'] for item in scored)
                eligible = [item for item in scored if item[0]]
                selected = max(eligible,key=lambda item:item[0][0].log_score,default=None)
                candidates,accepted,state = (selected[0],selected[1],selected[3].reconstruction) if selected else ([],None,None)
                diagnostic = dict(pipeline.summary(),correct=correct,wrong=wrong,abstain=not correct and not wrong,
                                  truth_rank=next((i+1 for i,c in enumerate(candidates) if c.plate==passage['ground_truth']),None),
                                  target_plate_iou=localisation[name],oracle_crops=False)
                evidence = [(h.reader,pipeline.crop_bank.get(track)) for track,h in pipeline.histories.items()]
            else:
                state = recon if name=='oracle_reconstruction' else None
                candidates,accepted,diagnostic = evaluate(readers[name],state,passage['ground_truth'])
                diagnostic['oracle_crops'] = True
                evidence = [(readers[name],bank.get('oracle'))]
            for reader,samples in evidence:
                lookup = {sample.source_id:sample for sample in samples}
                for pts,source,probabilities,_,quality in reader.samples:
                    sample = lookup.get(source)
                    if sample is not None:
                        frame_views.append(dict(image=sample.image,pts_ms=pts,index=len(frame_views),
                            reading=greedy_text(probabilities,ocr.tokens),quality=round(quality,3),
                            width=sample.image.shape[1],height=sample.image.shape[0],ocr_ms='paired cache'))
            widths = [crop.shape[1] for crop in crops[name]]
            row = dict(condition=spec['name'],method=name,**diagnostic,
                       median_native_width=statistics.median(widths) if widths else None,
                       median_stage_ms=round(statistics.median(durations[name]),2),
                       reconstruction_status=state.status if state else 'not_run')
            rows.append(row)
            visuals.append(dict(source_mode='benchmark',title=f"{spec['name']} · {name}",
                original=crops[name][-1] if crops[name] else None,candidates=candidates,
                accepted=accepted.plate if accepted else None,expected=passage['ground_truth'],
                outcome='wrong' if row['wrong'] else 'correct' if row['correct'] else 'abstain',
                reconstruction=state.result.image if state and state.result else None,
                source_count=len(state.result.source_ids) if state and state.result else 0,
                reason='One previously used development vehicle; oracle crops are NOT product accuracy.',
                duration_ms=row['median_stage_ms'],dimensions=list(crops[name][-1].shape[1::-1]) if crops[name] else None,
                frame_diagnostics=frame_views,
                score_note='Uncalibrated scores. Derived suggestions are rescored only against original observations.'))
        print(json.dumps({'condition':spec['name'],'elapsed_seconds':round(time.perf_counter()-started,2),
                          'methods':[{'method':r['method'],'correct':r['correct'],'wrong':r['wrong'],'truth_rank':r['truth_rank']} for r in rows[-4:]]}),flush=True)
    result = dict(source_mode='benchmark',unique_vehicles=1,original_frames=3,conditions=21,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        detector_hashes={name:HASHES[name] for name in ('yolo26s','parent')},
        timing_mode='assigned_replay',held_out_accuracy_claimed=False,live_validation=False,
        crop_collection_queue_evaluated=False,model_retrained=False,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        caveat='Paired crop/OCR/reconstruction ablation on one reused Indian development sequence. Queue timing is tested separately. Stage timings share cached posteriors and are not standalone throughput.',
        rows=rows,visuals=viewer.name)
    render_viewer(viewer,document['dataset'],visuals)
    report.write_text(json.dumps(result,indent=2))
    print(json.dumps({'report':report.name,'viewer':viewer.name,'rows':len(rows)}),flush=True)


if __name__=='__main__':
    main()
