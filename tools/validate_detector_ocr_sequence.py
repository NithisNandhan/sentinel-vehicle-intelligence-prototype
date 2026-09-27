"""Automatic tracked OCR on the licensed three-frame NIVU development sequence."""
import hashlib
import json
from pathlib import Path
import time

import cv2

from sentinel_edge.benchmark_viewer import render_viewer
from sentinel_edge.ctc_fusion import greedy_text
from sentinel_edge.degradation_benchmark import validate_manifest
from sentinel_edge.detector_ocr_shadow import load_ocr,load_profiles
from sentinel_edge.experimental_detectors import PROFILES


def main(name='detector-ocr-sequence'):
    if not name.replace('-','').isalnum():raise ValueError('Simple experiment name required')
    root = Path('/workspace/data/benchmarks/nivu')
    report_root = root.parent/'reports'
    manifest_path = root/'manifest.json'
    document = json.loads(manifest_path.read_text())
    document['dataset']['modifications'] = 'Automatic native crops and tracked CTC fusion; no image reconstruction or deblurring'
    passages = validate_manifest(document,root)
    if len(passages) != 1 or len(passages[0]['frames']) != 3:
        raise ValueError('Frozen three-frame development probe required')
    ocr = load_ocr(Path('/models/experiments/awiros'),Path('/models/experiments/awiros/adaptations/indian-corpus-encoder.safetensors'))
    vehicles,pipelines = load_profiles(ocr,Path('/models'),Path('/training'),list(PROFILES),roi_budget=8)
    passage = passages[0]
    truth = passage['ground_truth']
    material = []
    for row in passage['frames']:
        path = (root/row['path']).resolve()
        if not path.is_relative_to(root.resolve()):raise ValueError('Unsafe source path')
        image = cv2.imread(str(path))
        if image is None:raise ValueError('Source decode failed')
        material.append((row,image,hashlib.sha256(path.read_bytes()).hexdigest()))
    result = {'source_mode':'benchmark','dataset':document['dataset'],
        'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'input_sha256':[sha for _,_,sha in material],
        'unique_vehicles':1,'original_frames':3,'timing_mode':'assigned_replay',
        'ground_truth_used_for_cropping':False,'held_out_accuracy_claimed':False,
        'live_validation':False,'production_promoted':False,
        'caveat':'One previously used development sequence, approximate visually reviewed annotation. Assigned PTS are not original live timing; no independent new human labelling. Full-scene automatic vehicle/plate localisation, no oracle crops, no reconstruction.',
        'methods':{},'visuals':name+'.html'}
    visuals,displays,last_images = [],{name:[] for name in pipelines},{}
    started = time.perf_counter()
    for row,image,_ in material:
        tracks = {name:model.track(image) for name,model in vehicles.items()}
        cache = {}
        for name,pipeline in pipelines.items():
            proposals = pipeline.process(image,tracks[PROFILES[name][0]],row['pts_ms'],posterior_cache=cache)
            displays[name].extend(proposals)
            last_images[name] = image
    for name,pipeline in pipelines.items():
        candidates = []
        accepted = []
        for history in pipeline.histories.values():
            if history.reader.observation_count < 3:continue
            candidates.extend(history.reader.candidates())
            match = history.reader.accepted()
            if match:accepted.append(match)
        correct = sum(c.plate == truth for c in accepted)
        wrong = sum(c.plate != truth for c in accepted)
        retained = sorted(candidates,key=lambda c:-c.log_score)
        rank = next((i+1 for i,c in enumerate(retained) if c.plate == truth),None)
        outcome = 'correct' if correct and not wrong else 'wrong' if wrong else 'abstain'
        result['methods'][name] = dict(pipeline.summary(),correct_target_accepted=bool(correct),
            wrong_accepted_identifiers=wrong,truth_rank=rank,outcome=outcome)
        crops = displays[name]
        diagnostics = []
        for index,display in enumerate(crops[:12]):
            history = pipeline.histories.get(display['track_id'])
            source = hashlib.sha256(display['crop'].tobytes()).hexdigest()
            sample = next((s for s in history.reader.samples if s[1] == source),None) if history else None
            diagnostics.append({'image':display['crop'],'index':index,'pts_ms':sample[0] if sample else None,
                'reading':greedy_text(sample[2],ocr.tokens) if sample else 'history unavailable',
                'ocr_ms':'not isolated'})
        visuals.append({'source_mode':'benchmark','title':name+' · automatic full-scene detection + tracked OCR',
            'original':crops[-1]['crop'] if crops else None,'candidates':retained,
            'accepted':accepted[0].plate if accepted else None,'expected':truth,
            'outcome':outcome,'reason':result['caveat'],'source_count':pipeline.maximum_observations,
            'source_count_label':'Distinct original observations',
            'label_description':'Visually reviewed development label, not independently human-verified',
            'dimensions':list(crops[-1]['crop'].shape[1::-1]) if crops else None,
            'duration_ms':'paired run, not standalone timing',
            'frame_diagnostics':diagnostics,
            'score_note':'Automatic boxes; no source plate/vehicle annotations select input crops. Scores are not calibrated confidence.'})
    result['paired_elapsed_seconds'] = round(time.perf_counter()-started,2)
    report = report_root/(result['visuals'].replace('.html','.json'))
    if report.exists():raise ValueError('Existing result preserved; use a new explicit experiment')
    render_viewer(report_root/result['visuals'],document['dataset'],visuals)
    report.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--name',default='detector-ocr-sequence')
    main(parser.parse_args().name)
