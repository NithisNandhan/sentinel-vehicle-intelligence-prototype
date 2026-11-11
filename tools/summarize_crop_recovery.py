"""Aggregate a completed crop experiment without publishing images or plate text."""
import argparse
import json
from pathlib import Path

from PIL import Image
from render_crop_comparison import build_data,load,sha,validate_capture


def summarize(baseline,current,review):
    baseline,current,review=map(Path,(baseline,current,review))
    manifest,labels=load(review/'manifest.json'),load(review/'labels.json')
    old,old_accuracy=validate_capture(baseline,review,manifest,labels)
    new,new_accuracy=validate_capture(current,review,manifest,labels)
    if old['models']!=new['models']:raise ValueError('Frozen detector/OCR model identity changed')
    if not new['reached_eof'] or not new['complete_crops'] or new['labels_used_during_inference']:
        raise ValueError('Completed label-free crop-recovery inference required')
    data=build_data(old,new,old_accuracy,new_accuracy,manifest)
    cases=[]
    for row in data['results']:
        item=dict(number=row['number'],reference=row['reference_id'])
        for side,root in (('previous',baseline),('current',current)):
            evidence=row[side];size=None
            if evidence['crop']:
                with Image.open(root/evidence['crop']) as image:size=list(image.size)
            item[side]=dict(found=evidence['found'],crop_size=size,group=evidence['group_id'],
                            timestamp_ms=evidence['sample_position_ms'],correspondence=evidence['correspondence'])
        cases.append(item)
    replay=new.get('crop_replay')
    old_groups={g['id']:g for g in old['groups']}
    unchanged=changed=0;unexpected=[]
    if replay:
        for group in new['groups']:
            before=old_groups[group['id']]
            same=[s['source_id'] for s in before['samples']]==[s['source_id'] for s in group['samples']]
            if not same:changed+=1;continue
            unchanged+=1
            if any(before['methods'][m]['top']!=group['methods'][m]['top'] for m in ('single','multi','reconstruction')):
                unexpected.append(group['id'])
        if unexpected:raise ValueError('Unchanged crop groups produced different raw outputs')
    return dict(date='2026-11-11',scope='Selected development labels; found-anywhere coverage, not full-video precision',
        experiment_kind='controlled_retained_crop_replay' if replay else 'full_detector_rerun',crop_replay=replay,
        unchanged_crop_groups=unchanged if replay else None,changed_crop_groups=changed if replay else None,
        unchanged_crop_output_differences=unexpected if replay else None,
        baseline_predictions_sha256=sha(baseline/'predictions.json'),predictions_sha256=sha(current/'predictions.json'),
        source_sha256=new['source_sha256'],scores=data['scores'],
        gained_references=[r['reference_id'] for r in data['results'] if r['current']['found'] and not r['previous']['found']],
        lost_references=[r['reference_id'] for r in data['results'] if r['previous']['found'] and not r['current']['found']],
        cases=cases,models=new['models'],model_weights_changed=False,source_frames=new['decoded_frames'],
        sampled_frames=new['sampled_frames'],plate_histories=len(new['groups']),baseline_plate_histories=len(old['groups']),
        ocr_calls=new['ocr_calls'],baseline_ocr_calls=old['ocr_calls'],wall_seconds=new['wall_seconds'],
        baseline_wall_seconds=old['wall_seconds'],search=new['search_diagnostics'],validation=new['plate_validation'],
        exact_timestamp_reference_summary=new_accuracy['summary'],baseline_exact_timestamp_reference_summary=old_accuracy['summary'],
        inference_source_sha256=new['inference_source_sha256'],central_events_sent=0,
        limitations=['Same development clip; not held-out accuracy',
                     'Extra wrong guesses are not penalised by the user-requested coverage metric',
                     'No new detections; fixed crop timestamps and fusion weights; unchanged original OCR posteriors reused' if replay else 'Text helper runs on CPU after a failed GPU attempt; OCR and plate detector remain on GPU',
                     'Runtime is not comparable to a fresh detection run when replaying cached evidence' if replay else 'Wall-clock proposal budgets can change which optional detector calls run'],
        failed_attempts=[dict(capture='crop-recovery-20261111',reason='CUDA unknown error during GPU text-helper run; no completed score'),
                         dict(capture='crop-recovery-cpu-text-20261111',reason='CPU predictor memory_optimize_pass failed; fixed before final run')])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','current','review','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();report=summarize(args.baseline,args.current,args.review)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('scores','gained_references','lost_references','search')}))
