"""Publish aggregate plate-evidence regression results without images or text."""
import argparse
import json
from pathlib import Path

from run_owned_multiframe import sha,write_json


def summarize(baseline,current):
    old=json.loads((baseline/'predictions.json').read_bytes())
    new=json.loads((current/'predictions.json').read_bytes())
    old_score=json.loads((baseline/'accuracy.json').read_bytes())
    score=json.loads((current/'accuracy.json').read_bytes())
    if old_score['predictions_sha256']!=sha(baseline/'predictions.json') or score['predictions_sha256']!=sha(current/'predictions.json'):
        raise ValueError('Scores do not bind captures')
    if new['plate_validation_capture']['input_predictions_sha256']!=old_score['predictions_sha256']:
        raise ValueError('Validation did not derive from the stated baseline')
    before={g['id']:g for g in old['groups']};after={g['id']:g for g in new['groups']}
    if set(before)!=set(after):raise ValueError('Validation changed plate identities')
    for key in before:
        if before[key]['samples']!=after[key]['samples'] or before[key]['observed']!=after[key]['observed']:
            raise ValueError('Validation changed original evidence')
        for method in ('single','multi','reconstruction'):
            if any(before[key]['methods'][method][field]!=after[key]['methods'][method][field] for field in ('top','candidates')):
                raise ValueError('Validation rewrote raw model readings')
    old_parents={v['id']:v for v in old['vehicle_results']}
    changed=[v['id'] for v in new['vehicle_results'] if old_parents.get(v['id'],{}).get('methods')!=v['methods']]
    cases={c['id']:c for c in old_score['cases']}
    lost=[c['id'] for c in score['cases'] if cases[c['id']]['methods']['multi']['exact'] and not c['methods']['multi']['exact']]
    return dict(scope='Frozen saved-evidence development diagnostic; not full-video precision/recall',
        baseline_predictions_sha256=old_score['predictions_sha256'],predictions_sha256=score['predictions_sha256'],
        source_sha256=new['source_sha256'],validation=new['plate_validation'],provenance=new['plate_validation_capture'],
        original_histories=len(before),retained_native_crops=sum(len(g['samples']) for g in after.values()),
        suspected_non_plate_ids=[g['id'] for g in new['groups'] if g['plate_validation']['excluded_from_fusion']],
        active_plate_histories=sum(not g['plate_validation']['excluded_from_fusion'] for g in new['groups']),
        raw_hypotheses_and_native_evidence_unchanged=True,parent_readings_changed=changed,
        baseline_summary=old_score['summary'],summary=score['summary'],baseline_vehicle_summary=old_score['vehicle_summary'],
        vehicle_summary=score['vehicle_summary'],lost_exact_development_targets=lost,
        postprocess_seconds=new['postprocess_seconds'],models_rerun=False,central_events_sent=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--current',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report=summarize(args.baseline,args.current);write_json(args.output,report)
    print(json.dumps({k:report[k] for k in ('suspected_non_plate_ids','active_plate_histories','parent_readings_changed','lost_exact_development_targets')}))
