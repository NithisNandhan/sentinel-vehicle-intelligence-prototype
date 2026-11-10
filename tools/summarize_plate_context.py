"""Publish aggregate crop/tracking comparisons without plate texts or images."""
import argparse
import hashlib
import json
from pathlib import Path


def read(root):
    prediction=json.loads((root/'predictions.json').read_bytes())
    score=json.loads((root/'accuracy.json').read_bytes())
    if hashlib.sha256((root/'predictions.json').read_bytes()).hexdigest()!=score['predictions_sha256']:
        raise ValueError('Score/prediction mismatch')
    if not prediction['reached_eof'] or prediction['labels_used_during_inference']:
        raise ValueError('Incomplete or label-guided capture')
    return prediction,score


def summarize(runs):
    documents={name:read(root) for name,root in runs.items()}
    first_prediction,first_score=next(iter(documents.values()))
    methods=('single','multi','reconstruction');baseline=first_score
    output={}
    for name,(prediction,score) in documents.items():
        if (prediction['source_sha256']!=first_prediction['source_sha256']
                or prediction['tracking_report_sha256']!=first_prediction['tracking_report_sha256']
                or prediction['review_sha256']!=first_prediction['review_sha256']
                or prediction['models']!=first_prediction['models']
                or prediction['decoded_frames']!=first_prediction['decoded_frames']
                or score['labels_sha256']!=first_score['labels_sha256']
                or {c['id'] for c in score['cases']}!={c['id'] for c in first_score['cases']}):
            raise ValueError('Trials do not share source, models, tracking and labels')
        byid={c['id']:c for c in score['cases']};base={c['id']:c for c in baseline['cases']}
        changes={method:dict(
            gained=[key for key,c in byid.items() if c['methods'][method]['exact'] and not base[key]['methods'][method]['exact']],
            lost=[key for key,c in byid.items() if not c['methods'][method]['exact'] and base[key]['methods'][method]['exact']]) for method in methods}
        groups={g['id']:g for g in prediction['groups']}
        matched=[c for c in score['cases'] if c['group']]
        accepted=[(c,groups[c['group']]['methods']['multi']['accepted']) for c in matched]
        output[name]=dict(summary=score['summary'],changes_vs_baseline=changes,
            decoded_frames=prediction['decoded_frames'],sampled_frames=prediction['sampled_frames'],
            evidence_histories_not_unique_vehicles=len(groups),wall_seconds=prediction['wall_seconds'],
            ocr_calls=prediction['ocr_calls'],router=prediction['router'],
            context=prediction.get('context_diagnostics'),motion=prediction.get('motion_diagnostics'),
            matched_reference_model_accepted=sum(bool(value) for _,value in accepted),
            matched_reference_model_accepted_correct=sum(bool(value) and value==case['expected'] for case,value in accepted),
            matched_reference_model_accepted_wrong=sum(bool(value) and value!=case['expected'] for case,value in accepted),
            all_group_model_accepted=sum(bool(g['methods']['multi']['accepted']) for g in groups.values()),
            all_group_vehicle_owned_accepted=sum(bool(g['methods']['multi'].get('accepted_for_vehicle',
                g['methods']['multi']['accepted'] if g['owner'] else None)) for g in groups.values()),
            conflicting_groups_quarantined=len(score['conflicting_groups']),
            predictions_sha256=score['predictions_sha256'],
            inference_source_sha256=prediction.get('inference_source_sha256'))
    return dict(date='2026-11-10',scope='Same eighteen selected development references; not full-video or unseen-camera accuracy',
        source_sha256=first_prediction['source_sha256'],models=first_prediction['models'],
        human_labels_used_only_for_scoring=True,central_events_sent=0,runs=output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','context','motion','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    document=summarize(dict(baseline=args.baseline,context=args.context,motion=args.motion))
    args.output.write_text(json.dumps(document,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v['summary']['multi'] for k,v in document['runs'].items()}))
