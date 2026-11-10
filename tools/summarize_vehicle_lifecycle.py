"""Aggregate a lifecycle comparison; keep crops and plate transcriptions local."""
import argparse
import hashlib
import json
from pathlib import Path
from compare_temporal_approaches import box_iou


def review_overlap(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    sizes=[(v[2]-v[0])*(v[3]-v[1]) for v in (a,b)]
    if intersection/min(sizes)>=.90 and max(sizes)/min(sizes)<=6:
        return .5+.5*box_iou(a,b)
    return box_iou(a,b)


def load(root):
    payload=(root/'predictions.json').read_bytes();prediction=json.loads(payload)
    score=json.loads((root/'accuracy.json').read_bytes())
    if hashlib.sha256(payload).hexdigest()!=score['predictions_sha256']:raise ValueError('Unbound score')
    if not prediction['reached_eof'] or prediction['labels_used_during_inference']:raise ValueError('Incomplete or label-guided inference')
    return prediction,score


def parent_switches(root):
    timeline=json.loads((root/'vehicle-timeline.json').read_bytes());last={};transitions=0
    for frame in timeline['frames']:
        for vehicle in frame['tracks']:
            if not vehicle['observed']:continue
            for native in vehicle['native_ids']:
                if native in last and last[native]!=vehicle['id']:transitions+=1
                last[native]=vehicle['id']
    return transitions


def summarize(baseline,current,review_baseline=None):
    old,old_score=load(baseline);new,new_score=load(current)
    for key in ('source_sha256','tracking_report_sha256','review_sha256','models','decoded_frames'):
        if old[key]!=new[key]:raise ValueError('Incompatible '+key)
    if old_score['labels_sha256']!=new_score['labels_sha256']:raise ValueError('Different truth sets')
    old_cases={c['id']:c for c in old_score['cases']};new_cases={c['id']:c for c in new_score['cases']}
    if old_cases.keys()!=new_cases.keys():raise ValueError('Different reference targets')
    changes={m:dict(gained=[k for k,c in new_cases.items() if c['methods'][m]['exact'] and not old_cases[k]['methods'][m]['exact']],
                    lost=[k for k,c in new_cases.items() if not c['methods'][m]['exact'] and old_cases[k]['methods'][m]['exact']])
             for m in ('single','multi','reconstruction')}
    reported=[['PT0004','PT0006','PT0007'],['PT0015','PT0016'],['PT0036','PT0041'],['PT0047','PT0049','PT0050'],['PT0068','PT0069']]
    reviewed,review_score=load(review_baseline or baseline)
    if (reviewed['source_sha256']!=new['source_sha256'] or review_score['predictions_sha256']!=
            '16e5f78f635a3cb879b6d6e9e2c7f946be2e00e83810ac4e2729434ae15de727'):
        raise ValueError('Reported example IDs belong to the original plate-flow-checked run; supply --review-baseline')
    old_groups={g['id']:g for g in reviewed['groups']};checks=[]
    for cluster in reported:
        mapped={}
        for key in cluster:
            matches=[]
            for observed in old_groups[key]['observed']:
                choices=[(review_overlap(observed['box'],sample['box']),g) for g in new['groups'] for sample in g['observed']
                         if sample['position_ms']==observed['position_ms']]
                choices.sort(key=lambda c:-c[0])
                if choices and choices[0][0]>=.3 and (len(choices)==1 or choices[0][0]-choices[1][0]>=.05):
                    group=choices[0][1];matches.append(dict(plate_history=group['id'],vehicle=group.get('parent_vehicle_id')))
            mapped[key]=[dict(plate_history=pid,vehicle=vehicle) for pid,vehicle in sorted({(m['plate_history'],m['vehicle']) for m in matches},key=lambda p:p[0])]
        parents={m['vehicle'] for values in mapped.values() for m in values if m['vehicle']}
        all_assigned=all(values and all(m['vehicle'] for m in values) for values in mapped.values())
        checks.append(dict(previous_cluster=cluster,mapping=mapped,single_confirmed_parent=len(parents)==1 and all_assigned))
    negative=[]
    for old_id in ('PT0006','PT0023'):
        parents={m['vehicle'] for case in checks for key,values in case['mapping'].items() if key==old_id for m in values if m['vehicle']}
        if old_id=='PT0023':
            old_group=old_groups[old_id]
            for group in new['groups']:
                if any(box_iou(a['box'],b['box'])>=.3 and a['position_ms']==b['position_ms'] for a in old_group['observed'] for b in group['observed']):
                    if group.get('parent_vehicle_id'):parents.add(group['parent_vehicle_id'])
        negative.append(parents)
    byid={g['id']:g for g in new['groups']}
    accepted=[(c,byid[c['group']]['methods']['multi']['accepted']) for c in new_score['cases'] if c['group']]
    return dict(date='2026-11-10',scope='Same selected development references; no full-video accuracy claim',
        models=new['models'],source_sha256=new['source_sha256'],baseline_summary=old_score['summary'],summary=new_score['summary'],
        changes_vs_baseline=changes,vehicle_summary=new_score.get('vehicle_summary'),
        decoded_frames=new['decoded_frames'],sampled_frames=new['sampled_frames'],wall_seconds=new['wall_seconds'],
        baseline_wall_seconds=old['wall_seconds'],plate_histories=len(new['groups']),vehicle_histories=len(new['vehicles']),
        lifecycle=new['vehicle_lifecycle_counters'],motion=new['vehicle_motion_diagnostics'],router=new['router'],context=new['context_diagnostics'],
        vehicle_histories_with_confirmed_ocr_evidence=len(new['vehicle_results']),model_accepted_matched_correct=sum(bool(v) and v==c['expected'] for c,v in accepted),
        model_accepted_matched_wrong=sum(bool(v) and v!=c['expected'] for c,v in accepted),
        reported_case_checks=checks,car_vs_autorickshaw_parent_conflict=bool(negative[0]&negative[1]),
        reported_case_baseline_predictions_sha256=review_score['predictions_sha256'],
        car_vs_autorickshaw_both_assigned=bool(negative[0] and negative[1]),
        predictions_sha256=new_score['predictions_sha256'],inference_source_sha256=new['inference_source_sha256'],
        native_id_parent_switches=parent_switches(current),central_events_sent=0,labels_used_during_inference=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--current',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--review-baseline',type=Path)
    args=parser.parse_args()
    document=summarize(args.baseline,args.current,args.review_baseline)
    args.output.write_text(json.dumps(document,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(summary=document['summary'],vehicle_summary=document['vehicle_summary'],
        reported_clusters_resolved=sum(c['single_confirmed_parent'] for c in document['reported_case_checks']))))
