"""Selected source-frame identity checks, not exhaustive MOT accuracy.

Annotations are loaded only after all tracker strategies complete. An ambiguous
geometry match remains unmatched, rather than choosing the convenient identity.
"""
import argparse
import hashlib
import json
from pathlib import Path
from compare_temporal_approaches import box_iou


def review(root,annotations):
    spec=json.loads(annotations.read_bytes());capture=json.loads((root/'report.json').read_bytes())
    if spec['source_sha256']!=capture['source_sha256']:raise ValueError('Review/source mismatch')
    output={}
    for strategy in capture['strategies']:
        timeline=json.loads((root/strategy/'timeline.json').read_bytes());frames={f['t']:f for f in timeline['frames']}
        cases=[]
        for anchor in spec['anchors']:
            frame=frames[anchor['position_ms']]
            choices=sorted([(box_iou(anchor['box'],t[1:5]),t[0]) for t in frame['q'] if t[7]],reverse=True)
            selected=choices[0][1] if choices and choices[0][0]>=.5 and (len(choices)==1 or choices[0][0]-choices[1][0]>=.05) else None
            cases.append(dict(vehicle=anchor['vehicle'],position_ms=anchor['position_ms'],matched_track=selected,
                              best_iou=round(choices[0][0],4) if choices else 0.))
        per_vehicle={key:[c for c in cases if c['vehicle']==key] for key in {c['vehicle'] for c in cases}}
        groups={key:{c['matched_track'] for c in values if c['matched_track']} for key,values in per_vehicle.items()}
        identities={}
        for vehicle,ids in groups.items():
            for identity in ids:identities.setdefault(identity,set()).add(vehicle)
        output[strategy]=dict(cases=cases,anchors=len(cases),matched_anchors=sum(c['matched_track'] is not None for c in cases),
            selected_vehicle_cases=len(groups),fragmented_cases=sum(len(ids)>1 for ids in groups.values()),
            repeated_vehicle_cases=sum(len(values)>1 for values in per_vehicle.values()),
            fully_consistent_cases=sum(len(values)>1 and len(groups[k])==1 and all(c['matched_track'] for c in values) for k,values in per_vehicle.items()),
            cross_vehicle_ids=sum(len(v)>1 for v in identities.values()))
    return dict(scope='Selected assistant-reviewed source-frame bounding boxes; not human-verified MOT ground truth or full-video accuracy',
        annotations_sha256=hashlib.sha256(annotations.read_bytes()).hexdigest(),source_sha256=spec['source_sha256'],
        annotations_used_during_inference=False,match_iou=.5,unique_match_margin=.05,strategies=output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--annotations',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();document=review(args.run,args.annotations)
    args.output.write_text(json.dumps(document,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:{n:v[n] for n in ['matched_anchors','anchors','fully_consistent_cases','fragmented_cases','cross_vehicle_ids']} for k,v in document['strategies'].items()}))
