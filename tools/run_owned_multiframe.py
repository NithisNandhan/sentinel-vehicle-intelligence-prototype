"""Owned multi-frame OCR on frozen dense vehicle tracks and permitted video.

No training, downloads, threshold search or central events. Originals remain
distinct evidence; reconstruction only proposes strings rescored on originals.
"""
import argparse
import ast
from collections import Counter
import gzip
import hashlib
import html
import json
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'edge'))
from sentinel_edge.owned_plate import OwnedPlateRouter
from sentinel_edge.ctc_fusion import CTCPlateReader
from sentinel_edge.plate_quality import crop_quality,select_frames
from sentinel_edge.reconstruction import CropBank
from compare_temporal_approaches import box_iou,reference_box,sha,write_json

CHECKPOINT='c2ba896d994b174071c95ace0f5ffc8a61cec3fc6eecce8f0bc5feb7dec4456d'
METHODS=('single','multi','reconstruction')


def pack(candidate):
    return dict(plate=candidate.plate,score=candidate.score,minimum_character_score=candidate.minimum_character_score)


def run(args):
    if args.output.exists():raise ValueError('Use a fresh run folder')
    args.output.mkdir(parents=True);(args.output/'assets').mkdir()
    (args.output/'runner.py').write_bytes(Path(__file__).read_bytes())
    write_json(args.output/'progress.json',dict(stage='loading_models'))
    tracking_path=args.tracking/'report.json';tracking=json.loads(tracking_path.read_bytes())
    timeline=json.loads((args.tracking/'timeline.json').read_bytes())['frames']
    review_path=args.tracking/'vehicle-review.json'
    review=json.loads(review_path.read_bytes()) if review_path.exists() else None
    if review and review['report_sha256']!=sha(tracking_path):raise ValueError('Review belongs to another tracker run')
    excluded={k for k,v in (review or {}).get('labels',{}).items() if v['status'] in ('non_vehicle','mixed','uncertain')}
    if review:(args.output/'vehicle-review.snapshot.json').write_bytes(review_path.read_bytes())
    from sentinel_edge.recorded_source import RecordingLibrary,read_recording
    from sentinel_edge.config import Config
    from sentinel_edge.live_detection_viewer import load_models,model_identity,clipped_box,crop_box
    from sentinel_edge.tracked_live_viewer import load_text_detector
    from sentinel_edge.plate_search import TargetedPlateSearch
    from sentinel_edge.plate_proposals import ProposalPool
    from sentinel_edge.text_boundaries import warp_polygon
    from sentinel_edge.live_reconstruction import ReconstructionState,update,disagreement
    import cv2
    recordings,_=RecordingLibrary(args.recordings_dir).scan();source=recordings.get(tracking['recording_id'])
    if source is None or source.sha256!=tracking['source_sha256']:raise ValueError('Permitted source identity mismatch')
    checkpoint=args.cache/'adaptations/indian-corpus-encoder.safetensors'
    if sha(checkpoint)!=CHECKPOINT:raise ValueError('Deployed checkpoint changed')
    config=Config.from_env(live=False)
    ocr,detector,unused_vehicle=load_models(config,args.cache,checkpoint,detector_profile='general',device='gpu')
    models=model_identity(ocr,'general',detector=detector,vehicle_model=unused_vehicle)
    search=ProposalPool(TargetedPlateSearch(detector),tile_detector=detector,text_detector=load_text_detector())
    del unused_vehicle
    router=OwnedPlateRouter(excluded);bank=CropBank(max_bytes=128*1024*1024,max_frames=6,max_tracks=512,retain_best=True)
    groups={};decoded=sampled=roi_calls=0;last_slot=None;began=time.monotonic()
    for _,frame in read_recording(source,threading.Event(),paced=False):
        index=decoded;decoded+=1;pts=frame.pts_ms-source.start_pts_ms
        if index>=len(timeline) or abs(timeline[index]['t']-pts)>1:raise ValueError('Video/tracker timestamp mismatch')
        slot=pts//250
        if slot==last_slot:continue
        last_slot=slot;sampled+=1
        vehicles=[dict(id=t[0],box=clipped_box(frame.image,t[1:5]),qualified=bool(t[7])) for t in timeline[index]['q']]
        proposals=[];search.begin_frame()
        eligible=[v for v in vehicles if v['qualified'] and v['id'] not in excluded]
        if eligible:
            turn=sampled%len(eligible);eligible=eligible[turn:]+eligible[:turn]
        for vehicle in eligible:
            vx,vy,_,_=vehicle['box'];region=crop_box(frame.image,vehicle['box'])
            if region is None:continue
            local=search.box(region);roi_calls+=1
            if local is None:continue
            local=clipped_box(region,local)
            if local[2]-local[0]<24 or local[3]-local[1]<8:continue
            quad=getattr(search,'last_text_quad',None)
            relative=[(x-local[0],y-local[1]) for x,y in quad] if quad is not None else None
            proposals.append(dict(box=[local[0]+vx,local[1]+vy,local[2]+vx,local[3]+vy],
                                  confidence=float(search.last_confidence or 0.),quad=relative))
        for proposal in router.route(proposals,vehicles,pts):
            gid=proposal['group'];b=proposal['box'];native=crop_box(frame.image,b)
            if native is None:continue
            quality=crop_quality(native);digest=hashlib.sha256(native.tobytes()).hexdigest()
            group=groups.setdefault(gid,dict(id=gid,owner=proposal['owner'],observed=[],metadata={}))
            group['observed'].append(dict(position_ms=pts,box=b,source_id=digest))
            group['metadata'][digest]=dict(box=b,quad=proposal['quad'])
            bank.add(gid,native,pts,quality=quality)
        if sampled%20==0:
            progress=dict(stage='collecting_owned_crops',position_ms=pts,decoded=decoded,sampled=sampled,
                          groups=len(groups),router=dict(router.counters))
            write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    if decoded!=tracking['decoded_frames']:raise ValueError('Source did not complete the frozen frame inventory')
    posteriors=[];predictions=[];ocr_calls=0
    for group in groups.values():
        samples=select_frames(bank.get(group['id']),limit=6)
        if not samples:continue
        observations=[];multi=CTCPlateReader(ocr.tokens)
        for i,sample in enumerate(samples):
            metadata=group['metadata'][sample.source_id]
            working=warp_polygon(sample.image,metadata['quad']) if metadata['quad'] else None
            if working is None:working=sample.image
            rows=ocr.probabilities(working);ocr_calls+=1
            multi.add(rows,timestamp_ms=sample.pts_ms,source_id=sample.source_id,quality=sample.quality)
            relative=f"assets/{group['id']}-{i}.png"
            if not cv2.imwrite(str(args.output/relative),sample.image):raise ValueError('Native PNG save failed')
            observations.append(dict(position_ms=sample.pts_ms,source_id=sample.source_id,quality=sample.quality,
                box=metadata['box'],path=relative,rows=rows,working_quad=metadata['quad']))
        best=max(observations,key=lambda o:(o['quality'],-o['position_ms']))
        single=CTCPlateReader(ocr.tokens);single.add(best['rows'],timestamp_ms=best['position_ms'],source_id=best['source_id'])
        def result(reader):
            ranked=reader.candidates();accepted=reader.accepted(.85)
            return dict(top=ranked[0].plate if ranked else None,candidates=[pack(c) for c in ranked[:20]],
                        accepted=accepted.plate if accepted else None)
        values=dict(single=result(single),multi=result(multi))
        state=ReconstructionState();update(state,multi,samples,ocr,samples[-1].pts_ms)
        values['reconstruction']=result(multi)
        accepted=multi.accepted(.85)
        if disagreement(state,accepted,.85):values['reconstruction']['accepted']=None
        public_samples=[{k:v for k,v in o.items() if k!='rows'} for o in observations]
        predictions.append(dict(id=group['id'],owner=group['owner'],observed=group['observed'],
            samples=public_samples,observations=len(samples),methods=values,reconstruction_status=state.status))
        posteriors.append(dict(id=group['id'],observations=observations))
        if len(predictions)%10==0:
            write_json(args.output/'progress.json',dict(stage='recognizing',groups_done=len(predictions),groups=len(groups),ocr_calls=ocr_calls))
            print(json.dumps(dict(stage='recognizing',groups_done=len(predictions),ocr_calls=ocr_calls)),flush=True)
    with gzip.open(args.output/'posteriors.json.gz','wt',encoding='utf-8') as f:json.dump(dict(tokens=ocr.tokens,groups=posteriors),f,allow_nan=False)
    document=dict(schema_version=1,source_sha256=source.sha256,tracking_report_sha256=sha(tracking_path),
        timeline_sha256=sha(args.tracking/'timeline.json'),review_sha256=sha(review_path) if review else None,
        excluded_review_ids=sorted(excluded),models=models,decoded_frames=decoded,sampled_frames=sampled,
        sample_interval_ms=250,maximum_retained_originals=6,roi_calls=roi_calls,ocr_calls=ocr_calls,
        router=dict(router.counters),search_diagnostics=search.diagnostics(),groups_with_owned_crops=len(groups),
        groups_with_ocr=len(predictions),groups_evicted_before_ocr=len(groups)-len(predictions),
        wall_seconds=round(time.monotonic()-began,2),reached_eof=True,labels_used_during_inference=False,
        runner_sha256=sha(args.output/'runner.py'),ownership_source_sha256=sha(Path(__file__).resolve().parents[1]/'edge/sentinel_edge/owned_plate.py'),
        groups=predictions,scope='Full permitted source; frozen dense tracker; no events or training; selected human reference evaluation follows separately')
    write_json(args.output/'predictions.json',document)
    write_json(args.output/'progress.json',dict(stage='inference_complete',groups=len(predictions),ocr_calls=ocr_calls))
    print(json.dumps(dict(stage='inference_complete',groups=len(predictions),ocr_calls=ocr_calls,wall_seconds=document['wall_seconds'])),flush=True)


def score(args):
    path=args.output/'predictions.json';d=json.loads(path.read_bytes())
    if (args.output/'accuracy.json').exists():raise ValueError('Preserve the earlier score')
    ref=json.loads((args.review/'manifest.json').read_bytes());labels=json.loads((args.review/'labels.json').read_bytes())
    if (ref['source_sha256']!=d['source_sha256'] or labels['source_sha256']!=d['source_sha256']
            or labels['review_manifest_sha256']!=sha(args.review/'manifest.json')):raise ValueError('Wrong truth binding')
    corners={}
    for node in ast.parse((args.review/'ocr-model-dev/rectify.py').read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('CORNERS','TEXT_REGIONS') for t in node.targets):corners.update(ast.literal_eval(node.value))
    truth={r['id']:r for r in labels['labels']};cases=[]
    for reference in ref['samples']:
        label=truth[reference['id']]
        if label['status']!='readable' or reference['id']=='21':continue
        expected_box=reference_box(reference,corners)
        choices=[]
        for group in d['groups']:
            overlap=max((box_iou(expected_box,o['box']) for o in group['observed'] if abs(o['position_ms']-reference['position_ms'])<=20),default=0)
            if overlap>=.3:choices.append((overlap,group))
        choices.sort(key=lambda pair:-pair[0])
        group=choices[0][1] if choices and (len(choices)==1 or choices[0][0]-choices[1][0]>=.05) else None
        cases.append(dict(id=reference['id'],expected=label['plate'],position_ms=reference['position_ms'],
            group=group['id'] if group else None,methods={m:dict(top=group['methods'][m]['top'] if group else None,
                exact=bool(group and group['methods'][m]['top']==label['plate'])) for m in METHODS}))
    # Never choose a favourable truth when one inferred history hits conflicting labels.
    conflicts={}
    for c in cases:
        if c['group']:conflicts.setdefault(c['group'],set()).add(c['expected'])
    bad={g for g,v in conflicts.items() if len(v)>1}
    for c in cases:
        if c['group'] in bad:c['group']=None;c['methods']={m:dict(top=None,exact=False) for m in METHODS}
    matched=[c for c in cases if c['group']];byid={g['id']:g for g in d['groups']}
    eligible=[c for c in matched if byid[c['group']]['observations']>=3]
    summary={m:dict(exact=sum(c['methods'][m]['exact'] for c in cases),targets=len(cases),
        matched_exact=sum(c['methods'][m]['exact'] for c in matched),matched_targets=len(matched),
        multi_frame_exact=sum(c['methods'][m]['exact'] for c in eligible),multi_frame_targets=len(eligible),
        recovered_vs_single=[c['id'] for c in cases if not c['methods']['single']['exact'] and c['methods'][m]['exact']],
        regressed_vs_single=[c['id'] for c in cases if c['methods']['single']['exact'] and not c['methods'][m]['exact']]) for m in METHODS}
    report=dict(scope='Previously reviewed reference targets; not full-video accuracy',predictions_sha256=sha(path),
        labels_sha256=sha(args.review/'labels.json'),summary=summary,conflicting_groups=list(bad),cases=cases,
        processing={k:v for k,v in d.items() if k!='groups'})
    write_json(args.output/'accuracy.json',report);render(args.output,d,report)
    print(json.dumps(dict(stage='scored',summary=summary)),flush=True)


def render(root,d,report):
    cases={c['group']:c for c in report['cases'] if c['group']}
    cards=[]
    for g in d['groups']:
        c=cases.get(g['id']);pictures=''.join('<figure><img src="'+html.escape(o['path'])+'" alt="Native plate crop"><figcaption>'+str(o['position_ms']/1000)+' s</figcaption></figure>' for o in g['samples'])
        readings=''.join('<tr><td>'+m+'</td><td>'+html.escape(g['methods'][m]['top'] or 'No complete hypothesis')+'</td><td>'+html.escape(g['methods'][m]['accepted'] or 'Unconfirmed')+'</td><td>'+(', '.join(html.escape(x['plate']) for x in g['methods'][m]['candidates'][:5]) or 'None')+'</td></tr>' for m in METHODS)
        truth='<p>Existing human reference '+c['id']+': '+html.escape(c['expected'])+'</p>' if c else '<p>Unlabelled passage: correctness unmeasured.</p>'
        cards.append('<section><h2>'+html.escape(g['id'])+' · owner '+html.escape(g['owner'])+'</h2>'+truth+'<div class="frames">'+pictures+'</div><div class="scroll"><table><tr><th>Method</th><th>Top hypothesis</th><th>Model gate</th><th>Top five candidates</th></tr>'+readings+'</table></div></section>')
    summary=''.join('<tr><td>'+m+'</td><td>'+str(s['matched_exact'])+'/'+str(s['matched_targets'])+'</td><td>'+str(s['multi_frame_exact'])+'/'+str(s['multi_frame_targets'])+'</td></tr>' for m,s in report['summary'].items())
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sentinel · Owned multi-frame OCR</title><style>body{font:16px system-ui;background:#f6f7f2;color:#18352e;max-width:1350px;margin:30px auto;padding:0 18px}h1,h2{color:#0f766e}section{padding:18px;margin:20px 0;border:1px solid #cbd9ce;border-radius:10px;background:white}.frames{display:flex;gap:10px;overflow:auto}figure{margin:5px;flex:none}img{max-width:280px;max-height:130px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #cbd9ce;text-align:left}.scroll{overflow:auto}</style><h1>Owned multi-frame recognition</h1><p>Same OCR weights, same native evidence. Ambiguous plate ownership is rejected before fusion. All hypotheses and model scores remain unverified until labelled.</p><p>'+str(d['decoded_frames'])+' source frames · '+str(d['sampled_frames'])+' plate-search samples · '+str(d['groups_with_ocr'])+' evidence groups. <a href="http://127.0.0.1:7798/">Vehicle tracking viewer</a></p><table><tr><th>Method</th><th>Matched exact</th><th>≥3-frame exact</th></tr>'+summary+'</table><p>These are selected development references, not full-video accuracy. Unmatched references remain misses in the all-18 denominator. No watchlist events were emitted.</p>'+''.join(cards)+'</html>'
    (root/'index.html').write_bytes(page.encode('utf-8'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('run','score'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--tracking',type=Path,default=Path('/workspace/data/benchmarks/vehicle-tracking-960-20261110'))
    p.add_argument('--recordings-dir',type=Path,default=Path('/recordings'));p.add_argument('--cache',type=Path,default=Path('/models/experiments/awiros'))
    p.add_argument('--review',type=Path,default=Path('data/benchmarks/recorded-footage-20261104/manual-review'))
    args=p.parse_args();run(args) if args.phase=='run' else score(args)
