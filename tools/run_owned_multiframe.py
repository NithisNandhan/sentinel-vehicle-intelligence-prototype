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
from sentinel_edge.plate_identity import PlateIdentityTracker
from sentinel_edge.plate_context import context_proposals
from sentinel_edge.plate_motion import PlateMotion
from sentinel_edge.vehicle_lifecycle import VehicleLifecycle
from sentinel_edge.vehicle_plate import VehiclePlateRouter,deduplicate_views,one_view_per_timestamp,vehicle_signature,SceneBoundary
from sentinel_edge.plate_sampling import PlateBurstSchedule
from sentinel_edge.ctc_fusion import CTCPlateReader,greedy_reading
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
    source_hashes={}
    for name in ('owned_plate','plate_context','plate_identity','plate_motion','vehicle_lifecycle','vehicle_plate','plate_sampling'):
        payload=(Path(__file__).resolve().parents[1]/'edge/sentinel_edge'/(name+'.py')).read_bytes()
        (args.output/('source-'+name+'.py')).write_bytes(payload)
        source_hashes[name]=hashlib.sha256(payload).hexdigest()
    write_json(args.output/'progress.json',dict(stage='loading_models'))
    tracking_path=args.tracking/'report.json';tracking=json.loads(tracking_path.read_bytes())
    timeline=json.loads((args.tracking/'timeline.json').read_bytes())['frames']
    review_path=args.tracking/'vehicle-review.json'
    review_bytes=review_path.read_bytes() if review_path.exists() else None
    review=json.loads(review_bytes) if review_bytes else None
    if review and review['report_sha256']!=sha(tracking_path):raise ValueError('Review belongs to another tracker run')
    excluded={k for k,v in (review or {}).get('labels',{}).items() if v['status'] in ('non_vehicle','mixed','uncertain')}
    if review:(args.output/'vehicle-review.snapshot.json').write_bytes(review_bytes)
    from sentinel_edge.recorded_source import RecordingLibrary,read_recording
    from sentinel_edge.config import Config
    from sentinel_edge.live_detection_viewer import load_models,model_identity,clipped_box,crop_box
    from sentinel_edge.tracked_live_viewer import load_text_detector
    from sentinel_edge.plate_search import TargetedPlateSearch
    from sentinel_edge.plate_proposals import ProposalPool
    from sentinel_edge.text_boundaries import warp_polygon
    from sentinel_edge.live_reconstruction import ReconstructionState,update,disagreement
    import cv2
    import numpy as np
    recordings,_=RecordingLibrary(args.recordings_dir).scan();source=recordings.get(tracking['recording_id'])
    if source is None or source.sha256!=tracking['source_sha256']:raise ValueError('Permitted source identity mismatch')
    checkpoint=args.cache/'adaptations/indian-corpus-encoder.safetensors'
    if sha(checkpoint)!=CHECKPOINT:raise ValueError('Deployed checkpoint changed')
    config=Config.from_env(live=False)
    ocr,detector,unused_vehicle=load_models(config,args.cache,checkpoint,detector_profile='general',device='gpu')
    models=model_identity(ocr,'general',detector=detector,vehicle_model=unused_vehicle)
    search=ProposalPool(TargetedPlateSearch(detector),tile_detector=detector,text_detector=load_text_detector())
    del unused_vehicle
    router=VehiclePlateRouter(visual_identity=args.visual_identity) if args.vehicle_lifecycle else PlateIdentityTracker(excluded) if args.plate_first else OwnedPlateRouter(excluded)
    bank=CropBank(max_bytes=128*1024*1024,max_frames=6,max_tracks=512,retain_best=True)
    posterior_cache={};context_metrics=Counter();identity_ocr_calls=0
    motion=PlateMotion() if args.plate_flow else None
    lifecycle=VehicleLifecycle((tracking['height'],tracking['width']),excluded=excluded) if args.vehicle_lifecycle else None
    vehicle_motion=PlateMotion(max_active=128) if lifecycle else None
    scene=SceneBoundary() if lifecycle else None
    bursts=PlateBurstSchedule() if args.adaptive_plates else None
    burst_frames=0
    vehicle_frames=[];signature_cache={};parents=[]
    groups={};decoded=sampled=roi_calls=0;last_slot=None;began=time.monotonic()
    for _,frame in read_recording(source,threading.Event(),paced=False):
        index=decoded;decoded+=1;pts=frame.pts_ms-source.start_pts_ms
        if index>=len(timeline) or abs(timeline[index]['t']-pts)>1:raise ValueError('Video/tracker timestamp mismatch')
        reset=scene.changed(frame.image) if scene else False
        if reset:
            router.groups.clear();motion.states.clear();vehicle_motion.states.clear()
            signature_cache.clear()
        if motion:motion.advance(frame.image,pts)
        if lifecycle:
            vehicle_motion.advance(frame.image,pts)
            raw=[]
            for t in timeline[index]['q']:
                box=clipped_box(frame.image,t[1:5])
                if box[2]<=box[0] or box[3]<=box[1]:continue
                if t[0] not in signature_cache or pts-signature_cache[t[0]][0]>=250:
                    signature_cache[t[0]]=(pts,vehicle_signature(frame.image,box))
                raw.append(dict(id=t[0],box=box,qualified=bool(t[7]),vehicle_class=int(t[5]),
                    signature=signature_cache[t[0]][1],motion_support=vehicle_motion.support(box)))
            signature_cache={k:v for k,v in signature_cache.items() if pts-v[0]<=5000}
            parents=lifecycle.update(raw,pts,scene_reset=reset)
            for parent in parents:
                if parent['observed']:vehicle_motion.observe(parent['id'],parent['box'])
            vehicle_frames.append(dict(t=pts,tracks=[{k:v for k,v in parent.items() if k in
                ('id','box','status','observed','vehicle_class','native_ids')} for parent in parents]))
            router.retained_parent_ids={p['id'] for p in parents}
            if bursts:bursts.advance(parents,pts,reset=reset)
        slot=pts//250
        base_due=slot!=last_slot
        burst_due=bursts.due() if bursts else set()
        if not base_due and not burst_due:continue
        if not base_due:burst_frames+=1
        last_slot=slot;sampled+=1
        vehicles=[dict(id=t[0],box=clipped_box(frame.image,t[1:5]),qualified=bool(t[7])) for t in timeline[index]['q']]
        if lifecycle:
            vehicles=[{**p,'box':clipped_box(frame.image,p['box']),'qualified':p['qualified'] and p['observed']}
                      for p in parents if p['box'] is not None]
            vehicles=[p for p in vehicles if p['box'][2]>p['box'][0] and p['box'][3]>p['box'][1]]
            aliases=lifecycle.native_aliases
            vehicles.extend(dict(id='veto:'+t['id'],box=t['box'],qualified=False,observed=True,
                                 status='tentative',ownership_veto=t['id'] in excluded)
                            for t in raw if (not t['qualified'] or t['id'] in excluded) and t['id'] not in aliases)
        proposals=[];search.begin_frame()
        eligible=[v for v in vehicles if v['qualified'] and v['id'] not in excluded]
        if lifecycle:eligible=[v for v in vehicles if v.get('status') in ('visible','predicted')]
        if not base_due:eligible=[v for v in eligible if v['id'] in burst_due and v.get('observed')]
        if bursts:bursts.searched({v['id'] for v in eligible})
        if eligible:
            turn=sampled%len(eligible);eligible=eligible[turn:]+eligible[:turn]
        if args.plate_first:
            proposals,metrics=context_proposals(frame.image,eligible,search)
            context_metrics.update(metrics);roi_calls+=metrics['padded_search']+metrics['fragment_redetection']
            if lifecycle:
                proposals,metrics=deduplicate_views(proposals);context_metrics.update(metrics)
            # Deduplicate overlapping vehicle ROIs before running identity OCR.
            unique=[]
            for proposal in sorted(proposals,key=lambda p:-p['confidence']):
                if any(box_iou(proposal['box'],p['box'])>=.5 for p in unique):
                    context_metrics['same_frame_duplicates']+=1;continue
                native=crop_box(frame.image,proposal['box'])
                digest=hashlib.sha256(native.tobytes()).hexdigest()
                working=warp_polygon(native,proposal['quad']) if proposal['quad'] else None
                if working is None:working=native
                # Fingerprint includes working geometry: identical native pixels
                # with different text rectification are different OCR inputs.
                key=hashlib.sha256(working.tobytes()+str(working.shape).encode()).hexdigest()
                if key not in posterior_cache:
                    posterior_cache[key]=ocr.probabilities(working);identity_ocr_calls+=1
                gray=cv2.resize(cv2.cvtColor(working,cv2.COLOR_BGR2GRAY),(32,12)).astype(np.float32).ravel()
                gray-=gray.mean();norm=float(np.linalg.norm(gray))
                if norm<1e-6:
                    context_metrics['flat_crops_rejected']+=1;continue
                proposal.update(signature=(gray/norm).tolist(),reading=greedy_reading(posterior_cache[key],ocr.tokens),
                                posterior_key=key,source_id=digest)
                if motion:proposal['motion_support']=motion.support(proposal['box'])
                unique.append(proposal)
            proposals=unique
        for vehicle in ([] if args.plate_first else eligible):
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
            group['owner']=proposal['owner']
            group['owner_locked']=proposal.get('owner_locked',True)
            group['recognition_conflict']=proposal.get('recognition_conflict',False)
            group['observed'].append(dict(position_ms=pts,box=b,source_id=digest,
                owner=proposal['owner'],owner_candidate=proposal.get('owner_candidate',proposal['owner']),
                owner_locked=proposal.get('owner_locked',True),crop_origin=proposal.get('crop_origin','tight_vehicle')))
            group['metadata'][digest]=dict(box=b,quad=proposal['quad'],posterior_key=proposal.get('posterior_key'))
            bank.add(gid,native,pts,quality=quality)
            if motion:motion.observe(gid,b)
        if args.plate_first:
            # Only retained bank originals need posterior payloads after routing.
            retained={g['metadata'][s.source_id]['posterior_key'] for gid,g in groups.items() for s in bank.get(gid)}
            posterior_cache={k:v for k,v in posterior_cache.items() if k in retained}
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
            rows=posterior_cache.get(metadata.get('posterior_key'))
            if rows is None:rows=ocr.probabilities(working);ocr_calls+=1
            multi.add(rows,timestamp_ms=sample.pts_ms,source_id=sample.source_id,quality=sample.quality)
            relative=f"assets/{group['id']}-{i}.png"
            if not cv2.imwrite(str(args.output/relative),sample.image):raise ValueError('Native PNG save failed')
            ownership=next((o['owner'] for o in group['observed'] if o['source_id']==sample.source_id and o.get('owner_locked')),None)
            observations.append(dict(position_ms=sample.pts_ms,source_id=sample.source_id,quality=sample.quality,vehicle_id=ownership,
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
        if args.visual_identity and group.get('recognition_conflict'):
            for value in values.values():value['accepted']=None
        public_samples=[{k:v for k,v in o.items() if k!='rows'} for o in observations]
        for value in values.values():
            value['accepted_for_vehicle']=value['accepted'] if group.get('owner_locked') else None
        historical_parents={o['owner'] for o in group['observed'] if o.get('owner_locked') and o['owner']}
        predictions.append(dict(id=group['id'],owner=group['owner'],owner_locked=group.get('owner_locked'),observed=group['observed'],
            parent_vehicle_id=next(iter(historical_parents)) if len(historical_parents)==1 else None,
            parent_conflict=len(historical_parents)>1,
            recognition_conflict=group.get('recognition_conflict',False),
            samples=public_samples,observations=len(samples),methods=values,reconstruction_status=state.status))
        posteriors.append(dict(id=group['id'],observations=observations))
        if len(predictions)%10==0:
            write_json(args.output/'progress.json',dict(stage='recognizing',groups_done=len(predictions),groups=len(groups),ocr_calls=ocr_calls+identity_ocr_calls))
            print(json.dumps(dict(stage='recognizing',groups_done=len(predictions),ocr_calls=ocr_calls+identity_ocr_calls)),flush=True)
    with gzip.open(args.output/'posteriors.json.gz','wt',encoding='utf-8') as f:json.dump(dict(tokens=ocr.tokens,groups=posteriors),f,allow_nan=False)
    vehicle_results=[]
    if lifecycle:
        evidence={};children={}
        for group in predictions:
            parent=group['parent_vehicle_id']
            if parent:children.setdefault(parent,[]).append(group['id'])
        for group in posteriors:
            for observation in group['observations']:
                if observation['vehicle_id']:
                    evidence.setdefault(observation['vehicle_id'],[]).append(observation)
        for parent,items in evidence.items():
            selected,diagnostics=one_view_per_timestamp(items,ocr.tokens)
            selected=sorted(selected,key=lambda o:(-o['quality'],o['position_ms']))[:6]
            selected.sort(key=lambda o:o['position_ms']);reader=CTCPlateReader(ocr.tokens)
            for observation in selected:reader.add(observation['rows'],timestamp_ms=observation['position_ms'],source_id=observation['source_id'],quality=observation['quality'])
            values=result(reader)
            if diagnostics['conflicting_plate_views']:values['accepted']=None
            vehicle_results.append(dict(id=parent,children=children.get(parent,[]),methods=values,
                samples=[{k:v for k,v in o.items() if k!='rows'} for o in selected],observations=len(selected),**diagnostics))
        write_json(args.output/'vehicle-timeline.json',dict(width=tracking['width'],height=tracking['height'],frames=vehicle_frames))
    document=dict(schema_version=1,source_sha256=source.sha256,tracking_report_sha256=sha(tracking_path),
        timeline_sha256=sha(args.tracking/'timeline.json'),review_sha256=hashlib.sha256(review_bytes).hexdigest() if review else None,
        excluded_review_ids=sorted(excluded),models=models,decoded_frames=decoded,sampled_frames=sampled,
        sample_interval_ms=250,adaptive_plates=bool(bursts),additional_burst_frames=burst_frames,
        maximum_retained_originals=6,roi_calls=roi_calls,ocr_calls=ocr_calls+identity_ocr_calls,
        plate_first=args.plate_first,plate_flow=args.plate_flow,motion_diagnostics=dict(motion.metrics) if motion else None,
        identity_ocr_calls=identity_ocr_calls,context_diagnostics=dict(context_metrics),
        vehicle_lifecycle=bool(lifecycle),vehicle_lifecycle_counters=dict(lifecycle.counters) if lifecycle else None,
        visual_identity=args.visual_identity,
        vehicle_motion_diagnostics=dict(vehicle_motion.metrics) if lifecycle else None,
        vehicles=list(lifecycle.records.values()) if lifecycle else [],vehicle_results=vehicle_results,
        router=dict(router.counters),search_diagnostics=search.diagnostics(),groups_with_owned_crops=len(groups),
        groups_with_ocr=len(predictions),groups_evicted_before_ocr=len(groups)-len(predictions),
        wall_seconds=round(time.monotonic()-began,2),reached_eof=True,labels_used_during_inference=False,
        runner_sha256=sha(args.output/'runner.py'),inference_source_sha256=source_hashes,
        ownership_source_sha256=source_hashes['plate_identity' if args.plate_first else 'owned_plate'],
        groups=predictions,scope='Full permitted source; frozen dense tracker; no events or training; selected human reference evaluation follows separately')
    write_json(args.output/'predictions.json',document)
    write_json(args.output/'progress.json',dict(stage='inference_complete',groups=len(predictions),ocr_calls=ocr_calls+identity_ocr_calls))
    print(json.dumps(dict(stage='inference_complete',groups=len(predictions),ocr_calls=ocr_calls+identity_ocr_calls,wall_seconds=document['wall_seconds'])),flush=True)


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
        scorer_sha256=sha(Path(__file__)),
        labels_sha256=sha(args.review/'labels.json'),summary=summary,conflicting_groups=list(bad),cases=cases,
        processing={k:v for k,v in d.items() if k!='groups'})
    if d.get('vehicle_lifecycle'):
        parents={v['id']:v for v in d['vehicle_results']};vehicle_cases=[];parent_truth={}
        for case in cases:
            group=byid.get(case['group']);pid=group.get('parent_vehicle_id') if group else None
            parent=parents.get(pid)
            if pid:parent_truth.setdefault(pid,set()).add(case['expected'])
            top=parent['methods']['top'] if parent and not parent['conflicting_plate_views'] else None
            vehicle_cases.append(dict(id=case['id'],vehicle_id=pid,top=top,expected=case['expected'],
                exact=bool(top and top==case['expected']),accepted=parent['methods']['accepted'] if parent else None))
        mixed={pid for pid,truths in parent_truth.items() if len(truths)>1}
        for case in vehicle_cases:
            if case['vehicle_id'] in mixed:case.update(top=None,exact=False,accepted=None)
        report['vehicle_cases']=vehicle_cases
        report['vehicle_summary']=dict(exact=sum(c['exact'] for c in vehicle_cases),targets=len(vehicle_cases),
            matched_targets=sum(c['top'] is not None for c in vehicle_cases),conflicting_parents_quarantined=sorted(mixed),
            accepted_correct=sum(bool(c['accepted']) and c['accepted']==c['expected'] for c in vehicle_cases),
            accepted_wrong=sum(bool(c['accepted']) and c['accepted']!=c['expected'] for c in vehicle_cases))
    write_json(args.output/'accuracy.json',report);render(args.output,d,report)
    print(json.dumps(dict(stage='scored',summary=summary)),flush=True)


def render(root,d,report):
    cases={c['group']:c for c in report['cases'] if c['group']}
    cards=[]
    for g in d['groups']:
        c=cases.get(g['id']);pictures=''.join('<figure><img src="'+html.escape(o['path'])+'" alt="Native plate crop"><figcaption>'+str(o['position_ms']/1000)+' s</figcaption></figure>' for o in g['samples'])
        readings=''.join('<tr><td>'+m+'</td><td>'+html.escape(g['methods'][m]['top'] or 'No complete hypothesis')+'</td><td>'+html.escape(g['methods'][m]['accepted'] or 'Unconfirmed')+'</td><td>'+(', '.join(html.escape(x['plate']) for x in g['methods'][m]['candidates'][:5]) or 'None')+'</td></tr>' for m in METHODS)
        truth='<p>Existing human reference '+c['id']+': '+html.escape(c['expected'])+'</p>' if c else '<p>Unlabelled passage: correctness unmeasured.</p>'
        if g.get('recognition_conflict'):
            truth+='<p><strong>Conflicting OCR readings.</strong> The visual identity is retained; text acceptance is blocked. Inspect the alternative readings and original crops.</p>'
        cards.append('<section id="'+html.escape(g['id'])+'"><h2>'+html.escape(g['id'])+' · owner '+html.escape(g['owner'] or 'uncertain')+'</h2>'+truth+'<div class="frames">'+pictures+'</div><div class="scroll"><table><tr><th>Method</th><th>Top hypothesis</th><th>Model gate (text only)</th><th>Top five candidates</th></tr>'+readings+'</table></div></section>')
    summary=''.join('<tr><td>'+m+'</td><td>'+str(s['matched_exact'])+'/'+str(s['matched_targets'])+'</td><td>'+str(s['multi_frame_exact'])+'/'+str(s['multi_frame_targets'])+'</td></tr>' for m,s in report['summary'].items())
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sentinel · Owned multi-frame OCR</title><style>body{font:16px system-ui;background:#f6f7f2;color:#18352e;max-width:1350px;margin:30px auto;padding:0 18px}h1,h2{color:#0f766e}section{padding:18px;margin:20px 0;border:1px solid #cbd9ce;border-radius:10px;background:white}.frames{display:flex;gap:10px;overflow:auto}figure{margin:5px;flex:none}img{max-width:280px;max-height:130px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #cbd9ce;text-align:left}.scroll{overflow:auto}</style><h1>Owned multi-frame recognition</h1><p>Same OCR weights, same native evidence. Ambiguous plate ownership is rejected before fusion. All hypotheses and model scores remain unverified until labelled.</p><p>'+str(d['decoded_frames'])+' source frames · '+str(d['sampled_frames'])+' plate-search samples · '+str(d['groups_with_ocr'])+' evidence groups. <a href="http://127.0.0.1:7798/">Vehicle tracking viewer</a></p><table><tr><th>Method</th><th>Matched exact</th><th>≥3-frame exact</th></tr>'+summary+'</table><p>These are selected development references, not full-video accuracy. Unmatched references remain misses in the all-18 denominator. No watchlist events were emitted.</p>'+''.join(cards)+'</html>'
    if d.get('plate_first'):
        page=page.replace('Owned multi-frame recognition','Plate-first multi-frame recognition').replace(
            'Ambiguous plate ownership is rejected before fusion.',
            'Plate identities use geometry, appearance and OCR disagreement gates. Ambiguous vehicle ownership remains unassigned; text acceptance is separate from vehicle ownership. This is a local experiment, not the operator alert path.')
        rows=''.join('<tr><td>'+html.escape(c['id'])+'</td><td>'+html.escape(c['expected'])+'</td><td>'+(
            '<a href="#'+html.escape(c['group'])+'">'+html.escape(c['group'])+'</a>' if c['group'] else 'No unique matched crop')+
            '</td><td>'+('Exact' if c['methods']['multi']['exact'] else 'Miss / wrong')+'</td></tr>' for c in report['cases'])
        panel='<section><h2>All 18 development references</h2><p>Automatic crop association uses time and geometry. These labels did not guide inference.</p><table><tr><th>Reference</th><th>Human label</th><th>Plate history</th><th>Multi-frame result</th></tr>'+rows+'</table></section>'
        page=page.replace(''.join(cards),panel+''.join(cards))
        if d.get('plate_flow'):
            page=page.replace('Plate-first multi-frame recognition','Motion-supported plate recognition').replace(
                'Plate identities use geometry, appearance and OCR disagreement gates.',
                'Plate identities use geometry, appearance, OCR disagreement gates and forward/backward checked motion between decoded video frames. Optical flow supplies no additional OCR votes.')
    if d.get('vehicle_lifecycle'):
        from vehicle_lifecycle_view import augment
        page=augment(page,d,report)
    (root/'index.html').write_bytes(page.encode('utf-8'))


def render_saved(args):
    prediction=args.output/'predictions.json';document=json.loads(prediction.read_bytes())
    report=json.loads((args.output/'accuracy.json').read_bytes())
    if report['predictions_sha256']!=sha(prediction):raise ValueError('Score does not bind this capture')
    summary=json.loads(args.review_summary.read_bytes()) if args.review_summary else None
    if summary is not None and summary['predictions_sha256']!=sha(prediction):raise ValueError('Review mapping does not bind this capture')
    if summary is not None and not document.get('vehicle_lifecycle'):raise ValueError('Vehicle review mapping requires a vehicle-lifecycle capture')
    render(args.output,document,report)
    if summary is not None:
        from vehicle_lifecycle_view import case_panel
        page=args.output/'index.html';text=page.read_text(encoding='utf-8');marker='<section id="vehicle-lifecycle">'
        page.write_text(text.replace(marker,case_panel(summary)+marker,1),encoding='utf-8')
    print(json.dumps(dict(stage='viewer_rendered',inference_rerun=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('run','score','render'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--tracking',type=Path,default=Path('/workspace/data/benchmarks/vehicle-tracking-960-20261110'))
    p.add_argument('--plate-first',action='store_true',help='Padded native search and independent conservative plate identities')
    p.add_argument('--plate-flow',action='store_true',help='Forward/backward checked motion support between sampled plate detections')
    p.add_argument('--vehicle-lifecycle',action='store_true',help='Persistent vehicle parents, occlusion states and multiple plate views')
    p.add_argument('--visual-identity',action='store_true',help='Retain visually established identity through OCR conflicts; block uncertain text acceptance')
    p.add_argument('--adaptive-plates',action='store_true',help='Bounded extra plate searches for new or reappearing vehicles')
    p.add_argument('--recordings-dir',type=Path,default=Path('/recordings'));p.add_argument('--cache',type=Path,default=Path('/models/experiments/awiros'))
    p.add_argument('--review',type=Path,default=Path('data/benchmarks/recorded-footage-20261104/manual-review'))
    p.add_argument('--review-summary',type=Path,help='After-inference vehicle mapping for --phase render only')
    args=p.parse_args()
    if args.plate_flow and not args.plate_first:p.error('--plate-flow requires --plate-first')
    if args.vehicle_lifecycle and not args.plate_flow:p.error('--vehicle-lifecycle requires --plate-flow')
    if args.visual_identity and not args.vehicle_lifecycle:p.error('--visual-identity requires --vehicle-lifecycle')
    if args.adaptive_plates and not args.vehicle_lifecycle:p.error('--adaptive-plates requires --vehicle-lifecycle')
    if args.review_summary and args.phase!='render':p.error('--review-summary is only for after-inference rendering')
    if args.phase=='run':run(args)
    elif args.phase=='score':score(args)
    else:render_saved(args)
