"""Controlled native crop-extent replay on frozen tracks/times and OCR weights.

Decode the permitted source, refine only its existing retained crop boxes, and
reuse unchanged original posteriors. No labels, new detections or extra frame
votes enter inference. The separate scorer follows after the fresh run.
"""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time

from run_owned_multiframe import CHECKPOINT,reading_result,build_vehicle_results,sha,write_json
from sentinel_edge.plate_extent import recover_readable_extent
from sentinel_edge.recorded_source import RecordingLibrary,read_recording
from sentinel_edge.ctc_fusion import CTCPlateReader
from sentinel_edge.reconstruction import CropSample


def replay(args):
    import cv2
    from sentinel_edge.awiros_experiment import AwirosExperiment
    from sentinel_edge.live_reconstruction import ReconstructionState,update,disagreement
    if args.output.exists():raise ValueError('Use a fresh output folder')
    old=json.loads((args.capture/'predictions.json').read_bytes())
    frozen=json.loads(gzip.decompress((args.capture/'posteriors.json.gz').read_bytes()))
    if not old['reached_eof'] or old['labels_used_during_inference']:raise ValueError('Completed label-free source required')
    records,_=RecordingLibrary(args.recordings_dir).scan()
    source=next((v for v in records.values() if v.sha256==old['source_sha256']),None)
    if source is None:raise ValueError('Permitted recording does not match frozen capture')
    checkpoint=args.cache/'adaptations/indian-corpus-encoder.safetensors'
    if sha(checkpoint)!=CHECKPOINT:raise ValueError('OCR checkpoint changed')
    args.output.mkdir(parents=True);(args.output/'assets').mkdir()
    runner=Path(__file__).resolve();helper=runner.parents[1]/'edge/sentinel_edge'
    names=('plate_extent','crop_completion','plate_evidence','ctc_fusion','live_reconstruction','reconstruction')
    snapshots={}
    for name in names:
        content=(helper/(name+'.py')).read_bytes();snapshots[name]=hashlib.sha256(content).hexdigest()
        (args.output/('source-'+name+'.py')).write_bytes(content)
    (args.output/'runner.py').write_bytes(runner.read_bytes())
    shared=runner.with_name('run_owned_multiframe.py');shared_sha=sha(shared)
    (args.output/'source-run_owned_multiframe.py').write_bytes(shared.read_bytes())
    write_json(args.output/'progress.json',dict(stage='loading_frozen_ocr'))
    ocr=AwirosExperiment(args.cache,device='gpu',checkpoint=checkpoint)
    if tuple(ocr.tokens)!=tuple(frozen['tokens']):raise ValueError('OCR tokens changed')
    document=copy.deepcopy(old);groups={g['id']:g for g in document['groups']}
    pending={};other_boxes={}
    for group in frozen['groups']:
        original=groups[group['id']]
        if [{k:v for k,v in o.items() if k!='rows'} for o in group['observations']]!=original['samples']:
            raise ValueError('Posterior/crop metadata changed')
        for observation in group['observations']:pending.setdefault(observation['position_ms'],[]).append((group,observation))
        for observation in original['observed']:other_boxes.setdefault(observation['position_ms'],[]).append(observation['box'])
    started=time.monotonic();decoded=changed=ocr_calls=0;seen=set();images={}
    for _,frame in read_recording(source,threading.Event(),paced=False):
        decoded+=1;pts=frame.pts_ms-source.start_pts_ms
        for group,observation in pending.get(pts,[]):
            seed=observation['box'];x,y,x2,y2=map(int,seed);native=frame.image[y:y2,x:x2]
            if hashlib.sha256(native.tobytes()).hexdigest()!=observation['source_id']:
                raise ValueError('Frozen crop is not the same source frame/rectangle')
            seen.add((group['id'],pts));original_id=observation['source_id']
            others=[b for b in other_boxes.get(pts,[]) if b!=seed]
            recovery=recover_readable_extent(frame.image,seed,observation['rows'],ocr.tokens,others)
            if recovery:
                x,y,x2,y2=recovery['box'];native=frame.image[y:y2,x:x2]
                observation.update(box=recovery['box'],source_id=hashlib.sha256(native.tobytes()).hexdigest(),
                                   rows=ocr.probabilities(native),working_quad=None,crop_completion=recovery)
                changed+=1;ocr_calls+=1
                for observed in groups[group['id']]['observed']:
                    if observed['position_ms']==pts and observed['source_id']==original_id:
                        observed.update(box=observation['box'],source_id=observation['source_id'])
            images[(group['id'],pts)]=native.copy()
        if decoded%150==0:print(json.dumps(dict(stage='refining_saved_crops',decoded=decoded,changed=changed)),flush=True)
    if decoded!=old['decoded_frames'] or len(seen)!=sum(len(g['observations']) for g in frozen['groups']):
        raise ValueError('Source did not complete the frozen crop inventory')
    for count,group in enumerate(frozen['groups'],1):
        result=groups[group['id']];observations=group['observations'];samples=[];reader=CTCPlateReader(ocr.tokens)
        for i,o in enumerate(observations):
            native=images[(group['id'],o['position_ms'])];o['path']=f"assets/{group['id']}-{i}.png"
            if not cv2.imwrite(str(args.output/o['path']),native):raise ValueError('Native crop save failed')
            # Keep original fusion weights and timestamps for a crop-only comparison.
            sample=CropSample(native,o['position_ms'],o['source_id'],o['quality']);samples.append(sample)
            reader.add(o['rows'],timestamp_ms=o['position_ms'],source_id=o['source_id'],quality=o['quality'])
        best=max(observations,key=lambda o:(o['quality'],-o['position_ms']))
        single=CTCPlateReader(ocr.tokens);single.add(best['rows'],timestamp_ms=best['position_ms'],source_id=best['source_id'])
        methods=dict(single=reading_result(single),multi=reading_result(reader))
        state=ReconstructionState();update(state,reader,samples,ocr,samples[-1].pts_ms)
        methods['reconstruction']=reading_result(reader)
        if disagreement(state,reader.accepted(.85),.85):methods['reconstruction']['accepted']=None
        if result.get('recognition_conflict'):
            for value in methods.values():value['accepted']=None
        for value in methods.values():value['accepted_for_vehicle']=value['accepted'] if result.get('owner_locked') else None
        result.update(samples=[{k:v for k,v in o.items() if k!='rows'} for o in observations],
                      methods=methods,reconstruction_status=state.status)
        if count%40==0:print(json.dumps(dict(stage='reconstructing',groups_done=count,changed_crops=changed)),flush=True)
    timeline=json.loads((args.capture/'vehicle-timeline.json').read_bytes())
    from sentinel_edge.plate_evidence import annotate_plate_evidence
    document['plate_validation']=annotate_plate_evidence(document['groups'],frozen['groups'],ocr.tokens,timeline,
        lambda sample:cv2.imread(str(args.output/sample['path'])))
    document['vehicle_results']=build_vehicle_results(document['groups'],frozen['groups'],ocr.tokens)
    document.update(complete_crops=True,refine_characters=True,scope='Controlled crop-only replay; frozen tracks, timestamps, original weights and unchanged posteriors; no labels or events',
        wall_seconds=round(time.monotonic()-started,2),ocr_calls=ocr_calls,identity_ocr_calls=0,
        runner_sha256=sha(runner),inference_source_sha256=snapshots,
        search_diagnostics=dict(text_detector_enabled=False,crop_completion=dict(character_row_extent=changed),
                                note='No detector rerun: frozen original crop inventory'),
        crop_replay=dict(input_predictions_sha256=sha(args.capture/'predictions.json'),
                         input_posteriors_sha256=sha(args.capture/'posteriors.json.gz'),
                         changed_originals=changed,reused_original_posteriors=sum(len(g['observations']) for g in frozen['groups'])-changed,
                         decoded_frames=decoded,frame_weights_unchanged=True,labels_used=False,shared_runner_sha256=shared_sha))
    for name in names:
        if sha(helper/(name+'.py'))!=snapshots[name]:raise ValueError('Inference helper changed during replay')
    if sha(shared)!=shared_sha:raise ValueError('Shared inference helper changed during replay')
    shutil.copyfile(args.capture/'vehicle-timeline.json',args.output/'vehicle-timeline.json')
    with gzip.open(args.output/'posteriors.json.gz','wt',encoding='utf-8') as stream:json.dump(frozen,stream,allow_nan=False)
    write_json(args.output/'predictions.json',document)
    write_json(args.output/'progress.json',dict(stage='inference_complete',changed_crops=changed,ocr_calls=ocr_calls))
    print(json.dumps(dict(stage='inference_complete',changed_crops=changed,wall_seconds=document['wall_seconds'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--recordings-dir',type=Path,default=Path('/recordings'))
    parser.add_argument('--cache',type=Path,default=Path('/models/experiments/awiros'))
    replay(parser.parse_args())
