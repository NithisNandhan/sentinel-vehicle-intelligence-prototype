"""Freeze a complete recorded-pipeline run for local blind human review.

This observer never supplies labels, alters models or delivers central events.
Every decoded source frame is counted; inference keeps the dashboard's normal
sampling and gates. Review evidence is retained only for this permitted file.
"""
import argparse
import ast
from datetime import datetime,timezone
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time

from sentinel_edge.awiros_experiment import file_hash
from sentinel_edge.recorded_source import RecordingLibrary,read_recording,new_session_id

PREDICTION_POLICY='passed_vehicles_strongest_window_v1'
GROUPING_POLICY='recognition_passage_v1'
MAX_WITHHELD_PROPOSALS=512
MAX_WITHHELD_NATIVE_BYTES=32*1024*1024


def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')


def write_json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def identity_guard_settings():
    """Effective text-free ownership and veto-only OCR separation policies.

    The source hashes bind fixed thresholds not exposed as runtime options;
    these are conservative evidence guards, not calibrated identity proofs.
    """
    from sentinel_edge import plate_continuity,plate_identity,plate_ownership,tracked_ocr,crop_collector,sampled_frames,plate_filter
    return dict(
        plate_ownership=dict(policy='unique_native_vehicle_containment_v1',
            minimum_area_fraction=.85,plate_center_inside=True,
            compared_regions='all_current_vehicle_boxes',before_ocr_and_fusion=True),
        plate_continuity=asdict(plate_continuity.DEFAULT_POLICY),
        vehicle_tracking=dict(every_decoded_frame=os.getenv('SENTINEL_CONTINUOUS_TRACKING','false').lower()=='true',
            recorded_only=True,ocr_sampling_unchanged=True),
        fusion_text_support=dict(minimum_confident_characters=1,character_peak=.6,
            withheld_proposals_preserved=True,no_character_support_does_not_mean_non_plate=True),
        independent_ocr_conflict=dict(policy='independent_reading_conflict_v1',
            equal_length_differing_characters_min=3,character_score_min=.9,
            terminal_digits=4,terminal_differing_characters_min=2,
            unequal_prefix_lengths_allowed=True,veto_only_never_merge=True,
            scores_are_uncalibrated=True),
        source_sha256={module.__name__.rsplit('.',1)[-1]:file_hash(Path(module.__file__))
            for module in (plate_ownership,plate_continuity,plate_identity,tracked_ocr,crop_collector,sampled_frames,plate_filter)})


def reference_coverage(manifest,review,reference_view):
    """Suggest old-reference links by native geometry/time only, never OCR text.

    These are not imported labels: duplicate passages, missing vehicles and
    every current track still require human review before full-clip scoring.
    """
    from sentinel_edge.automatic_crops import overlap
    review,reference_view=Path(review),Path(reference_view)
    original=json.loads((review/'manifest.json').read_bytes())
    labels=json.loads((review/'labels.json').read_bytes())
    previous=json.loads(reference_view.read_bytes())
    if any(d.get('source_sha256')!=manifest['source']['sha256'] for d in (original,labels,previous)):
        raise ValueError('Existing human references belong to another source')
    bindings=previous.get('bindings',{})
    if (bindings.get('manifest_sha256')!=file_hash(review/'manifest.json')
            or bindings.get('labels_sha256')!=file_hash(review/'labels.json')):
        raise ValueError('Existing reference provenance changed')
    samples={r['id']:r for r in original['samples']}
    truth={r['id']:r for r in labels['labels']}
    points={}
    source=review/'ocr-model-dev/rectify.py'
    for node in ast.parse(source.read_text(encoding='utf-8')).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('CORNERS','TEXT_REGIONS') for t in node.targets):
            points.update(ast.literal_eval(node.value))
    rows=[]
    for reference in previous['references']:
        sid=reference['reference_id'];sample=samples[sid];label=truth[sid]
        if label['status']!='readable' or label['plate']!=reference['actual']:
            raise ValueError('Existing readable human reference is inconsistent')
        box=sample['box'];quad=points.get(sid)
        if quad:
            box=[box[0]+min(p[0] for p in quad),box[1]+min(p[1] for p in quad),
                 box[0]+max(p[0] for p in quad),box[1]+max(p[1] for p in quad)]
        candidates=[]
        for track in manifest['tracks']:
            best=max((overlap(o['box'],box) for o in track.get('plate_observations',[])
                      if abs(o['position_ms']-sample['position_ms'])<=1),default=0.)
            if best>=.3:candidates.append(dict(track_id=track['id'],iou=round(best,4)))
        rows.append(dict(reference_number=reference['number'],reference_id=sid,position_ms=sample['position_ms'],
            status='unique_geometry_needs_review' if len(candidates)==1 else 'ambiguous_geometry' if candidates else 'no_matching_observation',
            candidates=sorted(candidates,key=lambda r:-r['iou'])))
    return dict(source_sha256=manifest['source']['sha256'],human_reference_count=len(rows),
        labels_imported=0,predictions_used_for_linking=False,manual_inventory_complete=False,
        original_manifest_sha256=file_hash(review/'manifest.json'),original_labels_sha256=file_hash(review/'labels.json'),
        reference_view_sha256=file_hash(reference_view),references=rows,
        notice='Historical development references, linked only by source hash, video time and native geometry. All suggestions require human confirmation; no labels are imported and this is not full-clip accuracy.')


class CountedRun:
    def __init__(self,recording):
        self.recording=recording;self.session_id=new_session_id()
        self.start_ms=0;self.paced=False;self.paused=None
        self.decoded=0;self.last_pts=None;self.reached_eof=False

    def frames(self,stop):
        for protocol,frame in read_recording(self.recording,stop,paced=False):
            self.decoded+=1;self.last_pts=frame.pts_ms
            yield protocol,frame
        self.reached_eof=not stop.is_set()


class ReviewObserver:
    def __init__(self,recording,output):
        self.recording=recording;self.output=output
        self.segment=0;self.tracks={};self.predictions={};self.native={}
        self.last_position=0;self.samples=0;self.started=time.monotonic()
        self.pipeline=None
        self.withheld_proposals=[];self.withheld_predictions=[];self.withheld_native=[]
        self.withheld_seen=0;self.withheld_bytes=0;self.withheld_keys=set()

    def identity(self,track_id,passage_id=None):
        value=str(track_id)
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}',value):raise ValueError('Unsafe tracker identity')
        if passage_id:
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',str(passage_id)):
                raise ValueError('Unsafe passage identity')
            return f'passage-s{self.segment}-{passage_id}'
        # Small/unprocessed detections have no recognition history and cannot
        # contribute plate evidence. Do not silently mix them into a passage.
        return f'track-s{self.segment}-{value}-unprocessed'

    def ensure(self,track_id,position,*,history=None,passage_id=None):
        passage_id=passage_id or getattr(history,'passage_id',None)
        uid=self.identity(track_id,passage_id)
        if uid not in self.tracks:
            self.tracks[uid]=dict(id=uid,tracker_id=str(track_id),segment=self.segment,
                tracker_ids=[],passage_id=passage_id,
                identity_kind='recognition_passage' if passage_id else 'unprocessed_detection',
                passage_start_reason=getattr(history,'passage_start_reason',None),
                first_ms=position,last_ms=position,vehicle_class=None,analysed_samples=0,
                plate_crop_count=0,vehicle_preview=None,samples=[],plate_observations=[])
            self.predictions[uid]=dict(first_ms=position,last_ms=position,top_guess=None,
                candidate_ms=None,candidates=[],accepted=[],observations=0,selection_strength=[])
            self.native[uid]=dict(vehicle=None,area=0,plates=[],selected=None,latest_ocr=None)
        if str(track_id) not in self.tracks[uid]['tracker_ids']:
            self.tracks[uid]['tracker_ids'].append(str(track_id))
        return uid,self.tracks[uid],self.predictions[uid]

    def depth(self):return 0

    def accept(self,camera_id,track_id,candidate,pts_ms,*,recorded=None):
        if self.pipeline is not None:self.segment=self.pipeline.metrics['scene_resets']
        position=pts_ms-self.recording.start_pts_ms
        history=self.pipeline.histories.get(track_id) if self.pipeline is not None else None
        if self.pipeline is not None and not getattr(history,'passage_id',None):
            raise ValueError('Accepted reading has no recognition passage')
        _,_,prediction=self.ensure(track_id,position,history=history)
        if candidate.plate not in {r['plate'] for r in prediction['accepted']}:
            prediction['accepted'].append(dict(plate=candidate.plate,pts_ms=position))
        return True

    def on_passage(self,outcome):
        # The true acceptance callback above is the operational gate. A partial
        # final result is never silently upgraded into an accepted plate.
        if self.pipeline is not None:self.segment=self.pipeline.metrics['scene_resets']
        _,row,_=self.ensure(outcome.track_ids[-1],outcome.first_pts-self.recording.start_pts_ms,
                          passage_id=outcome.passage_id)
        row['tracker_ids']=list(dict.fromkeys([*row['tracker_ids'],*map(str,outcome.track_ids)]))
        row['passage_end_reason']=outcome.cause
        row['passage_first_ms']=outcome.first_pts-self.recording.start_pts_ms
        row['passage_last_ms']=outcome.last_pts-self.recording.start_pts_ms

    def publish(self,snapshot,stop=None):
        pass

    def capture_withheld(self,frame,pipeline):
        """Retain rejected observations for inspection, outside accuracy units.

        Low OCR evidence does not prove a region is not a plate. The native
        image and reason remain visible, while raw OCR stays in predictions
        hidden during blind labelling. No crop becomes a fusion/scoring vote.
        """
        import numpy as np
        for item in getattr(pipeline,'rejected_proposals',[]) or []:
            if item.get('used_for_fusion') is not False or item.get('pts_ms')!=frame.pts_ms:
                raise ValueError('Withheld proposal must be a non-fusion observation of the current frame')
            track_id=item['track_id'];passage_id=item.get('passage_id')
            self.identity(track_id,passage_id)
            crop=item.get('crop');box=item.get('box')
            if (not isinstance(crop,np.ndarray) or crop.dtype!=np.uint8 or crop.ndim!=3
                    or crop.shape[2]!=3 or min(crop.shape[:2])<1
                    or not isinstance(box,(tuple,list)) or len(box)!=4
                    or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in box)
                    or not 0<=box[0]<box[2]<=frame.image.shape[1]
                    or not 0<=box[1]<box[3]<=frame.image.shape[0]):
                raise ValueError('Invalid native withheld proposal')
            digest=hashlib.sha256(crop.tobytes()).hexdigest()
            key=(self.segment,str(track_id),passage_id,frame.pts_ms,digest)
            if key in self.withheld_keys:continue
            self.withheld_seen+=1
            if (len(self.withheld_proposals)>=MAX_WITHHELD_PROPOSALS
                    or self.withheld_bytes+crop.nbytes>MAX_WITHHELD_NATIVE_BYTES):
                continue
            raw=item.get('raw_text');characters=item.get('characters',())
            reason=item.get('reason')
            if (raw is not None and (not isinstance(raw,str) or len(raw)>128)
                    or not isinstance(reason,str) or not 1<=len(reason)<=256
                    or not isinstance(characters,(tuple,list)) or len(characters)>128):
                raise ValueError('Invalid withheld OCR diagnostic')
            def character(value):
                if isinstance(value,dict):text,score=value.get('character'),value.get('score')
                elif isinstance(value,(tuple,list)) and len(value)==2:text,score=value
                else:raise ValueError('Invalid withheld character score')
                if (not isinstance(text,str) or len(text)!=1 or isinstance(score,bool)
                        or not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=1):
                    raise ValueError('Invalid withheld character score')
                return dict(character=text,score=float(score))
            clean=[]
            for value in characters:
                entry=character(value)
                alternatives=value.get('alternatives',[]) if isinstance(value,dict) else []
                if not isinstance(alternatives,(tuple,list)) or len(alternatives)>3:
                    raise ValueError('Invalid withheld character alternatives')
                entry['alternatives']=[character(alt) for alt in alternatives]
                clean.append(entry)
            uid=f'withheld-{len(self.withheld_proposals)+1:06d}'
            position=frame.pts_ms-self.recording.start_pts_ms
            row=dict(id=uid,tracker_id=str(track_id),passage_id=passage_id,segment=self.segment,
                position_ms=position,box=list(map(float,box)),width=crop.shape[1],height=crop.shape[0],
                reason=reason,used_for_fusion=False,path=f'assets/{uid}.png',pixel_sha256=digest)
            history=pipeline.histories.get(track_id)
            if history is not None and getattr(history,'passage_id',None)==passage_id:
                row['vehicle_box']=list(map(float,history.box))
            self.withheld_proposals.append(row);self.withheld_native.append(crop.copy())
            self.withheld_predictions.append(dict(id=uid,raw_text=raw,characters=clean,
                position_ms=position,passage_id=passage_id,used_for_fusion=False))
            self.withheld_bytes+=crop.nbytes;self.withheld_keys.add(key)

    def withheld_retention(self):
        return dict(seen=self.withheld_seen,retained=len(self.withheld_proposals),
            omitted=self.withheld_seen-len(self.withheld_proposals),
            truncated=self.withheld_seen>len(self.withheld_proposals),
            maximum_proposals=MAX_WITHHELD_PROPOSALS,maximum_native_bytes=MAX_WITHHELD_NATIVE_BYTES)

    def on_sample(self,frame,tracks,displays,pipeline):
        from sentinel_edge.live_detection_viewer import crop_box,clipped_box
        from sentinel_edge.plate_quality import crop_quality
        from sentinel_edge.passed_vehicles import PassedVehicles
        from sentinel_edge.recognition_contract import recognition_state
        self.pipeline=pipeline
        self.segment=pipeline.metrics['scene_resets']
        position=frame.pts_ms-self.recording.start_pts_ms
        self.last_position=position;self.samples+=1
        for track in tracks:
            history=pipeline.histories.get(track.track_id)
            uid,row,prediction=self.ensure(track.track_id,position,history=history)
            row['last_ms']=position;row['analysed_samples']+=1
            row['vehicle_class']=int(track.vehicle_class)
            prediction['last_ms']=position
            box=clipped_box(frame.image,track.box)
            area=max(0,box[2]-box[0])*max(0,box[3]-box[1])
            if area>self.native[uid]['area']:
                crop=crop_box(frame.image,box)
                if crop is not None:
                    self.native[uid].update(vehicle=crop,area=area)
                    row['vehicle_preview']=dict(path=f'assets/{uid}-vehicle.png',position_ms=position,box=list(box))
        for display in displays:
            history=pipeline.histories.get(display['track_id'])
            passage_id=getattr(history,'passage_id',None)
            if not passage_id or display.get('passage_id',passage_id)!=passage_id:
                raise ValueError('Plate crop has no matching recognition passage')
            uid,row,prediction=self.ensure(display['track_id'],position,history=history)
            row['last_ms']=position;row['plate_crop_count']+=1
            crop=display['crop'];quality=crop_quality(crop)
            digest=hashlib.sha256(crop.tobytes()).hexdigest()
            sample=dict(position_ms=position,box=list(map(float,display['box'])),
                width=crop.shape[1],height=crop.shape[0],quality=quality,pixel_sha256=digest)
            row['plate_observations'].append(dict(position_ms=position,box=sample['box'],pixel_sha256=digest,
                passage_id=passage_id,tracker_id=str(display['track_id']),
                vehicle_box=list(map(float,display.get('vehicle_box',history.box)))))
            retained=self.native[uid]['plates']
            if not any(item[1]['pixel_sha256']==digest for item in retained):
                retained.append((crop.copy(),sample))
                retained.sort(key=lambda item:(-item[1]['quality'],item[1]['position_ms']))
                del retained[6:]
            if history is not None and history.reader.observation_count:
                candidate_ms=history.reader.samples[-1][0]-self.recording.start_pts_ms
                # A duplicate crop can leave the OCR clock unchanged. Preserve
                # the admitted image rather than attaching today's different
                # crop to an earlier reading merely because its ID is the same.
                if candidate_ms==position:
                    self.native[uid]['latest_ocr']=(crop.copy(),dict(sample))
                ranked=history.reader.candidates()
                if ranked:
                    state=recognition_state(display['ocr_status'])
                    strength=PassedVehicles._strength(state,display['observations'],ranked[0])
                    if prediction['top_guess'] is None or strength>tuple(prediction['selection_strength']):
                        prediction.update(top_guess=ranked[0].plate,
                            candidate_ms=candidate_ms,
                            selection_strength=list(strength),selected_state=state,
                            observations=history.reader.observation_count,
                            candidates=[dict(plate=c.plate,score=c.score,
                                minimum_character_score=c.minimum_character_score,
                                format_adjusted=c.format_adjusted) for c in ranked[:20]])
                        selected=self.native[uid]['latest_ocr']
                        if selected is None or selected[1]['position_ms']!=candidate_ms:
                            raise ValueError('Selected OCR window has no admitted crop')
                        self.native[uid]['selected']=selected
        self.capture_withheld(frame,pipeline)
        if self.samples%5==0:
            progress=dict(status='processing',position_ms=position,analysed_frames=self.samples,
                tracked_fragments=len(self.tracks),seconds=round(time.monotonic()-self.started,1))
            write_json(self.output/'progress.json',progress)
            print(json.dumps(progress),flush=True)

    def export_assets(self):
        import cv2
        assets=self.output/'assets';assets.mkdir()
        def save(path,image):
            ok,encoded=cv2.imencode('.png',image)
            if not ok:raise ValueError('Review PNG could not be encoded')
            blob=encoded.tobytes()
            (self.output/path).write_bytes(blob)
            return hashlib.sha256(blob).hexdigest()
        for uid,row in self.tracks.items():
            native=self.native[uid]
            if native['vehicle'] is not None:
                row['vehicle_preview']['image_sha256']=save(row['vehicle_preview']['path'],native['vehicle'])
            chosen=[]
            if native['selected'] is not None:chosen.append(native['selected'])
            chosen += [(crop,sample) for crop,sample in native['plates']
                       if not chosen or sample['pixel_sha256']!=chosen[0][1]['pixel_sha256']]
            for number,(crop,sample) in enumerate(chosen[:6]):
                sample['path']=f'assets/{uid}-plate-{number}.png'
                sample['image_sha256']=save(sample['path'],crop);row['samples'].append(sample)
        for row,crop in zip(self.withheld_proposals,self.withheld_native):
            row['image_sha256']=save(row['path'],crop)


def main(args):
    runner_sha=file_hash(Path(__file__))
    guards=identity_guard_settings()
    if args.output.exists():raise ValueError('Use a new run folder; frozen runs must not be overwritten')
    args.output.mkdir(parents=True)
    write_json(args.output/'progress.json',dict(status='loading_models',started_at=utc()))
    recordings,_=RecordingLibrary(args.recordings_dir).scan()
    recording=recordings.get(args.recording_id)
    if recording is None:raise ValueError('The selected recording requires a valid permission receipt')
    from sentinel_edge.config import Config
    from sentinel_edge.live_detection_viewer import load_models,model_identity
    from sentinel_edge.tracked_live_viewer import observe_tracked
    from sentinel_edge.plate_options import PlateOptions
    import av
    with av.open(str(recording.path)) as container:
        stream=container.streams.video[0]
        source=dict(name=recording.path.name,sha256=recording.sha256,
            recording_id=recording.recording_id,duration_ms=recording.duration_ms,
            width=stream.width,height=stream.height,
            fps=float(stream.average_rate) if stream.average_rate else None,
            advertised_frames=stream.frames,codec=stream.codec_context.name)
    config=Config.from_env(live=False)
    keep=os.getenv('SENTINEL_KEEP_UNCONFIRMED_VEHICLES','false').lower()=='true'
    checkpoint=args.cache/'adaptations/indian-corpus-encoder.safetensors'
    if file_hash(checkpoint)!='c2ba896d994b174071c95ace0f5ffc8a61cec3fc6eecce8f0bc5feb7dec4456d':
        raise ValueError('Deployed OCR checkpoint changed')
    reader,detector,vehicle=load_models(config,args.cache,checkpoint,
        detector_profile=args.detector_profile,device='gpu',keep_unconfirmed=keep)
    identity=model_identity(reader,args.detector_profile,detector=detector,vehicle_model=vehicle)
    options=PlateOptions.from_env()
    counted=CountedRun(recording);observer=ReviewObserver(recording,args.output)
    stop=threading.Event();started_at=utc();began=time.monotonic()
    summary=observe_tracked(config,recording,reader,detector,vehicle,observer,stop,
        delivery=observer,detector_profile=args.detector_profile,recorded=counted,on_sample=observer.on_sample)
    if not counted.reached_eof:raise ValueError('The source did not reach EOF; no completed review can be frozen')
    if not counted.decoded or not observer.samples:raise ValueError('No video frames were processed')
    accepted_strings={item['plate'] for row in observer.predictions.values() for item in row['accepted']}
    if len(accepted_strings)!=summary['unique_model_accepted_unverified']:
        raise ValueError('Acceptance observer does not reconcile with pipeline summary')
    if file_hash(Path(__file__))!=runner_sha or identity_guard_settings()!=guards:
        raise ValueError('Review runner or identity guard source changed during the capture; use a new run')
    observer.export_assets()
    settings=dict(options=options.manifest(),keep_unconfirmed_vehicles=keep,
        keep_unconfirmed_min_height=os.getenv('SENTINEL_KEEP_UNCONFIRMED_MIN_HEIGHT','0'),
        ocr_min_confidence=config.ocr_min_confidence,normal_dashboard_sampling=True,
        review_crops_per_track_max=6,review_track_count_cap=None,prediction_policy=PREDICTION_POLICY,
        grouping_policy=GROUPING_POLICY,
        identity_guards=guards,
        text_detector_configured=bool(os.getenv('SENTINEL_TEXT_DETECTOR_ASSETS','')))
    manifest=dict(schema_version=1,run_id=counted.session_id,source=source,models=identity,
        settings=settings,run=dict(started_at=started_at,completed_at=utc(),commit=args.commit,
            runner_sha256=runner_sha,reached_eof=True,decoded_frames=counted.decoded,
            last_decoded_ms=counted.last_pts-recording.start_pts_ms,
            last_analysed_ms=observer.last_position,analysed_frames=observer.samples,
            wall_seconds=round(time.monotonic()-began,2),pipeline_summary=summary),
        scope='Complete permitted recorded clip, previously used in development; not an unseen CCTV test',
        predictions_hidden_during_labelling=True,
        rejected_proposals=observer.withheld_proposals,
        withheld_proposal_retention=observer.withheld_retention(),
        tracks=sorted(observer.tracks.values(),key=lambda r:(r['first_ms'],r['id'])))
    write_json(args.output/'manifest.json',manifest)
    manifest_sha=file_hash(args.output/'manifest.json')
    write_json(args.output/'predictions.json',dict(schema_version=1,manifest_sha256=manifest_sha,
        source_sha256=recording.sha256,tracks=observer.predictions,
        withheld_proposals=observer.withheld_predictions,
        prediction_policy=PREDICTION_POLICY,
        caveat='Scores are uncalibrated. Top guesses and actual model acceptances are separate.'))
    if getattr(args,'reference_review',None) and getattr(args,'reference_view',None):
        write_json(args.output/'legacy-reference-links.json',reference_coverage(manifest,args.reference_review,args.reference_view))
    write_json(args.output/'review-status.json',dict(schema_version=1,manifest_sha256=manifest_sha,
        status='human_review_pending',tracked_fragments=len(observer.tracks),human_reviewed_tracks=0,
        pending_tracks=len(observer.tracks),inventory_complete=False,accuracy_available=False,
        prediction_policy=PREDICTION_POLICY,central_events_sent=0,
        notice='Human reviewer must label readable/unreadable/no-plate/non-vehicle/duplicate tracks and add missed vehicles. No current-run ground truth has been inferred from OCR.'))
    write_json(args.output/'progress.json',dict(status='completed',decoded_frames=counted.decoded,
        analysed_frames=observer.samples,tracked_fragments=len(observer.tracks),
        manifest_sha256=manifest_sha,review_ready=True))
    print(json.dumps(dict(stage='review_ready',decoded_frames=counted.decoded,
        analysed_frames=observer.samples,tracked_fragments=len(observer.tracks),
        reached_eof=True,output=str(args.output))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recordings-dir',type=Path,default=Path('/recordings'))
    parser.add_argument('--recording-id',default='rec-6b7064605b3d')
    parser.add_argument('--detector-profile',default='general')
    parser.add_argument('--cache',type=Path,default=Path('/models/experiments/awiros'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--commit',default='')
    parser.add_argument('--reference-review',type=Path,default=None)
    parser.add_argument('--reference-view',type=Path,default=None)
    args=parser.parse_args()
    if bool(args.reference_review)!=bool(args.reference_view):parser.error('Supply both reference provenance inputs or neither')
    main(args)
