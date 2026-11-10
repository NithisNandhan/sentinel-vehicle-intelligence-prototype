"""Freeze a complete recorded-pipeline run for local blind human review.

This observer never supplies labels, alters models or delivers central events.
Every decoded source frame is counted; inference keeps the dashboard's normal
sampling and gates. Review evidence is retained only for this permitted file.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time

from sentinel_edge.awiros_experiment import file_hash
from sentinel_edge.recorded_source import RecordingLibrary,read_recording,new_session_id


def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')


def write_json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


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

    def identity(self,track_id):
        value=str(track_id)
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}',value):raise ValueError('Unsafe tracker identity')
        return f'track-s{self.segment}-{value}'

    def ensure(self,track_id,position):
        uid=self.identity(track_id)
        if uid not in self.tracks:
            self.tracks[uid]=dict(id=uid,tracker_id=str(track_id),segment=self.segment,
                first_ms=position,last_ms=position,vehicle_class=None,analysed_samples=0,
                plate_crop_count=0,vehicle_preview=None,samples=[])
            self.predictions[uid]=dict(first_ms=position,last_ms=position,top_guess=None,
                candidate_ms=None,candidates=[],accepted=[],observations=0)
            self.native[uid]=dict(vehicle=None,area=0,plates=[])
        return uid,self.tracks[uid],self.predictions[uid]

    def depth(self):return 0

    def accept(self,camera_id,track_id,candidate,pts_ms,*,recorded=None):
        position=pts_ms-self.recording.start_pts_ms
        _,_,prediction=self.ensure(track_id,position)
        if candidate.plate not in {r['plate'] for r in prediction['accepted']}:
            prediction['accepted'].append(dict(plate=candidate.plate,pts_ms=position))
        return True

    def on_passage(self,outcome):
        # The true acceptance callback above is the operational gate. A partial
        # final result is never silently upgraded into an accepted plate.
        pass

    def publish(self,snapshot,stop=None):
        pass

    def on_sample(self,frame,tracks,displays,pipeline):
        from sentinel_edge.live_detection_viewer import crop_box,clipped_box
        from sentinel_edge.plate_quality import crop_quality
        self.segment=pipeline.metrics['scene_resets']
        position=frame.pts_ms-self.recording.start_pts_ms
        self.last_position=position;self.samples+=1
        for track in tracks:
            uid,row,prediction=self.ensure(track.track_id,position)
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
            uid,row,prediction=self.ensure(display['track_id'],position)
            row['last_ms']=position;row['plate_crop_count']+=1
            crop=display['crop'];quality=crop_quality(crop)
            digest=hashlib.sha256(crop.tobytes()).hexdigest()
            retained=self.native[uid]['plates']
            if not any(item[1]['pixel_sha256']==digest for item in retained):
                sample=dict(position_ms=position,box=list(map(float,display['box'])),
                    width=crop.shape[1],height=crop.shape[0],quality=quality,pixel_sha256=digest)
                retained.append((crop.copy(),sample))
                retained.sort(key=lambda item:(-item[1]['quality'],item[1]['position_ms']))
                del retained[6:]
            history=pipeline.histories.get(display['track_id'])
            if history is not None and history.reader.observation_count:
                ranked=history.reader.candidates()
                prediction['observations']=history.reader.observation_count
                if ranked:
                    prediction['top_guess']=ranked[0].plate
                    prediction['candidate_ms']=history.reader.samples[-1][0]-self.recording.start_pts_ms
                    prediction['candidates']=[dict(plate=c.plate,score=c.score,
                        minimum_character_score=c.minimum_character_score,
                        format_adjusted=c.format_adjusted) for c in ranked[:20]]
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
            for number,(crop,sample) in enumerate(native['plates']):
                sample['path']=f'assets/{uid}-plate-{number}.png'
                sample['image_sha256']=save(sample['path'],crop);row['samples'].append(sample)


def main(args):
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
    observer.export_assets()
    settings=dict(options=options.manifest(),keep_unconfirmed_vehicles=keep,
        keep_unconfirmed_min_height=os.getenv('SENTINEL_KEEP_UNCONFIRMED_MIN_HEIGHT','0'),
        ocr_min_confidence=config.ocr_min_confidence,normal_dashboard_sampling=True,
        review_crops_per_track_max=6,review_track_count_cap=None,
        text_detector_configured=bool(os.getenv('SENTINEL_TEXT_DETECTOR_ASSETS','')))
    manifest=dict(schema_version=1,run_id=counted.session_id,source=source,models=identity,
        settings=settings,run=dict(started_at=started_at,completed_at=utc(),commit=args.commit,
            runner_sha256=file_hash(Path(__file__)),reached_eof=True,decoded_frames=counted.decoded,
            last_decoded_ms=counted.last_pts-recording.start_pts_ms,
            last_analysed_ms=observer.last_position,analysed_frames=observer.samples,
            wall_seconds=round(time.monotonic()-began,2),pipeline_summary=summary),
        scope='Complete permitted recorded clip, previously used in development; not an unseen CCTV test',
        predictions_hidden_during_labelling=True,
        tracks=sorted(observer.tracks.values(),key=lambda r:(r['first_ms'],r['id'])))
    write_json(args.output/'manifest.json',manifest)
    manifest_sha=file_hash(args.output/'manifest.json')
    write_json(args.output/'predictions.json',dict(schema_version=1,manifest_sha256=manifest_sha,
        source_sha256=recording.sha256,tracks=observer.predictions,
        caveat='Scores are uncalibrated. Top guesses and actual model acceptances are separate.'))
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
    main(parser.parse_args())
