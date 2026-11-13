"""Full permitted-recording EOF check through the real shared inference pipeline.

No frames, crop images, OCR strings, service keys or central events are exported.
This verifies inference completion, not dashboard controls or plate accuracy.
"""
import argparse
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'edge'))


def run(args):
    from sentinel_edge.recorded_source import RecordingLibrary,RecordedRun,new_session_id
    from sentinel_edge.live_detection_viewer import load_models,model_identity
    from sentinel_edge.tracked_live_viewer import observe_tracked
    if args.output.exists():raise ValueError('Fresh report path required')
    recordings,_=RecordingLibrary(args.recordings_dir).scan()
    recording=recordings.get(args.recording_id)
    if recording is None:raise ValueError('Permission-receipted recording required')
    config=SimpleNamespace(model_cache_dir=args.cache.parents[1],ocr_min_confidence=.85,hmac_key=None)
    began=time.monotonic()
    reader,detector,vehicle=load_models(config,args.cache,args.cache/'adaptations/indian-corpus-encoder.safetensors',
                                       detector_profile=args.profile,device=args.device)
    identity=model_identity(reader,args.profile,detector=detector,vehicle_model=vehicle)
    if any(identity[k]!=args.device for k in ('ocr_device','plate_device','vehicle_device')):
        raise RuntimeError('Explicit inference device mismatch')
    loaded=time.monotonic();stop=threading.Event();paused=threading.Event()
    original=RecordedRun(recording,new_session_id(),paced=False,paused=paused)
    decoded=0;last_pts=None;discontinuities=0;backward_pts=0;duplicate_pts=0
    def frames(token):
        nonlocal decoded,last_pts,discontinuities,backward_pts,duplicate_pts
        for protocol,frame in original.frames(token):
            # The production source explicitly marks clock resets. Preserve that
            # behaviour here instead of rejecting a supported discontinuity.
            if last_pts is not None:
                backward_pts+=frame.pts_ms<last_pts
                duplicate_pts+=frame.pts_ms==last_pts
                if frame.pts_ms<last_pts and not frame.discontinuity:
                    raise ValueError('Unmarked source timestamp reset')
            discontinuities+=bool(frame.discontinuity)
            decoded+=1;last_pts=frame.pts_ms
            yield protocol,frame
    source=SimpleNamespace(recording=recording,session_id=original.session_id,paused=paused,
                           paced=False,start_ms=0,frames=frames)
    kept={};published=0
    class SummarySink:
        def publish(self,snapshot,capture_stop=None):
            nonlocal published
            published+=1
            for key in ('analysed_frames','plate_crops','unique_model_accepted_unverified','video_position_ms'):
                kept[key]=snapshot.get(key)
            if published%10==0:
                print(json.dumps(dict(stage='processing',decoded_frames=decoded,published_samples=published,
                                      video_position_ms=kept['video_position_ms'])),flush=True)
    timer=threading.Timer(args.timeout_seconds,stop.set);timer.daemon=True;timer.start()
    try:
        summary=observe_tracked(config,recording,reader,detector,vehicle,SummarySink(),stop,
                                detector_profile=args.profile,recorded=source)
        if stop.is_set():raise TimeoutError('Completion deadline reached; EOF not established')
    finally:
        timer.cancel()
    if not decoded or not published:raise RuntimeError('No recorded inference completed')
    report=dict(schema_version=1,check='full_recorded_inference_eof_not_accuracy',completed_eof=True,
                recording_sha256=recording.sha256,recording_duration_ms=recording.duration_ms,
                decoded_frames=decoded,last_decoded_pts_ms=last_pts,published_samples=published,
                source_discontinuities=discontinuities,backward_pts=backward_pts,duplicate_pts=duplicate_pts,
                profile=args.profile,device=args.device,models=identity,
                load_seconds=round(loaded-began,3),inference_seconds=round(time.monotonic()-loaded,3),
                counts={k:summary[k] for k in ('analysed_frames','plate_crops','unique_model_accepted_unverified') if k in summary},
                last_sample_position_ms=kept.get('video_position_ms'),central_events_sent=0,
                notice='Unpaced permitted recording; models execute on declared device. Model accepts are unverified; no strings/images saved.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording-id',required=True);p.add_argument('--recordings-dir',type=Path,default=Path('/recordings'))
    p.add_argument('--device',choices=('cpu','gpu'),required=True)
    p.add_argument('--profile',choices=('general','general_quality'),default='general')
    p.add_argument('--cache',type=Path,default=Path('/models/experiments/awiros'))
    p.add_argument('--timeout-seconds',type=int,default=900);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not 30<=a.timeout_seconds<=3600:p.error('Use a bounded 30–3600 second inference deadline')
    try:
        run(a)
    except Exception as error:
        print(json.dumps(dict(stage='failed',exception=type(error).__name__,
                              failure_code=getattr(error,'failure_code',None))),flush=True)
        raise
