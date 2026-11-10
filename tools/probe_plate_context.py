"""Targeted native-pixel crop diagnostic, separate from automatic evaluation.

The local JSON spec contains recording_id, source_sha256, position_ms and boxes.
Human-supplied boxes are diagnostic inputs, never automatic benchmark evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'edge'))
from sentinel_edge.ctc_fusion import CTCPlateReader
from sentinel_edge.plate_context import expanded_box


def main(args):
    from sentinel_edge.recorded_source import RecordingLibrary,read_recording
    from sentinel_edge.config import Config
    from sentinel_edge.live_detection_viewer import load_models,model_identity,crop_box
    from sentinel_edge.plate_search import TargetedPlateSearch
    from sentinel_edge.plate_proposals import ProposalPool
    from sentinel_edge.tracked_live_viewer import load_text_detector
    import cv2
    spec=json.loads(args.spec.read_bytes())
    sources,_=RecordingLibrary(args.recordings_dir).scan();source=sources[spec['recording_id']]
    if source.sha256!=spec['source_sha256']:raise ValueError('Diagnostic source mismatch')
    checkpoint=args.cache/'adaptations/indian-corpus-encoder.safetensors'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!='c2ba896d994b174071c95ace0f5ffc8a61cec3fc6eecce8f0bc5feb7dec4456d':
        raise ValueError('Diagnostic checkpoint changed')
    ocr,detector,vehicle=load_models(Config.from_env(live=False),args.cache,checkpoint,detector_profile='general',device='gpu')
    models=model_identity(ocr,'general',detector=detector,vehicle_model=vehicle);del vehicle
    search=ProposalPool(TargetedPlateSearch(detector),tile_detector=detector,text_detector=load_text_detector())
    if args.output.exists():raise ValueError('Fresh diagnostic folder required')
    args.output.mkdir(parents=True)
    for _,frame in read_recording(source,threading.Event(),paced=False):
        pts=frame.pts_ms-source.start_pts_ms
        if pts!=spec['position_ms']:continue
        boxes=spec['boxes'];union=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)]
        context=expanded_box(union,frame.image.shape,.20,24,128)
        region=crop_box(frame.image,context);search.begin_frame();detected=search.box(region)
        tests=[('fragment-'+str(i),b) for i,b in enumerate(boxes)]+[('manual-union',union)]
        if detected is not None:
            tests.append(('redetected',[round(detected[0]+context[0]),round(detected[1]+context[1]),round(detected[2]+context[0]),round(detected[3]+context[1])]))
        results=[]
        for name,box in tests:
            crop=crop_box(frame.image,box)
            if crop is None:raise ValueError('Empty diagnostic crop')
            if not cv2.imwrite(str(args.output/(name+'.png')),crop):raise ValueError('PNG save failed')
            reader=CTCPlateReader(ocr.tokens);reader.add(ocr.probabilities(crop),timestamp_ms=pts,source_id=name)
            candidates=reader.candidates()
            results.append(dict(name=name,box=box,top=candidates[0].plate if candidates else None,
                                candidates=[dict(plate=c.plate,score=c.score) for c in candidates[:5]],
                                accepted=reader.accepted(.85) is not None))
        report=dict(scope='Human-directed same-frame diagnostic; excluded from automatic evaluation',
                    spec_sha256=hashlib.sha256(args.spec.read_bytes()).hexdigest(),source_sha256=source.sha256,
                    position_ms=pts,independent_originals=1,models=models,results=results)
        (args.output/'diagnostic.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(dict(stage='targeted_diagnostic_complete',crops=len(results),independent_originals=1)),flush=True)
        return
    raise ValueError('Exact diagnostic PTS missing')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--recordings-dir',type=Path,default=Path('/recordings'))
    parser.add_argument('--cache',type=Path,default=Path('/models/experiments/awiros'))
    main(parser.parse_args())
