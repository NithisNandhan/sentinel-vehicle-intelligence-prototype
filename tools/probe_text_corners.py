"""Timestamp-targeted automatic vehicle text/corner/verifier diagnostic.

Schedule contains reference IDs/times only. No plate text or manual boxes are
read during capture. This is not a whole-video accuracy or tracking benchmark.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'edge'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def checked_schedule(document, source_hash):
    if document.get('source_sha256')!=source_hash:
        raise ValueError('Schedule source mismatch')
    rows=document.get('targets',[])
    if not rows or len(rows)>100 or len({r['id'] for r in rows})!=len(rows):
        raise ValueError('Unique bounded diagnostic targets required')
    if any(set(row)!={'id','position_ms'} or not isinstance(row['position_ms'],int) or row['position_ms']<0 for row in rows):
        raise ValueError('Schedule may contain IDs/timestamps only, not answers or boxes')
    return sorted({row['position_ms'] for row in rows})


def run(args):
    import cv2
    from sentinel_edge.awiros_experiment import AwirosExperiment
    from sentinel_edge.recorded_source import RecordingLibrary, read_recording
    from sentinel_edge.plate_context import expanded_box
    from sentinel_edge.text_boundaries import TextBoundaryDetector, warp_polygon
    from sentinel_edge.text_plate_geometry import text_plate_candidates
    from sentinel_edge.plate_region_verifier import PlateRegionVerifier
    from sentinel_edge.ctc_fusion import CTCPlateReader
    from sentinel_edge.low_contrast_search import contrast_view
    if args.output.exists() or not args.output.resolve().is_relative_to((ROOT/'data/benchmarks').resolve()):
        raise ValueError('Fresh local benchmark output required')
    tracking_path=args.tracking/'report.json'
    tracking=json.loads(tracking_path.read_bytes())
    timeline=json.loads((args.tracking/'timeline.json').read_bytes())['frames']
    schedule=json.loads(args.schedule.read_bytes())
    targets=set(checked_schedule(schedule,tracking['source_sha256']))
    library,_=RecordingLibrary(args.recordings_dir).scan()
    source=library.get(tracking['recording_id'])
    if source is None or source.sha256!=tracking['source_sha256']:
        raise ValueError('Permitted recording/tracking mismatch')
    reviewed=args.tracking/'vehicle-review.json'
    excluded=set()
    if reviewed.exists():
        review=json.loads(reviewed.read_bytes())
        if review['report_sha256']!=sha(tracking_path):
            raise ValueError('Vehicle review belongs to a different capture')
        excluded={name for name,value in review['labels'].items() if value['status'] in ('non_vehicle','mixed','uncertain')}
    checkpoint=args.cache/'adaptations/indian-corpus-encoder.safetensors'
    if sha(checkpoint)!='c2ba896d994b174071c95ace0f5ffc8a61cec3fc6eecce8f0bc5feb7dec4456d':
        raise ValueError('Frozen OCR checkpoint changed')
    args.output.mkdir(parents=True);(args.output/'assets').mkdir()
    write(args.output/'progress.json',{'stage':'loading_models'})
    verifier=PlateRegionVerifier(args.verifier)
    ocr=AwirosExperiment(args.cache,device=args.device,checkpoint=checkpoint)
    text=TextBoundaryDetector(args.text_assets,device=args.device)
    began=time.monotonic();frames=[];totals=Counter();decoded=0
    def save(relative,image):
        path=args.output/relative
        if not cv2.imwrite(str(path),image):raise ValueError('Diagnostic image save failed')
        return sha(path),hashlib.sha256(image.tobytes()).hexdigest()
    for _,frame in read_recording(source,threading.Event(),paced=False):
        index=decoded;decoded+=1;pts=frame.pts_ms-source.start_pts_ms
        if index>=len(timeline) or abs(timeline[index]['t']-pts)>1:
            raise ValueError('Source/frozen tracker frame mismatch')
        if pts not in targets:continue
        selected=dict(position_ms=pts,regions=[])
        for ri,track in enumerate(timeline[index]['q']):
            if track[0] in excluded:continue
            box=expanded_box(track[1:5],frame.image.shape)
            x,y,x2,y2=box
            region=frame.image[y:y2,x:x2]
            if not region.size or min(region.shape[:2])<24:continue
            key=f'{int(pts):06}-{ri:03}'
            relative=f'assets/{key}-context.png'
            file_hash,pixels=save(relative,region)
            polygons=text.polygons(region);totals['text_calls']+=1
            if not polygons and args.retry_contrast:
                polygons=text.polygons(contrast_view(region));totals['contrast_text_calls']+=1
            entry=dict(id=str(track[0]),box=box,path=relative,sha256=file_hash,native_source_id=pixels,
                polygons=polygons,proposals=[],qualified=bool(track[7]))
            totals['vehicle_regions']+=1;totals['raw_text_regions']+=len(polygons)
            candidates=text_plate_candidates(region,polygons,max_candidates=args.max_candidates)
            for ci,candidate in enumerate(candidates):
                a,b,c,d=candidate['box'];native=region[b:d,a:c]
                local=[(px-a,py-b) for px,py in candidate['quad']]
                rectified=warp_polygon(native,local)
                if rectified is None:rectified=native.copy()
                verdict=verifier.verify(rectified,context={'proposal_box':candidate['box'],
                    'vehicle_region_shape':list(region.shape),'geometry':candidate['geometry']})
                native_path=f'assets/{key}-{ci:02}-native.png'
                rectified_path=f'assets/{key}-{ci:02}-rectified.png'
                native_hash,native_id=save(native_path,native)
                rectified_hash,_=save(rectified_path,rectified)
                allowed=verdict['status']=='likely_plate'
                top=None
                if allowed:
                    reader=CTCPlateReader(ocr.tokens)
                    reader.add(ocr.probabilities(rectified),timestamp_ms=pts,
                               source_id=hashlib.sha256(rectified.tobytes()).hexdigest())
                    ranked=reader.candidates();top=ranked[0].plate if ranked else None
                    totals['ocr_calls']+=1
                totals[verdict['status']]+=1
                entry['proposals'].append(dict(**candidate,id=f'{key}-{ci:02}',verdict=verdict,
                    native_path=native_path,rectified_path=rectified_path,native_sha256=native_hash,
                    rectified_sha256=rectified_hash,native_source_id=native_id,ocr_ran=allowed,top=top))
            selected['regions'].append(entry)
            write(args.output/'progress.json',dict(stage='probing',position_ms=pts,regions=totals['vehicle_regions'],
                counts=dict(totals),elapsed_seconds=round(time.monotonic()-began,2)))
        frames.append(selected)
        print(json.dumps(dict(stage='frame_complete',position_ms=pts,frames=len(frames),counts=dict(totals))),flush=True)
        if {row['position_ms'] for row in frames}==targets:break
    if {row['position_ms'] for row in frames}!=targets:
        raise ValueError('Not all requested source timestamps were decoded')
    document=dict(schema_version=1,source_sha256=source.sha256,
        scope='Timestamp-selected automatic vehicle-ROI diagnostic, including tentative boxes; not whole-video accuracy',
        schedule_sha256=sha(args.schedule),tracking_sha256=sha(tracking_path),labels_used_during_inference=False,
        timestamp_selection='Existing human-review timestamps; no plate text or manual geometry in inference',
        decoded_frames=decoded,selected_frames=len(frames),elapsed_seconds=round(time.monotonic()-began,2),
        ocr_model_sha256=sha(checkpoint),verifier_metadata=json.loads(args.verifier.with_suffix('.json').read_bytes()),
        summary=dict(totals),frames=frames,central_events_sent=0,
        source_code_sha256={name:sha(ROOT/name) for name in ('tools/probe_text_corners.py',
            'edge/sentinel_edge/text_plate_geometry.py','edge/sentinel_edge/plate_region_verifier.py')})
    write(args.output/'audit.json',document)
    write(args.output/'progress.json',dict(stage='complete',selected_frames=len(frames),summary=dict(totals)))
    print(json.dumps(dict(stage='complete',frames=len(frames),counts=dict(totals))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schedule',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tracking',type=Path,default=Path('/workspace/data/benchmarks/vehicle-tracking-960-20261110'))
    parser.add_argument('--recordings-dir',type=Path,default=Path('/recordings'))
    parser.add_argument('--cache',type=Path,default=Path('/models/experiments/awiros'))
    parser.add_argument('--text-assets',type=Path,default=Path('/text-model'))
    parser.add_argument('--verifier',type=Path,default=ROOT/'edge/assets/plate-region-verifier/model.npz')
    parser.add_argument('--device',choices=('cpu','gpu'),default='cpu')
    parser.add_argument('--retry-contrast',action='store_true')
    parser.add_argument('--max-candidates',type=int,default=8)
    run(parser.parse_args())
