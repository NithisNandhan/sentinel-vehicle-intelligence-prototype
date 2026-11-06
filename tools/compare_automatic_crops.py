"""Frozen manual-vs-automatic geometry diagnostic; never loads or scores OCR."""
import argparse
import ast
import base64
from collections import defaultdict
import hashlib
import html
import json
from pathlib import Path
import statistics
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manual_corners(path):
    # Annotation is loaded for scoring only, never passed to the cropper.
    found = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('CORNERS','TEXT_REGIONS') for t in node.targets):
            found.update(ast.literal_eval(node.value))
    return found


def geometry(box, polygon):
    import cv2
    import numpy as np
    x,y,x2,y2 = box
    automatic = np.array([(x,y),(x2,y),(x2,y2),(x,y2)],np.float32)
    target = np.array(polygon,np.float32)
    intersection,_ = cv2.intersectConvexConvex(automatic,target)
    area,expected = (x2-x)*(y2-y),cv2.contourArea(target)
    return {'iou':float(intersection/(area+expected-intersection)) if area+expected>intersection else 0.,
            'target_coverage':float(intersection/expected) if expected else 0.,
            'target_fraction_in_crop':float(intersection/area) if area else 0.}


def png(image):
    import cv2
    ok,data = cv2.imencode('.png',image)
    if not ok:
        raise ValueError('PNG encode failed')
    return 'data:image/png;base64,'+base64.b64encode(data).decode('ascii')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review',type=Path,required=True)
    parser.add_argument('--recording',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('gpu','cpu'),default='gpu')
    parser.add_argument('--threshold',type=float,default=.25)
    parser.add_argument('--tile-sizes',type=int,nargs='+',default=[1280,768])
    args = parser.parse_args()
    if not .1 <= args.threshold <= .5:
        raise ValueError('Bounded proposal threshold required')
    if args.output.exists():
        raise FileExistsError('Choose a fresh report folder')
    import av
    import cv2
    import numpy as np
    import torch
    from sentinel_edge.automatic_crops import NativePlateCropper, native_crop, rectify_boundary
    from sentinel_edge.experimental_detectors import HASHES, PlateDetector, VehicleTracker
    from sentinel_edge.plate_search import TargetedPlateSearch
    manifest_path = args.review/'manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    if sha(args.recording) != manifest['source_sha256']:
        raise ValueError('Recording differs from the manually reviewed source')
    receipt = json.loads(args.recording.with_suffix(args.recording.suffix+'.receipt.json').read_bytes())
    if receipt.get('sha256') != manifest['source_sha256'] or not receipt.get('permission'):
        raise ValueError('Permitted recording receipt required')
    if not 1 <= len(manifest['samples']) <= 100:
        raise ValueError('Short review only')
    # CUDA libraries initialise before ONNX. No OCR model is constructed.
    vehicle = VehicleTracker('yolo26s',Path('/models'),Path('/training'),device=args.device)
    detector = PlateDetector('parent',Path('/models'),device=args.device)
    search = TargetedPlateSearch(detector)
    cropper = NativePlateCropper(detector,sizes=tuple(args.tile_sizes))
    corners = manual_corners(args.review/'ocr-model-dev/rectify.py')
    corrected = json.loads((args.review/'ocr-model-dev/rectified-extended-manifest.json').read_bytes())
    corrected = {r['id']:r for r in corrected['samples']}
    wanted = defaultdict(list)
    for row in manifest['samples']:
        if sha(args.review/row['crop']) != row['crop_sha256']:
            raise ValueError('Manual reference changed')
        wanted[row['frame_index']].append(row)
    args.output.mkdir(parents=True)
    rows,visuals,times = [],[],[]
    frame_stats = []
    with av.open(str(args.recording)) as source:
        stream = source.streams.video[0]
        for index,frame in enumerate(source.decode(stream)):
            if index not in wanted:
                continue
            image = frame.to_ndarray(format='bgr24')
            if list((image.shape[1],image.shape[0])) != manifest['source_resolution']:
                raise ValueError('Unexpected source resolution')
            started = time.perf_counter()
            detector.model.model.conf_thresh = .25
            baseline = []
            tracks = vehicle.track(image)
            for track in tracks:
                x,y,x2,y2 = (max(0,track.box[0]),max(0,track.box[1]),min(image.shape[1],track.box[2]),min(image.shape[0],track.box[3]))
                if x2-x < 64 or y2-y < 32:
                    continue
                box = search.box(image[y:y2,x:x2])
                if box:
                    baseline.append((box[0]+x,box[1]+y,box[2]+x,box[3]+y))
            baseline_ms = (time.perf_counter()-started)*1000
            started = time.perf_counter()
            detector.model.model.conf_thresh = args.threshold
            proposals = cropper.proposals(image)
            tile_calls = cropper.calls
            raw = [native_crop(image,p.box)[1] for p in proposals]
            refined = [cropper.refine(image,p) for p in proposals]
            final = [native_crop(image,p.box)[1] for p in refined]
            elapsed = (time.perf_counter()-started)*1000
            times.append(elapsed)
            frame_stats.append({'frame_index':index,'baseline_ms':round(baseline_ms,1),
                                'tile_and_refine_ms':round(elapsed,1),'tile_calls':tile_calls,
                                'total_plate_calls':cropper.calls,'budget_exhausted':cropper.budget_exhausted,
                                'candidate_count':len(raw),'vehicle_boxes':len(tracks)})
            # Everything above runs without labels. Labels start here solely to
            # associate each reference with the nearest already-produced crop.
            for sample in wanted[index]:
                identifier = sample['id']
                x,y,x2,y2 = sample['box']
                polygon = [[a+x,b+y] for a,b in corners[identifier]] if identifier in corners else [[x,y],[x2,y],[x2,y2],[x,y2]]
                row = {'id':identifier,'position_ms':sample['position_ms'],
                       'reference':'manual_polygon' if identifier in corners else 'manual_context_rectangle',
                       'methods':{}}
                manual = cv2.imread(str(args.review/sample['crop']))
                reference = manual
                if identifier in corrected:
                    spec = corrected[identifier]
                    file = args.review/'ocr-model-dev'/spec['crop_path']
                    if sha(file) != spec['crop_sha256']:
                        raise ValueError('Corrected reference changed')
                    reference = cv2.imread(str(file))
                pictures = {'manual':png(reference)}
                for name,boxes in (('baseline',baseline),('tiled',raw),('refined',final)):
                    scored = [(geometry(box,polygon),box) for box in boxes]
                    metrics,box = max(scored,key=lambda pair:pair[0]['iou'],default=({'iou':0.,'target_coverage':0.,'target_fraction_in_crop':0.},None))
                    hit = metrics['iou'] >= .5
                    row['methods'][name] = {**metrics,'box':box,'hit_iou50':hit}
                    if box and metrics['iou'] >= .15:
                        a,b,c,d = box
                        pictures[name] = png(image[b:d,a:c])
                        if name == 'refined':
                            rect,quad = rectify_boundary(image,box)
                            row['automatic_quadrilateral'] = quad
                            if rect is not None:
                                pictures['rectified'] = png(rect)
                    else:
                        pictures[name] = None
                rows.append(row)
                visuals.append((row,pictures))
            print(json.dumps({'frame_index':index,'reviewed':len(rows),'candidates':len(raw),'calls':cropper.calls,'ms':round(elapsed)}),flush=True)
            del wanted[index]
            if not wanted:
                break
    if wanted:
        raise ValueError('Recording ended before all annotated frame indices')
    summaries = {}
    for name in ('baseline','tiled','refined'):
        values = [r['methods'][name] for r in rows]
        hits = [v for v in values if v['hit_iou50']]
        summaries[name] = {'targets':len(values),'hit_iou50':len(hits),
                           'coverage_ge90_at_iou50':sum(v['target_coverage']>=.9 for v in hits),
                           'median_iou':statistics.median(v['iou'] for v in values),
                           'median_target_coverage':statistics.median(v['target_coverage'] for v in values)}
    report = {'scope':'Development crop geometry only; selected 21 targets, not exhaustive recall or OCR accuracy.',
              'ocr_loaded':False,'live_pipeline_changed':False,'device':args.device,
              'source_sha256':manifest['source_sha256'],'manifest_sha256':sha(manifest_path),
              'plate_model_sha256':HASHES['parent'],'vehicle_model_sha256':HASHES['yolo26s'],
              'tile_sizes':list(cropper.sizes),'overlap':.25,'max_calls':cropper.max_calls,
              'proposal_threshold':args.threshold,'baseline_threshold':.25,
              'mean_frame_ms':statistics.mean(times),'summary':summaries,'frames':frame_stats,'cases':rows}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    cards = []
    for row,pictures in visuals:
        panels = []
        for name,title in [('manual','Manual reference'),('baseline','Current vehicle-first crop'),('tiled','Direct tiled plate detection'),('refined','Context re-detection + margin'),('rectified','Optional automatic boundary warp')]:
            img = pictures.get(name)
            metrics = row['methods'].get(name)
            stat = (f"IoU {metrics['iou']:.2f} · plate coverage {metrics['target_coverage']:.0%} · plate fraction {metrics['target_fraction_in_crop']:.0%}" if metrics else '')
            panels.append(f'<figure><figcaption>{title}</figcaption><div class="image">'+(f'<img alt="{title} for target {row["id"]}" src="{img}">' if img else '<span>No matching crop / no safe boundary</span>')+f'</div><small>{stat}</small></figure>')
        cards.append(f'<article id="crop-{row["id"]}"><h2>Crop {row["id"]} · {row["position_ms"]/1000:.1f}s</h2><p>{html.escape(row["reference"])} · automatic proposals matched to references only after inference</p><div class="grid">'+''.join(panels)+'</div></article>')
    stats = ' · '.join(f'{name}: {value["hit_iou50"]}/{len(rows)}' for name,value in summaries.items())
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sentinel · Automatic crop review</title><style>
    *{box-sizing:border-box}body{background:#f6f5f0;color:#233b35;font:16px/1.5 system-ui;margin:0}main{max-width:1500px;margin:auto;padding:28px 20px}h1{margin:8px 0}h2{font-size:1.1rem}.bar{background:#e5f2ec;padding:16px;border:1px solid #a8c8bc;border-radius:10px}article{background:white;border:1px solid #d1dcd5;border-radius:10px;padding:16px;margin:20px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr));gap:12px}figure{margin:0;min-width:0}figcaption{font-weight:600;margin-bottom:8px}.image{min-height:150px;display:grid;place-items:center;background:#eef0eb;border-radius:5px;overflow:hidden;padding:6px}.image img{max-width:100%;max-height:240px;object-fit:contain;image-rendering:auto}small,p{color:#52665e}a{color:#086e64}nav{display:flex;flex-wrap:wrap;gap:12px}a:focus-visible{outline:3px solid #086e64;outline-offset:3px}
    </style><main><p>Sentinel · Image geometry investigation · No OCR</p><h1>Automatic crops beside your manual crops</h1><div class="bar">'''+stats+'''<br>Target coverage at IoU ≥ 0.50; 21 selected targets, not total scene recall.</div><p>Original native crops retain the source pixels. The optional boundary warp interpolates a working copy; it can decline when no clear plate boundary exists. No generated detail. No plate numbers are inferred here. These are still-frame development diagnostics; throughput and full-video operation need separate validation.</p><nav>'''+''.join(f'<a href="#crop-{r["id"]}">{r["id"]}</a>' for r in rows)+'''</nav>'''+''.join(cards)+'''</main></html>'''
    (args.output/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps({'summary':summaries,'mean_frame_ms':report['mean_frame_ms']}),flush=True)


if __name__ == '__main__':
    main()
