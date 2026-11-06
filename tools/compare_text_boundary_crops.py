"""Development text-region proposal diagnostic, with recognition disabled."""
import argparse
import json
from pathlib import Path
import statistics
import time

from compare_automatic_crops import geometry,manual_corners,png,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review',type=Path,required=True)
    parser.add_argument('--recording',type=Path,required=True)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('gpu','cpu'),default='gpu')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Existing report preserved')
    import av
    import cv2
    import numpy as np
    from sentinel_edge.automatic_crops import windows,overlap,clip
    from sentinel_edge.text_boundaries import TextBoundaryDetector,warp_polygon
    manifest=json.loads((args.review/'manifest.json').read_bytes())
    if sha(args.recording)!=manifest['source_sha256']:
        raise ValueError('Wrong source recording')
    receipt=json.loads(args.recording.with_suffix(args.recording.suffix+'.receipt.json').read_bytes())
    if receipt.get('sha256')!=manifest['source_sha256'] or not receipt.get('permission'):
        raise ValueError('Permitted recording required')
    corners=manual_corners(args.review/'ocr-model-dev/rectify.py')
    detector=TextBoundaryDetector(args.assets,device=args.device)
    wanted={}
    for s in manifest['samples']:
        if sha(args.review/s['crop'])!=s['crop_sha256']:
            raise ValueError('Reference image changed')
        wanted.setdefault(s['frame_index'],[]).append(s)
    args.output.mkdir(parents=True)
    rows,cards,times,frame_stats=[],[],[],[]
    with av.open(str(args.recording)) as source:
        for index,frame in enumerate(source.decode(video=0)):
            if index not in wanted:
                continue
            image=frame.to_ndarray(format='bgr24')
            proposals=[]
            started=time.perf_counter()
            for x,y,x2,y2 in windows(image.shape,1024):
                tile=image[y:y2,x:x2]
                for poly in detector.polygons(tile):
                    poly=np.array(poly)+np.array([x,y])
                    box=clip((*poly.min(axis=0),*poly.max(axis=0)),image.shape)
                    if box is not None:
                        proposals.append((box,poly.tolist()))
            retained=[]
            for box,poly in sorted(proposals,key=lambda p:-(p[0][2]-p[0][0])*(p[0][3]-p[0][1])):
                if not any(overlap(box,old[0])>=.45 for old in retained):
                    retained.append((box,poly))
            elapsed=(time.perf_counter()-started)*1000
            times.append(elapsed)
            frame_stats.append({'frame_index':index,'ms':elapsed,'text_proposals':len(retained)})
            # Match frozen references only after every automatic proposal exists.
            for sample in wanted.pop(index):
                identifier=sample['id']
                x,y,x2,y2=sample['box']
                target=[[a+x,b+y] for a,b in corners[identifier]] if identifier in corners else [[x,y],[x2,y],[x2,y2],[x,y2]]
                matches=[(geometry(box,target),box,poly) for box,poly in retained]
                score,box,poly=max(matches,key=lambda v:v[0]['iou'],default=({'iou':0.,'target_coverage':0.,'target_fraction_in_crop':0.},None,None))
                row={'id':identifier,'reference':'manual_polygon' if identifier in corners else 'manual_context_rectangle',**score,'box':box,'quad':poly,'hit_iou50':score['iou']>=.5}
                rows.append(row)
                manual=cv2.imread(str(args.review/sample['crop']))
                pics=[('Manual native reference',manual)]
                if box and score['iou']>=.15:
                    a,b,c,d=box
                    pics += [('Text region, native pixels',image[b:d,a:c]),('Automatic text-boundary warp',warp_polygon(image,poly))]
                panels=''.join('<figure><figcaption>'+label+'</figcaption><div>'+('<img alt="'+label+' '+identifier+'" src="'+png(im)+'">' if im is not None else 'No match')+'</div></figure>' for label,im in pics)
                cards.append(f'<article id="crop-{identifier}"><h2>Crop {identifier}</h2><p>IoU {score["iou"]:.2f} · target coverage {score["target_coverage"]:.0%} · target fraction {score["target_fraction_in_crop"]:.0%}</p><section>{panels}</section></article>')
            print(json.dumps({'frame_index':index,'reviewed':len(rows),'text_proposals':len(retained),'ms':round(elapsed)}),flush=True)
            (args.output/'partial.json').write_text(json.dumps({'status':'incomplete','cases':rows,'frames':frame_stats},indent=2)+'\n')
            if not wanted:
                break
    if wanted:
        raise ValueError('Source ended early')
    summary={'targets':len(rows),'hit_iou50':sum(r['hit_iou50'] for r in rows),
             'coverage_ge90_at_iou50':sum(r['hit_iou50'] and r['target_coverage']>=.9 for r in rows),
             'median_iou':statistics.median(r['iou'] for r in rows),'mean_frame_ms':statistics.mean(times)}
    report={'scope':'Previously examined development crops. Text proposals can include non-plate text; precision is unmeasured.',
            'ocr_loaded':False,'device':args.device,'model':'PP-OCRv5_mobile_det',
            'model_receipt_sha256':sha(args.assets/'receipt.json'),'source_sha256':manifest['source_sha256'],
            'manifest_sha256':sha(args.review/'manifest.json'),'summary':summary,'cases':rows,'frames':frame_stats}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sentinel · Text-boundary crop review</title><style>*{box-sizing:border-box}body{background:#f6f5f0;color:#233b35;font:16px/1.5 system-ui;margin:0}main{max-width:1280px;margin:auto;padding:24px}article{padding:16px;border:1px solid #cadbd2;border-radius:10px;background:white;margin:20px 0}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:14px}figure{margin:0;min-width:0}figure div{background:#eef0eb;min-height:130px;padding:10px;display:grid;place-items:center}img{max-width:100%;max-height:240px;object-fit:contain}a{color:#086e64}.bar{padding:16px;background:#e5f2ec;border-radius:10px}nav{display:flex;gap:12px;flex-wrap:wrap}</style><main><p>Sentinel · Crop geometry only · Recognition disabled</p><h1>Text-boundary fallback</h1><div class="bar">'''+f'{summary["hit_iou50"]}/{len(rows)} references matched at IoU ≥ 0.50 · {summary["mean_frame_ms"]/1000:.2f} seconds per tested frame'+'''</div><p>The detector locates text and its four corners. It does not read characters or verify that a text region is a number plate. Unrelated text remains a possible false proposal. Native crops preserve source pixels; straightening interpolates a separate working view. References are consulted only after inference for scoring. Results cover selected development crops, not whole-video plate recall.</p><p><a href="automatic-crops.html">Compare vehicle-first and tiled plate crops</a></p><nav>'''+''.join(f'<a href="#crop-{r["id"]}">{r["id"]}</a>' for r in rows)+'''</nav>'''+''.join(cards)+'''</main></html>'''
    (args.output/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
