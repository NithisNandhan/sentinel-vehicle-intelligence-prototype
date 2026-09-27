"""Paired fixed-threshold detection evaluation against publisher box labels.

This does not measure OCR, independent camera generalisation, or live success.
No checkpoint selection or thresholds use the reserved test partition.
"""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .detector_comparison import PlateModel, VehicleModel, iou, letterbox, transformed


def match_counts(predictions,targets,threshold):
    """Maximum-cardinality one-to-one IoU matching, not multiple credits."""
    neighbors = [sorted((j for j,t in enumerate(targets) if iou(p,t)>=threshold),
                        key=lambda j:-iou(p,targets[j])) for p in predictions]
    assigned = {}
    def augment(index,seen):
        for target in neighbors[index]:
            if target in seen:
                continue
            seen.add(target)
            if target not in assigned or augment(assigned[target],seen):
                assigned[target]=index
                return True
        return False
    matches=sum(augment(i,set()) for i in range(len(predictions)))
    return {'matched':matches,'unmatched_predictions':len(predictions)-matches,
            'missed_annotations':len(targets)-matches,'annotations':len(targets)}


class FinePlate:
    def __init__(self,path):
        import torch
        sys.path.insert(0,'/models/experiments/yolov9-training')
        from models.experimental import attempt_load
        from utils.general import non_max_suppression
        self.nms=non_max_suppression; self.torch=torch
        self.model=attempt_load(str(path),device=torch.device('cuda:0'),fuse=True).float().eval()
    def predict(self,image):
        tensor,scale,left,top=letterbox(image,512,512)
        with self.torch.inference_mode():
            output=self.model(self.torch.from_numpy(tensor).cuda())
            values=self.nms(output,.25,.45,max_det=300)[0][:,:4].cpu().tolist()
        return [[max(0,(b[0]-left)/scale),max(0,(b[1]-top)/scale),
                 min(image.shape[1],(b[2]-left)/scale),min(image.shape[0],(b[3]-top)/scale)] for b in values]


class FineVehicle:
    def __init__(self,path):
        from ultralytics import YOLO
        self.model=YOLO(str(path))
    def predict(self,image):
        return self.model.predict(image,imgsz=640,device=0,conf=.25,iou=.45,
                                  verbose=False,save=False)[0].boxes.xyxy.cpu().tolist()


def aggregate(rows):
    result={'image_cases':len(rows),'median_ms':round(statistics.median(r['elapsed_ms'] for r in rows),2)}
    for threshold in ('50','75'):
        counts={key:sum(r['iou'+threshold][key] for r in rows) for key in rows[0]['iou'+threshold]}
        matched=counts['matched']; returned=matched+counts['unmatched_predictions']
        counts['annotation_precision']=round(matched/returned,4) if returned else 0
        counts['annotation_recall']=round(matched/counts['annotations'],4) if counts['annotations'] else 0
        result['iou'+threshold]=counts
    return result


def evaluate(kind,root,checkpoint,report):
    import torch
    torch.set_num_threads(2); cv2.setNumThreads(2)
    manifest=json.loads((root/'manifest.json').read_text())
    receipt=json.loads((checkpoint.parent.parent/'training-receipt.json').read_text())
    if (manifest['kind']!=kind or receipt['kind']!=kind
            or receipt['training_manifest_sha256']!=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
            or receipt['best_sha256']!=hashlib.sha256(checkpoint.read_bytes()).hexdigest()):
        raise ValueError('Candidate receipt and frozen training manifest required')
    samples=[s for s in manifest['samples'] if s['split']=='test']
    if not samples or len(samples)>300:
        raise ValueError('Bounded prepared test partition required')
    cache=Path('/models/experiments/detectors')
    baseline_classes=[1,2,3,5,7] if manifest.get('source')=='iisc-aim/BMD-45' else [2,3,5,7]
    baseline=PlateModel('yolov9t512',cache,'cuda') if kind=='plate' else VehicleModel('yolo26s',cache,'cuda',640,baseline_classes)
    candidate=FinePlate(checkpoint) if kind=='plate' else FineVehicle(checkpoint)
    result={'kind':kind,'sample_images':len(samples),'confidence_threshold':.25,'nms_iou':.45,
        'manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        'candidate_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'baseline_coco_classes':baseline_classes if kind=='vehicle' else None,
        'candidate_categories':manifest.get('category_names'),
        'taxonomy_caveat':'BMD evaluation includes bicycles in baseline; candidate has 13 publisher subclasses; metrics pool vehicle boxes and do not prove subclass accuracy. COCO lacks a dedicated three-wheeler class.' if manifest.get('source')=='iisc-aim/BMD-45' else None,
        'independent_camera_holdout':False,'ocr_accuracy':None,'live_validation':False,
        'caveat':manifest['caveat']+'; fixed-threshold box metrics against publisher annotations; unmatched boxes may include annotation omissions; stress variants are not new vehicles',
        'models':[]}
    for name,model in (('parent',baseline),('fine_tuned',candidate)):
        first=cv2.imread(str(root/samples[0]['image_path']))
        for _ in range(3):model.predict(first)
        rows=[]
        for condition in ('native','resolution_quarter','motion9'):
            for sample in samples:
                path=root/sample['image_path']
                if hashlib.sha256(path.read_bytes()).hexdigest()!=sample['original_sha256']:
                    raise ValueError('Reserved test image changed')
                image=cv2.imread(str(path)); height,width=image.shape[:2]
                targets=[]
                for _,x,y,w,h in sample['boxes']:
                    box=[(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height]
                    _,box=transformed(image,condition,box); targets.append(box)
                changed,_=transformed(image,condition,[0,0,width,height])
                start=time.perf_counter(); predictions=model.predict(changed)
                elapsed=(time.perf_counter()-start)*1000
                rows.append({'id':sample['id'],'source':sample.get('source','BMD-45'),'condition':condition,
                    'elapsed_ms':round(elapsed,2),'iou50':match_counts(predictions,targets,.5),
                    'iou75':match_counts(predictions,targets,.75)})
        entry={'name':name,'summary':aggregate(rows),'conditions':{
            c:aggregate([r for r in rows if r['condition']==c]) for c in ('native','resolution_quarter','motion9')},'cases':rows}
        result['models'].append(entry); report.write_text(json.dumps(result,indent=2))
        print(json.dumps({k:v for k,v in entry.items() if k!='cases'}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--kind',choices=['plate','vehicle'],required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    evaluate(args.kind,args.root,args.checkpoint,args.report)
