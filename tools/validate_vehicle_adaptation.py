"""Cross-source regression check of the Indian vehicle candidate, not live proof."""
import hashlib
import json
from pathlib import Path
import time

import cv2
import torch

from sentinel_edge.detector_adaptation_eval import FineVehicle
from sentinel_edge.detector_comparison import (
    CONDITIONS, VehicleModel, load_inputs, score_target, summarize, transformed,
)


def main():
    torch.set_num_threads(2); cv2.setNumThreads(2)
    checkpoint=Path('/training/runs/vehicle-bmd/weights/best.pt')
    receipt=json.loads((checkpoint.parent.parent/'training-receipt.json').read_text())
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=receipt['best_sha256']:
        raise ValueError('Selected training checkpoint changed')
    root=Path('/workspace/data/benchmarks')
    _,images=load_inputs(root/'detector-comparison/manifest.json',root)
    models=[('parent',VehicleModel('yolo26s',Path('/models/experiments/detectors'),'cuda',640)),
            ('indian_adapted',FineVehicle(checkpoint))]
    result={'candidate_sha256':receipt['best_sha256'],'models':[],
        'live_validation':False,'production_promoted':False,'ocr_accuracy':None,
        'caveat':'13 development images, 11 targets with approximate AI-reviewed boxes; target coverage not exhaustive scene recall; candidate categories differ, boxes are pooled'}
    for name,model in models:
        for _ in range(3):model.predict(images[0][1])
        rows=[]
        for condition in CONDITIONS:
            for sample,image in images:
                changed,target=transformed(image,condition,sample['vehicle_box'])
                started=time.perf_counter(); predictions=model.predict(changed)
                rows.append({'id':sample['id'],'condition':condition,
                    'elapsed_ms':round((time.perf_counter()-started)*1000,2),**score_target(predictions,target)})
        entry={'name':name,'summary':summarize(rows),
               'conditions':{c:summarize([r for r in rows if r['condition']==c]) for c in CONDITIONS},'cases':rows}
        result['models'].append(entry)
        (root/'reports/vehicle-bmd-cross-source-regression.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({k:v for k,v in entry.items() if k!='cases'}),flush=True)


if __name__=='__main__':main()
