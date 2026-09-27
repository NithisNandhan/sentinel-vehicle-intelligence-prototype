"""Real-public-image CPU/CUDA parity and frozen diagnostic regression checks.

Never reads Sentinel frames, credentials, or changes live detector selection.
"""
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort
import torch

from sentinel_edge.detector_adaptation_eval import FinePlate
from sentinel_edge.detector_comparison import (
    CONDITIONS, decode_yolo11, letterbox, load_inputs, score_target,
    summarize, transformed,
)


def main():
    torch.set_num_threads(2); cv2.setNumThreads(2)
    # Compare FP32 export arithmetic, not Ampere's optional TF32 rounding.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cache = Path('/models/experiments/detectors')
    export = cache/'yolov9t512-indian-adapted.onnx'
    receipt = json.loads(export.with_suffix('.receipt.json').read_text())
    if hashlib.sha256(export.read_bytes()).hexdigest() != receipt['onnx_sha256']:
        raise ValueError('Export changed after receipt')
    checkpoint = Path('/training/runs/plate-r3/weights/best.pt')
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != receipt['checkpoint_sha256']:
        raise ValueError('Checkpoint changed after export')
    source = FinePlate(checkpoint)
    source.model.model[-1].export = True
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
    cpu = ort.InferenceSession(str(export), options, providers=['CPUExecutionProvider'])
    gpu = ort.InferenceSession(str(export), options,
                              providers=[('CUDAExecutionProvider',{'use_tf32':'0'}),'CPUExecutionProvider'])
    if 'CUDAExecutionProvider' not in gpu.get_providers():
        raise RuntimeError('Export GPU test fell back to CPU')
    root = Path('/workspace/data/benchmarks')
    _, images = load_inputs(root/'detector-comparison/manifest.json', root)
    errors = []
    # Same native pixels and stress transformations; no labels select export.
    for sample, image in images[:8]:
        for condition in ('native','resolution_quarter','motion9'):
            changed,_ = transformed(image, condition, sample['plate_box'])
            tensor,_,_,_ = letterbox(changed,512,512)
            with torch.inference_mode():
                expected = source.model(torch.from_numpy(tensor).cuda()).cpu().numpy()
            outputs = [('cpu',cpu),('cuda',gpu)]
            for device, session in outputs:
                actual = session.run(None, {session.get_inputs()[0].name:tensor})[0]
                passed = actual.shape == expected.shape and np.allclose(actual,expected,rtol=.002,atol=.05)
                errors.append({'id':sample['id'],'condition':condition,'device':device,
                    'passed':bool(passed),'max_absolute_difference':float(np.max(np.abs(actual-expected)))})
    (root/'reports/plate-export-parity.json').write_text(json.dumps(errors,indent=2))
    if not all(e['passed'] for e in errors):
        print(json.dumps({'failed_cases':[e for e in errors if not e['passed']]}),flush=True)
        raise ValueError('Real-image export parity failed; preserve evidence and do not promote')
    rows=[]
    first=letterbox(images[0][1],512,512)[0]
    for _ in range(3):gpu.run(None,{gpu.get_inputs()[0].name:first})
    for condition in CONDITIONS:
        for sample, image in images:
            changed,box = transformed(image,condition,sample['plate_box'])
            started=time.perf_counter()
            tensor,scale,left,top=letterbox(changed,512,512)
            output=gpu.run(None,{gpu.get_inputs()[0].name:tensor})[0]
            predictions=decode_yolo11(output,scale,left,top,changed.shape)
            rows.append({'id':sample['id'],'condition':condition,
                'elapsed_ms':round((time.perf_counter()-started)*1000,2),**score_target(predictions,box)})
    result={'onnx_sha256':receipt['onnx_sha256'],'real_image_parity_checks':len(errors),
        'real_image_parity_passed':True,'parity_rtol':.002,'parity_atol':.05,
        'providers':gpu.get_providers(),'summary':summarize(rows),
        'conditions':{c:summarize([r for r in rows if r['condition']==c]) for c in CONDITIONS},
        'parity_cases':errors,'cases':rows,'production_promoted':False,
        'caveat':'13 images, 11 distinct targets, approximate AI-reviewed boxes; diagnostic target coverage, not OCR or independent accuracy; provider audit does not guarantee all operators use GPU'}
    (root/'reports/plate-export-regression.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','parity_cases')}),flush=True)


if __name__=='__main__':main()
