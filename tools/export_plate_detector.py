"""Export the experimental selected plate model as plain safe ONNX data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,'/models/experiments/yolov9-training')
from models.experimental import attempt_load

torch.set_num_threads(2)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint',type=Path,default=Path('/training/runs/plate-r3/weights/best.pt'))
parser.add_argument('--target',type=Path,default=Path('/models/experiments/detectors/yolov9t512-indian-adapted.onnx'))
parser.add_argument('--report',type=Path,default=Path('/reports/plate-detector-export.json'))
parser.add_argument('--validation-root',type=Path,default=None,
                    help='Optional fixed validation export for real-image CPU/CUDA parity')
args=parser.parse_args()
checkpoint=args.checkpoint
receipt=json.loads((checkpoint.parent.parent/'training-receipt.json').read_text())
if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=receipt['best_sha256']:
    raise ValueError('Training checkpoint receipt mismatch')
model=attempt_load(str(checkpoint),device=torch.device('cpu'),fuse=True).float().eval()
model.model[-1].export=True
value=torch.zeros(1,3,512,512)
with torch.inference_mode():
    expected=model(value).numpy()
target=args.target
if target.exists():
    raise ValueError('Existing experimental export preserved')
torch.onnx.export(model,value,str(target),input_names=['images'],output_names=['predictions'],
                  opset_version=17,dynamo=False)
import onnxruntime as ort
session=ort.InferenceSession(str(target),providers=['CPUExecutionProvider'])
actual=session.run(None,{'images':value.numpy()})[0]
if actual.shape!=expected.shape or not np.allclose(actual,expected,rtol=.001,atol=.02):
    raise ValueError('Export parity failed; do not use checkpoint')
record={'architecture':'YOLOv9-tiny','input_shape':[1,3,512,512],
        'output_shape':list(actual.shape),'output_contract':'xywh plus one plate probability; external NMS required',
        'checkpoint_sha256':receipt['best_sha256'],'onnx_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'zero_input_cpu_parity':True,'production_promoted':False,
        'training_source_license':'GPL-3.0; model publisher training counterpart; dataset notices retained'}
if args.validation_root is not None:
    import cv2
    from sentinel_edge.detector_comparison import letterbox, transformed
    manifest=json.loads((args.validation_root/'manifest.json').read_bytes())
    if hashlib.sha256((args.validation_root/'manifest.json').read_bytes()).hexdigest()!=receipt['training_manifest_sha256']:
        raise ValueError('Parity validation belongs to a different data export')
    examples=[s for s in manifest['samples'] if s['split']=='val' and s.get('variant','original')=='original'][:8]
    if len(examples)!=8:
        raise ValueError('Eight original validation examples required for parity')
    gpu=ort.InferenceSession(str(target),providers=[('CUDAExecutionProvider',{'use_tf32':'0'}),'CPUExecutionProvider'])
    if 'CUDAExecutionProvider' not in gpu.get_providers():
        raise RuntimeError('Real export parity GPU unavailable')
    checks=[]
    for example in examples:
        path=args.validation_root/example['image_path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=example['original_sha256']:
            raise ValueError('Frozen parity image changed')
        image=cv2.imread(str(path))
        for condition in ('native','resolution_quarter','motion9'):
            changed,_=transformed(image,condition,[0,0,image.shape[1],image.shape[0]])
            tensor=letterbox(changed,512,512)[0]
            with torch.inference_mode():reference=model(torch.from_numpy(tensor)).numpy()
            for device,runtime in (('cpu',session),('cuda',gpu)):
                prediction=runtime.run(None,{'images':tensor})[0]
                passed=prediction.shape==reference.shape and np.allclose(prediction,reference,rtol=.002,atol=.05)
                checks.append(dict(id=example['id'],condition=condition,device=device,passed=bool(passed),
                                   max_absolute_difference=float(np.max(np.abs(prediction-reference)))))
    if not all(c['passed'] for c in checks):
        raise ValueError('Real-image export parity failed; export remains unverified')
    record.update(real_image_parity_passed=True,real_image_parity_checks=len(checks),
                  real_image_parity_rtol=.002,real_image_parity_atol=.05,real_image_parity_cases=checks)
(target.with_suffix('.receipt.json')).write_text(json.dumps(record,indent=2))
args.report.write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
