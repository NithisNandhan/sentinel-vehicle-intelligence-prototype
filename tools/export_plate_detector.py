"""Export the experimental selected plate model as plain safe ONNX data."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,'/models/experiments/yolov9-training')
from models.experimental import attempt_load

torch.set_num_threads(2)
checkpoint=Path('/training/runs/plate-r3/weights/best.pt')
receipt=json.loads((checkpoint.parent.parent/'training-receipt.json').read_text())
if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=receipt['best_sha256']:
    raise ValueError('Training checkpoint receipt mismatch')
model=attempt_load(str(checkpoint),device=torch.device('cpu'),fuse=True).float().eval()
model.model[-1].export=True
value=torch.zeros(1,3,512,512)
with torch.inference_mode():
    expected=model(value).numpy()
target=Path('/models/experiments/detectors/yolov9t512-indian-adapted.onnx')
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
(target.with_suffix('.receipt.json')).write_text(json.dumps(record,indent=2))
Path('/reports/plate-detector-export.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
