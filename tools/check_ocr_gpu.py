"""CPU/GPU full OCR inference parity on three licensed crops, not accuracy."""
import json
import gc
from pathlib import Path
import cv2
import numpy as np
import paddle
from sentinel_edge.awiros_experiment import AwirosExperiment, file_hash
from sentinel_edge.ctc_fusion import greedy_text

cache=Path('/models/experiments/awiros')
root=Path('/models/experiments/corpus-stage')
rows=json.loads((root/'adaptation_manifest.json').read_text())['samples'][:3]
images=[cv2.imread(str(root/r['crop_path'])) for r in rows]
cpu=AwirosExperiment(cache,device='cpu')
reference=[cpu.probabilities(image) for image in images]
tokens=cpu.tokens
del cpu; gc.collect()
gpu=AwirosExperiment(cache,device='gpu')
observed=[gpu.probabilities(image) for image in images]
errors=[float(np.max(np.abs(np.array(a)-np.array(b)))) for a,b in zip(reference,observed)]
same=[greedy_text(a,tokens)==greedy_text(b,tokens) for a,b in zip(reference,observed)]
print(json.dumps({'parity_differences':errors,'greedy_agreement':same}),flush=True)
if not all(same) or max(errors)>1e-4:
    raise RuntimeError('CPU/GPU OCR parity failed')
result={'paddle_version':paddle.__version__,'cuda_build':paddle.is_compiled_with_cuda(),
        'gpu_name':paddle.device.cuda.get_device_name(),'full_model_crops':len(rows),
        'greedy_text_identical':all(same),'maximum_probability_difference':max(errors),
        'parity_tolerance':1e-4,'model_sha256':file_hash(cache/'model.safetensors'),
        'accuracy_measured':False,'production_modified':False}
path=Path('/workspace/data/benchmarks/reports/ocr-gpu-check.json')
path.write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
