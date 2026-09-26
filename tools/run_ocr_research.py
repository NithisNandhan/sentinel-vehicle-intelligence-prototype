"""Linux-only launcher for isolated Paddle CUDA libraries, never production Torch."""
import os
import sys
from pathlib import Path

if __name__ == '__main__':
    cache=Path('/models/experiments/awiros')
    libraries=sorted((cache/'gpu-runtime/nvidia').glob('*/lib'))
    libraries += [cache/'gpu-deps/paddle/libs', Path('/usr/lib/wsl/lib')]
    environment=dict(os.environ)
    environment['LD_LIBRARY_PATH']=':'.join(str(p) for p in libraries if p.is_dir()) + ':' + environment.get('LD_LIBRARY_PATH','')
    environment['PYTHONPATH']=':'.join([str(cache/'gpu-deps'),str(cache/'deps'),'/workspace/edge'])
    environment['OMP_NUM_THREADS']='2'
    environment['OPENBLAS_NUM_THREADS']='2'
    # Require FP32 parity before training; TF32 defaults exceeded the test bound.
    environment['NVIDIA_TF32_OVERRIDE']='0'
    environment['FLAGS_allow_tf32_cublas']='False'
    environment['FLAGS_allow_tf32_cudnn']='False'
    os.execvpe(sys.executable,[sys.executable,*sys.argv[1:]],environment)
