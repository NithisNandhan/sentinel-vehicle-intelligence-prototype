"""Linux-only launcher for the isolated Paddle OCR runtime, never production Torch.

SENTINEL_OCR_DEVICE selects the runtime explicitly: `gpu` (the default, as
before) adds the isolated CUDA libraries; `cpu` loads only the CPU Paddle
dependencies so the session runs on a machine with no NVIDIA driver.
"""
import os
import sys
from pathlib import Path


def research_environment(base, cache=Path('/models/experiments/awiros')):
    environment = dict(base)
    device = environment.get('SENTINEL_OCR_DEVICE', 'gpu')
    if device not in ('cpu', 'gpu'):
        raise SystemExit('SENTINEL_OCR_DEVICE must be cpu or gpu')
    threads = environment.get('SENTINEL_CPU_THREADS', '2' if device == 'gpu' else '4')
    if not threads.isdigit() or not 1 <= int(threads) <= 64:
        raise SystemExit('SENTINEL_CPU_THREADS must be between 1 and 64')
    if device == 'gpu':
        libraries = sorted((cache/'gpu-runtime/nvidia').glob('*/lib'))
        libraries += [cache/'gpu-deps/paddle/libs', Path('/usr/lib/wsl/lib')]
        environment['LD_LIBRARY_PATH'] = (':'.join(str(p) for p in libraries if p.is_dir())
                                          + ':' + environment.get('LD_LIBRARY_PATH', ''))
        environment['PYTHONPATH'] = ':'.join([str(cache/'gpu-deps'), str(cache/'deps'), '/workspace/edge'])
        # Require FP32 parity before training; TF32 defaults exceeded the test bound.
        environment['NVIDIA_TF32_OVERRIDE'] = '0'
        environment['FLAGS_allow_tf32_cublas'] = 'False'
        environment['FLAGS_allow_tf32_cudnn'] = 'False'
    else:
        # No CUDA search path and no GPU Paddle on the import path: removing
        # the Docker GPU flag alone would still import the GPU build first.
        environment['PYTHONPATH'] = ':'.join([str(cache/'deps'), '/workspace/edge'])
        environment['CUDA_VISIBLE_DEVICES'] = ''
    environment['OMP_NUM_THREADS'] = threads
    environment['OPENBLAS_NUM_THREADS'] = threads
    return environment


if __name__ == '__main__':
    os.execvpe(sys.executable, [sys.executable, *sys.argv[1:]], research_environment(os.environ))
