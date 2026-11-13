"""Warm cached OCR, vehicle and plate models on zero inputs, without media.

Run through run_ocr_research.py so the explicitly selected CPU/GPU Paddle
runtime is on the import path. No settings files, downloads, camera streams,
plate readings or central events are used by this model-only diagnostic.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'edge'))
OCR_SHA256='c2ba896d994b174071c95ace0f5ffc8a61cec3fc6eecce8f0bc5feb7dec4456d'


def check_models(device,profile='general',cache=Path('/models/experiments/awiros'),*,
                 environment=None,loader=None,identity_reader=None,text_loader=None):
    """Return names/hash/device metadata only; injected loaders support tests."""
    if device not in ('cpu','gpu'):
        raise ValueError('Explicit cpu or gpu device required')
    environment=os.environ if environment is None else environment
    if environment.get('SENTINEL_OCR_DEVICE',device)!=device:
        raise RuntimeError('Requested device conflicts with the selected OCR runtime environment')
    configured_profile=environment.get('RECORDED_DETECTOR_PROFILE')
    if configured_profile and configured_profile!=profile:
        raise RuntimeError('Requested detector profile conflicts with the configured recorded profile')
    if loader is None or identity_reader is None:
        from sentinel_edge.live_detection_viewer import load_models,model_identity
        loader=loader or load_models;identity_reader=identity_reader or model_identity
    import numpy as np
    cache=Path(cache)
    # load_models only needs this model path. Avoid unrelated service settings.
    config=SimpleNamespace(model_cache_dir=cache.parents[1])
    reader,detector,vehicle=loader(config,cache,cache/'adaptations/indian-corpus-encoder.safetensors',
                                  detector_profile=profile,device=device)
    vehicle.track(np.zeros((720,1280,3),dtype=np.uint8))
    detector.boxes(np.zeros((512,512,3),dtype=np.uint8))
    reader.probabilities(np.zeros((32,120,3),dtype=np.uint8))
    providers=tuple(getattr(detector,'providers',()))
    if device=='gpu' and 'CUDAExecutionProvider' not in providers:
        raise RuntimeError('GPU preflight plate session did not load CUDA')
    if device=='cpu' and (not providers or any(p!='CPUExecutionProvider' for p in providers)):
        raise RuntimeError('CPU preflight plate session is not CPU-only')
    text_status='disabled'
    if environment.get('SENTINEL_TEXT_DETECTOR_ASSETS','').strip():
        if environment.get('SENTINEL_TEXT_DETECTOR_DEVICE',device)!=device:
            raise RuntimeError('Configured text detector device conflicts with requested runtime')
        if text_loader is None:
            from sentinel_edge.tracked_live_viewer import load_text_detector
            text_loader=load_text_detector
        from concurrent.futures import ThreadPoolExecutor
        text=text_loader()
        if text is None or text.device!=device:
            raise RuntimeError('Configured text detector did not load on the requested device')
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(text.polygons,np.zeros((160,320,3),dtype=np.uint8)).result(timeout=60)
        vehicle.track(np.zeros((720,1280,3),dtype=np.uint8))
        reader.probabilities(np.zeros((32,120,3),dtype=np.uint8))
        text_status=device
    identity=identity_reader(reader,profile,detector=detector,vehicle_model=vehicle)
    if any(identity.get(field)!=device for field in ('ocr_device','vehicle_device','plate_device')):
        raise RuntimeError('Preflight did not execute all three stages on the requested device')
    if identity.get('ocr_checkpoint_sha256')!=OCR_SHA256:
        raise RuntimeError('Preflight loaded a different OCR checkpoint')
    return {'model_preflight':'passed','device':device,'models':identity,'text_detector':text_status}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',required=True,choices=('cpu','gpu'))
    p.add_argument('--detector-profile',default=os.environ.get('RECORDED_DETECTOR_PROFILE','general'))
    p.add_argument('--cache',type=Path,default=Path('/models/experiments/awiros'))
    a=p.parse_args(argv)
    print(json.dumps(check_models(a.device,a.detector_profile,a.cache)),flush=True)


if __name__=='__main__':main()
