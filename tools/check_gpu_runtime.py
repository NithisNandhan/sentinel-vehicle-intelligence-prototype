"""Warm the real shared models before replacing a working CPU service.

No media files, OCR strings, credentials or screenshots are read or exported.
Only zero-valued in-memory inputs are used for this runtime check.
"""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'edge'))


def main():
    import numpy as np
    from sentinel_edge.config import Config
    from sentinel_edge.live_detection_viewer import load_models, model_identity
    profile = os.environ.get('RECORDED_DETECTOR_PROFILE', 'general')
    cache = Path('/models/experiments/awiros')
    reader, detector, vehicle = load_models(Config.from_env(live=False), cache,
        cache / 'adaptations/indian-corpus-encoder.safetensors',
        detector_profile=profile, device='gpu')
    # load_models warms Paddle before importing Torch, then constructs ONNX.
    # Run all three, since advertised provider availability is not sufficient.
    vehicle.track(np.zeros((720, 1280, 3), dtype=np.uint8))
    detector.boxes(np.zeros((512, 512, 3), dtype=np.uint8))
    reader.probabilities(np.zeros((32, 120, 3), dtype=np.uint8))
    identity = model_identity(reader, profile, detector=detector, vehicle_model=vehicle)
    if any(identity[field] != 'gpu' for field in ('ocr_device', 'vehicle_device', 'plate_device')):
        raise RuntimeError('GPU preflight did not load all three stages on GPU')
    print(json.dumps({'gpu_preflight': 'passed', 'models': identity}), flush=True)


if __name__ == '__main__':
    main()
