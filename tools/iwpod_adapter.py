"""Pinned, experiment-only IWPOD-NET adapter; no model/geometry training.

The upstream MIT code stays in the ignored benchmark asset directory. We run
its preprocessing, affine decoding, NMS and rectification without rewriting
those algorithms. Only package import names and the Keras invocation wrapper
differ, to avoid a global ``src`` collision and unnecessary tf.data threads.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import sys
import types

import numpy as np

REVISION = '8cb667e54baef7ccb14bee33e293624627b2838a'
HASHES = {
    'src/keras_utils.py': '47b2dfc5ff839f8352e47a66dc79211da86780f3423c93afe763754ebd4ca5bc',
    'src/utils.py': '20b58eb24355309e9c6476c0446ecfe41800384fa17ab2a68f912c265110d1a2',
    'src/label.py': '262e05aa4b565509c9747d4f9673bb105ea9aecd1025c87db4e02efefca70b23',
    'src/projection_utils.py': '22c6b2c45957b0f5d14c1320edd5893d7b5787ff587a812da9ddb2dd6c0a36f6',
    'src/drawing_utils.py': 'c26245f0310934165cebc3b90679ab2c7d8105dd6a792bcc5decace63f1ec2db',
    'weights/iwpod_net.h5': '35c0664e9392881ae6b3bc653e571e8b5cb9e7a06cef0d8ab63286a999004d86',
    'weights/iwpod_net.json': '6f36c4ec071a06484d1da13cc26abbc0c23f9403102b5c25013e960e0cfc5947',
}
DOCUMENT_HASHES = {
    'LICENSE': '41cb863e67a4cf46579990d520d7ef28e3e8944a4fc4eba447f80b494df30368',
    'README.md': 'f37723bbe69b443aa035b2ab0cf4ba3570356a1f967702e7ad82ec56973361c9',
}


def _asset_limit(relative):
    return 16 * 1024 * 1024 if relative.endswith('.h5') else 1024 * 1024


def prepare_assets(source):
    """Explicit pinned downloads only; inference never calls this function.

    Existing verified assets are reused. A download must pass both a size bound
    and its fixed SHA256 before atomically replacing an unverified destination.
    Unique temporary files are removed on every failure, preserving the prior
    asset. The receipt and upstream MIT license accompany the local bundle.
    """
    import json
    import tempfile
    import urllib.request
    source = Path(source).resolve()
    source.mkdir(parents=True, exist_ok=True)
    records = []
    for relative, expected in {**HASHES, **DOCUMENT_HASHES}.items():
        path = source / relative
        if not path.resolve().is_relative_to(source):
            raise ValueError('IWPOD asset destination escapes source directory')
        url = f'https://raw.githubusercontent.com/claudiojung/iwpod-net/{REVISION}/{relative}'
        cached = (path.is_file() and path.stat().st_size <= _asset_limit(relative)
                  and hashlib.sha256(path.read_bytes()).hexdigest() == expected)
        if not cached:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                digest, size = hashlib.sha256(), 0
                request = urllib.request.Request(url, headers={'User-Agent': 'Sentinel-IWPOD-benchmark'})
                with urllib.request.urlopen(request, timeout=90) as response:
                    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.iwpod-',
                                                     suffix='.partial', delete=False) as stream:
                        temporary = Path(stream.name)
                        while block := response.read(1024 * 1024):
                            size += len(block)
                            if size > _asset_limit(relative):
                                raise ValueError('IWPOD asset exceeds bounded download size')
                            digest.update(block)
                            stream.write(block)
                if digest.hexdigest() != expected:
                    raise ValueError(f'IWPOD download checksum mismatch: {relative}')
                temporary.replace(path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        records.append(dict(path=relative, sha256=expected, bytes=path.stat().st_size, url=url))
    receipt = dict(repository='https://github.com/claudiojung/iwpod-net', revision=REVISION,
                   license='MIT', files=records)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=source,
                                         prefix='.iwpod-receipt-', suffix='.partial', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(receipt, stream, indent=2)
            stream.write('\n')
        temporary.replace(source / 'receipt.json')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return receipt


def checked_modules(source):
    """Load only the exact reviewed upstream files into a private namespace."""
    source = Path(source).resolve()
    for relative, expected in HASHES.items():
        path = source / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'IWPOD asset checksum mismatch: {relative}')
    namespace = '_sentinel_iwpod_' + hashlib.sha256(str(source).encode()).hexdigest()[:12]
    if namespace + '.keras_utils' in sys.modules:
        return sys.modules[namespace + '.keras_utils'], sys.modules[namespace + '.utils']
    package = types.ModuleType(namespace)
    package.__path__ = [str(source / 'src')]
    sys.modules[namespace] = package
    # All transformations are import-path substitutions; decoder math is intact.
    for name in ('drawing_utils', 'label', 'utils', 'projection_utils', 'keras_utils'):
        path = source / 'src' / (name + '.py')
        text = path.read_text(encoding='utf-8').replace('from src.', 'from ' + namespace + '.')
        module = types.ModuleType(namespace + '.' + name)
        module.__file__ = str(path)
        module.__package__ = namespace
        sys.modules[module.__name__] = module
        exec(compile(text, str(path), 'exec'), module.__dict__)
    return sys.modules[namespace + '.keras_utils'], sys.modules[namespace + '.utils']


def settings(shape, vehicle_type='fullimage'):
    """Official example defaults, including vehicle-dependent input sizing."""
    h, w = shape[:2]
    if min(h, w) < 1:
        raise ValueError('Nonempty image required')
    if vehicle_type in ('car', 'bus', 'truck'):
        return 256 * max(1., min(2.75, w / h)), (240, 80)
    if vehicle_type == 'fullimage':
        return 480, (240, 80)
    if vehicle_type == 'bike':
        return 208, (120, 80)
    raise ValueError('Known IWPOD vehicle type required')


def native_predictions(labels, rectified, image):
    """Map official normalized quads to unchanged native vehicle coordinates."""
    import cv2
    h, w = image.shape[:2]
    if len(labels) != len(rectified):
        raise ValueError('IWPOD detection/crop count mismatch')
    result = []
    for label, working in zip(labels, rectified):
        points = np.asarray(label.pts, dtype=np.float64)
        probability = float(label.prob())
        if points.shape != (2, 4) or not np.isfinite(points).all() or not math.isfinite(probability):
            continue
        quad = (points * np.array([[w], [h]])).T
        contour = quad.astype(np.float32)
        if not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) < 1:
            continue
        low, high = quad.min(axis=0), quad.max(axis=0)
        box = [max(0., float(low[0])), max(0., float(low[1])),
               min(float(w), float(high[0])), min(float(h), float(high[1]))]
        if box[0] >= box[2] or box[1] >= box[3]:
            continue
        rect = np.asarray(working)
        if rect.ndim != 3 or rect.shape[2] != 3 or not np.isfinite(rect).all():
            continue
        result.append(dict(box=box, quad=quad.tolist(), confidence=probability,
            boundary_clipped=bool((low < 0).any() or (high > [w, h]).any()),
            rectified=np.rint(np.clip(rect, 0., 1.) * 255).astype(np.uint8)))
    return result


class IWPODDetector:
    """CPU-only cached checkpoint, default official fullimage geometry preset.

    Set vehicle_type to the frozen tracker class where available, rather than
    guessing from OCR. ``fullimage`` is the documented roughly-framed fallback.
    ``predict`` returns local native coordinates and BGR uint8 working images.
    """
    def __init__(self, source: Path, *, threshold=.35, vehicle_type='fullimage'):
        if not 0 < threshold < 1:
            raise ValueError('Probability threshold must be in (0,1)')
        import tensorflow as tf
        if tf.config.list_logical_devices('GPU'):
            raise RuntimeError('IWPOD comparison adapter requires CPU-only TensorFlow visibility')
        self.upstream, self.utils = checked_modules(source)
        self.model = self.upstream.load_model(str(Path(source) / 'weights/iwpod_net'))
        self.threshold, self.vehicle_type = threshold, vehicle_type
        self.model_record = dict(repository='https://github.com/claudiojung/iwpod-net',
            revision=REVISION, sha256=HASHES['weights/iwpod_net.h5'], license='MIT',
            device='CPU', threshold=threshold, default_vehicle_type=vehicle_type,
            preprocessing='official BGR im2single; stride16 rounded resize; no RGB conversion',
            rectification='official affine decode + SVD homography, fixed 240x80 except bike120x80')

    def predict(self, image, vehicle_type=None):
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or not image.size:
            raise ValueError('Nonempty native BGR uint8 image required')
        maximum, size = settings(image.shape, vehicle_type or self.vehicle_type)
        model = self.model
        class DirectKeras:
            def predict(self, tensor):
                return model(tensor, training=False).numpy()
        labels, working, _ = self.upstream.detect_lp_width(
            DirectKeras(), self.utils.im2single(image), maximum, 16, size, self.threshold)
        return native_predictions(labels, working, image)


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare',))
    parser.add_argument('--source', type=Path, required=True,
                        help='Local cache for pinned official source, weights, notices and receipt')
    args = parser.parse_args()
    receipt = prepare_assets(args.source)
    print(json.dumps(dict(stage='prepared', revision=receipt['revision'], files=len(receipt['files']))))
