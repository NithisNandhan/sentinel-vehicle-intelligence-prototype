"""Isolated Indian-plate OCR experiment; never imported by the live worker.

Preparation is an explicit network operation. Inference is cached-only and
checks all checkpoint keys/shapes before accepting a model. Author source is
parsed for its literal configuration, not executed as an inference script.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

MODEL_REVISION = 'eadbc5ae4faa8284c8a547a60142fd555c3059ca'
PADDLEOCR_REVISION = 'fc82f229112a03aff46cc53db6166a98ea65a8fa'
WEIGHTS_SHA256 = 'f9f1264e0c115a239ca6d6700a74ceffab17168b75dafc6c169c232012c68e47'


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_assets(cache: Path):
    """Explicit bounded downloads of pinned model, notices and framework source."""
    import tarfile
    import urllib.request
    cache.mkdir(parents=True, exist_ok=True)
    assets = []
    def download(url, path, limit, expected=None):
        if not path.is_file() or (expected and file_hash(path) != expected):
            temporary = path.with_suffix(path.suffix + '.partial')
            try:
                with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as output:
                    size = 0
                    while chunk := response.read(1024 * 1024):
                        size += len(chunk)
                        if size > limit:
                            raise ValueError('Research asset exceeds download bound')
                        output.write(chunk)
                if expected and file_hash(temporary) != expected:
                    raise ValueError('Research asset does not match publisher checksum')
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        assets.append({'name': path.name, 'url': url, 'sha256': file_hash(path)})
        print(json.dumps({'cached_asset': path.name, 'bytes': path.stat().st_size}), flush=True)
    base = f'https://huggingface.co/Awiros/anpr-ocr/resolve/{MODEL_REVISION}'
    for filename in ('LICENSE', 'en_dict.txt', 'test.py', 'README.md', 'TechnicalReport.pdf', 'model.safetensors'):
        download(f'{base}/{filename}?download=true', cache / filename, 160 * 1024 * 1024,
                 WEIGHTS_SHA256 if filename == 'model.safetensors' else None)
    archive = cache / 'paddleocr-source.tar.gz'
    download(f'https://codeload.github.com/PaddlePaddle/PaddleOCR/tar.gz/{PADDLEOCR_REVISION}',
             archive, 250 * 1024 * 1024)
    source = cache / 'PaddleOCR'
    with tarfile.open(archive) as bundle:
        total = 0
        for member in bundle:
            parts = Path(member.name).parts[1:]
            if not parts or parts[0] not in {'ppocr', 'LICENSE'} or not member.isfile():
                continue
            destination = source.joinpath(*parts).resolve()
            if not destination.is_relative_to(source.resolve()):
                raise ValueError('Unsafe research source archive')
            total += member.size
            if total > 100 * 1024 * 1024:
                raise ValueError('Research source exceeds extraction bound')
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(member) as stream, destination.open('wb') as output:
                output.write(stream.read())
    manifest = {'model_revision': MODEL_REVISION, 'paddleocr_revision': PADDLEOCR_REVISION,
                'model_license': 'Apache-2.0', 'weights_publisher_sha256': WEIGHTS_SHA256,
                'assets': assets,
                'source_files': {str(p.relative_to(source)): file_hash(p)
                                 for p in source.rglob('*') if p.is_file()}}
    (cache / 'assets.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def author_config(script: str):
    """Read only literal assignments needed by the author's architecture config."""
    constants = {}
    for statement in ast.parse(script).body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            continue
        name = target.id
        if name not in {'CTC_NUM_CLASSES', 'NRTR_NUM_CLASSES', 'MODEL_CONFIG'}:
            continue
        class Resolve(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id not in constants:
                    raise ValueError('Nonliteral author model configuration')
                return ast.copy_location(ast.Constant(constants[node.id]), node)
        constants[name] = ast.literal_eval(Resolve().visit(statement.value))
    if constants.get('CTC_NUM_CLASSES') != 64 or constants.get('NRTR_NUM_CLASSES') != 67:
        raise ValueError('Unexpected author model capacity')
    return constants['MODEL_CONFIG']['Architecture']


def check_state_shapes(expected, actual):
    if set(expected) != set(actual):
        raise ValueError('OCR checkpoint keys do not match the pinned architecture')
    if any(tuple(expected[key].shape) != tuple(actual[key].shape) for key in expected):
        raise ValueError('OCR checkpoint tensor shapes do not match the pinned architecture')


class AwirosExperiment:
    def __init__(self, cache: Path, *, device='cpu'):
        import sys
        if device not in {'cpu', 'gpu'}:
            raise ValueError('Explicit CPU/GPU device required')
        manifest = json.loads((cache / 'assets.json').read_text(encoding='utf-8'))
        if manifest['model_revision'] != MODEL_REVISION or manifest['paddleocr_revision'] != PADDLEOCR_REVISION:
            raise ValueError('Unpinned OCR experiment assets')
        for asset in manifest['assets']:
            path = cache / asset['name']
            if path.parent != cache or file_hash(path) != asset['sha256']:
                raise ValueError('Modified OCR experiment asset')
        if file_hash(cache / 'model.safetensors') != WEIGHTS_SHA256:
            raise ValueError('OCR checkpoint publisher checksum mismatch')
        source = cache / 'PaddleOCR'
        for filename, expected in manifest['source_files'].items():
            path = (source / filename).resolve()
            if not path.is_relative_to(source.resolve()) or file_hash(path) != expected:
                raise ValueError('Modified PaddleOCR experiment source')
        # Dependencies are installed separately, not into production site-packages.
        sys.path.insert(0, str(source))
        if (cache / 'deps').is_dir():
            sys.path.insert(0, str(cache / 'deps'))
        import paddle
        from safetensors.numpy import load_file
        from ppocr.modeling.architectures import build_model
        self.paddle = paddle
        if device == 'gpu' and not paddle.is_compiled_with_cuda():
            raise RuntimeError('Paddle GPU build is unavailable; no silent device fallback')
        paddle.set_device(device)
        config = author_config((cache / 'test.py').read_text(encoding='utf-8'))
        self.model = build_model(config)
        arrays = load_file(str(cache / 'model.safetensors'))
        check_state_shapes(self.model.state_dict(), arrays)
        missing, unexpected = self.model.set_state_dict({k: paddle.to_tensor(v) for k, v in arrays.items()})
        if missing or unexpected:
            raise ValueError('OCR checkpoint did not load strictly')
        self.model.eval()
        dictionary = (cache / 'en_dict.txt').read_text(encoding='utf-8').splitlines()
        self.tokens = ('', *dictionary, ' ')
        if len(self.tokens) != 64 or len(set(self.tokens)) != 64:
            raise ValueError('Unexpected OCR character dictionary')
        self.device = device

    def probabilities(self, image):
        import cv2
        import numpy as np
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or not image.size:
            raise ValueError('A nonempty native BGR uint8 crop is required')
        height, width = image.shape[:2]
        target_width = max(1, min(int(width * 48 / height), 320))
        padded = np.zeros((48, 320, 3), dtype=np.uint8)
        padded[:, :target_width] = cv2.resize(image, (target_width, 48))
        tensor = ((padded.astype(np.float32) / 255.0 - .5) / .5).transpose(2, 0, 1)[None]
        with self.paddle.no_grad():
            output = self.model(self.paddle.to_tensor(tensor))
        if isinstance(output, dict):
            if 'ctc' not in output:
                raise ValueError('No CTC output from OCR model')
            output = output['ctc']
        elif isinstance(output, (tuple, list)):
            output = output[0]
        rows = output.numpy()[0]
        if (rows.ndim != 2 or rows.shape[1] != len(self.tokens) or not np.isfinite(rows).all()
                or (rows < 0).any() or not np.allclose(rows.sum(axis=1), 1, atol=1e-4)):
            raise ValueError('Unexpected OCR posterior distributions')
        return rows.tolist()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Explicit preparation of isolated Indian OCR assets')
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    prepare_assets(args.cache)
