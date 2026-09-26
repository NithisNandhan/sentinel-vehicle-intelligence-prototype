"""Explicit opt-in learned alignment/restoration. No automatic network on load."""
from __future__ import annotations

import hashlib
import json
import math
import threading
from pathlib import Path

from .reconstruction import AlignedSample

RAFT_URL = 'https://download.pytorch.org/models/raft_small_C_T_V2-01064c6d.pth'
NAF_URL = 'https://drive.usercontent.google.com/download?id=1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj&export=download&authuser=0&confirm=t'
NAF_REVISION = '2b4af71ebe098a92a75910c233a3965a3e93ede4'


def verified_asset(cache: Path, name: str) -> Path:
    expected = {
        'raft-small': (RAFT_URL, 'Raft_Small_Weights.C_T_V2',
                       'https://github.com/pytorch/vision/blob/main/LICENSE'),
        'nafnet-gopro32': (NAF_URL, NAF_REVISION,
                          f'https://github.com/megvii-research/NAFNet/blob/{NAF_REVISION}/LICENSE'),
    }
    if name not in expected:
        raise ValueError('Unknown hybrid asset')
    document = json.loads((cache / 'assets.json').read_text(encoding='utf-8'))
    asset = document.get(name, {})
    path = (cache / asset.get('file', '')).resolve()
    if not path.is_relative_to(cache.resolve()) or not path.is_file():
        raise ValueError('Missing or invalid hybrid model asset')
    if (asset.get('source_url'), asset.get('source_revision'), asset.get('license_url')) != expected[name]:
        raise ValueError('Unexpected model provenance')
    if hashlib.sha256(path.read_bytes()).hexdigest() != asset.get('sha256'):
        raise ValueError('Hybrid model checksum mismatch')
    return path


class RAFTAlignment:
    name = 'raft-small-c-t-v2'

    def __init__(self, cache: Path):
        import torch
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        self.torch = torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = raft_small(weights=None).eval()
        self.model.load_state_dict(torch.load(verified_asset(cache, 'raft-small'),
                                              map_location='cpu', weights_only=True))
        self.model.to(self.device)
        self.transform = Raft_Small_Weights.C_T_V2.transforms()

    def __call__(self, reference, neighbour):
        import cv2
        import numpy as np
        torch = self.torch
        h, w = reference.shape[:2]
        if min(h, w) < 6:
            return None
        # Bounded network working grid. Native crops remain untouched.
        factor = min(1., 256 / max(h, w))
        oh = max(128, math.ceil(h * factor / 8) * 8)
        ow = max(128, math.ceil(w * factor / 8) * 8)
        resized = cv2.resize(neighbour, (w, h))
        def tensor(image):
            rgb = cv2.resize(image, (ow, oh))[:, :, ::-1].copy()
            return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        a, b = self.transform(tensor(reference), tensor(resized))
        with torch.inference_mode():
            forward = self.model(a, b, num_flow_updates=12)[-1][0].permute(1, 2, 0).cpu().numpy()
            backward = self.model(b, a, num_flow_updates=12)[-1][0].permute(1, 2, 0).cpu().numpy()
        forward = cv2.resize(forward, (w, h)) * np.array([w / ow, h / oh], np.float32)
        backward = cv2.resize(backward, (w, h)) * np.array([w / ow, h / oh], np.float32)
        if not np.isfinite(forward).all() or not np.isfinite(backward).all():
            return None
        xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        mx, my = xx + forward[:, :, 0], yy + forward[:, :, 1]
        reverse = cv2.remap(backward, mx, my, cv2.INTER_LINEAR)
        errors = np.linalg.norm(forward + reverse, axis=2)
        inside = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
        if not inside.any() or float(np.median(errors[inside])) > 1.5:
            return None
        valid = inside & (errors <= 1.5)
        if valid.mean() < .7:
            return None
        warped = cv2.remap(resized, mx, my, cv2.INTER_LINEAR)
        # Flow consistency alone cannot prove that unrelated plates match.
        first = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(float)
        second = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(float)
        if np.std(first[valid]) < 2 or np.std(second[valid]) < 2:
            return None
        correlation = float(np.corrcoef(first[valid], second[valid])[0, 1])
        if not math.isfinite(correlation) or correlation < .75:
            return None
        return AlignedSample(warped, valid, float(np.median(errors[valid])))


class NAFRestoration:
    name = 'nafnet-gopro-width32'

    def __init__(self, cache: Path):
        import torch
        from .nafnet_inference import NAFNet
        self.torch = torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = NAFNet().eval()
        checkpoint = torch.load(verified_asset(cache, 'nafnet-gopro32'), map_location='cpu', weights_only=True)
        self.model.load_state_dict(checkpoint.get('params', checkpoint), strict=True)
        self.model.to(self.device)

    def __call__(self, image):
        import numpy as np
        # Under 384x384, author's local pooling is global pooling. Larger inputs
        # are refused rather than silently running a different pooling recipe.
        h, w = image.shape[:2]
        if max(h, w) > 256 or image.dtype != np.uint8:
            raise ValueError('Restoration crop exceeds verified architecture bounds')
        tensor = self.torch.from_numpy(image[:, :, ::-1].copy()).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device, dtype=self.torch.float32) / 255
        with self.torch.inference_mode():
            result = self.model(tensor)[0].permute(1, 2, 0).cpu().numpy()
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite restoration output')
        return np.clip(np.rint(result[:, :, ::-1] * 255), 0, 255).astype(np.uint8)


class HeavyExecutor:
    """Shared fail-fast bounded executor. Overload bypasses optional processing."""
    def __init__(self):
        self._lock = threading.Lock()
        self._guard = threading.Lock()
        self.pending = 0
        self.bypass = False

    def run(self, function, *args):
        with self._guard:
            if self.pending >= 6:
                self.bypass = True
            if self.bypass and self.pending < 4:
                self.bypass = False
            if self.bypass or self.pending >= 8:
                return None, 'overload_bypass'
            self.pending += 1
        try:
            if not self._lock.acquire(timeout=.05):
                return None, 'executor_busy'
            try:
                return function(*args), None
            except Exception:
                # Do not log exception messages that might contain image/text.
                return None, 'model_failure'
            finally:
                self._lock.release()
        finally:
            with self._guard:
                self.pending -= 1


def prepare_assets(cache: Path):
    """Explicit official downloads, never invoked by inference constructors."""
    import requests
    cache.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, url, filename, revision, license_url in (
        ('raft-small', RAFT_URL, 'raft-small.pth', 'Raft_Small_Weights.C_T_V2',
         'https://github.com/pytorch/vision/blob/main/LICENSE'),
        ('nafnet-gopro32', NAF_URL, 'NAFNet-GoPro-width32.pth', NAF_REVISION,
         f'https://github.com/megvii-research/NAFNet/blob/{NAF_REVISION}/LICENSE'),
    ):
        path = cache / filename
        try:
            if not path.is_file():
                with requests.get(url, stream=True, timeout=(15, 60)) as response:
                    response.raise_for_status()
                    if 'text/' in response.headers.get('Content-Type', ''):
                        raise ValueError('Official model download returned a page, not weights')
                    temp = cache / (filename + '.download')
                    size = 0
                    try:
                        with temp.open('wb') as handle:
                            for chunk in response.iter_content(1024 * 1024):
                                size += len(chunk)
                                if size > 256 * 1024 * 1024:
                                    raise ValueError('Model asset exceeds download bound')
                                handle.write(chunk)
                        if size < 100000:
                            raise ValueError('Unexpectedly small model asset')
                        temp.replace(path)
                    finally:
                        temp.unlink(missing_ok=True)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if name == 'raft-small' and not digest.startswith('01064c6d'):
                raise ValueError('Official RAFT published hash does not match')
            manifest[name] = {'file': filename, 'source_url': url, 'source_revision': revision,
                              'license_url': license_url, 'sha256': digest,
                              'download_hash_is_not_an_independent_publisher_signature': True}
            print(json.dumps({'asset': name, 'available': True, 'bytes': path.stat().st_size}), flush=True)
        except (requests.RequestException, ValueError, OSError):
            print(json.dumps({'asset': name, 'available': False, 'reason': 'official_asset_unavailable'}), flush=True)
    (cache / 'assets.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest
