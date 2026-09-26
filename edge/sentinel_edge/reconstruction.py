"""Bounded native-image evidence and checked reconstruction. RAM only in live use."""
from __future__ import annotations

import hashlib
import math
from collections import OrderedDict, deque
from dataclasses import dataclass


def native_crop(image, box):
    import numpy as np
    if len(box) != 4 or any(not math.isfinite(float(v)) for v in box):
        raise ValueError('Invalid native crop coordinates')
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('Native crop requires decoded uint8 BGR pixels')
    h, w = image.shape[:2]
    x1, y1 = max(0, math.floor(box[0])), max(0, math.floor(box[1]))
    x2, y2 = min(w, math.ceil(box[2])), min(h, math.ceil(box[3]))
    if x2 <= x1 or y2 <= y1:
        raise ValueError('Empty native crop')
    return image[y1:y2, x1:x2].copy()


@dataclass(repr=False)
class CropSample:
    image: object
    pts_ms: int
    source_id: str
    quality: float


class CropBank:
    def __init__(self, *, max_bytes=64 * 1024 * 1024, max_frames=8, max_tracks=128):
        if min(max_bytes, max_frames, max_tracks) < 1:
            raise ValueError('Invalid crop bank bounds')
        self.max_bytes, self.max_frames, self.max_tracks = max_bytes, max_frames, max_tracks
        self.tracks = OrderedDict()
        self.bytes_used = 0

    def drop(self, track_id):
        for sample in self.tracks.pop(track_id, ()):
            self.bytes_used -= sample.image.nbytes

    def clear(self):
        self.tracks.clear()
        self.bytes_used = 0

    def get(self, track_id):
        return list(self.tracks.get(track_id, ()))

    def add(self, track_id, image, pts_ms, *, quality=1.0):
        import numpy as np
        if (not isinstance(pts_ms, int) or isinstance(pts_ms, bool) or pts_ms < 0 or
                not math.isfinite(quality) or quality <= 0):
            raise ValueError('Invalid native observation metadata')
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Invalid native crop pixels')
        if image.nbytes > self.max_bytes:
            return False
        fingerprint = hashlib.sha256(image.tobytes()).hexdigest()
        samples = self.tracks.get(track_id, ())
        if samples and (pts_ms - samples[-1].pts_ms < 150 or
                        any(s.source_id == fingerprint for s in samples)):
            return False
        # Budget eviction can remove the current track; fetch its deque afterwards.
        while self.tracks and (self.bytes_used + image.nbytes > self.max_bytes or
                               (track_id not in self.tracks and len(self.tracks) >= self.max_tracks)):
            self.drop(next(iter(self.tracks)))
        samples = self.tracks.setdefault(track_id, deque())
        if len(samples) == self.max_frames:
            self.bytes_used -= samples.popleft().image.nbytes
        copy = image.copy()
        samples.append(CropSample(copy, pts_ms, fingerprint, max(.1, min(1., quality))))
        self.bytes_used += copy.nbytes
        self.tracks.move_to_end(track_id)
        return True


@dataclass(repr=False)
class AlignedSample:
    image: object
    valid_mask: object
    error_px: float


@dataclass(repr=False)
class Reconstruction:
    image: object
    reference_image: object
    valid_mask: object
    support_mask: object
    source_ids: tuple[str, ...]
    reference_pts_ms: int
    rejected: int
    alignment: str


def align_ecc(reference, neighbour):
    """Affine alignment on working copies; originals never get resampled in place."""
    import cv2
    import numpy as np
    h, w = reference.shape[:2]
    if min(h, w) < 6 or min(neighbour.shape[:2]) < 6:
        return None
    if not .5 <= (neighbour.shape[1] / neighbour.shape[0]) / (w / h) <= 2:
        return None
    resized = cv2.resize(neighbour, (w, h), interpolation=cv2.INTER_LINEAR)
    first = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    second = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    if first.std() < .01 or second.std() < .01:
        return None
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        score, warp = cv2.findTransformECC(first, second, warp, cv2.MOTION_AFFINE,
                                          (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5),
                                          None, 3)
    except cv2.error:
        return None
    singular = np.linalg.svd(warp[:, :2], compute_uv=False)
    if (not np.isfinite(warp).all() or score < .75 or
            not (.5 <= singular.min() and singular.max() <= 2) or
            np.linalg.det(warp[:, :2]) <= 0 or
            abs(warp[0, 2]) > w * .25 or abs(warp[1, 2]) > h * .25):
        return None
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    aligned = cv2.warpAffine(resized, warp, (w, h), flags=flags)
    mask = cv2.warpAffine(np.ones((h, w), np.float32), warp, (w, h), flags=flags) > .999
    if mask.mean() < .7:
        return None
    # Photometric residual is diagnostic, not a forward/backward flow error.
    residual = float(np.abs(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY).astype(float) -
                            cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY))[mask].mean())
    if residual > 40:
        return None
    return AlignedSample(aligned, mask, residual)


def reconstruct(samples, *, aligner=align_ecc, alignment='ecc'):
    import numpy as np
    if len(samples) < 3 or len({s.source_id for s in samples}) < 3:
        return None
    reference = max(samples, key=lambda s: s.quality)
    h, w = reference.image.shape[:2]
    accumulated = reference.image.astype(np.float32) * reference.quality
    weights = np.full((h, w), reference.quality, np.float32)
    support = np.ones((h, w), np.uint8)
    ids, rejected = [reference.source_id], 0
    for sample in samples:
        if sample is reference:
            continue
        aligned = aligner(reference.image, sample.image)
        if aligned is None or aligned.image.shape != reference.image.shape or aligned.valid_mask.shape != (h, w):
            rejected += 1
            continue
        mask = aligned.valid_mask.astype(bool)
        if not np.isfinite(aligned.image).all() or mask.mean() < .7:
            rejected += 1
            continue
        weighted = mask * sample.quality
        accumulated += aligned.image.astype(np.float32) * weighted[:, :, None]
        weights += weighted
        support += mask.astype(np.uint8)
        ids.append(sample.source_id)
    valid = support >= 3
    if len(ids) < 3 or valid.mean() < .7:
        return None
    fused = np.clip(np.rint(accumulated / weights[:, :, None]), 0, 255).astype(np.uint8)
    # Unsupported borders remain original pixels, not inferred new evidence.
    fused[~valid] = reference.image[~valid]
    return Reconstruction(fused, reference.image.copy(), valid, support, tuple(ids),
                          reference.pts_ms, rejected, alignment)
