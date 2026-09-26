"""Image-only experimental frame selection; scores are not OCR confidence."""
from __future__ import annotations

import math

from .reconstruction import CropSample


def crop_quality(image):
    """Bounded heuristic for focus, contrast, exposure and native pixel size.

    Never reads OCR outputs or labels. Noise can mimic focus, so this score
    cannot establish readability or replace the recogniser evidence gates.
    """
    import cv2
    import numpy as np
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or not image.size:
        raise ValueError('Nonempty native BGR crop required')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    focus = min(1., float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 150.)
    contrast = min(1., float(gray.std()) / 45.)
    exposure = 1. - float(((gray < 8) | (gray > 247)).mean())
    size = min(1., h / 24., w / 100.)
    return max(.1, min(1., size * (.4 * focus + .35 * contrast + .25 * exposure)))


def select_frames(samples: list[CropSample], *, limit=8):
    """Select best native crops with distinct hashes and >=150 ms separation.

    This bounded offline selector considers at most forty observations. It
    does not treat resized/restored versions as additional camera frames.
    """
    if not 3 <= limit <= 12 or len(samples) > 40:
        raise ValueError('Frame selection exceeds diagnostic bounds')
    for sample in samples:
        if (not isinstance(sample.pts_ms, int) or isinstance(sample.pts_ms, bool) or sample.pts_ms < 0
                or not sample.source_id or not math.isfinite(sample.quality)
                or not .1 <= sample.quality <= 1.):
            raise ValueError('Invalid frame selection metadata')
    selected, identities = [], set()
    for sample in sorted(samples, key=lambda item: (-item.quality, item.pts_ms, item.source_id)):
        if sample.source_id in identities or any(abs(sample.pts_ms - old.pts_ms) < 150 for old in selected):
            continue
        selected.append(sample)
        identities.add(sample.source_id)
        if len(selected) == limit:
            break
    return sorted(selected, key=lambda item: item.pts_ms)


def deskew_view(image):
    """Optional working-copy affine correction when long edges agree.

    Returns (view, angle) or (None, None). Never claims a full perspective
    rectification or replaces the native evidence. Textured/no-edge plates
    safely abstain; supported slopes are limited to 2..20 degrees.
    """
    import cv2
    import numpy as np
    if min(image.shape[:2]) < 12:
        return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    lines = cv2.HoughLinesP(cv2.Canny(gray, 50, 150), 1, np.pi / 180.,
                            threshold=max(12, w // 4), minLineLength=max(12, w // 3),
                            maxLineGap=max(2, w // 20))
    angles = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4) if lines is not None else ():
        angle = math.degrees(math.atan2(int(y2) - int(y1), int(x2) - int(x1)))
        angle = (angle + 90.) % 180. - 90.
        if abs(angle) <= 20:
            angles.append(angle)
    if len(angles) < 2:
        return None, None
    angle = float(np.median(angles))
    if not 2 <= abs(angle) <= 20 or float(np.median(np.abs(np.array(angles) - angle))) > 2:
        return None, None
    # Expand the working canvas so rotation does not cut off corner characters.
    matrix = cv2.getRotationMatrix2D((w / 2., h / 2.), angle, 1.)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    ow, oh = math.ceil(w * cos + h * sin), math.ceil(h * cos + w * sin)
    matrix[0, 2] += (ow - w) / 2.
    matrix[1, 2] += (oh - h) / 2.
    view = cv2.warpAffine(image, matrix, (ow, oh), borderMode=cv2.BORDER_REPLICATE)
    return view, angle
