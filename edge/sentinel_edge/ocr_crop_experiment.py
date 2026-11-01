"""Bounded OCR working-copy ablations; never imported by the serving worker.

Views and separately read rows are one observation, not extra video frames.
No transformation changes the native crop or supplies invented characters.
"""
from __future__ import annotations

import re


POLICIES = ("native", "native_beam", "padding", "clahe", "deskew", "two_row", "two_row_beam")


def validate_image(image):
    import numpy as np
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
            or image.shape[2] != 3 or not image.size):
        raise ValueError("Nonempty native BGR uint8 crop required")


def padded_view(image):
    import cv2
    validate_image(image)
    h, w = image.shape[:2]
    y, x = max(1, round(h * .06)), max(1, round(w * .025))
    return cv2.copyMakeBorder(image, y, y, x, x, cv2.BORDER_REPLICATE)


def contrast_view(image):
    import cv2
    validate_image(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2., tileGridSize=(4, 4)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def row_boundary(image):
    """Image-only central whitespace split, conservative dark-on-light plates.

    The existing layout hypothesis supplies a candidate. Require a real valley
    of at least two native rows and sufficient ink on each side. This is a
    hypothesis, not verified two-line annotation or full perspective correction.
    """
    import cv2
    import numpy as np
    from .hybrid import layout_hypotheses
    validate_image(image)
    hypotheses = layout_hypotheses(image)
    if len(hypotheses) < 2:
        return None
    h, w = image.shape[:2]
    boundary = round(hypotheses[1][2] * h)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    middle = ink[:, max(1, w // 12):max(2, w - w // 12)]
    projection = middle.mean(axis=1)
    left = right = boundary
    while left > h // 3 and projection[left - 1] <= .12:
        left -= 1
    while right + 1 < h * 2 // 3 and projection[right + 1] <= .12:
        right += 1
    if right - left + 1 < max(2, round(h * .03)):
        return None
    boundary = (left + right) // 2
    if (not 0 < boundary < h or min(float(np.mean(middle[:boundary])),
                                    float(np.mean(middle[boundary:]))) < .04):
        return None
    return boundary


def decoded(rows, tokens, *, beam=False):
    from .ctc_fusion import greedy_text, prefix_candidates
    if beam:
        candidates = prefix_candidates(rows, tokens, beam_width=32, max_length=25)
        if candidates:
            return candidates[0][0]
    return re.sub("[^A-Z0-9]", "", greedy_text(rows, tokens))


def observe_policies(reader, image, policies=POLICIES):
    """Evaluate configured ablations without reading labels; cache native output.

    Native baseline is always computed. Non-native timings include that baseline
    cost where appropriate; beam/row hypotheses do not count as independent frames.
    """
    import time
    validate_image(image)
    if not policies or set(policies) - set(POLICIES):
        raise ValueError("Unknown or empty ablation policy")
    sync = getattr(reader, "synchronize", None)
    if sync is None:
        sync = lambda: reader.paddle.device.synchronize() if reader.device == "gpu" else None

    def read(view):
        sync()
        start = time.perf_counter()
        rows = reader.probabilities(view)
        sync()
        return rows, (time.perf_counter() - start) * 1000

    rows, native_ms = read(image)
    start = time.perf_counter()
    baseline = decoded(rows, reader.tokens)
    native_ms += (time.perf_counter() - start) * 1000
    results = {}
    for policy in policies:
        start = time.perf_counter()
        reading, detail, calls = baseline, {}, 1
        if policy == "native_beam":
            reading = decoded(rows, reader.tokens, beam=True)
        elif policy in {"padding", "clahe", "deskew"}:
            if policy == "padding":
                view = padded_view(image)
            elif policy == "clahe":
                view = contrast_view(image)
            else:
                from .plate_quality import deskew_view
                view, angle = deskew_view(image)
                detail["angle"] = angle
            if view is not None:
                values, _ = read(view)
                reading = decoded(values, reader.tokens)
                calls += 1
        elif policy in {"two_row", "two_row_beam"}:
            split = row_boundary(image)
            detail["boundary"] = split
            if split is not None:
                a, _ = read(image[:split])
                b, _ = read(image[split:])
                reading = decoded(a, reader.tokens, beam=policy.endswith("beam")) + decoded(b, reader.tokens, beam=policy.endswith("beam"))
                calls += 2
        extra_ms = (time.perf_counter() - start) * 1000
        results[policy] = {"reading": reading, "ocr_calls": calls,
                           "elapsed_ms": round(native_ms + extra_ms, 3), "detail": detail}
    return results
