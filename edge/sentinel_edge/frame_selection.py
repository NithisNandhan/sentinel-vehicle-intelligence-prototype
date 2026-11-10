"""Image-only candidate frame selection; experimental until paired validation.

Scores describe image evidence, not probability that OCR is correct. Native
pixels are never altered. Edge contact is a warning, not verified text clipping.
"""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class SelectionPolicy:
    min_width: int = 48
    min_height: int = 12
    min_sharpness: float = 40.
    min_contrast: float = 12.
    max_extreme_fraction: float = .9
    minimum_separation_ms: int = 150


DEFAULT_SELECTION = SelectionPolicy()


def assess_metrics(metrics, policy=DEFAULT_SELECTION):
    """Pure, fixed heuristic; accepts no OCR strings, confidence or ground truth."""
    required = ('width_px', 'height_px', 'sharpness', 'contrast', 'glare_fraction', 'dark_fraction')
    if any(isinstance(metrics.get(k), bool) or not isinstance(metrics.get(k), (int, float))
           or not math.isfinite(metrics[k]) or metrics[k] < 0 for k in required):
        raise ValueError('Finite nonnegative image indicators required')
    if max(metrics['glare_fraction'], metrics['dark_fraction']) > 1:
        raise ValueError('Exposure fractions must be between zero and one')
    reasons = []
    if metrics['width_px'] < policy.min_width or metrics['height_px'] < policy.min_height:
        reasons.append('too_few_native_pixels')
    if metrics['sharpness'] < policy.min_sharpness:
        reasons.append('blur_risk')
    if metrics['contrast'] < policy.min_contrast:
        reasons.append('low_contrast')
    if max(metrics['glare_fraction'], metrics['dark_fraction']) > policy.max_extreme_fraction:
        reasons.append('extreme_exposure')
    warnings = []
    if metrics.get('edge_contact_fraction', 0) >= .35:
        warnings.append('possible_edge_contact')
    angle = metrics.get('angle_deg')
    if angle is not None:
        if isinstance(angle, bool) or not isinstance(angle, (int, float)) or not math.isfinite(angle):
            raise ValueError('Finite angle or unknown required')
        if abs(angle) >= 8:
            warnings.append('tilt_risk')
    focus = metrics['sharpness']/(metrics['sharpness']+200.)
    contrast = metrics['contrast']/(metrics['contrast']+30.)
    size = min(1., metrics['width_px']/160.)*min(1., metrics['height_px']/32.)
    exposure = 1.-.5*max(0., metrics['glare_fraction']-.6)-.5*max(0., metrics['dark_fraction']-.6)
    score = size*(.6*focus+.25*contrast+.15*exposure)
    if 'possible_edge_contact' in warnings:
        score *= .85
    if angle is not None:
        score *= 1.-.01*min(abs(angle), 20.)
    return dict(eligible=not reasons, score=round(max(0., min(1., score)), 6),
                rejection_reasons=reasons, warnings=warnings,
                completeness_verified=False, policy=asdict(policy))


def assess_image(image, policy=DEFAULT_SELECTION):
    import cv2
    import numpy as np
    from .plate_quality import quality_indicators
    metrics = quality_indicators(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Dark content reaching the crop edge can be text, a border or vehicle
    # context. It receives a soft penalty; we cannot certify missing characters.
    edge = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    metrics['edge_contact_fraction'] = round(float((edge < 70).mean()), 4)
    return dict(metrics=metrics, **assess_metrics(metrics, policy))


def choose_frames(observations, assessments, *, limit=3, policy=DEFAULT_SELECTION):
    """Best qualifying distinct native views, returned in presentation-time order."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 8:
        raise ValueError('One to eight selected frames required')
    ranked = []
    for observation in observations:
        assessment = assessments[observation['source_id']]
        if assessment['eligible']:
            ranked.append(observation)
    ranked.sort(key=lambda o: (-assessments[o['source_id']]['score'], o['timestamp_ms'], o['source_id']))
    selected, seen = [], set()
    for observation in ranked:
        if observation['source_id'] in seen or any(abs(observation['timestamp_ms']-o['timestamp_ms']) < policy.minimum_separation_ms for o in selected):
            continue
        selected.append(observation)
        seen.add(observation['source_id'])
        if len(selected) == limit:
            break
    return sorted(selected, key=lambda o: o['timestamp_ms'])
