"""Conservative, opt-in plate-evidence triage, never a plate correctness score.

Only repeated, upper-body proposals with no confident OCR character and mostly
no character-sized strokes are quarantined. Unreadable crops without that
independent context remain uncertain. Thresholds are development heuristics;
original crops and hypotheses must remain available for inspection.
"""
from collections import Counter
import math

from .plate_filter import CONFIDENT_PEAK

MIN_COMPONENTS = 3
UPPER_BODY_LIMIT = .5
MIN_EVIDENCE_GAP_MS = 150


def character_support(rows, tokens):
    """Count confident, collapsed CTC ASCII alphanumeric characters only."""
    count, previous = 0, None
    for row in rows:
        if len(row) != len(tokens) or not len(row) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in row):
            raise ValueError('Finite CTC posterior rows matching tokens required')
        index = max(range(len(row)), key=row.__getitem__)
        token = tokens[index]
        if (index and index != previous and row[index] >= CONFIDENT_PEAK
                and len(token) == 1 and token.isascii() and token.isalnum()):
            count += 1
        previous = index
    return count


def character_components(image):
    """Count plausible stroke components in either polarity, without reading text.

    This is deliberately only a negative cue combined with OCR and vehicle
    context. Touching, italic and two-row characters can fail this heuristic.
    """
    import cv2
    import numpy as np
    if image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or not image.size:
        raise ValueError('Nonempty native BGR crop required')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, _ = gray.shape
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    counts = []
    for binary in (mask, 255-mask):
        _, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        counts.append(sum(.22*h <= ch <= .95*h and .06 <= cw/ch <= 1.3
                          and area >= 4 and .1 <= area/(cw*ch) <= .95
                          for _, _, cw, ch, area in stats[1:]))
    return int(max(counts))


def decide_plate_evidence(features):
    """Pure group rule. Never infer non-plate from missing OCR alone."""
    distinct = []
    seen = set()
    for feature in sorted(features, key=lambda item: item['position_ms']):
        if feature['source_id'] in seen or any(abs(feature['position_ms']-old['position_ms']) < MIN_EVIDENCE_GAP_MS for old in distinct):
            continue
        seen.add(feature['source_id'])
        distinct.append(feature)
    support = max((f['confident_characters'] for f in distinct), default=0)
    upper = [f for f in distinct if f['observed_parent_y'] is not None
             and 0 <= f['observed_parent_y'] < UPPER_BODY_LIMIT]
    sparse = sum(f['character_components'] < MIN_COMPONENTS for f in distinct)
    suspect = (len(distinct) >= 2 and support == 0 and sparse > len(distinct)/2
               and len(upper) >= 2 and len(upper) > len(distinct)/2)
    return dict(status='suspected_non_plate' if suspect else 'text_evidence_present' if support else 'uncertain',
                excluded_from_fusion=suspect, distinct_originals=len(distinct),
                maximum_confident_characters=support, sparse_stroke_originals=sparse,
                upper_body_originals=len(upper),
                reasons=['No confident alphanumeric OCR character in any original',
                         'Most originals lack character-sized stroke detail',
                         'Repeated confirmed ownership in the upper half of an observed vehicle'] if suspect else [],
                scope='Development heuristic; not proof of a non-plate or of correct text')


def annotate_plate_evidence(groups, posterior_groups, tokens, timeline, load_image):
    """Annotate retained originals; preserve IDs, images and candidate readings.

    `load_image` supplies the native image for a saved sample. Only exact PTS
    matches to actual observed parents establish the spatial context. A
    predicted box or a candidate/unlocked owner cannot provide that cue.
    """
    if not tokens or tokens[0] != '':
        raise ValueError('CTC blank must be token zero')
    frames = {frame['t']: frame for frame in timeline['frames']}
    if len(frames) != len(timeline['frames']):
        raise ValueError('Unique source timestamps required')
    posteriors = {group['id']: group for group in posterior_groups}
    if len(posteriors) != len(posterior_groups) or set(posteriors) != {g['id'] for g in groups}:
        raise ValueError('One posterior group per evidence history required')
    totals = Counter()
    for group in groups:
        parent = group.get('parent_vehicle_id')
        features = []
        for sample in posteriors[group['id']]['observations']:
            relative = None
            observed = [v for v in frames.get(sample['position_ms'], {}).get('tracks', [])
                        if parent and sample.get('vehicle_id') == parent and v['id'] == parent and v.get('observed')]
            if len(observed) == 1:
                p, b = observed[0]['box'], sample['box']
                if p and p[3] > p[1] and p[2] > p[0] and b[0] >= p[0] and b[2] <= p[2] and b[1] >= p[1] and b[3] <= p[3]:
                    relative = ((b[1]+b[3])/2-p[1])/(p[3]-p[1])
            features.append(dict(position_ms=sample['position_ms'], source_id=sample['source_id'],
                                 confident_characters=character_support(sample['rows'], tokens),
                                 character_components=character_components(load_image(sample)),
                                 observed_parent_y=relative))
        verdict = decide_plate_evidence(features)
        group['plate_validation'] = {**verdict, 'features': features}
        totals[verdict['status']] += 1
        if verdict['excluded_from_fusion']:
            for method in group['methods'].values():
                method['accepted'] = None
                method['accepted_for_vehicle'] = None
    return dict(rule_version=1, history_counts=dict(totals), histories=sum(totals.values()),
                thresholds=dict(ctc_character_peak=CONFIDENT_PEAK, minimum_components=MIN_COMPONENTS,
                                upper_body_limit=UPPER_BODY_LIMIT, minimum_gap_ms=MIN_EVIDENCE_GAP_MS),
                scope='Saved-native-evidence triage; no ground-truth labels used')
