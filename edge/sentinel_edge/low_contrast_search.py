"""Opt-in character-led and CLAHE plate discovery using existing models.

Enhanced images are search views only. All returned boxes refer to the native
input; callers crop/OCR that input. Text is an unverified proposal, not proof of
a plate, and no OCR answer or human label is used to generate these proposals.
"""
from collections import Counter
import math

from .automatic_crops import overlap
from .crop_completion import intersects_another_plate
from .ocr_crop_experiment import contrast_view
from .plate_proposals import checked_quad, with_margin
from .plate_search import TargetedPlateSearch


def character_support(image, box):
    """Require several aligned glyph-sized components, independently of border colour.

    This is a proposal heuristic, not a trained character/plate classifier. Both
    polarities are checked; no literal characters are predicted.
    """
    import cv2
    x1, y1, x2, y2 = (int(v) for v in box)
    region = image[y1:y2, x1:x2]
    if not region.size:
        return 0
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if h < 8 or w < 24:
        return 0
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    best = 0
    for mask in (binary, 255-binary):
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        glyphs = [(x, y, cw, ch) for x, y, cw, ch, area in stats[1:]
                  if max(5, .18*h) <= ch <= .98*h and .07 <= cw/ch <= 1.35
                  and area >= 5 and .10 <= area/(cw*ch) <= .96]
        # A single common row is sufficient; full two-row outlines are retained.
        for _, y, _, ch in glyphs:
            aligned = [(x, gy, gw, gh) for x, gy, gw, gh in glyphs
                       if .5 <= gh/ch <= 2. and abs((gy+gh/2)-(y+ch/2)) <= .4*max(ch, gh)]
            if len(aligned) >= 3:
                span = max(x+cw for x, _, cw, _ in aligned)-min(x for x, _, _, _ in aligned)
                if span >= .35*w:
                    best = max(best, len(aligned))
    return best


def character_groups(image, polygons, *, limit=6):
    """Keep text lines and conservatively join nearby fragments on the same row.

    Positive neighbouring plate boxes are vetoed by the caller. Each original
    line remains available: joining is only an additional search hypothesis.
    """
    h, w = image.shape[:2]
    boxes = []
    for polygon in polygons[:128]:
        quad = checked_quad(polygon, image.shape)
        if quad is None:
            continue
        box = (max(0, math.floor(min(x for x, _ in quad))),
               max(0, math.floor(min(y for _, y in quad))),
               min(w, math.ceil(max(x for x, _ in quad))+1),
               min(h, math.ceil(max(y for _, y in quad))+1))
        bw, bh = box[2]-box[0], box[3]-box[1]
        if bw >= 24 and bh >= 8 and 1.2 <= bw/bh <= 8 and (box[1]+box[3])/2 >= .20*h:
            boxes.append(box)
    candidates = list(boxes)
    for i, first in enumerate(boxes):
        for second in boxes[i+1:]:
            a, b = sorted((first, second))
            ah, bh = a[3]-a[1], b[3]-b[1]
            shared = min(a[3], b[3])-max(a[1], b[1])
            gap = b[0]-a[2]
            if (.5 <= ah/bh <= 2 and shared >= .65*min(ah, bh)
                    and 0 <= gap <= .6*min(ah, bh)):
                union = (a[0], min(a[1], b[1]), b[2], max(a[3], b[3]))
                if (union[2]-union[0])/(union[3]-union[1]) <= 8:
                    candidates.append(union)
    ranked = []
    for box in dict.fromkeys(candidates):
        support = character_support(image, box)
        if support >= 3:
            # Give characters some context, without relying on a closed rectangle.
            margin = max(2, round((box[3]-box[1])*.12))
            padded = (max(0, box[0]-margin), max(0, box[1]-margin),
                      min(w, box[2]+margin), min(h, box[3]+margin))
            ranked.append((support, padded))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    result = []
    for support, box in ranked:
        if not any(overlap(box, previous['box']) >= .65 for previous in result):
            result.append(dict(box=box, characters=support))
    return result[:limit]


class LowContrastPlateSearch:
    """Wrap the current search; keep legacy results and add bounded alternatives."""
    def __init__(self, primary, *, character_guided=False, contrast_pass=False,
                 max_extra_calls=24, max_candidates=4, text_corners=False):
        if not character_guided and not contrast_pass:
            raise ValueError('At least one discovery method is required')
        if (not 1 <= max_extra_calls <= 128
                or not 1+int(character_guided)+int(contrast_pass) <= max_candidates <= 8):
            raise ValueError('Bounded discovery work required')
        self.primary = primary
        self.character_guided, self.contrast_pass = character_guided, contrast_pass
        if text_corners and not character_guided:
            raise ValueError('Text-corner discovery requires character-guided search')
        self.text_corners = text_corners
        self.max_extra_calls, self.max_candidates = max_extra_calls, max_candidates
        self.remaining = max_extra_calls
        self.metrics = Counter()
        self.last_candidates = ()
        self.last_text_quad = self.last_confidence = self.last_completion = None
        self.last_calls, self.last_fallback = 0, False

    @property
    def detector(self):
        return self.primary.detector

    @property
    def pool(self):
        value = self.primary
        while not hasattr(value, 'text_detector') and hasattr(value, 'primary'):
            value = value.primary
        return value

    def begin_frame(self):
        self.primary.begin_frame()
        self.remaining = self.max_extra_calls

    def diagnostics(self):
        return dict(self.primary.diagnostics(), low_contrast=dict(self.metrics),
                    low_contrast_config=dict(character_guided=self.character_guided,
                                             contrast_pass=self.contrast_pass,
                                             text_corners=self.text_corners,
                                             max_extra_calls=self.max_extra_calls))

    def _spend(self, name):
        if self.remaining <= 0:
            self.metrics['extra_budget_skips'] += 1
            return False
        self.remaining -= 1
        self.last_calls += 1
        self.metrics[name] += 1
        return True

    def box(self, image):
        text_calls_before = getattr(self.pool,'metrics',{}).get('text_detector_calls',0)
        original = self.primary.box(image)
        text_already_ran = getattr(self.pool,'metrics',{}).get('text_detector_calls',0) > text_calls_before
        self.last_calls = self.primary.last_calls
        self.last_fallback = self.primary.last_fallback
        candidates = []
        if original is not None:
            candidates.append(dict(box=tuple(original), proposal_sources=['existing_search'],
                existing_support=list(getattr(self.pool,'last_sources',())),
                confidence=getattr(self.primary, 'last_confidence', None),
                quad=getattr(self.primary, 'last_text_quad', None),
                completion=getattr(self.primary, 'last_completion', None)))
        raw = list(getattr(self.pool, 'last_raw_proposals', ()))
        # Added detector results retain the original detector's own thresholds.
        if self.contrast_pass and self._spend('contrast_calls'):
            enhanced = contrast_view(image)
            if hasattr(self.detector, 'scored_boxes'):
                detected = self.detector.scored_boxes(enhanced)
            else:
                detected = [(box, None) for box in self.detector.boxes(enhanced)]
            for box, confidence in detected[:16]:
                if not TargetedPlateSearch.usable(box, image):
                    continue
                raw.append((tuple(box), 'plate', confidence))
                candidates.append(dict(box=with_margin(tuple(round(v) for v in box), image.shape),
                    proposal_sources=['contrast_plate'], confidence=confidence, quad=None, completion=None))
                self.metrics['contrast_proposals'] += 1
        if self.character_guided:
            polygons = [entry[3] for entry in raw if entry[1] == 'text' and len(entry) > 3]
            text_detector = getattr(self.pool, 'text_detector', None)
            if not polygons and not text_already_ran and text_detector is not None and self._spend('independent_text_calls'):
                polygons = text_detector.polygons(image)
            if self.text_corners:
                from .text_plate_geometry import text_plate_candidates
                groups = text_plate_candidates(image,polygons,
                    other_boxes=[entry[0] for entry in raw if entry[1] in ('plate','tile')])
            else:
                groups = character_groups(image, polygons)
            for group in groups:
                box = group['box']
                # A proposed text union must not encompass two separate plates.
                members = [entry[0] for entry in raw if entry[1] in ('plate', 'tile')
                           and box[0] <= (entry[0][0]+entry[0][2])/2 <= box[2]
                           and box[1] <= (entry[0][1]+entry[0][3])/2 <= box[3]]
                if len(members) > 1 and any(intersects_another_plate(seed, box, raw) for seed in members):
                    self.metrics['separate_plate_vetoes'] += 1
                    continue
                candidates.append(dict(box=box, proposal_sources=['text_corners' if self.text_corners else 'character_group'],
                    confidence=None, quad=group.get('quad'), completion=None,
                    geometry=group.get('geometry'), seed_quad=group.get('seed_quad'),text_members=group.get('text_members'),
                    character_components=group.get('character_components',group.get('characters',0))))
                self.metrics['character_group_proposals'] += 1
        merged = []
        for candidate in candidates:
            for previous in merged:
                # A new oriented enclosure is a different working geometry.
                # Verify it independently before selecting one view for OCR.
                if self.text_corners and ('text_corners' in candidate['proposal_sources']) != ('text_corners' in previous['proposal_sources']):
                    continue
                if overlap(candidate['box'], previous['box']) >= .5:
                    previous['proposal_sources'] = sorted(set(previous['proposal_sources']+candidate['proposal_sources']))
                    self.metrics['same_frame_duplicates'] += 1
                    break
            else:
                merged.append(candidate)
        # Preserve the old result and reserve a novel candidate per new method.
        # Several contrast detections must not silently crowd out text discovery.
        selected = [c for c in merged if 'existing_search' in c['proposal_sources']][:1]
        for source in ('contrast_plate','text_corners' if self.text_corners else 'character_group'):
            novel = next((c for c in merged if source in c['proposal_sources']
                          and 'existing_search' not in c['proposal_sources']
                          and all(c is not value for value in selected)), None)
            if novel is not None:
                selected.append(novel)
        selected.extend(c for c in merged if all(c is not value for value in selected))
        self.last_candidates = tuple(selected[:self.max_candidates])
        self.metrics['extra_candidates_retained'] += sum('existing_search' not in c['proposal_sources'] for c in self.last_candidates)
        best = self.last_candidates[0] if self.last_candidates else None
        self.last_confidence = best['confidence'] if best else None
        self.last_text_quad = best['quad'] if best else None
        self.last_completion = best['completion'] if best else None
        return best['box'] if best else None
