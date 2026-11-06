"""Bounded, native-pixel plate search after a whole-vehicle detector miss.

Tiles change the detector's field of view, not the evidence pixels. This is a
diagnostic fallback, not a calibrated confidence boost or a full-scene scan.
"""
import math


class TargetedPlateSearch:
    def __init__(self, detector):
        self.detector = detector
        self.last_calls = 0
        self.last_fallback = False
        # Uncalibrated detector score of the returned box, when the adapter reports one.
        self.last_confidence = None

    @staticmethod
    def usable(box, image):
        if box is None or len(box) != 4 or any(not math.isfinite(float(v)) for v in box):
            return False
        x1, y1, x2, y2 = box
        return (0 <= x1 < x2 <= image.shape[1] and 0 <= y1 < y2 <= image.shape[0]
                and x2-x1 >= 24 and y2-y1 >= 8)

    def box(self, image):
        self.last_calls, self.last_fallback, self.last_confidence = 1, False, None
        primary = self.detector.box(image)
        if self.usable(primary, image):
            self.last_confidence = getattr(self.detector, 'last_confidence', None)
            return primary
        h, w = image.shape[:2]
        if w < 192 or h < 96:
            return None
        self.last_fallback = True
        tw, th = math.ceil(w*.65), math.ceil(h*.65)
        proposals = []
        for y in (0, h-th):
            for x in (0, w-tw):
                tile = image[y:y+th, x:x+tw]
                self.last_calls += 1
                box = self.detector.box(tile)
                if not self.usable(box, tile):
                    continue
                x1, y1, x2, y2 = box
                # Avoid border-truncated tile proposals and obvious non-plates.
                if (x1 <= 0 or y1 <= 0 or x2 >= tw or y2 >= th
                        or not 1.2 <= (x2-x1)/(y2-y1) <= 8):
                    continue
                proposals.append(((x1+x, y1+y, x2+x, y2+y), getattr(self.detector, 'last_confidence', None)))
        # Largest usable native area is an explicit heuristic, not model confidence.
        chosen = max(proposals, key=lambda p:(p[0][2]-p[0][0])*(p[0][3]-p[0][1]), default=None)
        if chosen is None:
            return None
        self.last_confidence = chosen[1]
        return chosen[0]
