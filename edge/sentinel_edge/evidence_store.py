"""Bounded in-memory review evidence on the edge. Nothing is written to disk.

Partial and review-required passages keep their best native crops, one
downscaled original frame, the full partial pattern and character evidence
here, under an unguessable reference. The central API stores only the masked
result and the reference; authorised reviewers fetch the evidence through the
operator-only gateway while it is retained (default 2 h). After expiry or an
edge restart the evidence is gone and the review can only be closed without it.
"""
from __future__ import annotations

import base64
import secrets
import threading
import time
from collections import OrderedDict


class EvidenceStore:
    def __init__(self, *, max_items=256, ttl_seconds=7200, max_bytes=96 * 1024 * 1024, clock=time.monotonic):
        if not 1 <= max_items <= 10000 or not 60 <= ttl_seconds <= 86400 or max_bytes < 1024 * 1024:
            raise ValueError('Bounded evidence retention required')
        self.max_items, self.ttl, self.max_bytes, self.clock = max_items, ttl_seconds, max_bytes, clock
        self._items = OrderedDict()
        self._lock = threading.Lock()
        self.bytes_used = 0
        self.evicted = 0

    @staticmethod
    def _size(record):
        return sum(len(value) for value in record.get('_images', []))

    def put(self, record):
        reference = secrets.token_urlsafe(18)
        stored = dict(record, _stored_at=self.clock())
        with self._lock:
            self._items[reference] = stored
            self.bytes_used += self._size(stored)
            self._evict()
        return reference

    def _evict(self):
        now = self.clock()
        while self._items and (len(self._items) > self.max_items or self.bytes_used > self.max_bytes
                               or now - next(iter(self._items.values()))['_stored_at'] > self.ttl):
            _, old = self._items.popitem(last=False)
            self.bytes_used -= self._size(old)
            self.evicted += 1

    def get(self, reference):
        if not isinstance(reference, str) or not 16 <= len(reference) <= 64:
            return None
        with self._lock:
            self._evict()
            record = self._items.get(reference)
            if record is None:
                return None
            return {key: value for key, value in record.items() if not key.startswith('_')}

    def __len__(self):
        with self._lock:
            self._evict()
            return len(self._items)


def review_record(outcome, *, camera_id, source_mode, secondary=None):
    """Reviewer view of one passage: images as base64, full partial pattern, evidence."""
    import cv2
    result = outcome.result
    crops, images = [], []
    for item in outcome.evidence:
        ok, png = cv2.imencode('.png', item['crop'])
        if ok:
            images.append(png.tobytes())
            crops.append({'pts_ms': item['pts_ms'], 'quality': round(item['quality'], 3),
                          'width': int(item['crop'].shape[1]), 'height': int(item['crop'].shape[0]),
                          'image_png_base64': base64.b64encode(png.tobytes()).decode()})
    if outcome.frame_jpeg:
        images.append(outcome.frame_jpeg)
    return {
        'camera_id': camera_id, 'source_mode': source_mode, 'passage_id': outcome.passage_id,
        'first_pts_ms': outcome.first_pts, 'last_pts_ms': outcome.last_pts,
        'result_state': result.state, 'reason': result.reason, 'pattern': result.pattern,
        'unknown_positions': list(result.unknown_positions), 'length': result.length,
        'length_certain': result.length_certain, 'layout': result.layout,
        'positions': [position.public(reveal=True) for position in result.positions],
        'primary_score': result.score, 'secondary': secondary, 'quality': outcome.indicators,
        'observations': result.observations,
        'frame_pts_ms': outcome.frame_pts,
        'frame_jpeg_base64': base64.b64encode(outcome.frame_jpeg).decode() if outcome.frame_jpeg else None,
        'crops': crops, '_images': images,
        'notice': 'Unverified model evidence for authorised review only. Do not export or screenshot plate text.',
    }
