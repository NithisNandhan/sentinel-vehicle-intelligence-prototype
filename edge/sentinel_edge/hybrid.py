"""Opt-in plate reconstruction and original-supported sequence recognition."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

from .hybrid_models import HeavyExecutor, NAFRestoration, RAFTAlignment
from .reconstruction import CropBank, reconstruct
from .sequence_fusion import SequencePlateReader
from .temporal import TemporalPlateReader


def layout_hypotheses(image):
    """At most two layouts, with a native-coordinate two-row split hypothesis."""
    import cv2
    import numpy as np
    h, w = image.shape[:2]
    result = [('single', image, None)]
    if h < 16 or w / h > 3.5:
        return result
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    projection = ink[:, max(1, w // 12):max(2, w - w // 12)].mean(axis=1)
    low, high = max(2, h // 3), min(h - 2, h * 2 // 3)
    if low >= high or projection.max() < .1:
        return result
    split = low + int(np.argmin(projection[low:high]))
    if projection[split] > .12 or max(projection[:split], default=0) < .2 or max(projection[split:], default=0) < .2:
        return result
    result.append(('two-row', image, split / h))
    return result


def layout_probabilities(resources, image, split=None):
    """Recognise rows individually, preserving uncertainty in the row ending."""
    if split is None:
        return resources.probabilities(image)
    cfg = resources.ocr.config
    boundary = round(image.shape[0] * split)
    if not 0 < boundary < image.shape[0]:
        return None
    top = resources.probabilities(image[:boundary])
    bottom = resources.probabilities(image[boundary:])
    eos = cfg.alphabet.index(cfg.pad_char)
    unknown = cfg.alphabet.index('?')
    def ending(matrix):
        return next((i for i, row in enumerate(matrix)
                     if max(range(len(row)), key=row.__getitem__) == eos), None)
    a, b = ending(top), ending(bottom)
    if a is None or b is None or not a or not b or a + b >= cfg.max_plate_slots:
        return None
    rows = [row[:] for row in top[:a]]
    # If the first row ending is uncertain, do not invent certain row joins.
    end_support = top[a][eos]
    for row in bottom:
        value = [v * end_support for v in row]
        value[unknown] += 1 - end_support
        rows.append(value)
        if len(rows) == cfg.max_plate_slots:
            break
    return rows


class HybridResources:
    """One set of GPU models shared by camera instances in the worker."""
    def __init__(self, cache: Path, *, alignment='ecc', restoration=False):
        from .parseq_experiment import PARSeqExperiment
        self.ocr = PARSeqExperiment(cache, cached_only=True)
        self.lock = threading.Lock()
        self.executor = HeavyExecutor()
        if alignment not in {'ecc', 'raft'}:
            raise ValueError('Unknown hybrid alignment variant')
        self.aligner = RAFTAlignment(cache / 'hybrid') if alignment == 'raft' else None
        self.alignment = alignment
        self.restorer = NAFRestoration(cache / 'hybrid') if restoration else None
        self.model_version = ('hybrid-v1+parseq-sequence+' + alignment +
                              ('+nafnet-gopro32' if restoration else '+no-deblur'))

    def probabilities(self, image):
        with self.lock:
            return self.ocr.probabilities(image)


@dataclass(repr=False)
class HybridTrackState:
    readers: dict = field(default_factory=dict)
    row_splits: list = field(default_factory=list)
    reconstruction: object | None = None
    restored: object | None = None
    processed_candidates: list = field(default_factory=list)
    restoration_attempts: int = 0
    reconstruction_attempts: int = 0
    last_reference: str | None = None
    reason: str = 'insufficient_temporal_evidence'

    @property
    def observation_count(self):
        return max((r.observation_count for r in self.readers.values()), default=0)

    def candidates(self):
        # Processed-only candidates are intentionally diagnostic, never accepted.
        unique = {}
        for reader in self.readers.values():
            for candidate in reader.candidates():
                if candidate.plate not in unique or candidate.log_score > unique[candidate.plate].log_score:
                    unique[candidate.plate] = candidate
        return sorted(unique.values(), key=lambda c: (-c.log_score, c.plate))[:200]

    def accepted(self, threshold):
        values = [reader.accepted(threshold) for reader in self.readers.values()]
        values = [value for value in values if value]
        if not values or len({v.plate for v in values}) != 1:
            return None
        # A credible conflicting layout remains unresolved.
        if any(reader.candidates() and reader.candidates()[0].score >= threshold and
               reader.candidates()[0].plate != values[0].plate for reader in self.readers.values()):
            return None
        return values[0]

    def partial(self):
        values = [reader.partial() for reader in self.readers.values()]
        return max(values, key=lambda x: sum(c != '?' for c in x), default='')


class HybridEngine:
    def __init__(self, resources, *, threshold=.85, retain_visuals=False):
        self.resources, self.threshold = resources, threshold
        self.bank = CropBank()
        self.tracks = {}
        self.counters = {}
        self.retain_visuals = retain_visuals

    def count(self, name):
        self.counters[name] = self.counters.get(name, 0) + 1

    def drop(self, track_id):
        self.bank.drop(track_id)
        self.tracks.pop(track_id, None)

    def retain(self, track_ids):
        for key in list(self.tracks):
            if key not in track_ids or key not in self.bank.tracks:
                self.drop(key)

    def clear(self):
        self.bank.clear()
        self.tracks.clear()

    def process(self, track_id, crop, pts_ms, *, quality=1.):
        if crop.nbytes > 1024 * 512 * 3:
            self.count('oversized_crop_rejected')
            return None
        if not self.bank.add(track_id, crop, pts_ms, quality=quality):
            self.count('duplicate_or_budget_rejected')
            return None
        self.retain(set(self.bank.tracks))
        state = self.tracks.setdefault(track_id, HybridTrackState())
        cfg = self.resources.ocr.config
        for layout, working, split in layout_hypotheses(crop):
            if split is not None:
                state.row_splits.append(split)
                state.row_splits = state.row_splits[-8:]
                if max(state.row_splits) - min(state.row_splits) > .1:
                    state.readers.pop(layout, None)
                    state.row_splits = [split]
                    self.count('unstable_row_split')
                if len(state.row_splits) < 3:
                    continue
            reader = state.readers.setdefault(layout, SequencePlateReader(cfg.alphabet, cfg.pad_char,
                                                                          slots=cfg.max_plate_slots))
            matrix = layout_probabilities(self.resources, crop if split is not None else working, split)
            if matrix is not None:
                reader.add(matrix, timestamp_ms=pts_ms, quality=quality)
        samples = self.bank.get(track_id)
        if len(samples) >= 3 and state.reconstruction_attempts < 4:
            state.reconstruction_attempts += 1
            def build():
                if self.resources.aligner:
                    return reconstruct(samples, aligner=self.resources.aligner, alignment='raft-small')
                return reconstruct(samples)
            result, failure = self.resources.executor.run(build)
            state.reason = failure or ('alignment_rejected' if result is None else 'acceptance_evidence_below_threshold')
            state.reconstruction = result
            state.restored = None
            state.processed_candidates = []
            if result is not None:
                self.count('reconstructed_windows')
                state.last_reference = next(s.source_id for s in samples if s.pts_ms == result.reference_pts_ms)
                processed = result.image
                if self.resources.restorer and state.restoration_attempts < 4:
                    state.restoration_attempts += 1
                    restored, failure = self.resources.executor.run(self.resources.restorer, processed)
                    if restored is not None:
                        state.restored, processed = restored, restored
                        self.count('restored_windows')
                    elif failure:
                        state.reason = failure
                # A single processed view is a separate hypothesis, never another
                # temporal vote. Raw original-frame acceptance remains mandatory.
                for _, working, split in layout_hypotheses(processed):
                    diagnostic = TemporalPlateReader(cfg.alphabet, cfg.pad_char)
                    matrix = layout_probabilities(self.resources, processed if split is not None else working, split)
                    if matrix is not None:
                        diagnostic.add(matrix, timestamp_ms=pts_ms)
                        state.processed_candidates.extend(diagnostic.candidates())
        accepted = state.accepted(self.threshold)
        if accepted is not None and state.processed_candidates:
            processed_best = max(state.processed_candidates, key=lambda c: c.log_score)
            if processed_best.score >= self.threshold and processed_best.plate != accepted.plate:
                state.reason = 'original_processed_disagreement'
                self.count('cross_verification_rejected')
                if not self.retain_visuals:
                    state.reconstruction = state.restored = None
                return None
        if accepted:
            state.reason = 'accepted'
        if not self.retain_visuals:
            state.reconstruction = state.restored = None
        return accepted
