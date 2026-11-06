"""Bounded per-vehicle CTC histories; pixels and plate strings stay in memory."""
from dataclasses import dataclass, field
from collections import Counter, deque
import hashlib
import hmac
import secrets
import statistics
import time

from .ctc_fusion import CTCPlateReader
from .live_detection_viewer import clipped_box, crop_box, roi_schedule
from .plate_options import PlateOptions
from .plate_quality import crop_quality
from .privacy import mask_plate
from .two_line import row_aware_probabilities, two_line_split
from .vision import track_consistent

MAX_TRACK_HISTORIES = 512


@dataclass(repr=False)
class History:
    reader: CTCPlateReader
    box: tuple
    vehicle_class: int
    seen_pts: int
    plate_center: tuple | None = None
    searches: int = 0
    plate_detections: int = 0
    last_plate_failure: str | None = None
    reconstruction: object | None = None
    # Passage bookkeeping (result states and plate-level tracking; unused otherwise).
    passage_id: str = ''
    track_ids: list = field(default_factory=list)
    first_pts: int = 0
    first_box: tuple | None = None
    vehicle_samples: int = 0
    failures: Counter = field(default_factory=Counter)
    layouts: list = field(default_factory=list)
    max_plate_height: int = 0
    evidence: list = field(default_factory=list)
    frame_jpeg: bytes | None = None
    frame_pts: int | None = None
    signatures: list = field(default_factory=list)
    confirmed_plate: str | None = None


class TrackedOCR:
    def __init__(self, ocr, detector, *, roi_budget=None, gap_ms=2500, threshold=.85, debug=False, on_accepted=None, reconstruction=False,
                 options=None, on_finalized=None, result_policy=None, keep_evidence=False, secondary_veto=None):
        if (roi_budget is not None and not 1 <= roi_budget <= 8) or not 150 <= gap_ms <= 5000 or not 0 < threshold <= 1:
            raise ValueError('Invalid diagnostic tracking bounds')
        self.ocr, self.detector = ocr, detector
        self.options = options or PlateOptions()  # Every opt-in plate option is off unless given.
        self.non_plates = None
        if self.options.reject_non_plates:
            from .plate_filter import NonPlateFilter
            self.non_plates = NonPlateFilter()
        self.roi_budget, self.gap_ms, self.threshold = roi_budget, gap_ms, threshold
        self.histories = {}
        self.secret = secrets.token_bytes(32)
        self.accepted_ids = set()
        self.metrics = Counter()
        self.widths, self.qualities = deque(maxlen=512), deque(maxlen=512)
        self.last_pts = None
        self.last_sample = None
        self.maximum_observations = self.maximum_span = 0
        self.offset = 0
        self.on_accepted = on_accepted
        # Result states: one outcome per passage, delivered when the passage ends.
        self.on_finalized = on_finalized
        from .result_state import ResultPolicy
        self.result_policy = result_policy or ResultPolicy()
        self.keep_evidence = keep_evidence
        self.secondary_veto = secondary_veto if self.options.secondary_veto else None
        self.stage_ms = {}
        self.debug_enabled, self.reconstruction_enabled = debug, reconstruction
        self.crop_bank = None
        if debug or reconstruction:
            from .reconstruction import CropBank
            self.crop_bank = CropBank(max_bytes=32*1024*1024, max_frames=8, max_tracks=32,
                                      retain_best=True, max_age_ms=gap_ms)

    def _new_reader(self):
        return CTCPlateReader(self.ocr.tokens, **self.options.reader_options())

    def _read(self, crop, posterior_cache):
        """CTC rows for one crop, shared within the original frame through the cache."""
        source = hashlib.sha256(crop.tobytes()).hexdigest()
        rows = posterior_cache.get(source) if posterior_cache is not None else None
        if rows is not None:
            self.metrics['shared_frame_ocr_cache_hits'] += 1
            return rows
        started = time.monotonic()
        if self.options.two_line and two_line_split(crop) is not None:
            rows = row_aware_probabilities(self.ocr, crop)
            self.metrics['two_line_reads'] += 1
        else:
            rows = self.ocr.probabilities(crop)
        self.stage_ms['ocr'] += (time.monotonic()-started)*1000
        self.metrics['ocr_calls'] += 1
        if posterior_cache is not None:
            posterior_cache[source] = rows
        return rows

    def flush(self, cause='end_of_source'):
        """End every open passage (end of a recording or an operator stop)."""
        for key in list(self.histories):
            self._finalize(key, self.histories.pop(key), cause)

    def needs_more_evidence(self):
        """True while a vehicle seen in the latest sample has a plate but too little evidence."""
        return any(state.seen_pts == self.last_pts and state.plate_detections and state.confirmed_plate is None
                   and state.reader.observation_count < 3 for state in self.histories.values())

    def _finalize(self, key, state, cause):
        if not self.options.result_states or self.on_finalized is None:
            return
        from .layout import passage_layout
        from .passages import MIN_VEHICLE_SAMPLES, PassageOutcome, empty_reason
        from .result_state import CONFIRMED, classify
        if state.vehicle_samples < MIN_VEHICLE_SAMPLES and not state.plate_detections:
            self.metrics['transient_tracks_not_reported'] += 1
            return
        layout = passage_layout(state.layouts)
        veto = self._veto(state) if state.reader.observation_count else None
        result = classify(state.reader, threshold=self.threshold, policy=self.result_policy, layout=layout,
                          max_plate_height=state.max_plate_height or None, veto_positions=veto,
                          empty_reason=empty_reason(state.failures))
        revisited = False
        if state.confirmed_plate is not None and (result.state != CONFIRMED or result.plate != state.confirmed_plate):
            # A sighting was already sent; it stands, but the disagreement is recorded for review.
            revisited = True
            self.metrics['post_confirmation_disagreements'] += 1
        indicators = None
        if state.evidence:
            from .plate_quality import quality_indicators
            indicators = quality_indicators(state.evidence[0]['crop'])
        self.metrics['passages_'+('confirmed' if state.confirmed_plate else result.state)] += 1
        self.on_finalized(PassageOutcome(
            state.passage_id, tuple(state.track_ids), state.vehicle_class, state.first_pts, state.seen_pts, result, cause,
            plate_detections=state.plate_detections, searches=state.searches, vehicle_samples=state.vehicle_samples,
            first_box=state.first_box, last_box=state.box, evidence=list(state.evidence) if self.keep_evidence else [],
            frame_jpeg=state.frame_jpeg if self.keep_evidence else None, frame_pts=state.frame_pts,
            indicators=indicators, confirmed_plate=state.confirmed_plate, revisited=revisited,
            layouts=dict(Counter(state.layouts))))

    def _veto(self, state):
        if self.secondary_veto is None or not state.evidence:
            return None
        candidates = state.reader.candidates()
        if not candidates:
            return None
        return self.secondary_veto.disputed_positions(candidates[0].plate, [item['crop'] for item in state.evidence])

    def _gate(self, state, accepted):
        """With result states, an accepted reading must also be CONFIRMED to be delivered."""
        if accepted is None or not self.options.result_states:
            return accepted
        from .layout import passage_layout
        from .result_state import CONFIRMED, classify
        result = classify(state.reader, threshold=self.threshold, policy=self.result_policy,
                          layout=passage_layout(state.layouts), max_plate_height=state.max_plate_height or None,
                          veto_positions=self._veto(state))
        if result.state != CONFIRMED or result.plate != accepted.plate:
            self.metrics['confirmation_blocked_'+(result.reason or 'changed')] += 1
            return None
        return accepted

    def _remember(self, state, crop, quality, pts_ms, image):
        """Bounded best-crop evidence for review (edge memory only)."""
        from .passages import MAX_EVIDENCE_CROPS
        best = not state.evidence or quality > state.evidence[0]['quality']
        state.evidence.append({'pts_ms': pts_ms, 'crop': crop.copy(), 'quality': quality})
        state.evidence.sort(key=lambda item: -item['quality'])
        del state.evidence[MAX_EVIDENCE_CROPS:]
        if best and image is not None:
            from .live_detection_viewer import encoded
            import base64
            import cv2
            scale = min(1., 1280 / image.shape[1])
            view = image if scale == 1 else cv2.resize(image, (round(image.shape[1]*scale), round(image.shape[0]*scale)),
                                                       interpolation=cv2.INTER_AREA)
            state.frame_jpeg, state.frame_pts = base64.b64decode(encoded(view, '.jpg')), pts_ms

    def reset(self):
        for key in list(self.histories):
            self._finalize(key, self.histories[key], 'reset')
        self.histories.clear()
        if self.crop_bank is not None:
            self.crop_bank.clear()
        self.last_sample = self.last_pts = None
        self.metrics['scene_resets'] += 1

    def summary(self):
        return dict(self.metrics, unique_model_accepted_unverified=len(self.accepted_ids),
            maximum_distinct_observations=self.maximum_observations,
            maximum_plate_span_ms=self.maximum_span,
            median_plate_width=statistics.median(self.widths) if self.widths else None,
            median_crop_quality=round(statistics.median(self.qualities),3) if self.qualities else None)

    def debug_snapshot(self):
        if not self.debug_enabled:
            return {}
        from .recognition_debug import describe_track
        histories = sorted(self.histories.items(), key=lambda item: item[1].seen_pts, reverse=True)
        evidence_ids = {key for key, _ in histories[:16]}
        return {'tracks': [describe_track(key, state,
                    self.crop_bank.get(key) if key in evidence_ids else [], self.threshold)
                           for key, state in histories], 'stage_ms': dict(self.stage_ms),
                'roi_budget': self.roi_budget if self.roi_budget is not None else 'all eligible tracks',
                'track_history_capacity': MAX_TRACK_HISTORIES,
                'crop_bytes': self.crop_bank.bytes_used,
                'score_notice': 'Uncalibrated character scores; not probabilities of correct registration'}

    def process(self, image, tracks, pts_ms, *, posterior_cache=None, plate_cache=None):
        if isinstance(pts_ms,bool) or not isinstance(pts_ms,int) or pts_ms < 0:
            raise ValueError('PTS required, never substitute arrival time')
        if self.last_pts is not None and pts_ms <= self.last_pts:
            self.reset()
        self.last_pts = pts_ms
        self.metrics['analysed_frames'] += 1
        self.metrics['vehicle_box_instances'] += len(tracks)
        self.metrics['vehicle_frames'] += bool(tracks)
        self.metrics['peak_vehicle_boxes'] = max(self.metrics['peak_vehicle_boxes'],len(tracks))
        for key in [key for key, state in self.histories.items() if pts_ms-state.seen_pts > self.gap_ms]:
            self._finalize(key, self.histories.pop(key), 'expired')
        if self.crop_bank is not None:
            for key in list(self.crop_bank.tracks):
                if key not in self.histories:
                    self.crop_bank.drop(key)
        # A tracked vehicle may remain visible while no plate is found. Expire
        # its OCR evidence even on such misses, not only on the next plate hit.
        for key, history in self.histories.items():
            if history.reader.samples and pts_ms-history.reader.samples[-1][0] > self.gap_ms:
                history.reader = self._new_reader()
                history.reconstruction = None
                if self.crop_bank is not None:
                    self.crop_bank.drop(key)
                self.metrics['ocr_gap_resets'] += 1
        self.stage_ms = dict(plate=0., ocr=0., reconstruction=0., fusion=0.)
        begin = getattr(self.detector, 'begin_frame', None)
        if callable(begin):
            begin()  # Per-frame budget for pooled plate proposals.
        eligible = []
        transfers = {}
        if self.options.plate_level_tracking:
            from .passages import reassociations
            current = {track.track_id for track in tracks}
            fresh = [(track.track_id, clipped_box(image,track.box), track.vehicle_class)
                     for track in tracks if track.track_id not in self.histories]
            transfers = reassociations(fresh, self.histories, current, pts_ms)
        for new_id, old_id in transfers.items():
            moved = self.histories.pop(old_id)
            moved.track_ids.append(new_id)
            self.histories[new_id] = moved
            if self.crop_bank is not None:
                self.crop_bank.drop(old_id)
            self.metrics['track_reassociations'] += 1
        for track in tracks:
            box = clipped_box(image,track.box)
            if box[2]-box[0] < 64 or box[3]-box[1] < 32:
                continue
            old = self.histories.get(track.track_id)
            if old and not track_consistent(old.box,box,old.vehicle_class,track.vehicle_class):
                del self.histories[track.track_id]
                if self.crop_bank is not None:
                    self.crop_bank.drop(track.track_id)
                self.metrics['consistency_resets'] += 1
            state = self.histories.get(track.track_id)
            if state is None:
                if len(self.histories) >= MAX_TRACK_HISTORIES:
                    oldest = min(self.histories,key=lambda key:self.histories[key].seen_pts)
                    del self.histories[oldest]
                    if self.crop_bank is not None:
                        self.crop_bank.drop(oldest)
                from .passages import new_passage_id
                state = self.histories[track.track_id] = History(self._new_reader(),box,track.vehicle_class,pts_ms,
                                                                 passage_id=new_passage_id(),track_ids=[track.track_id],
                                                                 first_pts=pts_ms,first_box=box)
                if self.reconstruction_enabled:
                    from .live_reconstruction import ReconstructionState
                    state.reconstruction = ReconstructionState()
                self.metrics['vehicle_track_segments'] += 1
            state.box,state.vehicle_class,state.seen_pts = box,track.vehicle_class,pts_ms
            state.vehicle_samples += 1
            eligible.append((track,state))
        self.metrics['eligible_vehicle_tracks'] += len(eligible)
        displays = []
        selection = (roi_schedule(len(eligible),self.offset,self.roi_budget) if plate_cache is None else
                     [i for i,(track,_) in enumerate(eligible) if track.track_id in plate_cache])
        if self.roi_budget is not None:
            selection = selection[:self.roi_budget]
        self.metrics['plate_roi_searches'] += len(selection)
        self.metrics['plate_roi_unsearched_tracks'] += len(eligible)-len(selection)
        for index in selection:
            track,state = eligible[index]
            region = crop_box(image,state.box)
            self.metrics['vehicle_crop_calls'] += 1
            state.searches += 1
            state.last_plate_failure = None
            started = time.monotonic()
            prepared = plate_cache[track.track_id] if plate_cache is not None else None
            local = prepared['box'] if prepared is not None else self.detector.box(region)
            self.metrics['plate_detector_calls'] += prepared['calls'] if prepared is not None else getattr(self.detector,'last_calls',1)
            self.metrics['targeted_searches'] += prepared['fallback'] if prepared is not None else bool(getattr(self.detector,'last_fallback',False))
            self.stage_ms['plate'] += prepared['ms'] if prepared is not None else (time.monotonic()-started)*1000
            if local is None:
                state.last_plate_failure = 'Plate detector found no region in this vehicle ROI'
                state.failures[state.last_plate_failure] += 1
                continue
            local = clipped_box(region,local)
            if local[2]-local[0] < 24 or local[3]-local[1] < 8:
                self.metrics['too_small_plate_regions'] += 1
                state.last_plate_failure = 'Plate proposal is smaller than 24 by 8 native pixels'
                state.failures[state.last_plate_failure] += 1
                continue
            screened = None
            if self.non_plates is not None:
                # Screen before a non-plate can move the plate centre or reset this vehicle's reader.
                crop = prepared['crop'] if prepared is not None else crop_box(region,local)
                screened = self._read(crop,posterior_cache)
                score = prepared.get('confidence') if prepared is not None else getattr(self.detector,'last_confidence',None)
                reason = self.non_plates.reason(crop,screened,score)
                if reason:
                    self.metrics['non_plate_rejections'] += 1
                    state.last_plate_failure = reason
                    state.failures[reason] += 1
                    continue
            state.plate_detections += 1
            center = ((local[0]+local[2])/(2*region.shape[1]),(local[1]+local[3])/(2*region.shape[0]))
            if state.plate_center and sum((a-b)**2 for a,b in zip(center,state.plate_center)) > .12:
                state.reader = self._new_reader()
                state.reconstruction = None
                if self.crop_bank is not None:
                    self.crop_bank.drop(track.track_id)
                self.metrics['plate_region_resets'] += 1
            state.plate_center = center
            if state.reader.samples and pts_ms-state.reader.samples[-1][0] > self.gap_ms:
                state.reader = self._new_reader()
                state.reconstruction = None
                if self.crop_bank is not None:
                    self.crop_bank.drop(track.track_id)
                self.metrics['ocr_gap_resets'] += 1
            crop = prepared['crop'] if prepared is not None else crop_box(region,local)
            quality = crop_quality(crop)
            self.metrics['plate_crops'] += 1
            self.widths.append(crop.shape[1]); self.qualities.append(quality)
            source = hashlib.sha256(crop.tobytes()).hexdigest()
            # Never count the same crop or too-close frame twice. Shared cache
            # exists only for this original frame; it saves inference, not evidence.
            duplicate = any(sample[1] == source for sample in state.reader.samples)
            separated = not state.reader.samples or pts_ms-state.reader.samples[-1][0] >= 150
            thumb = None
            if self.options.near_duplicate_guard and not duplicate and separated:
                # A parked or stalled vehicle gives the same view again: not new evidence.
                from .passages import near_duplicate, signature
                thumb = signature(crop)
                if state.reader.samples and any(near_duplicate(old, thumb) for old in state.signatures
                                                if old.shape == thumb.shape):
                    duplicate = True
                    self.metrics['near_duplicate_crops'] += 1
            if self.options.result_states:
                from .layout import classify as classify_layout
                state.layouts.append(classify_layout(crop).layout)
                del state.layouts[:-16]
                state.max_plate_height = max(state.max_plate_height, crop.shape[0])
            if not duplicate and separated:
                rows = screened if screened is not None else self._read(crop,posterior_cache)
                if state.reader.add(rows,timestamp_ms=pts_ms,source_id=source,quality=quality):
                    self.metrics['distinct_ocr_observations'] += 1
                    if thumb is not None:
                        state.signatures.append(thumb)
                        del state.signatures[:-12]
                    if self.options.result_states:
                        self._remember(state, crop, quality, pts_ms, image if self.keep_evidence else None)
                    if self.crop_bank is not None:
                        self.crop_bank.add(track.track_id,crop,pts_ms,quality=quality)
                        # Eviction must also release processed previews/proposals.
                        for key, history in self.histories.items():
                            if key not in self.crop_bank.tracks and history.reconstruction is not None:
                                history.reconstruction = None
                                history.reader.propose(())
                        if self.reconstruction_enabled:
                            from .live_reconstruction import ReconstructionState, update
                            if state.reconstruction is None:
                                state.reconstruction = ReconstructionState()
                            if update(state.reconstruction, state.reader, self.crop_bank.get(track.track_id), self.ocr, pts_ms):
                                self.stage_ms['reconstruction'] += state.reconstruction.elapsed_ms
                                self.metrics['reconstruction_attempts'] += 1
                                self.metrics['reconstructed_windows'] += state.reconstruction.status == 'reconstructed'
                                self.metrics['reconstruction_failures'] += state.reconstruction.status == 'error'
            else:
                self.metrics['duplicate_or_close_crops'] += 1
            count = state.reader.observation_count
            self.maximum_observations = max(self.maximum_observations,count)
            span = pts_ms-state.reader.samples[0][0] if count else 0
            self.maximum_span = max(self.maximum_span,span)
            started = time.monotonic()
            candidates = state.reader.candidates() if count >= 3 else []
            accepted = state.reader.accepted(self.threshold) if count >= 3 else None
            from .live_reconstruction import disagreement
            if disagreement(state.reconstruction, accepted, self.threshold):
                accepted = None
                self.metrics['reconstruction_disagreements'] += 1
            accepted = self._gate(state, accepted)
            if accepted and state.confirmed_plate is None:
                state.confirmed_plate = accepted.plate
            self.stage_ms['fusion'] += (time.monotonic()-started)*1000
            self.metrics['eligible_fusion_windows'] += count >= 3
            self.metrics['candidate_windows'] += bool(candidates)
            if accepted:
                self.accepted_ids.add(hmac.digest(self.secret,accepted.plate.encode(),'sha256'))
                if self.on_accepted is not None:
                    self.on_accepted(track.track_id, accepted, pts_ms)
            status = 'accepted by model — NOT verified' if accepted else 'uncertain' if candidates else 'collecting distinct frames' if count < 3 else 'no complete candidate'
            global_box = (local[0]+state.box[0],local[1]+state.box[1],local[2]+state.box[0],local[3]+state.box[1])
            displays.append({'box':global_box,'crop':crop,'observations':count,'ocr_status':status,
                'masked_plate':mask_plate(candidates[0].plate) if candidates else None,
                'candidate_count':len(candidates),'track_id':track.track_id,
                'format_adjusted':bool(candidates and candidates[0].format_adjusted)})
        self.offset += len(selection) if self.roi_budget is None else self.roi_budget
        self.last_sample = displays[-1] if displays else None
        return displays
