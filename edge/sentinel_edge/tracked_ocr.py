"""Bounded per-vehicle CTC histories; pixels and plate strings stay in memory."""
from dataclasses import dataclass
from collections import Counter, deque
import hashlib
import hmac
import secrets
import statistics

from .ctc_fusion import CTCPlateReader
from .live_detection_viewer import clipped_box, crop_box, roi_schedule
from .plate_quality import crop_quality
from .privacy import mask_plate
from .vision import track_consistent


@dataclass(repr=False)
class History:
    reader: CTCPlateReader
    box: tuple
    vehicle_class: int
    seen_pts: int
    plate_center: tuple | None = None


class TrackedOCR:
    def __init__(self, ocr, detector, *, roi_budget=3, gap_ms=2500, threshold=.85):
        if not 1 <= roi_budget <= 8 or not 150 <= gap_ms <= 5000 or not 0 < threshold <= 1:
            raise ValueError('Invalid diagnostic tracking bounds')
        self.ocr, self.detector = ocr, detector
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

    def reset(self):
        self.histories.clear()
        self.last_sample = self.last_pts = None
        self.metrics['scene_resets'] += 1

    def summary(self):
        return dict(self.metrics, unique_model_accepted_unverified=len(self.accepted_ids),
            maximum_distinct_observations=self.maximum_observations,
            maximum_plate_span_ms=self.maximum_span,
            median_plate_width=statistics.median(self.widths) if self.widths else None,
            median_crop_quality=round(statistics.median(self.qualities),3) if self.qualities else None)

    def process(self, image, tracks, pts_ms, *, posterior_cache=None):
        if isinstance(pts_ms,bool) or not isinstance(pts_ms,int) or pts_ms < 0:
            raise ValueError('PTS required, never substitute arrival time')
        if self.last_pts is not None and pts_ms <= self.last_pts:
            self.reset()
        self.last_pts = pts_ms
        self.metrics['analysed_frames'] += 1
        self.metrics['vehicle_box_instances'] += len(tracks)
        self.metrics['vehicle_frames'] += bool(tracks)
        self.metrics['peak_vehicle_boxes'] = max(self.metrics['peak_vehicle_boxes'],len(tracks))
        self.histories = {key:state for key,state in self.histories.items() if pts_ms-state.seen_pts <= self.gap_ms}
        eligible = []
        for track in tracks:
            box = clipped_box(image,track.box)
            if box[2]-box[0] < 64 or box[3]-box[1] < 32:
                continue
            old = self.histories.get(track.track_id)
            if old and not track_consistent(old.box,box,old.vehicle_class,track.vehicle_class):
                del self.histories[track.track_id]
                self.metrics['consistency_resets'] += 1
            state = self.histories.get(track.track_id)
            if state is None:
                if len(self.histories) >= 128:
                    oldest = min(self.histories,key=lambda key:self.histories[key].seen_pts)
                    del self.histories[oldest]
                state = self.histories[track.track_id] = History(CTCPlateReader(self.ocr.tokens),box,track.vehicle_class,pts_ms)
                self.metrics['vehicle_track_segments'] += 1
            state.box,state.vehicle_class,state.seen_pts = box,track.vehicle_class,pts_ms
            eligible.append((track,state))
        displays = []
        for index in roi_schedule(len(eligible),self.offset,self.roi_budget):
            track,state = eligible[index]
            region = crop_box(image,state.box)
            self.metrics['vehicle_crop_calls'] += 1
            local = self.detector.box(region)
            if local is None:
                continue
            local = clipped_box(region,local)
            if local[2]-local[0] < 24 or local[3]-local[1] < 8:
                self.metrics['too_small_plate_regions'] += 1
                continue
            center = ((local[0]+local[2])/(2*region.shape[1]),(local[1]+local[3])/(2*region.shape[0]))
            if state.plate_center and sum((a-b)**2 for a,b in zip(center,state.plate_center)) > .12:
                state.reader = CTCPlateReader(self.ocr.tokens)
                self.metrics['plate_region_resets'] += 1
            state.plate_center = center
            if state.reader.samples and pts_ms-state.reader.samples[-1][0] > self.gap_ms:
                state.reader = CTCPlateReader(self.ocr.tokens)
                self.metrics['ocr_gap_resets'] += 1
            crop = crop_box(region,local)
            quality = crop_quality(crop)
            self.metrics['plate_crops'] += 1
            self.widths.append(crop.shape[1]); self.qualities.append(quality)
            source = hashlib.sha256(crop.tobytes()).hexdigest()
            # Never count the same crop or too-close frame twice. Shared cache
            # exists only for this original frame; it saves inference, not evidence.
            duplicate = any(sample[1] == source for sample in state.reader.samples)
            separated = not state.reader.samples or pts_ms-state.reader.samples[-1][0] >= 150
            if not duplicate and separated:
                rows = posterior_cache.get(source) if posterior_cache is not None else None
                if rows is None:
                    rows = self.ocr.probabilities(crop)
                    self.metrics['ocr_calls'] += 1
                    if posterior_cache is not None:
                        posterior_cache[source] = rows
                else:
                    self.metrics['shared_frame_ocr_cache_hits'] += 1
                if state.reader.add(rows,timestamp_ms=pts_ms,source_id=source,quality=quality):
                    self.metrics['distinct_ocr_observations'] += 1
            else:
                self.metrics['duplicate_or_close_crops'] += 1
            count = state.reader.observation_count
            self.maximum_observations = max(self.maximum_observations,count)
            span = pts_ms-state.reader.samples[0][0] if count else 0
            self.maximum_span = max(self.maximum_span,span)
            candidates = state.reader.candidates() if count >= 3 else []
            accepted = state.reader.accepted(self.threshold) if count >= 3 else None
            self.metrics['eligible_fusion_windows'] += count >= 3
            self.metrics['candidate_windows'] += bool(candidates)
            if accepted:
                self.accepted_ids.add(hmac.digest(self.secret,accepted.plate.encode(),'sha256'))
            status = 'accepted by model — NOT verified' if accepted else 'uncertain' if candidates else 'collecting distinct frames' if count < 3 else 'no complete candidate'
            global_box = (local[0]+state.box[0],local[1]+state.box[1],local[2]+state.box[0],local[3]+state.box[1])
            displays.append({'box':global_box,'crop':crop,'observations':count,'ocr_status':status,
                'masked_plate':mask_plate(candidates[0].plate) if candidates else None,
                'candidate_count':len(candidates),'track_id':track.track_id})
        self.offset += self.roi_budget
        self.last_sample = displays[-1] if displays else None
        return displays
