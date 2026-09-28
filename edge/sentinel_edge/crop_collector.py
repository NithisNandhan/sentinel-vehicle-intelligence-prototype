"""Single detector owner collects native crops while the OCR owner processes.

The decoder, detector collector and OCR consumer have separate bounded stages.
No Torch/Paddle model is shared across inference threads, and no media is saved.
"""
from dataclasses import dataclass, field
import time

from .live_detection_viewer import clipped_box, crop_box, roi_schedule
from .sampled_frames import SampledFrames
from .stream import VideoFrame


@dataclass
class CollectedFrame(VideoFrame):
    tracks: list = field(default_factory=list)
    plate_cache: dict = field(default_factory=dict)
    vehicle_ms: float = 0.
    collection_ms: float = 0.
    captured_at: float = 0.


class CropCollector(SampledFrames):
    def __init__(self, frames, stop, vehicle, detector, *, roi_budget=3,
                 adaptive_sampling=False, full_image_fallback=False, **kwargs):
        super().__init__(frames,stop,**kwargs)
        if not 1 <= roi_budget <= 8:
            raise ValueError('Invalid collector ROI budget')
        self.vehicle, self.detector, self.roi_budget = vehicle,detector,roi_budget
        self.collected_crops = self.input_dropped = 0
        self.adaptive_sampling, self.full_image_fallback = adaptive_sampling,full_image_fallback
        self.full_image_calls = self.full_image_crops = 0

    @staticmethod
    def size(frame):
        return frame.image.nbytes+sum(item['crop'].nbytes for item in frame.plate_cache.values() if item['crop'] is not None)

    def _drop(self):
        _,frame = self.queue.popleft()
        self.bytes_used -= self.size(frame)
        self.dropped_samples += 1

    def _read(self):
        offset = sequence = 0
        last_selected_pts = active_until_pts = last_full_scan_pts = None
        self.vehicle.reset()
        try:
            # Decode samples at the fastest permitted cadence; inference stays
            # at the idle cadence unless a recent sample contained a vehicle.
            with SampledFrames(self.frames,self.stop,interval_ms=250 if self.adaptive_sampling else self.interval_ms,
                    max_frames=self.max_frames,max_bytes=self.max_bytes,max_age_ms=self.max_age_ms) as tap:
                while not self.stop.is_set():
                    item = tap.take(sequence)
                    self.decoded, self.input_dropped = tap.decoded,tap.dropped_samples
                    if item is None:
                        if tap.finished:
                            if tap.failed:
                                self.failed = True
                                self.failure_code = getattr(tap,'failure_code','DecoderFailure')
                            break
                        continue
                    sequence,frame = item
                    started = time.monotonic()
                    if frame.discontinuity:
                        self.vehicle.reset()
                        offset = 0
                        last_selected_pts = active_until_pts = last_full_scan_pts = None
                    desired_interval = (250 if self.adaptive_sampling and active_until_pts is not None
                                        and frame.pts_ms <= active_until_pts else self.interval_ms)
                    if last_selected_pts is not None and frame.pts_ms-last_selected_pts < desired_interval:
                        continue
                    last_selected_pts = frame.pts_ms
                    tick = time.monotonic()
                    tracks = self.vehicle.track(frame.image)
                    vehicle_ms = (time.monotonic()-tick)*1000
                    eligible = []
                    for track in tracks:
                        box = clipped_box(frame.image,track.box)
                        if box[2]-box[0] >= 64 and box[3]-box[1] >= 32:
                            eligible.append((track,box))
                    if self.adaptive_sampling and eligible:
                        active_until_pts = frame.pts_ms + 750
                    proposals = {}
                    for index in roi_schedule(len(eligible),offset,self.roi_budget):
                        track,box = eligible[index]
                        region = crop_box(frame.image,box)
                        tick = time.monotonic()
                        local = self.detector.box(region)
                        elapsed = (time.monotonic()-tick)*1000
                        local = clipped_box(region,local) if local is not None else None
                        crop = crop_box(region,local) if local and local[2]-local[0]>=24 and local[3]-local[1]>=8 else None
                        proposals[track.track_id] = dict(box=local,crop=crop,ms=elapsed,
                            calls=getattr(self.detector,'last_calls',1),fallback=bool(getattr(self.detector,'last_fallback',False)),
                            source='vehicle_roi')
                        self.collected_crops += crop is not None
                    # One direct full-scene search per second, only after every
                    # selected vehicle ROI failed. Attribute the plate only if
                    # it fits completely inside exactly one best-matching track.
                    if (self.full_image_fallback and eligible and not any(p['crop'] is not None for p in proposals.values())
                            and (last_full_scan_pts is None or frame.pts_ms-last_full_scan_pts >= 1000)):
                        last_full_scan_pts = frame.pts_ms
                        direct = getattr(self.detector,'detector',self.detector)
                        tick = time.monotonic()
                        global_box = direct.box(frame.image)
                        elapsed = (time.monotonic()-tick)*1000
                        self.full_image_calls += 1
                        global_box = clipped_box(frame.image,global_box) if global_box is not None else None
                        if global_box is not None and global_box[2]-global_box[0] >= 24 and global_box[3]-global_box[1] >= 8:
                            matching = [(track,box) for track,box in eligible if
                                box[0] <= global_box[0] and box[1] <= global_box[1]
                                and global_box[2] <= box[2] and global_box[3] <= box[3]]
                            if matching:
                                track,box = min(matching,key=lambda item:(item[1][2]-item[1][0])*(item[1][3]-item[1][1]))
                                local = (global_box[0]-box[0],global_box[1]-box[1],
                                         global_box[2]-box[0],global_box[3]-box[1])
                                crop = crop_box(frame.image,global_box)
                                proposals[track.track_id] = dict(box=local,crop=crop,ms=elapsed,
                                    calls=1,fallback=False,source='full_image')
                                self.collected_crops += 1
                                self.full_image_crops += 1
                    offset += self.roi_budget
                    packet = CollectedFrame(frame.image,frame.pts_ms,frame.discontinuity,
                        tracks,proposals,vehicle_ms,(time.monotonic()-started)*1000,frame.captured_at)
                    size = self.size(packet)
                    with self.condition:
                        if packet.discontinuity:
                            while self.queue:
                                self._drop()
                            self.pending_cut = True
                        if size > self.max_bytes:
                            self.dropped_samples += 1
                            continue
                        while self.queue and (len(self.queue)>=self.max_frames
                                or self.bytes_used+size>self.max_bytes
                                or packet.pts_ms-self.queue[0][1].pts_ms>self.max_age_ms):
                            self._drop()
                        self.sequence += 1
                        self.sampled += 1
                        self.queue.append((self.sequence,packet))
                        self.bytes_used += size
                        self.peak_bytes = max(self.peak_bytes,self.bytes_used)
                        self.condition.notify_all()
        except Exception as exc:
            self.failed = True
            self.failure_code = type(exc).__name__
        finally:
            with self.condition:
                self.finished = True
                self.condition.notify_all()

    def take(self, previous):
        # Parent accounts only for decoded frames; release crop bytes here too.
        with self.condition:
            result = super().take(previous)
            if result is not None:
                actual = sum(item['crop'].nbytes for item in result[1].plate_cache.values() if item['crop'] is not None)
                self.bytes_used -= actual
            return result
