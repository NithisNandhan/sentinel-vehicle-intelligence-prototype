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
    def __init__(self, frames, stop, vehicle, detector, *, roi_budget=3, **kwargs):
        super().__init__(frames,stop,**kwargs)
        if not 1 <= roi_budget <= 8:
            raise ValueError('Invalid collector ROI budget')
        self.vehicle, self.detector, self.roi_budget = vehicle,detector,roi_budget
        self.collected_crops = self.input_dropped = 0

    @staticmethod
    def size(frame):
        return frame.image.nbytes+sum(item['crop'].nbytes for item in frame.plate_cache.values() if item['crop'] is not None)

    def _drop(self):
        _,frame = self.queue.popleft()
        self.bytes_used -= self.size(frame)
        self.dropped_samples += 1

    def _read(self):
        offset = sequence = 0
        self.vehicle.reset()
        try:
            with SampledFrames(self.frames,self.stop,interval_ms=self.interval_ms,
                    max_frames=self.max_frames,max_bytes=self.max_bytes,max_age_ms=self.max_age_ms) as tap:
                while not self.stop.is_set():
                    item = tap.take(sequence)
                    self.decoded, self.input_dropped = tap.decoded,tap.dropped_samples
                    if item is None:
                        if tap.finished:
                            if tap.failed:
                                self.failed = True
                            break
                        continue
                    sequence,frame = item
                    started = time.monotonic()
                    if frame.discontinuity:
                        self.vehicle.reset()
                        offset = 0
                    tick = time.monotonic()
                    tracks = self.vehicle.track(frame.image)
                    vehicle_ms = (time.monotonic()-tick)*1000
                    eligible = []
                    for track in tracks:
                        box = clipped_box(frame.image,track.box)
                        if box[2]-box[0] >= 64 and box[3]-box[1] >= 32:
                            eligible.append((track,box))
                    proposals = {}
                    for index in roi_schedule(len(eligible),offset,self.roi_budget):
                        track,box = eligible[index]
                        region = crop_box(frame.image,box)
                        tick = time.monotonic()
                        local = self.detector.box(region)
                        elapsed = (time.monotonic()-tick)*1000
                        local = clipped_box(region,local) if local is not None else None
                        crop = crop_box(region,local) if local and local[2]>local[0] and local[3]>local[1] else None
                        proposals[track.track_id] = dict(box=local,crop=crop,ms=elapsed,
                            calls=getattr(self.detector,'last_calls',1),fallback=bool(getattr(self.detector,'last_fallback',False)))
                        self.collected_crops += crop is not None
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
        except Exception:
            self.failed = True
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
