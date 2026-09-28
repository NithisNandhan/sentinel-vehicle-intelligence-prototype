"""Bounded timestamp-sampled decode queue, RAM only, for diagnostic inference."""
from collections import deque
from dataclasses import dataclass, replace
import time

from .detector_ocr_shadow import LatestFrames
from .stream import VideoFrame


@dataclass
class SampledFrame(VideoFrame):
    captured_at: float = 0.


class SampledFrames(LatestFrames):
    def __init__(self, frames, stop, *, interval_ms=500, max_frames=8,
                 max_bytes=48*1024*1024, max_age_ms=2500):
        if interval_ms < 150 or not 1 <= max_frames <= 16 or max_bytes < 1 or max_age_ms < interval_ms:
            raise ValueError('Invalid sampled decode bounds')
        super().__init__(frames, stop)
        self.interval_ms, self.max_frames = interval_ms, max_frames
        self.max_bytes, self.max_age_ms = max_bytes, max_age_ms
        self.queue = deque()
        self.bytes_used = self.peak_bytes = self.sampled = self.dropped_samples = 0

    def _drop(self):
        _, frame = self.queue.popleft()
        self.bytes_used -= frame.image.nbytes
        self.dropped_samples += 1

    def _read(self):
        last_pts = sampled_pts = None
        try:
            for _, frame in self.frames:
                if self.stop.is_set():
                    break
                with self.condition:
                    self.decoded += 1
                    pts = frame.pts_ms
                    cut = frame.discontinuity or (pts is not None and last_pts is not None
                           and (pts < last_pts or pts-last_pts > self.max_age_ms))
                    if cut:
                        while self.queue:
                            self._drop()
                        sampled_pts = None
                        self.pending_cut = True
                    if pts is None:
                        continue
                    last_pts = pts
                    if sampled_pts is not None and pts-sampled_pts < self.interval_ms:
                        continue
                    sampled_pts = pts
                    if frame.image.nbytes > self.max_bytes:
                        self.dropped_samples += 1
                        continue
                    while self.queue and (len(self.queue) >= self.max_frames
                            or self.bytes_used+frame.image.nbytes > self.max_bytes
                            or pts-self.queue[0][1].pts_ms > self.max_age_ms):
                        self._drop()
                    self.sequence += 1
                    self.sampled += 1
                    self.queue.append((self.sequence, SampledFrame(frame.image,pts,frame.discontinuity,time.monotonic())))
                    self.bytes_used += frame.image.nbytes
                    self.peak_bytes = max(self.peak_bytes, self.bytes_used)
                    self.condition.notify_all()
        except Exception:
            self.failed = True  # Never expose credential-bearing decoder errors.
        finally:
            with self.condition:
                self.finished = True
                self.condition.notify_all()

    def take(self, previous):
        with self.condition:
            self.condition.wait_for(lambda:self.queue or self.finished or self.stop.is_set(), timeout=1)
            if not self.queue or self.stop.is_set():
                return None
            sequence, frame = self.queue.popleft()
            self.bytes_used -= frame.image.nbytes
            frame = replace(frame, discontinuity=frame.discontinuity or self.pending_cut)
            self.pending_cut = False
            self.consumed = sequence
            return sequence, frame

    def __exit__(self, *args):
        super().__exit__(*args)
        with self.condition:
            self.queue.clear()
            self.bytes_used = 0
