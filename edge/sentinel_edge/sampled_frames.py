"""Bounded timestamp-sampled decode queue, RAM only, for diagnostic inference.

Live sources drop the oldest sample when the consumer falls behind, so the
view stays current. A recorded file is not a clock: with `block=True` the
decoder waits for room instead, so a slow CPU analyses every sampled frame.
"""
from collections import deque
from dataclasses import dataclass, replace
import time

from .detector_ocr_shadow import LatestFrames
from .stream import VideoFrame


class FrameTooLarge(ValueError):
    """A blocking source must fail visibly rather than silently skip a frame."""


@dataclass
class SampledFrame(VideoFrame):
    captured_at: float = 0.


class SampledFrames(LatestFrames):
    def __init__(self, frames, stop, *, interval_ms=500, max_frames=8,
                 max_bytes=48*1024*1024, max_age_ms=2500, block=False):
        if interval_ms < 150 or not 1 <= max_frames <= 16 or max_bytes < 1 or max_age_ms < interval_ms:
            raise ValueError('Invalid sampled decode bounds')
        super().__init__(frames, stop)
        self.interval_ms, self.max_frames = interval_ms, max_frames
        self.max_bytes, self.max_age_ms = max_bytes, max_age_ms
        self.block = block
        self.queue = deque()
        self.bytes_used = self.peak_bytes = self.sampled = self.dropped_samples = 0

    def _drop(self):
        _, frame = self.queue.popleft()
        self.bytes_used -= frame.image.nbytes
        self.dropped_samples += 1

    def _wait_for_room(self, size):
        """Backpressure: called with the condition held; returns False on stop."""
        while self.queue and (len(self.queue) >= self.max_frames or self.bytes_used+size > self.max_bytes):
            if self.stop.is_set():
                return False
            self.condition.wait(.25)
        return not self.stop.is_set()

    def _read(self):
        last_pts = sampled_pts = None
        carried_cut = False
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
                        sampled_pts = None
                        if self.block:
                            # Queued frames are still unprocessed evidence; mark
                            # the cut on the next sampled frame instead.
                            carried_cut = True
                        else:
                            while self.queue:
                                self._drop()
                            self.pending_cut = True
                    if pts is None:
                        continue
                    last_pts = pts
                    if sampled_pts is not None and pts-sampled_pts < self.interval_ms:
                        continue
                    sampled_pts = pts
                    if frame.image.nbytes > self.max_bytes:
                        if self.block:
                            raise FrameTooLarge('Frame exceeds the decode memory bound')
                        self.dropped_samples += 1
                        continue
                    if self.block:
                        if not self._wait_for_room(frame.image.nbytes):
                            break
                    else:
                        while self.queue and (len(self.queue) >= self.max_frames
                                or self.bytes_used+frame.image.nbytes > self.max_bytes
                                or pts-self.queue[0][1].pts_ms > self.max_age_ms):
                            self._drop()
                    self.sequence += 1
                    self.sampled += 1
                    self.queue.append((self.sequence, SampledFrame(frame.image,pts,
                        frame.discontinuity or carried_cut,time.monotonic())))
                    carried_cut = False
                    self.bytes_used += frame.image.nbytes
                    self.peak_bytes = max(self.peak_bytes, self.bytes_used)
                    self.condition.notify_all()
        except Exception as exc:
            self.failed = True  # Never expose credential-bearing decoder errors.
            self.failure_code = type(exc).__name__
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
            self.condition.notify_all()  # Wake a decoder waiting for room.
            return sequence, frame

    def __exit__(self, *args):
        super().__exit__(*args)
        with self.condition:
            self.queue.clear()
            self.bytes_used = 0
