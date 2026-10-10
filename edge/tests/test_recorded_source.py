"""Recorded-file plumbing on generated footage. No recognition model runs here,
so nothing in this file is evidence of plate-reading accuracy."""
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from sentinel_edge.crop_collector import CropCollector
from sentinel_edge.recorded_source import (
    ChangedRecording, CorruptRecording, InvalidSeek, MissingPts, RecordingLibrary,
    UnpermittedRecording, new_session_id, read_recording, write_receipt)
from sentinel_edge.sampled_frames import SampledFrames
from sentinel_edge.stream import VideoFrame

from recorded_fixtures import make_video, register


class Ticking:
    """A stop token whose waits advance a fake clock instead of sleeping."""
    def __init__(self):
        self.now, self.waited, self.stopped = 0., 0., False

    def clock(self):
        return self.now

    def is_set(self):
        return self.stopped

    def wait(self, seconds):
        self.now += seconds
        self.waited += seconds
        return self.stopped


class RecordingLibraryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)

    def test_only_videos_with_a_valid_receipt_are_listed(self):
        permitted = make_video(self.root/'permitted.mp4', frames=20)
        register(permitted, title='Junction A')
        make_video(self.root/'no-receipt.mp4', frames=5)
        incomplete = make_video(self.root/'incomplete.mp4', frames=5)
        Path(str(incomplete)+'.receipt.json').write_text(json.dumps(
            {'schema_version': '1.0', 'sha256': '0'*64, 'source': 'somewhere', 'licence': 'x', 'permission': ' '}))
        (self.root/'notes.txt').write_text('not a video')
        found, rejected = RecordingLibrary(self.root).scan()
        self.assertEqual(len(found), 1)
        recording = next(iter(found.values()))
        self.assertEqual(recording.name, 'Junction A')
        self.assertRegex(recording.recording_id, r'^rec-[a-f0-9]{12}$')
        self.assertEqual(recording.recording_id, 'rec-'+recording.sha256[:12])
        self.assertEqual(recording.duration_ms, 2000)
        self.assertIsNone(recording.started_at)
        self.assertEqual(recording.public()['time_basis'], 'processing_time')
        self.assertNotIn('path', recording.public())
        self.assertEqual({row['file']: row['reason'] for row in rejected},
                         {'no-receipt.mp4': 'UnpermittedRecording', 'incomplete.mp4': 'UnpermittedRecording'})

    def test_missing_folder_lists_nothing(self):
        self.assertEqual(RecordingLibrary(self.root/'absent').scan(), ({}, []))

    def test_receipt_is_never_overwritten_and_needs_real_permission_text(self):
        video = make_video(self.root/'clip.mp4', frames=5)
        with self.assertRaises(UnpermittedRecording):
            write_receipt(video, source='x', licence='y', permission='')
        register(video)
        with self.assertRaises(FileExistsError):
            register(video)
        with self.assertRaises(UnpermittedRecording):
            write_receipt(self.root/'notes.txt', source='x', licence='y', permission='z')

    def test_recording_time_is_used_only_when_the_receipt_states_utc(self):
        video = make_video(self.root/'timed.mp4', frames=5)
        register(video, recorded_start_utc='2026-09-01T06:30:00Z')
        recording = next(iter(RecordingLibrary(self.root).scan()[0].values()))
        self.assertEqual(recording.started_at.isoformat(), '2026-09-01T06:30:00+00:00')
        self.assertEqual(recording.public()['time_basis'], 'source_clock')
        other = make_video(self.root/'local-time.mp4', frames=6)
        with self.assertRaises(UnpermittedRecording):
            register(other, recorded_start_utc='2026-09-01T12:00:00+05:30')

    def test_a_file_that_is_not_video_is_rejected_by_name_only(self):
        broken = self.root/'broken.mp4'
        broken.write_bytes(b'this is not a video container' * 50)
        register(broken)
        found, rejected = RecordingLibrary(self.root).scan()
        self.assertEqual(found, {})
        self.assertEqual(rejected, [{'file': 'broken.mp4', 'reason': 'CorruptRecording'}])

    def test_session_ids_are_unique_and_fit_the_wire_pattern(self):
        ids = {new_session_id() for _ in range(50)}
        self.assertEqual(len(ids), 50)
        for value in ids:
            self.assertRegex(value, r'^[A-Za-z0-9_.:-]{1,64}$')


class RecordedDecodeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)
        register(make_video(self.root/'clip.mp4', frames=60, fps=10))
        self.recording = next(iter(RecordingLibrary(self.root).scan()[0].values()))

    def pts(self, **kwargs):
        return [frame.pts_ms for _, frame in read_recording(self.recording, threading.Event(), **kwargs)]

    def test_unpaced_decode_keeps_every_frame_and_its_original_pts(self):
        frames = list(read_recording(self.recording, threading.Event(), paced=False))
        self.assertEqual([frame.pts_ms for _, frame in frames], [i*100 for i in range(60)])
        self.assertEqual({transport for transport, _ in frames}, {'file'})
        self.assertEqual(frames[0][1].image.shape, (120, 160, 3))
        self.assertEqual(frames[0][1].image.dtype, np.uint8)
        self.assertFalse(any(frame.discontinuity for _, frame in frames))

    def test_paced_decode_waits_for_video_time_and_still_keeps_every_frame(self):
        token = Ticking()
        frames = list(read_recording(self.recording, token, paced=True, clock=token.clock))
        self.assertEqual(len(frames), 60)
        self.assertAlmostEqual(token.waited, 5.9, delta=.3)  # 59 intervals of 100 ms
        fast = Ticking()
        list(read_recording(self.recording, fast, paced=False, clock=fast.clock))
        self.assertEqual(fast.waited, 0)

    def test_seek_starts_at_the_requested_video_time(self):
        self.assertEqual(self.pts(paced=False, start_ms=3000), [i*100 for i in range(30, 60)])
        self.assertEqual(self.pts(paced=False, start_ms=3050)[0], 3100)
        for position in (-1, 6001, 1.5, True):
            with self.assertRaises(InvalidSeek):
                self.pts(paced=False, start_ms=position)

    def test_pause_holds_decoding_and_resume_continues_without_losing_frames(self):
        paused, stop, seen = threading.Event(), threading.Event(), []
        paused.set()
        thread = threading.Thread(target=lambda: seen.extend(
            frame.pts_ms for _, frame in read_recording(self.recording, stop, paced=False, paused=paused)))
        thread.start()
        time.sleep(.4)
        self.assertEqual(seen, [])
        paused.clear()
        thread.join(timeout=20)
        self.assertFalse(thread.is_alive())
        self.assertEqual(seen, [i*100 for i in range(60)])

    def test_pause_time_is_not_counted_as_playback_time(self):
        token = Ticking()
        paused = SimpleNamespace(calls=0)
        def is_set():
            paused.calls += 1
            return 20 <= paused.calls < 70  # Pause for 50 polls of 0.1 s part-way through.
        paused.is_set = is_set
        frames = list(read_recording(self.recording, token, paced=True, paused=paused, clock=token.clock))
        self.assertEqual(len(frames), 60)
        self.assertAlmostEqual(token.waited, 5.9 + 4.9, delta=.5)

    def test_stop_ends_decoding_promptly(self):
        stop, count = threading.Event(), 0
        for _ in read_recording(self.recording, stop, paced=False):
            count += 1
            if count == 5:
                stop.set()
        self.assertEqual(count, 5)

    def test_a_file_changed_after_its_receipt_is_refused(self):
        with self.recording.path.open('ab') as stream:
            stream.write(b'tampered')
        with self.assertRaises(ChangedRecording) as raised:
            self.pts(paced=False)
        self.assertEqual(raised.exception.failure_code, 'ChangedRecording')
        self.assertNotIn(str(self.root), str(raised.exception))

    def test_a_frame_without_pts_fails_instead_of_being_skipped(self):
        frame = SimpleNamespace(pts=None, time_base=None)
        stream = SimpleNamespace(type='video', time_base=1)
        class Container:
            streams = [stream]
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def decode(self, _): return iter([frame])
        with patch('av.open', return_value=Container()):
            with self.assertRaises(MissingPts):
                self.pts(paced=False, verify=False)

    def test_undecodable_content_is_reported_as_corrupt_without_the_path(self):
        import av
        with patch('av.open', side_effect=av.error.InvalidDataError(1, 'bad data', str(self.recording.path))):
            with self.assertRaises(CorruptRecording) as raised:
                self.pts(paced=False, verify=False)
        self.assertNotIn(str(self.root), str(raised.exception))


def synthetic(count, *, step=500, jump_at=None, size=(48, 64, 3)):
    """Timestamped frames with no file behind them."""
    pts = 0
    for index in range(count):
        if index == jump_at:
            pts += 60_000
        image = np.full(size, index % 250, np.uint8)
        yield 'file', VideoFrame(image, pts, False)
        pts += step


def drain(tap, delay=.01):
    received, sequence = [], 0
    while True:
        item = tap.take(sequence)
        if item is None:
            if tap.finished and not tap.queue:
                return received
            continue
        sequence, frame = item
        received.append(frame)
        time.sleep(delay)


class BackpressureTests(unittest.TestCase):
    def test_blocking_source_delivers_every_sample_to_a_slow_consumer(self):
        with SampledFrames(synthetic(40), threading.Event(), max_frames=2, block=True) as tap:
            received = drain(tap)
        self.assertEqual([frame.pts_ms for frame in received], [i*500 for i in range(40)])
        self.assertEqual(tap.dropped_samples, 0)
        self.assertFalse(tap.failed)
        self.assertLessEqual(tap.peak_bytes, 2*48*64*3)

    def test_live_behaviour_is_unchanged_and_still_drops_when_behind(self):
        with SampledFrames(synthetic(40), threading.Event(), max_frames=2) as tap:
            time.sleep(.3)  # Let the decoder run ahead of the consumer.
            received = drain(tap)
        self.assertLess(len(received), 40)
        self.assertGreater(tap.dropped_samples, 0)

    def test_a_cut_in_a_recording_is_flagged_without_discarding_queued_frames(self):
        with SampledFrames(synthetic(12, jump_at=6), threading.Event(), max_frames=2, block=True) as tap:
            received = drain(tap)
        self.assertEqual(len(received), 12)
        self.assertEqual([index for index, frame in enumerate(received) if frame.discontinuity], [6])

    def test_sampling_interval_still_applies_to_recordings(self):
        with SampledFrames(synthetic(40, step=100), threading.Event(), block=True) as tap:
            received = drain(tap, delay=0)
        self.assertEqual([frame.pts_ms for frame in received], [i*500 for i in range(8)])
        self.assertEqual(tap.decoded, 40)

    def test_an_oversized_frame_fails_a_recording_visibly(self):
        with SampledFrames(synthetic(3), threading.Event(), max_bytes=100, block=True) as tap:
            received = drain(tap)
        self.assertEqual(received, [])
        self.assertTrue(tap.failed)
        self.assertEqual(tap.failure_code, 'FrameTooLarge')

    def test_stop_releases_a_decoder_that_is_waiting_for_room(self):
        stop = threading.Event()
        with SampledFrames(synthetic(40), stop, max_frames=1, block=True) as tap:
            time.sleep(.2)
            stop.set()
        self.assertFalse(tap.thread.is_alive())

    def test_collector_delivers_every_sample_with_its_tracks_when_blocking(self):
        track = SimpleNamespace(track_id='1', box=(0, 0, 64, 48), vehicle_class=2)
        vehicle = SimpleNamespace(reset=lambda: None, track=lambda image: [track])
        detector = SimpleNamespace(box=lambda region: (4, 4, 40, 20))
        with CropCollector(lambda token: synthetic(30), threading.Event(), vehicle, detector,
                           max_frames=2, block=True) as tap:
            received = drain(tap)
        self.assertEqual([frame.pts_ms for frame in received], [i*500 for i in range(30)])
        self.assertEqual(tap.dropped_samples, 0)
        self.assertEqual(tap.collected_crops, 30)
        self.assertTrue(all(frame.plate_cache['1']['crop'].shape == (16, 36, 3) for frame in received))
        self.assertEqual(tap.bytes_used, 0)
