"""Recorded mode in the shared session, with a MOCKED recogniser.

The fake OCR below always returns the same text, so an "accepted" plate here
proves the plumbing from file to outbox only. It is not a recognition result.
"""
import http.client
import json
from pathlib import Path
import re
import socket
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sentinel_edge.events import sighting
from sentinel_edge.live_detection_viewer import LiveState, serve
from sentinel_edge.recognition_contract import operator_snapshot
from sentinel_edge.recorded_source import RecordedRun, RecordingLibrary
from sentinel_edge.shared_delivery import SharedDelivery
from sentinel_edge.tracked_live_viewer import observe_tracked

from recorded_fixtures import make_video, register

TOKENS = ('', *'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
TEXT = 'AP01AB1234'


def posterior(_):
    rows = []
    for char in TEXT:
        row = [0.]*len(TOKENS)
        row[TOKENS.index(char)] = 1.
        rows.append(row)
    return rows


def fakes(ocr_delay=0.):
    def probabilities(image):
        time.sleep(ocr_delay)
        return posterior(image)
    track = SimpleNamespace(track_id='7', box=(0, 0, 160, 120), vehicle_class=2)
    return (SimpleNamespace(tokens=TOKENS, probabilities=probabilities, device='cpu', adaptation=None),
            SimpleNamespace(box=lambda region: (5, 5, 150, 35)),
            SimpleNamespace(reset=lambda: None, track=lambda image: [track]))


class RecordedCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)
        (self.root/'recordings').mkdir()
        register(make_video(self.root/'recordings'/'clip.mp4', frames=60, fps=10), title='Test junction')
        self.recordings = RecordingLibrary(self.root/'recordings').scan()[0]
        self.recording = next(iter(self.recordings.values()))
        self.config = SimpleNamespace(db_path=self.root/'outbox.sqlite', central_api_url='http://example.invalid',
                                      edge_api_key='test-edge-key', hmac_key='test-hmac-key', edge_id='region-test',
                                      ocr_min_confidence=.85, model_cache_dir=self.root, camera_ids=())


class RecordedStateTests(RecordedCase):
    def state(self, clock=None):
        state = LiveState(**({'clock': clock} if clock else {}))
        state.authorise_recordings(self.recordings)
        return state

    def test_recording_is_selectable_and_always_labelled_recorded(self):
        state = self.state()
        self.assertEqual(state.snapshot()['source_mode'], 'live')
        state.select(self.recording.recording_id)
        snapshot = state.snapshot()
        self.assertEqual((snapshot['source_mode'], snapshot['status']), ('recorded', 'connecting'))
        self.assertEqual(snapshot['recording_name'], 'Test junction')
        self.assertEqual(snapshot['video_duration_ms'], 6000)
        self.assertEqual(snapshot['time_basis'], 'processing_time')
        self.assertRegex(snapshot['session_id'], r'^s-\d{8}T\d{6}-[a-f0-9]{8}$')
        self.assertNotIn(str(self.root), json.dumps(snapshot))
        state.select('cam01')
        self.assertEqual(state.snapshot()['source_mode'], 'live')
        self.assertNotIn('session_id', state.snapshot())
        with self.assertRaises(ValueError):
            state.select('rec-000000000000')

    def test_camera_catalogue_refresh_keeps_a_selected_recording(self):
        state = self.state()
        state.select(self.recording.recording_id)
        generation = state.capture_stop
        state.authorise(['cam12'])
        self.assertIs(state.capture_stop, generation)
        self.assertEqual(state.camera_id, self.recording.recording_id)
        state.authorise_recordings({})  # The file or its receipt was removed.
        self.assertIsNone(state.camera_id)
        self.assertTrue(generation.is_set())

    def test_pause_and_resume_do_not_restart_the_run(self):
        state = self.state()
        state.select(self.recording.recording_id)
        generation, session = state.capture_stop, state.session_id
        state.publish({'camera_id': self.recording.recording_id, 'status': 'processing'}, generation)
        state.control('pause')
        self.assertEqual(state.snapshot()['status'], 'paused')
        self.assertEqual(state.snapshot()['camera_id'], self.recording.recording_id)
        state.control('resume')
        self.assertEqual(state.snapshot()['status'], 'processing')
        self.assertIs(state.capture_stop, generation)
        self.assertEqual(state.session_id, session)

    def test_restart_and_seek_are_new_sessions_with_cleared_evidence(self):
        state = self.state()
        state.select(self.recording.recording_id)
        first_stop, first_session = state.capture_stop, state.session_id
        state.publish({'camera_id': self.recording.recording_id, 'status': 'processing',
                       'frame': 'pixels', 'debug': {'plate': TEXT}}, first_stop)
        state.control('pause')
        state.control('seek', 2500)
        self.assertTrue(first_stop.is_set())
        self.assertNotEqual(state.session_id, first_session)
        self.assertEqual(state.run_parameters()[4], 2500)
        self.assertFalse(state.paused.is_set())
        snapshot = state.snapshot(engineering=True)
        self.assertEqual(snapshot['status'], 'connecting')
        self.assertNotIn('frame', snapshot)
        self.assertNotIn(TEXT, json.dumps(snapshot))
        second_session = state.session_id
        state.control('restart')
        self.assertNotEqual(state.session_id, second_session)
        self.assertEqual(state.run_parameters()[4], 0)
        # A late publish from the replaced run must not reappear.
        state.publish({'camera_id': self.recording.recording_id, 'status': 'processing', 'frame': 'late'}, first_stop)
        self.assertNotIn('frame', state.snapshot())

    def test_invalid_controls_are_rejected_without_changing_the_session(self):
        state = self.state()
        with self.assertRaises(ValueError):
            state.control('pause')  # Nothing selected.
        state.select('cam01')
        with self.assertRaises(ValueError):
            state.control('pause')  # Cameras have no playback controls.
        state.select(self.recording.recording_id)
        generation, session = state.capture_stop, state.session_id
        for action, position in (('seek', None), ('seek', -1), ('seek', 6001), ('seek', 1.5), ('seek', True),
                                 ('pause', 100), ('rewind', None), ('', None)):
            with self.assertRaises(ValueError):
                state.control(action, position)
        self.assertIs(state.capture_stop, generation)
        self.assertEqual(state.session_id, session)

    def test_recorded_evidence_uses_its_own_stale_limit_and_is_held_while_paused(self):
        clock = [100.]
        state = self.state(clock=lambda: clock[0])
        rec = self.recording.recording_id
        state.select(rec)
        state.publish({'camera_id': rec, 'status': 'processing', 'published_at': 100., 'frame': 'pixels'})
        clock[0] = 112.  # Twelve seconds: stale for a camera, not for slow CPU analysis.
        snapshot = state.snapshot()
        self.assertEqual((snapshot['frame_age_seconds'], snapshot['stale_after_seconds']), (12., 30))
        self.assertIn('frame', snapshot)
        state.control('pause')
        clock[0] = 150.
        state.heartbeat(rec)
        held = state.snapshot()
        self.assertEqual((held['status'], held['frame_age_seconds']), ('paused', 0.))
        self.assertIn('frame', held)
        state.control('resume')
        stale = state.snapshot()
        self.assertEqual(stale['status'], 'processing')
        self.assertNotIn('frame', stale)
        clock[0] = 211.  # Idle expiry still clears everything.
        state.expire()
        self.assertIsNone(state.camera_id)
        self.assertNotIn('frame', state.snapshot())

    def test_live_stale_limit_is_unchanged(self):
        clock = [10.]
        state = self.state(clock=lambda: clock[0])
        state.select('cam01')
        state.publish({'camera_id': 'cam01', 'status': 'live', 'published_at': 10., 'frame': 'pixels'})
        clock[0] = 16.
        snapshot = state.snapshot()
        self.assertEqual((snapshot['status'], snapshot['stale_after_seconds']), ('unavailable', 5))
        self.assertNotIn('frame', snapshot)

    def test_completion_applies_only_to_the_run_that_reached_the_end(self):
        state = self.state()
        rec = self.recording.recording_id
        state.select(rec)
        old = state.capture_stop
        state.control('restart')
        state.complete(old)
        self.assertEqual(state.snapshot()['status'], 'connecting')
        state.complete(state.capture_stop)
        self.assertEqual(state.snapshot()['status'], 'completed')
        state.control('pause')  # Nothing to pause once finished.
        self.assertEqual(state.snapshot()['status'], 'completed')


class RecordedContractTests(RecordedCase):
    def run_for(self, session='s-20261010T101500-aaaaaaaa', recording=None):
        return RecordedRun(recording or self.recording, session, paced=False)

    def test_recorded_wire_event_needs_session_pts_and_time_basis(self):
        base = dict(edge_id='e', camera_id='rec-1', plate=TEXT, hmac_key='k', confidence=.9)
        ok = sighting(**base, source_mode='recorded', stream_pts_ms=500, session_id='s-1', time_basis='processing_time')
        self.assertEqual((ok['source_mode'], ok['session_id'], ok['time_basis']), ('recorded', 's-1', 'processing_time'))
        for missing in (dict(stream_pts_ms=500, session_id='s-1'), dict(stream_pts_ms=500, time_basis='processing_time'),
                        dict(session_id='s-1', time_basis='processing_time'),
                        dict(stream_pts_ms=500, session_id='s-1', time_basis='arrival')):
            with self.assertRaises(ValueError):
                sighting(**base, source_mode='recorded', **missing)
        live = sighting(**base, source_mode='live', stream_pts_ms=500)
        self.assertNotIn('session_id', live)  # Existing live payload shape is untouched.
        self.assertNotIn('processed_at', live)
        self.assertTrue(ok['processed_at'].endswith('Z'))
        with self.assertRaises(ValueError):
            sighting(**base, source_mode='live', session_id='s-1')
        with self.assertRaises(ValueError):
            sighting(**base, source_mode='downloaded')

    def test_delivery_is_masked_labelled_and_deduplicated_on_video_time(self):
        delivery = SharedDelivery(self.config, 'general')
        candidate, rec, run = SimpleNamespace(plate=TEXT, score=.99), self.recording.recording_id, self.run_for()
        self.assertTrue(delivery.accept(rec, '7', candidate, 1000, recorded=run))
        self.assertFalse(delivery.accept(rec, '8', candidate, 30_000, recorded=run))  # Same pass, tracker reset.
        self.assertTrue(delivery.accept(rec, '9', candidate, 95_000, recorded=run))   # A later, separate pass.
        self.assertTrue(delivery.accept(rec, '7', SimpleNamespace(plate='MH02EK0837', score=.9), 1000, recorded=run))
        batch = [payload for _, _, payload in delivery.outbox.next_batch()]
        self.assertEqual(len(batch), 3)
        self.assertNotIn(TEXT, json.dumps(batch))
        first = batch[0]
        self.assertEqual((first['source_mode'], first['camera_id'], first['stream_pts_ms']), ('recorded', rec, 1000))
        self.assertEqual((first['session_id'], first['time_basis']), (run.session_id, 'processing_time'))
        self.assertEqual(first['masked_plate'], 'AP******34')
        self.assertTrue(first['model_version'].startswith('recorded;'))
        self.assertLessEqual(len(first['model_version']), 100)
        self.assertNotIn(str(self.root), json.dumps(batch))

    def test_retransmission_is_idempotent_and_a_new_session_is_traceable(self):
        candidate, rec = SimpleNamespace(plate=TEXT, score=.99), self.recording.recording_id
        first = SharedDelivery(self.config, 'general')
        first.accept(rec, '7', candidate, 1000, recorded=self.run_for())
        # A restarted worker replays the same run position: same event, not a new one.
        again = SharedDelivery(self.config, 'general')
        self.assertTrue(again.accept(rec, '7', candidate, 1000, recorded=self.run_for()))
        self.assertEqual(again.depth(), 1)
        again.accept(rec, '7', candidate, 1000, recorded=self.run_for('s-20261010T103000-bbbbbbbb'))
        events = [payload for _, _, payload in again.outbox.next_batch()]
        self.assertEqual(len(events), 2)
        self.assertEqual(len({event['event_id'] for event in events}), 2)
        self.assertEqual({event['session_id'] for event in events},
                         {'s-20261010T101500-aaaaaaaa', 's-20261010T103000-bbbbbbbb'})
        self.assertEqual(len({event['plate_hmac'] for event in events}), 1)

    def test_recording_time_is_used_only_when_its_receipt_states_it(self):
        other = make_video(self.root/'recordings'/'timed.mp4', frames=12)
        register(other, recorded_start_utc='2026-09-01T06:30:00Z')
        timed = next(item for item in RecordingLibrary(self.root/'recordings').scan()[0].values()
                     if item.started_at is not None)
        delivery = SharedDelivery(self.config, 'general')
        candidate = SimpleNamespace(plate=TEXT, score=.99)
        delivery.accept(timed.recording_id, '1', candidate, 1500, recorded=self.run_for(recording=timed))
        delivery.accept(self.recording.recording_id, '1', candidate, 1500, recorded=self.run_for())
        stated, unknown = (payload for _, _, payload in delivery.outbox.next_batch())
        self.assertEqual((stated['observed_at'], stated['time_basis']), ('2026-09-01T06:30:01.500000Z', 'source_clock'))
        # The recording's clock does not replace the time this machine analysed it.
        self.assertTrue(stated['processed_at'].endswith('Z'))
        self.assertNotEqual(stated['processed_at'][:10], '2026-09-01')
        self.assertEqual(unknown['time_basis'], 'processing_time')
        # Unknown recording time: a real UTC processing time, explicitly labelled as such.
        self.assertTrue(unknown['observed_at'].endswith('Z'))
        self.assertNotEqual(unknown['observed_at'][:10], '2026-09-01')

    def test_recorded_event_is_accepted_by_the_central_schema(self):
        backend = Path(__file__).resolve().parents[2]/'backend'
        sys.path.insert(0, str(backend))
        try:
            from app.schemas import SightingIn
        except ImportError:
            self.skipTest('Central API dependencies are not installed in this environment')
        finally:
            sys.path.remove(str(backend))
        delivery = SharedDelivery(self.config, 'general')
        delivery.accept(self.recording.recording_id, '7', SimpleNamespace(plate=TEXT, score=.99), 1000,
                        recorded=self.run_for())
        delivery.accept('cam01', '7', SimpleNamespace(plate=TEXT, score=.99), 1000)
        for _, _, payload in delivery.outbox.next_batch():
            SightingIn.model_validate(payload)

    def test_operator_view_carries_recorded_labels_but_no_media_or_paths(self):
        snapshot = {'status': 'processing', 'camera_id': self.recording.recording_id, 'source_mode': 'recorded',
                    'session_id': 's-1', 'recording_name': 'Test junction', 'video_position_ms': 1500,
                    'video_duration_ms': 6000, 'time_basis': 'processing_time', 'pacing': 'unpaced',
                    'stale_after_seconds': 30, 'frame': 'private image', 'path': str(self.recording.path),
                    'debug': {'tracks': [{'track_id': '7', 'observations': 3, 'masked_plate': 'AP******34',
                                          'recognition_state': 'model_accepted', 'candidates': [{'plate': TEXT}]}]}}
        safe = operator_snapshot(snapshot)
        self.assertEqual(safe['source_mode'], 'recorded')
        self.assertEqual((safe['session_id'], safe['video_position_ms'], safe['recording_name']),
                         ('s-1', 1500, 'Test junction'))
        self.assertEqual(safe['recognition_state'], 'model_accepted')
        text = json.dumps(safe)
        for secret in (TEXT, 'private image', str(self.root), 'path'):
            self.assertNotIn(secret, text)
        self.assertEqual(operator_snapshot(dict(snapshot, status='completed'))['recognition_state'], 'completed')
        live = operator_snapshot({'status': 'live', 'camera_id': 'cam01', 'session_id': 'leak', 'recording_name': 'x'})
        self.assertEqual(live['source_mode'], 'live')
        self.assertNotIn('session_id', live)
        self.assertNotIn('recording_name', live)


class RecordedPipelineTests(RecordedCase):
    def test_file_to_outbox_with_a_slow_mocked_recogniser_loses_no_sample(self):
        state = LiveState()
        state.authorise_recordings(self.recordings)
        state.select(self.recording.recording_id)
        _, stop, recording, session, start = state.run_parameters()
        published = []
        original = state.publish
        state.publish = lambda value, capture_stop=None: (published.append(value), original(value, capture_stop))[1]
        delivery = SharedDelivery(self.config, 'general')
        model, detector, vehicle = fakes(ocr_delay=.05)
        run = RecordedRun(recording, session, start, paced=False, paused=state.paused)
        summary = observe_tracked(self.config, recording, model, detector, vehicle, state, stop,
                                  debug=True, delivery=delivery, detector_profile='general', recorded=run)
        # 6 s of video sampled every 500 ms of video time: all twelve samples analysed.
        self.assertEqual(summary['analysed_frames'], 12)
        self.assertEqual([item['video_position_ms'] for item in published], [i*500 for i in range(12)])
        self.assertEqual({item['status'] for item in published}, {'processing'})
        self.assertEqual({item['pacing'] for item in published}, {'unpaced'})
        self.assertEqual(published[-1]['debug']['dropped_sampled_frames'], 0)
        self.assertEqual(published[-1]['debug']['collector_input_drops'], 0)
        self.assertEqual(published[-1]['decoded_frames'], 60)
        state.complete(stop)
        final = state.snapshot(engineering=True)
        self.assertEqual((final['status'], final['source_mode'], final['session_id']), ('completed', 'recorded', session))
        self.assertIn('frame', final)  # The last analysed position stays inspectable until idle expiry.
        self.assertEqual(state.operator_snapshot()['recognition_state'], 'completed')
        events = [payload for _, _, payload in delivery.outbox.next_batch()]
        self.assertEqual(len(events), 1)  # One accepted identifier, one event for this session.
        self.assertEqual((events[0]['source_mode'], events[0]['session_id']), ('recorded', session))
        self.assertEqual(events[0]['masked_plate'], 'AP******34')
        self.assertNotIn(TEXT, json.dumps(events))

    def test_pause_stops_analysis_at_the_sample_in_hand_and_resume_loses_nothing(self):
        import threading
        state = LiveState()
        state.authorise_recordings(self.recordings)
        state.select(self.recording.recording_id)
        _, stop, recording, session, start = state.run_parameters()
        positions, original = [], state.publish
        def publish(value, capture_stop=None):
            positions.append(value['video_position_ms'])
            if len(positions) == 3:
                state.control('pause')  # Queued frames are already decoded at this point.
            return original(value, capture_stop)
        state.publish = publish
        model, detector, vehicle = fakes(ocr_delay=.03)
        worker = threading.Thread(target=observe_tracked, args=(self.config, recording, model, detector, vehicle, state, stop),
                                  kwargs=dict(detector_profile='general',
                                              recorded=RecordedRun(recording, session, start, paced=False, paused=state.paused)))
        worker.start()
        time.sleep(1.2)  # Long enough to analyse the whole clip if pause did not hold.
        self.assertEqual(positions, [0, 500, 1000])
        self.assertEqual(state.snapshot()['status'], 'paused')
        self.assertTrue(worker.is_alive())
        state.control('resume')
        worker.join(timeout=30)
        self.assertFalse(worker.is_alive())
        self.assertEqual(positions, [i*500 for i in range(12)])

    def test_seek_run_starts_at_the_requested_position(self):
        state = LiveState()
        state.authorise_recordings(self.recordings)
        state.select(self.recording.recording_id)
        state.control('seek', 4000)
        _, stop, recording, session, start = state.run_parameters()
        positions = []
        original = state.publish
        state.publish = lambda value, capture_stop=None: (positions.append(value['video_position_ms']), original(value, capture_stop))[1]
        model, detector, vehicle = fakes()
        observe_tracked(self.config, recording, model, detector, vehicle, state, stop, detector_profile='general',
                        recorded=RecordedRun(recording, session, start, paced=False, paused=state.paused))
        self.assertEqual(positions, [4000, 4500, 5000, 5500])

    def test_decoder_failure_is_an_error_but_end_of_file_is_not(self):
        state = LiveState()
        state.authorise_recordings(self.recordings)
        state.select(self.recording.recording_id)
        _, stop, recording, session, start = state.run_parameters()
        with recording.path.open('ab') as stream:
            stream.write(b'tampered after the receipt was written')
        model, detector, vehicle = fakes()
        with self.assertRaises(RuntimeError) as raised:
            observe_tracked(self.config, recording, model, detector, vehicle, state, stop, detector_profile='general',
                            recorded=RecordedRun(recording, session, start, paced=False, paused=state.paused))
        self.assertEqual(raised.exception.failure_code, 'ChangedRecording')


class RecordedServiceTests(RecordedCase):
    """The HTTP service end to end, still with the mocked recogniser."""
    def setUp(self):
        super().setUp()
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); self.port = sock.getsockname()[1]
        patcher = patch('sentinel_edge.live_detection_viewer.load_models', return_value=fakes(ocr_delay=.02))
        patcher.start(); self.addCleanup(patcher.stop)
        self.thread = threading.Thread(target=serve, args=(self.config, None, None), daemon=True, kwargs=dict(
            port=self.port, lifetime=60, operator_key='private-test-key', detector_profile='general',
            recordings_dir=self.root/'recordings', live_catalogue=False, paced=False, device='cpu'))
        self.thread.start()
        for _ in range(100):
            try:
                self.connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=5)
                self.connection.request('GET', '/'); page = self.connection.getresponse().read().decode(); break
            except OSError:
                time.sleep(.02)
        token = re.search(r'const viewerToken="([^"]+)"', page)[1]
        self.headers = {'X-Viewer-Token': token, 'X-Operator-Key': 'private-test-key',
                        'Host': f'127.0.0.1:{self.port}', 'Origin': f'http://127.0.0.1:{self.port}'}
        self.edge = {'X-Edge-Key': 'test-edge-key'}

    def call(self, method, path, body=None, headers=None):
        headers = dict(self.headers if headers is None else headers)
        payload = None
        if body is not None:
            payload = json.dumps(body); headers['Content-Type'] = 'application/json'
        self.connection.request(method, path, body=payload, headers=headers)
        response = self.connection.getresponse()
        return response.status, json.loads(response.read() or b'{}')

    def wait_for(self, predicate, seconds=30):
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            status, snapshot = self.call('GET', '/engineering')
            if status == 200 and predicate(snapshot):
                return snapshot
            time.sleep(.1)
        self.fail('Recorded session did not reach the expected state')

    def test_select_complete_restart_seek_and_stop_over_http(self):
        rec = self.recording.recording_id
        deadline = time.monotonic()+10
        while time.monotonic() < deadline:
            status, catalogue = self.call('GET', '/cameras')
            if catalogue.get('recordings'):
                break
            time.sleep(.1)
        self.assertEqual(catalogue['cameras'], [])
        self.assertEqual(catalogue['recordings'], [{'id': rec, 'name': 'Test junction', 'duration_ms': 6000,
                                                    'time_basis': 'processing_time'}])
        self.assertEqual(self.call('POST', '/control', {'action': 'pause'})[0], 409)  # Nothing selected yet.
        self.assertEqual(self.call('POST', '/session', {'camera_id': 'cam01'})[0], 400)  # No cameras in this mode.
        self.assertEqual(self.call('POST', '/session', {'camera_id': rec})[0], 200)
        done = self.wait_for(lambda s: s['status'] == 'completed')
        self.assertEqual((done['source_mode'], done['analysed_frames'], done['video_position_ms']), ('recorded', 12, 5500))
        self.assertEqual(done['models']['ocr_device'], 'cpu')
        first_session = done['session_id']

        status, operator = self.call('GET', '/internal/status', headers=self.edge)
        self.assertEqual(status, 200)
        self.assertEqual((operator['source_mode'], operator['status'], operator['inference_device']),
                         ('recorded', 'completed', 'cpu'))
        self.assertEqual(operator['session_id'], first_session)
        self.assertNotIn('frame', operator); self.assertNotIn('debug', operator)
        self.assertNotIn(TEXT, json.dumps(operator))
        self.assertEqual(operator['delivery'], {'enabled': False, 'outbox_depth': 0})

        for body in ({'action': 'seek', 'position_ms': 999_999}, {'action': 'rewind'}, {'action': 'seek'}):
            self.assertEqual(self.call('POST', '/control', body)[0], 409)
        for body in ({'position_ms': 5}, {'action': 5}, {'action': 'pause', 'extra': 1}):
            self.assertEqual(self.call('POST', '/control', body)[0], 400)
        self.assertEqual(self.call('POST', '/control', {'action': 'restart'}, headers={})[0], 403)

        status, operator = self.call('POST', '/internal/control', {'action': 'seek', 'position_ms': 4000}, headers=self.edge)
        self.assertEqual(status, 200)
        self.assertNotEqual(operator['session_id'], first_session)
        again = self.wait_for(lambda s: s['status'] == 'completed' and s['session_id'] != first_session)
        self.assertEqual(again['analysed_frames'], 4)

        self.assertEqual(self.call('POST', '/session', {'camera_id': None})[0], 200)
        stopped = self.wait_for(lambda s: s['status'] == 'paused')
        self.assertIsNone(stopped['camera_id'])
        self.assertNotIn('frame', stopped)
        self.assertEqual(stopped['source_mode'], 'live')  # No recording selected: nothing is labelled recorded.

    def test_a_changed_file_is_reported_by_code_and_not_retried_in_a_loop(self):
        rec = self.recording.recording_id
        self.wait_for(lambda s: True)
        deadline = time.monotonic()+10
        while time.monotonic() < deadline and not self.call('GET', '/cameras')[1].get('recordings'):
            time.sleep(.1)
        with self.recording.path.open('ab') as stream:
            stream.write(b'tampered')
        self.assertEqual(self.call('POST', '/session', {'camera_id': rec})[0], 200)
        failed = self.wait_for(lambda s: s['status'] == 'unavailable')
        self.assertEqual(failed['failure_code'], 'ChangedRecording')
        self.assertEqual(failed['source_mode'], 'recorded')
        # The operator view must be told why, by code only.
        operator = self.call('GET', '/internal/status', headers=self.edge)[1]
        self.assertEqual((operator['status'], operator['failure_code']), ('unavailable', 'ChangedRecording'))
        self.assertNotIn(str(self.root), json.dumps(failed))

    def test_absent_or_changed_models_are_reported_visibly_and_never_downloaded(self):
        deadline = time.monotonic()+10
        while time.monotonic() < deadline and not self.call('GET', '/cameras')[1].get('recordings'):
            time.sleep(.1)
        missing = ValueError('Diagnostic model missing or changed; no automatic download')
        with patch('sentinel_edge.live_detection_viewer.load_models', side_effect=missing) as loader:
            self.assertEqual(self.call('POST', '/session', {'camera_id': self.recording.recording_id})[0], 200)
            failed = self.wait_for(lambda s: s['status'] == 'unavailable')
            self.assertEqual((failed['failure_code'], failed['source_mode']), ('ValueError', 'recorded'))
            self.assertNotIn('frame', failed)
            self.assertNotIn('no automatic download', json.dumps(failed))  # A code, never the message.
            time.sleep(1.5)
            self.assertEqual(loader.call_count, 1)  # Held for the operator, not retried in a loop.

    def tearDown(self):
        try:
            self.call('POST', '/session', {'camera_id': None})
            self.connection.close()
        except OSError:
            pass
