"""Replay is historical, bounded, recorded-only and never an inference rerun."""
import json
import unittest
from sentinel_edge.recorded_replay import RecordedReplay, ReplayUnavailable


def frame(snapshot, *, mode):
    return dict(frame='QUJD' if mode == 'original' else 'REVG', vehicles=snapshot.get('test_vehicles', []))


class RecordedReplayTests(unittest.TestCase):
    def test_live_state_only_captures_opted_in_recordings_and_resets_after_seek(self):
        from pathlib import Path
        import numpy as np
        from sentinel_edge.live_detection_viewer import LiveState, encoded
        from sentinel_edge.recorded_source import Recording
        state = LiveState(recorded_diagnostics=True)
        recording = Recording('rec-one', 'Synthetic recording', Path('synthetic.mp4'), 'a'*64, duration_ms=1000)
        state.authorise_recordings({'rec-one': recording}); state.select('rec-one')
        image = encoded(np.zeros((16,32,3),dtype=np.uint8), '.jpg')
        state.publish(dict(camera_id='rec-one', status='processing', video_position_ms=0, frame=image,
            vehicles_seen=[], debug=dict(original_frame=image, tracks=[], passed=[])), state.capture_stop)
        session = state.session_id
        with self.assertRaises(ReplayUnavailable): state.replay_page(session)
        state.complete(state.capture_stop)
        self.assertEqual(state.replay_page(session)['sample_count'], 1)
        state.control('seek', 0)
        self.assertFalse(state.replay.samples)
        with self.assertRaises(ReplayUnavailable): state.replay_page(session)
        state.select(None)
        with self.assertRaises(ReplayUnavailable) as raised: state.replay_page(session)
        self.assertEqual(raised.exception.status, 404)
        live = LiveState(recorded_diagnostics=True); live.select('cam01')
        live.publish(dict(camera_id='cam01', status='live', frame=image), live.capture_stop)
        self.assertFalse(live.replay.samples)

    def test_completed_recording_survives_tab_inactivity_but_explicit_stop_clears_it(self):
        from pathlib import Path
        from sentinel_edge.live_detection_viewer import LiveState
        from sentinel_edge.recorded_source import Recording
        now = [0.]
        state = LiveState(lambda: now[0], recorded_diagnostics=True)
        state.authorise_recordings({'rec-one': Recording('rec-one', 'Synthetic', Path('synthetic.mp4'), 'a'*64, duration_ms=1000)})
        state.select('rec-one')
        self.append(state.replay, session=state.session_id)
        session = state.session_id
        state.complete(state.capture_stop)
        now[0] = 600.
        state.expire()
        self.assertEqual(state.camera_id, 'rec-one')
        self.assertEqual(state.replay_page(session)['sample_count'], 1)
        state.select(None)
        self.assertFalse(state.replay.samples)

    def append(self, store, position=0, **kwargs):
        snapshot = dict(video_position_ms=position, vehicles_seen=[], unique_model_accepted_unverified=0)
        snapshot.update(kwargs.pop('snapshot', {}))
        store.append(snapshot, session_id=kwargs.pop('session', 's-one'), camera_id='rec-one',
                     duration_ms=1000, frame_builder=kwargs.pop('builder', frame))
        return snapshot

    def page(self, store, **kwargs):
        return store.page(kwargs.pop('session', 's-one'), completed=kwargs.pop('completed', True), **kwargs)

    def test_pages_are_immutable_historical_evidence_not_final_readings(self):
        store = RecordedReplay()
        original = self.append(store, 0, snapshot={'test_vehicles': [dict(track_id='1', reading=None, secret='DROP')]})
        original['test_vehicles'][0]['reading'] = 'MUTATED'
        self.append(store, 500, snapshot={'test_vehicles': [dict(track_id='1', reading='GJ01AB1234')]})
        page = self.page(store)
        self.assertEqual((page['sample_count'], page['next_offset']), (2, 1))
        self.assertIsNone(page['samples'][0]['vehicles'][0]['reading'])
        self.assertNotIn('secret', json.dumps(page))
        page['samples'][0]['vehicles'][0]['reading'] = 'MUTATED'
        self.assertIsNone(self.page(store)['samples'][0]['vehicles'][0]['reading'])
        self.assertEqual(self.page(store, offset=1)['samples'][0]['vehicles'][0]['reading'], 'GJ01AB1234')
        self.assertIsNone(self.page(store, offset=1)['next_offset'])

    def test_completed_and_current_session_required(self):
        store = RecordedReplay(); self.append(store)
        for kwargs in ({'completed': False}, {'session': 's-old'}):
            with self.assertRaises(ReplayUnavailable) as raised: self.page(store, **kwargs)
            self.assertEqual(raised.exception.status, 409)
        self.assertEqual(self.page(store, mode='original')['samples'][0]['frame'], 'QUJD')
        self.assertEqual(self.page(store)['samples'][0]['frame'], 'REVG')

    def test_source_reset_invalidates_entire_replay_instead_of_sorting_history(self):
        store = RecordedReplay(); self.append(store, 500); self.append(store, 0)
        self.assertEqual(store.samples, [])
        with self.assertRaises(ReplayUnavailable) as raised: self.page(store)
        self.assertEqual(raised.exception.status, 503)

    def test_bounds_fail_closed_without_partial_recording(self):
        for store in (RecordedReplay(max_bytes=1), RecordedReplay(max_samples=1)):
            self.append(store); self.append(store, 500)
            self.assertEqual(store.samples, [])
            self.assertIsNotNone(store.reason)
            with self.assertRaises(ReplayUnavailable): self.page(store)

    def test_new_session_and_clear_remove_old_frames(self):
        store = RecordedReplay(); self.append(store)
        self.append(store, 500, session='s-two')
        with self.assertRaises(ReplayUnavailable): self.page(store)
        self.assertEqual(self.page(store, session='s-two')['sample_count'], 1)
        store.clear()
        self.assertFalse(store.samples)
        self.assertIsNone(store.session_id)

    def test_duplicate_sample_is_not_an_extra_observation_and_queries_are_bounded(self):
        store = RecordedReplay(); self.append(store); self.append(store)
        self.assertEqual(len(store.samples), 1)
        for query in ({'offset': -1}, {'offset': 2}, {'limit': 2}, {'mode': 'url'}, {'offset': True}):
            with self.assertRaises(ReplayUnavailable) as raised: self.page(store, **query)
            self.assertEqual(raised.exception.status, 400)

    def test_frame_failure_and_private_metadata_do_not_escape(self):
        store = RecordedReplay()
        self.append(store, snapshot={'secret': 'DROP', 'vehicles_seen': [dict(track_id='1', reading='DROP', secret='DROP')]})
        self.assertNotIn('DROP', json.dumps(self.page(store)))
        bad = RecordedReplay()
        self.append(bad, builder=lambda *a, **k: dict(frame='https://private.invalid/secret', vehicles=[]))
        self.assertFalse(bad.samples)
        self.assertNotIn('private', bad.reason)


if __name__ == '__main__': unittest.main()
