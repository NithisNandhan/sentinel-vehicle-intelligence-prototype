"""Synthetic RAM-only retention across recorded tracker history expiry."""
import base64
import json
from pathlib import Path
import unittest

from sentinel_edge.live_detection_viewer import LiveState
from sentinel_edge.recorded_source import Recording
from sentinel_edge.unified_diagnostics import DiagnosticUnavailable, diagnostic_view


PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII='


def evidence(track_id='204', state='model_accepted'):
    return dict(track_id=track_id, vehicle_box=[0, 0, 100, 100], vehicle_class=2,
        seen_pts_ms=500, observations=3, recognition_state=state,
        masked_plate='GJ******34', best_guess=dict(plate='GJ01AB1234', score=.91),
        candidates=[dict(plate='GJ01AB1234', score=.91, observations=3)],
        gate_reasons=['Model gates passed; correctness is not independently verified'],
        frames=[dict(pts_ms=500, width=1, height=1, quality=.8, text='GJ01AB1234', image=PNG,
                     working_view='text_quad', working_image=PNG, crop_sources=['text'])],
        reconstruction=dict(status='not_run', reason='No reconstruction'),
        source_url='rtsp://never-retain.invalid/private', secret='do-not-retain')


class RecordedDiagnosticRetentionTests(unittest.TestCase):
    def test_same_tracker_and_reading_cannot_reuse_another_passage_evidence(self):
        state = self.state()
        first = evidence()
        first['passage_id'] = 'aaaaaaaaaaaaaaaa'
        old = dict(key='0-p-aaaaaaaaaaaaaaaa', track_id='204', passage_id='aaaaaaaaaaaaaaaa',
                   state='model_accepted', reading='GJ01AB1234')
        self.publish(state, [first], passed=[old])
        new = dict(old, key='0-p-bbbbbbbbbbbbbbbb', passage_id='bbbbbbbbbbbbbbbb')
        self.publish(state, [], passed=[new, old], video_position_ms=900)
        self.assert_unavailable(state, '204')
        # Equal OCR text and state cannot authorize an old passage's image.
        self.publish(state, [first], passed=[new, old], video_position_ms=900)
        self.assert_unavailable(state, '204')
        second = evidence()
        second.update(passage_id='bbbbbbbbbbbbbbbb', frames=[], seen_pts_ms=900)
        self.publish(state, [second], passed=[new, old], video_position_ms=900)
        view = self.view(state)
        self.assertEqual(view['track']['passage_id'], 'bbbbbbbbbbbbbbbb')
        self.assertEqual(view['track']['frames'], [])

    def test_selected_uncertain_window_stays_coherent_when_current_guess_weakens(self):
        state = self.state()
        passed = [dict(key='0-204', track_id='204', state='uncertain', reading='GJ01AB1234')]
        self.publish(state, [evidence(state='uncertain')], passed=passed)
        weaker = evidence(state='uncertain')
        weaker['best_guess']['plate'] = 'GJ01AB1235'
        weaker['frames'][0]['text'] = 'GJ01AB1235'
        self.publish(state, [weaker], passed=passed, video_position_ms=900)
        view = self.view(state)
        self.assertEqual(view['video_position_ms'], 500)
        self.assertEqual(view['track']['best_reading'], 'GJ01AB1234')
        self.assertEqual(view['track']['frames'][0]['text'], 'GJ01AB1234')
        self.assertIn('Archived recorded evidence', view['track']['gate_reasons'][-1])
        passed[0]['reading'] = 'GJ01AB1235'
        self.publish(state, [weaker], passed=passed, video_position_ms=900)
        self.assertEqual(self.view(state)['track']['best_reading'], 'GJ01AB1235')

    def test_evicted_stronger_window_does_not_borrow_current_weaker_evidence(self):
        state = self.state()
        passed = [dict(key='0-204', track_id='204', state='uncertain', reading='GJ01AB1234')]
        self.publish(state, [evidence(state='uncertain')], passed=passed)
        state._clear_diagnostics()
        weaker = evidence(state='uncertain')
        weaker['best_guess']['plate'] = 'GJ01AB1235'
        self.publish(state, [weaker], passed=passed, video_position_ms=900)
        self.assert_unavailable(state, '204')

    def setUp(self):
        self.now = [1000.]

    def state(self, **kwargs):
        state = LiveState(lambda: self.now[0], recorded_diagnostics=True, **kwargs)
        recording = Recording('rec-test', 'Synthetic permitted recording', Path('synthetic-only.mp4'),
                              '0' * 64, duration_ms=1000)
        state.authorise_recordings({'rec-test': recording})
        state.select('rec-test')
        return state

    def publish(self, state, tracks, *, passed=None, **extra):
        value = dict(camera_id=state.camera_id, status='processing', published_at=self.now[0],
                     video_position_ms=500, debug=dict(tracks=tracks, stage_ms={'ocr': 12., 'secret': 3.}))
        if passed is not None:
            value['debug']['passed'] = passed
        value.update(extra)
        state.publish(value, state.capture_stop)
        return value

    def view(self, state, track_id='204', session=None):
        return diagnostic_view(state.snapshot(engineering=True), {'ocr_model': 'synthetic-test'},
                               session or state.session_id, track_id, enabled=True)

    def assert_unavailable(self, state, track_id, status=404, session=None):
        with self.assertRaises(DiagnosticUnavailable) as raised:
            self.view(state, track_id, session)
        self.assertEqual(raised.exception.status, status)

    def test_expired_track_remains_inspectable_after_eof_with_native_pixels(self):
        state = self.state()
        self.publish(state, [evidence()])
        self.publish(state, [], video_position_ms=1000)
        state.complete(state.capture_stop)
        self.now[0] += 100
        view = self.view(state)
        self.assertEqual(view['video_position_ms'], 500)
        self.assertEqual(view['track']['recognition_state'], 'model_accepted')
        self.assertEqual(view['track']['best_reading'], 'GJ01AB1234')
        self.assertEqual(view['track']['frames'][0]['image'], PNG)
        self.assertEqual(base64.b64decode(view['track']['frames'][0]['image']), base64.b64decode(PNG))
        self.assertEqual(view['track']['frames'][0]['working_image'], PNG)
        self.assertEqual(view['stage_ms'], {'ocr': 12.})
        self.assertNotIn('never-retain', json.dumps(view))
        self.assertNotIn('do-not-retain', repr(state._retained_diagnostics))

    def test_stop_restart_seek_and_new_source_invalidate_session_cache(self):
        for action in ('stop', 'restart', 'seek', 'new_source'):
            with self.subTest(action=action):
                state = self.state()
                self.publish(state, [evidence()])
                session = state.session_id
                if action == 'stop':
                    state.select(None)
                elif action == 'restart':
                    state.control('restart')
                elif action == 'seek':
                    state.control('seek', 100)
                else:
                    state.select('cam01')
                self.assertEqual(len(state._retained_diagnostics), 0)
                self.assertEqual(state._retained_diagnostic_bytes, 0)
                self.assert_unavailable(state, '204', 409 if action in ('restart', 'seek') else 404, session)

    def test_live_and_disabled_states_never_retain_recorded_evidence(self):
        for enabled in (True, False):
            state = LiveState(lambda: self.now[0], recorded_diagnostics=enabled)
            state.select('cam01')
            self.publish(state, [evidence()])
            self.assertFalse(state._retained_diagnostics)
            self.assertNotIn('_retained_recorded_diagnostics', state.snapshot(engineering=True))
            self.assert_unavailable(state, '204', session='s-live')
        state = self.state()
        state.recorded_diagnostics = False
        self.publish(state, [evidence()])
        self.publish(state, [])
        self.assertFalse(state._retained_diagnostics)
        self.assert_unavailable(state, '204')

    def test_track_budget_evicts_old_nonaccepted_rows_and_prioritizes_accepted(self):
        state = self.state(diagnostic_max_tracks=2)
        for identifier, status in [('204', 'model_accepted'), ('243', 'uncertain'), ('280', 'uncertain')]:
            self.publish(state, [evidence(identifier, status)])
        self.publish(state, [])
        self.assertEqual(set(state._retained_diagnostics), {'204', '280'})
        self.assert_unavailable(state, '243')
        self.assertEqual(self.view(state, '204')['track']['recognition_state'], 'model_accepted')

    def test_byte_budget_and_oversize_entry_eviction_are_explicit(self):
        state = self.state()
        self.publish(state, [evidence()])
        size = state._retained_diagnostic_bytes
        bounded = self.state(diagnostic_max_bytes=size + 1)
        self.publish(bounded, [evidence('204', 'uncertain')])
        self.publish(bounded, [evidence('243', 'uncertain')])
        self.publish(bounded, [])
        self.assertLessEqual(bounded._retained_diagnostic_bytes, size + 1)
        self.assertLessEqual(len(bounded._retained_diagnostics), 1)
        self.assert_unavailable(bounded, '204')
        tiny = self.state(diagnostic_max_bytes=100)
        self.publish(tiny, [evidence()])
        self.publish(tiny, [])
        self.assertEqual(tiny._retained_diagnostic_bytes, 0)
        self.assert_unavailable(tiny, '204')

    def test_old_generation_publish_cannot_repopulate_after_seek(self):
        state = self.state()
        old = state.capture_stop
        value = self.publish(state, [evidence()])
        state.control('seek', 100)
        state.publish(value, old)
        self.assertFalse(state._retained_diagnostics)
        self.assertEqual(state.snapshot()['status'], 'connecting')

    def test_replacing_recording_under_same_id_starts_new_generation(self):
        state = self.state()
        self.publish(state, [evidence()])
        previous = state.session_id
        replacement = Recording('rec-test', 'Different recording', Path('different-synthetic.mp4'),
                                '1' * 64, duration_ms=1000)
        state.authorise_recordings({'rec-test': replacement})
        self.assertNotEqual(state.session_id, previous)
        self.assertFalse(state._retained_diagnostics)
        self.assert_unavailable(state, '204', 409, previous)

    def test_cache_owns_copies_and_preserves_latest_true_recognition_state(self):
        state = self.state()
        value = self.publish(state, [evidence(state='uncertain')])
        value['debug']['tracks'][0]['frames'][0]['image'] = 'mutated-producer'
        latest = evidence()
        latest['best_guess']['plate'] = 'GJ01AB1235'
        self.publish(state, [latest])
        self.publish(state, [])
        view = self.view(state)
        self.assertEqual(view['track']['best_reading'], 'GJ01AB1235')
        self.assertEqual(view['track']['recognition_state'], 'model_accepted')
        view['track']['frames'][0]['image'] = 'mutated-consumer'
        self.assertEqual(self.view(state)['track']['frames'][0]['image'], PNG)

    def test_identical_metadata_keeps_richer_whole_snapshot_but_changed_reading_does_not(self):
        state = self.state()
        self.publish(state, [evidence()])
        thinner = evidence()
        thinner.update(frames=[], seen_pts_ms=900)
        self.publish(state, [thinner], video_position_ms=900)
        self.publish(state, [], video_position_ms=1000)
        retained = self.view(state)
        self.assertEqual(retained['video_position_ms'], 500)
        self.assertEqual(retained['track']['seen_pts_ms'], 500)
        self.assertEqual(retained['track']['frames'][0]['image'], PNG)
        thinner['best_guess']['plate'] = 'GJ01AB1235'
        self.publish(state, [thinner], video_position_ms=900)
        self.publish(state, [], video_position_ms=1000)
        changed = self.view(state)
        self.assertEqual(changed['video_position_ms'], 900)
        self.assertEqual(changed['track']['best_reading'], 'GJ01AB1235')
        self.assertEqual(changed['track']['frames'], [])

    def test_last_accepted_snapshot_matches_running_row_when_rolling_window_degrades(self):
        state = self.state()
        passed = [dict(key='0-204', track_id='204', state='model_accepted', reading='GJ01AB1234',
                       thumb='NOT-A-NATIVE-CROP')]
        self.publish(state, [evidence()], passed=passed)
        degraded = evidence(state='unreadable')
        degraded.update(best_guess=None, candidates=[], frames=[], observations=0, seen_pts_ms=900)
        self.publish(state, [degraded], passed=passed, video_position_ms=900)
        # An active empty row must not hide the coherent acceptance still shown
        # in the running list. No thumbnails or fabricated observations escape.
        view = self.view(state)
        self.assertEqual(view['video_position_ms'], 500)
        self.assertEqual(view['track']['seen_pts_ms'], 500)
        self.assertEqual(view['track']['observations'], 3)
        self.assertEqual(view['track']['recognition_state'], 'model_accepted')
        self.assertEqual(view['track']['best_reading'], passed[0]['reading'])
        self.assertEqual(view['track']['frames'][0]['image'], PNG)
        self.assertIn('Archived recorded evidence', view['track']['gate_reasons'][-1])
        self.assertNotIn('NOT-A-NATIVE-CROP', json.dumps(view))
        self.publish(state, [], passed=passed, video_position_ms=1000)
        state.complete(state.capture_stop)
        self.assertEqual(self.view(state)['track']['best_reading'], 'GJ01AB1234')

    def test_newest_passed_scene_identity_or_changed_reading_cannot_reveal_old_acceptance(self):
        for newest in (dict(key='1-204', track_id='204', state='unreadable', reading=None),
                       dict(key='0-204', track_id='204', state='model_accepted', reading='GJ01AB1235'),
                       dict(key='0-204', track_id='204', state='unreadable', reading=None)):
            with self.subTest(newest=newest):
                state = self.state()
                original = dict(key='0-204', track_id='204', state='model_accepted', reading='GJ01AB1234')
                self.publish(state, [evidence()], passed=[original])
                # Newest-first: the old same-ID row must not authorize fallback.
                self.publish(state, [], passed=[newest, original], video_position_ms=900)
                self.assert_unavailable(state, '204')
                degraded = evidence(state='unreadable')
                degraded.update(best_guess=None, candidates=[], frames=[], observations=0)
                self.publish(state, [degraded], passed=[newest, original], video_position_ms=900)
                if newest['reading']:
                    # The selected new answer has no matching retained window;
                    # don't present a different active window as its evidence.
                    self.assert_unavailable(state, '204')
                    continue
                view = self.view(state)
                self.assertEqual(view['video_position_ms'], 900)
                self.assertEqual(view['track']['recognition_state'], 'unreadable')
                self.assertIsNone(view['track']['best_reading'])
                self.assertEqual(view['track']['frames'], [])

    def test_new_accepted_reading_replaces_prior_whole_snapshot_without_borrowing_images(self):
        state = self.state()
        old = dict(key='0-204', track_id='204', state='model_accepted', reading='GJ01AB1234')
        self.publish(state, [evidence()], passed=[old])
        newer = evidence()
        newer.update(frames=[], seen_pts_ms=900)
        newer['best_guess']['plate'] = 'GJ01AB1235'
        new_row = dict(old, reading='GJ01AB1235')
        self.publish(state, [newer], passed=[new_row], video_position_ms=900)
        self.publish(state, [], passed=[new_row], video_position_ms=1000)
        view = self.view(state)
        self.assertEqual(view['video_position_ms'], 900)
        self.assertEqual(view['track']['best_reading'], 'GJ01AB1235')
        self.assertEqual(view['track']['frames'], [])

    def test_retention_does_not_change_operator_tracks_or_acceptance(self):
        state = self.state()
        self.publish(state, [evidence()])
        self.publish(state, [], vehicles_seen=[dict(track_id='204', state='model_accepted')])
        view = state.operator_snapshot()
        self.assertEqual(view['tracks'], [])
        self.assertEqual(view['vehicles_seen'][0]['state'], 'model_accepted')
        self.assertNotIn('_retained_recorded_diagnostics', view)
        self.assertNotIn('_retained_recorded_diagnostics', state.snapshot())

    def test_active_processing_stale_evidence_and_disabled_cache_remain_hidden(self):
        state = self.state()
        self.publish(state, [evidence()])
        self.publish(state, [])
        self.now[0] += 31
        self.assertNotIn('_retained_recorded_diagnostics', state.snapshot(engineering=True))
        self.assert_unavailable(state, '204')
        state.recorded_diagnostics = False
        state.snapshot(engineering=True)
        self.assertEqual(state._retained_diagnostic_bytes, 0)


if __name__ == '__main__':
    unittest.main()
