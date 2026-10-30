"""Synthetic-only contract/privacy tests for the unified worker adapter."""
import base64
import copy
import http.client
import json
import math
import socket
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sentinel_edge.live_detection_viewer import LiveState, serve
from sentinel_edge.unified_diagnostics import DiagnosticUnavailable, diagnostic_view

PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII='
SESSION = 's-20261030T100000-deadbeef'


def fixture():
    character = {'character': 'A', 'score': .9, 'alternatives': [{'character': 'A', 'score': .9}]}
    candidate = {'plate': 'GJ01AB1234', 'score': .9, 'observations': 3, 'log_score': -.5}
    return {'source_mode': 'recorded', 'session_id': SESSION, 'camera_id': 'rec-test', 'status': 'completed',
            'video_position_ms': 500, 'debug': {'stage_ms': {'ocr': 12., 'fusion': 1., 'secret': 1}, 'tracks': [{
                'track_id': 'det-1', 'vehicle_box': [0, 0, 100, 100], 'vehicle_class': 2,
                'seen_pts_ms': 500, 'observations': 3, 'recognition_state': 'model_accepted',
                'masked_plate': 'GJ******34', 'best_guess': candidate, 'candidates': [candidate],
                'gate_reasons': ['Model gates passed; correctness has not been independently verified'],
                'frames': [{'pts_ms': 500, 'width': 1, 'height': 1, 'text': 'GJ01AB1234', 'image': PNG,
                            'quality': .8, 'characters': [character]}],
                'reconstruction': {'status': 'reconstructed', 'reason': 'Derived evidence', 'image': PNG,
                                   'text': 'GJ01AB1234', 'width': 1, 'height': 1, 'contributing_originals': 3,
                                   'elapsed_ms': 3., 'characters': [character], 'candidates': [candidate]},
                'source_url': 'rtsp://secret.invalid/private'}]}}


class UnifiedDiagnosticTests(unittest.TestCase):
    def view(self, data=None, **kwargs):
        return diagnostic_view(data or fixture(), {'ocr_model': 'awiros-ctc-adapted', 'ocr_device': 'cpu',
                                                  'edge_api_key': 'DO-NOT-EXPOSE'}, SESSION, 'det-1', enabled=True, **kwargs)

    def test_exact_allowlisted_shape_native_pixels_and_best_guess(self):
        view = self.view()
        self.assertEqual(set(view), {'schema_version', 'source_mode', 'session_id', 'camera_id',
                                     'video_position_ms', 'models', 'stage_ms', 'track'})
        self.assertEqual(view['track']['best_reading'], 'GJ01AB1234')
        self.assertEqual(view['track']['model_score'], .9)
        self.assertEqual(base64.b64decode(view['track']['frames'][0]['image']), base64.b64decode(PNG))
        self.assertEqual(view['track']['reconstruction']['contributing_originals'], 3)
        self.assertEqual(view['stage_ms'], {'ocr': 12., 'fusion': 1.})
        self.assertNotIn('secret', json.dumps(view))
        self.assertNotIn('DO-NOT-EXPOSE', json.dumps(view))
        self.assertNotIn('source_url', json.dumps(view))

    def test_live_and_disabled_fail_closed_even_for_invalid_identifiers(self):
        for source, enabled in [('live', True), ('recorded', False)]:
            data = fixture(); data['source_mode'] = source
            with self.assertRaises(DiagnosticUnavailable) as raised:
                diagnostic_view(data, {}, 'bad/path', 'bad/path', enabled=enabled)
            self.assertEqual(raised.exception.status, 404)

    def test_session_mismatch_invalid_ids_and_eviction(self):
        for session, track, expected in [('s-other', 'det-1', 409), (SESSION, '../track', 400),
                                          (SESSION, 'missing', 404)]:
            with self.assertRaises(DiagnosticUnavailable) as raised:
                diagnostic_view(fixture(), {}, session, track, enabled=True)
            self.assertEqual(raised.exception.status, expected)

    def test_all_lists_are_bounded(self):
        data = fixture(); track = data['debug']['tracks'][0]
        track['frames'] = [copy.deepcopy(track['frames'][0]) for _ in range(40)]
        track['candidates'] = [copy.deepcopy(track['candidates'][0]) for _ in range(300)]
        for frame in track['frames']:
            frame['characters'] = [copy.deepcopy(frame['characters'][0]) for _ in range(30)]
            for char in frame['characters']:
                char['alternatives'] *= 5
        view = self.view(data)['track']
        self.assertEqual(len(view['frames']), 8)
        self.assertEqual(len(view['candidates']), 200)
        self.assertEqual(len(view['frames'][0]['characters']), 25)
        self.assertEqual(len(view['frames'][0]['characters'][0]['alternatives']), 3)

    def test_oversize_and_non_png_images_are_omitted_without_resizing(self):
        data = fixture(); track = data['debug']['tracks'][0]
        track['frames'][0]['image'] = 'A' * 80_001
        track['reconstruction']['image'] = base64.b64encode(b'<svg onload="alert(1)">').decode()
        view = self.view(data)['track']
        self.assertIsNone(view['frames'][0]['image'])
        self.assertIsNone(view['reconstruction']['image'])
        self.assertIn('not resized', view['gate_reasons'][-1])
        self.assertIn('not resized', view['reconstruction']['reason'])

    def test_nonfinite_values_and_unexpected_shapes_do_not_escape(self):
        data = fixture(); track = data['debug']['tracks'][0]
        track['best_guess']['score'] = math.nan
        track['recognition_state'] = {'unsafe': True}
        track['vehicle_class'] = {'unsafe': True}
        track['vehicle_box'] = [True, -1, 4, 5]
        data['debug']['stage_ms'] = {'ocr': math.inf, 'fusion': -1, 'vehicle': 4.}
        view = self.view(data)
        self.assertEqual(view['track']['recognition_state'], 'unreadable')
        self.assertEqual(view['track']['vehicle_class'], 'vehicle')
        self.assertEqual(view['track']['vehicle_box'], [])
        self.assertIsNone(view['track']['model_score'])
        self.assertEqual(view['stage_ms'], {'vehicle': 4.})
        json.dumps(view, allow_nan=False)


class AlignedEvidenceTests(unittest.TestCase):
    def test_original_and_overlay_share_source_pts_and_aspect_ratio(self):
        import cv2
        import numpy as np
        from sentinel_edge.live_detection_viewer import dashboard_view, encoded
        original = np.full((100, 200, 3), 40, np.uint8)
        overlay = original.copy()
        overlay[:, :40] = 220
        data = fixture()
        data['frame'] = encoded(overlay, '.jpg')
        data['debug']['original_frame'] = encoded(original, '.jpg')
        original_view = dashboard_view(data, width=100, mode='original')
        overlay_view = dashboard_view(data, width=100, mode='overlays')
        self.assertEqual(original_view['video_position_ms'], overlay_view['video_position_ms'])
        self.assertEqual(original_view['session_id'], overlay_view['session_id'])
        native = cv2.imdecode(np.frombuffer(base64.b64decode(original_view['frame']), np.uint8), cv2.IMREAD_COLOR)
        boxes = cv2.imdecode(np.frombuffer(base64.b64decode(overlay_view['frame']), np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(native.shape, (50, 100, 3))
        self.assertEqual(boxes.shape, native.shape)
        self.assertLess(float(native[:, :10].mean()), 60)
        self.assertGreater(float(boxes[:, :10].mean()), 180)
        self.assertEqual(dashboard_view(data)['frame'], dashboard_view(data, mode='overlays')['frame'])

    def test_missing_original_never_falls_back_to_mislabelled_overlays(self):
        from sentinel_edge.live_detection_viewer import dashboard_view
        data = fixture()
        data['frame'] = PNG
        self.assertIsNone(dashboard_view(data, mode='original')['frame'])


class UnifiedDiagnosticHttpTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture()
        outer = self

        class HeldState(LiveState):
            def snapshot(self, *, engineering=False):
                return copy.deepcopy(outer.fixture)

        self.state = HeldState()
        self.state_patch = patch('sentinel_edge.live_detection_viewer.LiveState', return_value=self.state)
        self.state_patch.start(); self.addCleanup(self.state_patch.stop)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); self.port = sock.getsockname()[1]
        self.thread = threading.Thread(target=serve, args=(SimpleNamespace(edge_api_key='edge-test'), None, None),
                                       kwargs={'port': self.port, 'lifetime': 2, 'live_catalogue': False,
                                               'dashboard_evidence': True}, daemon=True)
        self.thread.start()
        for _ in range(100):
            try:
                self.conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=3)
                self.conn.request('GET', '/'); response = self.conn.getresponse(); response.read(); break
            except OSError:
                time.sleep(.02)
        else:
            self.fail('Synthetic worker did not start')

    def call(self, path, headers=None, method='GET'):
        self.conn.request(method, path, headers=headers or {})
        response = self.conn.getresponse()
        raw = response.read()
        return response.status, response, json.loads(raw) if raw else {}

    def test_redirects_and_retired_browser_surfaces(self):
        for path in ('/', '/engineering', '/engineering?operator_key=never-reflect'):
            status, response, body = self.call(path)
            self.assertEqual(status, 307)
            self.assertEqual(response.getheader('Location'), 'http://127.0.0.1:5173/')
            self.assertEqual(body, {})
        for path in ('/snapshot', '/cameras'):
            self.assertEqual(self.call(path)[0], 410)
        self.assertEqual(self.call('/session', method='POST')[0], 410)
        self.assertEqual(self.call('/', {'Host': 'external.invalid'})[0], 403)

    def test_diagnostic_service_key_queries_and_context(self):
        path = f'/internal/diagnostics?session_id={SESSION}&track_id=det-1'
        self.assertEqual(self.call(path)[0], 403)
        self.assertEqual(self.call(path, {'X-Operator-Key': 'edge-test'})[0], 403)
        edge = {'X-Edge-Key': 'edge-test'}
        status, response, body = self.call(path, edge)
        self.assertEqual(status, 200)
        self.assertEqual(body['track']['best_reading'], 'GJ01AB1234')
        self.assertIn('no-store', response.getheader('Cache-Control'))
        self.assertEqual(self.call(path+'&track_id=det-1', edge)[0], 400)
        self.assertEqual(self.call('/internal/diagnostics?session_id=stale&track_id=det-1', edge)[0], 409)
        self.assertEqual(self.call(f'/internal/diagnostics?session_id={SESSION}&track_id=missing', edge)[0], 404)
        self.fixture['source_mode'] = 'live'
        self.assertEqual(self.call(path, edge)[0], 404)

    def test_bad_evidence_mode_is_rejected_before_image_processing(self):
        self.assertEqual(self.call('/internal/evidence?mode=wrong', {'X-Edge-Key': 'edge-test'})[0], 400)

    def tearDown(self):
        self.conn.close()
        self.thread.join(timeout=5)
        self.assertFalse(self.thread.is_alive())


if __name__ == '__main__':
    unittest.main()
