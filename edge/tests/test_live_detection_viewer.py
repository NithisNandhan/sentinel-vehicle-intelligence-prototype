import json
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from sentinel_edge.live_detection_viewer import LiveState, safe_request, detect_regions, crop_box


class LiveViewerTests(unittest.TestCase):
    def test_only_approved_pair_and_expiring_session(self):
        clock = [10.]
        state = LiveState(clock=lambda: clock[0])
        with self.assertRaises(ValueError): state.select('cam03')
        state.select('cam01')
        event = state.capture_stop
        state.publish({'frame': 'in-memory', 'camera_id': 'cam01'})
        clock[0] = 71.
        state.expire()
        self.assertTrue(event.is_set())
        self.assertIsNone(state.camera_id)
        self.assertNotIn('frame', state.snapshot())

    def test_switch_and_stop_clear_old_camera_media(self):
        state = LiveState()
        state.select('cam01')
        old = state.capture_stop
        state.publish({'camera_id': 'cam01', 'frame': 'old'})
        state.select('cam02')
        self.assertTrue(old.is_set())
        self.assertNotIn('frame', state.snapshot())
        state.publish({'camera_id': 'cam01', 'frame': 'late'})
        self.assertNotIn('frame', state.snapshot())
        state.publish({'camera_id': 'cam02', 'frame': 'new'})
        state.select(None)
        self.assertNotIn('frame', state.snapshot())

    def test_old_generation_cannot_publish_after_switching_back(self):
        state = LiveState()
        state.select('cam01')
        old = state.capture_stop
        state.select('cam02')
        state.select('cam01')
        state.publish({'camera_id': 'cam01', 'frame': 'stale'}, old)
        self.assertNotIn('frame', state.snapshot())

    def test_native_crop_is_encoded_before_overlay_and_text_is_masked(self):
        import numpy as np
        from sentinel_edge.live_detection_viewer import observe
        tokens = ('', *'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        text = 'AP01AB1234'
        probabilities = []
        for c in text:
            row = [0.]*len(tokens); row[tokens.index(c)] = 1.; probabilities.append(row)
        seen = []
        model = SimpleNamespace(tokens=tokens, probabilities=lambda crop: seen.append(crop.copy()) or probabilities)
        frames = []
        for i in range(3):
            image = np.full((40, 150, 3), 100, dtype=np.uint8)
            image[4, 4] = i
            frames.append(('rtsp', SimpleNamespace(image=image, pts_ms=i*500, discontinuity=False)))
        state = LiveState(); state.select('cam01')
        with patch('sentinel_edge.live_detection_viewer.viewer_frames', return_value=iter(frames)), \
                patch('sentinel_edge.live_detection_viewer.detect_regions', return_value=([], (0, 0, 150, 40))):
            observe(SimpleNamespace(), SimpleNamespace(camera_id='cam01'), model, None, None, state, state.capture_stop)
        np.testing.assert_array_equal(seen[0], frames[0][1].image)
        result = state.snapshot()
        self.assertEqual(result['crop']['observations'], 3)
        self.assertEqual(result['crop']['masked_plate'], 'AP******34')
        self.assertNotIn(text, json.dumps(result))
        self.assertEqual(result['pts_ms'], 1000)

    def test_startup_wait_is_bounded_and_parent_cancellation_propagates(self):
        import threading
        from sentinel_edge.live_detection_viewer import StartupStop
        clock = [0.]; parent = threading.Event()
        guard = StartupStop(parent, clock=lambda: clock[0])
        clock[0] = 31.
        self.assertTrue(guard.is_set())
        guard.started = True
        self.assertFalse(guard.is_set())
        parent.set()
        self.assertTrue(guard.is_set())

    def test_stalled_relay_falls_back_to_existing_direct_reader(self):
        import threading
        from dataclasses import dataclass
        from sentinel_edge.live_detection_viewer import viewer_frames
        @dataclass
        class Config:
            media_relay_base_url: str = 'rtsp://mediamtx:8554'
        attempts = []
        def read(config, camera, stop):
            attempts.append(config.media_relay_base_url)
            return iter([] if config.media_relay_base_url else [('rtsp', 'native frame')])
        with patch('sentinel_edge.stream.read_camera', side_effect=read):
            self.assertEqual(list(viewer_frames(Config(), None, threading.Event())), [('rtsp', 'native frame')])
        self.assertEqual(attempts, ['rtsp://mediamtx:8554', ''])
        with patch('sentinel_edge.stream.read_camera', return_value=iter([])):
            with self.assertRaises(TimeoutError):
                list(viewer_frames(Config(), None, threading.Event()))

    def test_http_security_headers_and_origin_token_requirements(self):
        import http.client
        import re
        import socket
        import threading
        import time
        from sentinel_edge.live_detection_viewer import serve
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        thread = threading.Thread(target=serve, args=(None, None, None),
                                  kwargs={'port': port, 'lifetime': 2}, daemon=True)
        thread.start()
        for _ in range(50):
            try:
                connection = http.client.HTTPConnection('127.0.0.1', port, timeout=2)
                connection.request('GET', '/')
                response = connection.getresponse(); page = response.read().decode(); break
            except OSError:
                time.sleep(.02)
        self.assertEqual(response.status, 200)
        self.assertIn('no-store', response.getheader('Cache-Control'))
        self.assertIn("frame-ancestors 'none'", response.getheader('Content-Security-Policy'))
        token = re.search(r'const token="([^"]+)"', page)[1]
        connection.request('GET', '/snapshot')
        response = connection.getresponse(); response.read()
        self.assertEqual(response.status, 403)
        connection.request('GET', '/snapshot', headers={'X-Viewer-Token': token})
        response = connection.getresponse(); self.assertEqual(json.loads(response.read())['status'], 'paused')
        connection.request('POST', '/session', json.dumps({'camera_id': None}),
                           headers={'Origin': 'http://evil.test', 'X-Viewer-Token': token})
        response = connection.getresponse(); response.read(); self.assertEqual(response.status, 403)
        connection.request('POST', '/session', json.dumps({'camera_id': None}),
                           headers={'Origin': f'http://127.0.0.1:{port}', 'X-Viewer-Token': token})
        response = connection.getresponse(); response.read(); self.assertEqual(response.status, 200)
        connection.close(); thread.join(timeout=5); self.assertFalse(thread.is_alive())

    def test_request_rejects_rebinding_and_cross_site(self):
        self.assertTrue(safe_request('127.0.0.1:7780', None, False))
        self.assertTrue(safe_request('localhost:7780', 'http://localhost:7780', True))
        self.assertFalse(safe_request('attacker.test:7780', None, False))
        self.assertFalse(safe_request('127.0.0.1:7780', 'http://attacker.test', True))
        self.assertFalse(safe_request('127.0.0.1:7780', None, True))

    def test_crop_uses_original_pixels_and_clips_coordinates(self):
        import numpy as np
        image = np.arange(40*120*3).reshape(40, 120, 3)
        crop = crop_box(image, (-4, 2, 118, 30))
        np.testing.assert_array_equal(crop, image[2:30, 0:118])
        self.assertIsNone(crop_box(image, (100, 20, 50, 25)))

    def test_vehicle_roi_plate_coordinates_stay_aligned(self):
        import numpy as np
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        boxes = SimpleNamespace(xyxy=SimpleNamespace(tolist=lambda: [[20, 10, 180, 90]]),
                                conf=SimpleNamespace(tolist=lambda: [.9]))
        model = SimpleNamespace(predict=lambda *a, **kw: [SimpleNamespace(boxes=boxes)])
        detector = SimpleNamespace(box=lambda frame: None if frame.shape[0] == 100 else (5, 6, 90, 30))
        vehicles, plate = detect_regions(image, model, detector)
        self.assertEqual(vehicles, [(20, 10, 180, 90)])
        self.assertEqual(plate, (25, 16, 110, 40))


if __name__ == '__main__':
    unittest.main()
