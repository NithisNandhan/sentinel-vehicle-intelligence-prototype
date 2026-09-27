import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from dataclasses import dataclass

from sentinel_edge.diagnostic_gateway import catalogue, discover, frames


class DiagnosticGatewayTests(unittest.TestCase):
    def test_direct_startup_failure_reuses_authorised_preview(self):
        from sentinel_edge.cameras import Camera
        from sentinel_edge.live_detection_viewer import viewer_frames
        @dataclass
        class Config:
            media_relay_base_url: str = ''
        camera = Camera('cam17', 'Test', 'rtsp://example.test/cam17')
        with patch('sentinel_edge.stream.read_camera', return_value=iter([])), \
             patch('sentinel_edge.diagnostic_gateway.frames', return_value=iter([('rtsp', 'frame')])) as fallback:
            self.assertEqual(list(viewer_frames(Config(), camera, threading.Event())), [('rtsp', 'frame')])
            self.assertEqual(fallback.call_args.args[1], camera)

    def test_authorised_metadata_only_and_offline_excluded(self):
        with patch('sentinel_edge.diagnostic_gateway.request', return_value={
            'total': 2, 'items': [{'id': 'cam17', 'name': 'Test', 'live': True,
                                  'rtsp_url': 'must not leak'},
                                 {'id': 'cam18', 'name': 'Offline', 'live': False}]}) as req:
            result = catalogue(None)
        self.assertEqual(req.call_args.args[1], '/search?q=cam&offset=0&limit=25')
        self.assertEqual([c.camera_id for c in result], ['cam17'])
        self.assertIsNone(result[0].rtsp_url)

    def test_fallback_requires_successful_gateway(self):
        with patch('sentinel_edge.cameras.discover', side_effect=ValueError), \
             patch('sentinel_edge.diagnostic_gateway.catalogue', side_effect=ValueError):
            with self.assertRaises(ValueError): discover(None)

    def test_preview_closed_after_capture_and_on_decoder_failure(self):
        for decoded in ([object()], None):
            stop = threading.Event()
            with patch('sentinel_edge.diagnostic_gateway.request', return_value={'id': 'a'*32}) as req, \
                 patch('sentinel_edge.stream._decode') as decode:
                if decoded is None:
                    decode.side_effect = RuntimeError('redacted')
                else:
                    def closed_by_operator(*args,**kwargs):
                        yield from decoded
                        stop.set()
                    decode.side_effect = closed_by_operator
                if decoded is None:
                    with self.assertRaises(RuntimeError):
                        list(frames(None, SimpleNamespace(camera_id='cam17'), threading.Event()))
                else:
                    self.assertEqual(len(list(frames(None, SimpleNamespace(camera_id='cam17'), stop))), 1)
                self.assertEqual(req.call_args.args[1:], ('/previews/'+'a'*32, 'DELETE'))

    def test_clean_eof_reconnects_same_preview_and_marks_discontinuity(self):
        from sentinel_edge.stream import VideoFrame
        delays = []
        stop = SimpleNamespace(is_set=lambda:False,wait=lambda seconds:delays.append(seconds))
        with patch('sentinel_edge.diagnostic_gateway.request',return_value={'id':'a'*32}) as req, \
             patch('sentinel_edge.stream._decode',side_effect=[iter([VideoFrame(None,100,False)]),
                                                              iter([VideoFrame(None,200,False)])]) as decode:
            stream = frames(None,SimpleNamespace(camera_id='cam17'),stop)
            self.assertFalse(next(stream)[1].discontinuity)
            self.assertTrue(next(stream)[1].discontinuity)
            stream.close()
        self.assertEqual(delays,[2])
        self.assertEqual(decode.call_args_list[0],decode.call_args_list[1])
        self.assertEqual(req.call_args.args[1:],('/previews/'+'a'*32,'DELETE'))


if __name__ == '__main__': unittest.main()
