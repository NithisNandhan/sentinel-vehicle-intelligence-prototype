from __future__ import annotations

import json
import io
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import threading

from sentinel_edge.api import CentralClient
from sentinel_edge.cameras import Camera, discover, parse_catalogue, select_two, source_urls
from sentinel_edge.config import Config
from sentinel_edge.events import sighting
from sentinel_edge.outbox import Outbox
from sentinel_edge.privacy import mask_plate, normalize_plate, plate_hmac
from sentinel_edge.relay import _ffmpeg_args
from sentinel_edge.simulator import seed_events
from sentinel_edge.stream import VideoFrame, read_camera
from sentinel_edge.vision import VisionPipeline


class PrivacyTests(unittest.TestCase):
    def test_normalize_mask_and_hmac(self):
        self.assertEqual(normalize_plate("gj 01-ab 1234"), "GJ01AB1234")
        self.assertEqual(mask_plate("GJ01AB1234"), "GJ******34")
        self.assertEqual(plate_hmac("gj 01-ab 1234", "secret"),
                         plate_hmac("GJ01AB1234", "secret"))

    def test_wire_event_does_not_contain_plaintext(self):
        event = sighting(edge_id="edge-1", camera_id="cam01", plate="GJ01AB1234",
                         hmac_key="secret", confidence=0.95, source_mode="live")
        self.assertNotIn("GJ01AB1234", json.dumps(event))
        self.assertEqual(event["masked_plate"], "GJ******34")
        self.assertEqual(len(event["plate_hmac"]), 64)


class CameraTests(unittest.TestCase):
    def test_catalogue_shapes_and_selection(self):
        cameras = parse_catalogue({"cameras": [
            {"id": 1, "name": "Bridge", "live": True, "urls": {
                "rtsp": "rtsp://grid.test:8554/stream/1", "hls": "http://grid.test/live/stream/1/index.m3u8"}},
            {"camera_id": "cam04", "location": "Circle", "status": "online",
             "rtsp_url": "rtsp://grid.test:8554/stream/cam04"},
            {"id": "wrong-path"},
        ]})
        self.assertEqual([c.camera_id for c in cameras], ["1", "cam04"])
        self.assertEqual([c.camera_id for c in select_two(cameras, ("cam04", "1"))],
                         ["cam04", "1"])
        self.assertEqual(source_urls(cameras[0]),
                         ("rtsp://grid.test:8554/stream/1", "http://grid.test/live/stream/1/index.m3u8"))

    def test_catalogue_skips_offline_and_missing_urls(self):
        cameras = parse_catalogue({"data": [
            {"id": 1, "live": False, "rtsp": "rtsp://grid.test/stream/1"},
            {"id": 2, "live": True, "hls": "http://grid.test/live/stream/2/index.m3u8"},
            {"id": 3, "live": True, "rtsp": "rtsp://grid.test/stream/3"},
            {"id": 4, "live": True},
        ]})
        self.assertEqual([c.camera_id for c in select_two(cameras, ())], ["2", "3"])

    def test_configured_first_two_portal_cameras(self):
        with patch.dict(os.environ, {"EDGE_API_KEY": "a", "WATCHLIST_HMAC_KEY": "b",
                                  "SENTINEL_EMAIL": "team@test.invalid", "SENTINEL_PASSWORD": "test",
                                  "CAMERA_IDS": "cam01,cam02"}, clear=True):
            config = Config.from_env(live=True)
        cameras = parse_catalogue({"cameras": [
            {"id": "cam01", "name": "Chiman bhai Bridge", "hls": "https://grid.test/cam01/index.m3u8"},
            {"id": "cam02", "name": "Janpath", "hls": "https://grid.test/cam02/index.m3u8"},
            {"id": "cam03", "name": "Other", "hls": "https://grid.test/cam03/index.m3u8"},
        ]})
        self.assertEqual([c.camera_id for c in select_two(cameras, config.camera_ids)],
                         ["cam01", "cam02"])

    def test_nested_camel_case_url_fields(self):
        cameras = parse_catalogue({"items": [{"id": "CAM-A", "status": "live", "endpoints": {
            "rtspUrl": {"url": "rtsp://grid.test:8554/stream/CAM-A"},
            "hlsUrl": {"url": "https://grid.test/live/stream/CAM-A/index.m3u8"},
        }}]})
        self.assertEqual(source_urls(cameras[0]),
                         ("rtsp://grid.test:8554/stream/CAM-A",
                          "https://grid.test/live/stream/CAM-A/index.m3u8"))

    def test_discovery_uses_official_catalogue_endpoint(self):
        with patch.dict(os.environ, {"EDGE_API_KEY": "a", "WATCHLIST_HMAC_KEY": "b",
                                  "SENTINEL_EMAIL": "team@test.invalid", "SENTINEL_PASSWORD": "secret",
                                  "SENTINEL_BASE_URL": "https://grid.test"}, clear=True):
            config = Config.from_env(live=True)
        response = io.BytesIO(json.dumps({"cameras": [
            {"id": 1, "rtsp": "rtsp://grid.test:8554/stream/1"}
        ]}).encode())
        with patch("sentinel_edge.cameras.urllib.request.urlopen", return_value=response) as open_url:
            cameras = discover(config)
        self.assertEqual(open_url.call_args.args[0].full_url, "https://grid.test/api/ingest")
        self.assertEqual(cameras[0].camera_id, "1")

    def test_live_requires_credentials(self):
        with patch.dict(os.environ, {"EDGE_API_KEY": "a", "WATCHLIST_HMAC_KEY": "b",
                                  "SENTINEL_EMAIL": "", "SENTINEL_PASSWORD": ""}):
            with self.assertRaisesRegex(ValueError, "SENTINEL_EMAIL"):
                Config.from_env(live=True)


class DeliveryTests(unittest.TestCase):
    def test_http_post_has_edge_key_and_privacy_safe_json(self):
        with tempfile.TemporaryDirectory() as directory:
            client = CentralClient("http://localhost:8000", "edge-key", Outbox(Path(directory) / "q.sqlite3"))
            event = sighting(edge_id="edge-1", camera_id="cam01", plate="GJ01AB1234",
                             hmac_key="secret", confidence=0.95, source_mode="live")
            class Response:
                status = 201
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    return False
            with patch("sentinel_edge.api.urllib.request.urlopen", return_value=Response()) as urlopen:
                client.post("/v1/sightings", event)
            request = urlopen.call_args.args[0]
            self.assertEqual(request.get_header("X-edge-key"), "edge-key")
            self.assertNotIn("GJ01AB1234", request.data.decode())

    def test_outbox_survives_reopen_and_retries_without_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queue.sqlite3"
            first = Outbox(path)
            event = sighting(edge_id="edge-1", camera_id="cam01", plate="GJ01AB1234",
                             hmac_key="secret", confidence=0.95, source_mode="synthetic")
            first.put("/v1/sightings", event)
            first.put("/v1/sightings", event)
            self.assertEqual(first.depth(), 1)
            reopened = Outbox(path)
            self.assertEqual(reopened.depth(), 1)
            client = CentralClient("http://localhost:8000", "edge-key", reopened)
            with patch.object(client, "post", side_effect=OSError("offline")):
                self.assertEqual(client.flush_once(), 0)
            self.assertEqual(reopened.depth(), 1)
            with patch.object(client, "post") as post:
                self.assertEqual(client.flush_once(), 1)
                post.assert_called_once()
            self.assertEqual(reopened.depth(), 0)

    def test_simulator_events_are_deterministic_and_private(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"EDGE_API_KEY": "a", "WATCHLIST_HMAC_KEY": "b",
                                      "EDGE_DB_PATH": str(Path(directory) / "outbox.sqlite3")}, clear=True):
                config = Config.from_env(live=False)
            outbox = Outbox(config.db_path)
            seed_events(config, outbox)
            first_ids = {event_id for event_id, _, _ in outbox.next_batch()
                         if event_id}
            seed_events(config, outbox)
            payloads = [payload for _, path, payload in outbox.next_batch() if path == "/v1/sightings"]
            self.assertEqual(len(payloads), 3)
            self.assertEqual({event_id for event_id, _, _ in outbox.next_batch()} & first_ids, first_ids)
            self.assertEqual([p["camera_id"] for p in payloads], ["cam01", "cam01", "cam04"])
            self.assertNotIn("GJ01AB1234", json.dumps(payloads))


class RelayTests(unittest.TestCase):
    def test_h264_copy_and_h265_transcode(self):
        copied = _ffmpeg_args("rtsp://source", "rtsp", None, "h264", "copy", "rtsp://target")
        self.assertEqual(copied[copied.index("-c:v") + 1], "copy")
        converted = _ffmpeg_args("http://source", "hls", None, "hevc", "h264_nvenc", "rtsp://target")
        self.assertEqual(converted[converted.index("-c:v") + 1], "h264_nvenc")

    def test_direct_reader_falls_back_to_hls(self):
        with patch.dict(os.environ, {"EDGE_API_KEY": "a", "WATCHLIST_HMAC_KEY": "b",
                                  "SENTINEL_EMAIL": "a@b.test", "SENTINEL_PASSWORD": "pass",
                                  "MEDIA_RELAY_BASE_URL": ""}, clear=True):
            config = Config.from_env(live=True)

        def fake_decode(url, *, transport, headers=None):
            if transport == "rtsp":
                raise OSError("port blocked")
            yield VideoFrame(object(), 1000, False)

        with patch("sentinel_edge.stream._decode", side_effect=fake_decode):
            camera = Camera("1", "Bridge", "rtsp://grid.test/stream/1",
                            "http://grid.test/live/stream/1/index.m3u8")
            transport, frame = next(read_camera(config, camera, stop=threading.Event()))
        self.assertEqual(transport, "hls")
        self.assertEqual(frame.pts_ms, 1000)


class VisionTests(unittest.TestCase):
    def test_two_read_consensus_and_dedup(self):
        class Image:
            shape = (200, 300, 3)
            def __getitem__(self, _):
                return self
        class Box:
            id = SimpleNamespace(item=lambda: 7)
            xyxy = [SimpleNamespace(tolist=lambda: [0, 0, 200, 100])]
        class Blur:
            def var(self):
                return 100
        class CV:
            COLOR_BGR2GRAY = 1
            CV_64F = 2
            def cvtColor(self, image, _):
                return image
            def Laplacian(self, image, _):
                return Blur()
        class Boxes:
            id = True
            def __iter__(self):
                return iter([Box()])
        class TrackingModel:
            def track(self, *args, **kwargs):
                return [SimpleNamespace(boxes=Boxes())]
        class ALPR:
            def predict(self, image):
                return [SimpleNamespace(ocr=SimpleNamespace(text="GJ01AB1234", confidence=0.96))]
        vision = VisionPipeline.__new__(VisionPipeline)
        vision.cv2 = CV()
        vision.model = TrackingModel()
        vision.alpr = ALPR()
        vision.ocr_min_confidence = 0.85
        vision.dedup_seconds = 300
        vision.tracks = {}
        vision.last_plate_seen = {}
        self.assertEqual(vision.process(Image()), [])
        result = vision.process(Image())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].plate, "GJ01AB1234")
        self.assertEqual(vision.process(Image()), [])


if __name__ == "__main__":
    unittest.main()
