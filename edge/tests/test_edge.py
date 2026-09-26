from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import threading

from sentinel_edge.api import CentralClient
from sentinel_edge.benchmark import latency_summary
from sentinel_edge.cameras import (Camera, CatalogueUnavailable, discover, discover_live_pair,
                                   parse_catalogue, rtsp_access_url, select_two, source_urls)
from sentinel_edge.catalogue_gateway import CatalogueState, GatewayError, GatewayHandler
from sentinel_edge.config import Config
from sentinel_edge.events import sighting
from sentinel_edge.outbox import Outbox
from sentinel_edge.privacy import mask_plate, normalize_plate, plate_hmac
from sentinel_edge.relay import _ffmpeg_args
from sentinel_edge.simulator import seed_events
from sentinel_edge.stream import VideoFrame, read_camera
from sentinel_edge.vision import VisionPipeline, preferred_onnx_providers


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
        with self.assertRaisesRegex(ValueError, "CAMERA_IDS"):
            select_two(cameras, (), require_requested=True)
        with self.assertRaisesRegex(ValueError, "distinct"):
            select_two(cameras, ("2", "2"), require_requested=True)

    def test_full_catalogue_keeps_offline_cameras_without_stream_urls(self):
        rows = {"cameras": [{"id": "cam01", "name": "Bridge", "location": "Ahmedabad",
                              "live": False, "codec": "h265"}]}
        self.assertEqual(parse_catalogue(rows), [])
        camera = parse_catalogue(rows, include_without_urls=True)[0]
        self.assertEqual((camera.location, camera.codec, camera.live), ("Ahmedabad", "h265", False))

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

    def test_direct_rtsp_credentials_are_encoded_and_host_scoped(self):
        with patch.dict(os.environ, {"EDGE_API_KEY": "a", "WATCHLIST_HMAC_KEY": "b",
                                  "SENTINEL_EMAIL": "alice@example.com",
                                  "SENTINEL_PASSWORD": "a@b:c/1",
                                  "SENTINEL_RTSP_HOST": "103.250.160.189"}, clear=True):
            config = Config.from_env(live=True)
        source = "rtsp://103.250.160.189:8554/live/cam01"
        self.assertEqual(rtsp_access_url(config, source),
                         "rtsp://alice%40example.com:a%40b%3Ac%2F1@103.250.160.189:8554/live/cam01")
        self.assertEqual(rtsp_access_url(config, "rtsp://outside.test/live/cam01"),
                         "rtsp://outside.test/live/cam01")
        self.assertEqual(rtsp_access_url(config, "rtsp://token@103.250.160.189:8554/live/cam01"),
                         "rtsp://token@103.250.160.189:8554/live/cam01")
        self.assertIsNone(rtsp_access_url(config, None))

    def test_discovery_uses_official_catalogue_endpoint(self):
        with patch.dict(os.environ, {"EDGE_API_KEY": "a", "WATCHLIST_HMAC_KEY": "b",
                                  "SENTINEL_EMAIL": "team@test.invalid", "SENTINEL_PASSWORD": "secret",
                                  "SENTINEL_BASE_URL": "https://grid.test"}, clear=True):
            config = Config.from_env(live=True)
        response = {"cameras": [
            {"id": "cam01", "name": "Bridge", "status": "live"},
            {"id": "cam02", "name": "Janpath", "status": "offline"},
        ]}
        with patch("sentinel_edge.cameras.portal_session") as session:
            session.return_value.get_json.return_value = response
            cameras = discover(config)
        session.return_value.get_json.assert_called_once_with("https://grid.test/cameras.json")
        self.assertEqual([camera.camera_id for camera in cameras], ["cam01", "cam02"])
        self.assertEqual(cameras[0].rtsp_url, "rtsp://103.250.160.189:8554/stream/cam01")
        self.assertEqual(cameras[0].hls_url, "https://grid.test/cam01/index.m3u8")
        self.assertFalse(cameras[1].live)

    def test_public_catalogue_accepts_id_only_list(self):
        with patch.dict(os.environ, {"EDGE_API_KEY": "a", "WATCHLIST_HMAC_KEY": "b",
                                  "SENTINEL_EMAIL": "team@test.invalid", "SENTINEL_PASSWORD": "test"}, clear=True):
            config = Config.from_env(live=True)
        with patch("sentinel_edge.cameras.portal_session") as session:
            session.return_value.get_json.return_value = ["cam01", "cam02"]
            cameras = discover(config)
        self.assertEqual([camera.camera_id for camera in cameras], ["cam01", "cam02"])
        self.assertEqual(cameras[0].hls_url, "https://cctv.corp8.cloud/cam01/index.m3u8")

    def test_discovery_reports_forbidden_without_exposing_credentials(self):
        with patch.dict(os.environ, {"EDGE_API_KEY": "a", "WATCHLIST_HMAC_KEY": "b",
                                  "SENTINEL_EMAIL": "team@test.invalid", "SENTINEL_PASSWORD": "secret"}, clear=True):
            config = Config.from_env(live=True)
        from sentinel_edge.auth import PortalAuthenticationError
        with patch("sentinel_edge.cameras.portal_session") as session:
            session.return_value.get_json.side_effect = PortalAuthenticationError("Sentinel catalogue HTTP 403")
            with self.assertRaisesRegex(CatalogueUnavailable, "HTTP 403") as context:
                discover(config)
        self.assertNotIn("secret", str(context.exception))

    def test_only_selected_pair_uses_documented_fallback(self):
        with patch.dict(os.environ, {"EDGE_API_KEY": "a", "WATCHLIST_HMAC_KEY": "b",
                                  "SENTINEL_EMAIL": "team@test.invalid", "SENTINEL_PASSWORD": "test",
                                  "CAMERA_IDS": "cam01,cam02"}, clear=True):
            config = Config.from_env(live=True)
        with patch("sentinel_edge.cameras.discover", side_effect=CatalogueUnavailable("sign-in")):
            cameras, fallback = discover_live_pair(config)
        self.assertTrue(fallback)
        self.assertEqual([camera.camera_id for camera in cameras], ["cam01", "cam02"])
        self.assertEqual(cameras[1].rtsp_url, "rtsp://103.250.160.189:8554/stream/cam02")
        self.assertEqual(cameras[1].hls_url, "https://cctv.corp8.cloud/cam02/index.m3u8")

        with patch.dict(os.environ, {"EDGE_API_KEY": "a", "WATCHLIST_HMAC_KEY": "b",
                                  "SENTINEL_EMAIL": "team@test.invalid", "SENTINEL_PASSWORD": "test",
                                  "CAMERA_IDS": "cam01,cam03"}, clear=True):
            other = Config.from_env(live=True)
        with patch("sentinel_edge.cameras.discover", side_effect=CatalogueUnavailable("sign-in")):
            with self.assertRaisesRegex(ValueError, "cam01,cam02"):
                discover_live_pair(other)

    def test_valid_catalogue_missing_selected_camera_never_falls_back(self):
        with patch.dict(os.environ, {"EDGE_API_KEY": "a", "WATCHLIST_HMAC_KEY": "b",
                                  "SENTINEL_EMAIL": "team@test.invalid", "SENTINEL_PASSWORD": "test",
                                  "CAMERA_IDS": "cam01,cam02"}, clear=True):
            config = Config.from_env(live=True)
        with patch("sentinel_edge.cameras.discover", return_value=[Camera("cam01", "Bridge")]):
            with self.assertRaisesRegex(ValueError, "cam02"):
                discover_live_pair(config)

    def test_valid_catalogue_uses_selected_pair_without_fallback(self):
        with patch.dict(os.environ, {"EDGE_API_KEY": "a", "WATCHLIST_HMAC_KEY": "b",
                                  "SENTINEL_EMAIL": "team@test.invalid", "SENTINEL_PASSWORD": "test",
                                  "CAMERA_IDS": "cam01,cam02"}, clear=True):
            config = Config.from_env(live=True)
        catalogue = [Camera("cam02", "Janpath"), Camera("cam01", "Bridge"), Camera("cam03", "Other")]
        with patch("sentinel_edge.cameras.discover", return_value=catalogue):
            cameras, fallback = discover_live_pair(config)
        self.assertFalse(fallback)
        self.assertEqual([camera.camera_id for camera in cameras], ["cam01", "cam02"])

    def test_valid_empty_catalogue_does_not_trigger_manual_fallback(self):
        with patch.dict(os.environ, {"EDGE_API_KEY": "a", "WATCHLIST_HMAC_KEY": "b",
                                  "SENTINEL_EMAIL": "team@test.invalid", "SENTINEL_PASSWORD": "test",
                                  "CAMERA_IDS": "cam01,cam02"}, clear=True):
            config = Config.from_env(live=True)
        with patch("sentinel_edge.cameras.portal_session") as session:
            session.return_value.get_json.return_value = []
            with self.assertRaisesRegex(ValueError, "no valid camera IDs"):
                discover_live_pair(config)

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

    def test_direct_reader_uses_encoded_rtsp_url(self):
        with patch.dict(os.environ, {"EDGE_API_KEY": "a", "WATCHLIST_HMAC_KEY": "b",
                                  "SENTINEL_EMAIL": "alice@example.com", "SENTINEL_PASSWORD": "p@ss",
                                  "MEDIA_RELAY_BASE_URL": ""}, clear=True):
            config = Config.from_env(live=True)
        received = []
        def fake_decode(url, *, transport, headers=None):
            received.append((url, transport))
            yield VideoFrame(object(), 1000, False)
        with patch("sentinel_edge.stream._decode", side_effect=fake_decode):
            next(read_camera(config, Camera("cam01", "Bridge",
                       "rtsp://103.250.160.189:8554/live/cam01"), stop=threading.Event()))
        self.assertEqual(received, [
            ("rtsp://alice%40example.com:p%40ss@103.250.160.189:8554/live/cam01", "rtsp")
        ])


class CatalogueGatewayTests(unittest.TestCase):
    def test_full_catalogue_search_never_uses_manual_pair(self):
        config = SimpleNamespace(edge_api_key="test")
        with patch("sentinel_edge.catalogue_gateway.discover",
                   side_effect=CatalogueUnavailable("sign-in")):
            state = CatalogueState(config)
            with self.assertRaisesRegex(GatewayError, "unavailable"):
                state.search("cam01", 0, 25)
        self.assertEqual(state.cameras, {})

    def test_internal_gateway_requires_service_key(self):
        handler = GatewayHandler.__new__(GatewayHandler)
        handler.state = SimpleNamespace(config=SimpleNamespace(edge_api_key="expected"))
        rejected = []
        handler._reply = lambda status, body: rejected.append((status, body))
        handler.headers = {"X-Edge-Key": "wrong"}
        self.assertFalse(handler._authorised())
        self.assertEqual(rejected[0][0], 401)
        handler.headers = {"X-Edge-Key": "expected"}
        self.assertTrue(handler._authorised())

    def test_search_eight_thousand_cameras_and_hide_stream_urls(self):
        cameras = [Camera(f"cam{index:04d}", f"Camera {index}",
                          rtsp_url=f"rtsp://private.invalid/secret/{index}", live=True,
                          location="Janpath" if index % 100 == 0 else "Elsewhere", codec="h264")
                   for index in range(8000)]
        state = CatalogueState(SimpleNamespace(edge_api_key="test"), fetcher=lambda _cfg: cameras)
        result = state.search("Janpath", 25, 25)
        self.assertEqual(result["total"], 80)
        self.assertEqual(len(result["items"]), 25)
        self.assertNotIn("rtsp://", json.dumps(result))
        self.assertEqual(result["items"][0]["id"], "cam2500")
        self.assertEqual(state.search("cam0000", 0, 25)["items"][0]["id"], "cam0000")

    def test_exact_id_then_prefix_then_name_or_location_ranking(self):
        cameras = [Camera("z1", "Cam office", location="Elsewhere"),
                   Camera("cam42", "Bridge", location="Elsewhere"),
                   Camera("cam", "Exact", location="Elsewhere"),
                   Camera("a1", "Bridge", location="Cam circle")]
        state = CatalogueState(SimpleNamespace(edge_api_key="test"), fetcher=lambda _cfg: cameras)
        self.assertEqual([item["id"] for item in state.search("CAM", 0, 25)["items"]],
                         ["cam", "cam42", "a1", "z1"])

    def test_sign_in_html_fails_closed_without_repeated_fetch(self):
        calls = []
        def html_response(_cfg):
            calls.append(1)
            raise ValueError("Sign in HTML, not JSON")
        state = CatalogueState(SimpleNamespace(edge_api_key="test"), fetcher=html_response,
                               clock=lambda: 1000.0)
        for _ in range(2):
            with self.assertRaisesRegex(GatewayError, "authorised JSON access"):
                state.search("cam", 0, 25)
        self.assertEqual(len(calls), 1)

    def test_actual_html_catalogue_response_is_unavailable(self):
        with patch.dict(os.environ, {"EDGE_API_KEY": "a", "WATCHLIST_HMAC_KEY": "b",
                                  "SENTINEL_EMAIL": "team@test.invalid", "SENTINEL_PASSWORD": "test"}, clear=True):
            config = Config.from_env(live=True)
        state = CatalogueState(config)
        from sentinel_edge.auth import PortalAuthenticationError
        with patch("sentinel_edge.cameras.portal_session") as session:
            session.return_value.get_json.side_effect = PortalAuthenticationError("Sentinel sign-in failed")
            with self.assertRaisesRegex(GatewayError, "authorised JSON access"):
                state.search("cam", 0, 25)

    def test_offline_camera_cannot_start_preview(self):
        camera = Camera("cam04", "Circle", live=False, location="Paldi Circle")
        state = CatalogueState(SimpleNamespace(edge_api_key="test"), fetcher=lambda _cfg: [camera])
        self.assertFalse(state.search("Paldi", 0, 25)["items"][0]["live"])
        with self.assertRaisesRegex(GatewayError, "offline"):
            state.start_preview("cam04")

    def test_live_camera_without_stream_cannot_start_preview(self):
        camera = Camera("cam05", "Missing stream", live=True)
        state = CatalogueState(SimpleNamespace(edge_api_key="test"), fetcher=lambda _cfg: [camera])
        with self.assertRaisesRegex(GatewayError, "stream is unavailable"):
            state.start_preview("cam05")

    def test_relay_failure_reports_failed_preview(self):
        camera = Camera("cam03", "Other", rtsp_url="rtsp://private.invalid/cam03")
        def broken_relay(_config, _camera, _stop, *, path):
            raise RuntimeError("sensitive source URL")
        state = CatalogueState(SimpleNamespace(edge_api_key="test"), fetcher=lambda _cfg: [camera],
                               relay=broken_relay, ready=lambda _path: False)
        preview = state.start_preview("cam03")
        state.previews[preview["id"]].thread.join(timeout=1)
        self.assertEqual(state.preview_status(preview["id"])["status"], "failed")
        state.stop_preview(preview["id"])

    def test_preview_cap_heartbeat_and_idle_cleanup(self):
        now = [1000.0]
        camera = Camera("cam03", "Other", rtsp_url="rtsp://private.invalid/cam03", live=True)
        def waiting_relay(_config, _camera, stop, *, path):
            self.assertTrue(path.startswith("preview-"))
            stop.wait()
        state = CatalogueState(SimpleNamespace(edge_api_key="test"), fetcher=lambda _cfg: [camera],
                               relay=waiting_relay, ready=lambda _path: True, clock=lambda: now[0])
        first = state.start_preview("cam03")
        second = state.start_preview("cam03")
        self.assertEqual(first["status"], "ready")
        self.assertNotEqual(first["preview_url"], second["preview_url"])
        with self.assertRaisesRegex(GatewayError, "Two previews"):
            state.start_preview("cam03")
        now[0] += 40
        state.heartbeat(first["id"])
        now[0] += 21
        state.reap_idle()
        self.assertIn(first["id"], state.previews)
        self.assertNotIn(second["id"], state.previews)
        state.stop_preview(first["id"])
        self.assertEqual(state.previews, {})


class VisionTests(unittest.TestCase):
    def test_latency_summary_reports_median_and_95th_percentile(self):
        self.assertEqual(latency_summary([10, 20, 30, 40, 50]),
                         {"median_ms": 30, "p95_ms": 50})

    def test_preferred_onnx_providers_skip_uninstalled_tensorrt(self):
        self.assertEqual(preferred_onnx_providers([
            "TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider",
        ]), ["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.assertEqual(preferred_onnx_providers(["CPUExecutionProvider"]), ["CPUExecutionProvider"])

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
