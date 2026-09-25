"""Internal, credential-bearing catalogue search and temporary media previews."""

from __future__ import annotations

import hmac
import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable
from uuid import uuid4

from .cameras import Camera, discover
from .config import Config
from .relay import relay_camera


CATALOGUE_REFRESH_SECONDS = 300
PREVIEW_IDLE_SECONDS = 60
PREVIEW_START_SECONDS = 20
MAX_PREVIEWS = 2
CAMERA_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


class GatewayError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class Preview:
    id: str
    camera_id: str
    path: str
    started_at: float
    last_heartbeat: float
    stop: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    failed: bool = False


class CatalogueState:
    def __init__(self, config: Config, *, fetcher: Callable[[Config], list[Camera]] | None = None,
                 relay: Callable[..., None] = relay_camera,
                 ready: Callable[[str], bool] | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.config = config
        self.fetcher = fetcher or (lambda cfg: discover(cfg, include_without_urls=True))
        self.relay = relay
        self.ready = ready or self._hls_ready
        self.clock = clock
        self.lock = threading.RLock()
        self.refresh_lock = threading.Lock()
        self.cameras: dict[str, Camera] = {}
        self.refreshed_at = 0.0
        self.attempted_at: float | None = None
        self.available = False
        self.previews: dict[str, Preview] = {}

    @staticmethod
    def _hls_ready(path: str) -> bool:
        try:
            with urllib.request.urlopen(f"http://mediamtx:8888/{path}/index.m3u8", timeout=2) as response:
                return response.status == 200
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    def refresh(self, *, force: bool = False) -> None:
        with self.refresh_lock:
            with self.lock:
                if not force and self.attempted_at is not None and self.clock() - self.attempted_at < CATALOGUE_REFRESH_SECONDS:
                    if self.available:
                        return
                    raise GatewayError(503, "Sentinel catalogue unavailable; authorised JSON access is required")
                self.attempted_at = self.clock()
            try:
                cameras = self.fetcher(self.config)
                if not cameras:
                    raise ValueError("empty catalogue")
            except (OSError, ValueError, TypeError, KeyError):
                with self.lock:
                    self.available = False
                    self.cameras = {}
                raise GatewayError(503, "Sentinel catalogue unavailable; authorised JSON access is required") from None
            with self.lock:
                self.cameras = {camera.camera_id: camera for camera in cameras}
                self.refreshed_at = self.clock()
                self.available = True

    def _catalogue(self) -> dict[str, Camera]:
        self.refresh()
        with self.lock:
            return dict(self.cameras)

    @staticmethod
    def _public(camera: Camera) -> dict:
        return {"id": camera.camera_id, "name": camera.name, "location": camera.location,
                "codec": camera.codec, "live": camera.live}

    def search(self, query: str, offset: int, limit: int) -> dict:
        query = query.strip().casefold()
        if not 2 <= len(query) <= 120 or offset < 0 or not 1 <= limit <= 25:
            raise GatewayError(422, "Invalid search query or page")
        matches: list[tuple[int, str, Camera]] = []
        for camera in self._catalogue().values():
            camera_id = camera.camera_id.casefold()
            name = camera.name.casefold()
            location = camera.location.casefold()
            if camera_id == query:
                rank = 0
            elif camera_id.startswith(query):
                rank = 1
            elif query in name or query in location:
                rank = 2
            else:
                continue
            matches.append((rank, camera_id, camera))
        matches.sort(key=lambda item: (item[0], item[1]))
        return {"total": len(matches), "offset": offset, "limit": limit,
                "items": [self._public(item[2]) for item in matches[offset:offset + limit]]}

    def start_preview(self, camera_id: str) -> dict:
        if not CAMERA_ID.fullmatch(camera_id):
            raise GatewayError(422, "Invalid camera ID")
        camera = self._catalogue().get(camera_id)
        if camera is None:
            raise GatewayError(404, "Camera not in authorised catalogue")
        if not camera.live:
            raise GatewayError(409, "Camera is offline")
        if not camera.rtsp_url and not camera.hls_url:
            raise GatewayError(409, "Camera stream is unavailable")
        with self.lock:
            if len(self.previews) >= MAX_PREVIEWS:
                raise GatewayError(429, "Two previews are already active")
            session_id = uuid4().hex
            now = self.clock()
            preview = Preview(session_id, camera_id, f"preview-{session_id}", now, now)
            self.previews[session_id] = preview

        def run() -> None:
            try:
                self.relay(self.config, camera, preview.stop, path=preview.path)
            except Exception:
                # Stream URLs can contain credentials; do not log exception details.
                preview.failed = True

        preview.thread = threading.Thread(target=run, daemon=True, name=f"preview-{session_id[:8]}")
        preview.thread.start()
        return self.preview_status(session_id)

    def preview_status(self, session_id: str) -> dict:
        with self.lock:
            preview = self.previews.get(session_id)
        if preview is None:
            raise GatewayError(404, "Preview session not found")
        if preview.failed or (preview.thread and not preview.thread.is_alive()):
            status = "failed"
        elif self.ready(preview.path):
            status = "ready"
        elif self.clock() - preview.started_at >= PREVIEW_START_SECONDS:
            preview.stop.set()
            status = "failed"
        else:
            status = "starting"
        return {"id": preview.id, "camera_id": preview.camera_id, "status": status,
                "preview_url": f"/media/{preview.path}/index.m3u8"}

    def heartbeat(self, session_id: str) -> dict:
        with self.lock:
            preview = self.previews.get(session_id)
            if preview is None:
                raise GatewayError(404, "Preview session not found")
            preview.last_heartbeat = self.clock()
        return self.preview_status(session_id)

    def stop_preview(self, session_id: str) -> None:
        with self.lock:
            preview = self.previews.pop(session_id, None)
        if preview is None:
            raise GatewayError(404, "Preview session not found")
        preview.stop.set()
        if preview.thread:
            preview.thread.join(timeout=5)

    def reap_idle(self) -> None:
        with self.lock:
            expired = [session_id for session_id, item in self.previews.items()
                       if self.clock() - item.last_heartbeat >= PREVIEW_IDLE_SECONDS]
        for session_id in expired:
            try:
                self.stop_preview(session_id)
            except GatewayError:
                pass


class GatewayHandler(BaseHTTPRequestHandler):
    state: CatalogueState

    def log_message(self, _format: str, *args) -> None:
        # Request paths and headers should not appear in container logs.
        pass

    def _reply(self, status: int, data: dict) -> None:
        body = json.dumps(data, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorised(self) -> bool:
        provided = self.headers.get("X-Edge-Key", "")
        if not provided or not hmac.compare_digest(provided, self.state.config.edge_api_key):
            self._reply(401, {"detail": "Unauthorised"})
            return False
        return True

    def _handle(self, method: str) -> None:
        if not self._authorised():
            return
        parsed = urllib.parse.urlsplit(self.path)
        parts = parsed.path.strip("/").split("/")
        try:
            if method == "GET" and parts == ["search"]:
                params = urllib.parse.parse_qs(parsed.query)
                result = self.state.search(params.get("q", [""])[0], int(params.get("offset", ["0"])[0]),
                                           int(params.get("limit", ["25"])[0]))
                self._reply(200, result)
            elif method == "POST" and parts == ["previews"]:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1024:
                    raise GatewayError(422, "Invalid preview request")
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise GatewayError(422, "Invalid preview request")
                self._reply(201, self.state.start_preview(str(body.get("camera_id", ""))))
            elif len(parts) == 2 and parts[0] == "previews" and method == "GET":
                self._reply(200, self.state.preview_status(parts[1]))
            elif len(parts) == 3 and parts[0] == "previews" and parts[2] == "heartbeat" and method == "POST":
                self._reply(200, self.state.heartbeat(parts[1]))
            elif len(parts) == 2 and parts[0] == "previews" and method == "DELETE":
                self.state.stop_preview(parts[1])
                self._reply(200, {"stopped": True})
            else:
                self._reply(404, {"detail": "Not found"})
        except (ValueError, TypeError, json.JSONDecodeError):
            self._reply(422, {"detail": "Invalid request"})
        except GatewayError as error:
            self._reply(error.status, {"detail": error.message})

    def do_GET(self) -> None:
        self._handle("GET")

    def do_POST(self) -> None:
        self._handle("POST")

    def do_DELETE(self) -> None:
        self._handle("DELETE")


def run_gateway(config: Config) -> None:
    state = CatalogueState(config)
    handler = type("BoundGatewayHandler", (GatewayHandler,), {"state": state})
    server = ThreadingHTTPServer(("0.0.0.0", 8765), handler)
    server.daemon_threads = True

    def maintain() -> None:
        while True:
            try:
                state.refresh()
            except GatewayError:
                pass
            state.reap_idle()
            time.sleep(5)

    threading.Thread(target=maintain, daemon=True, name="catalogue-maintenance").start()
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        for session_id in list(state.previews):
            state.stop_preview(session_id)
