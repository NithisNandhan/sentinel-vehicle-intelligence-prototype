"""Timestamp aware PyAV stream reader. Frames are never written to disk."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

from .auth import media_http_options, portal_session
from .cameras import Camera, hls_headers, rtsp_access_url, source_urls
from .config import Config


@dataclass
class VideoFrame:
    image: object
    pts_ms: int | None
    discontinuity: bool


def _decode(url: str, *, transport: str, headers: dict[str, str] | None = None) -> Iterator[VideoFrame]:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required for live streams: install edge[live]") from exc
    options = {"rtsp_transport": "tcp", "stimeout": "8000000"} if transport == "rtsp" else {}
    if headers:
        options.update(media_http_options(url, headers))
    try:
        with av.open(url, options=options, timeout=(8, 8)) as container:
            video_stream = next((s for s in container.streams if s.type == "video"), None)
            if video_stream is None:
                raise OSError("No video stream")
            previous_pts: int | None = None
            for frame in container.decode(video_stream):
                pts_ms = round(float(frame.pts * frame.time_base) * 1000) if frame.pts is not None else None
                discontinuity = previous_pts is not None and pts_ms is not None and (pts_ms < previous_pts or pts_ms - previous_pts > 5000)
                if pts_ms is not None:
                    previous_pts = pts_ms
                yield VideoFrame(frame.to_ndarray(format="bgr24"), pts_ms, discontinuity)
    except av.error.FFmpegError:
        # PyAV errors do not consistently inherit OSError; redact their source URL.
        raise OSError("Video decoding interrupted or unavailable") from None


def read_camera(config: Config, camera: Camera, *, stop=None) -> Iterator[tuple[str, VideoFrame]]:
    """Prefer local relay if configured; else try RTSP then HLS, with bounded backoff."""
    rtsp, hls = source_urls(camera)
    rtsp = rtsp_access_url(config, rtsp)
    sources = [(config.media_relay_base_url + "/" + camera.camera_id, "rtsp")] if config.media_relay_base_url else [
        (url, transport) for url, transport in ((rtsp, "rtsp"), (hls, "hls")) if url
    ]
    backoff = 2
    while stop is None or not stop.is_set():
        for url, transport in sources:
            if stop is not None and stop.is_set():
                return
            try:
                headers = hls_headers(config, url) if transport == "hls" else None
                for frame in _decode(url, transport=transport, headers=headers):
                    backoff = 2
                    yield transport, frame
            except (OSError, ValueError, RuntimeError):
                # Never log URLs: RTSP userinfo and HLS headers contain credentials.
                if transport == "hls":
                    portal_session(config).invalidate()
                continue
        if stop is not None:
            stop.wait(backoff)
        else:
            time.sleep(backoff)
        backoff = min(backoff * 2, 30)
