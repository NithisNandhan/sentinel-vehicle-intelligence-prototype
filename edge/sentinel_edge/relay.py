"""Live-only FFmpeg relay. It never records media."""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .cameras import Camera, hls_headers, source_urls
from .config import Config


def _probe(url: str, transport: str, headers: dict[str, str] | None) -> str:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required for relay probing") from exc
    options = {"rtsp_transport": "tcp"} if transport == "rtsp" else {}
    if headers:
        options["headers"] = "".join(f"{key}: {value}\r\n" for key, value in headers.items())
    with av.open(url, options=options, timeout=(8, 8)) as container:
        video = next((stream for stream in container.streams if stream.type == "video"), None)
        if video is None:
            raise OSError("No video stream")
        return video.codec_context.name.lower()


def _nvenc_available() -> bool:
    result = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                            capture_output=True, text=True, timeout=10, check=False)
    return "h264_nvenc" in result.stdout


def _ffmpeg_args(url: str, transport: str, headers: dict[str, str] | None,
                 codec: str, encoder: str, target: str) -> list[str]:
    args = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error"]
    if transport == "rtsp":
        args += ["-rtsp_transport", "tcp"]
    elif headers:
        args += ["-headers", "".join(f"{key}: {value}\r\n" for key, value in headers.items())]
    args += ["-i", url, "-map", "0:v:0", "-an"]
    if codec == "h264":
        args += ["-c:v", "copy"]
    else:
        args += ["-c:v", encoder, "-pix_fmt", "yuv420p"]
        if encoder == "libx264":
            args += ["-preset", "veryfast", "-tune", "zerolatency"]
        else:
            args += ["-preset", "p4", "-tune", "ll"]
    return args + ["-f", "rtsp", "-rtsp_transport", "tcp", target]


def relay_camera(config: Config, camera: Camera, stop: threading.Event, *, path: str | None = None) -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg executable not found")
    rtsp, hls = source_urls(camera)
    sources = [(url, transport, headers) for url, transport, headers in
               ((rtsp, "rtsp", None), (hls, "hls", hls_headers(config))) if url]
    target = config.mediamtx_publish_url + "/" + (path or camera.camera_id)
    backoff = 2
    while not stop.is_set():
        for url, transport, headers in sources:
            if stop.is_set():
                return
            try:
                codec = _probe(url, transport, headers)
            except (OSError, ValueError, RuntimeError):
                continue
            encoders = ["copy"] if codec == "h264" else (["h264_nvenc", "libx264"] if _nvenc_available() else ["libx264"])
            for encoder in encoders:
                args = _ffmpeg_args(url, transport, headers, codec, encoder, target)
                launched_at = time.monotonic()
                with subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL) as process:
                    while process.poll() is None and not stop.wait(1):
                        pass
                    if stop.is_set():
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                        return
                    if process.returncode == 0:
                        break
                if time.monotonic() - launched_at > 10:
                    break  # Stream interruption, not encoder initialization failure.
                # NVENC may be listed but unavailable at runtime; use CPU fallback.
        stop.wait(backoff)
        backoff = min(backoff * 2, 30)


def run_relays(config: Config, cameras: list[Camera], stop: threading.Event) -> None:
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="relay")
    try:
        futures = [pool.submit(relay_camera, config, camera, stop) for camera in cameras]
        for future in as_completed(futures):
            future.result()
    finally:
        stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
