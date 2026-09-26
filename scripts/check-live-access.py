"""Bounded live decode check: no recordings, plates, cookies or source URLs in output."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "edge"))

from sentinel_edge.auth import media_http_options
from sentinel_edge.cameras import discover, hls_headers, rtsp_access_url, select_two
from sentinel_edge.config import Config


def probe(config, camera, seconds, protocol_choice="auto"):
    import av
    av.logging.set_level(av.logging.PANIC)  # Native diagnostics can contain credential-bearing URLs.
    attempts = []
    for protocol, url in (("rtsp", rtsp_access_url(config, camera.rtsp_url)), ("hls", camera.hls_url)):
        if not url or protocol_choice not in {"auto", protocol}:
            continue
        started = time.monotonic()
        try:
            options = {"rtsp_transport": "tcp"} if protocol == "rtsp" else media_http_options(url, hls_headers(config, url))
            with av.open(url, options=options, timeout=(8, 8)) as container:
                video = next(stream for stream in container.streams if stream.type == "video")
                frames = 0
                first_pts = last_pts = None
                first_at = None
                discontinuities = 0
                for frame in container.decode(video):
                    now = time.monotonic()
                    if first_at is None:
                        first_at = now
                    pts = float(frame.pts * frame.time_base) if frame.pts is not None else None
                    if first_pts is None:
                        first_pts = pts
                    if pts is not None and last_pts is not None and (pts < last_pts or pts - last_pts > 5):
                        discontinuities += 1
                    last_pts = pts
                    frames += 1
                    if now - first_at >= seconds:
                        break
                if not frames:
                    raise RuntimeError("No frames")
                return {"camera_id": camera.camera_id, "protocol": protocol, "decoded_frames": frames,
                        "codec": video.codec_context.name, "width": video.codec_context.width,
                        "height": video.codec_context.height, "startup_seconds": round(first_at - started, 3),
                        "decode_seconds": round(time.monotonic() - first_at, 3),
                        "first_pts_seconds": first_pts, "last_pts_seconds": last_pts,
                        "pts_discontinuities": discontinuities, "prior_attempts": attempts}
        except Exception as error:
            # Do not print exception messages: third-party errors may include stream URLs.
            attempts.append({"protocol": protocol, "error_type": type(error).__name__})
    return {"camera_id": camera.camera_id, "decoded_frames": 0, "attempts": attempts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=20)
    parser.add_argument("--protocol", choices=("auto", "rtsp", "hls"), default="auto")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 900:
        parser.error("--seconds must be between 1 and 900")
    load_dotenv(ROOT / ".env")
    config = Config.from_env(live=True)
    try:
        cameras = discover(config)
        selected = select_two(cameras, config.camera_ids, require_requested=True)
    except Exception as error:
        print(json.dumps({"catalogue_available": False, "error_type": type(error).__name__}))
        return 1
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda camera: probe(config, camera, args.seconds, args.protocol), selected))
    print(json.dumps({"source": "live Sentinel; decode only, no AI or recording",
                      "catalogue_count": len(cameras), "cameras": results}, indent=2))
    return 0 if all(item["decoded_frames"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
