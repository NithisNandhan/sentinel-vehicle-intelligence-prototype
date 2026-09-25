"""Sentinel catalogue and stream source construction."""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import quote, urlparse, urlsplit, urlunsplit

from .config import Config


@dataclass(frozen=True)
class Camera:
    camera_id: str
    name: str
    rtsp_url: str | None = None
    hls_url: str | None = None
    live: bool = True
    location: str = ""
    codec: str = ""


def _camera_rows(data: object) -> list[dict]:
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("cameras", "items", "data", "streams"):
            if key in data:
                return _camera_rows(data[key])
        # Also tolerate {"cam01": {"name": ...}} catalogues.
        return [dict(value, id=key) for key, value in data.items()
                if isinstance(value, dict)]
    return []


def parse_catalogue(data: object, *, include_without_urls: bool = False) -> list[Camera]:
    cameras: list[Camera] = []
    seen: set[str] = set()
    for row in _camera_rows(data):
        raw_id = next((row[key] for key in ("id", "camera_id", "path") if row.get(key) is not None), None)
        camera_id = str(raw_id).strip() if raw_id is not None else ""
        if not camera_id or len(camera_id) > 80 or not all(c.isalnum() or c in "_-" for c in camera_id) or camera_id in seen:
            continue
        urls = row.get("urls") or row.get("streams") or row.get("endpoints") or {}
        if not isinstance(urls, dict):
            urls = {}
        def get_url(*keys: str, schemes: tuple[str, ...]) -> str | None:
            for source in (row, urls):
                for key in keys:
                    value = source.get(key)
                    if isinstance(value, dict):
                        value = value.get("url")
                    if isinstance(value, str) and urlparse(value).scheme in schemes and urlparse(value).hostname:
                        return value
            return None
        rtsp = get_url("rtsp_url", "rtspUrl", "rtsp", schemes=("rtsp", "rtsps"))
        hls = get_url("hls_url", "hlsUrl", "hls", schemes=("http", "https"))
        if not (rtsp or hls) and not include_without_urls:
            continue
        status = row.get("live", row.get("is_live", row.get("status", True)))
        live = status if isinstance(status, bool) else str(status).strip().lower() not in {"false", "0", "offline", "inactive", "down"}
        seen.add(camera_id)
        raw_location = row.get("location")
        if isinstance(raw_location, dict):
            location = ", ".join(str(raw_location[key]).strip() for key in ("name", "area", "city", "district", "region")
                                  if raw_location.get(key))
        else:
            location = str(raw_location or row.get("area") or row.get("region") or "").strip()
        codec = str(row.get("codec") or row.get("video_codec") or "").strip()
        cameras.append(Camera(camera_id, str(row.get("name") or location or camera_id), rtsp, hls, live,
                              location, codec))
    return cameras


def discover(config: Config, *, include_without_urls: bool = False) -> list[Camera]:
    token = base64.b64encode(f"{config.sentinel_email}:{config.sentinel_password}".encode()).decode()
    headers = {"Authorization": "Basic " + token, "Accept": "application/json"}
    if config.sentinel_cookie:
        headers["Cookie"] = config.sentinel_cookie
    request = urllib.request.Request(
        config.sentinel_base_url + "/api/ingest",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            cameras = parse_catalogue(json.load(response), include_without_urls=include_without_urls)
    except urllib.error.HTTPError as error:
        raise ValueError(f"Sentinel catalogue HTTP {error.code}; authorised API access is required") from None
    except json.JSONDecodeError:
        raise ValueError("Sentinel catalogue did not return JSON; authorised API access is required") from None
    if not cameras:
        raise ValueError("Camera catalogue contains no valid camera IDs")
    return cameras


def select_two(catalogue: list[Camera], requested: tuple[str, ...], *,
               require_requested: bool = False) -> list[Camera]:
    if require_requested and len(requested) != 2:
        raise ValueError("Set CAMERA_IDS to two exact authorised catalogue IDs before live processing")
    by_id = {camera.camera_id: camera for camera in catalogue if camera.live}
    if requested:
        missing = set(requested) - set(by_id)
        if missing:
            raise ValueError("Requested camera IDs absent from catalogue: " + ", ".join(sorted(missing)))
        selected = [by_id[item] for item in requested]
    else:
        selected = [camera for camera in catalogue if camera.live][:2]
    if len(selected) != 2:
        raise ValueError("Exactly two cameras are required for the live prototype")
    return selected


def source_urls(camera: Camera) -> tuple[str | None, str | None]:
    """Use catalogue URLs; endpoint patterns are not a stable contract."""
    return camera.rtsp_url, camera.hls_url


def rtsp_access_url(config: Config, url: str | None) -> str | None:
    """Add registered-user credentials only to the approved direct RTSP host.

    A catalogue URL may already carry its own userinfo. Never send the local
    password to a different host supplied by a catalogue response.
    """
    if not url:
        return None
    parsed = urlsplit(url)
    if (parsed.scheme not in {"rtsp", "rtsps"} or parsed.username is not None or
            parsed.hostname != config.sentinel_rtsp_host):
        return url
    user = quote(config.sentinel_email, safe="")
    password = quote(config.sentinel_password, safe="")
    return urlunsplit(parsed._replace(netloc=f"{user}:{password}@{parsed.netloc}"))


def hls_headers(config: Config) -> dict[str, str]:
    token = base64.b64encode(f"{config.sentinel_email}:{config.sentinel_password}".encode()).decode()
    headers = {"Authorization": "Basic " + token}
    if config.sentinel_cookie:
        headers["Cookie"] = config.sentinel_cookie
    return headers

