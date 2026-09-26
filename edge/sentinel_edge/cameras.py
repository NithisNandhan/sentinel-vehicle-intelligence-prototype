"""Sentinel catalogue and stream source construction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from urllib.parse import quote, urlparse, urlsplit, urlunsplit

from .auth import PortalAuthenticationError, portal_session
from .config import Config


class CatalogueUnavailable(ValueError):
    """The catalogue could not be read as authorised camera JSON."""


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
        return [row if isinstance(row, dict) else {"id": row}
                for row in data if isinstance(row, (dict, str, int)) and not isinstance(row, bool)]
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


def discover(config: Config) -> list[Camera]:
    try:
        data = portal_session(config).get_json(config.sentinel_base_url + "/cameras.json")
        cameras = parse_catalogue(data, include_without_urls=True)
    except PortalAuthenticationError as error:
        raise CatalogueUnavailable(str(error)) from None
    if not cameras:
        raise ValueError("Camera catalogue contains no valid camera IDs")
    result = []
    for camera in cameras:
        rtsp, hls = documented_stream_urls(config, camera.camera_id)
        result.append(replace(camera, rtsp_url=camera.rtsp_url or rtsp,
                              hls_url=camera.hls_url or hls))
    return result


def documented_stream_urls(config: Config, camera_id: str) -> tuple[str, str]:
    """Use the public-grid path pattern only for IDs validated from JSON or manual pair."""
    escaped = quote(camera_id, safe="")
    return (f"rtsp://{config.sentinel_rtsp_host}:8554/stream/{escaped}",
            f"{config.sentinel_base_url}/{escaped}/index.m3u8")


def discover_live_pair(config: Config) -> tuple[list[Camera], bool]:
    """Use JSON first; only CAM01/CAM02 may use the documented fallback."""
    try:
        catalogue = discover(config)
    except (CatalogueUnavailable, OSError):
        if len(config.camera_ids) != 2 or set(config.camera_ids) != {"cam01", "cam02"}:
            raise ValueError("Catalogue unavailable; manual fallback requires CAMERA_IDS=cam01,cam02") from None
        names = {"cam01": "Chiman bhai Bridge", "cam02": "Janpath"}
        cameras = [Camera(camera_id, names[camera_id], *documented_stream_urls(config, camera_id))
                   for camera_id in config.camera_ids]
        return cameras, True
    return select_two(catalogue, config.camera_ids, require_requested=True), False


def select_two(catalogue: list[Camera], requested: tuple[str, ...], *,
               require_requested: bool = False) -> list[Camera]:
    if require_requested and (len(requested) != 2 or len(set(requested)) != 2):
        raise ValueError("Set CAMERA_IDS to two distinct authorised catalogue IDs before live processing")
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
    """Return catalogue URLs or explicitly documented public-grid patterns."""
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


def hls_headers(config: Config, url: str | None = None) -> dict[str, str]:
    return portal_session(config).headers_for(url or config.sentinel_base_url)

