import hashlib
import hmac
import re
import base64
import json
import secrets
import time
from urllib.parse import urlsplit

from fastapi import Header, HTTPException, Request

from .config import get_settings


def normalize_plate(plate: str) -> str:
    result = re.sub(r"[^A-Z0-9]", "", plate.upper())
    if len(result) < 5 or len(result) > 20:
        raise ValueError("plate must contain 5 to 20 letters or digits")
    return result


def plate_token(plate: str) -> str:
    key = get_settings().watchlist_hmac_key.encode("utf-8")
    return hmac.new(key, normalize_plate(plate).encode("ascii"), hashlib.sha256).hexdigest()


def check_edge_key(x_edge_key: str | None = Header(default=None)) -> None:
    expected = get_settings().edge_api_key
    if not x_edge_key or not hmac.compare_digest(x_edge_key, expected):
        raise HTTPException(status_code=401, detail="invalid edge key")


COOKIE_NAME = "sentinel_local"
SESSION_SECONDS = 3600


def local_request_allowed(connection, *, require_origin=False) -> bool:
    settings = get_settings()
    if not settings.local_dashboard_mode:
        return False
    origin = connection.headers.get("origin")
    host = connection.headers.get("host", "").lower()
    allowed = {urlsplit(value).netloc.lower() for value in settings.local_origin_list}
    if host not in allowed:
        return False
    if connection.headers.get("sec-fetch-site", "") == "cross-site":
        return False
    return (origin in settings.local_origin_list and urlsplit(origin).netloc.lower() == host
            if origin else not require_origin)


def _signature(value: str) -> str:
    key = get_settings().operator_api_key.encode("utf-8")
    return hmac.new(key, ("sentinel-local-v1:" + value).encode("ascii"), hashlib.sha256).hexdigest()


def create_local_session():
    expiry = int(time.time()) + SESSION_SECONDS
    nonce = secrets.token_urlsafe(24)
    payload = base64.urlsafe_b64encode(json.dumps([expiry, nonce], separators=(",", ":")).encode()).decode().rstrip("=")
    token = payload + "." + _signature(payload)
    csrf = _signature("csrf:" + nonce)
    return token, csrf, expiry


def local_session(connection, *, require_origin=False):
    if not local_request_allowed(connection, require_origin=require_origin):
        return None
    value = connection.cookies.get(COOKIE_NAME, "")
    if len(value) > 512:
        return None
    try:
        payload, signature = value.split(".")
        if not hmac.compare_digest(signature, _signature(payload)):
            return None
        expiry, nonce = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        if type(expiry) is not int or expiry <= time.time() or not isinstance(nonce, str) or len(nonce) > 64:
            return None
        return _signature("csrf:" + nonce)
    except (ValueError, TypeError, UnicodeError):
        return None


def check_operator_key(request: Request, x_operator_key: str | None = Header(default=None)) -> None:
    expected = get_settings().operator_api_key
    if x_operator_key and hmac.compare_digest(x_operator_key, expected):
        return
    mutation = request.method not in {"GET", "HEAD", "OPTIONS"}
    csrf = local_session(request, require_origin=mutation)
    if csrf is None:
        raise HTTPException(status_code=401, detail="Local dashboard access unavailable")
    if mutation and not hmac.compare_digest(request.headers.get("x-csrf-token", ""), csrf):
        raise HTTPException(status_code=403, detail="Local request verification failed")

