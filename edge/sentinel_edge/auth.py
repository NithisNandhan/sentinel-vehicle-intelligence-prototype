"""Same-origin form login with memory-only cookies and safe failure messages."""

from __future__ import annotations

import http.cookiejar
import json
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlencode, urlsplit

from .config import Config

USER_AGENT = "Mozilla/5.0"


class PortalAuthenticationError(ValueError):
    """Never includes credentials, cookies, URLs or response bodies."""


def _origin(url: str) -> tuple[str, str | None, int | None]:
    parsed = urlsplit(url)
    if parsed.username is not None or parsed.password is not None:
        raise PortalAuthenticationError("Credential-bearing portal URLs are not allowed")
    return parsed.scheme, parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)


class _SameOriginRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, origin):
        self.origin = origin

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if _origin(newurl) != self.origin:
            raise PortalAuthenticationError("Portal redirected to an unexpected host")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class PortalSession:
    def __init__(self, config: Config, *, clock=time.monotonic):
        self.config = config
        self.origin = _origin(config.sentinel_base_url)
        self.cookies = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            _SameOriginRedirect(self.origin), urllib.request.HTTPCookieProcessor(self.cookies))
        self.lock = threading.RLock()
        self.clock = clock
        self.active = False
        self.login_at = 0.0
        self.failed_until = 0.0
        self.override_cookie = config.sentinel_cookie

    def _request(self, url: str, *, data: bytes | None = None, accept="text/html"):
        if _origin(url) != self.origin:
            raise PortalAuthenticationError("Portal request uses an unexpected host")
        request = urllib.request.Request(url, data=data,
                                         headers={"User-Agent": USER_AGENT, "Accept": accept})
        if data is not None:
            request.add_header("Content-Type", "application/x-www-form-urlencoded")
        if self.override_cookie:
            request.add_unredirected_header("Cookie", self.override_cookie)
        return self.opener.open(request, timeout=15)

    def _login(self, *, force=False) -> None:
        now = self.clock()
        if not force and self.active and now - self.login_at < 900:
            return
        if now < self.failed_until:
            raise PortalAuthenticationError("Sentinel sign-in failed; retry is temporarily delayed")
        self.active = False
        self.override_cookie = ""
        self.cookies.clear()
        try:
            with self._request(self.config.sentinel_base_url + "/auth/login"):
                pass
            body = urlencode({"email": self.config.sentinel_email,
                              "password": self.config.sentinel_password}).encode()
            with self._request(self.config.sentinel_base_url + "/auth/login", data=body) as response:
                if urlsplit(response.geturl()).path.rstrip("/") == "/auth/login" or not list(self.cookies):
                    raise PortalAuthenticationError("Sentinel rejected sign-in; check the registered email and access password")
            self.active = True
            self.login_at = now
        except (OSError, ValueError):
            self.failed_until = now + 60
            self.cookies.clear()
            raise PortalAuthenticationError("Sentinel sign-in failed; check account access and credentials") from None

    def _json_request(self, url: str):
        try:
            with self._request(url, accept="application/json") as response:
                if urlsplit(response.geturl()).path.rstrip("/") == "/auth/login":
                    return None
                payload = response.read(16 * 1024 * 1024 + 1)
                if len(payload) > 16 * 1024 * 1024:
                    raise PortalAuthenticationError("Sentinel catalogue is too large")
                if "text/html" in response.headers.get("Content-Type", "") or payload.lstrip().startswith(b"<"):
                    return None
                return json.loads(payload)
        except urllib.error.HTTPError as error:
            if error.code in {401, 403}:
                return None
            raise PortalAuthenticationError(f"Sentinel catalogue HTTP {error.code}") from None

    def get_json(self, url: str):
        with self.lock:
            try:
                result = self._json_request(url)
                if result is None:
                    self._login(force=True)
                    result = self._json_request(url)
                if result is None:
                    raise PortalAuthenticationError("Sentinel catalogue remains unavailable after sign-in")
                return result
            except (OSError, json.JSONDecodeError):
                raise PortalAuthenticationError("Sentinel catalogue could not be read") from None

    def headers_for(self, url: str) -> dict[str, str]:
        headers = {"User-Agent": USER_AGENT}
        if _origin(url) != self.origin:
            return headers
        with self.lock:
            if self.override_cookie:
                headers["Cookie"] = self.override_cookie
                return headers
            self._login()
            request = urllib.request.Request(url)
            self.cookies.add_cookie_header(request)
            cookie = request.get_header("Cookie")
            if cookie:
                headers["Cookie"] = cookie
            return headers

    def invalidate(self) -> None:
        with self.lock:
            # Prevent reconnect failures from creating a tight login loop.
            if self.clock() - self.login_at >= 60:
                self.active = False
                self.override_cookie = ""


_sessions: dict[tuple[str, str, str, str], PortalSession] = {}
_sessions_lock = threading.Lock()


def media_http_options(url: str, headers: dict[str, str]) -> dict[str, str]:
    """FFmpeg cookie options enforce host/path scope for nested HLS requests."""
    options = {}
    plain = {key: value for key, value in headers.items() if key.lower() != "cookie"}
    if plain:
        options["headers"] = "".join(f"{key}: {value}\r\n" for key, value in plain.items())
    cookie = next((value for key, value in headers.items() if key.lower() == "cookie"), "")
    if cookie:
        host = urlsplit(url).hostname
        options["cookies"] = "\n".join(f"{item.strip()}; path=/; domain={host};"
                                       for item in cookie.split(";") if "=" in item)
    return options


def portal_session(config: Config) -> PortalSession:
    key = (config.sentinel_base_url, config.sentinel_email,
           config.sentinel_password, config.sentinel_cookie)
    with _sessions_lock:
        if key not in _sessions:
            _sessions[key] = PortalSession(config)
        return _sessions[key]
