"""Real HTTP fixture for login, cookie scope, expiry and rejected credentials."""

import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs
from unittest.mock import patch

from sentinel_edge.auth import PortalSession, PortalAuthenticationError, media_http_options
from sentinel_edge.config import Config


class PortalAuthTests(unittest.TestCase):
    def setUp(self):
        self.state = {"posts": 0, "cookie": "session-one", "reject": False}
        state = self.state

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def reply(self, body, content_type="text/html"):
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/escape":
                    self.send_response(302)
                    self.send_header("Location", "https://outside.invalid/auth/login")
                    self.end_headers()
                elif self.path in {"/auth/login", "/"}:
                    self.reply(b'<form method="POST" action="/auth/login"></form>')
                elif self.headers.get("Cookie") != "sentinel=" + state["cookie"]:
                    self.send_response(302)
                    self.send_header("Location", "/auth/login")
                    self.end_headers()
                else:
                    self.reply(b'[{"id":"cam01","name":"Bridge"}]', "application/json")

            def do_POST(self):
                state["posts"] += 1
                body = parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode())
                if state["reject"] or body != {"email": ["alice@example.test"], "password": ["p@ss&word"]}:
                    self.reply(b'Email or access password is incorrect')
                else:
                    self.send_response(302)
                    self.send_header("Set-Cookie", "sentinel=" + state["cookie"] + "; Path=/; HttpOnly")
                    self.send_header("Location", "/")
                    self.end_headers()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        with patch.dict(os.environ, {"EDGE_API_KEY": "a", "WATCHLIST_HMAC_KEY": "b",
                                    "SENTINEL_EMAIL": "alice@example.test",
                                    "SENTINEL_PASSWORD": "p@ss&word",
                                    "SENTINEL_BASE_URL": self.base}, clear=True):
            self.config = Config.from_env(live=True)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_redirect_signs_in_and_reuses_memory_cookie(self):
        session = PortalSession(self.config)
        self.assertEqual(session.get_json(self.base + "/cameras.json")[0]["id"], "cam01")
        session.get_json(self.base + "/cameras.json")
        self.assertEqual(self.state["posts"], 1)
        self.assertEqual(session.headers_for(self.base + "/cam01/index.m3u8")["Cookie"],
                         "sentinel=session-one")
        self.assertNotIn("Cookie", session.headers_for("https://outside.invalid/live"))
        self.assertNotIn("Authorization", session.headers_for("https://outside.invalid/live"))

    def test_expired_cookie_gets_one_new_login(self):
        session = PortalSession(self.config)
        session.get_json(self.base + "/cameras.json")
        self.state["cookie"] = "session-two"
        session.get_json(self.base + "/cameras.json")
        self.assertEqual(self.state["posts"], 2)
        self.assertEqual(session.headers_for(self.base + "/cam01/index.m3u8")["Cookie"],
                         "sentinel=session-two")

    def test_rejection_is_redacted_and_rate_limited(self):
        self.state["reject"] = True
        session = PortalSession(self.config)
        for _ in range(2):
            with self.assertRaises(PortalAuthenticationError) as raised:
                session.get_json(self.base + "/cameras.json")
            self.assertNotIn("alice", str(raised.exception))
            self.assertNotIn("p@ss", str(raised.exception))
        self.assertEqual(self.state["posts"], 1)

    def test_cross_origin_redirect_is_rejected(self):
        session = PortalSession(self.config)
        with self.assertRaises(PortalAuthenticationError):
            session.get_json(self.base + "/escape")
        self.assertEqual(self.state["posts"], 0)

    def test_parallel_catalogue_requests_share_one_login(self):
        from concurrent.futures import ThreadPoolExecutor
        session = PortalSession(self.config)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: session.get_json(self.base + "/cameras.json"), range(2)))
        self.assertEqual(len(results), 2)
        self.assertEqual(self.state["posts"], 1)

    def test_media_cookies_are_domain_scoped_not_raw_headers(self):
        options = media_http_options("https://grid.test/cam01/index.m3u8",
                                     {"Cookie": "sentinel=dummy; other=example", "User-Agent": "test"})
        self.assertNotIn("Cookie", options["headers"])
        self.assertEqual(options["cookies"],
                         "sentinel=dummy; path=/; domain=grid.test;\nother=example; path=/; domain=grid.test;")


if __name__ == "__main__":
    unittest.main()
