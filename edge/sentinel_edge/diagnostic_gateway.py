"""Reuse the protected, temporary preview gateway; never expose source URLs."""
import json
import re
import threading
import urllib.request

from .cameras import Camera


def request(config, path, method='GET', data=None):
    req = urllib.request.Request('http://catalogue-gateway:8765' + path,
        data=json.dumps(data).encode() if data is not None else None,
        headers={'X-Edge-Key': config.edge_api_key, 'Content-Type': 'application/json'}, method=method)
    with urllib.request.urlopen(req, timeout=25) as response:
        body = response.read(1024 * 1024)
        return json.loads(body) if body else None


def catalogue(config):
    cameras = []
    for offset in range(0, 8000, 25):
        # The documented grid uses cam-prefixed IDs; the gateway requires >=2 characters.
        page = request(config, f'/search?q=cam&offset={offset}&limit=25')
        for row in page['items']:
            camera_id = row['id']
            if re.fullmatch(r'[A-Za-z0-9_-]{1,80}', camera_id) and row['live']:
                cameras.append(Camera(camera_id, row['name']))
        if offset + 25 >= page['total']:
            return cameras
    raise ValueError('Diagnostic catalogue exceeds bounded page limit')


def frames(config, camera, stop):
    from .stream import _decode
    preview = request(config, '/previews', 'POST', {'camera_id': camera.camera_id})
    session_id = preview['id']
    if not re.fullmatch(r'[a-f0-9]{32}', session_id):
        raise ValueError('Invalid gateway preview identifier')
    finished = threading.Event()
    def heartbeat():
        while not finished.wait(20):
            if stop.is_set():
                return
            try:
                request(config, f'/previews/{session_id}/heartbeat', 'POST', {})
            except Exception:
                return  # Existing gateway idle timeout cleans up failed sessions.
    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        # Path is an identifier supplied by the authorised gateway, not a guessed camera URL.
        for attempt in range(5):
            if stop.is_set():
                return
            try:
                for frame in _decode(f'rtsp://mediamtx:8554/preview-{session_id}', transport='rtsp'):
                    if stop.is_set():
                        return
                    yield 'rtsp', frame
                return
            except OSError:
                stop.wait(min(2 ** (attempt + 1), 8))
        raise TimeoutError('Authorised temporary preview unavailable')
    finally:
        finished.set()
        try:
            request(config, f'/previews/{session_id}', 'DELETE')
        except Exception:
            pass  # Gateway's 60-second idle cleanup remains the backstop.


def discover(config):
    from .cameras import discover as direct
    try:
        return direct(config)
    except Exception:
        # Gateway performs its own current authorisation and never serves stale-on-error data.
        return catalogue(config)
