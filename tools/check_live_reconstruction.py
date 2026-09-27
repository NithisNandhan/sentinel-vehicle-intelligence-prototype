"""Short protected viewer smoke test. Outputs metadata only, never live pixels/text.

Run inside the existing viewer container; keys are read from its environment.
Requires an initially paused session and returns it to paused after testing.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
import re
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=int, default=90)
    args = parser.parse_args()
    if not 15 <= args.seconds <= 180:
        raise ValueError('Smoke window must be between 15 and 180 seconds')
    base = 'http://127.0.0.1:7780'
    headers = {'X-Edge-Key':os.environ['EDGE_API_KEY']}
    def request(path, payload=None, engineering=False):
        data = json.dumps(payload).encode() if payload is not None else None
        access = browser_headers if engineering else headers
        req = urllib.request.Request(base+path, data=data, headers={**access, 'Content-Type':'application/json'})
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.load(response)
    ready_deadline = time.monotonic()+45
    while True:
        try:
            initial = request('/internal/status')
            break
        except urllib.error.URLError:
            if time.monotonic() >= ready_deadline:
                raise ValueError('Viewer did not become ready within 45 seconds') from None
            time.sleep(1)
    if initial['status'] != 'paused' or initial.get('camera_id'):
        raise ValueError('Shared session is already in use; smoke test did not change it')
    with urllib.request.urlopen(base+'/', timeout=10) as response:
        page = response.read().decode()
    token = re.search(r'const viewerToken=("[^"\n]+")', page)
    if not token:
        raise ValueError('Viewer token unavailable')
    browser_headers = {'X-Viewer-Token':json.loads(token[1]), 'X-Operator-Key':os.environ['OPERATOR_API_KEY']}
    catalogue_deadline = time.monotonic()+60
    while True:
        catalogue = request('/internal/cameras')
        if catalogue.get('available'):
            break
        if time.monotonic() >= catalogue_deadline:
            raise ValueError('Authorised camera scope did not become ready within 60 seconds')
        time.sleep(1)
    ids = {row['id'] for row in catalogue.get('cameras', [])}
    requested = [camera.strip() for camera in os.environ.get('CAMERA_IDS', 'cam19,cam26').split(',') if camera.strip()]
    if len(requested) != 2 or len(set(requested)) != 2 or any(camera not in ids for camera in requested):
        raise ValueError('Both configured cameras must be in the current authorised catalogue or approved relay scope')
    cameras = requested
    report = dict(started_at_utc=datetime.now(timezone.utc).isoformat(), source_mode='live',
                  requested_seconds_per_camera=args.seconds, camera_scope=catalogue.get('scope'),
                  images_saved=False, plaintext_saved=False, ground_truth_available=False,
                  accuracy=None, learned_restoration='not_run', cameras=[])
    for camera in cameras:
        last = {}
        statuses = Counter()
        reconstructed = set()
        counts = Counter()
        timing = []
        request('/internal/session', {'camera_id':camera})
        deadline = time.monotonic()+args.seconds
        try:
            while time.monotonic() < deadline:
                request('/internal/heartbeat', {'camera_id':camera})
                value = request('/engineering', engineering=True)
                if value.get('camera_id') != camera:
                    raise ValueError('Shared session changed; smoke test stopped')
                last = value
                for key in ('decoded_frames','analysed_frames','plate_crops','reconstruction_attempts','reconstructed_windows','reconstruction_failures','unique_model_accepted_unverified','maximum_distinct_observations'):
                    counts[key] = max(counts[key], value.get(key, 0) or 0)
                debug = value.get('debug', {})
                ms = debug.get('stage_ms', {}).get('reconstruction', 0)
                if ms:timing.append(ms)
                for track in debug.get('tracks', []):
                    result = track.get('reconstruction', {})
                    statuses[result.get('status', 'missing')] += 1
                    if result.get('status') == 'reconstructed':
                        reconstructed.add((track['track_id'], result.get('reference_pts_ms'), result.get('attempts')))
                time.sleep(2)
        finally:
            # Do not pause a camera an operator switched to while we were testing.
            current = request('/internal/status')
            if current.get('camera_id') == camera:
                request('/internal/session', {'camera_id':None})
        report['cameras'].append(dict(camera_id=camera, status=last.get('status'),
                                     counts=dict(counts), polled_reconstruction_states=dict(statuses),
                                     observed_reconstructed_windows=len(reconstructed),
                                     reconstruction_stage_ms_max=max(timing, default=None),
                                     frame_age_seconds=last.get('frame_age_seconds')))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':main()
