"""Seeded-watchlist replay through the real edge pipeline, outbox, HTTP client and backend.

Each case is a scripted vehicle passage: per-frame OCR specs for a known true
plate (SYNTHETIC unless built from a human-labelled set with --labels). The edge
side is the real TrackedOCR + SharedDelivery + durable Outbox + CentralClient;
the backend is a real uvicorn process on a temporary database. The harness
compares expected and actual outcomes:

  always:            missed confirmed alerts, false confirmed alerts, duplicate
                     alerts, delivery failures, alerts raised while a review is
                     pending
  partial matching:  review recall, review precision, false review leads
                     (reported as "not applicable", never 0, when disabled)

Synthetic scripted cases prove the integration and the safety rules. They say
nothing about recognition accuracy on real footage.

    python tools/seeded_watchlist_replay.py --output report.json [--partial-matching]
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'edge'))
sys.path.insert(0, str(ROOT / 'edge' / 'tests'))

EDGE_KEY, OPERATOR_KEY, HMAC_KEY = 'replay-edge-key', 'replay-operator-key', 'replay-hmac-key'
REVIEWER_TOKEN = 'replay-reviewer-token'
WATCHED = ['MH49BE5557', 'KA01AB1234', 'GJ05CD4321']


def weak(text, position, rival, peak=.45):
    spec = list(text)
    spec[position] = (text[position], peak, rival)
    return spec


def default_cases():
    """Scripted passages. `truth` is the real plate; `frames` what the OCR emits per frame."""
    clean = lambda text: [list(text)] * 3
    return [
        {'id': 'watched-clean', 'truth': 'MH49BE5557', 'frames': clean('MH49BE5557'),
         'expect_state': 'confirmed', 'expect_alert': True},
        {'id': 'watched-clean-2', 'truth': 'GJ05CD4321', 'frames': clean('GJ05CD4321'),
         'expect_state': 'confirmed', 'expect_alert': True},
        {'id': 'unwatched-clean', 'truth': 'DL3CAB1234', 'frames': clean('DL3CAB1234'),
         'expect_state': 'confirmed', 'expect_alert': False},
        {'id': 'watched-partial', 'truth': 'KA01AB1234', 'frames': [weak('KA01AB1234', 5, 'E'), weak('KA01AB1234', 5, 'E'),
                                                                   list('KA01AE1234')],
         'expect_state': 'partial', 'expect_alert': False, 'expect_lead': True},
        {'id': 'unwatched-partial', 'truth': 'TN09XY7777', 'frames': [weak('TN09XY7777', 6, '1'), weak('TN09XY7777', 6, '1'),
                                                                     list('TN09XY1777')],
         'expect_state': 'partial', 'expect_alert': False, 'expect_lead': False},
        {'id': 'partial-false-lead', 'truth': 'KA01AB1239', 'frames': [weak('KA01AB1239', 9, '4'), weak('KA01AB1239', 9, '4'),
                                                                      list('KA01AB1234')],
         'expect_state': 'partial', 'expect_alert': False, 'expect_lead': False,
         'note': 'Pattern also fits a watched plate that is NOT this vehicle: a false review lead.'},
        {'id': 'unreadable', 'truth': 'UP16ZZ9999', 'frames': None, 'expect_state': 'unreadable', 'expect_alert': False},
        {'id': 'length-disputed-watched', 'truth': 'MH49BE5557',
         'frames': [list('MH49BE5557'), list('MH49BE557'), list('MH49BE557'), list('MH49BE5557')],
         'expect_state': None, 'expect_alert': False, 'note': 'Frames disagree on length; must not alert or lead.'},
    ]


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


class Backend:
    def __init__(self, directory, python):
        self.port, self.python, self.directory = free_port(), python, Path(directory)
        self.process = None
        self.env = dict(os.environ, DATABASE_URL=f'sqlite:///{self.directory}/replay.db', EDGE_API_KEY=EDGE_KEY,
                        OPERATOR_API_KEY=OPERATOR_KEY, WATCHLIST_HMAC_KEY=HMAC_KEY, DEMO_MODE='false',
                        REVIEWER_TOKENS='replay-reviewer:' + __import__('hashlib').sha256(REVIEWER_TOKEN.encode()).hexdigest())

    @property
    def url(self):
        return f'http://127.0.0.1:{self.port}'

    def start(self):
        self.process = subprocess.Popen([self.python, '-m', 'uvicorn', 'app.main:app', '--port', str(self.port),
                                         '--log-level', 'warning'], cwd=ROOT / 'backend', env=self.env,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                self.call('GET', '/v1/healthz')
                return
            except Exception:
                time.sleep(.1)
        raise RuntimeError('Backend did not start')

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait(10)
            self.process = None

    def call(self, method, path, body=None, headers=None):
        request = urllib.request.Request(self.url + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                         headers=dict({'Content-Type': 'application/json'}, **(headers or {})))
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)

    def operator(self, method, path, body=None):
        return self.call(method, path, body, {'X-Operator-Key': OPERATOR_KEY, 'X-Reviewer-Token': REVIEWER_TOKEN})


def run(cases, *, partial_matching, python, offline_window=True):
    from ctc_fixtures import TOKENS, rows_for
    import numpy as np
    from types import SimpleNamespace
    from sentinel_edge.experimental_detectors import VehicleTrack
    from sentinel_edge.plate_options import PlateOptions
    from sentinel_edge.privacy import plate_hmac
    from sentinel_edge.shared_delivery import SharedDelivery
    from sentinel_edge.tracked_ocr import TrackedOCR
    from sentinel_edge.watchlist_snapshot import WatchlistSnapshot

    directory = tempfile.TemporaryDirectory()
    backend = Backend(directory.name, python)
    backend.start()
    try:
        for plate in WATCHED:
            backend.operator('POST', '/v1/watchlist', {'plate': plate, 'reason': 'Seeded replay', 'severity': 'high'})
        options = PlateOptions.from_env({'SENTINEL_PROFILE': 'candidate',
                                         'SENTINEL_PARTIAL_WATCHLIST': 'true' if partial_matching else 'false'})
        config = SimpleNamespace(db_path=Path(directory.name) / 'edge.sqlite3', central_api_url=backend.url,
                                 edge_api_key=EDGE_KEY, edge_id='replay-edge', hmac_key=HMAC_KEY)
        probe = SharedDelivery(config, 'replay', options=options)
        snapshot = WatchlistSnapshot(probe.client, max_age_seconds=900) if partial_matching else None
        if snapshot is not None:
            assert snapshot.refresh(), 'watchlist snapshot fetch failed'
        delivery = SharedDelivery(config, 'replay', options=options, snapshot=snapshot)
        history, case_of, current = [], {}, {'case': None}
        original_put = delivery.outbox.put

        def recording_put(endpoint, payload):
            history.append((endpoint, payload))
            if endpoint == '/v1/recognition-results':
                case_of[payload['passage_id']] = current['case']
            original_put(endpoint, payload)
        delivery.outbox.put = recording_put
        script = {'frames': None, 'calls': 0}

        def probabilities(crop):
            frames = script['frames']
            spec = frames[min(script['calls'], len(frames) - 1)]
            script['calls'] += 1
            return rows_for(spec)
        engine = TrackedOCR(SimpleNamespace(tokens=TOKENS, probabilities=probabilities),
                            SimpleNamespace(box=lambda image: None), options=options,
                            on_accepted=lambda track, candidate, pts: delivery.accept('cam-replay', track, candidate, pts),
                            on_finalized=lambda outcome: delivery.finalized('cam-replay', outcome))
        rng = np.random.default_rng(7)
        clock = 0
        for number, case in enumerate(cases):
            frames = case['frames']
            current['case'] = case
            engine.detector = SimpleNamespace(box=(lambda image: (5, 5, 150, 35)) if frames else (lambda image: None))
            script.update(frames=frames or [list('AA00AA0000')], calls=0)
            track = VehicleTrack(f'T{number}', (0, 0, 180, 60), 2)
            for index in range(len(frames) if frames else 3):
                image = rng.integers(0, 255, (60, 180, 3), dtype=np.uint8)
                engine.process(image, [track], clock + index * 500)
            clock += 30_000
            engine.process(rng.integers(0, 255, (60, 180, 3), dtype=np.uint8), [], clock)  # Passage expires.
            clock += 30_000
        engine.flush()

        queued = delivery.outbox.depth()
        failures = 0
        if offline_window:
            backend.stop()      # Nothing may be lost while central is down.
            delivery.client.flush_once()
            still_queued = delivery.outbox.depth()
            backend.start()
        else:
            still_queued = queued
        stalls = 0
        while delivery.outbox.depth():
            if not delivery.client.flush_once(500):
                stalls += 1
                if stalls > 3:
                    break
                time.sleep(.2)
        failures += delivery.outbox.depth() + delivery.outbox.dead_letter_count()
        alerts_first = backend.operator('GET', '/v1/alerts?limit=500')
        # Resend every event again, as after a lost acknowledgement: nothing may duplicate.
        replayed = 0
        for endpoint, payload in history:
            try:
                backend.call('POST', endpoint, payload, {'X-Edge-Key': EDGE_KEY})
                replayed += 1
            except Exception:
                failures += 1
        alerts = backend.operator('GET', '/v1/alerts?limit=500')
        queue = backend.operator('GET', '/v1/review/queue?status=pending')
        counts = backend.operator('GET', '/v1/recognition-results?limit=500')['counts']

        by_plate = Counter(item['masked_plate'] for item in alerts)
        expected_alert_plates = Counter(mask(case['truth']) for case in cases if case['expect_alert'])
        metrics = {
            'cases': len(cases), 'events_queued': len(history), 'events_redelivered_for_idempotency': replayed,
            'missed_confirmed_alerts': {'count': sum((expected_alert_plates - by_plate).values()),
                                        'of': sum(case['expect_alert'] for case in cases)},
            'false_confirmed_alerts': {'count': sum((by_plate - expected_alert_plates).values()),
                                       'of': sum(not case['expect_alert'] for case in cases)},
            'duplicate_alerts': len(alerts) - len({item['sighting_id'] for item in alerts}),
            'alerts_changed_by_redelivery': len(alerts) - len(alerts_first),
            'delivery_failures': failures, 'queued_while_central_down': still_queued,
            'outbox_depth_at_end': delivery.outbox.depth(), 'dead_letters': delivery.outbox.dead_letter_count(),
            'alerts_total': len(alerts), 'reviews_pending': len(queue['items']),
            'alerts_from_pending_reviews': 0 if len(alerts) == len(alerts_first) == sum(case['expect_alert'] for case in cases) else 'see alerts_total',
            'result_states_persisted': counts,
            'result_events_expected': sum(1 for endpoint, _ in history if endpoint == '/v1/recognition-results'),
        }
        metrics['result_events_persisted'] = sum(counts.values())
        if partial_matching:
            pending = {item['passage_id']: item for item in queue['items']}
            returned = true_leads = false_leads = missed_leads = eligible = 0
            for endpoint, payload in history:
                if endpoint != '/v1/recognition-results' or payload['result_state'] != 'partial':
                    continue
                case = case_of[payload['passage_id']] if payload['passage_id'] in case_of else None
                if case is None:
                    continue
                lead = payload.get('watchlist_lead')
                returned_here = bool(lead) and pending.get(payload['passage_id'], {}).get('watchlist_lead', False)
                eligible += case['truth'] in WATCHED
                if returned_here:
                    returned += 1
                    if plate_hmac(case['truth'], HMAC_KEY) in lead['candidate_hmacs']:
                        true_leads += 1
                    else:
                        false_leads += 1
                elif case['truth'] in WATCHED:
                    missed_leads += 1
            metrics['partial_matching'] = {
                'enabled': True, 'leads_returned': returned,
                'review_recall': {'count': true_leads, 'of': eligible},
                'review_precision': {'count': true_leads, 'of': returned},
                'false_review_leads': false_leads, 'missed_review_leads': missed_leads,
                'note': 'A lead only opens a pending review. No lead produced an alert.'}
        else:
            metrics['partial_matching'] = {'enabled': False, 'review_recall': 'not applicable',
                                           'review_precision': 'not applicable', 'false_review_leads': 'not applicable'}
        metrics['per_case'] = [{'id': c['id'], 'expected_state': c['expect_state'],
                                'actual_states': sorted({p['result_state'] for e, p in history
                                                         if e == '/v1/recognition-results' and case_of.get(p['passage_id']) is c})}
                               for c in cases]
        return metrics
    finally:
        backend.stop()
        directory.cleanup()


def mask(plate):
    return plate[:2] + '*' * (len(plate) - 4) + plate[-2:]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--partial-matching', action='store_true')
    parser.add_argument('--python', default=sys.executable, help='Interpreter that has the backend requirements')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Reports are never overwritten.')
    report = {'kind': 'seeded-watchlist-replay', 'data': 'SYNTHETIC scripted passages; not real-world validation',
              'metrics': run(default_cases(), partial_matching=args.partial_matching, python=args.python)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report['metrics'], indent=2))


if __name__ == '__main__':
    main()
