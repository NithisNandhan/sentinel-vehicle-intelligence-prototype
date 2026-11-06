"""Result-state events, delivery, evidence retention and the watchlist snapshot.

Plates here are SYNTHETIC. No real camera data is used.
"""
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import urllib.error

import numpy as np

from ctc_fixtures import reader_with
from sentinel_edge.api import CentralClient
from sentinel_edge.events import recognition_result
from sentinel_edge.evidence_store import EvidenceStore, review_record
from sentinel_edge.outbox import Outbox
from sentinel_edge.passages import PassageOutcome
from sentinel_edge.plate_options import PlateOptions
from sentinel_edge.privacy import plate_hmac
from sentinel_edge.result_state import classify
from sentinel_edge.shared_delivery import SharedDelivery
from sentinel_edge.watchlist_snapshot import WatchlistSnapshot, partial_leads

KEY = 'test-hmac-key'
PLATE = 'MH49BE5557'
COMMON = dict(edge_id='edge-1', camera_id='cam01', passage_id='p1', source_mode='live',
              observed_at='2026-11-06T10:00:00Z', last_observed_at='2026-11-06T10:00:02Z', event_id='e1')


def config(directory):
    return SimpleNamespace(db_path=Path(directory) / 'outbox.sqlite3', central_api_url='http://127.0.0.1:9',
                           edge_api_key='edge-key', edge_id='edge-1', hmac_key=KEY)


def outcome(frames, **extra):
    result = classify(reader_with(frames))
    crop = np.full((30, 120, 3), 200, np.uint8)
    values = dict(evidence=[{'pts_ms': 0, 'crop': crop, 'quality': .8}], frame_jpeg=b'\xff\xd8jpeg',
                  first_box=(0, 0, 100, 60), last_box=(10, 0, 110, 60))
    values.update(extra)
    return PassageOutcome('passage-1', ('3',), 2, 0, 1200, result, 'expired', **values)


def weak(position, rival):
    spec = list(PLATE)
    spec[position] = (PLATE[position], .45, rival)
    return spec


class EventContractTests(unittest.TestCase):
    def test_confirmed_carries_fingerprint_and_masked_plate_only(self):
        event = recognition_result(result_state='confirmed', plate=PLATE, hmac_key=KEY, **COMMON)
        self.assertEqual((event['schema_version'], event['masked_plate']), ('2.0', 'MH******57'))
        self.assertEqual(event['plate_hmac'], plate_hmac(PLATE, KEY))
        self.assertNotIn(PLATE, json.dumps(event))

    def test_partial_and_unreadable_never_carry_fingerprints_or_text(self):
        partial = recognition_result(result_state='partial', masked_pattern='MH******57', plate_length=10,
                                     unknown_positions=(5,), reason='uncertain_characters', **COMMON)
        self.assertNotIn('plate_hmac', partial)
        unreadable = recognition_result(result_state='unreadable', reason='no_plate_found', **COMMON)
        self.assertNotIn('masked_plate', unreadable)
        self.assertNotIn('plate_hmac', unreadable)

    def test_state_specific_validation(self):
        bad = [dict(result_state='confirmed', plate=PLATE, hmac_key=None),
               dict(result_state='confirmed', plate=PLATE, hmac_key=KEY, reason='x'),
               dict(result_state='partial', masked_pattern='MH49B?5557', plate_length=10, reason='r'),
               dict(result_state='partial', masked_pattern='MH******57', plate_length=9, reason='r'),
               dict(result_state='partial', masked_pattern='MH******57', plate_length=10, unknown_positions=(12,), reason='r'),
               dict(result_state='partial', masked_pattern='MH******57', plate_length=10),
               dict(result_state='unreadable'),
               dict(result_state='unreadable', reason='r', plate=PLATE),
               dict(result_state='maybe', reason='r')]
        for fields in bad:
            with self.assertRaises(ValueError, msg=fields):
                recognition_result(**fields, **COMMON)
        with self.assertRaises(ValueError):
            recognition_result(result_state='unreadable', reason='r', **dict(COMMON, source_mode='recorded'))


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.options = PlateOptions.from_env({'SENTINEL_PROFILE': 'candidate', 'SENTINEL_PARTIAL_WATCHLIST': 'true'})

    def delivery(self, **kwargs):
        return SharedDelivery(config(self.directory.name), 'general', options=self.options, **kwargs)

    def test_partial_with_evidence_and_lead_is_queued_once(self):
        snapshot = SimpleNamespace(entries=lambda: frozenset({plate_hmac(PLATE, KEY)}), max_age=900)
        delivery = self.delivery(evidence=EvidenceStore(), snapshot=snapshot)
        frames = [weak(5, 'F'), weak(5, 'F'), list('MH49BF5557')]
        with unittest.mock.patch.dict('os.environ', {'SENTINEL_REVIEW_EVIDENCE_LIVE': 'true'}):
            event = delivery.finalized('cam01', outcome(frames))
        self.assertEqual(event['result_state'], 'partial')
        self.assertEqual(event['watchlist_lead']['candidate_hmacs'], [plate_hmac(PLATE, KEY)])
        self.assertIsNotNone(event['evidence_ref'])
        record = delivery.evidence.get(event['evidence_ref'])
        self.assertEqual(record['pattern'], 'MH49B?5557')
        self.assertNotIn('_images', record)
        queued = delivery.outbox.next_batch()
        self.assertEqual([(endpoint, payload['passage_id']) for _, endpoint, payload in queued],
                         [('/v1/recognition-results', 'passage-1')])
        delivery.outbox.put('/v1/recognition-results', event)  # Retry: idempotent by event_id.
        self.assertEqual(delivery.outbox.depth(), 1)
        self.assertNotIn('MH49B', json.dumps(event))

    def test_live_evidence_stays_on_the_edge_by_default(self):
        delivery = self.delivery(evidence=EvidenceStore())
        event = delivery.finalized('cam01', outcome([weak(5, 'F'), weak(5, 'F'), list('MH49BF5557')]))
        self.assertIsNone(event['evidence_ref'])

    def test_fragments_and_compatible_partials_are_deduplicated(self):
        delivery = self.delivery()
        unreadable = PassageOutcome('a', ('1',), 2, 0, 1000, classify(None), 'expired',
                                    first_box=(0, 0, 100, 60), last_box=(10, 0, 110, 60))
        fragment = PassageOutcome('b', ('2',), 2, 1500, 3000, classify(None), 'expired',
                                  first_box=(14, 0, 114, 60), last_box=(20, 0, 120, 60))
        self.assertIsNotNone(delivery.finalized('cam01', unreadable))
        self.assertIsNone(delivery.finalized('cam01', fragment))
        self.assertEqual(delivery.result_metrics['suppressed_tracker_fragment'], 1)
        other_camera = PassageOutcome('c', ('2',), 2, 1500, 3000, classify(None), 'expired',
                                      first_box=(14, 0, 114, 60), last_box=(20, 0, 120, 60))
        self.assertIsNotNone(delivery.finalized('cam02', other_camera))

    def test_confirmed_links_to_its_sighting(self):
        delivery = self.delivery()
        candidate = SimpleNamespace(plate=PLATE, score=.95)
        delivery.accept('cam01', '3', candidate, 1000)
        sighting_id = delivery.outbox.next_batch()[0][0]
        event = delivery.finalized('cam01', outcome([list(PLATE)] * 3, confirmed_plate=PLATE))
        self.assertEqual((event['result_state'], event['sighting_event_id']), ('confirmed', sighting_id))

    def test_permanent_rejection_is_dead_lettered_not_retried_forever(self):
        outbox = Outbox(Path(self.directory.name) / 'dead.sqlite3')
        outbox.put('/v1/recognition-results', {'event_id': 'bad'})
        outbox.put('/v1/recognition-results', {'event_id': 'good'})
        client = CentralClient('http://127.0.0.1:9', 'k', outbox, edge_id='edge-1')
        posted = []
        def post(endpoint, payload):
            if payload['event_id'] == 'bad':
                raise urllib.error.HTTPError('u', 422, 'invalid', {}, io.BytesIO())
            posted.append(payload['event_id'])
        client.post = post
        self.assertEqual(client.flush_once(), 1)
        self.assertEqual((posted, outbox.depth(), outbox.dead_letter_count()), (['good'], 0, 1))
        self.assertEqual(outbox.requeue_dead_letters(), 1)
        self.assertEqual(outbox.depth(), 1)
        offline = CentralClient('http://127.0.0.1:9', 'k', outbox)
        offline.post = lambda *_: (_ for _ in ()).throw(urllib.error.URLError('down'))
        self.assertEqual(offline.flush_once(), 0)
        self.assertEqual(outbox.depth(), 1, 'network failure keeps the event queued')
        self.assertEqual(client._headers()['X-Edge-Id'], 'edge-1')


class SnapshotTests(unittest.TestCase):
    def test_expiry_revocation_and_minimal_contents(self):
        now = [1000.]
        entries = [{'plate_hmac': 'a' * 64, 'reason': 'not kept'}, {'plate_hmac': 'short'}]
        expires = (datetime.fromtimestamp(1000, timezone.utc) + timedelta(seconds=60)).isoformat()
        client = SimpleNamespace(watchlist_snapshot=lambda: {'version': 3, 'entries': entries, 'expires_at': expires})
        snapshot = WatchlistSnapshot(client, max_age_seconds=300, clock=lambda: now[0])
        self.assertIsNone(snapshot.entries(), 'nothing before the first fetch')
        self.assertTrue(snapshot.refresh())
        self.assertEqual(snapshot.entries(), frozenset({'a' * 64}))
        now[0] += 61
        self.assertIsNone(snapshot.entries(), 'server expiry is enforced')
        def revoked():
            raise urllib.error.HTTPError('u', 403, 'revoked', {}, io.BytesIO())
        snapshot.client = SimpleNamespace(watchlist_snapshot=revoked)
        self.assertFalse(snapshot.refresh())
        self.assertTrue(snapshot.status()['revoked'])
        self.assertIsNone(snapshot.entries())
        with self.assertRaises(ValueError):
            WatchlistSnapshot(client, max_age_seconds=10)

    def test_partial_leads_are_bounded_and_never_guess_length(self):
        fingerprints = frozenset({plate_hmac(PLATE, KEY), plate_hmac('MH49BF5557', KEY)})
        lead = partial_leads('MH49B?5557', KEY, fingerprints)
        self.assertEqual(sorted(lead['candidate_hmacs']), sorted(fingerprints))
        self.assertEqual(lead['candidates_considered'], 26)
        self.assertEqual(partial_leads('MH49B?5557', KEY, None), {'skipped': 'snapshot_unavailable'})
        self.assertEqual(partial_leads('MH49B?5557', KEY, fingerprints, length_certain=False)['skipped'],
                         'length_uncertain')
        self.assertEqual(partial_leads('M??9B?5557', KEY, fingerprints)['skipped'], 'unknown_positions_out_of_bounds')
        self.assertEqual(partial_leads('MH49B?5557', KEY, frozenset({plate_hmac('KA01AB1234', KEY)}))['candidate_hmacs'], [])


class EvidenceStoreTests(unittest.TestCase):
    def test_ttl_capacity_and_unguessable_references(self):
        now = [0.]
        store = EvidenceStore(max_items=2, ttl_seconds=60, clock=lambda: now[0])
        first = store.put({'a': 1, '_images': [b'x']})
        self.assertGreaterEqual(len(first), 16)
        self.assertEqual(store.get(first), {'a': 1})
        store.put({'b': 1}); store.put({'c': 1})
        self.assertIsNone(store.get(first), 'capacity evicts the oldest')
        now[0] = 1000
        self.assertEqual(len(store), 0, 'time-to-live evicts everything')
        self.assertIsNone(store.get('short'))

    def test_review_record_contents(self):
        record = review_record(outcome([weak(5, 'F'), weak(5, 'F'), list('MH49BF5557')]), camera_id='cam01',
                               source_mode='recorded')
        self.assertEqual(record['pattern'], 'MH49B?5557')
        self.assertEqual(len(record['crops']), 1)
        self.assertTrue(record['frame_jpeg_base64'])
        self.assertIn('Do not export', record['notice'])


import unittest.mock  # noqa: E402

if __name__ == '__main__':
    unittest.main()
