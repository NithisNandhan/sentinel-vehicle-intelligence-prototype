"""The running vehicles-seen list: bounded, masked for the operator, full text only where allowed."""
import json
from types import SimpleNamespace
import unittest

import numpy as np

from sentinel_edge.passed_vehicles import PassedVehicles
from sentinel_edge.privacy import mask_plate, plate_hmac
from sentinel_edge.recognition_contract import operator_snapshot

TEXT, KEY = 'TS07FS8429', 'test-hmac-key'
READING, UNCERTAIN, ACCEPTED = 'collecting distinct frames', 'uncertain', 'accepted by model — NOT verified'


def track(track_id='5', vehicle_class=2, confidence=.39):
    return SimpleNamespace(track_id=track_id, box=(0, 0, 200, 100), vehicle_class=vehicle_class, confidence=confidence)


def display(track_id='5', status=READING, observations=1, masked=None):
    return {'track_id': track_id, 'ocr_status': status, 'observations': observations, 'masked_plate': masked,
            'crop': np.full((30, 120, 3), 200, np.uint8), 'box': (10, 60, 130, 90)}


class PassedVehiclesTests(unittest.TestCase):
    def test_a_vehicle_is_listed_as_soon_as_it_appears_and_updates_as_evidence_arrives(self):
        seen = PassedVehicles(keep_text=True, hmac_key=KEY)
        seen.update(0, 1000, [track()], [], {})
        row = seen.engineering()[0]
        self.assertEqual((row['vehicle_class'], row['confidence'], row['observations'], row['reading']), ('car', .39, 0, None))
        seen.update(0, 1500, [track(confidence=.61)], [display(observations=1)], {'5': TEXT})
        seen.update(0, 2500, [track(confidence=.2)], [display(status=UNCERTAIN, observations=3, masked=mask_plate(TEXT))], {'5': TEXT})
        row = seen.engineering()[0]
        self.assertEqual((row['first_ms'], row['last_ms'], row['state'], row['observations']), (1000, 2500, 'uncertain', 3))
        self.assertEqual((row['confidence'], row['reading']), (.61, TEXT))   # Best detector score seen, not the last.
        self.assertTrue(row['thumb'])
        self.assertEqual(len(seen.engineering()), 1)

    def test_newest_first_a_reused_track_id_after_a_scene_cut_is_a_new_vehicle_and_the_list_is_bounded(self):
        seen = PassedVehicles(limit=3)
        for index in range(5):
            seen.update(0, index*1000, [track(str(index))], [], {})
        self.assertEqual([row['track_id'] for row in seen.engineering()], ['4', '3', '2'])
        seen.update(1, 9000, [track('4')], [], {})
        self.assertEqual([row['key'] for row in seen.engineering()][:2], ['1-4', '0-4'])

    def test_operator_view_is_masked_and_never_carries_text_or_pixels(self):
        seen = PassedVehicles(keep_text=True, hmac_key=KEY)
        seen.watchlist = {plate_hmac(TEXT, KEY)}
        seen.update(0, 0, [track()], [display(status=UNCERTAIN, observations=3, masked=mask_plate(TEXT))], {'5': TEXT})
        masked = seen.masked()
        self.assertEqual(masked[0]['masked_plate'], 'TS******29')
        self.assertIsNone(masked[0]['watchlist'])            # Not accepted: no flag reaches the operator.
        self.assertEqual(seen.engineering()[0]['watchlist'], 'possible_match')
        text = json.dumps(masked)
        self.assertNotIn(TEXT, text); self.assertNotIn('thumb', text); self.assertNotIn('reading', text)
        seen.update(0, 500, [track()], [display(status=ACCEPTED, observations=4, masked=mask_plate(TEXT))], {'5': TEXT})
        self.assertEqual(seen.masked()[0]['watchlist'], 'possible_match')
        safe = operator_snapshot({'status': 'processing', 'source_mode': 'recorded', 'camera_id': 'rec-1',
                                  'vehicles_seen': seen.masked(), 'debug': {'passed': seen.engineering(), 'tracks': []}})
        self.assertEqual(safe['vehicles_seen'][0]['masked_plate'], 'TS******29')
        self.assertNotIn(TEXT, json.dumps(safe)); self.assertNotIn('thumb', json.dumps(safe))

    def test_watchlist_is_unknown_until_fetched_and_text_is_withheld_outside_the_protected_view(self):
        seen = PassedVehicles(keep_text=False, hmac_key=KEY)
        seen.update(0, 0, [track()], [display(status=ACCEPTED, observations=3, masked=mask_plate(TEXT))], {'5': TEXT})
        self.assertIsNone(seen.engineering()[0]['watchlist'])   # Central not reachable yet: unknown, not "no match".
        self.assertIsNone(seen.engineering()[0]['reading'])
        seen.watchlist = set()
        self.assertEqual(seen.masked()[0]['watchlist'], 'no_match')
