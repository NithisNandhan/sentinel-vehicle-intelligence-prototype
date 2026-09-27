import base64
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from sentinel_edge.ctc_fusion import CTCPlateReader
from sentinel_edge.live_reconstruction import ReconstructionState, update
from sentinel_edge.reconstruction import CropBank
from sentinel_edge.recognition_contract import operator_snapshot
from sentinel_edge.tracked_ocr import TrackedOCR

TOKENS = ('', *'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
TEXT = 'AP01AB1234'


def posterior(text=TEXT):
    rows = []
    for char in text:
        row = [0.] * len(TOKENS)
        row[TOKENS.index(char)] = 1.
        rows.append(row)
    return rows


def plate(index):
    image = np.full((48, 160, 3), 220, np.uint8)
    cv2.putText(image, 'AP01AB1234', (8, 32), cv2.FONT_HERSHEY_SIMPLEX, .65, (20, 20, 20), 2)
    image[0, index] = index
    return image


class LiveReconstructionTests(unittest.TestCase):
    def engine(self, callback=None):
        return TrackedOCR(SimpleNamespace(tokens=TOKENS, probabilities=lambda image:posterior()),
                          SimpleNamespace(box=lambda image:(0, 0, 160, 48)),
                          debug=True, reconstruction=True, on_accepted=callback)

    def run_frames(self, engine, count=3):
        track = SimpleNamespace(track_id='1', box=(0, 0, 160, 48), vehicle_class=2)
        for i in range(count):
            engine.process(plate(i), [track], i * 500)
        return track

    def test_live_path_reconstructs_without_adding_votes_or_changing_native_pixels(self):
        accepted = []
        engine = self.engine(lambda *args:accepted.append(args))
        self.run_frames(engine)
        debug = engine.debug_snapshot()
        track = debug['tracks'][0]
        result = track['reconstruction']
        self.assertEqual(result['status'], 'reconstructed')
        self.assertEqual(result['contributing_originals'], 3)
        self.assertEqual(result['extra_observations'], 0)
        self.assertEqual(result['text'], TEXT)
        self.assertEqual(track['observations'], 3)
        self.assertEqual(track['candidates'][0]['observations'], 3)
        self.assertEqual(len(accepted), 1)
        original = cv2.imdecode(np.frombuffer(base64.b64decode(track['frames'][0]['image']), np.uint8), cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(original, plate(0))
        self.assertGreater(debug['stage_ms']['reconstruction'], 0)
        safe = operator_snapshot({'status':'live', 'camera_id':'cam01', 'debug':debug})
        self.assertNotIn(TEXT, json.dumps(safe))
        self.assertNotIn('image', json.dumps(safe))

    def test_failure_is_redacted_and_original_ocr_still_accepts(self):
        for failure in (None, RuntimeError('private secret')):
            engine = self.engine()
            kwargs = {'return_value':None} if failure is None else {'side_effect':failure}
            with patch('sentinel_edge.live_reconstruction.reconstruct', **kwargs):
                self.run_frames(engine)
            row = engine.debug_snapshot()['tracks'][0]
            self.assertEqual(row['recognition_state'], 'model_accepted')
            self.assertNotIn('private secret', json.dumps(row))
            self.assertNotIn('image', row['reconstruction'])
            self.assertEqual(row['reconstruction']['status'], 'alignment_rejected' if failure is None else 'error')

    def test_conflicting_processed_reading_is_not_delivered(self):
        accepted = []
        engine = self.engine(lambda *args:accepted.append(args))
        calls = []
        def probabilities(image):
            calls.append(1)
            return posterior('AP01AB9999' if len(calls) == 4 else TEXT)
        engine.ocr.probabilities = probabilities
        self.run_frames(engine)
        row = engine.debug_snapshot()['tracks'][0]
        self.assertEqual(len(accepted), 0)
        self.assertEqual(row['recognition_state'], 'uncertain')
        self.assertIn('disagree', ' '.join(row['gate_reasons']))

    def test_original_scores_and_counts_are_unchanged_by_derived_suggestions(self):
        reader = CTCPlateReader(TOKENS)
        for i in range(3):reader.add(posterior(), timestamp_ms=i*500, source_id=str(i))
        before = reader.accepted()
        reader.propose((TEXT, 'AP01AB9999'))
        self.assertEqual(reader.accepted(), before)
        self.assertEqual(reader.observation_count, 3)
        self.assertNotIn('AP01AB9999', [c.plate for c in reader.candidates()])

    def test_suggestion_can_enter_search_but_is_scored_only_against_originals(self):
        reader = CTCPlateReader(TOKENS)
        for i in range(3):reader.add(posterior(), timestamp_ms=i*500, source_id=str(i))
        # Simulate a sequence pruned from the original frame beams, not absent pixels.
        for sample in reader.samples:sample[3].clear()
        reader._cache = None
        self.assertEqual(reader.candidates(), [])
        reader.propose((TEXT,))
        self.assertEqual(reader.accepted().plate, TEXT)
        self.assertEqual(reader.accepted().observations, 3)

    def test_duplicate_crop_does_not_run_reconstruction(self):
        engine = self.engine()
        track = SimpleNamespace(track_id='1', box=(0, 0, 160, 48), vehicle_class=2)
        for stamp in (0, 500, 1000):engine.process(plate(0), [track], stamp)
        self.assertEqual(engine.metrics['reconstruction_attempts'], 0)
        self.assertEqual(engine.histories['1'].reader.observation_count, 1)

    def test_gap_reset_clears_reconstruction_and_full_text(self):
        engine = self.engine()
        track = self.run_frames(engine)
        engine.detector.box = lambda image:None
        engine.process(plate(4), [track], 4000)
        row = engine.debug_snapshot()['tracks'][0]
        self.assertNotIn(TEXT, json.dumps(row))
        self.assertNotIn('image', row['reconstruction'])
        self.assertEqual(row['observations'], 0)
        engine.reset()
        self.assertEqual(engine.crop_bank.bytes_used, 0)

    def test_eviction_clears_processed_preview(self):
        engine = self.engine()
        self.run_frames(engine)
        engine.crop_bank.max_bytes = plate(0).nbytes
        other = SimpleNamespace(track_id='2', box=(0, 0, 160, 48), vehicle_class=2)
        engine.process(plate(5), [other], 1500)
        self.assertIsNone(engine.histories['1'].reconstruction)
        self.assertEqual(engine.histories['1'].reader._proposals, ())

    def test_attempt_budget_clears_stale_derived_result(self):
        bank = CropBank()
        reader = CTCPlateReader(TOKENS)
        state = ReconstructionState()
        for i in range(9):
            bank.add('1', plate(i), i*500)
            reader.add(posterior(), timestamp_ms=i*500, source_id=str(i))
            update(state, reader, bank.get('1'), SimpleNamespace(tokens=TOKENS, probabilities=lambda image:posterior()), i*500)
        self.assertEqual(state.attempts, 6)
        self.assertEqual(state.status, 'budget_exhausted')
        self.assertIsNone(state.result)
        self.assertEqual(reader._proposals, ())


if __name__ == '__main__':unittest.main()
