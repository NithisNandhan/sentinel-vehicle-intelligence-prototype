from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

from sentinel_edge.temporal import TemporalPlateReader, indian_plate_format
from sentinel_edge.vision import VisionPipeline


ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"


def probabilities(text, confidence=0.96, overrides=None):
    rows = []
    for position, character in enumerate(text.ljust(12, "_")):
        row = [(1 - confidence) / (len(ALPHABET) - 1)] * len(ALPHABET)
        row[ALPHABET.index(character)] = confidence
        for token, probability in (overrides or {}).get(position, {}).items():
            row[ALPHABET.index(token)] = probability
        total = sum(row)
        rows.append([value / total for value in row])
    return rows


class TemporalTests(unittest.TestCase):
    def reader(self):
        return TemporalPlateReader(ALPHABET, "_", min_observations=3)

    def test_indian_formats_and_unknowns(self):
        self.assertEqual(indian_plate_format("GJ01AB1234"), "state")
        self.assertEqual(indian_plate_format("24BH1234AB"), "bharat")
        self.assertIsNone(indian_plate_format("1234567"))
        self.assertIsNone(indian_plate_format("GJ01AB1234567"))

    def test_single_high_score_does_not_emit(self):
        reader = self.reader()
        reader.add(probabilities("GJ01AB1234"), timestamp_ms=1000)
        self.assertIsNone(reader.accepted(0.85))

    def test_identical_timestamps_do_not_create_support(self):
        reader = self.reader()
        for _ in range(4):
            reader.add(probabilities("GJ01AB1234"), timestamp_ms=1000)
        self.assertEqual(reader.observation_count, 1)
        self.assertIsNone(reader.accepted(0.85))

    def test_consistent_three_observations_accept(self):
        reader = self.reader()
        for index in range(3):
            reader.add(probabilities("GJ01AB1234"), timestamp_ms=1000 + index * 300)
        result = reader.accepted(0.85)
        self.assertEqual(result.plate, "GJ01AB1234")
        self.assertGreater(result.score, 0.9)

    def test_fuses_complementary_errors_without_identical_strings(self):
        reader = self.reader()
        # Each frame has a different weak wrong digit. Two other observations
        # provide strong evidence for that position; no complete strings agree.
        for index, position in enumerate((7, 8, 9)):
            rows = probabilities("GJ01AB1234", 0.99)
            correct = ALPHABET.index("GJ01AB1234"[position])
            rows[position] = [0.0] * len(ALPHABET)
            rows[position][correct] = 0.48
            rows[position][ALPHABET.index("9")] = 0.52
            reader.add(rows, timestamp_ms=1000 + index * 300)
        result = reader.accepted(0.85)
        self.assertEqual(result.plate, "GJ01AB1234")

    def test_format_does_not_inflate_weak_visual_evidence(self):
        reader = self.reader()
        for index in range(3):
            rows = probabilities("GJ01AB1234")
            rows[0] = [1 / len(ALPHABET)] * len(ALPHABET)
            reader.add(rows, timestamp_ms=1000 + index * 300)
        self.assertIsNone(reader.accepted(0.85))

    def test_equal_numeric_alternatives_stay_unresolved(self):
        reader = self.reader()
        for index in range(3):
            rows = probabilities("GJ01AB1234")
            rows[9] = [0.0] * len(ALPHABET)
            rows[9][ALPHABET.index("4")] = 0.5
            rows[9][ALPHABET.index("8")] = 0.5
            reader.add(rows, timestamp_ms=1000 + index * 300)
        self.assertIsNone(reader.accepted(0.85))
        self.assertLessEqual(len(reader.candidates()), 200)

    def test_window_is_bounded_and_reset_discards_evidence(self):
        reader = self.reader()
        for index in range(40):
            reader.add(probabilities("GJ01AB1234"), timestamp_ms=index * 300)
        self.assertEqual(reader.observation_count, 12)
        reader.reset()
        self.assertEqual(reader.observation_count, 0)

    def test_repetition_does_not_inflate_uncalibrated_scores(self):
        reader = self.reader()
        rows = probabilities("GJ01AB1234", 0.82)
        reader.add(rows, timestamp_ms=1000)
        original = reader.candidates()[0].score
        for index in range(1, 10):
            reader.add(rows, timestamp_ms=1000 + index * 300)
        self.assertAlmostEqual(reader.candidates()[0].score, original)
        self.assertIsNone(reader.accepted(0.85))

    def test_invalid_probabilities_rejected(self):
        reader = self.reader()
        for invalid in ([], [[math.nan] * len(ALPHABET)], [[0.0] * len(ALPHABET)]):
            with self.assertRaises(ValueError):
                reader.add(invalid, timestamp_ms=1000)

    def test_invalid_evidence_bounds_rejected(self):
        for options in ({"window": 0}, {"beam_width": 201}, {"min_observations": 0},
                        {"window": 2, "min_observations": 3}):
            with self.assertRaises(ValueError):
                TemporalPlateReader(ALPHABET, "_", **options)

    def test_invalid_padding_prefix_cannot_consume_the_beam(self):
        reader = TemporalPlateReader(ALPHABET, "_", beam_width=1)
        rows = probabilities("GJ01AB1234")
        rows[0] = [0.0] * len(ALPHABET)
        rows[0][ALPHABET.index("_")] = 0.55
        rows[0][ALPHABET.index("G")] = 0.45
        for index in range(3):
            reader.add(rows, timestamp_ms=index * 300)
        self.assertEqual(reader.candidates()[0].plate, "GJ01AB1234")
        # Grammar must not upgrade the actual weak G probability into evidence.
        self.assertAlmostEqual(reader.candidates()[0].minimum_character_score, 0.45)
        self.assertIsNone(reader.accepted(0.85))

    def test_confident_prefix_does_not_hide_uncertain_trailing_padding(self):
        reader = self.reader()
        rows = probabilities("GJ01123")
        for position in range(7, len(rows)):
            rows[position] = [0.0] * len(ALPHABET)
            rows[position][ALPHABET.index("_")] = 0.55
            rows[position][ALPHABET.index("X")] = 0.45
        for index in range(3):
            reader.add(rows, timestamp_ms=index * 300)
        self.assertGreater(reader.candidates()[0].score, 0.9)
        self.assertIsNone(reader.accepted(0.85))

    def test_trailing_padding_requires_two_separate_observations(self):
        reader = self.reader()
        for index, (pad_score, quality) in enumerate(((0.99, 1.0), (0.59, 0.1), (0.59, 0.1))):
            rows = probabilities("GJ01123")
            for position in range(7, len(rows)):
                rows[position] = [0.0] * len(ALPHABET)
                rows[position][ALPHABET.index("_")] = pad_score
                rows[position][ALPHABET.index("X")] = 1 - pad_score
            reader.add(rows, timestamp_ms=index * 300, quality=quality)
        self.assertIsNone(reader.accepted(0.85))

    def test_prefix_pruning_preserves_supported_common_layouts_and_scores(self):
        for plate in ("GJ01123", "DL1CAA1234", "GJ123ABC1234", "24BH1234A", "24BH1234AB"):
            with self.subTest(plate=plate):
                reader = self.reader()
                for index in range(3):
                    reader.add(probabilities(plate), timestamp_ms=index * 300)
                result = reader.accepted(0.85)
                self.assertEqual(result.plate, plate)
                self.assertAlmostEqual(result.score, 0.96)

    def test_padding_inside_text_and_too_few_slots_cannot_be_accepted(self):
        for rows in (probabilities("GJ01_AB1234"), probabilities("GJ01123")[:6]):
            reader = self.reader()
            for index in range(3):
                reader.add(rows, timestamp_ms=index * 300)
            self.assertIsNone(reader.accepted(0.85))

    def test_unsupported_alphabet_tokens_never_become_plate_characters(self):
        alphabet = ALPHABET + "?"
        reader = TemporalPlateReader(alphabet, "_")
        rows = [[0.0] * len(alphabet) for _ in range(10)]
        for row in rows:
            row[alphabet.index("?")] = 1.0
        for index in range(3):
            reader.add(rows, timestamp_ms=index * 300)
        self.assertEqual(reader.candidates(), [])
        self.assertIsNone(reader.accepted(0.85))


class PipelineTemporalTests(unittest.TestCase):
    def test_motion_discontinuity_discards_evidence_without_resetting_budget(self):
        vision, image, box = self.pipeline([probabilities("GJ01AB1234")] * 3)
        box.xyxy = [SimpleNamespace(tolist=lambda: [0, 0, 80, 40])]
        vision.process(image, pts_ms=1000)
        vision.process(image, pts_ms=1300)
        box.xyxy = [SimpleNamespace(tolist=lambda: [180, 140, 260, 180])]
        self.assertEqual(vision.process(image, pts_ms=1600), [])
        self.assertEqual(vision.tracks["7"].reader.observation_count, 1)
        self.assertEqual(vision.tracks["7"].attempts, 3)
        self.assertEqual(vision.metrics["track_consistency_resets"], 1)

    def test_long_video_time_gap_does_not_combine_old_reads(self):
        vision, image, _ = self.pipeline([probabilities("GJ01AB1234")] * 3)
        vision.process(image, pts_ms=1000)
        vision.process(image, pts_ms=1300)
        self.assertEqual(vision.process(image, pts_ms=4000), [])
        self.assertEqual(vision.tracks["7"].reader.observation_count, 1)
    def pipeline(self, matrices):
        class Image:
            shape = (200, 300, 3)
            def __getitem__(self, _):
                return self
        box = SimpleNamespace(id=SimpleNamespace(item=lambda: 7),
                              xyxy=[SimpleNamespace(tolist=lambda: [0, 0, 200, 100])])
        class Boxes:
            id = True
            def __iter__(self):
                return iter([box])
        vision = VisionPipeline.__new__(VisionPipeline)
        vision.cv2 = SimpleNamespace(COLOR_BGR2GRAY=1, CV_64F=2,
                                     cvtColor=lambda image, _: image,
                                     Laplacian=lambda *_: SimpleNamespace(var=lambda: 100))
        vision.model = SimpleNamespace(track=lambda *_, **__: [SimpleNamespace(boxes=Boxes())])
        detection = SimpleNamespace(confidence=0.9,
                                    bounding_box=SimpleNamespace(x1=40, y1=120, x2=120, y2=145))
        cfg = SimpleNamespace(alphabet=ALPHABET, pad_char="_")
        vision.alpr = SimpleNamespace(detector=SimpleNamespace(predict=lambda _: [detection]),
                                      ocr=SimpleNamespace(ocr_model=SimpleNamespace(config=cfg)))
        iterator = iter(matrices)
        vision._probabilities = lambda _: next(iterator)
        vision.multi_frame = True
        vision.ocr_min_confidence = 0.85
        vision.dedup_seconds = 300
        vision.tracks = {}
        vision.last_plate_seen = {}
        vision.metrics = {}
        return vision, Image(), box

    def test_real_pipeline_emits_once_after_three_separated_observations(self):
        vision, image, _ = self.pipeline([probabilities("GJ01AB1234")] * 3)
        self.assertEqual(vision.process(image, pts_ms=1000), [])
        self.assertEqual(vision.process(image, pts_ms=1300), [])
        observed = vision.process(image, pts_ms=1600)
        self.assertEqual(observed[0].plate, "GJ01AB1234")
        self.assertEqual(vision.process(image, pts_ms=1900), [])
        self.assertEqual(vision.metrics["accepted_sightings"], 1)
        self.assertNotIn("GJ01AB1234", repr(observed[0]))

    def test_track_reset_prevents_cross_scene_consensus(self):
        vision, image, _ = self.pipeline([probabilities("GJ01AB1234")] * 3)
        vision.process(image, pts_ms=1000)
        vision.process(image, pts_ms=1300)
        vision.reset()
        self.assertEqual(vision.process(image, pts_ms=1600), [])
        self.assertEqual(vision.tracks["7"].reader.observation_count, 1)

    def test_low_confidence_attempts_are_bounded_and_not_emitted(self):
        vision, image, _ = self.pipeline([probabilities("GJ01AB1234", 0.2)] * 40)
        for index in range(60):
            self.assertEqual(vision.process(image, pts_ms=1000 + index * 400), [])
        self.assertEqual(vision.metrics["probability_observations"], 40)
        self.assertEqual(vision.tracks["7"].reader.observation_count, 12)
        self.assertEqual(vision.tracks["7"].no_read_reason, "attempt_budget_exhausted")
        self.assertEqual(vision.metrics["no_read_tracks"], 1)
        self.assertNotIn("GJ01AB1234", str(vision.metrics))

    def test_frames_too_close_are_not_counted_twice(self):
        vision, image, _ = self.pipeline([probabilities("GJ01AB1234")] * 3)
        for pts in (1000, 1000, 1100, 1200):
            vision.process(image, pts_ms=pts)
        self.assertEqual(vision.metrics["probability_observations"], 1)

    def test_second_track_for_same_plate_is_suppressed(self):
        vision, image, box = self.pipeline([probabilities("GJ01AB1234")] * 6)
        for pts in (1000, 1300, 1600):
            observations = vision.process(image, pts_ms=pts)
        self.assertEqual(len(observations), 1)
        box.id = SimpleNamespace(item=lambda: 8)
        for pts in (1900, 2200, 2500):
            self.assertEqual(vision.process(image, pts_ms=pts), [])
        self.assertEqual(vision.metrics["accepted_sightings"], 1)

    def test_spatial_shift_clears_old_plate_evidence(self):
        vision, image, _ = self.pipeline([probabilities("GJ01AB1234")] * 3)
        vision.process(image, pts_ms=1000)
        vision.process(image, pts_ms=1300)
        vision.tracks["7"].plate_center = (0.95, 0.05)
        self.assertEqual(vision.process(image, pts_ms=1600), [])
        self.assertEqual(vision.metrics["plate_region_resets"], 1)
        self.assertEqual(vision.tracks["7"].reader.observation_count, 1)


if __name__ == "__main__":
    unittest.main()
