from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from sentinel_edge.degradation_benchmark import (
    degradation_levels,
    score_passage,
    summarize_outcomes,
    validate_manifest,
    validate_sequence,
    validate_annotation_bounds,
    _target_box,
)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        # These empty files exercise manifest path validation, not decoding.
        for name in ("one.png", "two.png", "three.png", "passage.mp4"):
            (self.root / name).touch()
        self.document = {
            "dataset": {
                "name": "Permitted controlled fixture",
                "source_url": "https://example.org/dataset",
                "license": "CC-BY-NC-4.0",
                "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
                "permitted_use": "noncommercial_benchmark",
            },
            "passages": [{
                "id": "passage-1", "vehicle_id": "vehicle-1",
                "split": "development", "ground_truth": "GJ01AB1234",
                "country": "IN", "kind": "sequence",
                "timing_mode": "verified_pts",
                "frames": [
                    {"path": "one.png", "pts_ms": 1000},
                    {"path": "two.png", "pts_ms": 1300},
                    {"path": "three.png", "pts_ms": 1600},
                ],
            }],
        }

    def reject(self, document):
        with self.assertRaises((ValueError, FileNotFoundError)):
            validate_manifest(document, self.root)

    def test_permitted_sequence_is_accepted(self):
        result = validate_manifest(self.document, self.root)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "passage-1")

    def test_explicit_supported_licenses_only(self):
        for license_id in ("CC0-1.0", "CC-BY-4.0", "CC-BY-NC-4.0"):
            document = copy.deepcopy(self.document)
            document["dataset"]["license"] = license_id
            self.assertEqual(len(validate_manifest(document, self.root)), 1)
        for license_id in ("unknown", "all-rights-reserved", "MIT", ""):
            document = copy.deepcopy(self.document)
            document["dataset"]["license"] = license_id
            self.reject(document)

    def test_noncommercial_scope_cannot_be_silently_changed(self):
        document = copy.deepcopy(self.document)
        document["dataset"]["permitted_use"] = "commercial"
        self.reject(document)

    def test_missing_permission_provenance_is_rejected(self):
        for field in ("name", "source_url", "license", "license_url", "permitted_use"):
            document = copy.deepcopy(self.document)
            del document["dataset"][field]
            self.reject(document)

    def test_insecure_or_retention_forbidden_urls_rejected(self):
        for url in (
            "http://example.org/dataset", "https://sentinel.gujarat.gov.in/video",
            "https://cctv.corp8.cloud/cam01/index.m3u8",
        ):
            document = copy.deepcopy(self.document)
            document["dataset"]["source_url"] = url
            self.reject(document)

    def test_same_vehicle_may_have_variants_only_in_one_split(self):
        second = copy.deepcopy(self.document["passages"][0])
        second["id"] = "passage-2"
        self.document["passages"].append(second)
        self.assertEqual(len(validate_manifest(self.document, self.root)), 2)
        second["split"] = "test"
        self.reject(self.document)

    def test_unknown_split_and_non_indian_scope_are_rejected(self):
        for field, value in (("split", "train"), ("country", "BR")):
            document = copy.deepcopy(self.document)
            document["passages"][0][field] = value
            self.reject(document)

    def test_plate_split_leak_cannot_hide_under_another_vehicle_id(self):
        second = copy.deepcopy(self.document["passages"][0])
        second.update(id="passage-2", vehicle_id="vehicle-2", split="test")
        self.document["passages"].append(second)
        self.reject(self.document)

    def test_sequence_timing_must_be_labelled(self):
        for mode in (None, "guessed_fps"):
            document = copy.deepcopy(self.document)
            document["passages"][0]["timing_mode"] = mode
            self.reject(document)

    def test_duplicate_passage_ids_are_rejected(self):
        self.document["passages"].append(copy.deepcopy(self.document["passages"][0]))
        self.reject(self.document)

    def test_labels_must_be_independently_normalized(self):
        for text in ("gj01ab1234", "GJ 01 AB 1234", "", "गJ01AB1234"):
            document = copy.deepcopy(self.document)
            document["passages"][0]["ground_truth"] = text
            self.reject(document)

    def test_invalid_annotation_box_is_rejected(self):
        for box in ([10, 20, 10, 30], [10, 20, 30], [-1, 20, 30, 40], [0, 0, float("nan"), 40]):
            document = copy.deepcopy(self.document)
            document["passages"][0]["frames"][0]["plate_box"] = box
            self.reject(document)

    def test_still_image_is_not_declared_a_temporal_passage(self):
        document = copy.deepcopy(self.document)
        document["passages"][0]["kind"] = "image"
        self.reject(document)
        document = copy.deepcopy(self.document)
        document["passages"][0]["frames"] = document["passages"][0]["frames"][:2]
        self.reject(document)

    def test_missing_and_outside_root_frames_rejected(self):
        for path in ("missing.png", "../outside.png", str(self.root.parent / "outside.png")):
            document = copy.deepcopy(self.document)
            document["passages"][0]["frames"][0]["path"] = path
            self.reject(document)

    def test_video_requires_a_valid_time_window(self):
        document = copy.deepcopy(self.document)
        document["passages"][0].update(kind="video", path="passage.mp4", start_ms=0, end_ms=3000)
        del document["passages"][0]["frames"]
        self.assertEqual(len(validate_manifest(document, self.root)), 1)
        for start, end in ((1000, 1000), (2000, 1000), (-1, 1000), (False, 1000), (0, True)):
            invalid = copy.deepcopy(document)
            invalid["passages"][0].update(start_ms=start, end_ms=end)
            self.reject(invalid)


class SequenceTests(unittest.TestCase):
    def test_annotation_bounds_reject_invented_crop_area(self):
        with self.assertRaises(ValueError):
            validate_annotation_bounds({"plate_box": [0, 0, 101, 10]}, (20, 100, 3))
        validate_annotation_bounds({"plate_box": [0, 0, 100, 10]}, (20, 100, 3))

    def test_target_matching_rejects_overlapping_large_distractor(self):
        annotation = {"vehicle_box": [10, 10, 30, 30], "plate_box": [15, 20, 25, 25]}
        self.assertTrue(_target_box([11, 10, 31, 30], annotation))
        self.assertFalse(_target_box([0, 0, 100, 100], annotation))
        self.assertFalse(_target_box([10, 10, 30, 30], {"plate_box": [15, 20, 25, 25]}))
    def test_real_changes_and_strict_timestamps_are_accepted(self):
        self.assertIsNone(validate_sequence([0, 300, 600, 900], ["a", "b", "b", "c"]))

    def test_repeated_still_cannot_masquerade_as_video(self):
        with self.assertRaises(ValueError):
            validate_sequence([0, 300, 600, 900], ["still"] * 4)

    def test_duplicate_and_reverse_timestamps_are_rejected(self):
        for timestamps in ([0, 0, 300], [300, 200, 100]):
            with self.assertRaises(ValueError):
                validate_sequence(timestamps, ["a", "b", "c"])

    def test_insufficient_or_mismatched_sequence_is_rejected(self):
        for timestamps, hashes in (([0, 300], ["a", "b"]), ([0, 300, 600], ["a", "b"])):
            with self.assertRaises(ValueError):
                validate_sequence(timestamps, hashes)


class ScoringTests(unittest.TestCase):
    def test_exact_plate_read_is_correct(self):
        result = score_passage({"GJ01AB1234"}, {"GJ01AB1234"})
        self.assertEqual(result["outcome"], "correct")
        self.assertEqual((result["correct"], result["wrong"], result["abstain"]), (1, 0, 0))

    def test_wrong_high_confidence_plate_is_wrong(self):
        result = score_passage({"GJ01AB1234"}, {"GJ01AB1238"})
        self.assertEqual(result["outcome"], "wrong")
        self.assertEqual((result["correct"], result["wrong"], result["abstain"]), (0, 1, 0))

    def test_no_accepted_read_is_abstention(self):
        result = score_passage({"GJ01AB1234"}, set())
        self.assertEqual(result["outcome"], "abstain")
        self.assertEqual((result["correct"], result["wrong"], result["abstain"]), (0, 0, 1))

    def test_extra_wrong_emission_does_not_count_as_exact_success(self):
        result = score_passage({"GJ01AB1234"}, {"GJ01AB1234", "GJ01AB1238"})
        self.assertEqual(result["outcome"], "wrong")

    def test_scores_do_not_expose_plate_strings(self):
        result = score_passage({"GJ01AB1234"}, {"GJ01AB1238"})
        self.assertNotIn("GJ01AB", json.dumps(result))

    def test_aggregate_reports_accuracy_precision_and_coverage_separately(self):
        rows = []
        for predictions in ({"GJ01AB1234"}, {"GJ01AB1238"}, set()):
            rows.append({"stage": "end_to_end", "variant": "resolution_1", "split": "test",
                         **score_passage({"GJ01AB1234"}, predictions),
                         "duration_ms": 120, "probability_observations": 3,
                         "ground_truth": "GJ01AB1234", "path": "private/video.mp4",
                         "source_url": "https://example.org/private"})
        result = summarize_outcomes(rows)
        # Exact naming of group containers may evolve; privacy must not.
        encoded = json.dumps(result)
        self.assertNotIn("GJ01AB", encoded)
        self.assertNotIn("private/video", encoded)
        self.assertNotIn("example.org", encoded)
        self.assertNotIn("ground_truth", encoded)
        self.assertNotIn("score_is_accuracy", encoded)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result["groups"]), 1)
        group = result["groups"][0]
        self.assertEqual(group["passages"], 3)
        self.assertEqual((group["correct"], group["wrong"], group["abstain"]), (1, 1, 1))
        self.assertAlmostEqual(group["exact_accuracy"], 1 / 3)
        self.assertAlmostEqual(group["accepted_precision"], 1 / 2)
        self.assertAlmostEqual(group["coverage"], 2 / 3)
        self.assertIs(result["confidence_is_accuracy"], False)

    def test_all_abstentions_have_zero_coverage_not_perfect_accuracy(self):
        row = {"stage": "end_to_end", "variant": "resolution_0.25", "split": "test",
               **score_passage({"GJ01AB1234"}, set()), "duration_ms": 1}
        group = summarize_outcomes([row])["groups"][0]
        self.assertEqual(group["coverage"], 0)
        self.assertEqual(group["exact_accuracy"], 0)
        self.assertIsNone(group["accepted_precision"])

    def test_development_and_test_results_are_not_pooled(self):
        rows = [{"stage": "end_to_end", "variant": "resolution_1", "split": split,
                 **score_passage({"GJ01AB1234"}, prediction), "duration_ms": 1}
                for split, prediction in (("development", {"GJ01AB1234"}), ("test", set()))]
        groups = summarize_outcomes(rows)["groups"]
        self.assertEqual(len(groups), 2)
        indexed = {group["split"]: group for group in groups}
        self.assertEqual(indexed["development"]["exact_accuracy"], 1)
        self.assertEqual(indexed["test"]["exact_accuracy"], 0)

    def test_empty_aggregate_does_not_divide_by_zero(self):
        self.assertIsInstance(summarize_outcomes([]), dict)


class DegradationPlanTests(unittest.TestCase):
    def test_resolution_plan_is_repeatable_and_has_clear_original(self):
        levels = degradation_levels()
        self.assertEqual(levels, degradation_levels())
        resolution = [item for item in levels if item.get("family") == "resolution"]
        self.assertEqual([item["scale"] for item in resolution], [1, .75, .5, .375, .25, .1875, .125])


if __name__ == "__main__":
    unittest.main()
