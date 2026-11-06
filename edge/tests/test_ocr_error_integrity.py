"""Failure-review source bindings and private-viewer destination checks."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


def load_tool(name):
    path = Path(__file__).resolve().parents[2] / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ReviewIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = load_tool("audit_ocr_failures")
        cls.summary = load_tool("summarize_ocr_crops")

    def fixture(self):
        manifest = {"samples": [{"id": "fixture", "split": "test", "transcription": "GJ01AB1234",
                                "crop_path": "crop.png", "crop_sha256": "fixture_hash", "height": 24, "width": 100}]}
        row = {"id": "fixture", "expected": "GJ01AB1234", "reading": "GJ01AB1235", "exact": False,
               "crop_path": "crop.png", "crop_sha256": "fixture_hash", "native_shape": [24, 100]}
        report = {"split": "test", "stills": [row]}
        index = {"samples": [{"number": 1, "id": "fixture", "expected": row["expected"], "reading": row["reading"],
                              "crop_sha256": "fixture_hash", "native_width": 100, "native_height": 24}]}
        return manifest, report, index

    def test_report_cannot_substitute_reference_pixels_or_sample_identity(self):
        manifest, report, _ = self.fixture()
        self.audit.validate_report(manifest, report)
        for key, replacement in (("expected", "GJ01AB1235"), ("crop_sha256", "changed"),
                                 ("crop_path", "other.png"), ("id", "other"), ("exact", True)):
            changed = copy.deepcopy(report)
            changed["stills"][0][key] = replacement
            with self.assertRaises(ValueError):
                self.audit.validate_report(manifest, changed)
        report["stills"].append(report["stills"][0])
        with self.assertRaises(ValueError):
            self.audit.validate_report(manifest, report)

    def test_failure_index_is_bound_to_actual_failed_readings_and_order(self):
        _, report, index = self.fixture()
        self.audit.validate_index(index, report)
        for key, replacement in (("reading", "substituted"), ("crop_sha256", "changed"),
                                 ("number", 2), ("native_width", 99)):
            changed = copy.deepcopy(index)
            changed["samples"][0][key] = replacement
            with self.assertRaises(ValueError):
                self.audit.validate_index(changed, report)
        with self.assertRaises(ValueError):
            self.audit.validate_index({"samples": []}, report)

    def test_paired_comparison_rejects_missing_samples_and_reference_drift(self):
        _, report, _ = self.fixture()
        rows = report["stills"]
        self.assertEqual(self.summary.paired(rows, rows)["agree_wrong"], 1)
        for second in ([], rows + rows, [{**rows[0], "expected": "different"}]):
            with self.assertRaises(ValueError):
                self.summary.paired(rows, second)

    def test_matching_directory_name_outside_checkout_cannot_receive_private_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                self.summary.private_output_path(Path(temporary) / "data/benchmarks/reports/private.html")


if __name__ == "__main__":
    unittest.main()
