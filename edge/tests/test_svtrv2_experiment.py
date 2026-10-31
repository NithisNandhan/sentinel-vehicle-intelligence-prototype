"""Publisher resize-policy and CTC adapter boundary checks; no model downloads."""
import unittest
import hashlib
import importlib.util
import tempfile
from pathlib import Path

from sentinel_edge.svtrv2_experiment import input_size


class ResizePolicyTests(unittest.TestCase):
    def test_publisher_cli_and_dataset_are_explicitly_distinct(self):
        self.assertEqual(input_size(120, 30), (160, 32))
        self.assertEqual(input_size(120, 30, policy="dataset_round"), (128, 32))
        self.assertEqual(input_size(120, 100), (96, 48))
        self.assertEqual(input_size(50, 100), (64, 64))
        self.assertEqual(input_size(600, 10), (384, 32))

    def test_invalid_input_is_not_silently_resized(self):
        for width, height in ((0, 4), (4, 0), (-1, 4), (True, 4), (4.5, 3)):
            with self.assertRaises(ValueError):
                input_size(width, height)
        with self.assertRaises(ValueError):
            input_size(100, 30, policy="invented")

    def test_tensor_preserves_publisher_rgb_and_normalisation(self):
        import numpy as np
        from sentinel_edge.svtrv2_experiment import input_tensor
        image = np.zeros((30, 120, 3), dtype=np.uint8)
        image[:, :, 0] = 255  # Blue BGR input must become the third RGB channel.
        tensor = input_tensor(image)
        self.assertEqual(tuple(tensor.shape), (1, 3, 32, 160))
        self.assertTrue(bool((tensor[:, 0] == -1).all()))
        self.assertTrue(bool((tensor[:, 2] == 1).all()))
        with self.assertRaises(ValueError):
            input_tensor(image.astype(np.float32))


class DiagnosticIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script = Path(__file__).resolve().parents[2] / "tools/compare_svtrv2.py"
        spec = importlib.util.spec_from_file_location("compare_svtrv2_test", script)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def fixture(self, root):
        (root / "crop.png").write_bytes(b"test-asset")
        return {"samples": [{"id": "one", "crop_path": "crop.png", "crop_sha256": hashlib.sha256(b"test-asset").hexdigest(),
                             "transcription": "GJ01AB1234", "label_tier": "synthetic_unit_fixture", "split": "test", "quarantined": False}]}

    def test_registration_leakage_and_duplicates_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = self.fixture(root)
            self.assertEqual(len(self.module.validate_samples(document, root, "test")), 1)
            document["samples"].append({**document["samples"][0], "id": "two", "split": "train"})
            with self.assertRaises(ValueError):
                self.module.validate_samples(document, root, "test")
            document["samples"][1]["split"] = "test"
            document["samples"][1]["id"] = "one"
            with self.assertRaises(ValueError):
                self.module.validate_samples(document, root, "test")

    def test_changed_image_and_empty_holdout_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = self.fixture(root)
            (root / "crop.png").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                self.module.validate_samples(document, root, "test")
            with self.assertRaises(ValueError):
                self.module.validate_samples(document, root, "validation")
