"""External-reference normalisation and evaluation protocol checks."""
import importlib.util
import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[2] / f"tools/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExternalReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepare = load_tool("prepare_external_ocr_test")

    def test_references_are_not_ocr_corrected(self):
        self.assertEqual(self.prepare.normalise_reference("mh 02-EK 0837"), "MH02EK0837")
        self.assertEqual(self.prepare.normalise_reference("MH02EKO837"), "MH02EKO837")
        self.assertIsNone(self.prepare.normalise_reference("MH02??0837"))
        # Normalisation is not a legal-format or human-readability validator.
        self.assertEqual(self.prepare.normalise_reference("unknown text"), "UNKNOWNTEXT")

    def test_pixel_hash_agrees_with_existing_corpus_identity(self):
        import numpy as np
        from PIL import Image
        from sentinel_edge.plate_corpus import pixel_hash
        bgr = np.array([[[1, 20, 255], [100, 45, 7]]], dtype=np.uint8)
        rgb = Image.fromarray(bgr[:, :, ::-1])
        self.assertEqual(self.prepare.pixel_hash(bgr), pixel_hash(rgb))

    def test_perceptual_similarity_survives_small_pixel_changes(self):
        import numpy as np
        image = np.random.default_rng(8).integers(10, 240, size=(30, 100, 3), dtype=np.uint8)
        adjusted = image + np.uint8(1)
        self.assertLessEqual((self.prepare.phash(image) ^ self.prepare.phash(adjusted)).bit_count(), 6)


class PrivateViewerTests(unittest.TestCase):
    def test_private_images_require_opt_in_and_stay_in_ignored_viewer_tree(self):
        render = load_tool("render_svtrv2_comparison").render
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus"; corpus.mkdir()
            reports = root / "results"; reports.mkdir()
            image = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")
            (corpus / "crop.png").write_bytes(image)
            row = {"id":"fixture", "crop_path":"crop.png", "crop_sha256":hashlib.sha256(image).hexdigest(),
                   "label_tier":"synthetic_unit_fixture", "native_shape":[1,1], "expected":"GJ01AB1234", "reading":"GJ01AB1234", "exact":True}
            manifest = {"dataset":{"purpose":"private_noncommercial_evaluation_only", "no_redistribution_of_images_or_crops":True},
                        "no_training":True, "samples":[{**row, "provenance":[{"license":"CC-BY-NC-ND-4.0", "source_url":"https://example.org/fixture"}]}]}
            path = corpus / "adaptation_manifest.json"; path.write_text(json.dumps(manifest))
            report = {"model":{"name":"synthetic fixture"}, "manifest_sha256":hashlib.sha256(path.read_bytes()).hexdigest(), "split":"test",
                      "summary":{"exact":1,"samples":1,"character_errors":0,"characters":10,"median_ocr_ms":1,"p95_ocr_ms":1},
                      "stills":[row],"sequences":[],"evaluation_caveat":"Synthetic unit fixture only"}
            for name in ["awiros-parent", "awiros-adapted", "svtrv2"]:
                (reports / (name + "-test.json")).write_text(json.dumps(report))
            output = root / "data/benchmarks/reports/private.html"
            with self.assertRaises(ValueError):
                render(reports, corpus, output)
            with self.assertRaises(ValueError):
                render(reports, corpus, root / "public.html", allow_private_evaluation=True)
            render(reports, corpus, output, allow_private_evaluation=True)
            self.assertIn("Do not redistribute", output.read_text())
            self.assertIsNone(json.loads((reports / "aggregate.json").read_text())["models"][0]["sequence"])
            path.write_text(json.dumps({**manifest, "changed":True}))
            with self.assertRaises(ValueError):
                render(reports, corpus, output, allow_private_evaluation=True)


if __name__ == "__main__":
    unittest.main()
