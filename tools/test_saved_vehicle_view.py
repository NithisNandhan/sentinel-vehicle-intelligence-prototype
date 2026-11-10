import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from run_owned_multiframe import render_saved


class SavedViewBindingTests(unittest.TestCase):
    def test_mismatched_score_or_mapping_cannot_overwrite_existing_viewer(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);payload=b'{"vehicle_lifecycle":true}'
            (root/'predictions.json').write_bytes(payload);page=root/'index.html';page.write_text('preserve this viewer')
            (root/'accuracy.json').write_text(json.dumps(dict(predictions_sha256='wrong')))
            with self.assertRaises(ValueError):render_saved(SimpleNamespace(output=root,review_summary=None))
            (root/'accuracy.json').write_text(json.dumps(dict(predictions_sha256=hashlib.sha256(payload).hexdigest())))
            summary=root/'mapping.json';summary.write_text(json.dumps(dict(predictions_sha256='another capture')))
            with self.assertRaises(ValueError):render_saved(SimpleNamespace(output=root,review_summary=summary))
            self.assertEqual(page.read_text(),'preserve this viewer')


if __name__=='__main__':unittest.main()
