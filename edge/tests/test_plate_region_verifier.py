"""Synthetic contract/split tests; never read real plate transcriptions."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from sentinel_edge.plate_region_verifier import FEATURE_DIM,FEATURE_VERSION,PlateRegionVerifier,decision,extract_features,sha256

spec=importlib.util.spec_from_file_location('train_region',Path(__file__).resolve().parents[2]/'tools/train_plate_region_verifier.py')
trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)


class VerifierTests(unittest.TestCase):
    def artifact(self,root):
        path=Path(root)/'model.npz'
        np.savez_compressed(path,weights=np.zeros(FEATURE_DIM,np.float32),bias=np.array([0.],np.float64))
        receipt=dict(model_sha256=sha256(path),feature_version=FEATURE_VERSION,feature_dimension=FEATURE_DIM,
                     model_type='linear-logistic-hog',thresholds=dict(reject_below=.2,accept_above=.8),label_scope='synthetic test')
        path.with_suffix('.json').write_text(json.dumps(receipt),encoding='utf-8')
        return path

    def test_three_way_thresholds_and_nonfinite_inputs(self):
        self.assertEqual(decision(.1,.2,.8),'likely_non_plate')
        self.assertEqual(decision(.2,.2,.8),'uncertain')
        self.assertEqual(decision(.5,.2,.8),'uncertain')
        self.assertEqual(decision(.8,.2,.8),'likely_plate')
        for args in ((np.nan,.2,.8),(.5,.9,.8),(-.1,.2,.8)):
            with self.assertRaises(ValueError):decision(*args)

    def test_features_preserve_input_and_have_fixed_finite_contract(self):
        image=np.full((45,130,3),150,np.uint8)
        cv2.putText(image,'AB123',(3,32),cv2.FONT_HERSHEY_SIMPLEX,.7,(20,20,20),2)
        before=image.copy(); features=extract_features(image)
        self.assertEqual(features.shape,(FEATURE_DIM,))
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_array_equal(image,before)
        for bad in (np.zeros((3,20,3),np.uint8),np.zeros((40,100)),np.zeros((40,100,3),np.float32)):
            with self.assertRaises(ValueError):extract_features(bad)

    def test_model_verifies_without_reading_characters(self):
        with tempfile.TemporaryDirectory() as directory:
            path=self.artifact(directory)
            value=PlateRegionVerifier(path).verify(np.full((32,100,3),100,np.uint8),context={'vehicle_id':'synthetic'})
            self.assertEqual(value['status'],'uncertain')
            self.assertFalse(value['allow_ocr'])
            self.assertEqual(value['classifier_score'],.5)
            self.assertNotIn('plate',value)
            self.assertEqual(value['model_sha256'],sha256(path))

    def test_tampered_weights_and_feature_version_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=self.artifact(directory)
            path.write_bytes(path.read_bytes()+b'tamper')
            with self.assertRaisesRegex(ValueError,'identity'):PlateRegionVerifier(path)
            path=self.artifact(directory)
            receipt=json.loads(path.with_suffix('.json').read_bytes());receipt['feature_dimension']=3
            path.with_suffix('.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError,'feature'):PlateRegionVerifier(path)

    def test_malformed_arrays_cannot_be_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            path=self.artifact(directory)
            np.savez_compressed(path,weights=np.zeros(3,np.float32),bias=np.array([0.]))
            receipt=json.loads(path.with_suffix('.json').read_bytes());receipt['model_sha256']=sha256(path)
            path.with_suffix('.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError,'weights'):PlateRegionVerifier(path)

    def test_group_pixel_and_reserved_source_leakage_are_rejected(self):
        row=dict(id='a',group='g',pixels='p',split='train')
        self.assertEqual(trainer.validate_splits([row]),{'train':1})
        for extra in (dict(id='b',group='g',pixels='q',split='test'),
                      dict(id='b',group='h',pixels='p',split='val'),
                      dict(id='b',group='h',pixels='q',split='test',original_sha256=trainer.MUMBAI_HASH)):
            with self.assertRaises(ValueError):trainer.validate_splits([row,extra])

    def test_background_negatives_never_touch_any_known_plate(self):
        row=dict(id='a'*64,width=640,height=480)
        positives=[(100,100,220,140),(260,100,380,140)]
        negatives=trainer.negatives(row,positives)
        self.assertTrue(negatives)
        self.assertTrue(all(not trainer.overlaps_any(box,positives) for box in negatives))
        self.assertEqual(negatives,trainer.negatives(row,positives))

    def test_threshold_selection_uses_positive_retention_and_uncertainty(self):
        scores=np.concatenate((np.linspace(.2,.99,100),np.linspace(.01,.8,100)))
        truth=np.r_[np.ones(100),np.zeros(100)]
        thresholds=trainer.threshold_selection(scores,truth)
        report=trainer.metrics(scores,truth,thresholds)
        self.assertGreaterEqual(report['positive_retention'],.99)
        self.assertEqual(thresholds['selection_split'],'val')
        self.assertGreater(report['uncertain_positive']+report['uncertain_negative'],0)


if __name__=='__main__':unittest.main()
