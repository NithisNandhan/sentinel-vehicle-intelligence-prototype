import hashlib
import tempfile
import unittest
import json
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

from sentinel_edge.awiros_experiment import adaptation_keys
from sentinel_edge.corpus_ocr_adaptation import validate


class CorpusAdaptationTests(unittest.TestCase):
    def test_selected_scope_is_bounded(self):
        keys = {'head.ctc_head.fc.weight', 'head.ctc_head.fc.bias',
                'head.ctc_encoder.encoder.svtr_block.1.norm1.weight',
                'backbone.stages.3.blocks.0.layers.5.conv2.conv.weight',
                'backbone.stem.conv.weight', 'head.ctc_encoder.encoder.conv1.norm._mean'}
        selected = adaptation_keys(keys, 'bounded_image_encoder_tail_and_ctc')
        self.assertEqual(len(selected), 4)
        self.assertNotIn('backbone.stem.conv.weight', selected)
        self.assertNotIn('head.ctc_encoder.encoder.conv1.norm._mean', selected)
        with self.assertRaises(ValueError): adaptation_keys(keys, 'all_parameters')
        with self.assertRaises(ValueError): adaptation_keys({'head.ctc_head.fc.weight'}, 'bounded_image_encoder_tail_and_ctc')

    def document(self, root):
        rows=[]
        for i in range(50):
            content=str(i).encode(); path=root/f'{i}.png'; path.write_bytes(content)
            split='validation' if i<10 else 'test' if i<20 else 'train'
            rows.append({'id':str(i),'crop_path':path.name,'crop_sha256':hashlib.sha256(content).hexdigest(),
                         'split':split,'origin':'photograph_publisher_declared',
                         'transcription':f'AP01AB{i:04d}',
                         'label_tier':'AI_blind_visual_publisher_agreement_not_human_truth' if i<20 else 'publisher_silver'})
        return {'schema_version':1,'evaluation_caveat':'AI-reviewed diagnostic only','samples':rows}

    def test_separate_silver_training_and_reviewed_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); document=self.document(root)
            self.assertEqual(len(validate(document,root)),50)
            document['samples'][0]['label_tier']='publisher_silver'
            with self.assertRaises(ValueError): validate(document,root)

    def test_registration_leakage_and_modified_pixels_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); document=self.document(root)
            document['samples'][20]['transcription']=document['samples'][0]['transcription']
            with self.assertRaises(ValueError): validate(document,root)
            document=self.document(root); (root/'0.png').write_bytes(b'changed')
            with self.assertRaises(ValueError): validate(document,root)


class LiveShadowPrivacyTests(unittest.TestCase):
    def test_vehicle_fallback_preserves_native_crop_pixels(self):
        import numpy as np
        from sentinel_edge.awiros_live_shadow import probe
        tokens=('',*'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        probabilities=[]
        for c in 'AP01AB1234':
            row=[0.]*len(tokens); row[tokens.index(c)]=1.; probabilities.append(row)
        original=np.arange(32*120*3,dtype=np.uint16).astype(np.uint8).reshape(32,120,3)
        seen=[]
        def read(image):
            seen.append(image.copy()); return probabilities
        model=SimpleNamespace(tokens=tokens,probabilities=read)
        boxes=SimpleNamespace(xyxy=SimpleNamespace(tolist=lambda:[[0,0,120,30]]),conf=SimpleNamespace(tolist=lambda:[.9]))
        vehicle=SimpleNamespace(predict=lambda *a,**k:[SimpleNamespace(boxes=boxes)])
        detector=SimpleNamespace(box=lambda image:None if image.shape[0]==32 else (1,1,119,29))
        frame=SimpleNamespace(image=original,pts_ms=0,discontinuity=False)
        with patch('sentinel_edge.awiros_live_shadow.read_camera',return_value=iter([('rtsp',frame)])):
            result=probe(SimpleNamespace(),SimpleNamespace(camera_id='cam01'),{'test':model},detector,15,vehicle)
        np.testing.assert_array_equal(seen[0],original[1:29,1:119])
        self.assertEqual(result['vehicle_regions'],1)

    def test_shadow_serializes_only_counts_not_media_or_plate_text(self):
        import numpy as np
        from sentinel_edge.awiros_live_shadow import probe
        tokens=('',*'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        plate='AP01AB1234'
        probabilities=[]
        for c in plate:
            row=[0.]*len(tokens); row[tokens.index(c)]=1.; probabilities.append(row)
        model=SimpleNamespace(tokens=tokens,probabilities=lambda image:probabilities)
        frames=[]
        for i in range(5):
            image=np.full((32,120,3),120,dtype=np.uint8); image[0,0]=i
            frames.append(('rtsp',SimpleNamespace(image=image,pts_ms=i*500,discontinuity=False)))
        detector=SimpleNamespace(box=lambda image:(0,0,120,32))
        with patch('sentinel_edge.awiros_live_shadow.read_camera',return_value=iter(frames)):
            result=probe(SimpleNamespace(),SimpleNamespace(camera_id='cam01'),{'test':model},detector,15)
        text=json.dumps(result)
        self.assertNotIn(plate,text)
        self.assertNotIn('image',text)
        self.assertEqual(result['sampled_frames'],5)
        self.assertEqual(result['methods']['test']['unique_accepted_unverified_identifiers'],1)
