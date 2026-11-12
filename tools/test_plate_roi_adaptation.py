import unittest
import hashlib
import json
from pathlib import Path
import tempfile
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from prepare_plate_roi_adaptation import absolute_boxes,context_crop,map_boxes,validate_lineage,prepare,pixel_hash


class PlateRoiAdaptationTests(unittest.TestCase):
    def test_absolute_mapping_and_bounds(self):
        self.assertEqual(absolute_boxes([[0,.5,.5,.2,.4]],(100,200,3)),[[0,80.,30.,120.,70.]])
        with self.assertRaises(ValueError):absolute_boxes([[0,.95,.5,.3,.2]],(100,200,3))
        with self.assertRaises(ValueError):absolute_boxes([], (100,200,3))

    def test_context_maps_all_intersecting_labels(self):
        rows=map_boxes([[0,10,20,30,40],[0,40,20,60,40],[0,80,80,90,90]],(0,0,70,60))
        self.assertEqual(len(rows),2)
        self.assertAlmostEqual(rows[0][1],20/70)
        self.assertAlmostEqual(rows[1][3],20/70)

    def test_partial_other_plate_never_silently_disappears(self):
        self.assertIsNone(map_boxes([[0,10,20,30,40],[0,60,20,80,40]],(0,0,70,60)))
        # A tiny <=5% box boundary intersection is retained explicitly.
        self.assertEqual(len(map_boxes([[0,10,20,30,40]],(10.5,0,70,60))),1)

    def test_deterministic_crop_retains_anchor_and_native_bounds(self):
        boxes=[[0,.5,.7,.15,.08]]
        first=context_crop('source-id',boxes,(500,800,3))
        self.assertEqual(first,context_crop('source-id',boxes,(500,800,3)))
        crop,mapped,_=first
        self.assertTrue(0<=crop[0]<crop[2]<=800 and 0<=crop[1]<crop[3]<=500)
        self.assertEqual(len(mapped),1)
        self.assertAlmostEqual(mapped[0][3]*(crop[2]-crop[0]),120.)
        self.assertAlmostEqual(mapped[0][4]*(crop[3]-crop[1]),40.)

    def test_partial_plate_policy_falls_back_to_full_image(self):
        # Dense grid guarantees that candidate crops cut another plate.
        boxes=[[0,x,y,.1,.08] for x in [.1,.3,.5,.7,.9] for y in [.1,.3,.5,.7,.9]]
        found=False
        for i in range(20):
            crop,mapped,details=context_crop(str(i),boxes,(500,800,3))
            if details['full_image_fallback']:
                self.assertEqual(crop,(0,0,800,500));self.assertEqual(len(mapped),25);found=True;break
        self.assertTrue(found)

    def test_frozen_group_and_pixel_partitions_cannot_leak(self):
        rows=[dict(group='g1',pixels='p1',split='train'),dict(group='g1',pixels='p2',split='train'),dict(group='g2',pixels='p3',split='test')]
        validate_lineage(rows)
        with self.assertRaises(ValueError):validate_lineage(rows+[dict(group='g1',pixels='new',split='test')])
        with self.assertRaises(ValueError):validate_lineage(rows+[dict(group='new',pixels='p1',split='test')])

    def test_export_augments_train_val_and_preserves_test_bytes_and_groups(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.mkdir();samples=[]
            rng=np.random.default_rng(42)
            for split in ('train','val','test'):
                folder=source/'images'/split;folder.mkdir(parents=True)
                for i in range(3):
                    image=rng.integers(0,256,(64,128,3),dtype=np.uint8)
                    identifier=split+str(i);path=folder/(identifier+'.png')
                    self.assertTrue(cv2.imwrite(str(path),image))
                    samples.append(dict(id=identifier,group=identifier,split=split,pixels=pixel_hash(image),
                        image_path=path.relative_to(source).as_posix(),original_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        width=128,height=64,boxes=[[0,.5,.7,.15,.1]]))
            (source/'manifest.json').write_text(json.dumps(dict(kind='plate',samples=samples,
                counts=dict(train=3,val=3,test=3),caveat='Synthetic fixture only')))
            output=root/'output';receipt=prepare(source,output)
            self.assertEqual(receipt['counts'],dict(train=6,val=6,test=3))
            rows=json.loads((output/'manifest.json').read_bytes())['samples']
            originals={r['id']:r for r in samples}
            for row in rows:
                parent=originals[row['parent_id']]
                self.assertEqual((row['group'],row['split']),(parent['group'],parent['split']))
                if row['split']=='test':
                    self.assertEqual(row['variant'],'original')
                    self.assertEqual(row['original_sha256'],parent['original_sha256'])
            # An upstream image edit must fail before producing a new dataset.
            (source/samples[0]['image_path']).write_bytes(b'changed')
            with self.assertRaises(ValueError):prepare(source,root/'invalid')


if __name__=='__main__':unittest.main()
