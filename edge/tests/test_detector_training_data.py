import unittest

import numpy as np

from sentinel_edge.detector_training_data import exclude_cross_split_similar, grouping_key, parse_yolo, partition_groups, perceptual_hash


class DetectorDataTests(unittest.TestCase):
    def test_cross_split_similarity_quarantines_transitive_group(self):
        rows=[{'id':i,'pixels':str(i),'source':'bmd','phash':h,'split':s}
              for i,h,s in ((0,0,'train'),(1,1,'train'),(2,3,'test'),(3,255,'val'))]
        kept,excluded=exclude_cross_split_similar(rows,max_distance=1)
        self.assertEqual([r['id'] for r in kept],[3])
        self.assertEqual({r['id'] for r in excluded},{0,1,2})

    def test_same_partition_similar_images_are_not_moved(self):
        rows=[{'pixels':str(i),'source':'one','phash':0,'split':'test'} for i in range(2)]
        kept,excluded=exclude_cross_split_similar(rows)
        self.assertEqual(kept,rows)
        self.assertEqual(excluded,[])

    def test_existing_export_is_preserved(self):
        from pathlib import Path
        import tempfile
        from sentinel_edge.detector_training_data import prepare
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory); (output/'manifest.json').write_text('original')
            with self.assertRaisesRegex(ValueError,'Existing export preserved'):
                prepare(Path('/unused'),'plate',output)
            self.assertEqual((output/'manifest.json').read_text(),'original')
    def test_reject_invalid_labels_and_unknown_negatives(self):
        for value in ('','0 nan .5 .2 .2','0 .5 .5 -.1 .2','1 .5 .5 .1 .1','0 .5 .5 .1 .1 .4','0.5 .5 .5 .1 .1'):
            with self.assertRaises(ValueError):
                parse_yolo(value,1)

    def test_clip_bounds_not_classes(self):
        boxes=parse_yolo('0 .99 .5 .1 .2',1)
        self.assertEqual(boxes[0][0],0)
        self.assertAlmostEqual(boxes[0][1]+boxes[0][3]/2,1)

    def test_filename_and_temporal_groups(self):
        self.assertEqual(grouping_key('madrid','abc_jpg.rf.123.jpg'),grouping_key('madrid','abc_jpg.rf.456.jpg'))
        self.assertEqual(grouping_key('gujarat','IMG_20260226_123333 (2).jpg'),grouping_key('gujarat','IMG_20260226_123357.jpg'))

    def test_groups_remain_together_and_deterministic(self):
        rows=[{'pixels':str(i),'source':'one','group_key':str(i//2),'phash':(2**63-1 if i%2 else 0),'publisher_split':'train'} for i in range(8)]
        first=partition_groups([dict(r) for r in rows])
        second=partition_groups([dict(r) for r in rows])
        self.assertEqual(first,second)
        for a,b in zip(first[::2],first[1::2]):
            self.assertEqual(a['split'],b['split'])

    def test_holdout_precedence(self):
        rows=[{'pixels':'same','source':'one','group_key':'one','phash':0,'publisher_split':s} for s in ('train','val','test')]
        self.assertEqual({r['split'] for r in partition_groups(rows,preserve=True)},{'test'})

    def test_perceptual_hash_stable(self):
        image=np.zeros((40,40,3),np.uint8); image[:,20:]=255
        self.assertEqual(perceptual_hash(image),perceptual_hash(image.copy()))


if __name__=='__main__':unittest.main()
