import unittest
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'edge'))
from sentinel_edge.plate_scale_search import POLICY,detect_views,search_views


class PlateScaleSearchTests(unittest.TestCase):
    def test_small_views_search_after_first_hit_and_respect_budget(self):
        image=np.zeros((200,400,3),np.uint8);calls=[]
        def predict(tile):
            calls.append(tile.shape)
            return [([20,20,80,40],.9)]
        rows,count=detect_views(image,predict,max_calls=6)
        self.assertEqual(count,6);self.assertEqual(len(calls),6)
        self.assertTrue(any(row['view']=='full' for row in rows))
        self.assertEqual(search_views(image.shape,1),[('full',(0,0,400,200))])
        with self.assertRaises(ValueError):detect_views(image,predict,max_calls=7)

    def test_offsets_are_native_without_resampling(self):
        image=np.arange(200*400*3,dtype=np.uint8).reshape(200,400,3)
        calls=[]
        def predict(tile):
            calls.append(tile)
            return [([10,10,60,30],.8)] if len(calls)==3 else []
        rows,count=detect_views(image,predict)
        self.assertEqual(count,6);self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['view'],'corner-br')
        self.assertEqual(rows[0]['box'],[150.,80.,200.,100.])
        self.assertTrue(np.shares_memory(calls[2],image))
        np.testing.assert_array_equal(calls[2],image[70:200,140:400])

    def test_higher_confidence_tile_never_replaces_full_prediction(self):
        image=np.zeros((200,400,3),np.uint8);count=0
        def predict(tile):
            nonlocal count
            count+=1
            return [([20,20,80,40],.6 if count==1 else .95)] if count<=2 else []
        rows,_=detect_views(image,predict)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['confidence'],.6)
        self.assertEqual(rows[0]['view'],'full')
        self.assertEqual(POLICY,'full_view_protected_v2')

    def test_full_view_survives_equal_or_lower_confidence_duplicate(self):
        image=np.zeros((200,400,3),np.uint8)
        for tile_confidence in (.8,.9):
            with self.subTest(tile_confidence=tile_confidence):
                count=0
                def predict(tile):
                    nonlocal count
                    count+=1
                    return [([20,20,80,40],.9 if count==1 else tile_confidence)] if count<=2 else []
                rows,_=detect_views(image,predict)
                self.assertEqual(len(rows),1);self.assertEqual(rows[0]['view'],'full')

    def test_internal_cut_discarded_but_source_edge_kept(self):
        image=np.zeros((200,400,3),np.uint8);count=0
        def predict(tile):
            nonlocal count
            count+=1
            if count==1:return [([0,30,60,50],.7)]
            if count==2:return [([220,30,260,50],.95)] # right artificial cut
            if count==3:return [([0,30,60,50],.95)] # left artificial cut
            return []
        rows,_=detect_views(image,predict)
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['view'],'full')

    def test_invalid_geometry_and_tiny_extra_proposals_are_filtered(self):
        image=np.zeros((100,300,3),np.uint8)
        calls=0
        def predict(_):
            nonlocal calls
            calls+=1
            return [] if calls==1 else [([20,20,40,30],.8),([30,30,100,35],.8),
                ([40,10,70,60],.8),([0,0,float('nan'),30],.8)]
        rows,_=detect_views(image,predict)
        self.assertEqual(rows,[])

    def test_small_and_unusual_full_predictions_preserved_unchanged(self):
        image=np.zeros((100,300,3),np.uint8)
        source=[([10,10,20,14],.9),([60,50,64,70],.8),([299,90,310,110],.7),
                ([0,0,float('nan'),20],.6),([40,40,40,50],.6)]
        rows,count=detect_views(image,lambda _:source,max_calls=1)
        self.assertEqual(count,1)
        self.assertEqual([p['box'] for p in rows],[[10.,10.,20.,14.],[60.,50.,64.,70.],[299.,90.,300.,100.]])
        self.assertTrue(all(p['view']=='full' for p in rows))

    def test_parent_full_outputs_do_not_receive_new_nms(self):
        image=np.zeros((100,300,3),np.uint8)
        source=[([10,10,80,30],.9),([20,10,90,30],.8)]
        rows,count=detect_views(image,lambda _:source,max_calls=1)
        self.assertEqual(count,1);self.assertEqual(len(rows),2)
        self.assertEqual([p['box'] for p in rows],[v[0] for v in source])

    def test_large_native_windows_spread_and_cap(self):
        shape=(1000,1800,3);views=search_views(shape,6)
        self.assertEqual(len(views),6)
        self.assertEqual(views[0],('full',(0,0,1800,1000)))
        boxes=[b for _,b in views[1:]]
        self.assertTrue(any(b[0]==0 and b[1]==0 for b in boxes))
        self.assertTrue(any(b[2]==1800 and b[3]==1000 for b in boxes))
        self.assertTrue(all(b[2]-b[0]<=640 and b[3]-b[1]<=640 for b in boxes))
        self.assertEqual(views,search_views(shape,6))


if __name__=='__main__':unittest.main()
