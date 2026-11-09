import unittest

from sentinel_edge.roadtext_corpus import archive_member_kind,blocked_splits,check_splits,grouped_splits,native_box,plate_text


def label(text='MH01AB1234',**changes):
    return dict(attributes={'English':[1,'Licence_Plates']},category='English',ocr=text,**changes)


class RoadTextCorpusTests(unittest.TestCase):
    def test_archive_admits_bundled_annotations_without_admitting_arbitrary_paths(self):
        self.assertEqual(archive_member_kind('I_400/2402.mp4','I_400'),'video')
        self.assertEqual(archive_member_kind('I_400/400.json','I_400'),'annotations')
        for name in ('I_400/../2402.mp4','/I_400/2402.mp4','I_200/2402.mp4','I_400/other.json'):
            with self.assertRaises(ValueError):archive_member_kind(name,'I_400')

    def test_source_plate_category_required_and_text_not_corrected(self):
        self.assertEqual(plate_text(label('mh-01 ab 1234'))[0],'MH01AB1234')
        self.assertEqual(plate_text(dict(label(),attributes={'English':[0,'English_Text']}))[1],'not_plate_category')
        self.assertEqual(plate_text(label('MH0IAB1234'))[0],'MH0IAB1234')

    def test_fragments_unknown_and_illegible_not_supervised_as_full_plates(self):
        for text in (None,'','###','1234','MH01AB','MH01AB12?4'):
            self.assertIsNone(plate_text(label(text))[0])
        self.assertEqual(plate_text(dict(label(),category='Illegible'))[1],'illegible')
        self.assertIsNotNone(plate_text(label('23BH9715D'))[0])

    def test_fractional_native_bounds_expand_and_clamp(self):
        self.assertEqual(native_box(dict(x1=-.5,y1=1.5,x2=30.2,y2=12.1),30,20),(0,1,30,13))
        self.assertIsNone(native_box(dict(x1=0,y1=0,x2=float('nan'),y2=12),30,20))
        self.assertIsNone(native_box(dict(x1=0,y1=0,x2=10,y2=12),30,20))

    def test_shared_video_or_registration_cannot_cross_splits(self):
        rows=[dict(video_id='1',transcription='MH01AB1234',decoded_pixel_sha256='a'),
              dict(video_id='1',transcription='MH02AB5678',decoded_pixel_sha256='b'),
              dict(video_id='2',transcription='MH02AB5678',decoded_pixel_sha256='c')]
        result=grouped_splits(rows)
        self.assertEqual(len({r['split'] for r in result}),1)
        self.assertEqual(grouped_splits(list(reversed(rows))),list(reversed(result)))
        self.assertTrue(check_splits(result))
        changed=[dict(r) for r in result]
        changed[0]['split']='test' if result[0]['split']!='test' else 'train'
        with self.assertRaises(ValueError):check_splits(changed)

    def test_repeated_pixels_do_not_cross_sets(self):
        with self.assertRaises(ValueError):
            check_splits([dict(video_id='1',transcription='MH01AB1234',decoded_pixel_sha256='same',split='train'),
                          dict(video_id='2',transcription='MH02AB5678',decoded_pixel_sha256='same',split='test')])

    def test_video_blocks_quarantine_every_occurrence_of_a_crossing_plate(self):
        rows=[dict(video_id=str(i),transcription=f'MH01AB{i:04}',decoded_pixel_sha256=str(i)) for i in range(1,11)]
        rows.extend([dict(video_id='1',transcription='MH02AB1234',decoded_pixel_sha256='x'),
                     dict(video_id='8',transcription='MH02AB1234',decoded_pixel_sha256='y')])
        selected,excluded=blocked_splits(rows)
        self.assertEqual(excluded,dict(cross_block_registrations=1,cross_block_crops=2))
        self.assertEqual([sum(r['split']==s for r in selected) for s in ('train','validation','test')],[5,2,3])
        self.assertTrue(check_splits(selected))
        reverse,counts=blocked_splits(list(reversed(rows)))
        self.assertEqual(reverse,list(reversed(selected)))
        self.assertEqual(counts,excluded)


if __name__=='__main__':unittest.main()
