import unittest
from render_plate_results import build_results,render_html


def group(name,top,**extra):
    return dict(id=name,methods=dict(reconstruction=dict(top=top)),samples=[],**extra)


class PlateResultsTests(unittest.TestCase):
    def test_correct_reconstruction_anywhere_counts_once_without_frame_matching(self):
        doc=dict(source_sha256='source',groups=[group('PT1','AB123'),group('PT2','AB123')])
        score=dict(cases=[dict(id='09',expected='AB123',group=None)])
        manifest=dict(source_sha256='source',samples=[dict(id='09',crop='plate09.png')])
        result=build_results(doc,score,manifest)
        self.assertEqual((result['correct'],result['total']),(1,1))
        self.assertEqual(result['results'][0]['number'],1)
        self.assertEqual(result['results'][0]['reference_id'],'09')

    def test_other_methods_and_quarantined_outputs_cannot_inflate_the_score(self):
        excluded=group('PT1','AB123',plate_validation=dict(excluded_from_fusion=True))
        wrong=group('PT2','WRONG');wrong['methods']['single']=dict(top='AB123')
        wrong['methods']['reconstruction']['candidates']=[dict(plate='AB123')]
        doc=dict(source_sha256='source',groups=[excluded,wrong])
        score=dict(cases=[dict(id='01',expected='AB123',group='PT2')])
        manifest=dict(source_sha256='source',samples=[dict(id='01',crop='plate01.png')])
        result=build_results(doc,score,manifest)
        self.assertEqual(result['correct'],0)
        self.assertEqual(result['results'][0]['ocr'],'WRONG')

    def test_wrong_source_cannot_reuse_labels(self):
        with self.assertRaises(ValueError):build_results(dict(source_sha256='a'),{},dict(source_sha256='b'))

    def test_saved_labels_cannot_inject_html_or_scripts(self):
        html=render_html(dict(results=[dict(actual='</script><script>alert(1)</script>')],correct=0,total=1,percent=0))
        self.assertNotIn('</script><script>alert(1)</script>',html)
        self.assertIn('\\u003c/script\\u003e',html)


if __name__=='__main__':unittest.main()
