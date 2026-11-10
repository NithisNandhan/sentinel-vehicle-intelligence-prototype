import json
from pathlib import Path
import tempfile
import unittest
from review_vehicle_association import review


class IdentityReviewTests(unittest.TestCase):
    def test_single_anchor_is_not_a_continuity_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'mode').mkdir()
            (root/'report.json').write_text(json.dumps(dict(source_sha256='source',strategies={'mode':{}})))
            spec=root/'annotations.json'
            spec.write_text(json.dumps(dict(source_sha256='source',anchors=[
                dict(vehicle='A',position_ms=0,box=[0,0,100,100]),dict(vehicle='A',position_ms=100,box=[0,0,100,100]),
                dict(vehicle='B',position_ms=0,box=[200,0,300,100])])) )
            frames=[dict(t=0,q=[['V1',0,0,100,100,2,.9,1],['V3',200,0,300,100,2,.9,1]]),
                    dict(t=100,q=[['V2',0,0,100,100,2,.9,1]])]
            path=root/'mode/timeline.json';path.write_text(json.dumps(dict(frames=frames)))
            result=review(root,spec)['strategies']['mode']
            self.assertEqual(result['matched_anchors'],3);self.assertEqual(result['repeated_vehicle_cases'],1)
            self.assertEqual(result['fully_consistent_cases'],0);self.assertEqual(result['fragmented_cases'],1)
            frames[1]['q'][0][0]='V1';path.write_text(json.dumps(dict(frames=frames)))
            self.assertEqual(review(root,spec)['strategies']['mode']['fully_consistent_cases'],1)


if __name__=='__main__':unittest.main()
