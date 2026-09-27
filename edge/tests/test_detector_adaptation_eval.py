import unittest
from sentinel_edge.detector_adaptation_eval import match_counts


class MatchingTests(unittest.TestCase):
    def test_no_double_credit(self):
        result=match_counts([[0,0,10,10],[0,0,10,10]],[[0,0,10,10]],.5)
        self.assertEqual(result['matched'],1)
        self.assertEqual(result['unmatched_predictions'],1)

    def test_empty_predictions(self):
        self.assertEqual(match_counts([],[[0,0,10,10]],.5)['missed_annotations'],1)

    def test_two_objects(self):
        boxes=[[0,0,10,10],[20,20,30,30]]
        self.assertEqual(match_counts(boxes,list(reversed(boxes)),.5)['matched'],2)

    def test_threshold_matters(self):
        self.assertEqual(match_counts([[0,0,10,10]],[[2,2,12,12]],.75)['matched'],0)


if __name__=='__main__':unittest.main()
