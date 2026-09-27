import unittest
from sentinel_edge.detector_adaptation_eval import class_coverage, match_assignments, match_counts, normalised_targets


class MatchingTests(unittest.TestCase):
    def test_label_restore_uses_actual_degraded_dimensions(self):
        import numpy as np
        from sentinel_edge.detector_comparison import transformed
        image=np.zeros((101,201,3),np.uint8)
        labels=[[0,.5,.5,.4,.2]]
        target=normalised_targets(labels,image.shape)[0]
        changed,expected=transformed(image,'resolution_quarter',target)
        np.testing.assert_allclose(normalised_targets(labels,changed.shape)[0],expected)

    def test_subclass_coverage_is_localisation_not_classification(self):
        rows=[{'target_class_iou50':{'7':{'matched':2,'annotations':4}}},
              {'target_class_iou50':{'7':{'matched':1,'annotations':2}}}]
        self.assertEqual(class_coverage(rows)['7'],{'matched':3,'annotations':6,'annotation_recall':.5})
        self.assertEqual(set(match_assignments([[0,0,10,10]],[[0,0,10,10]],.5)),{0})
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

    def test_augmenting_path_prevents_greedy_undercount(self):
        # First prediction overlaps both objects; second only the first.
        # A greedy assignment would credit one rather than two detections.
        self.assertEqual(match_counts([[0,0,15,10],[0,0,10,10]],
                                      [[0,0,10,10],[5,0,15,10]],.5)['matched'],2)


if __name__=='__main__':unittest.main()
