import copy
import unittest

from sentinel_edge.frame_selection import assess_metrics, choose_frames


def metrics(**changes):
    return dict(width_px=200, height_px=50, sharpness=200., contrast=40.,
                glare_fraction=.3, dark_fraction=.1, angle_deg=None, **changes)


class FrameSelectionTests(unittest.TestCase):
    def test_blur_size_and_exposure_are_explicit_rejections(self):
        for key, value, reason in [('sharpness', 5., 'blur_risk'), ('width_px', 20, 'too_few_native_pixels'),
                                   ('contrast', 2., 'low_contrast'), ('glare_fraction', .95, 'extreme_exposure')]:
            assessment = assess_metrics({**metrics(), key:value})
            self.assertFalse(assessment['eligible'])
            self.assertIn(reason, assessment['rejection_reasons'])

    def test_whiteness_alone_does_not_reject_a_legitimate_white_plate(self):
        self.assertTrue(assess_metrics({**metrics(), 'glare_fraction':.65})['eligible'])

    def test_edge_contact_is_not_claimed_to_prove_clipping(self):
        normal = assess_metrics(metrics())
        edge = assess_metrics(metrics(edge_contact_fraction=.8))
        self.assertTrue(edge['eligible'])
        self.assertFalse(edge['completeness_verified'])
        self.assertLess(edge['score'], normal['score'])

    def test_selection_preserves_best_frame_and_returns_chronological_unique_views(self):
        frames = [dict(source_id=str(i), timestamp_ms=i*500) for i in range(6)]
        assessments = {str(i):dict(eligible=True, score=1.-i*.1) for i in range(6)}
        before = copy.deepcopy(frames)
        selected = choose_frames(frames, assessments, limit=3)
        self.assertEqual([f['source_id'] for f in selected], ['0','1','2'])
        self.assertEqual(frames, before)
        # An exact duplicate never supplies another independent view.
        self.assertEqual(len(choose_frames(frames+[frames[0]], assessments, limit=8)), 6)

    def test_rejected_or_too_close_frames_do_not_get_forced_into_fusion(self):
        frames = [dict(source_id=str(i), timestamp_ms=i*100) for i in range(4)]
        assessments = {str(i):dict(eligible=i!=0, score=1.-i*.1) for i in range(4)}
        self.assertEqual([f['source_id'] for f in choose_frames(frames, assessments)], ['1','3'])

    def test_invalid_features_and_unknown_angle_are_handled_explicitly(self):
        self.assertTrue(assess_metrics(metrics())['eligible'])
        for value in (float('nan'), float('inf'), -1., True):
            with self.assertRaises(ValueError):
                assess_metrics({**metrics(), 'sharpness':value})
