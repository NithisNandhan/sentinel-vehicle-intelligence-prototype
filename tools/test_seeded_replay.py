"""Edge -> outbox -> HTTP -> backend replay on SYNTHETIC scripted passages. Skipped without backend deps."""
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent))
HAVE = all(importlib.util.find_spec(name) for name in ('uvicorn', 'fastapi', 'sqlalchemy', 'cv2', 'numpy'))


@unittest.skipUnless(HAVE, 'needs backend requirements plus numpy/opencv in one interpreter')
class ReplayTests(unittest.TestCase):
    def run_replay(self, partial):
        import seeded_watchlist_replay as replay
        return replay.run(replay.default_cases(), partial_matching=partial, python=sys.executable)

    def test_matching_disabled_marks_review_metrics_not_applicable(self):
        metrics = self.run_replay(False)
        self.assertEqual((metrics['missed_confirmed_alerts']['count'], metrics['false_confirmed_alerts']['count'],
                          metrics['duplicate_alerts'], metrics['delivery_failures']), (0, 0, 0, 0))
        self.assertEqual(metrics['alerts_total'], 2)
        self.assertEqual(metrics['queued_while_central_down'], metrics['events_queued'])
        self.assertEqual((metrics['outbox_depth_at_end'], metrics['alerts_changed_by_redelivery']), (0, 0))
        self.assertEqual(metrics['result_events_persisted'], metrics['result_events_expected'])
        self.assertEqual(metrics['partial_matching']['review_recall'], 'not applicable')

    def test_partial_leads_open_reviews_but_never_alert(self):
        metrics = self.run_replay(True)
        self.assertEqual(metrics['alerts_total'], 2, 'a lead must never create an alert')
        self.assertEqual(metrics['reviews_pending'], 2)
        match = metrics['partial_matching']
        self.assertEqual(match['review_recall'], {'count': 1, 'of': 1})
        self.assertEqual(match['false_review_leads'], 1, 'the known false-lead scenario is measured, not hidden')
        self.assertEqual(metrics['false_confirmed_alerts']['count'], 0)


if __name__ == '__main__':
    unittest.main()


class ShadowCompareTests(unittest.TestCase):
    def report(self, wrong=0, states=True, passed=True, labels='h'):
        base = {'labels_sha256': labels, 'totals': {'accepted_readings': {'correct': 1, 'wrong': wrong},
                                                    'exact_read_coverage': {'count': 1, 'of': 2, 'rate': .5}}, 'recordings': []}
        if states:
            base.update(profile={'profile': 'candidate'}, release_gates={'passed': passed},
                        states_totals={'accepted': {'correct': 1, 'wrong': wrong}, 'coverage': {'count': 1, 'of': 2},
                                       'partial': {'with_wrong_known_character': 0}, 'unreadable_rate': {'count': 0, 'of': 2}})
        return base

    def test_promotion_blockers(self):
        import shadow_compare
        self.assertTrue(shadow_compare.compare(self.report(states=False), self.report())['promotion_allowed'])
        self.assertIn('candidate confirmed a wrong plate', shadow_compare.compare(self.report(), self.report(wrong=1))['blockers'])
        self.assertFalse(shadow_compare.compare(self.report(), self.report(passed=False))['promotion_allowed'])
        self.assertFalse(shadow_compare.compare(self.report(), self.report(states=False))['promotion_allowed'])
        with self.assertRaises(ValueError):
            shadow_compare.compare(self.report(), self.report(labels='other'))
