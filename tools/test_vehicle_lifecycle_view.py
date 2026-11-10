import unittest
from vehicle_lifecycle_view import group_vehicle_cards


class VehicleViewTests(unittest.TestCase):
    def test_all_crops_preserved_and_grouped_without_guessing_owner(self):
        document=dict(groups=[dict(id='PT0001',parent_vehicle_id='VT0001'),dict(id='PT0002',parent_vehicle_id=None)],
                      vehicles=[dict(id='VT0001',status='visible')],vehicle_results=[])
        page='<section><h2>All 18 development references</h2></section><section id="PT0001"><img src="one.png"></section><section id="PT0002"><img src="two.png"></section>'
        result=group_vehicle_cards(page,document)
        self.assertEqual(result.count('id="PT0001"'),1);self.assertEqual(result.count('id="PT0002"'),1)
        self.assertIn('id="VT0001"',result);self.assertIn('1 plate histories remain unassigned',result)
        self.assertIn('one.png',result);self.assertIn('two.png',result)
        self.assertLess(result.index('id="VT0001"'),result.index('id="PT0001"'))
        self.assertLess(result.index('id="unassigned-plate-views"'),result.index('id="PT0002"'))

    def test_missing_evidence_fails_instead_of_hiding_a_history(self):
        document=dict(groups=[dict(id='PT0001')],vehicles=[],vehicle_results=[])
        with self.assertRaises(ValueError):group_vehicle_cards('',document)

    def test_duplicate_identity_cards_fail_instead_of_dropping_evidence(self):
        document=dict(groups=[dict(id='PT0001')],vehicles=[],vehicle_results=[])
        page='<section id="PT0001">first</section><section id="PT0001">second</section>'
        with self.assertRaises(ValueError):group_vehicle_cards(page,document)


if __name__=='__main__':unittest.main()
