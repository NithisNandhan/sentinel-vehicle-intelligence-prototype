import unittest
from sentinel_edge.owned_plate import OwnedPlateRouter


def vehicle(name,box=(0,0,100,100),qualified=True):return dict(id=name,box=box,qualified=qualified)
def plate(box=(20,50,60,70),confidence=.8):return dict(box=box,confidence=confidence)


class OwnershipTests(unittest.TestCase):
    def test_different_vehicles_never_share_a_plate_history(self):
        router=OwnedPlateRouter()
        first=router.route([plate()],[vehicle('A')],0)[0]
        second=router.route([plate()],[vehicle('B')],250)[0]
        self.assertNotEqual(first['group'],second['group'])

    def test_overlapping_tentative_neighbour_blocks_fusion(self):
        router=OwnedPlateRouter()
        self.assertEqual(router.route([plate()],[vehicle('A'),vehicle('B',qualified=False)],0),[])
        self.assertEqual(router.counters['ambiguous_owner'],1)

    def test_reviewed_unsafe_histories_are_excluded(self):
        router=OwnedPlateRouter(['sign'])
        self.assertEqual(router.route([plate()],[vehicle('sign')],0),[])

    def test_two_simultaneous_nonoverlapping_vehicles_keep_separate_evidence(self):
        router=OwnedPlateRouter()
        rows=router.route([plate(),plate((140,50,180,70))],
                         [vehicle('A'),vehicle('B',(120,0,220,100))],0)
        self.assertEqual({r['owner'] for r in rows},{'A','B'})
        self.assertEqual(len({r['group'] for r in rows}),2)

    def test_excluded_overlapping_history_cannot_redirect_a_crop_to_its_neighbour(self):
        router=OwnedPlateRouter(['unsafe'])
        self.assertEqual(router.route([plate()],[vehicle('unsafe'),vehicle('other')],0),[])
        self.assertEqual(router.counters['ambiguous_owner'],1)

    def test_duplicates_do_not_add_votes_and_two_distinct_plates_do_not_mix(self):
        router=OwnedPlateRouter()
        self.assertEqual(len(router.route([plate(),plate()],[vehicle('A')],0)),1)
        self.assertEqual(router.route([plate((5,5,35,20)),plate((60,70,90,90))],[vehicle('A')],250),[])

    def test_gap_restarts_evidence_and_normal_updates_preserve_it(self):
        router=OwnedPlateRouter()
        first=router.route([plate()],[vehicle('A')],0)[0]
        second=router.route([plate()],[vehicle('A')],250)[0]
        third=router.route([plate()],[vehicle('A')],1501)[0]
        self.assertEqual(first['group'],second['group'])
        self.assertNotEqual(second['group'],third['group'])

    def test_no_tentative_unowned_or_nonforward_observations(self):
        router=OwnedPlateRouter()
        self.assertEqual(router.route([plate()],[vehicle('A',qualified=False)],0),[])
        self.assertEqual(router.route([plate()],[],250),[])
        with self.assertRaises(ValueError):router.route([],[],250)
