import unittest
from sentinel_edge.plate_sampling import PlateBurstSchedule


def parent(key='V',observed=True):return dict(id=key,observed=observed,status='visible' if observed else 'predicted')


class SamplingTests(unittest.TestCase):
    def test_new_vehicle_has_only_three_bounded_extra_searches(self):
        schedule=PlateBurstSchedule();schedule.advance([parent()],0)
        self.assertEqual(schedule.due(),{'V'});schedule.searched({'V'})
        schedule.advance([parent()],50);self.assertEqual(schedule.due(),set())
        schedule.advance([parent()],100);schedule.searched({'V'})
        schedule.advance([parent()],200);schedule.searched({'V'})
        schedule.advance([parent()],300);self.assertEqual(schedule.due(),set())

    def test_per_frame_cap_and_camera_reset(self):
        schedule=PlateBurstSchedule();schedule.advance([parent(str(i)) for i in range(10)],0)
        self.assertEqual(len(schedule.due()),2)
        schedule.advance([],33,reset=True);self.assertEqual(schedule.pending,{})
        with self.assertRaises(ValueError):schedule.advance([],33)

    def test_short_flicker_does_not_continuously_restart_bursts(self):
        schedule=PlateBurstSchedule();schedule.advance([parent()],0);schedule.searched({'V'})
        schedule.advance([parent(observed=False)],33)
        schedule.advance([parent()],67)
        self.assertEqual(schedule.pending['V']['left'],2)
        schedule.advance([parent(observed=False)],1100)
        schedule.advance([parent()],1200);self.assertEqual(schedule.pending['V']['left'],3)
