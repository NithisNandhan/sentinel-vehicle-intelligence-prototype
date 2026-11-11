import unittest
from probe_text_corners import checked_schedule


class ProbeScheduleTests(unittest.TestCase):
    def test_timestamp_schedule_cannot_supply_plate_text_or_manual_geometry(self):
        self.assertEqual(checked_schedule({'source_sha256':'s','targets':[{'id':'x','position_ms':0}]},'s'),[0])
        for field,value in [('plate','AB123'),('box',[0,0,100,20])]:
            with self.subTest(field=field),self.assertRaises(ValueError):
                checked_schedule({'source_sha256':'s','targets':[{'id':'x','position_ms':0,field:value}]},'s')

    def test_wrong_source_or_ambiguous_targets_fail(self):
        with self.assertRaises(ValueError):checked_schedule({'source_sha256':'other'},'s')
        with self.assertRaises(ValueError):checked_schedule({'source_sha256':'s','targets':[]},'s')


if __name__=='__main__':unittest.main()
