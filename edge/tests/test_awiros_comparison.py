import importlib.util
from pathlib import Path
import unittest


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parents[2]/'tools/compare_awiros_checkpoints.py'
        spec=importlib.util.spec_from_file_location('awiros_comparison',path)
        cls.tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.tool)

    def test_equal_totals_can_hide_one_recovery_and_one_regression(self):
        original=[dict(id='a',expected='MH01AB1234',reading='MH01AB1230'),
                  dict(id='b',expected='MH02AB1234',reading='MH02AB1234')]
        adapted=[dict(original[1],reading='MH02AB1230'),dict(original[0],reading='MH01AB1234')]
        result=self.tool.paired_summary(original,adapted)
        self.assertEqual(result['original']['exact'],1)
        self.assertEqual(result['adapted']['exact'],1)
        self.assertEqual((result['recovered'],result['regressed']),(1,1))
        self.assertEqual(result['unchanged_readings'],0)

    def test_different_samples_labels_or_duplicate_ids_are_rejected(self):
        row=dict(id='a',expected='MH01AB1234',reading='MH01AB1234')
        for other in ([dict(row,id='b')],[dict(row,expected='MH02AB1234')],[row,row]):
            with self.assertRaises(ValueError):self.tool.paired_summary([row],other)


if __name__=='__main__':unittest.main()
