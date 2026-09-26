import unittest
from types import SimpleNamespace

from sentinel_edge.awiros_experiment import author_config, check_state_shapes


class AwirosTests(unittest.TestCase):
    def test_config_is_literal_not_executed(self):
        source = "raise RuntimeError('must not execute')\nCTC_NUM_CLASSES=64\nNRTR_NUM_CLASSES=67\nMODEL_CONFIG={'Architecture': {'count': CTC_NUM_CLASSES}}"
        self.assertEqual(author_config(source), {'count': 64})
        with self.assertRaises(ValueError):
            author_config(source.replace('CTC_NUM_CLASSES}}', 'danger()}}'))

    def test_strict_tensor_loading(self):
        tensors = {'weight': SimpleNamespace(shape=(3, 64))}
        check_state_shapes(tensors, tensors)
        for invalid in ({}, {'extra': SimpleNamespace(shape=(3, 64))},
                        {'weight': SimpleNamespace(shape=(3, 63))}):
            with self.assertRaises(ValueError):
                check_state_shapes(tensors, invalid)
