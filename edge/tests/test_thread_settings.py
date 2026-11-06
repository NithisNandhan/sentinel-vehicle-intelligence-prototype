"""Opt-in CPU thread counts keep the long-standing default of 2."""
import os
import unittest
from unittest.mock import patch

from sentinel_edge.detector_comparison import configured_threads


class ThreadSettingTests(unittest.TestCase):
    def test_default_is_unchanged(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(configured_threads('SENTINEL_VEHICLE_THREADS'), 2)
        with patch.dict(os.environ, {'SENTINEL_VEHICLE_THREADS': ''}):
            self.assertEqual(configured_threads('SENTINEL_VEHICLE_THREADS'), 2)

    def test_explicit_value_is_used(self):
        with patch.dict(os.environ, {'SENTINEL_PLATE_THREADS': '6'}):
            self.assertEqual(configured_threads('SENTINEL_PLATE_THREADS'), 6)

    def test_invalid_values_fail_loudly(self):
        for value in ('0', '65', 'four', '-2'):
            with patch.dict(os.environ, {'SENTINEL_PLATE_THREADS': value}):
                with self.assertRaises(ValueError):
                    configured_threads('SENTINEL_PLATE_THREADS')


if __name__ == '__main__':
    unittest.main()
