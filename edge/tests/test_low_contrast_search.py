"""Discovery tests with synthetic glyphs and stub models; no real plate labels."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from sentinel_edge.low_contrast_search import LowContrastPlateSearch, character_groups, contrast_view
from sentinel_edge.plate_context import context_proposals


def scene():
    image = np.full((200, 400, 3), 180, np.uint8)
    cv2.putText(image, 'AB12345', (100, 155), cv2.FONT_HERSHEY_SIMPLEX, 1., (60, 60, 60), 2)
    return image


class Primary:
    last_confidence = .8
    last_text_quad = last_completion = None
    last_calls = 1
    last_fallback = False

    def __init__(self, box=None, polygons=(), detections=()):
        self.result = box
        self.last_raw_proposals = [(box, 'plate', .8)] if box else []
        self.text_detector = SimpleNamespace(polygons=lambda image: list(polygons))
        self.detector = SimpleNamespace(scored_boxes=lambda image: list(detections))

    def box(self, image):
        return self.result

    def begin_frame(self):
        pass

    def diagnostics(self):
        return {'text_detector_enabled': True}


class DiscoveryTests(unittest.TestCase):
    def test_contrast_is_same_geometry_and_does_not_mutate_native_pixels(self):
        image = scene()
        original = image.copy()
        view = contrast_view(image)
        self.assertEqual(view.shape, image.shape)
        self.assertEqual(view.dtype, np.uint8)
        self.assertTrue(np.array_equal(original, image))
        self.assertFalse(np.shares_memory(view, image))
        self.assertFalse(np.array_equal(view, image))

    def test_character_group_can_be_found_without_any_plate_detector_seed(self):
        image = scene()
        polygon = [[98, 130], [244, 130], [244, 158], [98, 158]]
        pool = LowContrastPlateSearch(Primary(polygons=[polygon]), character_guided=True)
        result = pool.box(image)
        self.assertIsNotNone(result)
        self.assertEqual(pool.last_candidates[0]['proposal_sources'], ['character_group'])
        self.assertIsNone(pool.last_confidence, 'Text is not a calibrated plate confidence')
        self.assertEqual(character_groups(np.full_like(image, 180), [polygon]), [])

    def test_contrast_detector_receives_working_copy_native_is_preserved(self):
        image = scene()
        primary = Primary(detections=[([95, 125, 250, 165], .61)])
        received = []
        primary.detector.scored_boxes = lambda value: received.append(value.copy()) or [([95,125,250,165], .61)]
        pool = LowContrastPlateSearch(primary, contrast_pass=True)
        original = image.copy()
        self.assertIsNotNone(pool.box(image))
        self.assertTrue(np.array_equal(received[0], contrast_view(original)))
        self.assertTrue(np.array_equal(image, original))
        self.assertEqual(pool.last_confidence, .61)

    def test_baseline_wins_duplicate_without_union_or_extra_votes(self):
        pool = LowContrastPlateSearch(Primary(box=[90,120,260,170],
            detections=[([92,122,258,168], .9)]), contrast_pass=True)
        self.assertEqual(pool.box(scene()), (90,120,260,170))
        self.assertEqual(len(pool.last_candidates), 1)
        self.assertEqual(pool.last_candidates[0]['proposal_sources'], ['contrast_plate','existing_search'])

    def test_many_contrast_boxes_cannot_starve_character_discovery(self):
        polygon = [[98,130],[244,130],[244,158],[98,158]]
        primary = Primary(box=[10,20,70,40],polygons=[polygon],detections=[
            ([10,60,70,80],.8),([100,50,160,70],.7),([210,40,270,60],.6),([300,100,370,120],.5)])
        pool = LowContrastPlateSearch(primary,character_guided=True,contrast_pass=True)
        pool.box(scene())
        self.assertEqual(len(pool.last_candidates),4)
        self.assertTrue(any('character_group' in c['proposal_sources'] for c in pool.last_candidates))
        self.assertTrue(any('contrast_plate' in c['proposal_sources'] for c in pool.last_candidates))
        self.assertIn('existing_search',pool.last_candidates[0]['proposal_sources'])

    def test_broad_text_region_cannot_swallow_two_small_positive_plates(self):
        first,second=(20,80,60,95),(100,80,140,95)
        primary=Primary(box=first)
        primary.last_raw_proposals.append((second,'plate',.7))
        pool=LowContrastPlateSearch(primary,character_guided=True)
        with patch('sentinel_edge.low_contrast_search.character_groups',return_value=[
                dict(box=(0,60,200,120),characters=6)]):
            pool.box(scene())
        self.assertFalse(any('character_group' in c['proposal_sources'] for c in pool.last_candidates))
        self.assertEqual(pool.metrics['separate_plate_vetoes'],1)

    def test_primary_text_support_is_not_reported_as_plate_detector_support(self):
        primary=Primary(box=[90,120,260,170])
        primary.last_sources=('text',)
        primary.last_confidence=None
        pool=LowContrastPlateSearch(primary,contrast_pass=True)
        pool.box(scene())
        self.assertEqual(pool.last_candidates[0]['proposal_sources'],['existing_search'])
        self.assertEqual(pool.last_candidates[0]['existing_support'],['text'])
        self.assertIsNone(pool.last_confidence)

    def test_an_empty_completed_text_pass_is_not_repeated_on_identical_pixels(self):
        primary=Primary()
        primary.metrics={'text_detector_calls':0}
        def original(image):
            primary.metrics['text_detector_calls']+=1
            return None
        primary.box=original
        pool=LowContrastPlateSearch(primary,character_guided=True)
        pool.box(scene())
        self.assertEqual(pool.metrics['independent_text_calls'],0)

    def test_budget_is_bounded_and_resets_per_frame(self):
        pool = LowContrastPlateSearch(Primary(), contrast_pass=True, max_extra_calls=1)
        pool.begin_frame()
        pool.box(scene()); pool.box(scene())
        self.assertEqual(pool.metrics['contrast_calls'], 1)
        self.assertEqual(pool.metrics['extra_budget_skips'], 1)
        pool.begin_frame(); pool.box(scene())
        self.assertEqual(pool.metrics['contrast_calls'], 2)

    def test_context_retains_alternatives_and_projects_to_native_coordinates(self):
        primary = Primary(box=[30,60,90,85], detections=[([120,80,190,105], .7)])
        pool = LowContrastPlateSearch(primary, contrast_pass=True)
        frame = np.zeros((400,600,3), np.uint8)
        proposals, metrics = context_proposals(frame,[dict(box=[100,100,300,300])], pool, max_recoveries=0)
        self.assertEqual(len(proposals), 2)
        self.assertEqual(proposals[0]['box'], [82,112,142,137])
        self.assertIn('contrast_plate', proposals[1]['proposal_sources'])
        self.assertEqual(metrics['discovery_alternatives'], 1)

    def test_invalid_polygons_and_far_apart_text_are_not_joined(self):
        self.assertEqual(character_groups(scene(),[[[0,0],[400,0],[400,199],[0,199]]]), [])
        with self.assertRaises(ValueError):
            LowContrastPlateSearch(Primary())


if __name__ == '__main__':
    unittest.main()
