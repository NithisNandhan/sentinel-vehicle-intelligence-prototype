"""Cached GPU OCR/reconstruction plumbing check on explicitly synthetic pixels.

Not live footage, an accuracy test, or evidence of improved recognition.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from sentinel_edge.awiros_experiment import AwirosExperiment
from sentinel_edge.tracked_ocr import TrackedOCR


def main():
    ocr = AwirosExperiment(Path('/models/experiments/awiros'), device='gpu',
                          checkpoint=Path('/models/experiments/awiros/adaptations/indian-corpus-encoder.safetensors'))
    engine = TrackedOCR(ocr, SimpleNamespace(box=lambda image:(0, 0, 320, 80)),
                        debug=True, reconstruction=True)
    track = SimpleNamespace(track_id='synthetic-1', box=(0, 0, 320, 80), vehicle_class=2)
    for i in range(3):
        image = np.full((80, 320, 3), 235, np.uint8)
        cv2.putText(image, 'AP01AB1234', (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (15, 15, 15), 2)
        image[0, i] = i
        engine.process(image, [track], i*500)
    result = engine.debug_snapshot()['tracks'][0]['reconstruction']
    print(json.dumps(dict(source_mode='synthetic', diagnostic_only=True, accuracy=None,
                          device='gpu', reconstruction_status=result['status'],
                          contributing_originals=result.get('contributing_originals', 0),
                          original_observations=engine.histories['synthetic-1'].reader.observation_count,
                          additional_temporal_votes=result.get('extra_observations'),
                          derived_character_count=len(result.get('characters', [])),
                          reconstruction_and_derived_ocr_ms=result['elapsed_ms'],
                          image_saved=False, summary=engine.summary()), indent=2))
    engine.reset()
    if result['status'] != 'reconstructed':
        raise RuntimeError('GPU reconstruction/OCR plumbing check failed')


if __name__ == '__main__':main()
