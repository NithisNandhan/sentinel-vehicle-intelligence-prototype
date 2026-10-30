"""Read labelled plate crops with the pipeline's OCR under several variants.

Run inside the recorded-analysis container (same OCR model and runtime):

    python /workspace/tools/run_plate_variants.py /tmp/review/crops /tmp/review/predictions.csv \
        --features /tmp/review/features.csv --scores /tmp/review/probe-general/report.json \
        --sources /tmp/review/crops.csv

Each variant yields two columns: `<variant>_raw` (greedy CTC text, what the network
literally emits) and `<variant>_format` (best Indian-format candidate from a one-crop
fusion reader, empty when none). An empty cell means no reading. Score the output with
tools/eval_plates.py. Diagnostic only: no setting changes.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, '/workspace/edge')
from sentinel_edge.awiros_experiment import AwirosExperiment  # noqa: E402
from sentinel_edge.ctc_fusion import CTCPlateReader, greedy_text  # noqa: E402

CACHE = Path('/models/experiments/awiros')
CHECKPOINT = CACHE / 'adaptations/indian-corpus-encoder.safetensors'


def format_reading(rows, tokens):
    reader = CTCPlateReader(tokens)
    reader.add(rows, timestamp_ms=0, source_id='crop')
    candidates = reader.candidates()
    return candidates[0].plate if candidates else ''


def peak_stats(rows, tokens):
    """Peak probability of each non-blank greedy character: uncalibrated, diagnostics only."""
    peaks, previous = [], None
    for row in rows:
        index = max(range(len(row)), key=row.__getitem__)
        if index and index != previous:
            peaks.append(row[index])
        previous = index
    return (round(sum(peaks) / len(peaks), 4) if peaks else 0.), sum(p >= .6 for p in peaks), len(peaks)


def variants(reader, enabled):
    """name -> function(crop) returning CTC rows, or None for no reading."""
    def enlarge(factor):
        return lambda crop: reader.probabilities(cv2.resize(crop, None, fx=factor, fy=factor,
                                                            interpolation=cv2.INTER_CUBIC))

    def two_line(crop):
        from sentinel_edge.two_line import row_aware_probabilities
        return row_aware_probabilities(reader, crop)

    def reject_non_plates():
        from sentinel_edge.plate_filter import NonPlateFilter  # Needs the detector score; see main().
        return NonPlateFilter()
    found = {'baseline': reader.probabilities, 'enlarge2x': enlarge(2), 'enlarge3x': enlarge(3),
             'two_line': two_line, 'reject_non_plates': reject_non_plates}
    return {name: found[name] for name in enabled}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('crops', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--variants', default='baseline')
    parser.add_argument('--features', type=Path)
    parser.add_argument('--scores', type=Path, help='plate_crop_probe report.json with detector scores')
    parser.add_argument('--sources', type=Path, help='crops.csv mapping crop_id to the probe file')
    args = parser.parse_args()
    reader = AwirosExperiment(CACHE, device='cpu', checkpoint=CHECKPOINT)
    tokens = reader.tokens
    chosen = variants(reader, args.variants.split(','))
    scores = {}
    if args.scores and args.sources:
        by_file = {row['file']: row.get('detector_confidence') for row in json.loads(args.scores.read_text())['crops']}
        with open(args.sources, newline='') as handle:
            scores = {row['crop_id']: by_file.get(row['source_file']) for row in csv.DictReader(handle)}

    predictions, features = [], []
    for path in sorted(args.crops.glob('*.png')):
        crop = cv2.imread(str(path))
        crop_id = path.stem
        row = {'crop_id': crop_id}
        base_rows = reader.probabilities(crop)
        for name, function in chosen.items():
            if name == 'reject_non_plates':
                reason = function().reason(crop, base_rows, scores.get(crop_id))
                rows = None if reason else base_rows
            else:
                rows = function(crop)
            row[f'{name}_raw'] = greedy_text(rows, tokens) if rows else ''
            row[f'{name}_format'] = format_reading(rows, tokens) if rows else ''
        predictions.append(row)
        mean_peak, confident, characters = peak_stats(base_rows, tokens)
        features.append({'crop_id': crop_id, 'width': crop.shape[1], 'height': crop.shape[0],
                         'aspect': round(crop.shape[1] / crop.shape[0], 3), 'detector_confidence': scores.get(crop_id),
                         'mean_peak': mean_peak, 'confident_characters': confident, 'greedy_characters': characters})
    for path, rows in ((args.output, predictions), (args.features, features)):
        if path is None:
            continue
        with open(path, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f'{len(predictions)} crops, variants: {", ".join(chosen)}')


if __name__ == '__main__':
    main()
