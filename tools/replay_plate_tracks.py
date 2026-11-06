"""Compare the opt-in plate-reading settings on saved probe crops, per crop and per track.

Run inside the recorded-analysis container (same OCR model and runtime):

    python /workspace/tools/run_ocr_research.py /workspace/tools/replay_plate_tracks.py \
        /tmp/eval/probe-general /tmp/eval/crops.csv /tmp/eval/labels_draft.csv /tmp/eval/out

Inputs are the outputs of tools/plate_crop_probe.py (crops plus report.json with track and
PTS), the crop_id -> probe file map and a labels CSV for tools/eval_plates.py. Every probe
crop is read once with the real OCR; the settings then only change decoding and fusion.

Writes, for tools/eval_plates.py:
- `crop_details.csv`: each labelled crop's track, label and validator status, with
  `format-adjusted, unverified` in `*_note` wherever the validator corrected or created
  the reading.
- `track_details.csv`: the same note for each track's accepted and top readings.
- `crop_predictions.csv`: labelled crops read alone: greedy text and the one-crop reader,
  each with the format validator off and on. Voting needs several frames, so it has no
  single-crop column.
- `track_labels.csv` / `track_predictions.csv`: each probe track replayed in PTS order
  through CTCPlateReader with the settings off/on. `<variant>_accepted` is the first plate
  the unchanged gate accepted; `<variant>_top` is the top candidate at the end, which is
  what the overlay would show (masked). A track's label comes from its labelled crops.

The replay keeps the pipeline's 2,500 ms gap reset and 150 ms spacing but not its
plate-centre reset or the non-plate filter, and it reuses the probe's tracks rather than
re-running tracking. Outputs hold full plate text: keep them out of Git.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, '/workspace/edge')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_plates import NO_PLATE, UNREADABLE, load_labels  # noqa: E402
from sentinel_edge.ctc_fusion import CTCPlateReader, greedy_text  # noqa: E402
from sentinel_edge.plate_format import check_plate  # noqa: E402

CACHE = Path('/models/experiments/awiros')
CHECKPOINT = CACHE / 'adaptations/indian-corpus-encoder.safetensors'
GAP_MS = 2500
ADJUSTED = 'format-adjusted, unverified'
VARIANTS = {'baseline': {}, 'validator': {'validate_format': True},
            'vote': {'position_vote': True}, 'validator_vote': {'validate_format': True, 'position_vote': True}}


def note(candidate):
    return ADJUSTED if candidate is not None and candidate.format_adjusted else ''


def one_crop(rows, tokens, **options):
    """Top candidate of a one-crop reader and its format-adjusted note."""
    reader = CTCPlateReader(tokens, **options)
    reader.add(rows, timestamp_ms=0, source_id='crop')
    candidates = reader.candidates()
    return (candidates[0].plate, note(candidates[0])) if candidates else ('', '')


def track_label(labels):
    """A track's label from its labelled crops: a readable plate if they agree on one."""
    plates = {label for label in labels if label not in (NO_PLATE, UNREADABLE)}
    if len(plates) == 1:
        return plates.pop()
    if plates:
        return UNREADABLE   # Conflicting readings: a person must decide.
    return UNREADABLE if UNREADABLE in labels else NO_PLATE


def replay(observations, tokens, options):
    """observations: [(pts_ms, source_id, rows)] in PTS order -> (first accepted, final top)."""
    result = replay_with_notes(observations, tokens, options)
    return result['accepted'], result['top']


def replay_with_notes(observations, tokens, options):
    """As `replay`, plus the format-adjusted note of the accepted and top readings."""
    reader, accepted, accepted_note = None, '', ''
    for pts, source, rows in observations:
        if reader is None or (reader.samples and pts - reader.samples[-1][0] > GAP_MS):
            reader = CTCPlateReader(tokens, **options)
        if reader.samples and pts - reader.samples[-1][0] < 150:
            continue
        reader.add(rows, timestamp_ms=pts, source_id=source)
        if not accepted and reader.observation_count >= 3:
            best = reader.accepted()
            accepted, accepted_note = (best.plate, note(best)) if best else ('', '')
    candidates = reader.candidates() if reader is not None and reader.observation_count >= 3 else []
    top = candidates[0] if candidates else None
    return {'accepted': accepted, 'accepted_note': accepted_note,
            'top': top.plate if top else '', 'top_note': note(top)}


def write(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('probe', type=Path, help='plate_crop_probe output: crops and report.json')
    parser.add_argument('sources', type=Path, help='crops.csv: crop_id -> source_file')
    parser.add_argument('labels', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from sentinel_edge.awiros_experiment import AwirosExperiment
    ocr = AwirosExperiment(CACHE, device='cpu', checkpoint=CHECKPOINT)
    tokens = ocr.tokens
    report = json.loads((args.probe / 'report.json').read_text())
    labels = load_labels(args.labels)
    with open(args.sources, newline='', encoding='utf-8') as handle:
        crop_of = {row['source_file']: row['crop_id'] for row in csv.DictReader(handle)}

    tracks = {}
    crop_rows, details = [], []
    for entry in report['crops']:
        rows = ocr.probabilities(cv2.imread(str(args.probe / entry['file'])))
        tracks.setdefault(entry['track'], []).append((entry['pts_ms'], entry['file'], rows))
        crop_id = crop_of.get(entry['file'])
        if crop_id in labels:
            greedy = greedy_text(rows, tokens)
            checked = check_plate(greedy)
            validator_format, validator_format_note = one_crop(rows, tokens, validate_format=True)
            crop_rows.append({'crop_id': crop_id,
                              'baseline_raw': greedy, 'validator_raw': checked.plate or '',
                              'baseline_format': one_crop(rows, tokens)[0],
                              'validator_format': validator_format})
            details.append({'crop_id': crop_id, 'track': entry['track'], 'label': labels[crop_id],
                            'validator_status': checked.status,
                            'validator_raw_note': ADJUSTED if checked.status == 'corrected' else '',
                            'validator_format_note': validator_format_note})
    write(args.output / 'crop_predictions.csv', sorted(crop_rows, key=lambda row: row['crop_id']))
    write(args.output / 'crop_details.csv', sorted(details, key=lambda row: row['crop_id']))

    labelled_tracks = {}
    for row in details:
        labelled_tracks.setdefault(row['track'], []).append(row['label'])
    track_rows, track_notes, label_rows = [], [], []
    for track, observations in sorted(tracks.items(), key=lambda item: str(item[0]).zfill(8)):
        observations.sort(key=lambda item: item[0])
        row, notes = {'crop_id': f't{track}'}, {'crop_id': f't{track}'}
        for name, options in VARIANTS.items():
            result = replay_with_notes(observations, tokens, options)
            row[f'{name}_accepted'], row[f'{name}_top'] = result['accepted'], result['top']
            notes[f'{name}_accepted_note'], notes[f'{name}_top_note'] = result['accepted_note'], result['top_note']
        track_rows.append(row)
        track_notes.append(notes)
        if track in labelled_tracks:
            label_rows.append({'crop_id': f't{track}', 'label': track_label(labelled_tracks[track])})
    write(args.output / 'track_predictions.csv', track_rows)
    write(args.output / 'track_details.csv', track_notes)
    write(args.output / 'track_labels.csv', label_rows)
    print(f'{len(report["crops"])} crops read, {len(crop_rows)} labelled; '
          f'{len(track_rows)} tracks, {len(label_rows)} with a label')


if __name__ == '__main__':
    main()
