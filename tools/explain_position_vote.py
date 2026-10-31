"""Why SENTINEL_POSITION_VOTE did or did not add a candidate, per vehicle track.

Run inside the recorded-analysis container (same OCR model and runtime):

    python /workspace/tools/run_ocr_research.py /workspace/tools/explain_position_vote.py \
        /tmp/eval/probe-general /tmp/eval/out/vote_per_track.csv [--validator]

Replays each probe track in PTS order as tools/replay_plate_tracks.py does (2,500 ms gap
reset, 150 ms spacing), with the vote on. After every frame it records why the vote did
or did not run (`ctc_fusion.explain_vote`). Wherever fusion would run (3+ readings) and a
vote exists, it records `CTCPlateReader.vote_report()`: whether the voted plate differs
from the best reading without the vote, and whether it was added as a new candidate.

The CSV holds full plate text: keep it out of Git. The printed summary holds counts only.
"""
import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, '/workspace/edge')
from sentinel_edge.ctc_fusion import CTCPlateReader, explain_vote  # noqa: E402

CACHE = Path('/models/experiments/awiros')
CHECKPOINT = CACHE / 'adaptations/indian-corpus-encoder.safetensors'
GAP_MS = 2500


def explain_track(observations, tokens, *, validator=False):
    """observations: [(pts_ms, source_id, rows)] in PTS order -> one summary row."""
    reasons, outcomes = collections.Counter(), collections.Counter()
    most, fusion_steps, differs, examples = 0, 0, 0, []
    reader = None
    for pts, source, rows in observations:
        if reader is None or (reader.samples and pts - reader.samples[-1][0] > GAP_MS):
            reader = CTCPlateReader(tokens, position_vote=True, validate_format=validator)
        if reader.samples and pts - reader.samples[-1][0] < 150:
            continue
        reader.add(rows, timestamp_ms=pts, source_id=source)
        most = max(most, reader.observation_count)
        if reader.observation_count < 3:
            reasons['fewer_than_3_readings'] += 1   # Fusion does not run yet.
            continue
        fusion_steps += 1
        vote = explain_vote(list(reader._readings), [sample[4] for sample in reader.samples])
        reasons[vote['reason']] += 1
        if vote['reason'] != 'voted':
            continue
        report = reader.vote_report()
        outcomes[report['outcome']] += 1
        differs += report['differs_from_top']
        examples.append(f"{report['admitted'] or vote['voted']}|{report['top_without_vote'] or ''}"
                        f"|{report['outcome']}|lengths={json.dumps(report['lengths'])}")
    join = lambda counter: ';'.join(f'{key}:{value}' for key, value in sorted(counter.items()))
    return {'crops': len(observations), 'max_readings_in_window': most, 'fusion_steps': fusion_steps,
            'vote_reasons': join(reasons), 'vote_outcomes': join(outcomes),
            'steps_vote_differs_from_top': differs, 'examples_voted|top|outcome': ' / '.join(examples[-3:])}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('probe', type=Path, help='plate_crop_probe output: crops and report.json')
    parser.add_argument('output', type=Path, help='per-track CSV (contains plate text)')
    parser.add_argument('--validator', action='store_true', help='also turn the format validator on')
    args = parser.parse_args()
    from sentinel_edge.awiros_experiment import AwirosExperiment
    ocr = AwirosExperiment(CACHE, device='cpu', checkpoint=CHECKPOINT)
    tracks = {}
    for entry in json.loads((args.probe / 'report.json').read_text())['crops']:
        rows = ocr.probabilities(cv2.imread(str(args.probe / entry['file'])))
        tracks.setdefault(entry['track'], []).append((entry['pts_ms'], entry['file'], rows))
    rows, totals = [], collections.Counter()
    for track, observations in sorted(tracks.items(), key=lambda item: str(item[0]).zfill(8)):
        observations.sort(key=lambda item: item[0])
        row = {'track': track, **explain_track(observations, ocr.tokens, validator=args.validator)}
        rows.append(row)
        totals['tracks'] += 1
        totals['tracks_reaching_3_readings'] += row['max_readings_in_window'] >= 3
        totals['fusion_steps'] += row['fusion_steps']
        totals['steps_vote_differs_from_top'] += row['steps_vote_differs_from_top']
        for field in ('vote_reasons', 'vote_outcomes'):
            for item in filter(None, row[field].split(';')):
                key, value = item.split(':')
                totals[f'{field[5:]}:{key}'] += int(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    readings = collections.Counter(row['max_readings_in_window'] for row in rows)
    print(json.dumps({'totals': dict(sorted(totals.items())),
                      'tracks_by_max_readings_in_window': dict(sorted(readings.items()))}, indent=1))


if __name__ == '__main__':
    main()
