"""Score plate readings for single crops against draft or verified labels.

Labels CSV: `crop_id,label,...` where label is a plate (capital letters and digits),
`NO_PLATE` (not a plate) or `UNREADABLE` (a person cannot be sure of the full text).
Predictions CSV: `crop_id,<variant>,<variant>...`; an empty cell means the pipeline
produced no reading (rejected or abstained).

    python tools/eval_plates.py --labels labels.csv --predictions predictions.csv

Every rate is reported with its denominator and is `null` when the denominator is 0.
Scores describe this labelled sample only; draft labels are not ground truth.
"""
import argparse
import csv
import json
import re

NO_PLATE, UNREADABLE = 'NO_PLATE', 'UNREADABLE'
_PLATE = re.compile(r'^[A-Z0-9]{1,20}$')


def normalise(text):
    return re.sub(r'[^A-Z0-9]', '', (text or '').upper())


def edit_distance(first, second):
    previous = list(range(len(second) + 1))
    for i, a in enumerate(first, 1):
        current = [i]
        for j, b in enumerate(second, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def load_labels(path):
    labels = {}
    with open(path, newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            crop, label = row['crop_id'].strip(), row['label'].strip()
            if label not in (NO_PLATE, UNREADABLE) and not _PLATE.fullmatch(label):
                raise ValueError(f'{crop}: label must be a plate, {NO_PLATE} or {UNREADABLE}')
            if crop in labels:
                raise ValueError(f'{crop}: duplicate label')
            labels[crop] = label
    return labels


def load_predictions(path):
    with open(path, newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        variants = [name for name in reader.fieldnames if name != 'crop_id']
        rows = {row['crop_id'].strip(): row for row in reader}
    return variants, {variant: {crop: normalise(row.get(variant)) for crop, row in rows.items()}
                      for variant in variants}


def _rate(numerator, denominator):
    return round(numerator / denominator, 4) if denominator else None


def score(labels, predictions):
    """Metrics for one variant: {crop_id: reading or ''}."""
    missing = sorted(set(labels) - set(predictions))
    if missing:
        raise ValueError(f'No prediction row for: {", ".join(missing)}')
    readable = {crop: label for crop, label in labels.items() if label not in (NO_PLATE, UNREADABLE)}
    no_plate = [crop for crop, label in labels.items() if label == NO_PLATE]
    unreadable = [crop for crop, label in labels.items() if label == UNREADABLE]
    exact = sum(predictions[crop] == label for crop, label in readable.items())
    errors = sum(edit_distance(predictions[crop], label) for crop, label in readable.items())
    characters = sum(len(label) for label in readable.values())
    false_reads = sum(bool(predictions[crop]) for crop in no_plate)
    return {
        'readable_n': len(readable), 'exact_matches': exact, 'exact_match_rate': _rate(exact, len(readable)),
        'abstained_on_readable': sum(not predictions[crop] for crop in readable),
        'character_errors': errors, 'label_characters': characters,
        'character_error_rate': _rate(errors, characters),
        'no_plate_n': len(no_plate), 'false_reads_on_no_plate': false_reads,
        'false_read_rate': _rate(false_reads, len(no_plate)),
        'unreadable_n': len(unreadable), 'readings_on_unreadable': sum(bool(predictions[crop]) for crop in unreadable),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--labels', required=True)
    parser.add_argument('--predictions', required=True)
    args = parser.parse_args(argv)
    labels = load_labels(args.labels)
    variants, predictions = load_predictions(args.predictions)
    report = {variant: score(labels, predictions[variant]) for variant in variants}
    print(json.dumps(report, indent=1))
    return report


if __name__ == '__main__':
    main()
