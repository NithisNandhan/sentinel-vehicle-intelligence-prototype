"""Annotation-set validation: schema, splits, duplicates, sealed-set rules, agreement, manifest.

Works on the recorded label format (recorded_evaluation.load_labels, schema 1.1).
Nothing here creates or changes a label: it only reports problems so that a
person can fix them. See docs/ANNOTATION_GUIDE.md.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .recorded_evaluation import SEALED, SPLITS, LabelError, load_labels, normalise

SET_A_MINIMUM_READABLE = 50
SEALED_MINIMUM_RECORDINGS = 2
DEVELOPMENT_MINIMUM_RECORDINGS = 2
SECOND_CHECK_TARGET = .20


def validate(path):
    """Return a report dict; raises LabelError only when the file cannot be parsed at all."""
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    report = {'labels_sha256': hashlib.sha256(raw).hexdigest(), 'schema_version': data.get('schema_version'),
              'errors': [], 'warnings': [], 'splits': {}}
    loaded = {}
    for split in SPLITS:
        try:
            selected, _ = load_labels(path, split=split)
        except LabelError as error:
            report['errors'].append(str(error))
            return report
        loaded[split] = selected
    plates_by_split = defaultdict(set)
    for split, recordings in loaded.items():
        passages = [p for entry in recordings.values() for p in entry['passages']]
        if not recordings:
            continue
        readable = [p for p in passages if p.readable]
        report['splits'][split] = {
            'recordings': len(recordings), 'passages': len(passages), 'readable': len(readable),
            'unreadable': len(passages) - len(readable),
            'two_row_readable': sum(p.layout == 'two_row' for p in readable),
            'layout_unlabelled': sum(p.layout is None for p in readable),
            'conditions': dict(Counter(c for p in readable for c in p.conditions)),
        }
        for p in readable:
            plates_by_split[split].add(p.plate)
        # Duplicate passages: the same plate twice in one recording with overlapping time.
        for rid, entry in recordings.items():
            seen = defaultdict(list)
            for p in entry['passages']:
                if p.readable:
                    for other in seen[p.plate]:
                        if p.start_ms <= other.end_ms and other.start_ms <= p.end_ms:
                            report['errors'].append(f'{rid}: passages {other.passage_id} and {p.passage_id} overlap with the same plate (duplicate passage)')
                    seen[p.plate].append(p)
        if split in SEALED:
            if len(recordings) < SEALED_MINIMUM_RECORDINGS:
                report['errors'].append(f'{split}: a sealed set needs at least {SEALED_MINIMUM_RECORDINGS} recordings')
            if split == 'sealed_a' and len(readable) < SET_A_MINIMUM_READABLE:
                report['errors'].append(f'sealed_a: {len(readable)} readable passages; at least {SET_A_MINIMUM_READABLE} are required')
    if 'development' in report['splits'] and report['splits']['development']['recordings'] < DEVELOPMENT_MINIMUM_RECORDINGS:
        report['warnings'].append('Keep at least two recordings for development.')
    if 'development' not in report['splits']:
        report['warnings'].append('No development split is labelled.')
    # The same registration in two splits leaks between tuning and testing.
    splits = list(plates_by_split)
    for i, first in enumerate(splits):
        for second in splits[i + 1:]:
            shared = plates_by_split[first] & plates_by_split[second]
            if shared:
                report['errors'].append(f'{len(shared)} registration(s) appear in both {first} and {second}')
    # Recording files reused across splits.
    owners = defaultdict(set)
    for entry in data.get('recordings', []):
        owners[entry.get('sha256')].add(entry.get('split'))
    for sha, owner in owners.items():
        if len(owner) > 1:
            report['errors'].append(f'recording {str(sha)[:12]} is labelled in several splits: {sorted(owner)}')
    report['agreement'] = agreement(data)
    if report['agreement']['checked_passages'] < SECOND_CHECK_TARGET * report['agreement']['eligible_passages']:
        report['warnings'].append('Fewer than 20% of passages have an independent second readability check.')
    report['valid'] = not report['errors']
    return report


def agreement(data):
    """Second-annotator readability agreement and disagreements needing adjudication."""
    eligible = checked = agreed = 0
    unresolved = []
    for entry in data.get('recordings', []):
        for item in entry.get('passages') or []:
            eligible += 1
            second = item.get('second_check')
            if not isinstance(second, dict):
                continue
            checked += 1
            same_readability = second.get('readable') == item.get('readable')
            same_text = (not item.get('readable') or second.get('plate') is None
                         or normalise(second.get('plate') or '') == normalise(item.get('plate') or ''))
            if same_readability and same_text:
                agreed += 1
            elif not second.get('adjudicated_by'):
                unresolved.append(f"{entry.get('recording_id')}/{item.get('passage_id')}")
    return {'eligible_passages': eligible, 'checked_passages': checked, 'agreed': agreed,
            'agreement_rate': round(agreed / checked, 4) if checked else None,
            'unresolved_disagreements': unresolved}


def manifest(path, recordings_dir=None):
    """Hashes that pin an evaluation: label file, and each labelled recording if present."""
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    files = []
    for entry in data.get('recordings', []):
        row = {'recording_id': entry.get('recording_id'), 'split': entry.get('split'), 'sha256': entry.get('sha256'),
               'label_source': entry.get('label_source', 'blind_human')}
        files.append(row)
    body = {'labels_sha256': hashlib.sha256(raw).hexdigest(), 'recordings': sorted(files, key=lambda r: str(r['recording_id']))}
    body['manifest_sha256'] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body


class SealedLedger:
    """Append-only record of sealed-set openings. An opened set becomes development data."""
    def __init__(self, path):
        self.path = Path(path)

    def entries(self):
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding='utf-8'))['openings']

    def opened(self, split, labels_sha256):
        return [e for e in self.entries() if e['split'] == split]

    def record(self, split, labels_sha256, *, commit=None, profile=None):
        if split not in SEALED:
            raise ValueError('Only sealed splits are recorded')
        previous = self.opened(split, labels_sha256)
        entries = self.entries() + [{'split': split, 'labels_sha256': labels_sha256, 'commit': commit,
                                     'profile': profile, 'opened_at': datetime.now(timezone.utc).isoformat(),
                                     'opening_number': len(previous) + 1}]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps({'openings': entries}, indent=2), encoding='utf-8')
        temporary.replace(self.path)
        return bool(previous)
