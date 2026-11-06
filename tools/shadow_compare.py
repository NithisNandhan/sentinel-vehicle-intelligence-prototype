"""Compare a BASELINE and a CANDIDATE evaluation report from the same labels (shadow mode).

Both reports come from tools/evaluate_recorded.py on identical labels and
recordings (checked by labels hash). Shows confirmed/partial/unreadable counts,
safety counters, runtime and release gates side by side, and states whether
the candidate may be promoted. It never edits either report.
"""
import argparse
import json
from pathlib import Path


def summarise(report):
    states = report.get('states_totals') or {}
    totals = report['totals']
    runtime = [r.get('runtime', {}) for r in report.get('recordings', [])]
    return {
        'profile': (report.get('profile') or {}).get('profile', 'baseline'),
        'confirmed_correct': states.get('accepted', {}).get('correct', totals['accepted_readings']['correct']),
        'confirmed_wrong': states.get('accepted', {}).get('wrong', totals['accepted_readings']['wrong']),
        'coverage': states.get('coverage') or totals['exact_read_coverage'],
        'partial': states.get('partial'),
        'unreadable_rate': states.get('unreadable_rate'),
        'wall_seconds': round(sum(r.get('wall_seconds', 0) for r in runtime), 2),
        'video_seconds': round(sum(r.get('video_seconds') or 0 for r in runtime), 2),
        'commit': report.get('commit'),
        'release_gates_passed': (report.get('release_gates') or {}).get('passed'),
    }


def compare(baseline, candidate):
    if baseline['labels_sha256'] != candidate['labels_sha256']:
        raise ValueError('Reports use different labels; they are not comparable')
    a, b = summarise(baseline), summarise(candidate)
    blockers = []
    if b['confirmed_wrong']:
        blockers.append('candidate confirmed a wrong plate')
    if b['partial'] and b['partial'].get('with_wrong_known_character'):
        blockers.append('candidate partial result shows a wrong known character')
    if not candidate.get('states_totals'):
        blockers.append('candidate report has no result-state scoring (profile not enabled)')
    if b['release_gates_passed'] is not True:
        blockers.append('release gates did not pass on this report')
    return {'baseline': a, 'candidate': b, 'same_labels': True, 'promotion_allowed': not blockers, 'blockers': blockers,
            'note': 'Promotion also needs a first-opened sealed set, a tested rollback and the seeded-watchlist replay.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text())), indent=2))
