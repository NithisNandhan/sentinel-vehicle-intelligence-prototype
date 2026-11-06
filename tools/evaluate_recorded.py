"""Measure recognition on labelled recorded footage with the unchanged models.

Runs inside the CPU engineering container (see docs/RECORDED_VIDEO.md). The
report contains counts with denominators and masked plates only. The `test`
split is for one final run after settings are frozen and needs --final.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'edge'))


def peak_memory_mb():
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 1)  # Linux reports KiB
    except ImportError:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--split', choices=('development', 'validation', 'test', 'sealed_a', 'sealed_b', 'sealed_c'), required=True)
    parser.add_argument('--final', action='store_true', help='Required for a sealed split: scored after freezing settings')
    parser.add_argument('--ledger', type=Path, default=Path('data/evaluation/sealed-ledger.json'),
                        help='Append-only record of sealed-set openings (an opened set becomes development data)')
    parser.add_argument('--allow-reopen', action='store_true',
                        help='Re-run an already opened sealed set as DEVELOPMENT data; the report says it is no longer unseen')
    parser.add_argument('--recordings-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('/models/experiments/awiros'))
    parser.add_argument('--checkpoint', type=Path,
                        default=Path('/models/experiments/awiros/adaptations/indian-corpus-encoder.safetensors'))
    parser.add_argument('--parent-ocr', action='store_true', help='Use the parent OCR weights instead of the adapted checkpoint')
    parser.add_argument('--detector-profile', choices=('legacy', 'general', 'plate_adapted', 'indian_adapted'), default='general')
    parser.add_argument('--training-root', type=Path, default=Path('/training'))
    parser.add_argument('--device', choices=('cpu', 'gpu'), default=os.getenv('SENTINEL_OCR_DEVICE', 'gpu'))
    parser.add_argument('--plate-recovery', action='store_true')
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--commit', default=os.getenv('SENTINEL_COMMIT', ''), help='Source commit being evaluated')
    args = parser.parse_args()
    sealed = args.split in ('test', 'sealed_a', 'sealed_b', 'sealed_c')
    if sealed and not args.final:
        parser.error('A sealed split is scored once, after freezing. Pass --final to confirm this is that run.')
    from sentinel_edge.annotations import SealedLedger
    ledger = SealedLedger(args.ledger)
    already_opened = sealed and bool(ledger.opened(args.split, ''))
    if already_opened and not args.allow_reopen:
        parser.error(f'{args.split} was already opened. It is development data now; pass --allow-reopen to re-run it as such.')
    if args.output.exists():
        parser.error('The output file already exists; reports are never overwritten.')
    if not 1 <= args.top_k <= 200:
        parser.error('--top-k must be between 1 and 200')

    from sentinel_edge.config import Config
    from sentinel_edge.live_detection_viewer import load_models, model_identity
    from sentinel_edge.recorded_evaluation import LabelError, evaluate, load_labels
    try:
        load_labels(args.labels, split=args.split)  # Fail on bad labels before loading any model.
    except LabelError as error:
        raise SystemExit(f'Labels rejected: {error}')
    config = Config.from_env(live=False)
    checkpoint = None if args.parent_ocr else args.checkpoint
    models = load_models(config, args.cache, checkpoint, detector_profile=args.detector_profile,
                         training_root=args.training_root, device=args.device)
    try:
        report = evaluate(args.labels, args.split, args.recordings_dir, config, models,
                          detector_profile=args.detector_profile, top_k=args.top_k,
                          plate_recovery=args.plate_recovery)
    except LabelError as error:
        raise SystemExit(f'Evaluation stopped: {error}')
    from sentinel_edge.annotations import manifest as label_manifest
    from sentinel_edge.recorded_evaluation import release_gates
    if sealed:
        ledger.record(args.split, report['labels_sha256'], commit=args.commit or None, profile=report.get('profile'))
    report['manifest'] = label_manifest(args.labels)
    report['sealed'] = {'split_is_sealed': sealed, 'previously_opened': already_opened,
                        'status': ('sealed, first opening' if sealed and not already_opened else
                                   'previously opened: development data, not unseen' if sealed else 'development/validation')}
    if report.get('states_totals'):
        report['release_gates'] = release_gates(report['states_totals'], split=args.split, previously_opened=already_opened)
    report.update(generated_at=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                  commit=args.commit or None, models=model_identity(models[0], args.detector_profile,
                                                                  detector=models[1], vehicle_model=models[2]),
                  environment={'platform': platform.platform(), 'machine': platform.machine(),
                               'processor': platform.processor() or None, 'cpu_count': os.cpu_count(),
                               'python': platform.python_version(),
                               'cpu_threads': os.getenv('OMP_NUM_THREADS'), 'peak_memory_mb': peak_memory_mb()},
                  notice='Scores are model outputs judged against human labels on this footage only. '
                         'OCR ranking scores are not calibrated probabilities.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    totals = report['totals']
    print(json.dumps({key: totals[key] for key in ('passages', 'exact_read_coverage', 'wrong_reads', 'abstentions',
                                                   'accepted_precision', 'candidate_recall')}, indent=2))
    if report.get('states_totals'):
        states = report['states_totals']
        print(json.dumps({'coverage': states['coverage'], 'accepted': states['accepted'], 'partial': states['partial'],
                          'unreadable_rate': states['unreadable_rate'],
                          'release_gates_passed': report.get('release_gates', {}).get('passed')}, indent=2))
    print(f'Report written: {args.output}')
