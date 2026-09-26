"""Static local summary of a matched OCR adaptation comparison."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path

from .benchmark_viewer import validate_viewer_scope


def render_summary(output, dataset, parent, tuned, stills, training):
    validate_viewer_scope(output, dataset)
    for field in ('manifest_sha256', 'backend', 'device', 'stage', 'quality_weighted',
                  'reconstruction_requested', 'restoration_requested', 'deskew_requested', 'family'):
        if parent[field] != tuned[field]:
            raise ValueError('Unmatched before/after benchmark configuration')
    if parent['further_fine_tuned_here'] or not tuned['further_fine_tuned_here']:
        raise ValueError('Explicit original and adapted reports required')
    if tuned['adapted_checkpoint_sha256'] != stills['checkpoint_sha256']:
        raise ValueError('Different adapted checkpoint used across reports')
    old, new = parent['rows'], tuned['rows']
    if len(old) != len(new) or len(old) > 100:
        raise ValueError('Mismatched or oversized condition list')
    esc = lambda value: html.escape(str(value), quote=True)
    comparisons = []
    for before, after in zip(old, new):
        if (before['condition'], before['passage_index']) != (after['condition'], after['passage_index']):
            raise ValueError('Conditions are not paired')
        comparisons.append('<tr>' + ''.join(f'<td>{esc(value)}</td>' for value in
                                           (before['condition'], before['outcome'], after['outcome'],
                                            before['truth_rank'], after['truth_rank'])) + '</tr>')
    metrics = ''.join('<tr>' + ''.join(f'<td>{esc(value)}</td>' for value in
                                      (name, f"{row['exact']}/{row['samples']}",
                                       f"{row['character_errors']}/{row['characters']}",
                                       f"{row['cer'] * 100:.2f}%",
                                       f"{row['candidate_recall_counts']['10']}/{row['samples']}")) + '</tr>'
                      for name, row in stills['models'].items())
    splits = training['split_counts']
    output.write_text(f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>Indian OCR fine-tuning comparison</title><style>
body{{font-family:system-ui;background:#101827;color:#e5edf8;max-width:1100px;margin:auto;padding:30px}}
a{{color:#84c7ff}}table{{border-collapse:collapse;width:100%;margin:20px 0}}th,td{{border:1px solid #405069;padding:10px;text-align:left}}
.note{{padding:16px;background:#26354c;border-radius:10px}}h2{{margin-top:35px}}</style></head><body>
<h1>Indian OCR: original versus fine-tuned</h1>
<p class="note">Experimental recognition-head adaptation only. Live worker unchanged. This pilot does not establish 90% real-camera accuracy or calibrated confidence.</p>
<p>{esc(sum(splits.values()))} visually reviewed Indian registrations: {esc(splits['train'])} train, {esc(splits['validation'])} validation, {esc(splits['test'])} test. Native detector crops preserve pixels; mild training augmentations do not count as independent video frames.
{esc(training['trainable_parameters'])} output-layer parameters trained on CPU; image encoder and all its buffers remained unchanged. Validation selected epoch {esc(training['selected_epoch'])}.</p>
<h2>Ten separate test registrations — still-image diagnostic</h2>
<table><thead><tr><th>Model</th><th>Whole plates exact</th><th>Character errors</th><th>Character error rate</th><th>True plate in top 10</th></tr></thead><tbody>{metrics}</tbody></table>
<p><a href="awiros-finetune-test.html">Inspect every test crop, before/after reading and candidate list</a>.</p>
<h2>Existing multi-frame development regression</h2>
<p>21 controlled conditions of ONE already-seen vehicle with three distinct source frames. Original: {esc(parent['correct'])} correct, {esc(parent['wrong'])} wrong, {esc(parent['abstain'])} abstained.
Fine-tuned: {esc(tuned['correct'])} correct, {esc(tuned['wrong'])} wrong, {esc(tuned['abstain'])} abstained.
Evidence gates and automatic crop selection are unchanged.</p>
<p><a href="awiros-parent-rerun-all.html">Original model multi-frame viewer</a> · <a href="awiros-finetuned-all.html">Fine-tuned multi-frame viewer</a></p>
<table><thead><tr><th>Condition</th><th>Original result</th><th>Tuned result</th><th>Original truth rank</th><th>Tuned truth rank</th></tr></thead><tbody>{''.join(comparisons)}</tbody></table>
<h2>Limits and provenance</h2><p>Test registrations were excluded from this adaptation and checkpoint selection. Overlap with the author's earlier pretraining is unknown.
No two-row/night/live-camera accuracy claim; no footage or credentials from Sentinel included. Concurrent CPU runs are not a controlled speed comparison.
Source: {esc(dataset['name'])}; {esc(dataset['license'])}; {esc(dataset['attribution'])}.</p>
<p>Checkpoint SHA256: {esc(stills['checkpoint_sha256'])}</p></body></html>''', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reports', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    read = lambda name: json.loads((args.reports / name).read_text())
    render_summary(args.reports / 'awiros-finetune-summary.html',
                   json.loads(args.manifest.read_text())['dataset'],
                   read('awiros-parent-rerun-all.json'), read('awiros-finetuned-all.json'),
                   read('awiros-finetune-test.json'), read('awiros-finetune.json'))
