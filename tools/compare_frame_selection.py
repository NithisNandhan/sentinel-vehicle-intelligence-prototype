"""Fixed-policy frame selection on frozen native crops and original OCR rows.

Features are computed without labels or OCR confidence. This is a development
comparison, never a trainer, threshold search or automatic deployment.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import html
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'edge'))
from sentinel_edge.frame_selection import DEFAULT_SELECTION, assess_image, choose_frames
from sentinel_edge.plate_continuity import DEFAULT_POLICY, continuity_reason
from compare_temporal_approaches import replay_reader, sha, write_json

POLICIES = ('current_reconstruction', 'legacy_best', 'continuity_all', 'quality_best',
            'quality_top3_fusion', 'quality_assisted')


def trace_data(root):
    with gzip.open(root/'trace.json.gz', 'rt', encoding='utf-8') as source:
        trace = json.load(source)
    count = sum(len(s['observations']) for s in trace['segments'])
    if not trace['reached_eof'] or count != trace['pipeline_summary']['distinct_ocr_observations']:
        raise ValueError('A complete reconciled trace is required')
    return trace


def features(args):
    import cv2
    if args.output.exists():
        raise ValueError('Use a fresh feature folder')
    args.output.mkdir(parents=True)
    trace = trace_data(args.trace)
    assessments = {}
    for segment in trace['segments']:
        for row in segment['observations']:
            if row['source_id'] in assessments:
                continue
            path = args.trace/row['native_path']
            if sha(path) != row['native_sha256']:
                raise ValueError('Frozen evidence changed')
            image = cv2.imread(str(path))
            if image is None or hashlib.sha256(image.tobytes()).hexdigest() != row['source_id']:
                raise ValueError('Native pixels do not match the OCR source identity')
            assessments[row['source_id']] = assess_image(image)
    write_json(args.output/'features.json', dict(trace_sha256=sha(args.trace/'trace.json.gz'),
                feature_module_sha256=sha(Path(__file__).resolve().parents[1]/'edge/sentinel_edge/frame_selection.py'),
                policy=asdict(DEFAULT_SELECTION), labels_used=False, ocr_confidence_used=False,
                assessments=assessments))
    reasons = Counter(reason for a in assessments.values() for reason in a['rejection_reasons'])
    print(json.dumps(dict(stage='features_ready', frames=len(assessments),
                         eligible=sum(a['eligible'] for a in assessments.values()), rejected=dict(reasons))), flush=True)


def split_continuity(observations):
    groups, resets = [], []
    previous = None
    for observation in observations:
        reason = continuity_reason(previous['box'] if previous else None,
                                   previous['timestamp_ms'] if previous else None,
                                   observation['box'], observation['timestamp_ms'])
        if not groups or reason:
            groups.append([])
        if reason:
            resets.append(dict(position_ms=observation['position_ms'], reason=reason))
        groups[-1].append(observation)
        previous = observation
    return groups, resets


def reading(tokens, observations, validate_format=False, threshold=.85):
    if not observations:
        return dict(top=None, model_accepted=None, selected_positions_ms=[], observations=0)
    reader = replay_reader(tokens, observations, validate_format=validate_format)
    ranked = reader.candidates()
    accepted = reader.accepted(threshold)
    return dict(top=ranked[0].plate if ranked else None,
                model_accepted=accepted.plate if accepted else None,
                selected_positions_ms=[r['position_ms'] for r in observations],
                observations=len(observations))


def evaluate(tokens, original, clean, assessments, validate_format, threshold):
    last = original[-1]
    current = dict(top=last['current_candidates'][0]['plate'] if last['current_candidates'] else None,
                   model_accepted=last['current_accepted'], observations=len(original),
                   selected_positions_ms=[r['position_ms'] for r in original[-8:]])
    best = max(original, key=lambda r:(r['quality'], -r['position_ms']))
    selected_one = choose_frames(clean, assessments, limit=1)
    selected_three = choose_frames(clean, assessments, limit=3)
    one = reading(tokens, selected_one, validate_format, threshold)
    three = reading(tokens, selected_three, validate_format, threshold)
    assisted = dict(three if three['model_accepted'] else one)
    assisted['decision'] = 'multi_frame_gate_passed' if three['model_accepted'] else 'best_frame_hypothesis'
    return dict(current_reconstruction=current,
                legacy_best=reading(tokens, [best], validate_format, threshold),
                continuity_all=reading(tokens, clean, validate_format, threshold),
                quality_best=one, quality_top3_fusion=three, quality_assisted=assisted)


def score(args):
    if (args.output/'report.json').exists():
        raise ValueError('Preserve the previous comparison')
    trace = trace_data(args.trace)
    feature_path = args.output/'features.json'
    feature = json.loads(feature_path.read_bytes())
    raw = json.loads((args.trace/'report.json').read_bytes())
    if feature['trace_sha256'] != sha(args.trace/'trace.json.gz') or raw['trace_sha256'] != feature['trace_sha256']:
        raise ValueError('Features and labels must refer to the same trace')
    assessments = feature['assessments']
    by_id = {s['id']:s for s in trace['segments']}
    cache, cases = {}, []
    for reference in raw['cases']:
        row = dict(id=reference['id'], expected=reference['expected'],
                   position_ms=reference['position_ms'], association=reference['association'],
                   original_observations=reference['observations'], clean_observations=0,
                   methods={}, resets=[])
        if reference['observations']:
            segment = by_id[reference['association']['segment']]
            groups, resets = split_continuity(segment['observations'])
            # Follow the geometrically associated anchor into its own continuity
            # group; never choose the group by whether OCR matches the answer.
            clean = next(g for g in groups if any(abs(o['position_ms']-reference['position_ms'])<=20 for o in g))
            key = (segment['id'], clean[0]['timestamp_ms'])
            if key not in cache:
                cache[key] = evaluate(trace['tokens'], segment['observations'], clean, assessments,
                                      trace['options']['format_validator'], trace['threshold'])
            row.update(clean_observations=len(clean), methods={k:dict(v) for k,v in cache[key].items()}, resets=resets)
        for name in POLICIES:
            result = row['methods'].setdefault(name, dict(top=None, model_accepted=None, observations=0, selected_positions_ms=[]))
            result['exact'] = result['top'] == row['expected']
        cases.append(row)
        print(json.dumps(dict(stage='scoring', targets=len(cases))), flush=True)
    matched = [c for c in cases if c['original_observations']]
    multi = [c for c in matched if c['clean_observations']>=3]
    summary = {}
    baseline = 'current_reconstruction'
    for name in POLICIES:
        summary[name] = dict(exact=sum(c['methods'][name]['exact'] for c in cases), targets=len(cases),
            matched_exact=sum(c['methods'][name]['exact'] for c in matched), matched_targets=len(matched),
            clean_multi_exact=sum(c['methods'][name]['exact'] for c in multi), clean_multi_targets=len(multi),
            outputs=sum(bool(c['methods'][name]['top']) for c in cases),
            recovered=[c['id'] for c in cases if not c['methods'][baseline]['exact'] and c['methods'][name]['exact']],
            regressed=[c['id'] for c in cases if c['methods'][baseline]['exact'] and not c['methods'][name]['exact']],
            selected_at_reference=sum(c['position_ms'] in c['methods'][name]['selected_positions_ms'] for c in matched),
            final_gate_correct=sum(c['methods'][name]['model_accepted']==c['expected'] for c in cases),
            final_gate_wrong=sum(bool(c['methods'][name]['model_accepted']) and c['methods'][name]['model_accepted']!=c['expected'] for c in cases))
    best = max(s['matched_exact'] for s in summary.values())
    winners = [name for name,s in summary.items() if s['matched_exact']==best]
    report = dict(scope='Fixed-policy development diagnostic on previously inspected reference anchors; not full-video accuracy',
                  trace_sha256=feature['trace_sha256'], features_sha256=sha(feature_path),
                  original_report_sha256=sha(args.trace/'report.json'), models=trace['models'],
                  policy=feature['policy'], continuity_policy=asdict(DEFAULT_POLICY),
                  selector_sha256=sha(Path(__file__).resolve().parents[1]/'edge/sentinel_edge/frame_selection.py'),
                  continuity_sha256=sha(Path(__file__).resolve().parents[1]/'edge/sentinel_edge/plate_continuity.py'),
                  runner_sha256=sha(__file__), summary=summary, winners=winners, cases=cases,
                  settings_changed_in_production=False,
                  limitations=['Only admitted observations are present in frozen trace; missing/duplicate detector observations can affect live continuity, requiring full-pipeline validation.',
                               'Edge contact and tilt indicators do not prove a complete or readable plate.',
                               'Final gate metrics are endpoint diagnostics, not a count of all earlier operational acceptances.'])
    write_json(args.output/'report.json', report)
    render_page(args, report, assessments)
    print(json.dumps(dict(stage='completed', winners=winners, summary=summary)), flush=True)


def render_page(args, report, assessments):
    import shutil
    # The local viewer receives only the specific frozen native crops used here.
    assets = args.output/'assets'
    assets.mkdir(exist_ok=True)
    trace = trace_data(args.trace)
    observations = {o['position_ms']:o for s in trace['segments'] for o in s['observations']}
    segments = {s['id']:s for s in trace['segments']}
    sections = []
    for case in report['cases']:
        if not case['original_observations']:
            continue
        frames = []
        for row in segments[case['association']['segment']]['observations']:
            shutil.copyfile(args.trace/row['native_path'], args.output/row['native_path'])
            assessment = assessments[row['source_id']]
            frames.append('<figure><figcaption>'+str(row['position_ms']/1000)+' s · score '+str(assessment['score'])+
                ' · '+('eligible' if assessment['eligible'] else ', '.join(assessment['rejection_reasons']))+
                '</figcaption><img src="'+html.escape(row['native_path'])+'" alt="Original plate crop"><small>'+html.escape(', '.join(assessment['warnings']))+'</small></figure>')
        methods = ''.join('<tr><td>'+html.escape(name)+'</td><td>'+html.escape(result['top'] or 'No reading')+
                          '</td><td>'+('Correct' if result['exact'] else 'Incorrect / missing')+'</td><td>'+html.escape(str(result['selected_positions_ms']))+'</td></tr>' for name,result in case['methods'].items())
        sections.append('<section><h2>Reference '+html.escape(case['id'])+' · human label '+html.escape(case['expected'])+
                        '</h2><p>Continuity resets: '+html.escape(str(case['resets']))+'</p><div class="frames">'+''.join(frames)+
                        '</div><div class="scroll"><table><tr><th>Rule</th><th>Hypothesis</th><th>Exact</th><th>Selected times (ms)</th></tr>'+methods+'</table></div></section>')
    summary = ''.join('<tr><td>'+html.escape(name)+'</td><td>'+str(s['matched_exact'])+'/'+str(s['matched_targets'])+'</td><td>'+str(s['clean_multi_exact'])+'/'+str(s['clean_multi_targets'])+'</td></tr>' for name,s in report['summary'].items())
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sentinel · Frame selection</title><style>body{font:16px system-ui;background:#f6f7f2;color:#17322d;max-width:1300px;margin:30px auto;padding:0 20px}h1,h2{color:#0f766e}section{background:white;border:1px solid #ccd9d0;border-radius:12px;padding:20px;margin:20px 0}table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #ccd9d0;text-align:left}.frames{display:flex;gap:15px;flex-wrap:wrap;margin:20px 0}figure{margin:0;padding:12px;border:1px solid #ccd9d0}img{display:block;max-width:300px;max-height:130px;margin:12px 0}.scroll{overflow:auto}small{display:block}</style><h1>Choose evidence before combining readings</h1><p>'+html.escape(report['scope'])+'</p><p>Rules use native image properties and geometric continuity, not the answer or OCR confidence. A displayed hypothesis is unverified; image-quality scores are not correctness probabilities.</p><div class="scroll"><table><tr><th>Rule</th><th>Matched exact</th><th>Clean ≥3-frame exact</th></tr>'+summary+'</table></div>'+''.join(sections)+'</html>'
    (args.output/'index.html').write_bytes(page.encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('features','score'), required=True)
    parser.add_argument('--trace', type=Path, default=Path('data/benchmarks/temporal-approaches-20261110'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    features(args) if args.phase=='features' else score(args)
