"""Paired fixed-input diagnostic of single-frame, fusion and reconstruction.

Capture uses the unchanged GPU pipeline, without reading human transcriptions.
Scoring reuses its exact OCR posteriors and per-frame beams. No training or
deployment is performed. Detailed evidence stays in the ignored output folder.
"""
import argparse
import ast
import gzip
import hashlib
import html
import json
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'edge'))
from sentinel_edge.ctc_fusion import CTCPlateReader, validate_ctc
from run_video_review import CountedRun, utc, write_json

DEPLOYED = 'c2ba896d994b174071c95ace0f5ffc8a61cec3fc6eecce8f0bc5feb7dec4456d'
METHODS = ('single_quality', 'multi_originals', 'multi_reconstruction')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate(value):
    return dict(plate=value.plate, score=value.score,
                minimum_character_score=value.minimum_character_score,
                log_score=value.log_score)


class TraceObserver:
    def __init__(self, output, source):
        self.output, self.source = output, source
        self.readers, self.segments = {}, []
        self.samples = 0

    def publish(self, snapshot, stop=None):
        pass

    def on_sample(self, frame, tracks, displays, pipeline):
        import cv2
        from sentinel_edge.live_reconstruction import disagreement
        position = frame.pts_ms-self.source.start_pts_ms
        for display in displays:
            state = pipeline.histories.get(display['track_id'])
            if state is None or not state.reader.samples:
                continue
            reader = state.reader
            digest = hashlib.sha256(display['crop'].tobytes()).hexdigest()
            latest = reader.samples[-1]
            if latest[0] != frame.pts_ms or latest[1] != digest:
                continue  # No new original observation on this display.
            if reader not in self.readers:
                segment = dict(id=f'ocr-{len(self.segments)+1:04}',
                               track_id=str(display['track_id']), observations=[])
                self.readers[reader] = segment
                self.segments.append(segment)
            segment = self.readers[reader]
            if any(o['source_id'] == digest and o['timestamp_ms'] == latest[0]
                   for o in segment['observations']):
                continue
            relative = f"assets/{segment['id']}-{len(segment['observations']):02}.png"
            ok, encoded = cv2.imencode('.png', display['crop'])
            if not ok:
                raise ValueError('Cannot preserve native evidence')
            (self.output/relative).write_bytes(encoded.tobytes())
            ranked = reader.candidates()
            accepted = reader.accepted(pipeline.threshold)
            vetoed = disagreement(state.reconstruction, accepted, pipeline.threshold)
            if vetoed:
                accepted = None
            accepted = pipeline._gate(state, accepted)
            reconstruction = state.reconstruction
            segment['observations'].append(dict(
                position_ms=position, timestamp_ms=latest[0], source_id=digest,
                quality=latest[4], rows=latest[2], frame_candidates=latest[3],
                box=list(display['box']), native_path=relative,
                native_sha256=sha(self.output/relative),
                current_candidates=[candidate(c) for c in ranked[:20]],
                current_accepted=accepted.plate if accepted else None,
                reconstruction_status=getattr(reconstruction, 'status', 'disabled'),
                reconstruction_veto=bool(vetoed),
                reconstruction_proposals=list(reader._proposals)))
        self.samples += 1
        if self.samples % 10 == 0:
            progress = dict(stage='capture', position_ms=position,
                            analysed_frames=self.samples, ocr_segments=len(self.segments))
            write_json(self.output/'progress.json', progress)
            print(json.dumps(progress), flush=True)


def capture(args):
    from sentinel_edge.awiros_experiment import file_hash
    from sentinel_edge.config import Config
    from sentinel_edge.live_detection_viewer import load_models, model_identity
    from sentinel_edge.plate_options import PlateOptions
    from sentinel_edge.recorded_source import RecordingLibrary
    from sentinel_edge.tracked_live_viewer import observe_tracked
    if args.output.exists():
        raise ValueError('Use a fresh capture folder; earlier evidence is immutable')
    args.output.mkdir(parents=True)
    (args.output/'assets').mkdir()
    write_json(args.output/'progress.json', dict(stage='loading_models', started_at=utc()))
    recordings, _ = RecordingLibrary(args.recordings_dir).scan()
    source = recordings.get(args.recording_id)
    if source is None:
        raise ValueError('A permitted recording is required')
    options = PlateOptions.from_env()
    if options.position_vote or options.result_states:
        raise ValueError('This comparison requires the deployed baseline decoder/gate')
    checkpoint = args.cache/'adaptations/indian-corpus-encoder.safetensors'
    if file_hash(checkpoint) != DEPLOYED:
        raise ValueError('Deployed checkpoint identity changed')
    config = Config.from_env(live=False)
    import os
    keep = os.getenv('SENTINEL_KEEP_UNCONFIRMED_VEHICLES', 'false').lower() == 'true'
    reader, detector, vehicle = load_models(config, args.cache, checkpoint,
                                           detector_profile='general', device='gpu', keep_unconfirmed=keep)
    observer = TraceObserver(args.output, source)
    counted, stop = CountedRun(source), threading.Event()
    began = time.monotonic()
    summary = observe_tracked(config, source, reader, detector, vehicle, observer, stop,
                              recorded=counted, detector_profile='general', on_sample=observer.on_sample)
    if not counted.reached_eof or not counted.decoded:
        raise ValueError('Capture did not complete the source')
    if sum(len(s['observations']) for s in observer.segments) != summary['distinct_ocr_observations']:
        raise ValueError('Trace omitted an admitted original observation')
    accepts = {o['current_accepted'] for s in observer.segments for o in s['observations'] if o['current_accepted']}
    if len(accepts) != summary['unique_model_accepted_unverified']:
        raise ValueError('Captured acceptance decisions disagree with the pipeline summary')
    document = dict(schema_version=1, source_sha256=source.sha256, source_name=source.path.name,
                    models=model_identity(reader, 'general', detector=detector, vehicle_model=vehicle),
                    tokens=reader.tokens, options=options.manifest(), threshold=config.ocr_min_confidence,
                    reached_eof=True, decoded_frames=counted.decoded, analysed_frames=observer.samples,
                    wall_seconds=round(time.monotonic()-began, 2), pipeline_summary=summary,
                    runner_sha256=sha(__file__),
                    decoder_sha256=sha(Path(__file__).resolve().parents[1]/'edge/sentinel_edge/ctc_fusion.py'),
                    labels_used_during_capture=False, segments=observer.segments)
    with gzip.open(args.output/'trace.json.gz', 'wt', encoding='utf-8') as target:
        json.dump(document, target, allow_nan=False)
    write_json(args.output/'progress.json', dict(stage='captured', segments=len(observer.segments),
                original_observations=sum(len(s['observations']) for s in observer.segments),
                reached_eof=True, trace_sha256=sha(args.output/'trace.json.gz')))
    print(json.dumps(dict(stage='captured', segments=len(observer.segments),
                         decoded_frames=counted.decoded)), flush=True)


def replay_reader(tokens, observations, *, validate_format=False):
    """Restore exact cached original-frame beams; same CTC scoring implementation."""
    reader = CTCPlateReader(tokens, validate_format=validate_format)
    for observation in observations[-8:]:
        validate_ctc(observation['rows'], tokens)
        reader.samples.append((observation['timestamp_ms'], observation['source_id'],
                               observation['rows'], observation['frame_candidates'], observation['quality']))
    return reader


def evaluate_segment(tokens, segment, threshold, validate_format=False):
    observations = segment['observations']
    chosen = max(observations, key=lambda o: (o['quality'], -o['position_ms'], o['source_id']))
    single = replay_reader(tokens, [chosen], validate_format=validate_format)
    ranked_single = single.candidates()
    native = replay_reader(tokens, observations, validate_format=validate_format)
    ranked_native = native.candidates()
    original_accepts = set()
    for end in range(3, len(observations)+1):
        window = replay_reader(tokens, observations[max(0, end-8):end], validate_format=validate_format)
        accepted = window.accepted(threshold)
        if accepted:
            original_accepts.add(accepted.plate)
    last = observations[-1]
    return dict(
        single_quality=dict(top=ranked_single[0].plate if ranked_single else None,
                            accepted=None, selected_position_ms=chosen['position_ms']),
        multi_originals=dict(top=ranked_native[0].plate if ranked_native else None,
                             accepted=sorted(original_accepts)),
        multi_reconstruction=dict(top=last['current_candidates'][0]['plate'] if last['current_candidates'] else None,
                                  accepted=sorted({o['current_accepted'] for o in observations if o['current_accepted']})))


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0]))*max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union > 0 else 0.


def reference_box(sample, corners):
    points = corners.get(sample['id'])
    if not points:
        return sample['box']
    x, y = sample['box'][:2]
    return [x+min(p[0] for p in points), y+min(p[1] for p in points),
            x+max(p[0] for p in points), y+max(p[1] for p in points)]


def match_reference(sample, segments, corners, *, minimum_iou=.3):
    reference = reference_box(sample, corners)
    choices = [(box_iou(reference, observation['box']), segment['id'], observation)
               for segment in segments for observation in segment['observations']
               if abs(observation['position_ms']-sample['position_ms']) <= 20]
    choices.sort(key=lambda row: (-row[0], row[1]))
    if not choices or choices[0][0] < minimum_iou:
        return None
    # Do not silently assign an overlapping, equally plausible alternative.
    if len(choices) > 1 and choices[1][0] >= minimum_iou and choices[0][0]-choices[1][0] < .05:
        return dict(ambiguous=True, best_iou=choices[0][0])
    overlap, segment, observation = choices[0]
    return dict(ambiguous=False, iou=overlap, segment=segment,
                anchor_native_path=observation['native_path'])


def quarantine_conflicts(cases):
    """One uninterrupted evidence segment cannot have two distinct human truths."""
    truths = {}
    for row in cases:
        association = row['association']
        if association and not association.get('ambiguous'):
            truths.setdefault(association['segment'], set()).add(row['expected'])
    conflicts = {key for key, values in truths.items() if len(values) > 1}
    for row in cases:
        association = row['association']
        if association and association.get('segment') in conflicts:
            row['association'] = {**association, 'ambiguous': True, 'conflicting_human_truths': True}
            row['observations'] = 0
            row['methods'] = {method: dict(top=None, exact=False,
                             accepted=None if method == 'single_quality' else []) for method in METHODS}
    return sorted(conflicts)


def score(args):
    if (args.output/'report.json').exists():
        raise ValueError('Preserve the completed comparison')
    with gzip.open(args.output/'trace.json.gz', 'rt', encoding='utf-8') as source:
        trace = json.load(source)
    if not trace['reached_eof'] or trace['labels_used_during_capture']:
        raise ValueError('Invalid frozen inference trace')
    captured = sum(len(s['observations']) for s in trace['segments'])
    if captured != trace['pipeline_summary']['distinct_ocr_observations']:
        raise ValueError('Captured observations do not reconcile with the pipeline')
    for segment in trace['segments']:
        for observation in segment['observations']:
            if sha(args.output/observation['native_path']) != observation['native_sha256']:
                raise ValueError('An original evidence image changed')
    decoder = Path(__file__).resolve().parents[1]/'edge/sentinel_edge/ctc_fusion.py'
    if sha(decoder) != trace['decoder_sha256']:
        raise ValueError('Decoder changed since capture')
    manifest_path, labels_path = args.review/'manifest.json', args.review/'labels.json'
    manifest = json.loads(manifest_path.read_bytes())
    labels = json.loads(labels_path.read_bytes())
    if (manifest['source_sha256'] != trace['source_sha256'] or labels['source_sha256'] != trace['source_sha256']
            or labels['review_manifest_sha256'] != sha(manifest_path)):
        raise ValueError('Human labels are not bound to this recording and manifest')
    corners = {}
    for node in ast.parse((args.review/'ocr-model-dev/rectify.py').read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('CORNERS', 'TEXT_REGIONS') for t in node.targets):
            corners.update(ast.literal_eval(node.value))
    truth = {row['id']: row for row in labels['labels']}
    cases, cache = [], {}
    segments = {segment['id']: segment for segment in trace['segments']}
    for sample in manifest['samples']:
        label = truth[sample['id']]
        if label['status'] != 'readable' or sample['id'] == '21':
            continue
        if sha(args.review/sample['crop']) != sample['crop_sha256']:
            raise ValueError('A human reference image changed')
        association = match_reference(sample, trace['segments'], corners)
        row = dict(id=sample['id'], expected=label['plate'], position_ms=sample['position_ms'],
                   association=association, observations=0, methods={})
        if association and not association.get('ambiguous'):
            key = association['segment']
            segment = segments[key]
            if key not in cache:
                cache[key] = evaluate_segment(trace['tokens'], segment, trace['threshold'], trace['options']['format_validator'])
            row.update(observations=len(segment['observations']),
                       methods={name: dict(value) for name, value in cache[key].items()})
        for method in METHODS:
            result = dict(row['methods'].get(method, dict(top=None, accepted=None if method == 'single_quality' else [])))
            result['exact'] = result['top'] == row['expected']
            row['methods'][method] = result
        cases.append(row)
        print(json.dumps(dict(stage='scoring', reviewed=len(cases), matched=len(cache))), flush=True)
    conflicts = quarantine_conflicts(cases)
    matched = [row for row in cases if row['observations']]
    genuine_multi = [row for row in matched if row['observations'] >= 3]
    metrics = {}
    for method in METHODS:
        values = [row['methods'][method] for row in cases]
        metrics[method] = dict(exact=sum(v['exact'] for v in values), targets=len(cases),
                               matched_exact=sum(row['methods'][method]['exact'] for row in matched), matched_targets=len(matched),
                               multi_frame_exact=sum(row['methods'][method]['exact'] for row in genuine_multi), multi_frame_targets=len(genuine_multi),
                               outputs=sum(bool(v['top']) for v in values),
                               accepted_correct=None if method == 'single_quality' else sum(row['expected'] in row['methods'][method]['accepted'] for row in cases),
                               accepted_wrong=None if method == 'single_quality' else sum(sum(p != row['expected'] for p in row['methods'][method]['accepted']) for row in cases))
    pairs = {}
    for first, second in ((METHODS[0], METHODS[1]), (METHODS[1], METHODS[2])):
        pairs[first+'__'+second] = dict(recovered=[r['id'] for r in cases if not r['methods'][first]['exact'] and r['methods'][second]['exact']],
                                       regressed=[r['id'] for r in cases if r['methods'][first]['exact'] and not r['methods'][second]['exact']])
    wins = max((m['exact'], m['matched_exact']) for m in metrics.values())
    winners = [name for name, m in metrics.items() if (m['exact'], m['matched_exact']) == wins]
    report = dict(scope='Selected previously reviewed Mumbai reference targets; not whole-video or unseen-camera accuracy.',
                  source_sha256=trace['source_sha256'], trace_sha256=sha(args.output/'trace.json.gz'),
                  capture_runner_sha256=trace['runner_sha256'], scorer_sha256=sha(__file__),
                  observations_reconciled_with_pipeline=captured,
                  labels_sha256=sha(labels_path), review_manifest_sha256=sha(manifest_path),
                  geometry_sha256=sha(args.review/'ocr-model-dev/rectify.py'), models=trace['models'],
                  capture={k: trace[k] for k in ('decoded_frames','analysed_frames','wall_seconds','options','pipeline_summary')},
                  protocol=dict(association='Same PTS within 20ms; bounding box of prior manual polygon where available; IoU >= 0.30; ambiguous overlaps rejected',
                                disputed_ids_excluded=['21'], single_frame_selection='Maximum image quality within the same OCR-reader segment; no transcription/confidence selection',
                                multi_frame_window=8, reconstruction='Captured operational ECC proposals and veto; no extra independent observation',
                                comparison_unit='Labelled reference target linked to one uninterrupted OCR-reader instance; not assumed unique vehicles',
                                single_frame_acceptance='Not applicable: one image cannot satisfy the unchanged multi-frame confirmation gate'),
                  summary=metrics, paired=pairs, winners=winners, matched_targets=len(matched),
                  unmatched_targets=len(cases)-len(matched), genuine_multi_frame_targets=len(genuine_multi),
                  conflicting_segments=conflicts,
                  distinct_matched_segments=len({r['association']['segment'] for r in matched}), cases=cases)
    write_json(args.output/'report.json', report)
    make_page(args.output, report)
    write_json(args.output/'progress.json', dict(stage='completed', winners=winners, summary=metrics))
    print(json.dumps(dict(stage='completed', summary=metrics, winners=winners)), flush=True)


def make_page(output, report):
    names = dict(single_quality='One quality-selected frame', multi_originals='Multi-frame originals', multi_reconstruction='Multi-frame + reconstruction')
    rows = []
    for row in report['cases']:
        picture = row['association'].get('anchor_native_path') if row['association'] else None
        cells = ''.join('<td>'+html.escape(row['methods'][m]['top'] or 'No reading')+('<br>Correct' if row['methods'][m]['exact'] else '<br>Incorrect / missing')+'</td>' for m in METHODS)
        rows.append('<tr><td>'+html.escape(row['id'])+'<br>'+str(row['observations'])+' frames'+
                    (f'<br><img src="{html.escape(picture)}" alt="Automatic plate crop">' if picture else '')+
                    '</td><td>'+html.escape(row['expected'])+'</td>'+cells+'</tr>')
    summary = ''.join('<li>'+names[m]+': '+str(v['exact'])+'/'+str(v['targets'])+' reference targets; '+str(v['multi_frame_exact'])+'/'+str(v['multi_frame_targets'])+' on the ≥3-frame subset</li>' for m, v in report['summary'].items())
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sentinel · Temporal comparison</title><style>body{font:16px system-ui;background:#f6f7f2;color:#17322d;margin:35px auto;padding:0 20px;max-width:1400px}h1{color:#0f766e}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #ccd9d0;padding:12px;text-align:left;vertical-align:top}img{max-width:260px;max-height:100px}small{color:#53655c}</style><h1>Same OCR, three reading strategies</h1><p>'+html.escape(report['scope'])+'</p><ul>'+summary+'</ul><p>Unmatched targets remain in the 18-target denominator. The ≥3-frame subset isolates cases where temporal evidence exists. Model scores are not verified probabilities. No model was changed.</p><div style="overflow:auto"><table><thead><tr><th>Target / automatic crop</th><th>Existing human label</th>'+''.join('<th>'+names[m]+'</th>' for m in METHODS)+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div></html>'
    (output/'index.html').write_bytes(page.encode('utf-8'))


def integrity_audit(args):
    """Explicit post-review sensitivity analysis; never changes the raw report."""
    if not args.mixed_segment or not args.integrity_note or not args.integrity_reviewer:
        raise ValueError('Name visually reviewed mixed segments and explain the evidence')
    target = args.output/'integrity-audit.json'
    if target.exists():
        raise ValueError('Preserve the existing integrity review')
    report = json.loads((args.output/'report.json').read_bytes())
    with gzip.open(args.output/'trace.json.gz', 'rt', encoding='utf-8') as source:
        trace = json.load(source)
    if sha(args.output/'trace.json.gz') != report['trace_sha256']:
        raise ValueError('The trace changed after scoring')
    segments = {s['id']:s for s in trace['segments']}
    excluded = set(args.mixed_segment)
    if not excluded <= segments.keys():
        raise ValueError('Unknown OCR-reader segment')
    matched = [c for c in report['cases'] if c['observations']]
    retained = [c for c in matched if c['association']['segment'] not in excluded]
    excluded_cases = [c['id'] for c in matched if c not in retained]
    clean = {method:dict(exact=sum(c['methods'][method]['exact'] for c in retained), targets=len(retained),
                        multi_frame_exact=sum(c['methods'][method]['exact'] for c in retained if c['observations']>=3),
                        multi_frame_targets=sum(c['observations']>=3 for c in retained)) for method in METHODS}
    audit = dict(scope='Post-review integrity sensitivity analysis, not a replacement for the frozen raw result',
                 report_sha256=sha(args.output/'report.json'), trace_sha256=report['trace_sha256'],
                 reviewer=args.integrity_reviewer,
                 reason=args.integrity_note, excluded_segments=sorted(excluded),
                 excluded_reference_ids=excluded_cases, raw_matched_targets=len(matched),
                 retained_matched_targets=len(retained), summary=clean)
    pictures = []
    for key in sorted(excluded):
        for observation in segments[key]['observations']:
            if sha(args.output/observation['native_path']) != observation['native_sha256']:
                raise ValueError('Integrity evidence changed')
            pictures.append('<figure><figcaption>'+html.escape(key)+' · '+str(observation['position_ms']/1000)+
                            ' s</figcaption><img alt="Original crop from mixed OCR history" src="'+
                            html.escape(observation['native_path'])+'"></figure>')
    rows = []
    for method in METHODS:
        raw, corrected = report['summary'][method], clean[method]
        rows.append('<tr><td>'+html.escape(method.replace('_',' '))+'</td><td>'+str(raw['matched_exact'])+'/'+
                    str(raw['matched_targets'])+'</td><td>'+str(corrected['exact'])+'/'+str(corrected['targets'])+
                    '</td><td>'+str(corrected['multi_frame_exact'])+'/'+str(corrected['multi_frame_targets'])+'</td></tr>')
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sentinel · OCR comparison and integrity finding</title><style>body{font:17px system-ui;background:#f6f7f2;color:#17322d;max-width:1150px;margin:40px auto;padding:0 20px}h1,h2{color:#0f766e}table{border-collapse:collapse;background:white;width:100%}td,th{padding:15px;border:1px solid #ccd9d0;text-align:left}.frames{display:flex;gap:25px;flex-wrap:wrap}figure{background:white;border:1px solid #ccd9d0;padding:15px;margin:0}img{display:block;margin-top:20px;max-width:90vw;width:auto;min-height:110px;object-fit:contain}.notice{padding:18px;background:#e2f2e9;border-left:4px solid #0f766e}</style><h1>What the three-approach test actually found</h1><p class="notice">The quality-selected single frame leads on the raw matched subset. Its only advantage comes from a history that mixed two different vehicles. Excluding that contaminated history, all three approaches tie.</p><div style="overflow:auto"><table><tr><th>Approach</th><th>Raw matched exact</th><th>After excluding mixed history</th><th>Retained targets with ≥3 frames</th></tr>'+''.join(rows)+'</table></div><p>18 previously labelled reference targets were considered. Only six had safe timestamp-and-geometry matches; 12 are association-unavailable at the reference instant, not proven whole-passage detector misses. This is development evidence, not full-video accuracy.</p><h2>Two different plates in one OCR history</h2><p>'+html.escape(args.integrity_note)+'</p><div class="frames">'+''.join(pictures)+'</div><p>The exclusion is a post-review sensitivity analysis. The original report is preserved, and no model or production setting was changed.</p><p><a href="index.html">Open the original per-target comparison</a> · <a href="integrity-audit.json">Download integrity audit</a></p></html>'
    write_json(target, audit)
    (args.output/'integrity.html').write_bytes(page.encode('utf-8'))
    print(json.dumps(dict(stage='integrity_reviewed', excluded_reference_ids=excluded_cases, summary=clean)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('capture', 'score', 'audit'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--review', type=Path, default=Path('data/benchmarks/recorded-footage-20261104/manual-review'))
    parser.add_argument('--recordings-dir', type=Path, default=Path('/recordings'))
    parser.add_argument('--recording-id', default='rec-6b7064605b3d')
    parser.add_argument('--cache', type=Path, default=Path('/models/experiments/awiros'))
    parser.add_argument('--mixed-segment', action='append')
    parser.add_argument('--integrity-note')
    parser.add_argument('--integrity-reviewer')
    args = parser.parse_args()
    {'capture': capture, 'score': score, 'audit': integrity_audit}[args.phase](args)
