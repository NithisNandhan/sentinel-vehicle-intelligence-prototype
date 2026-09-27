"""Static local viewer of actual detector reports; no media or credentials."""
import html
import json
from pathlib import Path


ROOT=Path('/workspace/data/benchmarks/reports')


def escape(value):
    return html.escape(str(value))


def table(headings,rows):
    return '<table><thead><tr>'+''.join('<th>'+escape(h)+'</th>' for h in headings)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+escape(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'


def read(name):
    path=ROOT/name
    return json.loads(path.read_text()) if path.exists() else None


def comparison(name):
    report=read(name)
    if report is None:return '<p>Result not available.</p>'
    rows=[]
    for entry in report['models']:
        if entry['device']!='cuda' or entry['status']!='completed':continue
        score=entry['summary']
        rows.append([entry['name'],entry['input_size'],
                     str(score['target_hits_iou50'])+'/'+str(score['cases']),
                     score['target_hits_iou75'],score['median_ms']])
    return table(['Model','Input size','Targets found (overlap ≥ 0.5)',
                  'Tightly aligned (overlap ≥ 0.75)','GPU median milliseconds'],rows)


def adaptation(name,source=None):
    report=read(name)
    if report is None:return '<p class="notice">Training or evaluation is not yet complete. No result is claimed.</p>'
    rows=[]
    for entry in report['models']:
        for condition,score in entry['conditions'].items():
            if source:
                selected=[r for r in entry['cases'] if r['source']==source and r['condition']==condition]
                score={t:{k:sum(r[t][k] for r in selected) for k in ('matched','annotations','unmatched_predictions')}
                       for t in ('iou50','iou75')}
                values=score['iou50']
                values['annotation_recall']=values['matched']/values['annotations']
                values['annotation_precision']=values['matched']/(values['matched']+values['unmatched_predictions'])
            values=score['iou50']; tight=score['iou75']
            rows.append([entry['name'],condition,
                         str(values['matched'])+'/'+str(values['annotations']),
                         str(round(values['annotation_recall']*100,2))+'%',
                         str(round(values['annotation_precision']*100,2))+'%',tight['matched']])
    image_count=len({r['id'] for r in report['models'][0]['cases'] if r['source']==source}) if source else report['sample_images']
    return '<p>'+escape(image_count)+' reserved original images. Degraded copies are not additional independent vehicles.</p>'+table(
        ['Model','Test condition','Matched / labelled objects','Label coverage',
         'Annotation precision','Tightly aligned objects'],rows)+'<p class="small">'+escape(report['caveat'])+'</p>'


def main():
    parts=['''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>Sentinel detector laboratory</title><style>
body{font:16px/1.5 system-ui,sans-serif;background:#101827;color:#edf3ff;max-width:1150px;margin:auto;padding:30px}
h1,h2{color:#9bd7ff}table{border-collapse:collapse;width:100%;background:#182338;margin:20px 0}
td,th{padding:10px;border:1px solid #34445e;text-align:left}th{background:#263751}
.notice{background:#3c301c;padding:16px;border-left:4px solid #ffc75d}.small{font-size:13px;color:#b4c3d9}
a{color:#9bd7ff}.scroll{overflow:auto}</style></head><body>
<h1>Vehicle &amp; plate detector laboratory</h1>
<p class="notice">Detector tables measure box localisation, not plate-text accuracy. Separate
tracked OCR probes appear below. No live-camera accuracy or 90% plate-text accuracy is claimed.
Experimental weights are not production defaults.</p>
<h2>Initial vehicle comparison</h2><p>13 public development images, 11 target vehicles, 65 stress cases.
Target boxes are approximate, not an exhaustively labelled scene.</p>''',
        '<div class="scroll">'+comparison('detector-comparison-vehicle.json')+'</div>',
        '<h2>Initial plate comparison</h2><div class="scroll">'+comparison('detector-comparison-plate-cuda.json')+'</div>',
        '<h2>Fine-tuned plate model: reserved publisher-labelled images</h2><div class="scroll">'+adaptation('plate-detector-adaptation-test.json')+'</div>',
        '<h2>Gujarat-only plate locations</h2><p>Indian subset, separate from the mixed-country result above. These are box matches, not recognised plate numbers.</p><div class="scroll">'+adaptation('plate-detector-adaptation-test.json','gujarat')+'</div>',
        '<h2>Vehicle pilot: Madrid camera</h2><p>One camera and six test images cannot establish Indian camera generalisation.</p><div class="scroll">'+adaptation('vehicle-detector-adaptation-madrid-test.json')+'</div>',
        '<h2>Indian CCTV vehicle adaptation</h2><div class="scroll">'+adaptation('vehicle-detector-adaptation-bmd-test.json')+'</div>']
    export=read('plate-export-regression.json')
    if export:
        score=export['summary']
        parts.append('<h2>Plate export checks</h2><p>'+escape(export['real_image_parity_checks'])+
            ' real-image CPU/GPU numerical checks passed with TF32 disabled. On the previous diagnostic: '+
            escape(score['target_hits_iou50'])+'/'+escape(score['cases'])+' targets found, '+
            escape(score['target_hits_iou75'])+' tightly aligned. Parent tight alignment was 51: this regression blocks global promotion.</p>')
    regression=read('vehicle-bmd-cross-source-regression.json')
    if regression:
        parts.append('<h2>Vehicle cross-source regression</h2>'+table(
            ['Model','Earlier targets found / 65','Tightly aligned'],
            [[r['name'],str(r['summary']['target_hits_iou50'])+'/65',r['summary']['target_hits_iou75']]
             for r in regression['models']])+
            '<p class="notice">The Indian adaptation helps distant CCTV vehicles but loses coverage on earlier examples and some car categories. Keep original YOLO26s as the general candidate; treat adapted YOLO26s as a separate CCTV specialist. Do not copy the 13-class checkpoint over the COCO-filtered live model.</p>')
    shadow=read('detector-ocr-live-20260927.json')
    if shadow:
        rows=[]
        for camera in shadow['cameras']:
            for name,score in camera['methods'].items():
                rows.append([camera['camera_id'],name,camera['sampled_frames'],score.get('vehicle_box_instances',0),
                    score.get('plate_crops',0),score.get('maximum_distinct_observations',0),
                    score.get('candidate_windows',0),score['unique_model_accepted_unverified']])
        parts.append('<h2>Live detector → tracked multi-frame OCR</h2><p class="notice">'+
            'Live model proposals only: no independent plate transcription was available. Model acceptance is NOT a verified reading. Native frames and plate text were not retained. '+
            escape(shadow['caveat'])+'</p><div class="scroll">'+table(
            ['Camera','Profile','Paired sampled frames','Vehicle box instances','Plate crops',
             'Max distinct observations','Candidate windows','Accepted by model, UNVERIFIED'],rows)+'</div>')
    sequence=read('detector-ocr-sequence-r2.json') or read('detector-ocr-sequence.json')
    if sequence:
        parts.append('<h2>Automatic detection → OCR on labelled Indian sequence</h2><p>'+escape(sequence['caveat'])+
            '</p>'+table(['Profile','Original frames','Plate crops','Maximum distinct observations','Result'],
            [[name,sequence['original_frames'],s.get('plate_crops',0),s['maximum_distinct_observations'],s['outcome']]
             for name,s in sequence['methods'].items()])+
            '<p><a href="'+escape(sequence['visuals'])+'">Inspect the original automatic crops and ranked plate readings</a></p>')
    standalone=read('detector-ocr-general-live-20260927.json')
    if standalone:
        parts.append('<h2>Selected profile running alone on CAM17</h2>'+table(
            ['Camera','Sampled frames','Plate crops','Distinct observations, max','Candidate windows','Accepted, UNVERIFIED'],
            [[c['camera_id'],c['sampled_frames'],c['methods']['general'].get('plate_crops',0),
              c['methods']['general']['maximum_distinct_observations'],c['methods']['general'].get('candidate_windows',0),
              c['methods']['general']['unique_model_accepted_unverified']] for c in standalone['cameras']])+
            '<p>90-second bounded diagnostic, ROI budget four. This is a separate window, not a paired accuracy comparison.</p>')
    smoke=read('detector-ocr-viewer-profile-recheck-20260927.json')
    if smoke:
        parts.append('<h2>Final viewer-profile live check</h2>'+table(
            ['Camera','Seconds observed','Window complete','Analysed frames','Plate crops','Candidate windows','Accepted, UNVERIFIED'],
            [[c['camera_id'],c['observed_seconds'],c['requested_window_completed'],c['sampled_frames'],
              c['methods']['general'].get('plate_crops',0),c['methods']['general'].get('candidate_windows',0),
              c['methods']['general']['unique_model_accepted_unverified']] for c in smoke['cameras']])+
            '<p>Exact viewer profile and deployed image, ROI budget three. The preceding attempt ended early at 15.43 seconds and remains an incomplete report; the final 30-second recheck completed after the clean-EOF retry fix. Neither established a verified plate reading.</p>')
    parts.append('<h2>How to interpret this</h2><p>A matched box overlaps its source annotation by at least half of their combined area. Tight alignment uses a stricter three-quarter threshold. Annotation precision counts unmatched predictions against publisher labels; incomplete labels can affect it.</p><p>No credentials or genuine Sentinel frames are stored in this report. Timing is a local adapter measurement, not guaranteed camera throughput.</p><p><a href="indian-corpus-summary.html">Separate OCR experiment results</a></p></body></html>')
    (ROOT/'detector-summary.html').write_text('\n'.join(parts),encoding='utf-8')
    print(json.dumps({'viewer':'detector-summary.html','live_validation':False}),flush=True)


if __name__=='__main__':main()
