"""Small local landing page for measured experimental OCR results."""
import html
import json
from pathlib import Path

root=Path('data/benchmarks/reports')
training=json.loads((root/'indian-corpus-adaptation.json').read_text())
gpu=json.loads((root/'ocr-gpu-check.json').read_text())
parent=json.loads((root/'indian-corpus-multiframe-parent.json').read_text())
tuned=json.loads((root/'indian-corpus-multiframe-adapted.json').read_text())
metrics=[]
for label,a,b in [('Validation exact',training['baseline_validation'],training['tuned_validation']),
                   ('Test exact',training['baseline_test'],training['tuned_test'])]:
    metrics.append(f'<tr><td>{label}</td><td>{a["exact"]}/{a["samples"]}</td><td>{b["exact"]}/{b["samples"]}</td></tr>')
metrics.append(f'<tr><td>Test character errors</td><td>{training["baseline_test"]["character_errors"]}/295</td><td>{training["tuned_test"]["character_errors"]}/295</td></tr>')
metrics.append(f'<tr><td>Multi-frame correct / wrong / uncertain</td><td>{parent["correct"]} / {parent["wrong"]} / {parent["abstain"]}</td><td>{tuned["correct"]} / {tuned["wrong"]} / {tuned["abstain"]}</td></tr>')
live=[]
for name in ('indian-corpus-live-shadow.json','indian-corpus-live-vehicle-shadow.json','indian-corpus-live-direct-shadow.json'):
    path=root/name
    if path.is_file():
        doc=json.loads(path.read_text())
        live.append({'test':name,'ground_truth_available':doc['ground_truth_available'],
                     'recordings_saved':doc['recordings_saved'],'plaintext_plates_saved':doc['plaintext_plates_saved'],
                     'cameras':doc['cameras']})
content='''<!doctype html><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'"><title>Indian OCR deeper adaptation</title>
<style>body{font:18px system-ui;background:#101b2a;color:#e8eff9;max-width:1050px;margin:36px auto;padding:20px}a{color:#8fd0ff}table{border-collapse:collapse;width:100%}td,th{border:1px solid #40516a;padding:12px;text-align:left}.warning{background:#442f12;padding:18px;border-radius:10px}pre{font-size:14px;white-space:pre-wrap;overflow-wrap:anywhere}</style>
<h1>Deeper Indian-plate OCR: measured results</h1><p>GPU installation, bounded encoder training, native-crop diagnostics and multi-frame reconstruction tests.</p>
<p class="warning">Experimental checkpoint, NOT deployed. 28/31 exact test crops is 90.3% on this small still-image diagnostic, not 90% live-camera reliability. New labels were reviewed by AI and compared to publisher text, not independently human-verified. Prior pretraining overlap is unknown. Most photographic training labels remain publisher-silver.</p>
<h2>Parent versus adapted</h2><table><tr><th>Metric</th><th>Parent</th><th>Adapted</th></tr>'''+''.join(metrics)+'''</table>
<p>12 epochs, epoch 8 selected only on validation. 133,864 trainable parameters; other weights and batch-normalisation buffers stayed unchanged. Full checkpoint reload passed. CPU/GPU inference parity passed with TF32 disabled.</p>
<h2>Inspect the readings and candidates</h2><p><a href="indian-corpus-adaptation.html">31 test crops: side-by-side readings and ranked candidates</a> (large page, about 17 MB)</p>
<p><a href="indian-corpus-multiframe-parent.html">Parent multi-frame degradation viewer</a> · <a href="indian-corpus-multiframe-adapted.html">Adapted multi-frame degradation viewer</a></p>
<p>Multi-frame evaluation: one development-visible vehicle, three distinct source frames, 21 degradations. Darkness improved; severe downsampling and motion-blur boundaries did not. Candidate scores are not calibrated probabilities.</p>
<h2>Live-camera diagnostic</h2><p>Counts below are not accuracy. Plate-box association is a read-only diagnostic, not production YOLO/ByteTrack validation. No recordings, plaintext plates or central writes.</p>
<pre>'''+html.escape(json.dumps(live,indent=2))+'''</pre>
<h2>Evidence files</h2><p><a href="indian-corpus-adaptation.json">Training results</a> · <a href="ocr-gpu-check.json">GPU parity check</a> · <a href="indian-corpus-multiframe-adapted.json">Multi-frame results</a></p>'''
(root/'indian-corpus-summary.html').write_text(content,encoding='utf-8')
print(json.dumps({'summary':'indian-corpus-summary.html','live_reports':len(live),'production_promoted':False}))
