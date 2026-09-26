"""Local licensed benchmark visual report. Never called by live workers."""
from __future__ import annotations

import base64
import html
import math
from pathlib import Path
from urllib.parse import urlparse

from .degradation_benchmark import LICENSES


def validate_viewer_scope(output: Path, dataset: dict, *, source_mode='benchmark'):
    if source_mode != 'benchmark':
        raise ValueError('Live/Sentinel visual reports are forbidden')
    if dataset.get('license') not in LICENSES or dataset.get('permitted_use') != 'noncommercial_benchmark':
        raise ValueError('Explicit benchmark permission required')
    for field in ('source_url', 'license_url'):
        parsed = urlparse(dataset.get(field, ''))
        host = (parsed.hostname or '').lower()
        if (parsed.scheme != 'https' or not host or parsed.username or parsed.password or
                'sentinel' in host or host.endswith('corp8.cloud')):
            raise ValueError('Invalid benchmark provenance')
    parts = output.resolve().parts
    if not any(tuple(parts[i:i + 3]) == ('data', 'benchmarks', 'reports') for i in range(len(parts) - 2)):
        raise ValueError('Viewer output must remain under ignored data/benchmarks/reports')
    if output.suffix.lower() != '.html':
        raise ValueError('Viewer output must be HTML')


def ranked_candidates(original, processed=()):
    unique = {}
    for origin, candidates in (('original-frame fusion', original), ('processed view only', processed)):
        for candidate in candidates:
            if not all(math.isfinite(v) for v in (candidate.score, candidate.minimum_character_score, candidate.log_score)):
                continue
            # Original evidence takes precedence over a correlated processed view.
            if candidate.plate not in unique:
                unique[candidate.plate] = (candidate, origin)
    return sorted(unique.values(), key=lambda item: (-item[0].log_score, item[0].plate))[:200]


def _image(image):
    if image is None:
        return '<p class="unavailable">Unavailable — no valid result for this stage</p>'
    import cv2
    valid, encoded = cv2.imencode('.png', image)
    if not valid:
        raise ValueError('Benchmark image encoding failed')
    data = base64.b64encode(encoded).decode('ascii')
    return f'<img alt="Licensed benchmark plate crop" src="data:image/png;base64,{data}">'


def render_viewer(output: Path, dataset: dict, cases: list, *, source_mode='benchmark'):
    validate_viewer_scope(output, dataset, source_mode=source_mode)
    if len(cases) > 100:
        raise ValueError('Visual report exceeds case bound')
    esc = lambda value: html.escape(str(value), quote=True)
    sections = []
    for case in cases:
        if case.get('source_mode') != 'benchmark':
            raise ValueError('Every visual case must be permitted benchmark data')
        candidates = ranked_candidates(case.get('candidates', []), case.get('processed_candidates', []))
        accepted = case.get('accepted')
        status = 'RECOGNISED' if accepted else 'UNCERTAIN' if candidates else 'NO-READ'
        alternatives = [c.plate for c, _ in candidates]
        differing = {i for i in range(max((len(t) for t in alternatives), default=0))
                     if len({t[i] if i < len(t) else None for t in alternatives}) > 1}
        def marked(text):
            return ''.join(f'<mark>{esc(char)}</mark>' if i in differing else esc(char)
                           for i, char in enumerate(text))
        rows = ''.join(f'<tr><td>{rank}</td><td class="plate">{marked(c.plate)}</td>'
                       f'<td>{c.log_score:.4f}</td><td>{c.score:.4f}</td><td>{c.minimum_character_score:.4f}</td>'
                       f'<td>{c.observations}</td><td>{esc(origin)}</td></tr>'
                       for rank, (c, origin) in enumerate(candidates, 1))
        if not rows:
            rows = '<tr><td colspan="7">No retained complete candidate. No plate has been invented.</td></tr>'
        images = ''.join(f'<figure><figcaption>{label}</figcaption>{_image(case.get(key))}</figure>'
                         for label, key in (
                             ('Original native crop', 'original'),
                             ('Processed reconstruction — not original footage', 'reconstruction'),
                             ('Learned deblurring — may alter character details', 'restored')))
        diagnostics = case.get('frame_diagnostics', [])
        if len(diagnostics) > 40:
            raise ValueError('Too many frame diagnostics')
        diagnostic_html = ''.join(
            f'<figure><figcaption>Original frame {esc(item.get("index"))} · {esc(item.get("pts_ms"))} ms'
            f'<br>Raw OCR: {esc(item.get("reading"))} · OCR time: {esc(item.get("ocr_ms"))} ms</figcaption>'
            f'{_image(item.get("image"))}</figure>' for item in diagnostics)
        extra = (f'<p>Manually verified evaluation label: {esc(case["expected"])}. '
                 'This label was not supplied to inference.</p>' if case.get('expected') else '')
        if case.get('score_note'):
            extra += f'<p>{esc(case["score_note"])}</p>'
        if diagnostics:
            extra += ('<details><summary>Inspect raw OCR on each original frame — diagnostic, not separate product detections</summary>'
                      f'<div class="images">{diagnostic_html}</div></details>')
        sections.append(f'''<section><h2>{esc(case.get('title', 'Benchmark passage'))}</h2>
<p class="status">{status}</p><p>Accepted plate: {esc(accepted or 'none')} · Partial reading: {esc(case.get('partial') or 'unavailable')}</p>
<p>Reason: {esc(case.get('reason', 'unknown'))} · Native dimensions: {esc(case.get('dimensions'))} · Reconstruction sources: {esc(case.get('source_count', 0))}
 · Rejected alignments: {esc(case.get('rejected', 0))} · Processing: {esc(case.get('duration_ms'))} ms</p>
<p>Evaluation outcome: {esc(case.get('outcome', 'not scored'))}. Ground truth was used only after inference.</p>
<div class="images">{images}</div>
{extra}
<p>Retained candidates: {len(candidates)} / maximum 200. This is not an exhaustive list; the correct plate may be absent.
Scores are uncalibrated model scores, not probability of correctness. Ranking uses whole-sequence log score, not the mean character score.
Processed views are correlated hypotheses, not extra frames.</p>
<div class="scroll"><table><thead><tr><th>Rank</th><th>Candidate</th><th>Sequence log score</th><th>Mean character score</th><th>Weakest character</th><th>Source observations</th><th>Evidence origin</th></tr></thead><tbody>{rows}</tbody></table></div></section>''')
    body = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>Sentinel local multi-frame benchmark</title><style>
body{{background:#101925;color:#edf2f7;font:16px system-ui;margin:0;padding:24px;line-height:1.5}}
main{{max-width:1400px;margin:auto}}section{{border-top:1px solid #536174;padding:24px 0}}h1,h2{{line-height:1.2}}
.images{{display:flex;gap:20px;flex-wrap:wrap}}figure{{margin:0;flex:1;min-width:240px}}figcaption{{min-height:48px;color:#b9c8dc}}
img{{display:block;width:100%;height:160px;object-fit:contain;background:#050a10;image-rendering:auto}}
.status{{font-weight:700;color:#ffce72}}.scroll{{overflow:auto;max-height:480px}}table{{border-collapse:collapse;width:100%;text-align:left}}
td,th{{padding:10px;border-bottom:1px solid #465568}}th{{position:sticky;top:0;background:#182535}}.plate{{font-family:monospace;white-space:nowrap}}
mark{{background:#ffce72;color:#101925}}.unavailable{{color:#c5ceda}}footer{{margin-top:24px;color:#b9c8dc}}
</style></head><body><main><h1>Local multi-frame plate benchmark</h1>
<p>Licensed offline evaluation only — not genuine Sentinel evidence or production accuracy.</p>
<p>These visual cases use manually annotated plate crops to isolate OCR. They do not demonstrate automatic vehicle/plate localisation.</p>
{''.join(sections)}<footer>Dataset: {esc(dataset.get('name'))}. Licence: {esc(dataset.get('license'))}.
Source: {esc(dataset.get('source_url'))}. Licence notice: {esc(dataset.get('license_url'))}.
Modifications: cropping, controlled degradation, geometric alignment, reconstruction and optional deblurring.
This report performs no external network requests and sends no alerts.</footer></main></body></html>'''
    if len(body.encode('utf-8')) > 64 * 1024 * 1024:
        raise ValueError('Visual report exceeds byte bound')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(body, encoding='utf-8')
    return output
