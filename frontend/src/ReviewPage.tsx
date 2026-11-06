import { useCallback, useEffect, useState, type FormEvent } from 'react';
import { ShieldCheck, RefreshCw } from 'lucide-react';
import { formatTime } from './api';
import { imageData } from './Inspector';
import DetailSheet from './DetailSheet';
import { REASON_LABELS, confirmAgrees, decideReview, getReviewDetail, getReviewEvidence, getReviewQueue, normalisePlate,
  patternCells, type ReviewAction, type ReviewDetail, type ReviewEvidence, type ReviewItem, type Quality } from './review';

function Pattern({ item, evidence }: { item: ReviewItem; evidence?: ReviewEvidence | null }) {
  return <span className="pattern" aria-label="Plate pattern; question marks are unknown positions">{patternCells(item, evidence?.pattern).map((cell, index) =>
    <span key={index} className={`cell ${cell.state}`}>{cell.character}</span>)}</span>;
}

function QualityFacts({ quality }: { quality: Quality | null | undefined }) {
  if (!quality) return <p className="caption">No quality indicators were reported.</p>;
  const flags = (['blurred', 'low_contrast', 'glare', 'low_light', 'tilted'] as const).filter(key => quality[key]);
  return <dl className="facts">
    <div><dt>Plate size</dt><dd>{quality.width_px ?? '—'} × {quality.height_px ?? '—'} native px</dd></div>
    <div><dt>Sharpness</dt><dd>{quality.sharpness ?? '—'}</dd></div>
    <div><dt>Contrast</dt><dd>{quality.contrast ?? '—'}</dd></div>
    <div><dt>Angle</dt><dd>{quality.angle_deg == null ? 'not measured' : `${quality.angle_deg}°`}</dd></div>
    <div><dt>Indicators</dt><dd>{flags.length ? flags.map(flag => flag.replace('_', ' ')).join(', ') : 'none'}</dd></div>
  </dl>;
}

function Detail({ token, id, onDecided, onClose }: { token: string; id: number; onDecided: () => void; onClose: () => void }) {
  const [detail, setDetail] = useState<ReviewDetail | null>(null);
  const [evidence, setEvidence] = useState<ReviewEvidence | null>(null);
  const [error, setError] = useState('');
  const [plate, setPlate] = useState('');
  const [note, setNote] = useState('');
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    let cancelled = false;
    getReviewDetail(token, id).then(next => { if (!cancelled) setDetail(next); }).catch(failure => { if (!cancelled) setError((failure as Error).message); });
    getReviewEvidence(token, id).then(next => { if (!cancelled) { setEvidence(next); if (next?.pattern && !next.pattern.includes('?')) setPlate(next.pattern); } }).catch(() => { if (!cancelled) setEvidence(null); });
    return () => { cancelled = true; };
  }, [token, id]);
  async function decide(action: ReviewAction, event?: FormEvent) {
    event?.preventDefault();
    if (!detail) return;
    if ((action === 'confirm' || action === 'correct') && !normalisePlate(plate)) { setError('Enter the full plate you can read in the evidence.'); return; }
    if (action === 'confirm' && !confirmAgrees(plate, detail, evidence?.pattern)) { setError('This plate contradicts the known characters. Use Correct if the model was wrong.'); return; }
    setBusy(true); setError('');
    try { await decideReview(token, id, action, action === 'confirm' || action === 'correct' ? normalisePlate(plate) : undefined, note.trim() || undefined); setPlate(''); onDecided(); }
    catch (failure) { setError((failure as Error).message); }
    finally { setBusy(false); }
  }
  const open = detail && (detail.review_status === 'pending' || detail.review_status === 'not_required');
  const frame = imageData(evidence?.frame_jpeg_base64, 'jpeg');
  return <DetailSheet title={`Review passage ${id}`} subtitle="Unverified model evidence · authorised reviewers only" onClose={onClose}>
    {error && <div className="notice error" role="alert">{error}</div>}
    {!detail ? <p>Loading…</p> : <>
      <div className="review-summary"><Pattern item={detail} evidence={evidence} /><span className={`status ${detail.result_state === 'partial' ? 'warning' : 'neutral'}`}>{detail.result_state}</span>{detail.watchlist_lead && <span className="status warning">Possible watchlist lead · review only</span>}</div>
      <dl className="facts">
        <div><dt>Why not confirmed</dt><dd>{REASON_LABELS[detail.reason || ''] || detail.reason || '—'}</dd></div>
        <div><dt>Camera / source</dt><dd>{detail.camera_id} · {detail.source_mode === 'recorded' ? 'Recorded footage' : detail.source_mode}</dd></div>
        <div><dt>Observed</dt><dd>{formatTime(detail.observed_at)} → {formatTime(detail.last_observed_at)}</dd></div>
        <div><dt>Distinct frames</dt><dd>{detail.observations}</dd></div>
        <div><dt>Layout</dt><dd>{detail.layout.replace('_', ' ')}</dd></div>
        <div><dt>Length agreed by frames</dt><dd>{detail.length_certain === false ? 'No — watchlist search disabled' : 'Yes'}</dd></div>
        <div><dt>Profile / policy</dt><dd className="mono">{detail.profile || '—'} · {detail.policy_version || '—'}</dd></div>
        <div><dt>Evidence retained until</dt><dd>{evidence ? 'Edge memory (short-lived)' : 'Not available'} · record expires {formatTime(detail.expires_at)}</dd></div>
      </dl>
      <h3>Image evidence</h3>
      {!evidence ? <p className="caption">Edge evidence is disabled, expired, or was never kept for this passage. Decide only from what you can verify; otherwise mark it unreadable.</p> : <>
        <p className="caption">{evidence.notice}</p>
        {frame && <img className="review-frame" src={frame} alt="Best original frame for this passage" />}
        <div className="review-crops">{evidence.crops.map(crop => { const src = imageData(crop.image_png_base64, 'png'); return src ? <figure key={crop.pts_ms}><img src={src} alt={`Exact OCR input crop at ${crop.pts_ms} ms`} style={{ width: Math.min(crop.width * 3, 480) }} /><figcaption>{crop.width} × {crop.height} px · quality {crop.quality}</figcaption></figure> : null; })}</div>
        <table className="character-table"><thead><tr><th>Position</th><th>Model character</th><th>Known</th><th>Supporting frames</th><th>Mean peak</th><th>Alternatives</th></tr></thead><tbody>{evidence.positions.map(position => <tr key={position.index} className={position.known ? '' : 'unknown'}><td>{position.index + 1}</td><td className="mono">{position.predicted ?? '—'}</td><td>{position.known ? 'yes' : 'no'}</td><td>{position.support_frames}</td><td>{position.mean_peak.toFixed(3)}</td><td className="mono">{position.alternatives?.join(' ') || '—'}</td></tr>)}</tbody></table>
        <p className="caption">Primary score {evidence.primary_score ?? '—'} (uncalibrated). Secondary reader: {evidence.secondary ? JSON.stringify(evidence.secondary) : 'not run'}.</p>
      </>}
      <h3>Quality indicators</h3><QualityFacts quality={evidence?.quality || detail.quality} />
      {open ? <form className="review-actions" onSubmit={event => void decide('confirm', event)}>
        <label htmlFor="review-plate">Full plate as you read it</label>
        <input id="review-plate" autoComplete="off" spellCheck={false} value={plate} onChange={event => setPlate(event.target.value)} placeholder="Only if every character is visible" />
        <label htmlFor="review-note">Note (optional, no plate text)</label>
        <input id="review-note" value={note} maxLength={500} onChange={event => setNote(event.target.value)} />
        <div className="button-row">
          <button className="button primary" type="submit" disabled={busy}>Confirm reading</button>
          <button className="button" type="button" disabled={busy} onClick={() => void decide('correct')}>Correct reading</button>
          <button className="button" type="button" disabled={busy} onClick={() => void decide('unreadable')}>Mark unreadable</button>
          <button className="button quiet" type="button" disabled={busy} onClick={() => void decide('reject')}>Reject (not a plate / wrong vehicle)</button>
        </div>
        <p className="caption">Confirm or Correct creates a human-reviewed sighting. Only then can a watchlist alert be raised. Pending review never raises one.</p>
      </form> : <p className="notice">Decided: {detail.review_status}.</p>}
      <h3>Audit history</h3>
      {detail.decisions.length ? <ol className="audit">{detail.decisions.map((decision, index) => <li key={index}>{formatTime(decision.decided_at)} · {decision.reviewer} · {decision.action}{decision.masked_plate ? ` · ${decision.masked_plate}` : ''}{decision.alert_id ? ` · alert ${decision.alert_id}` : ''}{decision.note ? ` — ${decision.note}` : ''}</li>)}</ol> : <p className="caption">No decisions yet.</p>}
    </>}
  </DetailSheet>;
}

export default function ReviewPage() {
  const [token, setToken] = useState('');
  const [draft, setDraft] = useState('');
  const [status, setStatus] = useState<'pending' | 'decided' | 'all'>('pending');
  const [queue, setQueue] = useState<{ reviewer: string; pending: number; items: ReviewItem[] } | null>(null);
  const [error, setError] = useState('');
  const [selected, setSelected] = useState<number | null>(null);
  const load = useCallback(async () => {
    if (!token) return;
    try { setQueue(await getReviewQueue(token, status)); setError(''); }
    catch (failure) { setError((failure as Error).message); setQueue(null); }
  }, [token, status]);
  useEffect(() => { void load(); }, [load]);
  if (!token) return <section className="surface review-gate"><ShieldCheck size={28} /><h1>Review queue</h1><p>Partial readings and review leads are restricted to named reviewers. Your token stays in this tab's memory only.</p>
    <form onSubmit={event => { event.preventDefault(); setToken(draft.trim()); setDraft(''); }}><label htmlFor="reviewer-token">Reviewer token</label><input id="reviewer-token" type="password" autoComplete="off" value={draft} onChange={event => setDraft(event.target.value)} /><button className="button primary" type="submit" disabled={!draft.trim()}>Open queue</button></form></section>;
  return <div>
    <div className="page-heading"><div><h1>Review</h1><p>{queue ? `Signed in as ${queue.reviewer} · ${queue.pending} pending` : 'Reviewer access'}</p></div><div className="button-row"><button className="button quiet" onClick={() => void load()}><RefreshCw size={15} /> Refresh</button><button className="button quiet" onClick={() => { setToken(''); setQueue(null); }}>Sign out</button></div></div>
    {error && <div className="notice error" role="alert">{error}</div>}
    <div className="filter-tabs">{(['pending', 'decided', 'all'] as const).map(value => <button key={value} aria-pressed={status === value} className={status === value ? 'active' : ''} onClick={() => setStatus(value)}>{value}</button>)}</div>
    {queue && !queue.items.length && <div className="empty"><strong>Nothing to review</strong><p>Partial readings that need a decision appear here.</p></div>}
    <div className="review-list">{queue?.items.map(item => <button key={item.id} className="reading-row" onClick={() => setSelected(item.id)}>
      <div className="reading-line"><Pattern item={item} />{item.watchlist_lead && <span className="status warning">Watchlist lead</span>}</div>
      <div className="reading-line"><span className="caption">{item.camera_id} · {formatTime(item.observed_at)}</span><span className="caption">{REASON_LABELS[item.reason || ''] || item.reason}</span></div>
      <div className="reading-status"><span className="status neutral">{item.review_status}</span>{item.evidence_available && <span className="caption">Evidence on edge</span>}</div>
    </button>)}</div>
    {selected != null && <Detail token={token} id={selected} onClose={() => setSelected(null)} onDecided={() => { setSelected(null); void load(); }} />}
  </div>;
}
