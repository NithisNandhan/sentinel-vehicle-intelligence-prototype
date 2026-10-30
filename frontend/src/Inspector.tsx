import { useEffect, useState } from 'react';
import { getDiagnostics } from './api';
import { formatVideoTime } from './recognition';
import type { CharacterEvidence, DiagnosticsSnapshot } from './unified-contract';
import DetailSheet from './DetailSheet';

export function imageData(value: string | null | undefined, kind = 'png'): string | undefined {
  return value && value.length <= 1_500_000 && /^[A-Za-z0-9+/]*={0,2}$/.test(value) ? `data:image/${kind};base64,${value}` : undefined;
}

function Characters({ characters }: { characters?: CharacterEvidence[] }) {
  if (!characters?.length) return null;
  return <details className="character-details"><summary>Character scores and alternatives</summary><div className="characters">{characters.map((char, index) => <div key={index}><strong>{char.character}</strong><span>{char.score.toFixed(3)}</span><small>{char.alternatives?.map(alt => `${alt.character} ${alt.score.toFixed(3)}`).join(' · ')}</small></div>)}</div></details>;
}

export default function Inspector({ sessionId, trackId, onClose }: { sessionId: string; trackId: string; onClose: () => void }) {
  const [snapshot, setSnapshot] = useState<DiagnosticsSnapshot | null>(null);
  const [error, setError] = useState('');
  const [expanded, setExpanded] = useState(false);
  const [zoom, setZoom] = useState(false);
  useEffect(() => {
    let cancelled = false, pending = false;
    setSnapshot(null); setError(''); setExpanded(false);
    async function load() {
      if (pending || document.visibilityState !== 'visible') return;
      pending = true;
      try {
        const next = await getDiagnostics(sessionId, trackId);
        if (!cancelled) {
          if (next.session_id !== sessionId || next.track.track_id !== trackId) { setSnapshot(null); setError('The analysis session changed. Select the vehicle again.'); }
          else { setSnapshot(next); setError(''); }
        }
      } catch (failure) { if (!cancelled) { setSnapshot(null); setError((failure as Error).message); } }
      finally { pending = false; }
    }
    void load(); const timer = window.setInterval(() => void load(), 2000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [sessionId, trackId]);
  const track = snapshot?.track;
  const frames = track?.frames || [];
  const native = frames[frames.length - 1];
  const candidates = track?.candidates || [];
  const reconstruction = track?.reconstruction;
  return <DetailSheet title={`Track ${trackId}`} subtitle="Recognition Inspector" onClose={onClose}>
    {error && <div role="alert" className="notice error">{error}<p>Evidence may be disabled, expired or unavailable. Closing this panel does not stop analysis.</p></div>}
    {!track && !error && <div className="empty">Loading model evidence…</div>}
    {track && <>
      <div className="inspector-reading"><span className="overline">Best available reading · unverified</span><strong className="plate-text">{track.best_reading || track.masked_plate || 'No complete reading'}</strong><p>{track.observations || 0} distinct observations · video {formatVideoTime(track.seen_pts_ms)}</p></div>
      <section className="inspection-section"><div className="section-heading"><h3>Native crop</h3><button className="button quiet" onClick={() => setZoom(!zoom)} aria-pressed={zoom}>{zoom ? 'Fit crop' : 'Enlarge crop'}</button></div>
        {imageData(native?.image) ? <div className={`crop-frame ${zoom ? 'zoomed' : ''}`}><img src={imageData(native?.image)} alt="Native plate crop, unchanged source pixels" /></div> : <p className="muted">No retained native crop for this track.</p>}
        <p className="caption">{native ? `${native.width} × ${native.height} native pixels · ${formatVideoTime(native.pts_ms)}. ` : ''}Enlargement adds no image detail.</p>
      </section>
      <section className="inspection-section"><h3>Acceptance checks</h3><ul className="gate-list">{(track.gate_reasons || []).map((reason, index) => <li key={index}>{reason}</li>)}</ul><p className="caption">Model acceptance does not establish an independently correct registration.</p></section>
      <section className="inspection-section"><h3>Reconstruction · derived evidence</h3><p>{reconstruction?.status || 'Not available'}{reconstruction?.reason ? ` · ${reconstruction.reason}` : ''}</p>{imageData(reconstruction?.image) && <div className="crop-frame"><img src={imageData(reconstruction?.image)} alt="Derived multi-frame reconstruction, not an independent source frame" /></div>}{reconstruction?.text && <p className="plate-text">{reconstruction.text}</p>}{!!reconstruction?.contributing_originals && <p className="caption">{reconstruction.contributing_originals} contributing originals · not an extra observation</p>}<Characters characters={reconstruction?.characters} /></section>
      <section className="inspection-section"><div className="section-heading"><h3>Ranked candidates</h3><span className="caption">{candidates.length} retained</span></div>
        {!candidates.length ? <p className="muted">No complete candidate yet.</p> : <><table className="compact-table"><thead><tr><th>Plate hypothesis</th><th>Model score</th></tr></thead><tbody>{candidates.slice(0, expanded ? 200 : 5).map((candidate, index) => <tr key={index}><td className="mono">{candidate.plate}</td><td>{candidate.score.toFixed(3)}</td></tr>)}</tbody></table>{candidates.length > 5 && <button className="button quiet" onClick={() => setExpanded(!expanded)}>{expanded ? 'Show first 5' : `Show ${Math.min(200, candidates.length)} retained candidates`}</button>}</>}
        <p className="caption">Scores are uncalibrated model outputs, not accuracy percentages.</p>
      </section>
      <section className="inspection-section"><h3>Original frame observations</h3>{frames.length ? frames.map((frame, index) => <article className="frame-evidence" key={`${frame.pts_ms}-${index}`}><div className="section-heading"><span className="mono">{frame.text || 'No complete text'}</span><span className="caption">{formatVideoTime(frame.pts_ms)}</span></div>{imageData(frame.image) && <div className="crop-frame"><img src={imageData(frame.image)} alt={`Original plate crop at ${formatVideoTime(frame.pts_ms)}`} /></div>}<p className="caption">{frame.width} × {frame.height} pixels</p><Characters characters={frame.characters} /></article>) : <p className="muted">No retained source-frame evidence.</p>}</section>
      <details className="inspection-section"><summary>Model and processing details</summary><dl className="facts">{Object.entries(snapshot?.models || {}).map(([key, value]) => <div key={key}><dt>{key.replaceAll('_', ' ')}</dt><dd className="mono">{value || 'Unavailable'}</dd></div>)}</dl><h3>Measured stages</h3><dl className="facts">{Object.entries(snapshot?.stage_ms || {}).map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{value.toFixed(1)} ms</dd></div>)}</dl></details>
    </>}
  </DetailSheet>;
}
