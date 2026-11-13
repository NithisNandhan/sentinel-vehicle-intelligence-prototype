import { useEffect, useState } from 'react';
import { getDiagnostics } from './api';
import { diagnosticIdentityMatches, formatVideoTime } from './recognition';
import type { CharacterEvidence, DiagnosticsSnapshot, RejectedProposal } from './unified-contract';
import DetailSheet from './DetailSheet';

export function imageData(value: string | null | undefined, kind = 'png'): string | undefined {
  return value && value.length <= 1_500_000 && /^[A-Za-z0-9+/]*={0,2}$/.test(value) ? `data:image/${kind};base64,${value}` : undefined;
}

function Characters({ characters }: { characters?: CharacterEvidence[] }) {
  if (!characters?.length) return null;
  return <details className="character-details"><summary>Character scores and alternatives</summary><div className="characters">{characters.map((char, index) => <div key={index}><strong>{char.character}</strong><span>{char.score.toFixed(3)}</span><small>{char.alternatives?.map(alt => `${alt.character} ${alt.score.toFixed(3)}`).join(' · ')}</small></div>)}</div></details>;
}


export function WithheldProposals({ proposals }: { proposals: RejectedProposal[] }) {
  if (!proposals.length) return null;
  return <section className="inspection-section"><h3>Withheld proposals · not used for reading</h3>
    <p className="caption">These candidate crops did not pass the evidence checks. They contribute no observations or reconstruction votes. A withheld proposal may still be a real plate; this is not a non-plate verdict.</p>
    {proposals.map((proposal, index) => <article className="frame-evidence" key={`withheld-${proposal.pts_ms}-${index}`}>
      <div className="section-heading"><span className="mono">{proposal.text || 'No raw OCR text'}</span><span className="caption">{formatVideoTime(proposal.pts_ms)}</span></div>
      {imageData(proposal.image) ? <div className="crop-frame"><img src={imageData(proposal.image)} alt={`Withheld candidate crop at ${formatVideoTime(proposal.pts_ms)}`} /></div> : <p className="muted">Native preview unavailable within the display limit.</p>}
      <p>{proposal.reason || 'Insufficient evidence for the reading history'}</p><p className="caption">{proposal.width} × {proposal.height} native pixels · raw OCR suggestion · excluded from fusion</p>
      <Characters characters={proposal.characters} />
    </article>)}
  </section>;
}

export default function Inspector({ sessionId, trackId, rowKey, onClose }: { sessionId: string; trackId: string; rowKey?: string; onClose: () => void }) {
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
          if (!diagnosticIdentityMatches(next, sessionId, trackId, rowKey)) {
            setSnapshot(null); setError('The selected vehicle passage changed. Close this panel and select its current row.');
          }
          else { setSnapshot(next); setError(''); }
        }
      } catch (failure) { if (!cancelled) { setSnapshot(null); setError((failure as Error).message); } }
      finally { pending = false; }
    }
    void load(); const timer = window.setInterval(() => void load(), 2000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [sessionId, trackId, rowKey]);
  const track = snapshot?.track;
  const frames = track?.frames || [];
  // Keep the native/working comparison on the same timestamp. Prefer the
  // latest text-guided input when the track has one; the full history is below.
  const native = [...frames].reverse().find(frame => frame.working_view === 'text_quad') || frames[frames.length - 1];
  const candidates = track?.candidates || [];
  const reconstruction = track?.reconstruction;
  const rejected = track?.rejected_proposals || [];
  return <DetailSheet title={`Track ${trackId}`} subtitle="Recognition Inspector" onClose={onClose}>
    {error && <div role="alert" className="notice error">{error}<p>Evidence may be disabled, expired or unavailable. Closing this panel does not stop analysis.</p></div>}
    {!track && !error && <div className="empty">Loading model evidence…</div>}
    {track && <>
      <div className="inspector-reading"><span className="overline">Best available reading · unverified</span><strong className="plate-text">{track.best_reading || track.masked_plate || 'No complete reading'}</strong><p>{track.observations || 0} distinct observations · video {formatVideoTime(track.seen_pts_ms)}</p></div>
      {track.passage_id && <p className="caption">Recognition passage {track.passage_id}{track.passage_start_reason ? ` · started: ${track.passage_start_reason.replaceAll('_', ' ')}` : ''}. Evidence is combined only within this passage.</p>}
      <section className="inspection-section"><div className="section-heading"><h3>Native crop</h3><button className="button quiet" onClick={() => setZoom(!zoom)} aria-pressed={zoom}>{zoom ? 'Fit crop' : 'Enlarge crop'}</button></div>
        {imageData(native?.image) ? <div className={`crop-frame ${zoom ? 'zoomed' : ''}`}><img src={imageData(native?.image)} alt="Native plate crop, unchanged source pixels" /></div> : <p className="muted">No admitted native crop for this passage.{rejected.length ? ' Withheld candidate crops are shown below.' : ''}</p>}
        <p className="caption">{native ? `${native.width} × ${native.height} native pixels · ${formatVideoTime(native.pts_ms)}. ` : ''}Enlargement adds no image detail.</p>
        {!!native?.crop_sources?.length && <p className="caption">Crop source: {native.crop_sources.join(' + ')}.</p>}
      </section>
      {native?.working_view === 'text_quad' && <section className="inspection-section"><h3>Text-guided OCR input</h3>
        {imageData(native.working_image) ? <div className="crop-frame"><img src={imageData(native.working_image)} alt="OCR working crop straightened using detected text corners" /></div> : <p className="muted">The straightened view exceeds the display limit.</p>}
        <p className="caption">Automatically outlined and straightened. Interpolation adds no detail. This and the native crop are one source observation; the OCR character scores describe this working view.</p>
      </section>}
      <WithheldProposals proposals={rejected} />
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
