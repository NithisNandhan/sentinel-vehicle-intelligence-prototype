import { useEffect, useRef, useState } from 'react';
import { Activity, Camera, CircleAlert, ExternalLink, Pause, Play, RefreshCw, ShieldCheck } from 'lucide-react';
import {
  getRecognitionCameras, getRecognitionStatus, heartbeatRecognition, setRecognitionSession,
  type RecognitionCameras, type RecognitionStatus,
} from './api';
import { recognitionCount, recognitionIsFresh, recognitionLabel, recognitionTone } from './recognition';

/** Both views observe one explicitly started session; mounting this panel never starts inference. */
export default function RecognitionPanel({ operatorKey }: { operatorKey: string }) {
  const [status, setStatus] = useState<RecognitionStatus | null>(null);
  const [catalogue, setCatalogue] = useState<RecognitionCameras | null>(null);
  const [selected, setSelected] = useState('');
  const [error, setError] = useState('');
  const [catalogueError, setCatalogueError] = useState('');
  const [busy, setBusy] = useState(false);
  const active = useRef<RecognitionStatus | null>(null);
  const generation = useRef(0);
  const mounted = useRef(true);
  const changing = useRef(false);

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; generation.current += 1; active.current = null; };
  }, []);

  useEffect(() => {
    let cancelled = false;
    let polling = false;
    let heartbeating = false;
    let loadingCameras = false;
    let catalogueAvailable = false;
    let lastCameraRefresh = 0;
    generation.current += 1;
    active.current = null;
    setStatus(null); setCatalogue(null); setError(''); setCatalogueError('');
    if (!operatorKey) return;

    async function cameras() {
      if (cancelled || loadingCameras || document.visibilityState !== 'visible') return;
      loadingCameras = true;
      lastCameraRefresh = Date.now();
      try {
        const next = await getRecognitionCameras();
        if (!cancelled && document.visibilityState === 'visible') {
          catalogueAvailable = next.available;
          setCatalogue(next); setCatalogueError('');
        }
      } catch (failure) {
        catalogueAvailable = false;
        if (!cancelled) { setCatalogue(null); setCatalogueError((failure as Error).message); }
      } finally { loadingCameras = false; }
    }
    async function poll() {
      if (cancelled || polling || changing.current || document.visibilityState !== 'visible') return;
      polling = true;
      const before = generation.current;
      try {
        const next = await getRecognitionStatus();
        if (!cancelled && before === generation.current && document.visibilityState === 'visible') {
          active.current = next; setStatus(next); setError('');
        }
      } catch (failure) {
        if (!cancelled && before === generation.current) {
          active.current = null; setStatus(null); setError((failure as Error).message);
        }
      } finally { polling = false; }
    }
    async function heartbeat() {
      const current = active.current;
      if (cancelled || heartbeating || changing.current || document.visibilityState !== 'visible' || !current?.camera_id
        || !['live', 'connecting'].includes(current.status)) return;
      heartbeating = true;
      try { await heartbeatRecognition(current.camera_id); }
      catch { if (!cancelled) { active.current = null; setStatus(null); void poll(); } }
      finally { heartbeating = false; }
    }
    function visibility() {
      if (document.visibilityState === 'visible') { void poll(); void cameras(); }
      else { generation.current += 1; active.current = null; setStatus(null); }
    }
    void cameras(); void poll();
    const timer = window.setInterval(() => void poll(), 1_000);
    const keepalive = window.setInterval(() => void heartbeat(), 20_000);
    // Recover when this page opens before the model service is ready, without a reload.
    const catalogueRetry = window.setInterval(() => {
      if (Date.now() - lastCameraRefresh >= (catalogueAvailable ? 30_000 : 15_000)) void cameras();
    }, 15_000);
    document.addEventListener('visibilitychange', visibility);
    return () => {
      cancelled = true; active.current = null;
      window.clearInterval(timer); window.clearInterval(keepalive);
      window.clearInterval(catalogueRetry);
      document.removeEventListener('visibilitychange', visibility);
    };
  }, [operatorKey]);

  async function changeSession(cameraId: string | null) {
    changing.current = true;
    setBusy(true); setError('');
    generation.current += 1;
    // A pending poll must not overwrite an explicit session change.
    const before = generation.current;
    try {
      await setRecognitionSession(cameraId);
      const next = await getRecognitionStatus();
      if (mounted.current && before === generation.current) { active.current = next; setStatus(next); }
    } catch (failure) {
      if (mounted.current && before === generation.current) {
        active.current = null; setStatus(null); setError((failure as Error).message);
      }
    } finally { changing.current = false; if (mounted.current) setBusy(false); }
  }

  const fresh = recognitionIsFresh(status);
  const cameraName = catalogue?.cameras.find(camera => camera.id === status?.camera_id)?.name;
  const tracks = fresh ? status?.tracks || [] : [];
  const running = !!status?.camera_id && ['live', 'connecting'].includes(status.status);
  const sessionLabel = !status ? 'Unavailable' : status.status === 'connecting' ? 'Connecting'
    : status.status === 'live' && !fresh ? 'Waiting for fresh frames' : status.status === 'live' ? 'Live analysis'
      : status.status === 'paused' ? 'Paused' : 'Unavailable';

  return <section className="panel recognition-panel" aria-labelledby="recognition-title">
    <div className="panel-heading"><div><span className="eyebrow">SHARED LIVE PROCESSING</span><h2 id="recognition-title">Vehicle recognition</h2></div>
      <span className={`pill ${fresh ? 'good' : status?.status === 'connecting' ? 'warn' : 'neutral'}`}><Activity size={13} />{sessionLabel}</span>
    </div>
    <div className="recognition-body">
      <div className="recognition-toolbar"><div className="recognition-select"><label htmlFor="recognition-camera">Camera to analyse</label>
        <select id="recognition-camera" value={selected} onChange={event => setSelected(event.target.value)} disabled={busy || !catalogue?.available}>
          <option value="">Select a live camera</option>
          {catalogue?.cameras.map(camera => <option key={camera.id} value={camera.id}>{camera.id.toUpperCase()} · {camera.name}</option>)}
        </select></div>
        <button className="button primary" disabled={busy || !selected || !catalogue?.available} onClick={() => void changeSession(selected)}>
          {busy ? <RefreshCw size={15} className="spinning" /> : <Play size={15} />}{running && selected !== status?.camera_id ? 'Switch analysis' : 'Start analysis'}</button>
        <button className="button ghost" disabled={busy || !running} onClick={() => void changeSession(null)}><Pause size={15} />Pause</button>
        <a className="button ghost recognition-console-link" href="http://127.0.0.1:7780/" target="_blank" rel="noopener noreferrer"><ExternalLink size={15} />Engineering console</a>
      </div>
      <p className="recognition-session-note"><Camera size={14} />{status?.camera_id
        ? <>Analysing <strong>{status.camera_id.toUpperCase()}{cameraName ? ` · ${cameraName}` : ''}</strong>. Both interfaces observe this same session.</>
        : <>Select a camera and start analysis explicitly. Opening this page does not start the models.</>}</p>
      {catalogue?.scope === 'configured_relays' && <p className="recognition-session-note" role="status">Configured relays only · full Sentinel catalogue unavailable.</p>}
      {error && <div className="recognition-error" role="alert"><CircleAlert size={16} /><span>{error}</span></div>}
      {catalogueError && !error && <div className="recognition-error" role="alert"><CircleAlert size={16} /><span>Camera selection unavailable: {catalogueError}</span></div>}
      {catalogue && !catalogue.available && !error && <div className="recognition-error" role="status"><CircleAlert size={16} /><span>Live analysis is unavailable. Check the local processing service and authorised camera access.</span></div>}
      <div className="recognition-metrics">
        <div><span>Vehicles in current view</span><strong>{fresh ? recognitionCount(status?.vehicle_boxes) : '—'}</strong><small>Detected boxes, not total traffic</small></div>
        <div><span>Plate crops examined</span><strong>{status ? recognitionCount(status.plate_crops) : '—'}</strong><small>Processing observations this session</small></div>
        <div><span>Model-accepted readings</span><strong>{status ? recognitionCount(status.unique_model_accepted_unverified) : '—'}</strong><small>Not independently verified</small></div>
      </div>
      {tracks.length ? <div className="table-scroll"><table className="data-table recognition-table"><thead><tr><th>Vehicle track</th><th>Reading</th><th>State</th><th>Frames supporting OCR</th></tr></thead><tbody>
        {tracks.map(track => <tr key={track.track_id}><td className="mono">{track.track_id}</td><td className="mono">{track.masked_plate || 'No complete reading'}</td>
          <td><span className={`pill ${recognitionTone(track.recognition_state)}`}>{recognitionLabel(track.recognition_state)}</span></td><td>{recognitionCount(track.observations)}</td></tr>)}
      </tbody></table></div> : <div className="recognition-empty"><ShieldCheck size={20} /><div><strong>{status?.status === 'connecting' ? 'Connecting to the camera' : running && !fresh ? 'Waiting for a fresh observation' : fresh ? 'No active plate-bearing tracks' : 'Ready when you are'}</strong>
        <span>{fresh ? 'Recognition evidence will appear as vehicles are processed.' : 'The existing live preview remains separate. Aligned boxes and crops are available in the engineering console.'}</span></div></div>}
      <div className="recognition-footnote"><ShieldCheck size={14} /><p>Live source · Plates are masked here. Uncertain results are not confirmed alerts; model scores are not calibrated probabilities.
        {status?.delivery && <> {status.delivery.enabled ? `${recognitionCount(status.delivery.outbox_depth)} queued event(s) awaiting delivery.` : 'Diagnostic analysis only; central event delivery is disabled.'}</>}</p></div>
    </div>
  </section>;
}
