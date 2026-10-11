import { useEffect, useRef, useState } from 'react';
import { Activity, Camera, CircleAlert, ExternalLink, Film, Pause, Play, RefreshCw, RotateCcw, ShieldCheck, Square } from 'lucide-react';
import {
  controlRecognition, getRecognitionCameras, getRecognitionEvidence, getRecognitionRecordings, getRecognitionStatus,
  heartbeatRecognition, setRecognitionSession,
  type PlaybackAction, type RecognitionCameras, type RecognitionEvidence, type RecognitionRecordings, type RecognitionStatus,
} from './api';
import {
  failureMessage, formatVideoTime, isRecorded, recognitionCount, recognitionIsFresh, recognitionIsRunning,
  recognitionLabel, recognitionTone, sessionLabel,
} from './recognition';

/** Both views observe one explicitly started session; mounting this panel never starts inference. */
export default function RecognitionPanel({ operatorKey }: { operatorKey: string }) {
  const [status, setStatus] = useState<RecognitionStatus | null>(null);
  const [catalogue, setCatalogue] = useState<RecognitionCameras | null>(null);
  const [recordings, setRecordings] = useState<RecognitionRecordings['recordings']>([]);
  const [selected, setSelected] = useState('');
  const [error, setError] = useState('');
  const [catalogueError, setCatalogueError] = useState('');
  const [busy, setBusy] = useState(false);
  const [seekTo, setSeekTo] = useState<number | null>(null);
  const [evidence, setEvidence] = useState<RecognitionEvidence | null>(null);
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
    setStatus(null); setCatalogue(null); setRecordings([]); setError(''); setCatalogueError('');
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
      }
      // Recordings are independent of camera access: list them even when the catalogue is down.
      try {
        const next = await getRecognitionRecordings();
        if (!cancelled && document.visibilityState === 'visible') setRecordings(next.recordings);
      } catch { if (!cancelled) setRecordings([]); }
      finally { loadingCameras = false; }
    }
    async function poll() {
      if (cancelled || polling || changing.current || document.visibilityState !== 'visible') return;
      polling = true;
      const before = generation.current;
      try {
        const next = await getRecognitionStatus();
        if (!cancelled && before === generation.current && document.visibilityState === 'visible') {
          active.current = next; setStatus(next); setError('');
          // Permitted recordings only, and only when switched on: the analysed frame and full readings.
          const view = next.source_mode === 'recorded' && next.camera_id ? await getRecognitionEvidence() : null;
          if (!cancelled && before === generation.current) setEvidence(view);
        }
      } catch (failure) {
        if (!cancelled && before === generation.current) {
          active.current = null; setStatus(null); setEvidence(null); setError((failure as Error).message);
        }
      } finally { polling = false; }
    }
    async function heartbeat() {
      const current = active.current;
      if (cancelled || heartbeating || changing.current || document.visibilityState !== 'visible' || !current?.camera_id
        || !recognitionIsRunning(current)) return;
      heartbeating = true;
      try { await heartbeatRecognition(current.camera_id); }
      catch { if (!cancelled) { active.current = null; setStatus(null); void poll(); } }
      finally { heartbeating = false; }
    }
    function visibility() {
      if (document.visibilityState === 'visible') { void poll(); void cameras(); }
      else { generation.current += 1; active.current = null; setStatus(null); setEvidence(null); }
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

  async function apply(change: () => Promise<unknown>) {
    changing.current = true;
    setBusy(true); setError(''); setEvidence(null);
    generation.current += 1;
    // A pending poll must not overwrite an explicit session change.
    const before = generation.current;
    try {
      await change();
      const next = await getRecognitionStatus();
      if (mounted.current && before === generation.current) { active.current = next; setStatus(next); }
    } catch (failure) {
      if (mounted.current && before === generation.current) {
        active.current = null; setStatus(null); setError((failure as Error).message);
      }
    } finally { changing.current = false; if (mounted.current) { setBusy(false); setSeekTo(null); } }
  }
  const changeSession = (sourceId: string | null) => apply(() => setRecognitionSession(sourceId));
  const playback = (action: PlaybackAction, positionMs?: number) => apply(() => controlRecognition(action, positionMs));

  const recorded = isRecorded(status);
  const fresh = recognitionIsFresh(status);
  const running = recognitionIsRunning(status);
  const sourceName = recorded ? status?.recording_name
    : catalogue?.cameras.find(camera => camera.id === status?.camera_id)?.name;
  const tracks = fresh ? status?.tracks || [] : [];
  const selectable = !!catalogue?.available || recordings.length > 0;
  const selectedIsRecording = recordings.some(recording => recording.id === selected);
  const duration = status?.video_duration_ms;
  const failure = status?.status === 'unavailable' ? failureMessage(status.failure_code) : '';

  return <section className="panel recognition-panel" aria-labelledby="recognition-title">
    <div className="panel-heading"><div><span className="eyebrow">{recorded ? 'SHARED RECORDED PROCESSING' : 'SHARED LIVE PROCESSING'}</span><h2 id="recognition-title">Vehicle recognition</h2></div>
      <span className={`pill ${fresh && status?.status !== 'paused' && status?.status !== 'completed' ? 'good' : status?.status === 'connecting' ? 'warn' : 'neutral'}`}><Activity size={13} />{sessionLabel(status)}</span>
    </div>
    <div className="recognition-body">
      <div className="recognition-toolbar"><div className="recognition-select"><label htmlFor="recognition-camera">Source to analyse</label>
        <select id="recognition-camera" value={selected} onChange={event => setSelected(event.target.value)} disabled={busy || !selectable}>
          <option value="">Select a camera or recording</option>
          {!!catalogue?.cameras.length && <optgroup label="Live cameras">
            {catalogue.cameras.map(camera => <option key={camera.id} value={camera.id}>{camera.id.toUpperCase()} · {camera.name}</option>)}</optgroup>}
          {!!recordings.length && <optgroup label="Recorded footage">
            {recordings.map(recording => <option key={recording.id} value={recording.id}>RECORDED · {recording.name} · {formatVideoTime(recording.duration_ms)}</option>)}</optgroup>}
        </select></div>
        <button className="button primary" disabled={busy || !selected || !selectable} onClick={() => void changeSession(selected)}>
          {busy ? <RefreshCw size={15} className="spinning" /> : <Play size={15} />}{running && selected !== status?.camera_id ? 'Switch analysis' : selectedIsRecording ? 'Analyse recording' : 'Start analysis'}</button>
        {recorded && running ? <>
          {status?.status === 'paused'
            ? <button className="button ghost" disabled={busy} onClick={() => void playback('resume')}><Play size={15} />Resume</button>
            : <button className="button ghost" disabled={busy || status?.status !== 'processing'} onClick={() => void playback('pause')}><Pause size={15} />Pause</button>}
          <button className="button ghost" disabled={busy} onClick={() => void playback('restart')}><RotateCcw size={15} />Restart</button>
          <button className="button ghost" disabled={busy} onClick={() => void changeSession(null)}><Square size={15} />Stop</button>
        </> : <button className="button ghost" disabled={busy || !running} onClick={() => void changeSession(null)}><Pause size={15} />Pause</button>}
        <a className="button ghost recognition-console-link" href="http://127.0.0.1:7780/" target="_blank" rel="noopener noreferrer"><ExternalLink size={15} />Engineering console</a>
      </div>
      <p className="recognition-session-note">{recorded ? <Film size={14} /> : <Camera size={14} />}{status?.camera_id
        ? <>{recorded && <span className="source-tag recorded">RECORDED</span>} Analysing <strong>{recorded ? sourceName || status.camera_id : `${status.camera_id.toUpperCase()}${sourceName ? ` · ${sourceName}` : ''}`}</strong>. Both interfaces observe this same session.</>
        : <>Select a source and start analysis explicitly. Opening this page does not start the models.</>}</p>
      {recorded && running && <div className="recognition-playback">
        <span className="mono">{formatVideoTime(seekTo ?? status?.video_position_ms)} / {formatVideoTime(duration)}</span>
        {typeof duration === 'number' && duration > 0 && <input type="range" aria-label="Position in the recording" min={0} max={duration} step={500}
          value={seekTo ?? status?.video_position_ms ?? 0} disabled={busy}
          onChange={event => setSeekTo(Number(event.target.value))}
          onPointerUp={() => { if (seekTo !== null) void playback('seek', seekTo); }}
          onKeyUp={event => { if (seekTo !== null && ['ArrowLeft', 'ArrowRight', 'Home', 'End', 'PageUp', 'PageDown'].includes(event.key)) void playback('seek', seekTo); }} />}
        <small>{status?.pacing === 'unpaced' ? 'Processed as fast as this machine allows' : 'Played at no more than real time'} · every sampled frame is analysed · moving the position starts a new run{status?.inference_device ? ` · ${status.inference_device.toUpperCase()} inference` : ''}</small>
      </div>}
      {catalogue?.scope === 'configured_relays' && <p className="recognition-session-note" role="status">Configured relays only · full Sentinel catalogue unavailable.</p>}
      {error && <div className="recognition-error" role="alert"><CircleAlert size={16} /><span>{error}</span></div>}
      {failure && !error && <div className="recognition-error" role="alert"><CircleAlert size={16} /><span>{failure}</span></div>}
      {catalogueError && !error && !recordings.length && <div className="recognition-error" role="alert"><CircleAlert size={16} /><span>Source selection unavailable: {catalogueError}</span></div>}
      {catalogue && !catalogue.available && !error && !recordings.length && <div className="recognition-error" role="status"><CircleAlert size={16} /><span>Live analysis is unavailable. Check the local processing service and authorised camera access.</span></div>}
      {catalogue && !catalogue.available && !error && !!recordings.length && !recorded && <p className="recognition-session-note" role="status">No live cameras available · recorded footage can still be analysed.</p>}
      <div className="recognition-metrics">
        <div><span>{recorded ? 'Vehicles at this position' : 'Vehicles in current view'}</span><strong>{fresh ? recognitionCount(status?.vehicle_boxes) : '—'}</strong><small>Detected boxes, not total traffic</small></div>
        <div><span>Plate crops examined</span><strong>{status ? recognitionCount(status.plate_crops) : '—'}</strong><small>Processing observations this session</small></div>
        <div><span>Model-accepted readings</span><strong>{status ? recognitionCount(status.unique_model_accepted_unverified) : '—'}</strong><small>Not independently verified</small></div>
      </div>
      {tracks.length ? <div className="table-scroll"><table className="data-table recognition-table"><thead><tr><th>Vehicle track</th><th>Reading</th><th>State</th><th>Frames supporting OCR</th></tr></thead><tbody>
        {tracks.map(track => <tr key={track.track_id}><td className="mono">{track.track_id}</td><td className="mono">{track.masked_plate || 'No complete reading'}</td>
          <td><span className={`pill ${recognitionTone(track.recognition_state)}`}>{recognitionLabel(track.recognition_state)}</span></td><td>{recognitionCount(track.observations)}</td></tr>)}
      </tbody></table></div> : <div className="recognition-empty"><ShieldCheck size={20} /><div><strong>{status?.status === 'connecting' ? (recorded ? 'Opening the recording' : 'Connecting to the camera') : status?.status === 'completed' ? 'Recording finished' : running && !fresh ? 'Waiting for a fresh observation' : fresh ? 'No active plate-bearing tracks' : 'Ready when you are'}</strong>
        <span>{fresh ? 'Recognition evidence will appear as vehicles are processed.' : 'The existing live preview remains separate. Aligned boxes and crops are available in the engineering console.'}</span></div></div>}
      {evidence && recorded && <div className="recognition-evidence">
        <figure>{evidence.frame
          ? <img src={`data:image/jpeg;base64,${evidence.frame}`} alt="Analysed frame from the recording with vehicle boxes and plate call-outs" />
          : <div className="recognition-evidence-wait">Waiting for the next analysed frame</div>}
          <figcaption><span className="source-tag recorded">RECORDED</span> Analysed frame at video {formatVideoTime(evidence.video_position_ms)} · updates as each sample is analysed</figcaption></figure>
        <div className="vehicles-seen" aria-live="polite">
          <div className="vehicles-seen-head"><strong>Vehicles detected this run</strong><span>{evidence.vehicles.length} listed · newest first</span></div>
          {evidence.vehicles.length ? <ul>{evidence.vehicles.map(vehicle => <li key={`${evidence.session_id || ''}-${vehicle.track_id}-${vehicle.first_ms}`}
            className={vehicle.watchlist === 'possible_match' ? 'match' : vehicle.state === 'model_accepted' ? 'accepted' : vehicle.state === 'uncertain' ? 'uncertain' : ''}>
            <span className={`plate-chip ${vehicle.reading ? '' : 'none'}`}><i>IND</i><b>{vehicle.reading || 'no reading'}</b></span>
            <span className={`pill ${vehicle.watchlist === 'possible_match' ? 'danger' : recognitionTone(vehicle.state)}`}>{vehicle.watchlist === 'possible_match' ? 'Watchlist · unverified' : vehicle.observations ? recognitionLabel(vehicle.state) : 'No plate yet'}</span>
            <small>{vehicle.vehicle_class.toUpperCase()}{typeof vehicle.confidence === 'number' ? ` ${Math.round(vehicle.confidence * 100)}%` : ''} · track {vehicle.track_id} · {recognitionCount(vehicle.observations)} plate frame(s)</small>
            <small>video {formatVideoTime(vehicle.first_ms)} – {formatVideoTime(vehicle.last_ms)}</small>
            {vehicle.thumb && <img src={`data:image/png;base64,${vehicle.thumb}`} alt={`Plate crop for track ${vehicle.track_id}`} />}
          </li>)}</ul> : <p>No vehicle has been tracked yet in this run.</p>}
          <p>Readings are the model's current best guess and are not verified; one marked Reading comes from fewer than three frames. The percentage is the vehicle detector's score for the box. Shown for recorded footage only.</p>
        </div>
      </div>}
      {!evidence && !!status?.vehicles_seen?.length && <div className="vehicles-seen" aria-live="polite">
        <div className="vehicles-seen-head"><strong>Vehicles detected this run</strong><span>{status.vehicles_seen.length} listed · newest first</span></div>
        <ul>{status.vehicles_seen.map(vehicle => <li key={`${status.session_id || ''}-${vehicle.track_id}-${vehicle.first_ms}`}
          className={vehicle.watchlist === 'possible_match' ? 'match' : vehicle.state === 'model_accepted' ? 'accepted' : vehicle.state === 'uncertain' ? 'uncertain' : ''}>
          <span className={`plate-chip ${vehicle.masked_plate ? '' : 'none'}`}><i>IND</i><b>{vehicle.masked_plate || 'no reading'}</b></span>
          <span className={`pill ${vehicle.watchlist === 'possible_match' ? 'danger' : recognitionTone(vehicle.state)}`}>{vehicle.watchlist === 'possible_match' ? 'Watchlist · see alerts' : vehicle.observations ? recognitionLabel(vehicle.state) : 'No plate yet'}</span>
          <small>{vehicle.vehicle_class.toUpperCase()}{typeof vehicle.confidence === 'number' ? ` ${Math.round(vehicle.confidence * 100)}%` : ''} · track {vehicle.track_id} · {recognitionCount(vehicle.observations)} plate frame(s)</small>
          <small>{recorded ? `video ${formatVideoTime(vehicle.first_ms)} – ${formatVideoTime(vehicle.last_ms)}` : ''}</small>
        </li>)}</ul>
        <p>The percentage is the vehicle detector's score for the box, not the chance a reading is right. Plates are masked here; full text and crops stay in the engineering console.</p>
      </div>}
      <div className="recognition-footnote"><ShieldCheck size={14} /><p>{recorded ? 'Recorded footage, not a live camera' : 'Live source'} · {evidence ? 'Full unverified readings and the analysed frame are shown for this permitted recording only; live cameras stay masked.' : 'Plates are masked here.'} Uncertain results are not confirmed alerts; model scores are not calibrated probabilities.
        {recorded && status?.time_basis === 'processing_time' && <> The recording&rsquo;s real time is unknown, so its events carry the time of analysis.</>}
        {status?.delivery && <> {status.delivery.enabled ? `${recognitionCount(status.delivery.outbox_depth)} queued event(s) awaiting delivery.` : 'Diagnostic analysis only; central event delivery is disabled.'}</>}</p></div>
    </div>
  </section>;
}
