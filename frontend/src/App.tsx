import { useCallback, useEffect, useState } from 'react';
import { Activity, ArrowUpRight, Check, CheckCheck, ChevronRight, Film, Pause, Play, RefreshCw, Settings, ShieldAlert, Square, Wifi, WifiOff } from 'lucide-react';
import { changeAlertStatus, ensureLocalSession, formatTime, getAlerts, isActionable, wsUrl, type Alert, type AlertStatus, type RecognitionStatus } from './api';
import { failureMessage, formatVideoTime, recognitionIsFresh, recognitionIsRunning, recognitionLabel, sessionLabel } from './recognition';
import { useRecognitionSession } from './useRecognitionSession';
import Inspector, { imageData } from './Inspector';
import DetailSheet from './DetailSheet';
import WatchlistPage from './WatchlistPage';

type Page = 'analyse' | 'alerts' | 'watchlist';
const title = (value: string) => value.replaceAll('_', ' ').replace(/^./, c => c.toUpperCase());

function AlertsPage({ alerts, error, refresh }: { alerts: Alert[]; error: string; refresh: () => Promise<void> }) {
  const [filter, setFilter] = useState('new');
  const [selected, setSelected] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState('');
  const [notice, setNotice] = useState('');
  const alert = alerts.find(row => row.id === selected);
  const filtered = alerts.filter(row => filter === 'all' || row.status === filter);
  async function update(status: AlertStatus) {
    if (!alert) return; setBusy(true); setActionError('');
    try { await changeAlertStatus(alert.id, status); await refresh(); setNotice(`Alert ${title(status).toLowerCase()}.`); }
    catch (failure) { setActionError((failure as Error).message); }
    finally { setBusy(false); }
  }
  return <><div className="page-heading"><div><h1>Alerts</h1><p>Review central watchlist matches and record a decision.</p></div><button className="button" onClick={() => void refresh()}><RefreshCw size={15} /> Refresh</button></div>
    {error && <div className="notice error" role="alert">{error} Available results may be outdated.</div>}{notice && <div className="notice success" role="status">{notice}</div>}
    <div className="filter-tabs" aria-label="Filter alert status">{['new', 'acknowledged', 'resolved', 'false_positive', 'all'].map(value => <button key={value} aria-pressed={filter === value} className={filter === value ? 'active' : ''} onClick={() => setFilter(value)}>{title(value)} <span>{value === 'all' ? alerts.length : alerts.filter(row => row.status === value).length}</span></button>)}</div>
    <section className="surface">{!filtered.length ? <div className="empty"><ShieldAlert size={28} /><strong>No {filter === 'all' ? '' : title(filter).toLowerCase()} alerts</strong><p>Accepted sightings that match the watchlist appear here. Uncertain guesses do not create confirmed alerts.</p></div> : <div className="alert-list">{filtered.map(row => <button className="alert-row" key={row.id} onClick={() => { setSelected(row.id); setActionError(''); }}><span className={`alert-marker ${row.status === 'new' ? 'danger' : ''}`}><ShieldAlert size={18} /></span><span className="alert-main"><strong className="mono">{row.masked_plate}</strong><span>{row.reason}</span><small>{row.source_mode === 'recorded' ? `Recorded · video ${formatVideoTime(row.stream_pts_ms)}` : row.source_mode === 'synthetic' ? 'Simulated demonstration' : 'Live source metadata'} · {row.camera_id}</small></span><span className="alert-meta"><span className={`status ${row.status === 'new' ? 'danger' : 'neutral'}`}>{title(row.status)}</span><small>{formatTime(row.processed_at || row.created_at)}</small></span><ChevronRight size={17} /></button>)}</div>}</section>
    {alert && <DetailSheet title={alert.masked_plate} subtitle={`Watchlist alert #${alert.id}`} onClose={() => setSelected(null)}>
      <div className="notice danger"><strong>{alert.reason}</strong><p>{title(alert.severity)} priority · {title(alert.status)}</p></div>{actionError && <div className="notice error" role="alert">{actionError}</div>}
      <dl className="facts"><div><dt>Source</dt><dd>{alert.camera_id}</dd></div><div><dt>Source type</dt><dd>{alert.source_mode === 'recorded' ? 'Recorded footage' : alert.source_mode === 'synthetic' ? 'Simulated event' : 'Live metadata'}</dd></div>{alert.source_mode === 'recorded' && <div><dt>Video position</dt><dd>{formatVideoTime(alert.stream_pts_ms)}</dd></div>}<div><dt>Processing time</dt><dd>{formatTime(alert.processed_at || alert.created_at)}</dd></div><div><dt>Observed time basis</dt><dd>{alert.time_basis === 'source_clock' ? 'Recording clock' : 'Processing time; capture time not established'}</dd></div><div><dt>Analysis run</dt><dd className="mono">{alert.session_id || 'Not provided'}</dd></div></dl>
      <div className="row-actions">{alert.status === 'new' && <button className="button primary" disabled={busy} onClick={() => void update('acknowledged')}><Check size={16} /> Acknowledge</button>}{alert.status === 'acknowledged' && <button className="button primary" disabled={busy} onClick={() => void update('resolved')}><CheckCheck size={16} /> Resolve</button>}{isActionable(alert) && <button className="button quiet" disabled={busy} onClick={() => void update('false_positive')}>Mark false positive</button>}</div>
      <section className="inspection-section"><h3>Decision history</h3>{alert.audit?.length ? alert.audit.map(item => <div className="audit-entry" key={item.id}><span>{title(item.to_status)} · {item.actor || 'local operator'}</span><small>{formatTime(item.changed_at)}</small></div>) : <p className="muted">No operator decisions yet.</p>}</section><p className="caption">Acknowledging an alert is an operator action, not independent verification of the plate reading.</p>
    </DetailSheet>}
  </>;
}

function ReadingStatus({ state, observations }: { state: string; observations: number }) {
  return <span className={`status ${state === 'uncertain' ? 'warning' : 'neutral'}`}>{observations ? recognitionLabel(state) : 'No plate yet'}</span>;
}

export default function App() {
  const [page, setPage] = useState<Page>('analyse');
  const [ready, setReady] = useState(false);
  const [bootError, setBootError] = useState('');
  const [attempt, setAttempt] = useState(0);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [alertError, setAlertError] = useState('');
  const [liveUpdates, setLiveUpdates] = useState(false);
  const [selectedSource, setSelectedSource] = useState('');
  const [inspect, setInspect] = useState<{ sessionId: string; trackId: string } | null>(null);
  const session = useRecognitionSession(ready);

  useEffect(() => { let cancelled = false; setBootError(''); setReady(false); ensureLocalSession().then(() => { if (!cancelled) setReady(true); }).catch(failure => { if (!cancelled) setBootError((failure as Error).message); }); return () => { cancelled = true; }; }, [attempt]);
  const refreshAlerts = useCallback(async () => {
    try { setAlerts(await getAlerts()); setAlertError(''); }
    catch (failure) { setAlertError((failure as Error).message); }
  }, []);
  useEffect(() => {
    if (!ready) return;
    let cancelled = false, socket: WebSocket | null = null, reconnect: number | undefined;
    void refreshAlerts(); const timer = window.setInterval(() => void refreshAlerts(), 10_000);
    function connect() {
      if (cancelled) return;
      socket = new WebSocket(wsUrl());
      socket.onopen = () => { if (!cancelled) setLiveUpdates(true); };
      socket.onmessage = () => void refreshAlerts();
      socket.onclose = () => { if (!cancelled) { setLiveUpdates(false); reconnect = window.setTimeout(connect, 4000); } };
      socket.onerror = () => socket?.close();
    }
    connect(); return () => { cancelled = true; window.clearInterval(timer); if (reconnect) window.clearTimeout(reconnect); socket?.close(); };
  }, [ready, refreshAlerts]);
  useEffect(() => { if (!selectedSource && session.recordings.length) setSelectedSource(session.recordings[0].id); }, [selectedSource, session.recordings]);
  useEffect(() => { setInspect(null); }, [session.status?.session_id, session.status?.camera_id]);

  const status = session.status;
  const fresh = recognitionIsFresh(status) && !session.error;
  const rows = fresh && status?.source_mode === 'recorded' ? status.vehicles_seen || [] : [];
  const matchingAlerts = status?.session_id ? alerts.filter(alert => alert.session_id === status.session_id && alert.source_mode === 'recorded') : [];
  const activeAlerts = alerts.filter(isActionable).length;
  const running = recognitionIsRunning(status);
  const evidence = fresh && session.evidence?.session_id === status?.session_id ? session.evidence : null;
  const frame = imageData(evidence?.frame, 'jpeg');
  const position = evidence?.video_position_ms ?? status?.video_position_ms;
  const duration = status?.video_duration_ms;
  const progress = typeof position === 'number' && typeof duration === 'number' && duration > 0 ? Math.min(100, Math.max(0, position / duration * 100)) : 0;
  const device = status?.inference_device?.toUpperCase();
  async function start() {
    if (!selectedSource) return;
    if (status?.camera_id && status.camera_id !== selectedSource && !window.confirm('Stop the current source and start the selected recording?')) return;
    setInspect(null); await session.start(selectedSource);
  }
  const deviceNotice = (current: RecognitionStatus | null) => current?.model_profile ? `${current.model_profile}${device ? ` · ${device} OCR` : ''}` : 'Runtime not loaded';

  return <div className="app-shell">
    <header className="app-header"><button className="brand" aria-label="Sentinel home" onClick={() => setPage('analyse')}><span className="brand-symbol"><Activity size={20} strokeWidth={1.6} /></span>Sentinel</button><nav aria-label="Main navigation">{(['analyse', 'alerts', 'watchlist'] as Page[]).map(value => <button key={value} aria-current={page === value ? 'page' : undefined} className={page === value ? 'active' : ''} onClick={() => setPage(value)}>{title(value)}{value === 'alerts' && activeAlerts > 0 && <span className="nav-count">{activeAlerts}</span>}</button>)}</nav><div className="header-meta"><span className={`connection ${ready && !alertError ? '' : 'disconnected'}`}>{ready && !alertError ? <Wifi size={14} /> : <WifiOff size={14} />}{!ready ? 'Connecting' : alertError ? 'Reconnecting' : liveUpdates ? 'Service connected' : 'Connected · polling'}</span><details className="settings"><summary aria-label="Application settings"><Settings size={18} /></summary><div className="settings-panel"><strong>Local single-user mode</strong><p>One shared analysis session. No operator key is stored in this browser.</p><p>Recordings are processed only after you start analysis. Technical runtime details are in the Inspector.</p></div></details></div></header>
    <main className="app-content">
      {!ready ? <div className="setup-state"><Activity size={28} /><h1>{bootError ? 'Local service unavailable' : 'Connecting to Sentinel'}</h1><p>{bootError || 'Preparing the local dashboard session…'}</p>{bootError && <button className="button primary" onClick={() => setAttempt(attempt + 1)}><RefreshCw size={16} /> Retry connection</button>}</div> : <>
        {page !== 'alerts' && alertError && <div className="notice warning" role="alert">Alert updates unavailable: {alertError} Analysis may still be available; central matches cannot be refreshed.</div>}
        <div hidden={page !== 'analyse'}>
          <div className="page-heading"><div><h1>Analyse footage</h1><p>Recorded CCTV · one shared recognition session</p></div><span className="runtime-label">{deviceNotice(status)}</span></div>
          <section className="analysis-toolbar" aria-label="Recording controls"><div className="source-picker"><label htmlFor="recording-source">Recording</label><select id="recording-source" value={selectedSource} onChange={event => setSelectedSource(event.target.value)} disabled={session.busy || !session.recordings.length}>{!session.recordings.length && <option value="">No registered recordings</option>}{session.recordings.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select></div><div className="recording-actions"><button className="button primary" onClick={() => void start()} disabled={!selectedSource || session.busy || (status?.camera_id === selectedSource && status?.status === 'processing')}><Play size={15} />{session.busy ? 'Working…' : status?.camera_id === selectedSource && status?.status === 'completed' ? 'Analyse again' : 'Start analysis'}</button>{status?.camera_id && status.status === 'paused' ? <button className="button" onClick={() => void session.control('resume')} disabled={session.busy}><Play size={15} /> Resume</button> : <button className="button" onClick={() => void session.control('pause')} disabled={session.busy || !running || status?.status === 'connecting'}><Pause size={15} /> Pause</button>}<button className="button quiet" onClick={() => { setInspect(null); void session.stop(); }} disabled={!status?.camera_id || session.busy}><Square size={14} /> Stop</button></div></section>
          {session.recordingError && <div className="notice warning" role="alert">Recording list unavailable: {session.recordingError} <button className="text-button" onClick={() => void session.refresh()}>Retry</button></div>}
          {session.error && <div className="notice error" role="alert">{session.error} <button className="text-button" onClick={() => void session.refresh()}>Retry</button></div>}
          {status?.failure_code && <div className="notice error" role="alert">{failureMessage(status.failure_code)}</div>}
          {!session.recordingError && !session.recordings.length && <div className="notice">No permitted recordings are registered. Add the video and its permission receipt using the recording-registration tool, then refresh.</div>}
          <div className="analysis-workspace"><section className="frame-column"><div className="section-heading"><h2>Analysed frame</h2><span className="status neutral">{sessionLabel(status)}</span></div><div className="display-controls" role="group" aria-label="Frame display"><button className={session.mode === 'overlays' ? 'active' : ''} aria-pressed={session.mode === 'overlays'} onClick={() => session.setMode('overlays')}>Detection overlays</button><button className={session.mode === 'original' ? 'active' : ''} aria-pressed={session.mode === 'original'} onClick={() => session.setMode('original')}>Original frame</button></div>
            <div className="frame-viewport">{frame ? <img src={frame} alt={`Analysed recorded frame at video ${formatVideoTime(position)}${session.mode === 'overlays' ? ', with aligned detection boxes' : ', original without overlays'}`} /> : <div className="frame-empty"><Film size={32} /><strong>{status?.status === 'connecting' ? 'Loading recording and models' : status?.camera_id ? session.evidenceError ? 'Frame unavailable' : 'Waiting for analysis evidence' : 'Select a recording to begin'}</strong><span>{session.evidenceError || (status?.camera_id ? 'Recorded evidence may be disabled, unavailable or still processing.' : 'Previewing this page does not start recognition.')}</span></div>}</div>
            <div className="video-progress" role="progressbar" aria-label="Analysed video position" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(progress)}><span style={{ width: `${progress}%` }} /></div><div className="frame-meta"><span className="mono">{formatVideoTime(position)} / {formatVideoTime(duration)}</span><span>Analysed samples · not guaranteed real-time playback</span></div>
          </section><section className="surface reading-panel"><div className="section-heading padded"><h2>Recent readings</h2><span className="caption">{rows.length} tracks shown</span></div>{!rows.length ? <div className="empty"><strong>{status?.camera_id ? 'No fresh vehicle results' : 'No analysis started'}</strong><p>{status?.camera_id ? 'Vehicle and plate evidence will appear as frames are processed.' : 'Choose a permitted recording and start analysis.'}</p></div> : <div className="reading-list">{rows.slice(0, 200).map(row => {
            const visual = evidence?.vehicles.find(item => item.track_id === row.track_id);
            return <button className="reading-row" key={row.track_id} onClick={() => status?.session_id && setInspect({ sessionId: status.session_id, trackId: row.track_id })} disabled={!status?.session_id}><div className="reading-line"><strong className="plate-text">{visual?.reading || row.masked_plate || 'No complete reading'}</strong><ArrowUpRight size={15} /></div><div className="reading-line"><span className="caption">{title(row.vehicle_class)} · track {row.track_id}</span><span className="caption mono">{formatVideoTime(row.last_ms)}</span></div><div className="reading-status"><ReadingStatus state={row.state} observations={row.observations} />{visual?.reading && row.state !== 'model_accepted' && <span className="caption">Unverified suggestion</span>}</div></button>;
          })}</div>}<div className="reading-footer">Bounded recent model tracks; counts are not independently verified vehicles. Select a row to inspect evidence.</div></section></div>
          <div className="run-summary"><span><strong>{rows.length}</strong> recent tracks</span><span><strong>{status?.unique_model_accepted_unverified ?? 0}</strong> model-accepted readings</span><span><strong>{rows.filter(row => row.state === 'uncertain').length}</strong> recent uncertain</span><button className="text-button" onClick={() => setPage('alerts')}><strong>{matchingAlerts.length}</strong> central matches this run</button><span className="summary-note">Accepted by model ≠ independently verified</span></div>
        </div>
        {page === 'alerts' && <AlertsPage alerts={alerts} error={alertError} refresh={refreshAlerts} />}
        {page === 'watchlist' && <WatchlistPage />}
      </>}
    </main><footer className="app-footer"><span>Sentinel · local vehicle intelligence</span><span>Recorded footage is not live camera evidence.</span></footer>
    {inspect && <Inspector sessionId={inspect.sessionId} trackId={inspect.trackId} onClose={() => setInspect(null)} />}
  </div>;
}
