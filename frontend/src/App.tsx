import { useCallback, useEffect, useMemo, useState, type FormEvent } from 'react';
import {
  Activity, ArrowDownRight, ArrowRight, BellRing, Camera as CameraIcon, Check,
  CheckCircle2, ChevronRight, CircleAlert, Clock3, FileText, KeyRound,
  LayoutDashboard, MapPinned, Play, Printer, Radio, RefreshCw, Route,
  Search, ShieldCheck, TriangleAlert, Wifi, WifiOff, X,
} from 'lucide-react';
import {
  changeAlertStatus, formatTime, getAlerts, getCameras, getHourlyReports,
  getIncidentReport, getOperatorKey, injectDemo, isActionable, searchRoute,
  setOperatorKey, wsUrl,
  type Alert, type AlertStatus, type Camera, type HourlyReport,
  type IncidentReport, type Sighting,
} from './api';
import MapPanel from './MapPanel';
import LivePlayer from './LivePlayer';

type Page = 'overview' | 'alerts' | 'cameras' | 'routes' | 'reports';
type Connection = 'connected' | 'reconnecting' | 'locked';

const navigation: { id: Page; label: string; icon: typeof LayoutDashboard }[] = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'alerts', label: 'Alert queue', icon: BellRing },
  { id: 'cameras', label: 'Cameras', icon: CameraIcon },
  { id: 'routes', label: 'Route search', icon: Route },
  { id: 'reports', label: 'Reports', icon: FileText },
];

function titleCase(value?: string | null): string {
  return (value || 'unknown').replaceAll('_', ' ').replace(/\b\w/g, match => match.toUpperCase());
}

function statusClass(value?: string | null): string {
  const clean = (value || 'unknown').toLowerCase();
  if (['online', 'healthy', 'connected', 'live'].includes(clean)) return 'good';
  if (['degraded', 'reconnecting', 'acknowledged', 'unscheduled'].includes(clean)) return 'warn';
  if (['offline', 'new', 'critical'].includes(clean)) return 'danger';
  return 'neutral';
}

function alertSeverity(alert: Alert): string {
  return (alert.severity || 'watchlist').toLowerCase();
}

function useControlRoom(operatorKey: string) {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [reports, setReports] = useState<HourlyReport[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null);

  const refresh = useCallback(async (showLoading = false) => {
    if (!getOperatorKey()) {
      setCameras([]); setAlerts([]); setReports([]); setLoading(false); setError('');
      return;
    }
    if (showLoading) setLoading(true);
    const results = await Promise.allSettled([getCameras(), getAlerts(), getHourlyReports()]);
    if (results[0].status === 'fulfilled') setCameras(results[0].value);
    if (results[1].status === 'fulfilled') setAlerts(results[1].value);
    if (results[2].status === 'fulfilled') setReports(results[2].value);
    const failures = results.filter(result => result.status === 'rejected');
    setError(failures.length === 3 ? 'The central service is unreachable. Showing the last available data.' :
      failures.length ? 'Some information could not be refreshed. Showing available data.' : '');
    if (failures.length < 3) setUpdatedAt(new Date());
    setLoading(false);
  }, []);

  useEffect(() => {
    if (!operatorKey) {
      setCameras([]); setAlerts([]); setReports([]); setLoading(false);
      return;
    }
    void refresh(true);
    const timer = window.setInterval(() => void refresh(), 10_000);
    return () => window.clearInterval(timer);
  }, [operatorKey, refresh]);

  return { cameras, alerts, reports, loading, error, updatedAt, refresh };
}

function StatCard({ icon: Icon, label, value, note, tone }: {
  icon: typeof Activity; label: string; value: number | string; note: string; tone?: string;
}) {
  return <div className={`stat-card ${tone || ''}`}>
    <div className="stat-top"><span>{label}</span><Icon size={18} strokeWidth={1.8} /></div>
    <strong>{value}</strong><small>{note}</small>
  </div>;
}

function SourceTag({ mode }: { mode?: string | null }) {
  return <span className={`source-tag ${(mode || 'live').toLowerCase()}`}>
    {mode === 'synthetic' ? 'SIMULATED' : 'LIVE SOURCE'}
  </span>;
}

function AlertList({ alerts, selectedId, onSelect, limit }: {
  alerts: Alert[]; selectedId?: string | null; onSelect: (id: string) => void; limit?: number;
}) {
  const visible = limit ? alerts.slice(0, limit) : alerts;
  if (!visible.length) return <div className="empty-state"><ShieldCheck size={30} /><strong>No alerts yet</strong><span>Watchlist matches will appear here as soon as they arrive.</span></div>;
  return <div className="alert-list">
    {visible.map(alert => <button key={alert.id} className={`alert-row ${selectedId === alert.id ? 'selected' : ''}`} onClick={() => onSelect(alert.id)}>
      <div className={`alert-icon ${alert.status === 'new' ? 'active' : ''}`}><TriangleAlert size={18} /></div>
      <div className="alert-row-main"><div className="alert-row-title"><strong>{alert.masked_plate || 'Masked plate'}</strong><SourceTag mode={alert.source_mode} /></div>
        <span>{alert.camera_id} <span className="bullet">·</span> {alert.reason || 'Watchlist match'}</span>
        <small>{formatTime(alert.observed_at)}</small></div>
      <div className="alert-row-right"><span className={`pill ${statusClass(alert.status)}`}>{titleCase(alert.status)}</span><ChevronRight size={17} /></div>
    </button>)}
  </div>;
}

function AlertDetail({ alert, camera, busy, onStatus, onReport, onPreview }: {
  alert?: Alert; camera?: Camera; busy: boolean;
  onStatus: (status: AlertStatus) => void; onReport: () => void; onPreview: () => void;
}) {
  if (!alert) return <div className="detail-placeholder"><CircleAlert size={28} /><strong>Select an alert</strong><span>Review its context and record an operator decision.</span></div>;
  return <div className="alert-detail">
    <div className="detail-head"><div><span className="eyebrow">ALERT DETAIL</span><h3>{alert.masked_plate || 'Masked plate'}</h3></div><SourceTag mode={alert.source_mode} /></div>
    <div className={`alert-callout ${alert.status === 'new' ? 'unreviewed' : ''}`}><TriangleAlert size={19} /><div><strong>{alert.reason || 'Watchlist match'}</strong><span>{titleCase(alertSeverity(alert))} priority · {titleCase(alert.status)}</span></div></div>
    <dl className="detail-grid"><div><dt>Camera</dt><dd>{camera?.name || alert.camera_id}</dd></div><div><dt>Observed</dt><dd>{formatTime(alert.observed_at)}</dd></div>
      <div><dt>Source</dt><dd>{alert.source_mode === 'synthetic' ? 'Simulated event' : 'Live camera'}</dd></div><div><dt>Sighting #</dt><dd className="mono short-id">{alert.sighting_id || '—'}</dd></div></dl>
    <div className="detail-actions">
      {alert.status === 'new' && <button disabled={busy} className="button primary" onClick={() => onStatus('acknowledged')}><Check size={16} /> Acknowledge</button>}
      {alert.status === 'acknowledged' && <button disabled={busy} className="button" onClick={() => onStatus('resolved')}><CheckCircle2 size={16} /> Resolve</button>}
      {isActionable(alert) && <button disabled={busy} className="button ghost" onClick={() => onStatus('false_positive')}>Mark false positive</button>}
      <button className="button ghost" onClick={onReport}><FileText size={16} /> Incident report</button>
      {camera?.source_mode === 'live' && <button className="button ghost" onClick={onPreview}><Play size={16} /> Live preview</button>}
    </div>
    {!!alert.audit?.length && <div className="audit-trail"><strong>Decision history</strong>{alert.audit.map(entry => <div key={entry.id}><span>{titleCase(entry.to_status)} by {entry.actor || 'operator'}</span><small>{formatTime(entry.changed_at)}</small></div>)}</div>}
    {alert.source_mode === 'live' && <p className="privacy-note"><ShieldCheck size={15} /> Genuine camera imagery and full plate text are not retained.</p>}
  </div>;
}

function CameraTable({ cameras, onPreview, onSelect }: {
  cameras: Camera[]; onPreview: (camera: Camera) => void; onSelect: (camera: Camera) => void;
}) {
  if (!cameras.length) return <div className="empty-state"><CameraIcon size={30} /><strong>No cameras discovered</strong><span>The catalogue will populate this view when the edge service connects.</span></div>;
  return <div className="table-scroll"><table className="data-table"><thead><tr><th>Camera</th><th>Location</th><th>State</th><th>Source</th><th>Last seen</th><th /></tr></thead><tbody>
    {cameras.map(camera => <tr key={camera.id} onClick={() => onSelect(camera)}><td><span className="camera-name">{camera.name || camera.id}</span><small className="muted mono">{camera.id}</small></td>
      <td>{camera.location_name || camera.region || 'Location not provided'}</td><td><span className={`pill ${statusClass(camera.status)}`}><span className="status-dot" />{titleCase(camera.status)}</span></td>
      <td><SourceTag mode={camera.source_mode} /></td><td>{formatTime(camera.last_seen_at || camera.updated_at)}</td>
      <td>{camera.source_mode === 'live' && <button className="icon-button" aria-label={`Preview ${camera.name}`} onClick={event => { event.stopPropagation(); onPreview(camera); }}><Play size={16} /></button>}</td></tr>)}
  </tbody></table></div>;
}

function RouteView({ cameras, onError }: { cameras: Camera[]; onError: (message: string) => void }) {
  const [plate, setPlate] = useState('');
  const [results, setResults] = useState<Sighting[]>([]);
  const [searched, setSearched] = useState(false);
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    const query = plate.trim();
    if (!query) return;
    setBusy(true); onError('');
    try {
      const sightings = await searchRoute(query);
      setResults([...sightings].sort((a, b) => new Date(a.observed_at).getTime() - new Date(b.observed_at).getTime()));
      setSearched(true); setPlate('');
    } catch (failure) { onError((failure as Error).message); }
    finally { setBusy(false); }
  }

  return <div className="route-layout">
    <div className="panel route-panel"><div className="panel-heading"><div><span className="eyebrow">MOVEMENT HISTORY</span><h2>Route search</h2></div><MapPinned size={21} /></div>
      <p className="panel-intro">Enter an authorized plate number to find its ordered camera sightings. The number is not stored in this browser after the search.</p>
      <form className="search-form" onSubmit={submit}><label htmlFor="plate-query">Plate number</label><div className="search-line"><input id="plate-query" value={plate} onChange={event => setPlate(event.target.value)} autoComplete="off" spellCheck={false} placeholder="Enter full plate number" /><button className="button primary" disabled={busy || !plate.trim()}><Search size={16} />{busy ? 'Searching…' : 'Search'}</button></div></form>
      <div className="route-results-head"><strong>Sightings</strong><span>{searched ? `${results.length} found` : 'Awaiting search'}</span></div>
      {!searched ? <div className="empty-state compact"><Route size={27} /><span>Search for a plate to see its route.</span></div> : !results.length ? <div className="empty-state compact"><Search size={27} /><strong>No sightings found</strong><span>Check the plate or try a different one.</span></div> :
        <ol className="route-timeline">{results.map((sighting, index) => <li key={sighting.id || sighting.event_id || `${sighting.camera_id}-${index}`}><span className="timeline-index">{index + 1}</span><div><strong>{sighting.camera_name || cameras.find(camera => camera.id === sighting.camera_id)?.name || sighting.camera_id}</strong><span>{formatTime(sighting.observed_at)} · {sighting.masked_plate}</span><SourceTag mode={sighting.source_mode} /></div></li>)}</ol>}
    </div>
    <div className="panel route-map-panel"><div className="panel-heading"><div><span className="eyebrow">GEOGRAPHIC VIEW</span><h2>Camera sequence</h2></div></div><MapPanel cameras={cameras} route={results} /></div>
  </div>;
}

function ReportsView({ reports }: { reports: HourlyReport[] }) {
  return <div className="panel reports-panel"><div className="panel-heading"><div><span className="eyebrow">REGIONAL OPERATIONS</span><h2>Hourly reports</h2></div><Clock3 size={21} /></div>
    <p className="panel-intro">Central hourly aggregation of sightings, alerts and regional health. Times are displayed in India Standard Time.</p>
    {!reports.length ? <div className="empty-state"><FileText size={30} /><strong>No hourly reports yet</strong><span>Reports appear after a region completes its first reporting window.</span></div> :
      <div className="table-scroll"><table className="data-table"><thead><tr><th>Period</th><th>Sightings</th><th>Alerts</th><th>Health reports</th><th>Decoded frames</th><th>Analysed frames</th><th>Reconnects</th></tr></thead><tbody>
        {reports.map((report, index) => <tr key={report.id || index}><td>{formatTime(report.window_start || report.period_start || report.generated_at)}</td><td>{report.sightings ?? report.sighting_count ?? '—'}</td><td>{report.alerts ?? report.alert_count ?? '—'}</td><td>{report.health_reports ?? '—'}</td><td>{report.decoded_frames ?? '—'}</td><td>{report.inference_frames ?? '—'}</td><td>{report.reconnects ?? '—'}</td></tr>)}
      </tbody></table></div>}
  </div>;
}

function printableValue(value: unknown): string {
  if (value == null) return '—';
  if (typeof value === 'object') return JSON.stringify(value, null, 2);
  return String(value);
}

function IncidentModal({ report, onClose }: { report: IncidentReport; onClose: () => void }) {
  return <div className="modal-backdrop" onMouseDown={event => { if (event.target === event.currentTarget) onClose(); }}><section className="modal incident-modal" role="dialog" aria-modal="true" aria-label="Incident report">
    <div className="modal-header"><div><span className="eyebrow">SENTINEL COMMAND CENTRE</span><h2>Incident report</h2></div><button className="icon-button print-hide" onClick={onClose} aria-label="Close"><X size={19} /></button></div>
    <div className="report-banner">{report.source_mode === 'synthetic' ? 'SIMULATED EVENT · DEMONSTRATION ONLY' : 'LIVE SOURCE METADATA · NO STORED CAMERA IMAGE'}</div>
    <div className="report-fields">{Object.entries(report).filter(([key]) => key !== 'evidence_url' && key !== 'image_url').map(([key, value]) => <div key={key}><span>{titleCase(key)}</span><strong>{printableValue(value)}</strong></div>)}</div>
    <div className="modal-footer print-hide"><button className="button" onClick={onClose}>Close</button><button className="button primary" onClick={() => window.print()}><Printer size={16} /> Print report</button></div>
  </section></div>;
}

export default function App() {
  const [page, setPage] = useState<Page>('overview');
  const [operatorKey, setOperatorKeyState] = useState(getOperatorKey);
  const { cameras, alerts, reports, loading, error, updatedAt, refresh } = useControlRoom(operatorKey);
  const [selectedAlertId, setSelectedAlertId] = useState<string | null>(null);
  const [selectedCameraId, setSelectedCameraId] = useState<string | null>(null);
  const [previewCamera, setPreviewCamera] = useState<Camera | null>(null);
  const [incident, setIncident] = useState<IncidentReport | null>(null);
  const [toast, setToast] = useState('');
  const [actionError, setActionError] = useState('');
  const [busy, setBusy] = useState(false);
  const [keyDraft, setKeyDraft] = useState('');
  const [keyOpen, setKeyOpen] = useState(false);
  const [connection, setConnection] = useState<Connection>('locked');

  const orderedAlerts = useMemo(() => [...alerts].sort((a, b) => {
    if (isActionable(a) !== isActionable(b)) return isActionable(a) ? -1 : 1;
    return new Date(b.observed_at).getTime() - new Date(a.observed_at).getTime();
  }), [alerts]);
  const selectedAlert = alerts.find(alert => alert.id === selectedAlertId) || orderedAlerts[0];
  const selectedCamera = cameras.find(camera => camera.id === selectedCameraId) || cameras[0];
  const online = cameras.filter(camera => camera.source_mode === 'live' && ['online', 'healthy', 'connected'].includes(camera.status.toLowerCase())).length;
  const degraded = cameras.filter(camera => camera.source_mode === 'live' && ['degraded', 'reconnecting'].includes(camera.status.toLowerCase())).length;
  const activeAlerts = alerts.filter(isActionable).length;
  const syntheticAlerts = alerts.filter(alert => alert.source_mode === 'synthetic').length;

  useEffect(() => {
    if (!operatorKey) { setConnection('locked'); return; }
    let ws: WebSocket;
    let timer: number | undefined;
    let cancelled = false;
    function connect() {
      if (cancelled) return;
      setConnection('reconnecting');
      ws = new WebSocket(wsUrl(), ['sentinel-operator', operatorKey]);
      ws.onopen = () => setConnection('connected');
      ws.onmessage = () => void refresh();
      ws.onclose = () => { if (!cancelled) { setConnection('reconnecting'); timer = window.setTimeout(connect, 4000); } };
      ws.onerror = () => ws.close();
    }
    connect();
    return () => { cancelled = true; if (timer) window.clearTimeout(timer); ws?.close(); };
  }, [operatorKey, refresh]);

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(''), 5000);
    return () => window.clearTimeout(timer);
  }, [toast]);

  function saveKey(event: FormEvent) {
    event.preventDefault();
    setOperatorKey(keyDraft);
    setOperatorKeyState(getOperatorKey());
    setKeyDraft(''); setKeyOpen(false); setActionError('');
    setToast('Operator key saved for this browser session.');
  }

  async function updateStatus(status: AlertStatus) {
    if (!selectedAlert) return;
    setBusy(true); setActionError('');
    try {
      await changeAlertStatus(selectedAlert.id, status);
      setToast(`Alert ${titleCase(status).toLowerCase()}.`);
      await refresh();
    } catch (failure) { setActionError((failure as Error).message); }
    finally { setBusy(false); }
  }

  async function openIncident() {
    if (!selectedAlert) return;
    setActionError('');
    try { setIncident(await getIncidentReport(selectedAlert.id)); }
    catch (failure) { setActionError((failure as Error).message); }
  }

  async function createDemoEvent() {
    setBusy(true); setActionError('');
    try { await injectDemo(); setToast('Simulated watchlist event added.'); await refresh(); setPage('alerts'); }
    catch (failure) { setActionError((failure as Error).message); }
    finally { setBusy(false); }
  }

  function showCamera(cameraId: string) {
    setSelectedCameraId(cameraId); setPage('cameras');
  }

  const pageLabel = navigation.find(item => item.id === page)?.label || 'Overview';
  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark"><ShieldCheck size={23} /></div><div><strong>SENTINEL</strong><span>VEHICLE INTELLIGENCE</span></div></div>
      <div className="workspace-label">COMMAND CENTRE</div>
      <nav className="side-nav" aria-label="Main navigation">{navigation.map(item => { const Icon = item.icon; return <button key={item.id} className={page === item.id ? 'active' : ''} onClick={() => setPage(item.id)}><Icon size={19} /><span>{item.label}</span>{item.id === 'alerts' && activeAlerts > 0 && <em>{activeAlerts}</em>}</button>; })}</nav>
      <div className="sidebar-bottom"><div className="sidebar-system"><span className="system-glow" /><div><strong>Regional prototype</strong><span>{online} connected · {cameras.length} catalogued</span></div></div><p>Live events contain metadata only. Synthetic events are visibly marked.</p></div>
    </aside>

    <div className="main-column"><header className="topbar"><div className="crumb">Operations <ChevronRight size={15} /> <strong>{pageLabel}</strong></div><div className="topbar-actions"><span className={`connection-chip ${connection}`}>{connection === 'connected' ? <Wifi size={15} /> : <WifiOff size={15} />}{connection === 'connected' ? 'Live updates' : connection === 'locked' ? 'Operator locked' : 'Reconnecting'}</span><button className="icon-button" onClick={() => void refresh(true)} title="Refresh data" aria-label="Refresh data"><RefreshCw size={17} /></button><button className="operator-button" onClick={() => setKeyOpen(true)}><KeyRound size={16} />{operatorKey ? 'Operator unlocked' : 'Unlock operator'}</button></div></header>
      <main className="content">
        <div className="page-intro"><div><span className="eyebrow">GUJARAT · REGIONAL MONITORING</span><h1>{page === 'overview' ? 'Command overview' : pageLabel}</h1><p>{page === 'overview' ? 'Live operational picture across connected regional cameras.' : page === 'alerts' ? 'Review watchlist matches and record each decision.' : page === 'cameras' ? 'Inspect camera health and open live previews when needed.' : page === 'routes' ? 'Trace an authorized plate through observed camera sightings.' : 'Review hourly worker health and activity.'}</p></div><div className="intro-actions"><span className="last-updated"><Clock3 size={14} /> {updatedAt ? `Updated ${updatedAt.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' })}` : 'Waiting for data'}</span>{page === 'overview' && <button className="button demo-button" onClick={() => void createDemoEvent()} disabled={busy}><Play size={16} /> Inject simulated alert</button>}</div></div>

        {!operatorKey && <div className="locked-banner"><KeyRound size={20} /><div><strong>Operator access is locked</strong><span>Enter the local operator key to load cameras, alerts and reports.</span></div><button className="button primary" onClick={() => setKeyOpen(true)}>Unlock</button></div>}
        {(error || actionError) && <div className="error-banner" role="alert"><CircleAlert size={18} /><span>{actionError || error}</span><button onClick={() => { setActionError(''); void refresh(); }} aria-label="Retry"><RefreshCw size={16} /></button></div>}
        {loading && <div className="loading-note"><RefreshCw size={16} className="spinning" /> Loading command centre…</div>}

        {operatorKey && page === 'overview' && <>
          <div className="stats-grid"><StatCard icon={CameraIcon} label="Active cameras" value={online} note={`${cameras.length} in current catalogue`} tone="teal" /><StatCard icon={BellRing} label="Open alerts" value={activeAlerts} note="Awaiting operator action" tone="amber" /><StatCard icon={Activity} label="Degraded feeds" value={degraded} note="Reconnecting or reduced health" tone="blue" /><StatCard icon={ShieldCheck} label="Demo events" value={syntheticAlerts} note="Clearly labelled as simulated" /></div>
          <div className="overview-grid"><section className="panel map-panel"><div className="panel-heading"><div><span className="eyebrow">LIVE NETWORK</span><h2>Camera coverage</h2></div><span className="subtle-label"><span className="status-dot" /> {online} online</span></div><MapPanel cameras={cameras} selectedCameraId={selectedCameraId} onSelectCamera={showCamera} /><div className="map-legend"><span><i className="legend-dot online" />Online</span><span><i className="legend-dot degraded" />Degraded</span><span><i className="legend-dot offline" />Offline</span><span><i className="legend-dot unscheduled" />Unscheduled</span></div></section>
            <section className="panel recent-alerts"><div className="panel-heading"><div><span className="eyebrow">WATCHLIST EVENTS</span><h2>Recent alerts</h2></div><button className="text-link" onClick={() => setPage('alerts')}>View queue <ArrowRight size={16} /></button></div><AlertList alerts={orderedAlerts} selectedId={selectedAlert?.id} onSelect={id => { setSelectedAlertId(id); setPage('alerts'); }} limit={5} /></section></div>
          <section className="panel health-panel"><div className="panel-heading"><div><span className="eyebrow">SYSTEM HEALTH</span><h2>Camera status</h2></div><button className="text-link" onClick={() => setPage('cameras')}>All cameras <ArrowRight size={16} /></button></div><CameraTable cameras={cameras.slice(0, 6)} onPreview={setPreviewCamera} onSelect={camera => showCamera(camera.id)} /></section>
        </>}

        {operatorKey && page === 'alerts' && <div className="alerts-layout"><section className="panel queue-panel"><div className="panel-heading"><div><span className="eyebrow">PRIORITIZED QUEUE</span><h2>Watchlist alerts</h2></div><span className="count-chip">{activeAlerts} open</span></div><AlertList alerts={orderedAlerts} selectedId={selectedAlert?.id} onSelect={setSelectedAlertId} /></section><section className="panel detail-panel"><AlertDetail alert={selectedAlert} camera={cameras.find(camera => camera.id === selectedAlert?.camera_id)} busy={busy} onStatus={status => void updateStatus(status)} onReport={() => void openIncident()} onPreview={() => { const camera = cameras.find(item => item.id === selectedAlert?.camera_id); if (camera) setPreviewCamera(camera); }} /></section></div>}

        {operatorKey && page === 'cameras' && <div className="camera-layout"><section className="panel camera-table-panel"><div className="panel-heading"><div><span className="eyebrow">DISCOVERED FEEDS</span><h2>Camera catalogue</h2></div><span className="count-chip">{cameras.length} cameras</span></div><CameraTable cameras={cameras} onPreview={setPreviewCamera} onSelect={camera => setSelectedCameraId(camera.id)} /></section>{selectedCamera && <section className="panel camera-side-panel"><div className="panel-heading"><div><span className="eyebrow">CAMERA DETAIL</span><h2>{selectedCamera.name}</h2></div><span className={`pill ${statusClass(selectedCamera.status)}`}>{titleCase(selectedCamera.status)}</span></div><dl className="detail-grid"><div><dt>Camera ID</dt><dd>{selectedCamera.id}</dd></div><div><dt>Location</dt><dd>{selectedCamera.location_name || selectedCamera.region || 'Not provided'}</dd></div><div><dt>Source</dt><dd><SourceTag mode={selectedCamera.source_mode} /></dd></div><div><dt>Last updated</dt><dd>{formatTime(selectedCamera.updated_at)}</dd></div></dl>{selectedCamera.source_mode === 'live' ? <><button className="button primary" onClick={() => setPreviewCamera(selectedCamera)}><Play size={16} /> Open live preview</button><p className="privacy-note"><Radio size={15} /> Live preview starts only when opened.</p></> : <p className="privacy-note"><Radio size={15} /> Simulated cameras have no live video.</p>}</section>}</div>}
        {operatorKey && page === 'routes' && <RouteView cameras={cameras} onError={setActionError} />}
        {operatorKey && page === 'reports' && <ReportsView reports={reports} />}
      </main>
      <footer className="footer"><span>Sentinel prototype · Operator decision support</span><span><ArrowDownRight size={13} /> Times shown in IST · Data stored in UTC</span></footer>
    </div>

    {toast && <div className="toast" role="status"><CheckCircle2 size={18} />{toast}<button onClick={() => setToast('')} aria-label="Dismiss"><X size={15} /></button></div>}
    {keyOpen && <div className="modal-backdrop" onMouseDown={event => { if (event.target === event.currentTarget) setKeyOpen(false); }}><section className="modal key-modal" role="dialog" aria-modal="true" aria-label="Operator access"><div className="modal-header"><div><span className="eyebrow">OPERATOR ACCESS</span><h2>Unlock actions</h2></div><button className="icon-button" onClick={() => setKeyOpen(false)} aria-label="Close"><X size={19} /></button></div><p>The operator key enables alert decisions, route searches and simulated events. It is stored only for this browser session.</p><form onSubmit={saveKey}><label htmlFor="operator-key">Operator key</label><input id="operator-key" type="password" value={keyDraft} onChange={event => setKeyDraft(event.target.value)} autoComplete="off" placeholder="Enter local operator key" required /><div className="modal-footer"><button type="button" className="button ghost" onClick={() => { setOperatorKey(''); setOperatorKeyState(''); setKeyOpen(false); setToast('Operator access locked.'); }}>Lock</button><button className="button primary" type="submit">Unlock</button></div></form></section></div>}
    {previewCamera && <div className="modal-backdrop" onMouseDown={event => { if (event.target === event.currentTarget) setPreviewCamera(null); }}><section className="modal preview-modal" role="dialog" aria-modal="true" aria-label={`${previewCamera.name} live preview`}><div className="modal-header"><div><span className="eyebrow">LIVE CAMERA PREVIEW</span><h2>{previewCamera.name}</h2></div><button className="icon-button" onClick={() => setPreviewCamera(null)} aria-label="Close"><X size={19} /></button></div><LivePlayer camera={previewCamera} /><div className="modal-footer"><span>{previewCamera.location_name || previewCamera.id} · {titleCase(previewCamera.status)}</span><button className="button" onClick={() => setPreviewCamera(null)}>Close preview</button></div></section></div>}
    {incident && <IncidentModal report={incident} onClose={() => setIncident(null)} />}
  </div>;
}
