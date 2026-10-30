export type SourceMode = 'live' | 'synthetic' | 'recorded' | string;
/** Whether a recorded event's time is the recording's own clock or only when it was processed. */
export type TimeBasis = 'source_clock' | 'processing_time';
export type AlertStatus = 'new' | 'acknowledged' | 'resolved' | 'false_positive' | string;

export interface Camera {
  id: string;
  name: string;
  region?: string | null;
  location_name?: string | null;
  latitude?: number | null;
  longitude?: number | null;
  status: string;
  source_mode?: SourceMode | null;
  preview_url?: string | null;
  last_seen_at?: string | null;
  updated_at?: string | null;
  coordinates_approximate?: boolean;
}

export interface CatalogueCamera {
  id: string;
  name: string;
  location: string;
  codec: string;
  live: boolean;
}

export interface CatalogueSearchResponse {
  total: number;
  offset: number;
  limit: number;
  items: CatalogueCamera[];
}

export interface PreviewSession {
  id: string;
  camera_id: string;
  status: 'starting' | 'ready' | 'failed';
  preview_url: string;
}

export type RecognitionState = 'reading' | 'uncertain' | 'model_accepted' | 'unreadable' | 'paused' | 'unavailable' | 'completed';

/** Operator-safe metadata only. Crops, full hypotheses and source URLs stay in the local console. */
export interface RecognitionStatus {
  schema_version: '1.0';
  /** `processing` and `completed` occur only for a recording; `live` only for a camera. */
  status: 'paused' | 'connecting' | 'live' | 'unavailable' | 'processing' | 'completed';
  /** A camera ID, or a recording ID when source_mode is `recorded`. */
  camera_id: string | null;
  source_mode: 'live' | 'recorded';
  session_id?: string | null;
  recording_name?: string | null;
  video_position_ms?: number | null;
  video_duration_ms?: number | null;
  time_basis?: TimeBasis | null;
  pacing?: 'paced' | 'unpaced' | null;
  stale_after_seconds?: number;
  inference_device?: 'cpu' | 'gpu' | null;
  failure_code?: string;
  frame_age_seconds?: number | null;
  resolution?: [number, number] | null;
  vehicle_boxes?: number;
  decoded_frames?: number;
  analysed_frames?: number;
  plate_crops?: number;
  unique_model_accepted_unverified?: number;
  model_profile?: string;
  recognition_state: RecognitionState;
  tracks: {
    track_id: string;
    observations: number;
    masked_plate: string | null;
    recognition_state: RecognitionState;
    candidate_count: number;
  }[];
  /** Bounded recent tracks, newest first. The status route carries masked metadata only. */
  vehicles_seen?: {
    track_id: string;
    vehicle_class: string;
    confidence: number | null;
    first_ms: number | null;
    last_ms: number | null;
    observations: number;
    state: RecognitionState;
    masked_plate: string | null;
    watchlist: 'possible_match' | 'no_match' | null;
  }[];
  delivery?: { enabled: boolean; outbox_depth: number };
}

export interface RecognitionCameras {
  available: boolean;
  scope?: 'configured_relays' | 'authorised_catalogue' | 'unavailable';
  cameras: { id: string; name: string }[];
}

export interface RecognitionRecordings {
  recordings: { id: string; name: string; duration_ms: number | null; time_basis: TimeBasis }[];
}

export type PlaybackAction = 'pause' | 'resume' | 'restart' | 'seek';

export interface Alert {
  id: string;
  sighting_id: string;
  camera_id: string;
  observed_at: string;
  masked_plate: string;
  reason: string;
  severity: string;
  status: AlertStatus;
  source_mode: SourceMode;
  stream_pts_ms?: number | null;
  session_id?: string | null;
  time_basis?: TimeBasis | null;
  processed_at?: string | null;
  confidence?: number | null;
  created_at?: string | null;
  acknowledged_at?: string | null;
  resolved_at?: string | null;
  audit?: { id: string; from_status?: string | null; to_status: string; actor?: string; note?: string | null; changed_at: string }[];
}

export interface Sighting {
  id?: string;
  event_id?: string;
  camera_id: string;
  camera_name?: string;
  observed_at: string;
  masked_plate: string;
  source_mode?: SourceMode;
  stream_pts_ms?: number | null;
  session_id?: string | null;
  time_basis?: TimeBasis | null;
  latitude?: number | null;
  longitude?: number | null;
}

export interface HourlyReport {
  id?: string;
  edge_id?: string;
  camera_id?: string;
  period_start?: string;
  window_start?: string;
  window_end?: string;
  generated_at?: string;
  sightings?: number;
  sighting_count?: number;
  alerts?: number;
  alert_count?: number;
  uptime_percent?: number;
  reconnects?: number;
  health_reports?: number;
  decoded_frames?: number;
  inference_frames?: number;
  source_mode?: SourceMode;
  [key: string]: unknown;
}

export interface HourlyReportResponse {
  hours?: number;
  buckets: HourlyReport[];
}

export type IncidentReport = Record<string, unknown>;

import type { DiagnosticsSnapshot, LocalSessionOut, WatchlistCreate, WatchlistItem, WatchlistPage, WatchlistPatch } from './unified-contract';

// Same-origin only: cookies and browser request tokens never go to another host.
const API_BASE = '';
let localSession: LocalSessionOut | null = null;
let sessionPending: Promise<LocalSessionOut> | null = null;

export class ApiError extends Error {
  constructor(message: string, public readonly status: number) { super(message); }
}

/** Tokens are transient request-forgery protection, not operator/service keys. */
export function resetLocalSession(): void { localSession = null; sessionPending = null; }

export async function ensureLocalSession(): Promise<LocalSessionOut> {
  if (localSession && Date.parse(localSession.expires_at) > Date.now() + 10_000) return localSession;
  if (sessionPending) return sessionPending;
  sessionPending = (async () => {
    // Remove obsolete credentials from older installations without ever reading them.
    try { sessionStorage.removeItem('sentinel.operatorKey'); } catch { /* storage may be disabled */ }
    let response: Response;
    try { response = await fetch('/v1/local/session', { method: 'POST', credentials: 'same-origin', cache: 'no-store', headers: { 'Content-Type': 'application/json' }, body: '{}' }); }
    catch { throw new Error('The central service is unavailable. Check Docker, then retry.'); }
    if (!response.ok) throw new ApiError('Local access is unavailable. Check the laptop-only service configuration, then retry.', response.status);
    const session = await response.json() as LocalSessionOut;
    if (typeof session.csrf_token !== 'string' || !session.csrf_token || !Number.isFinite(Date.parse(session.expires_at))) throw new Error('The local session response was invalid.');
    localSession = session;
    return session;
  })();
  try { return await sessionPending; } finally { sessionPending = null; }
}

function unwrapList<T>(value: unknown, keys: string[]): T[] {
  if (Array.isArray(value)) return value as T[];
  if (value && typeof value === 'object') {
    const record = value as Record<string, unknown>;
    for (const key of keys) if (Array.isArray(record[key])) return record[key] as T[];
  }
  return [];
}

async function request<T>(path: string, init?: RequestInit, protectedRequest = false, retried = false): Promise<T> {
  const headers = new Headers(init?.headers);
  if (init?.body) headers.set('Content-Type', 'application/json');
  if (protectedRequest) {
    const session = await ensureLocalSession();
    if (!['GET', 'HEAD', 'OPTIONS'].includes((init?.method || 'GET').toUpperCase())) headers.set('X-CSRF-Token', session.csrf_token);
  }
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, { ...init, headers, credentials: 'same-origin' });
  } catch {
    throw new Error('The central service is unreachable. Check that it is running.');
  }
  if (!response.ok) {
    // An authentication rejection occurs before an operation executes. Retry once;
    // never automatically replay conflicts, validation errors or network failures.
    if (response.status === 401 && protectedRequest && !retried) { resetLocalSession(); return request<T>(path, init, protectedRequest, true); }
    let detail = '';
    try {
      const body = await response.json() as { detail?: unknown };
      detail = typeof body.detail === 'string' ? body.detail : '';
    } catch { /* server may return plain text */ }
    if (response.status === 401 || response.status === 403) throw new ApiError('Local access was rejected. Retry or check the service configuration.', response.status);
    throw new ApiError(detail || `Request failed (${response.status}).`, response.status);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export async function getCameras(): Promise<Camera[]> {
  return unwrapList<Camera>(await request<unknown>('/v1/cameras', undefined, true), ['cameras', 'items']);
}

export function getRecognitionStatus(): Promise<RecognitionStatus> {
  return request<RecognitionStatus>('/v1/recognition/status', { cache: 'no-store' }, true);
}

export function getRecognitionCameras(): Promise<RecognitionCameras> {
  return request<RecognitionCameras>('/v1/recognition/cameras', { cache: 'no-store' }, true);
}

/** Opt-in view for permitted recordings only: the analysed frame and full, unverified readings. */
export interface RecognitionEvidence {
  source_mode: 'recorded';
  session_id: string | null;
  video_position_ms: number | null;
  frame: string | null;
  vehicles: {
    track_id: string; vehicle_class: string; confidence: number | null;
    first_ms: number | null; last_ms: number | null; observations: number;
    state: RecognitionState; reading: string | null; thumb: string | null;
    watchlist: 'possible_match' | 'no_match' | null;
  }[];
}

/** Resolves to null when the view is switched off or no recording is being analysed. */
export async function getRecognitionEvidence(mode: 'original' | 'overlays' = 'overlays'): Promise<RecognitionEvidence | null> {
  try { return await request<RecognitionEvidence>(`/v1/recognition/evidence?mode=${mode}`, { cache: 'no-store' }, true); }
  catch (error) { if (error instanceof ApiError && error.status === 404) return null; throw error; }
}

export function getDiagnostics(sessionId: string, trackId: string): Promise<DiagnosticsSnapshot> {
  const query = new URLSearchParams({ session_id: sessionId, track_id: trackId });
  return request(`/v1/recognition/diagnostics?${query}`, { cache: 'no-store' }, true);
}

export function getWatchlist(offset = 0): Promise<WatchlistPage> {
  return request(`/v1/watchlist?offset=${offset}&limit=25`, { cache: 'no-store' }, true);
}
export function searchWatchlist(query: string, offset = 0): Promise<WatchlistPage> {
  return request('/v1/watchlist/search', { method: 'POST', body: JSON.stringify({ query, offset, limit: 25 }) }, true);
}
export function addWatchlist(item: WatchlistCreate): Promise<WatchlistItem> {
  return request('/v1/watchlist', { method: 'POST', body: JSON.stringify(item) }, true);
}
export function patchWatchlist(id: number, patch: WatchlistPatch): Promise<WatchlistItem> {
  return request(`/v1/watchlist/${id}`, { method: 'PATCH', body: JSON.stringify(patch) }, true);
}

export function getRecognitionRecordings(): Promise<RecognitionRecordings> {
  return request<RecognitionRecordings>('/v1/recognition/recordings', { cache: 'no-store' }, true);
}

/** Playback control for the selected recording. `positionMs` is required for seek only. */
export function controlRecognition(action: PlaybackAction, positionMs?: number): Promise<RecognitionStatus> {
  const body = action === 'seek' ? { action, position_ms: Math.max(0, Math.round(positionMs ?? 0)) } : { action };
  return request<RecognitionStatus>('/v1/recognition/control', { method: 'POST', body: JSON.stringify(body) }, true);
}

export function setRecognitionSession(cameraId: string | null): Promise<RecognitionStatus> {
  return request<RecognitionStatus>('/v1/recognition/session', {
    method: 'POST', body: JSON.stringify({ camera_id: cameraId }),
  }, true);
}

export function heartbeatRecognition(cameraId: string): Promise<unknown> {
  return request<unknown>('/v1/recognition/heartbeat', {
    method: 'POST', body: JSON.stringify({ camera_id: cameraId }),
  }, true);
}

export async function searchCatalogue(query: string, offset = 0): Promise<CatalogueSearchResponse> {
  const params = new URLSearchParams({ q: query.trim(), offset: String(offset), limit: '25' });
  return request<CatalogueSearchResponse>(`/v1/catalogue/search?${params}`, undefined, true);
}

export async function startCataloguePreview(cameraId: string): Promise<PreviewSession> {
  return request<PreviewSession>(`/v1/catalogue/${encodeURIComponent(cameraId)}/preview`,
    { method: 'POST' }, true);
}

export async function getCataloguePreview(sessionId: string): Promise<PreviewSession> {
  return request<PreviewSession>(`/v1/previews/${encodeURIComponent(sessionId)}`, undefined, true);
}

export async function heartbeatCataloguePreview(sessionId: string): Promise<PreviewSession> {
  return request<PreviewSession>(`/v1/previews/${encodeURIComponent(sessionId)}/heartbeat`,
    { method: 'POST' }, true);
}

export async function stopCataloguePreview(sessionId: string): Promise<void> {
  await request<{ stopped: boolean }>(`/v1/previews/${encodeURIComponent(sessionId)}`,
    { method: 'DELETE' }, true);
}

export async function getAlerts(): Promise<Alert[]> {
  const alerts = unwrapList<Alert>(await request<unknown>('/v1/alerts', undefined, true), ['alerts', 'items']);
  return alerts.map(alert => ({
    ...alert,
    id: String(alert.id),
    sighting_id: String(alert.sighting_id),
    audit: alert.audit?.map(entry => ({ ...entry, id: String(entry.id) })),
  }));
}

export async function changeAlertStatus(id: string, status: AlertStatus): Promise<Alert> {
  const alert = await request<Alert>(`/v1/alerts/${encodeURIComponent(id)}`, {
    method: 'PATCH', body: JSON.stringify({ status }),
  }, true);
  return { ...alert, id: String(alert.id), sighting_id: String(alert.sighting_id) };
}

export async function searchRoute(plate: string): Promise<Sighting[]> {
  const response = await request<unknown>('/v1/routes/search', {
    method: 'POST', body: JSON.stringify({ plate: plate.trim() }),
  }, true);
  return unwrapList<Sighting>(response, ['sightings', 'route', 'items']);
}

export async function getHourlyReports(): Promise<HourlyReport[]> {
  return unwrapList<HourlyReport>(await request<unknown>('/v1/reports/hourly', undefined, true), ['buckets', 'reports', 'items']);
}

export async function getIncidentReport(id: string): Promise<IncidentReport> {
  return request<IncidentReport>(`/v1/reports/incidents/${encodeURIComponent(id)}`, undefined, true);
}

export async function injectDemo(): Promise<unknown> {
  return request<unknown>('/v1/demo/inject', { method: 'POST', body: JSON.stringify({}) }, true);
}

export function wsUrl(): string {
  return `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/v1/ws/alerts`;
}

export function formatTime(value?: string | null): string {
  if (!value) return '—';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return '—';
  return new Intl.DateTimeFormat('en-IN', {
    timeZone: 'Asia/Kolkata', day: '2-digit', month: 'short',
    hour: '2-digit', minute: '2-digit', hour12: false,
  }).format(parsed) + ' IST';
}

export function cameraPreview(camera: Camera): string {
  return camera.preview_url || `/media/${encodeURIComponent(camera.id)}/index.m3u8`;
}

export function isActionable(alert: Alert): boolean {
  return alert.status === 'new' || alert.status === 'acknowledged';
}
