export type SourceMode = 'live' | 'synthetic' | string;
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

const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');
const OPERATOR_KEY_NAME = 'sentinel.operatorKey';

export function getOperatorKey(): string {
  return sessionStorage.getItem(OPERATOR_KEY_NAME) || '';
}

export function setOperatorKey(value: string): void {
  const key = value.trim();
  if (key) sessionStorage.setItem(OPERATOR_KEY_NAME, key);
  else sessionStorage.removeItem(OPERATOR_KEY_NAME);
}

function unwrapList<T>(value: unknown, keys: string[]): T[] {
  if (Array.isArray(value)) return value as T[];
  if (value && typeof value === 'object') {
    const record = value as Record<string, unknown>;
    for (const key of keys) if (Array.isArray(record[key])) return record[key] as T[];
  }
  return [];
}

async function request<T>(path: string, init?: RequestInit, protectedRequest = false): Promise<T> {
  const headers = new Headers(init?.headers);
  if (init?.body) headers.set('Content-Type', 'application/json');
  if (protectedRequest) {
    const key = getOperatorKey();
    if (!key) throw new Error('Enter the operator key to use this action.');
    headers.set('X-Operator-Key', key);
  }
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, { ...init, headers });
  } catch {
    throw new Error('The central service is unreachable. Check that it is running.');
  }
  if (!response.ok) {
    let detail = '';
    try {
      const body = await response.json() as { detail?: unknown };
      detail = typeof body.detail === 'string' ? body.detail : '';
    } catch { /* server may return plain text */ }
    if (response.status === 401 || response.status === 403) throw new Error('Operator key rejected. Check it and try again.');
    throw new Error(detail || `Request failed (${response.status}).`);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export async function getCameras(): Promise<Camera[]> {
  return unwrapList<Camera>(await request<unknown>('/v1/cameras', undefined, true), ['cameras', 'items']);
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
  const configured = import.meta.env.VITE_WS_BASE_URL as string | undefined;
  const base = configured || API_BASE;
  if (base.startsWith('http')) return `${base.replace(/^http/, 'ws')}/v1/ws/alerts`;
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
