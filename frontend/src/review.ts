import { ApiError, request } from './api';

/** Central, masked view of one passage outcome. Never contains a full plate or image. */
export interface ReviewItem {
  id: number;
  event_id: string;
  camera_id: string;
  passage_id: string;
  result_state: 'confirmed' | 'partial' | 'unreadable';
  observed_at: string;
  last_observed_at: string;
  masked_plate: string | null;
  plate_length: number | null;
  unknown_positions: number[];
  reason: string | null;
  review_required: boolean;
  review_status: 'pending' | 'not_required' | 'confirmed' | 'corrected' | 'rejected' | 'unreadable';
  length_certain: boolean | null;
  observations: number;
  layout: string;
  format_status: string | null;
  source_mode: string;
  stream_pts_ms: number | null;
  end_pts_ms: number | null;
  session_id: string | null;
  profile: string | null;
  policy_version: string | null;
  watchlist_lead: boolean;
  evidence_available: boolean;
  expires_at: string;
}

export interface Quality {
  height_px?: number; width_px?: number; sharpness?: number; contrast?: number; glare_fraction?: number;
  angle_deg?: number | null; blurred?: boolean; low_contrast?: boolean; glare?: boolean; low_light?: boolean; tilted?: boolean;
}

export interface ReviewDetail extends ReviewItem {
  quality: Quality | null;
  edge_id: string;
  model_version: string | null;
  track_ids: string[];
  lead_candidates: number;
  lead_candidates_considered: number | null;
  decisions: { reviewer: string; action: string; from_status: string; note: string | null; masked_plate: string | null; alert_id: number | null; decided_at: string }[];
}

/** Edge-held evidence, proxied without central storage, only for authorised reviewers. */
export interface ReviewEvidence {
  passage_id: string;
  pattern: string | null;
  unknown_positions: number[];
  length: number | null;
  length_certain: boolean;
  layout: string;
  positions: { index: number; known: boolean; support_frames: number; mean_peak: number; predicted?: string; alternatives?: string[] }[];
  primary_score: number | null;
  secondary: { text?: string; disputed_positions?: number[] } | null;
  quality: Quality | null;
  frame_jpeg_base64: string | null;
  frame_pts_ms: number | null;
  crops: { pts_ms: number; quality: number; width: number; height: number; image_png_base64: string }[];
  notice: string;
}

export type ReviewAction = 'confirm' | 'correct' | 'reject' | 'unreadable';

function reviewerHeaders(token: string): HeadersInit {
  return { 'X-Reviewer-Token': token };
}

async function reviewRequest<T>(token: string, path: string, init: RequestInit = {}): Promise<T> {
  if (!token) throw new ApiError('Enter your reviewer token to open the review queue.', 403);
  try {
    return await request<T>(path, { ...init, headers: { ...(init.headers || {}), ...reviewerHeaders(token) } }, true);
  } catch (failure) {
    if (failure instanceof ApiError && failure.status === 403) throw new ApiError('Reviewer access was rejected. Check your reviewer token.', 403);
    throw failure;
  }
}

export function getReviewQueue(token: string, status: 'pending' | 'decided' | 'all' = 'pending') {
  return reviewRequest<{ reviewer: string; pending: number; items: ReviewItem[] }>(token, `/v1/review/queue?status=${status}`);
}

export function getReviewDetail(token: string, id: number) {
  return reviewRequest<ReviewDetail>(token, `/v1/review/${id}`);
}

export async function getReviewEvidence(token: string, id: number): Promise<ReviewEvidence | null> {
  try { return await reviewRequest<ReviewEvidence>(token, `/v1/review/${id}/evidence`); }
  catch (failure) { if (failure instanceof ApiError && failure.status === 404) return null; throw failure; }
}

export function decideReview(token: string, id: number, action: ReviewAction, plate?: string, note?: string) {
  const body: Record<string, string> = { action };
  if (plate) body.plate = plate;
  if (note) body.note = note;
  return reviewRequest<{ result: ReviewItem; alert: unknown }>(token, `/v1/review/${id}/decision`, { method: 'POST', body: JSON.stringify(body) });
}

export function normalisePlate(value: string): string {
  return value.toUpperCase().replace(/[^A-Z0-9]/g, '');
}

export type Cell = { character: string; state: 'known' | 'unknown' | 'hidden' };

/**
 * Character cells for display. With edge evidence the reviewer sees the full
 * pattern; otherwise only the centrally stored mask (middle positions hidden).
 */
export function patternCells(item: Pick<ReviewItem, 'masked_plate' | 'unknown_positions' | 'plate_length'>, evidencePattern?: string | null): Cell[] {
  const unknown = new Set(item.unknown_positions || []);
  const text = evidencePattern || item.masked_plate || '';
  return Array.from(text).map((character, index) => unknown.has(index) || character === '?'
    ? { character: '?', state: 'unknown' as const }
    : character === '*' ? { character: '•', state: 'hidden' as const } : { character, state: 'known' as const });
}

/** Mirrors the server rule: a confirmation must agree with every known, visible position. */
export function confirmAgrees(plate: string, item: Pick<ReviewItem, 'masked_plate' | 'plate_length'>, evidencePattern?: string | null): boolean {
  const value = normalisePlate(plate);
  const reference = evidencePattern || item.masked_plate || '';
  if (!value || (item.plate_length != null && value.length !== item.plate_length) || value.length !== reference.length) return false;
  return Array.from(reference).every((character, index) => character === '*' || character === '?' || character === value[index]);
}

export const REASON_LABELS: Record<string, string> = {
  uncertain_characters: 'Some characters uncertain', insufficient_observations: 'Too few distinct frames',
  length_uncertain: 'Plate length disputed', length_disagreement: 'Frames disagree on length',
  insufficient_known_characters: 'Too few known characters', format_impossible: 'Not a valid Indian format',
  format_lookalike: 'Look-alike character (e.g. O/0) — not fixed automatically', format_ambiguous: 'Several format fixes possible',
  format_invalid: 'Invalid Indian format', format_adjusted: 'Format-adjusted reading', layout_conflict: 'One-row/two-row layout conflict',
  insufficient_resolution: 'Plate too small to confirm', secondary_reader_veto: 'Secondary reader disagreed',
  no_plate_found: 'No plate found', plate_too_small: 'Plate too small', non_plate_rejected: 'Proposal was not a plate',
  no_complete_candidate: 'No complete candidate',
};
