import type { RecognitionState, RecognitionStatus } from './api';

export function recognitionLabel(state: RecognitionState | string): string {
  switch (state) {
    case 'reading': return 'Reading';
    case 'uncertain': return 'Uncertain';
    case 'model_accepted': return 'Model-accepted · unverified';
    case 'unreadable': return 'Unreadable';
    case 'paused': return 'Paused';
    default: return 'Unavailable';
  }
}

export function recognitionTone(state: RecognitionState | string): string {
  if (state === 'model_accepted' || state === 'uncertain') return 'warn';
  if (state === 'reading') return 'good';
  return 'neutral';
}

export function recognitionIsFresh(status: RecognitionStatus | null): boolean {
  return status?.status === 'live' && typeof status.frame_age_seconds === 'number'
    && Number.isFinite(status.frame_age_seconds) && status.frame_age_seconds >= 0
    && status.frame_age_seconds <= 5;
}

export function recognitionCount(value?: number): string {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0
    ? Math.floor(value).toLocaleString('en-IN') : '—';
}
