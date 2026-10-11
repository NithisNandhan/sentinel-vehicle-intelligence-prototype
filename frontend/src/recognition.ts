import type { RecognitionState, RecognitionStatus } from './api';

export function recognitionLabel(state: RecognitionState | string): string {
  switch (state) {
    case 'reading': return 'Reading';
    case 'uncertain': return 'Uncertain';
    case 'model_accepted': return 'Model-accepted · unverified';
    case 'unreadable': return 'Unreadable';
    case 'paused': return 'Paused';
    case 'completed': return 'Completed';
    default: return 'Unavailable';
  }
}

export function recognitionTone(state: RecognitionState | string): string {
  if (state === 'model_accepted' || state === 'uncertain') return 'warn';
  if (state === 'reading') return 'good';
  return 'neutral';
}

export function isRecorded(status: RecognitionStatus | null): boolean {
  return status?.source_mode === 'recorded';
}

/** Seconds after which shown evidence is no longer current. A recording analysed on a slow CPU gets longer than a live feed. */
export function staleLimit(status: RecognitionStatus | null): number {
  const limit = status?.stale_after_seconds;
  return isRecorded(status) && typeof limit === 'number' && Number.isFinite(limit) && limit > 0 && limit <= 600 ? limit : 5;
}

export function recognitionIsFresh(status: RecognitionStatus | null): boolean {
  if (!status) return false;
  // A camera is only current while live. A recording's last analysed position
  // also stays current while it is paused or has reached the end.
  const current = isRecorded(status)
    ? ['processing', 'paused', 'completed'].includes(status.status)
    : status.status === 'live';
  return current && typeof status.frame_age_seconds === 'number'
    && Number.isFinite(status.frame_age_seconds) && status.frame_age_seconds >= 0
    && status.frame_age_seconds <= staleLimit(status);
}

/** A session that holds a source and must be kept alive by heartbeats. */
export function recognitionIsRunning(status: RecognitionStatus | null): boolean {
  if (!status?.camera_id) return false;
  return isRecorded(status)
    ? ['connecting', 'processing', 'paused', 'completed'].includes(status.status)
    : ['live', 'connecting'].includes(status.status);
}

export function recognitionCount(value?: number): string {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0
    ? Math.floor(value).toLocaleString('en-IN') : '—';
}

/** Downloaded footage is never presented as a live source. */
export function sourceLabel(mode?: string | null): string {
  if (mode === 'synthetic') return 'SIMULATED';
  if (mode === 'recorded') return 'RECORDED';
  return 'LIVE SOURCE';
}

export function sourceDescription(mode?: string | null): string {
  if (mode === 'synthetic') return 'Simulated event';
  if (mode === 'recorded') return 'Recorded footage';
  return 'Live camera';
}

/** Position inside a video, e.g. 1:05 or 1:02:03. Unknown values are never shown as zero. */
export function formatVideoTime(milliseconds?: number | null): string {
  if (typeof milliseconds !== 'number' || !Number.isFinite(milliseconds) || milliseconds < 0) return '—';
  const total = Math.floor(milliseconds / 1000);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = String(total % 60).padStart(2, '0');
  return hours ? `${hours}:${String(minutes).padStart(2, '0')}:${seconds}` : `${minutes}:${seconds}`;
}

export function sessionLabel(status: RecognitionStatus | null): string {
  if (!status) return 'Unavailable';
  const fresh = recognitionIsFresh(status);
  switch (status.status) {
    case 'connecting': return isRecorded(status) ? 'Opening recording' : 'Connecting';
    case 'live': return fresh ? 'Live analysis' : 'Waiting for fresh frames';
    case 'processing': return fresh ? 'Analysing recording' : 'Analysing · waiting for the next frame';
    case 'completed': return 'Recording finished';
    case 'paused': return status.camera_id ? 'Recording paused' : 'Paused';
    default: return 'Unavailable';
  }
}

/** Plain words for the failure codes the edge can report; never the raw exception text. */
export function failureMessage(code?: string | null): string {
  switch (code) {
    case 'ChangedRecording': return 'This file no longer matches its permission receipt, so it was not analysed.';
    case 'CorruptRecording': return 'This file could not be decoded as video.';
    case 'MissingPts': return 'This file has frames without timestamps and cannot be analysed reliably.';
    case 'NoVideoStream': return 'This file contains no video.';
    case 'FrameTooLarge': return 'The frames in this file are larger than the analysis memory limit.';
    case 'InvalidSeek': return 'That position is outside the recording.';
    case undefined: case null: case '': return '';
    default: return `Analysis stopped (${code}).`;
  }
}
