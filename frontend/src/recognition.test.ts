import { describe, expect, it } from 'vitest';
import type { RecognitionStatus } from './api';
import {
  failureMessage, formatVideoTime, recognitionCount, recognitionIsFresh, recognitionIsRunning, recognitionLabel,
  recognitionTone, sessionLabel, sourceDescription, sourceLabel, staleLimit,
} from './recognition';

const status: RecognitionStatus = {
  schema_version: '1.0', status: 'live', camera_id: 'cam17', source_mode: 'live',
  recognition_state: 'uncertain', frame_age_seconds: 0.4, tracks: [],
};
const recording: RecognitionStatus = {
  schema_version: '1.0', status: 'processing', camera_id: 'rec-0123456789ab', source_mode: 'recorded',
  recognition_state: 'reading', frame_age_seconds: 2, stale_after_seconds: 30, tracks: [],
  session_id: 's-1', recording_name: 'Junction A', video_position_ms: 1500, video_duration_ms: 6000,
};

describe('operator recognition presentation', () => {
  it('never describes a model acceptance as a verified plate', () => {
    expect(recognitionLabel('model_accepted')).toBe('Model-accepted · unverified');
    expect(recognitionTone('model_accepted')).toBe('warn');
    expect(recognitionTone('uncertain')).toBe('warn');
  });
  it('keeps missing or unknown states unavailable instead of confirmed', () => {
    expect(recognitionLabel('unreadable')).toBe('Unreadable');
    expect(recognitionLabel('unknown')).toBe('Unavailable');
  });
  it('only shows current tracks when the live frame age is known and at most five seconds', () => {
    expect(recognitionIsFresh(status)).toBe(true);
    expect(recognitionIsFresh({ ...status, frame_age_seconds: 5 })).toBe(true);
    for (const age of [undefined, null, -1, 5.01, Number.NaN, Infinity]) {
      expect(recognitionIsFresh({ ...status, frame_age_seconds: age })).toBe(false);
    }
    expect(recognitionIsFresh({ ...status, status: 'connecting' })).toBe(false);
    expect(recognitionIsFresh(null)).toBe(false);
  });
  it('never presents recorded footage as a live source', () => {
    expect(sourceLabel('recorded')).toBe('RECORDED');
    expect(sourceLabel('synthetic')).toBe('SIMULATED');
    expect(sourceLabel('live')).toBe('LIVE SOURCE');
    expect(sourceDescription('recorded')).toBe('Recorded footage');
    expect(sessionLabel({ ...recording, status: 'processing' })).toBe('Analysing recording');
    expect(sessionLabel({ ...recording, status: 'completed' })).toBe('Recording finished');
    expect(sessionLabel({ ...recording, status: 'paused' })).toBe('Recording paused');
    expect(sessionLabel({ ...status, status: 'paused', camera_id: null })).toBe('Paused');
    expect(sessionLabel(status)).toBe('Live analysis');
    expect(recognitionLabel('completed')).toBe('Completed');
  });
  it('treats a recording as current while analysing, paused or finished, within its own limit', () => {
    for (const state of ['processing', 'paused', 'completed'] as const) {
      expect(recognitionIsFresh({ ...recording, status: state })).toBe(true);
    }
    expect(recognitionIsFresh({ ...recording, frame_age_seconds: 12 })).toBe(true);   // slow CPU, still current
    expect(recognitionIsFresh({ ...recording, frame_age_seconds: 30.1 })).toBe(false);
    expect(recognitionIsFresh({ ...recording, status: 'connecting' })).toBe(false);
    expect(recognitionIsFresh({ ...recording, status: 'unavailable' })).toBe(false);
    expect(staleLimit(recording)).toBe(30);
    expect(staleLimit({ ...recording, stale_after_seconds: 99999 })).toBe(5);
    // A camera can never borrow the longer recorded limit or the recorded states.
    expect(staleLimit({ ...status, stale_after_seconds: 30 })).toBe(5);
    expect(recognitionIsFresh({ ...status, stale_after_seconds: 30, frame_age_seconds: 12 })).toBe(false);
    expect(recognitionIsFresh({ ...status, status: 'processing' })).toBe(false);
  });
  it('keeps a held recording alive but not a stopped or failed session', () => {
    for (const state of ['connecting', 'processing', 'paused', 'completed'] as const) {
      expect(recognitionIsRunning({ ...recording, status: state })).toBe(true);
    }
    expect(recognitionIsRunning({ ...recording, status: 'unavailable' })).toBe(false);
    expect(recognitionIsRunning({ ...status, status: 'paused', camera_id: null })).toBe(false);
    expect(recognitionIsRunning(status)).toBe(true);
    expect(recognitionIsRunning(null)).toBe(false);
  });
  it('shows video time without inventing a position', () => {
    expect(formatVideoTime(0)).toBe('0:00');
    expect(formatVideoTime(65_400)).toBe('1:05');
    expect(formatVideoTime(3_723_000)).toBe('1:02:03');
    for (const value of [undefined, null, -1, Number.NaN, Infinity]) expect(formatVideoTime(value)).toBe('—');
  });
  it('explains failure codes in plain words and never echoes unknown text as fact', () => {
    expect(failureMessage('ChangedRecording')).toContain('permission receipt');
    expect(failureMessage('MissingPts')).toContain('timestamps');
    expect(failureMessage('SomethingNew')).toBe('Analysis stopped (SomethingNew).');
    expect(failureMessage(undefined)).toBe('');
  });
  it('keeps unavailable counts distinct from measured zero', () => {
    expect(recognitionCount(0)).toBe('0');
    expect(recognitionCount(undefined)).toBe('—');
    expect(recognitionCount(-1)).toBe('—');
    expect(recognitionCount(Number.NaN)).toBe('—');
  });
});
