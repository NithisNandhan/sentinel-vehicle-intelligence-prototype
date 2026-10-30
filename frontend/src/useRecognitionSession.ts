import { useCallback, useEffect, useRef, useState } from 'react';
import { controlRecognition, getRecognitionEvidence, getRecognitionRecordings, getRecognitionStatus, heartbeatRecognition, setRecognitionSession, type PlaybackAction, type RecognitionEvidence, type RecognitionRecordings, type RecognitionStatus } from './api';
import { recognitionIsRunning } from './recognition';

/** Owned by App, not a navigation page: switching views cannot reset the worker. */
export function useRecognitionSession(enabled: boolean) {
  const [status, setStatus] = useState<RecognitionStatus | null>(null);
  const [recordings, setRecordings] = useState<RecognitionRecordings['recordings']>([]);
  const [evidence, setEvidence] = useState<RecognitionEvidence | null>(null);
  const [error, setError] = useState('');
  const [evidenceError, setEvidenceError] = useState('');
  const [recordingError, setRecordingError] = useState('');
  const [mode, setModeState] = useState<'original' | 'overlays'>('overlays');
  const [busy, setBusy] = useState(false);
  const active = useRef<RecognitionStatus | null>(null);
  const generation = useRef(0);
  const changing = useRef(false);
  const modeRef = useRef(mode);
  const alive = useRef(true);

  useEffect(() => { alive.current = true; return () => { alive.current = false; generation.current++; }; }, []);
  const refresh = useCallback(async () => {
    if (!enabled || changing.current) return;
    const before = generation.current;
    try {
      const next = await getRecognitionStatus();
      if (!alive.current || before !== generation.current) return;
      active.current = next; setStatus(next); setError('');
      if (next.source_mode !== 'recorded' || !next.camera_id) { setEvidence(null); return; }
      try {
        const view = await getRecognitionEvidence(modeRef.current);
        if (alive.current && before === generation.current) {
          setEvidence(view?.session_id === next.session_id ? view : null);
          setEvidenceError('');
        }
      } catch (failure) {
        if (alive.current && before === generation.current) { setEvidence(null); setEvidenceError((failure as Error).message); }
      }
    } catch (failure) {
      if (alive.current && before === generation.current) { setEvidence(null); setError((failure as Error).message); }
    }
  }, [enabled]);

  useEffect(() => {
    if (!enabled) { active.current = null; setStatus(null); setEvidence(null); return; }
    let cancelled = false, polling = false, heartbeating = false;
    async function poll() {
      if (cancelled || polling || document.visibilityState !== 'visible') return;
      polling = true; try { await refresh(); } finally { polling = false; }
    }
    async function sources() {
      try { const next = await getRecognitionRecordings(); if (!cancelled) { setRecordings(next.recordings); setRecordingError(''); } }
      catch (failure) { if (!cancelled) setRecordingError((failure as Error).message); }
    }
    async function heartbeat() {
      const current = active.current;
      if (cancelled || heartbeating || changing.current || document.visibilityState !== 'visible' || !current?.camera_id || !recognitionIsRunning(current)) return;
      heartbeating = true;
      try { await heartbeatRecognition(current.camera_id); }
      catch { if (!cancelled) void poll(); }
      finally { heartbeating = false; }
    }
    const visibility = () => { generation.current++; setEvidence(null); if (document.visibilityState === 'visible') { void poll(); void sources(); } };
    void poll(); void sources();
    const timer = window.setInterval(() => void poll(), 1000);
    const keepalive = window.setInterval(() => void heartbeat(), 20_000);
    const listTimer = window.setInterval(() => void sources(), 30_000);
    document.addEventListener('visibilitychange', visibility);
    return () => { cancelled = true; generation.current++; window.clearInterval(timer); window.clearInterval(keepalive); window.clearInterval(listTimer); document.removeEventListener('visibilitychange', visibility); };
  }, [enabled, refresh]);

  async function apply(change: () => Promise<unknown>) {
    changing.current = true; setBusy(true); setEvidence(null); setError(''); generation.current++;
    const before = generation.current;
    try {
      const next = await change() as RecognitionStatus;
      if (alive.current && before === generation.current) { active.current = next; setStatus(next); }
    } catch (failure) { if (alive.current && before === generation.current) setError((failure as Error).message); }
    finally { changing.current = false; if (alive.current) setBusy(false); }
    await refresh();
  }
  function setMode(next: 'original' | 'overlays') { modeRef.current = next; setModeState(next); generation.current++; setEvidence(null); void refresh(); }
  function start(id: string) {
    return apply(async () => {
      if (active.current?.camera_id && active.current.camera_id !== id) await setRecognitionSession(null);
      if (active.current?.camera_id === id && active.current.status === 'completed') return controlRecognition('restart');
      return setRecognitionSession(id);
    });
  }
  return { status, recordings, evidence, error, evidenceError, recordingError, busy, mode, setMode, refresh,
    start, stop: () => apply(() => setRecognitionSession(null)), control: (action: PlaybackAction) => apply(() => controlRecognition(action)) };
}
