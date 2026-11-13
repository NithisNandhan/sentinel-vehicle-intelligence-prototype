import { useEffect, useRef, useState } from 'react';
import { Pause, Play, X } from 'lucide-react';
import { getRecognitionReplay, type RecognitionReplaySample } from './api';
import { formatVideoTime } from './recognition';
import { loadReplay, ReplayClock, replaySampleAt, type ReplayData } from './replay';

export default function ReplayPanel({ sessionId, cameraId, mode, onSample, onExit }: {
  sessionId: string; cameraId: string; mode: 'original' | 'overlays';
  onSample: (sample: RecognitionReplaySample | null, positionMs: number) => void; onExit: () => void;
}) {
  const [data, setData] = useState<ReplayData | null>(null);
  const [progress, setProgress] = useState({ loaded: 0, total: 0 });
  const [error, setError] = useState('');
  const [playing, setPlaying] = useState(false);
  const [position, setPosition] = useState(0);
  const clock = useRef<ReplayClock | null>(null);
  const shown = useRef<RecognitionReplaySample | null>(null);

  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    setData(null); setError(''); setPlaying(false); setPosition(0); setProgress({ loaded: 0, total: 0 });
    shown.current = null; onSample(null, 0);
    void loadReplay((offset, signal) => getRecognitionReplay(sessionId, offset, mode, signal),
      { sessionId, cameraId }, { signal: controller.signal,
        progress: (loaded, total) => { if (active) setProgress({ loaded, total }); },
      }).then(value => {
      if (!active) return;
      clock.current = new ReplayClock(value.durationMs);
      clock.current.play();
      setData(value); setPlaying(clock.current.playing);
      const first = replaySampleAt(value.samples, 0);
      shown.current = first; onSample(first, 0);
    }).catch(failure => { if (active) setError(failure instanceof Error ? failure.message : 'Replay could not be loaded.'); });
    return () => { active = false; controller.abort(); clock.current?.pause(); clock.current = null; };
  }, [sessionId, cameraId, mode, onSample]);

  useEffect(() => {
    if (!data || !playing || !clock.current) return;
    let handle = 0, lastPaint = -Infinity;
    const tick = (now: number) => {
      const current = clock.current;
      if (!current) return;
      const value = current.read();
      if (now - lastPaint >= 80 || !current.playing) { setPosition(value); lastPaint = now; }
      const sample = replaySampleAt(data.samples, value);
      if (sample !== shown.current || !current.playing) { shown.current = sample; onSample(sample, value); }
      if (current.playing) handle = window.requestAnimationFrame(tick);
      else setPlaying(false);
    };
    handle = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(handle);
  }, [data, playing, onSample]);

  function toggle() {
    const current = clock.current;
    if (!current) return;
    if (current.playing) current.pause(); else current.play();
    setPlaying(current.playing); setPosition(current.read());
    if (data) { const sample = replaySampleAt(data.samples, current.read()); shown.current = sample; onSample(sample, current.read()); }
  }
  function seek(value: number) {
    if (!data || !clock.current) return;
    clock.current.seek(value); setPosition(clock.current.read()); setPlaying(clock.current.playing);
    const sample = replaySampleAt(data.samples, clock.current.read());
    shown.current = sample; onSample(sample, clock.current.read());
  }

  return <section className="surface replay-panel" aria-label="Recorded analysis replay">
    <div className="section-heading"><h2>Analysis replay · 1×</h2><button className="button quiet" onClick={onExit}><X size={15} />{data || error ? 'Exit replay' : 'Cancel replay loading'}</button></div>
    {error ? <div className="notice error" role="alert">Replay unavailable: {error}</div>
      : !data ? <div role="status" className="replay-loading"><span>Loading saved analysis samples{progress.total ? `: ${progress.loaded} / ${progress.total}` : '…'}</span><progress value={progress.loaded} max={progress.total || 1} /></div>
        : <div className="replay-controls"><button className="button" onClick={toggle}>{playing ? <Pause size={15} /> : <Play size={15} />}{playing ? 'Pause replay' : position >= data.durationMs ? 'Replay again' : 'Play replay'}</button><label className="replay-scrub"><span className="sr-only">Replay source position</span><input type="range" min={0} max={data.durationMs} step={1} value={position} onChange={event => seek(Number(event.target.value))} /><span className="mono">{formatVideoTime(position)} / {formatVideoTime(data.durationMs)}</span></label></div>}
    <p className="caption">Saved analysis samples at 1× source-video time. Frames and readings update at their original sample positions. No AI is rerun and replay creates no alerts. Exit replay to inspect the latest retained diagnostics.</p>
  </section>;
}
