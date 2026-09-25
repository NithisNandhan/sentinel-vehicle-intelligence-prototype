import { useEffect, useRef, useState } from 'react';
import Hls from 'hls.js';
import { cameraPreview, type Camera } from './api';

export default function LivePlayer({ camera }: { camera: Camera }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [started, setStarted] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    setStarted(false);
    setError('');
  }, [camera.id]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !started) return;
    const source = cameraPreview(camera);
    let hls: Hls | undefined;
    if (Hls.isSupported()) {
      hls = new Hls({ liveDurationInfinity: true, lowLatencyMode: true, enableWorker: true });
      hls.on(Hls.Events.ERROR, (_event, data) => {
        if (data.fatal) setError('Live preview is unavailable. The camera feed may be reconnecting.');
      });
      hls.loadSource(source);
      hls.attachMedia(video);
    } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
      video.src = source;
    } else {
      setError('This browser cannot play HLS video.');
    }
    return () => { hls?.destroy(); video.pause(); video.removeAttribute('src'); video.load(); };
  }, [camera, started]);

  return <div className="preview-box">
    {started ? <video ref={videoRef} controls autoPlay muted playsInline aria-label={`${camera.name} live feed`} /> :
      <div className="preview-idle"><span className="preview-lens" /><p>Live preview is paused</p><button className="button primary" onClick={() => setStarted(true)}>Open live feed</button></div>}
    {error && <div className="preview-error" role="alert">{error}</div>}
    <div className="preview-caption">Live only · No recording or seeking</div>
  </div>;
}
