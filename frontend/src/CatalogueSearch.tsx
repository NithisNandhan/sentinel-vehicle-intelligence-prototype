import { useEffect, useRef, useState } from 'react';
import { Camera as CameraIcon, MapPin, Play, Search, X } from 'lucide-react';
import {
  getCataloguePreview, heartbeatCataloguePreview, searchCatalogue,
  startCataloguePreview, stopCataloguePreview,
  type CatalogueCamera, type CatalogueSearchResponse, type PreviewSession,
} from './api';
import LivePlayer from './LivePlayer';

export default function CatalogueSearch() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<CatalogueSearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState('');
  const [selected, setSelected] = useState<CatalogueCamera | null>(null);
  const [session, setSession] = useState<PreviewSession | null>(null);
  const [previewError, setPreviewError] = useState('');
  const activeSession = useRef<string | null>(null);
  const previewGeneration = useRef(0);
  const currentQuery = useRef(query);
  currentQuery.current = query;

  useEffect(() => {
    const term = query.trim();
    if (term.length < 2) {
      setResults(null); setError(''); setLoading(false);
      return;
    }
    let cancelled = false;
    setLoading(true); setError(''); setResults(null);
    const timer = window.setTimeout(() => {
      void searchCatalogue(term).then(data => {
        if (!cancelled) setResults(data);
      }).catch((failure: Error) => {
        if (!cancelled) setError(failure.message);
      }).finally(() => { if (!cancelled) setLoading(false); });
    }, 300);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [query]);

  useEffect(() => () => {
    previewGeneration.current += 1;
    if (activeSession.current) void stopCataloguePreview(activeSession.current).catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!session || session.status !== 'starting' || previewError) return;
    let cancelled = false;
    const timer = window.setInterval(() => {
      void getCataloguePreview(session.id).then(next => {
        if (cancelled) return;
        setSession(next);
        if (next.status === 'failed') {
          setPreviewError('The camera did not become available. Please try again later.');
          activeSession.current = null;
          void stopCataloguePreview(next.id).catch(() => undefined);
        }
      }).catch((failure: Error) => {
        if (!cancelled) {
          setPreviewError(failure.message);
          activeSession.current = null;
          void stopCataloguePreview(session.id).catch(() => undefined);
        }
      });
    }, 1500);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [session?.id, session?.status, previewError]);

  useEffect(() => {
    if (!session || session.status === 'failed' || previewError) return;
    const timer = window.setInterval(() => {
      void heartbeatCataloguePreview(session.id).then(next => {
        setSession(next);
        if (next.status === 'failed') {
          setPreviewError('The camera connection was lost. Please try again later.');
          activeSession.current = null;
          void stopCataloguePreview(next.id).catch(() => undefined);
        }
      }).catch((failure: Error) => {
        setPreviewError(failure.message);
        activeSession.current = null;
        void stopCataloguePreview(session.id).catch(() => undefined);
      });
    }, 20_000);
    return () => window.clearInterval(timer);
  }, [session?.id, session?.status, previewError]);

  async function loadMore() {
    if (!results || loadingMore) return;
    const term = query.trim();
    setLoadingMore(true);
    try {
      const next = await searchCatalogue(term, results.items.length);
      if (currentQuery.current.trim() === term) {
        setResults({ ...next, items: [...results.items, ...next.items] });
      }
    } catch (failure) { setError((failure as Error).message); }
    finally { setLoadingMore(false); }
  }

  function closePreview() {
    previewGeneration.current += 1;
    const id = activeSession.current;
    activeSession.current = null;
    if (id) void stopCataloguePreview(id).catch(() => undefined);
    setSelected(null); setSession(null); setPreviewError('');
  }

  async function openPreview(camera: CatalogueCamera) {
    if (!camera.live) return;
    const generation = ++previewGeneration.current;
    setSelected(camera); setSession(null); setPreviewError('');
    try {
      const started = await startCataloguePreview(camera.id);
      if (generation !== previewGeneration.current) {
        void stopCataloguePreview(started.id).catch(() => undefined);
        return;
      }
      activeSession.current = started.id;
      setSession(started);
      if (started.status === 'failed') {
        setPreviewError('The camera did not become available. Please try again later.');
        activeSession.current = null;
        void stopCataloguePreview(started.id).catch(() => undefined);
      }
    } catch (failure) {
      if (generation === previewGeneration.current) setPreviewError((failure as Error).message);
    }
  }

  const hasMore = results && results.items.length < results.total;
  return <>
    <section className="panel catalogue-search-panel" aria-label="Find a camera">
      <div className="panel-heading"><div><span className="eyebrow">AUTHORISED CAMERA GRID</span><h2>Find a camera</h2></div><CameraIcon size={20} /></div>
      <p>Search by camera ID, name or place name. Opening a preview does not change the two AI-monitored feeds.</p>
      <label htmlFor="catalogue-query">Camera ID, name or location</label>
      <div className="catalogue-search-input"><Search size={18} /><input id="catalogue-query" value={query}
        onChange={event => setQuery(event.target.value)} placeholder="e.g. cam01 or Janpath"
        autoComplete="off" spellCheck={false} /></div>
      <div className="catalogue-search-feedback" aria-live="polite">
        {query.trim().length < 2 ? 'Enter at least two characters to search.' :
          loading ? 'Searching the authorised catalogue…' :
          error ? <span className="catalogue-error">{error === 'Sentinel catalogue unavailable' ?
            'Camera catalogue unavailable. The Sentinel sign-in or connection needs attention.' : error}</span> :
          results ? `${results.total} matching ${results.total === 1 ? 'camera' : 'cameras'}` : ''}
      </div>
      {!!results?.items.length && <div className="catalogue-results" role="list">
        {results.items.map(camera => <div className="catalogue-result" role="listitem" key={camera.id}>
          <div className="catalogue-result-main"><strong>{camera.name || camera.id}</strong><span className="catalogue-id">{camera.id}</span>
            <small><MapPin size={13} /> {camera.location || 'Location not provided'}{camera.codec ? ` · ${camera.codec}` : ''}</small></div>
          <div className="catalogue-result-actions"><span className={`pill ${camera.live ? 'good' : 'danger'}`}>{camera.live ? 'Live' : 'Offline'}</span>
            <button className="button" disabled={!camera.live} onClick={() => void openPreview(camera)}
              aria-label={`Open preview for ${camera.name || camera.id}`}><Play size={15} /> Preview</button></div>
        </div>)}
      </div>}
      {results && !results.items.length && <div className="catalogue-empty">No matching cameras found.</div>}
      {hasMore && <button className="button ghost catalogue-more" disabled={loadingMore} onClick={() => void loadMore()}>
        {loadingMore ? 'Loading…' : 'Load more cameras'}</button>}
    </section>

    {selected && <div className="modal-backdrop" onMouseDown={event => { if (event.target === event.currentTarget) closePreview(); }}>
      <section className="modal preview-modal" role="dialog" aria-modal="true" aria-label={`${selected.name} live preview`}>
        <div className="modal-header"><div><span className="eyebrow">ON-DEMAND CAMERA PREVIEW</span><h2>{selected.name || selected.id}</h2></div>
          <button className="icon-button" onClick={closePreview} aria-label="Close"><X size={19} /></button></div>
        {previewError ? <div className="catalogue-preview-state catalogue-error" role="alert">{previewError}</div> :
          session?.status === 'ready' ? <LivePlayer camera={{ id: selected.id, name: selected.name,
            status: 'ONLINE', source_mode: 'live', preview_url: session.preview_url }} /> :
            <div className="catalogue-preview-state">Connecting to {selected.id}…</div>}
        <div className="modal-footer"><span>{selected.location || selected.id} · Preview only</span>
          <button className="button" onClick={closePreview}>Close preview</button></div>
      </section>
    </div>}
  </>;
}
