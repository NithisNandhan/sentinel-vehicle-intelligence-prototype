import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react';
import { Plus, Search, ChevronLeft, ChevronRight } from 'lucide-react';
import { addWatchlist, getWatchlist, patchWatchlist, searchWatchlist, formatTime } from './api';
import type { WatchlistCreate, WatchlistItem, WatchlistPage as Page } from './unified-contract';
import DetailSheet from './DetailSheet';

export default function WatchlistPage() {
  const [page, setPage] = useState<Page | null>(null);
  const [query, setQuery] = useState('');
  const [search, setSearch] = useState('');
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [editing, setEditing] = useState<WatchlistItem | 'new' | null>(null);
  const [busy, setBusy] = useState(false);
  const [formError, setFormError] = useState('');
  const [plate, setPlate] = useState('');
  const [reason, setReason] = useState('');
  const [severity, setSeverity] = useState<WatchlistCreate['severity']>('high');
  const generation = useRef(0);
  useEffect(() => () => { generation.current++; }, []);
  const load = useCallback(async () => {
    const before = ++generation.current;
    setLoading(true);
    try {
      const next = search ? await searchWatchlist(search, offset) : await getWatchlist(offset);
      if (before === generation.current) { setPage(next); setError(''); }
    } catch (failure) { if (before === generation.current) setError((failure as Error).message); }
    finally { if (before === generation.current) setLoading(false); }
  }, [search, offset]);
  useEffect(() => { void load(); }, [load]);
  function edit(item: WatchlistItem | 'new') { setEditing(item); setFormError(''); setPlate(''); setReason(item === 'new' ? '' : item.reason); setSeverity(item === 'new' ? 'high' : item.severity); }
  function close() { if (busy) return; setEditing(null); setPlate(''); setReason(''); setFormError(''); }
  async function save(event: FormEvent) {
    event.preventDefault(); if (!editing) return; setBusy(true); setFormError('');
    try {
      if (editing === 'new') await addWatchlist({ plate: plate.trim(), reason: reason.trim(), severity });
      else await patchWatchlist(editing.id, { expected_version: editing.version, reason: reason.trim(), severity });
      setEditing(null); setPlate(''); setReason(''); setNotice('Watchlist saved. New accepted sightings use the updated list.'); await load();
    } catch (failure) { setPlate(''); setFormError((failure as Error).message); }
    finally { setBusy(false); }
  }
  async function toggle(item: WatchlistItem) {
    if (item.active && !window.confirm(`Deactivate ${item.masked_plate || `entry #${item.id}`}? Existing alerts and history will be preserved.`)) return;
    setBusy(true); setError('');
    try { await patchWatchlist(item.id, { expected_version: item.version, active: !item.active }); setNotice(item.active ? 'Entry deactivated. Existing alert history is preserved.' : 'Entry reactivated.'); await load(); }
    catch (failure) { setError((failure as Error).message); }
    finally { setBusy(false); }
  }
  return <>
    <div className="page-heading"><div><h1>Watchlist</h1><p>Manage identifiers that should raise an alert.</p></div><button className="button primary" onClick={() => edit('new')}><Plus size={16} /> Add plate</button></div>
    <form className="search-toolbar" onSubmit={event => { event.preventDefault(); setSearch(query.trim()); setQuery(''); setOffset(0); }}><label className="sr-only" htmlFor="watchlist-search">Search watchlist</label><input id="watchlist-search" value={query} onChange={event => setQuery(event.target.value)} placeholder="Search plate or reason" autoComplete="off" spellCheck={false} maxLength={240} /><button className="button" disabled={loading}><Search size={16} /> Search</button>{search && <button className="button quiet" type="button" onClick={() => { setSearch(''); setOffset(0); }}>Clear search</button>}</form>
    {error && <div className="notice error" role="alert">{error} <button className="text-button" onClick={() => void load()}>Retry / refresh</button></div>}
    {notice && <div className="notice success" role="status">{notice}</div>}
    <section className="surface" aria-busy={loading}><div className="section-heading padded"><h2>{search ? 'Search results' : 'Monitored identifiers'}</h2><span className="caption">{page ? `${page.total} entries · revision ${page.version}` : 'Loading'}</span></div>
      {loading && !page ? <div className="empty">Loading watchlist…</div> : !page?.items.length ? <div className="empty"><strong>{search ? 'No matching entries' : 'No watchlist entries'}</strong><p>{search ? 'Try another plate or reason.' : 'Add a plate to monitor future accepted sightings.'}</p></div> : <div className="table-scroll"><table className="data-table"><thead><tr><th>Identifier</th><th>Reason</th><th>Priority</th><th>State</th><th>Created</th><th><span className="sr-only">Actions</span></th></tr></thead><tbody>{page.items.map(item => <tr key={item.id}><td><strong className="mono">{item.masked_plate || `Entry #${item.id}`}</strong>{!item.masked_plate && <small>Legacy identifier hidden</small>}</td><td className="reason-cell">{item.reason}</td><td>{item.severity}</td><td><span className={`status ${item.active ? 'neutral' : 'muted'}`}>{item.active ? 'Active' : 'Inactive'}</span></td><td className="caption">{formatTime(item.created_at)}</td><td><div className="row-actions"><button className="text-button" disabled={busy} onClick={() => edit(item)}>Edit</button><button className={`text-button ${item.active ? 'danger-text' : ''}`} disabled={busy} onClick={() => void toggle(item)}>{item.active ? 'Deactivate' : 'Reactivate'}</button></div></td></tr>)}</tbody></table></div>}
      <div className="table-footer"><span>Identifiers are stored as keyed values, not plaintext plates.</span><div className="row-actions"><button className="icon-button" aria-label="Previous watchlist page" disabled={loading || !offset} onClick={() => setOffset(Math.max(0, offset - 25))}><ChevronLeft size={18} /></button><span>{page?.total ? `${offset + 1}–${Math.min(offset + 25, page.total)}` : '0'} of {page?.total ?? '—'}</span><button className="icon-button" aria-label="Next watchlist page" disabled={loading || !page || offset + page.limit >= page.total} onClick={() => setOffset(offset + 25)}><ChevronRight size={18} /></button></div></div>
    </section>
    {editing && <DetailSheet title={editing === 'new' ? 'Add a plate' : 'Edit watchlist entry'} subtitle="Changes apply to future accepted sightings" onClose={close}><form className="entry-form" onSubmit={event => void save(event)}>
      {formError && <div className="notice error" role="alert">{formError}<p>If another edit changed this entry, close this panel and refresh before retrying.</p></div>}
      {editing === 'new' ? <div className="form-field"><label htmlFor="watchlist-plate">Plate number</label><input id="watchlist-plate" value={plate} onChange={event => setPlate(event.target.value)} autoComplete="off" spellCheck={false} maxLength={40} required placeholder="GJ01AB1234" /></div> : <p className="mono">{editing.masked_plate || `Legacy entry #${editing.id}`}</p>}
      <div className="form-field"><label htmlFor="watchlist-reason">Reason</label><textarea id="watchlist-reason" value={reason} onChange={event => setReason(event.target.value)} maxLength={240} rows={4} required /></div>
      <div className="form-field"><label htmlFor="watchlist-priority">Priority</label><select id="watchlist-priority" value={severity} onChange={event => setSeverity(event.target.value as WatchlistCreate['severity'])}>{['low', 'medium', 'high', 'critical'].map(value => <option key={value} value={value}>{value}</option>)}</select></div>
      <p className="caption">The server creates a keyed identifier. Full plate input is not retained in this browser after saving. Editing an identifier requires deactivating it and adding a new plate.</p>
      <div className="row-actions"><button className="button primary" disabled={busy || !reason.trim() || (editing === 'new' && !plate.trim())}>{busy ? 'Saving…' : 'Save entry'}</button><button type="button" className="button quiet" onClick={close} disabled={busy}>Cancel</button></div>
    </form></DetailSheet>}
  </>;
}
