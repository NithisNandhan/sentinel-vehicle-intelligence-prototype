import { useEffect, useRef, type ReactNode } from 'react';
import { X } from 'lucide-react';

export default function DetailSheet({ title, onClose, children, subtitle }: { title: string; subtitle?: string; onClose: () => void; children: ReactNode }) {
  const ref = useRef<HTMLDialogElement>(null);
  const close = useRef(onClose); close.current = onClose;
  useEffect(() => {
    const previouslyFocused = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const dialog = ref.current;
    dialog?.showModal();
    return () => { dialog?.close(); if (previouslyFocused?.isConnected) previouslyFocused.focus(); };
  }, []);
  return <dialog ref={ref} className="detail-sheet" aria-labelledby="sheet-title" onCancel={event => { event.preventDefault(); close.current(); }} onClick={event => { if (event.target === event.currentTarget) close.current(); }}>
    <div className="sheet-header"><div><h2 id="sheet-title">{title}</h2>{subtitle && <p>{subtitle}</p>}</div><button type="button" className="icon-button" aria-label="Close details" onClick={onClose} autoFocus><X size={20} /></button></div>
    <div className="sheet-content">{children}</div>
  </dialog>;
}
