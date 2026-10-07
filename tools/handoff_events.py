"""Snapshot and restore event metadata with no credentials or media URLs.

PostgreSQL export uses a repeatable-read transaction. SQLite uses online backup.
Restore refuses to replace a populated central database.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

TABLES = ('cameras', 'sightings', 'watchlist_entries', 'alerts', 'alert_audit', 'health_reports')
SECRETS = [v for k, v in os.environ.items() if v and len(v) >= 8 and
           any(part in k.upper() for part in ('PASSWORD', 'COOKIE', 'API_KEY', 'HMAC_KEY', 'TOKEN'))]


def scrub(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, str):
        for secret in SECRETS:
            value = value.replace(secret, '[credential removed]')
        value = re.sub(r'(?:rtsp|https?)://[^\s/]+:[^\s/]+@[^\s]+', '[source URL removed]', value)
    return value


def connect():
    import psycopg
    from psycopg.rows import dict_row
    url = os.environ.get('DATABASE_URL')
    if url:
        return psycopg.connect(url.replace('postgresql+psycopg://', 'postgresql://', 1),
                              row_factory=dict_row)
    return psycopg.connect(host='database', dbname='sentinel', user='sentinel',
                          password=os.environ['DB_PASSWORD'], row_factory=dict_row)


def export_central(output):
    from psycopg import sql
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'central-events.json'
    if path.exists():
        raise ValueError('Existing snapshot preserved')
    rows = {}
    with connect() as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        for table in TABLES:
            data = db.execute(sql.SQL('SELECT * FROM {} ORDER BY id').format(sql.Identifier(table))).fetchall()
            rows[table] = [{key: None if key == 'preview_url' else scrub(value)
                            for key, value in row.items()} for row in data]
        migrations = db.execute('SELECT version_num FROM alembic_version').fetchall()
    record = {'format': 'sentinel-event-metadata-v1',
              'captured_at_utc': datetime.now(timezone.utc).isoformat(),
              'migrations': [row['version_num'] for row in migrations],
              'tables': rows, 'counts': {key: len(value) for key, value in rows.items()},
              'redactions': 'Credentials and camera preview URLs excluded. Media never exported.',
              'identity_note': 'Existing plate HMACs are preserved; their original secret is not exported.'}
    path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'central_snapshot_counts': record['counts']}), flush=True)


def export_outboxes(output, sources):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for source in sources:
        directory, label = source.rsplit('=', 1)
        for path in sorted(Path(directory).glob('*.sqlite*')):
            if path.name.endswith(('-shm', '-wal')):
                continue
            destination = output / f'{label}-{path.name}'
            if destination.exists():
                raise ValueError('Existing SQLite snapshot preserved')
            # Read-only Docker mounts cannot create SQLite WAL shared-memory
            # locks. Copy a stable DB/WAL pair into a disposable working folder;
            # SQLite then recovers and backs up that copy, never the source.
            with tempfile.TemporaryDirectory(prefix='sentinel-snapshot-') as temp:
                copied = Path(temp) / path.name
                related = [path, Path(str(path) + '-wal')]
                for attempt in range(8):
                    before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
                              for p in related if p.exists()}
                    for p in related:
                        target = Path(temp) / p.name
                        if p.exists():
                            shutil.copyfile(p, target)
                        else:
                            target.unlink(missing_ok=True)
                    after = {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
                             for p in related if p.exists()}
                    if before == after:
                        break
                else:
                    raise ValueError('SQLite source remained busy; no inconsistent snapshot exported')
                with sqlite3.connect(copied) as original, sqlite3.connect(destination) as backup:
                    original.backup(backup)
                    tables = {row[0] for row in backup.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                    count = 0
                    if 'outbox' in tables:
                        for event_id, payload in backup.execute('SELECT event_id,payload FROM outbox').fetchall():
                            clean = scrub(payload)
                            if clean != payload:
                                backup.execute('UPDATE outbox SET payload=? WHERE event_id=?', (clean, event_id))
                        count = backup.execute('SELECT COUNT(*) FROM outbox').fetchone()[0]
            summary.append({'file': destination.name, 'queued_events': count})
    (output / 'outboxes.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'outbox_snapshots': summary}), flush=True)


def restore_central(path):
    from psycopg import sql
    record = json.loads(Path(path).read_text(encoding='utf-8'))
    if record.get('format') != 'sentinel-event-metadata-v1' or set(record['tables']) != set(TABLES):
        raise ValueError('Unsupported event snapshot')
    with connect() as db:
        for table in TABLES:
            if db.execute(sql.SQL('SELECT COUNT(*) AS count FROM {}').format(sql.Identifier(table))).fetchone()['count']:
                raise ValueError('Central restore requires a migrated, empty database; existing records preserved')
        for table in TABLES:
            for row in record['tables'][table]:
                columns = list(row)
                query = sql.SQL('INSERT INTO {} ({}) VALUES ({})').format(
                    sql.Identifier(table), sql.SQL(',').join(map(sql.Identifier, columns)),
                    sql.SQL(',').join(sql.Placeholder() for _ in columns))
                db.execute(query, [row[column] for column in columns])
            if table != 'cameras':
                db.execute(sql.SQL("SELECT setval(pg_get_serial_sequence(%s,'id'),"
                                   "COALESCE((SELECT MAX(id) FROM {}),1),"
                                   "EXISTS(SELECT 1 FROM {}))").format(sql.Identifier(table), sql.Identifier(table)),
                           (table,))
    print('Restored central metadata, alerts, operator audits and health records.', flush=True)


def restore_outbox(snapshot, destination):
    snapshot, destination = Path(snapshot), Path(destination)
    if destination.exists():
        raise ValueError('Existing outbox preserved; restore only into a fresh destination')
    destination.parent.mkdir(parents=True, exist_ok=True)
    # These are finalized SQLite backup snapshots, not active databases. WAL
    # mode may persist in the header even though all pages are in this file.
    # Immutable reads avoid creating WAL/SHM files on read-only Docker mounts.
    with sqlite3.connect(f'file:{snapshot.as_posix()}?mode=ro&immutable=1', uri=True) as source:
        if source.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('SQLite snapshot failed its integrity check')
        with sqlite3.connect(destination) as target:
            source.backup(target)
    print('Restored SQLite snapshot without replacing an existing database.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    central = sub.add_parser('export-central')
    central.add_argument('--output', required=True)
    outboxes = sub.add_parser('export-outboxes')
    outboxes.add_argument('--output', required=True)
    outboxes.add_argument('--source', required=True, action='append')
    restore = sub.add_parser('restore-central')
    restore.add_argument('--snapshot', required=True)
    queue = sub.add_parser('restore-outbox')
    queue.add_argument('--snapshot', required=True)
    queue.add_argument('--destination', required=True)
    args = parser.parse_args()
    if args.command == 'export-central':
        export_central(args.output)
    elif args.command == 'export-outboxes':
        export_outboxes(args.output, args.source)
    elif args.command == 'restore-central':
        restore_central(args.snapshot)
    else:
        restore_outbox(args.snapshot, args.destination)


if __name__ == '__main__':
    main()
