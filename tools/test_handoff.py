"""Run with python -m unittest discover -s tools -p test_handoff.py."""

import contextlib
import io
import json
import os
import sqlite3
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

import handoff_archives as archives
import handoff_events as events
import publish_handoff_release as publisher
import finish_handoff_upload as finisher


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.output = self.root / 'assets'

    def tearDown(self):
        self.temp.cleanup()

    def export(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return archives.export('test', [(self.source, '')], self.output, part_mib=1)

    def test_multipart_roundtrip_preserves_checkpoint(self):
        content = os.urandom(2 * 1024 * 1024 + 100)
        (self.source / 'model.pt').write_bytes(content)
        (self.source / '.env').write_text('PASSWORD=not-to-export')
        manifest = self.export()
        self.assertGreater(len(manifest['parts']), 1)
        destination = self.root / 'restored'
        with contextlib.redirect_stdout(io.StringIO()):
            archives.restore(self.output / 'test.manifest.json', destination)
        self.assertEqual((destination / 'model.pt').read_bytes(), content)
        self.assertFalse((destination / '.env').exists())

    def test_corrupted_archive_is_not_extracted(self):
        (self.source / 'file.txt').write_text('important')
        manifest = self.export()
        part = self.output / manifest['parts'][0]['name']
        part.write_bytes(b'corrupt')
        destination = self.root / 'restored'
        with self.assertRaisesRegex(ValueError, 'integrity mismatch'):
            archives.restore(self.output / 'test.manifest.json', destination)
        self.assertFalse(destination.exists())

    def test_nonempty_destination_is_preserved(self):
        (self.source / 'file.txt').write_text('new')
        self.export()
        destination = self.root / 'restored'
        destination.mkdir()
        (destination / 'file.txt').write_text('existing')
        with self.assertRaisesRegex(ValueError, 'must be empty'):
            archives.restore(self.output / 'test.manifest.json', destination)
        self.assertEqual((destination / 'file.txt').read_text(), 'existing')

    def test_archive_cannot_escape_destination(self):
        self.output.mkdir()
        part = self.output / 'test.tar.gz.part001'
        with tarfile.open(part, 'w:gz') as archive:
            member = tarfile.TarInfo('../escaped.txt')
            member.size = 4
            archive.addfile(member, io.BytesIO(b'evil'))
        manifest = {'format': 'sentinel-multipart-tar-gzip-v1', 'name': 'test',
                    'files': 1, 'parts': [{'name': part.name, 'bytes': part.stat().st_size,
                                          'sha256': archives.digest_file(part)}]}
        path = self.output / 'test.manifest.json'
        path.write_text(json.dumps(manifest))
        with self.assertRaises(tarfile.FilterError):
            archives.restore(path, self.root / 'restored')
        self.assertFalse((self.root / 'escaped.txt').exists())

    def test_existing_export_is_preserved(self):
        (self.source / 'file.txt').write_text('test')
        self.export()
        with self.assertRaisesRegex(ValueError, 'Existing archive preserved'):
            self.export()

    def test_resume_reproduces_existing_parts_and_completes_partial_tail(self):
        (self.source / 'model.pt').write_bytes(os.urandom(2 * 1024 * 1024 + 100))
        with contextlib.redirect_stdout(io.StringIO()):
            original = archives.export('test', [(self.source, '')], self.output, 1,
                                       stream_buffer=10240)
        manifest_path = self.output / 'test.manifest.json'
        manifest_path.unlink()
        last = self.output / original['parts'][-1]['name']
        last.write_bytes(last.read_bytes()[:-100])
        with contextlib.redirect_stdout(io.StringIO()):
            resumed = archives.export('test', [(self.source, '')], self.output, 1, resume=True)
        self.assertEqual(resumed['parts'], original['parts'])

    def test_sqlite_backup_preserves_source_and_queue(self):
        path = self.source / 'outbox.sqlite3'
        secret_url = 'rtsp://email%40example.com:password@camera/stream/cam01'
        with sqlite3.connect(path) as db:
            db.execute('CREATE TABLE outbox (event_id TEXT PRIMARY KEY,payload TEXT)')
            db.execute('INSERT INTO outbox VALUES (?,?)', ('one', secret_url))
        with contextlib.redirect_stdout(io.StringIO()):
            events.export_outboxes(self.output, [str(self.source) + '=edge'])
        with sqlite3.connect(path) as db:
            self.assertEqual(db.execute('SELECT payload FROM outbox').fetchone()[0], secret_url)
        with sqlite3.connect(self.output / 'edge-outbox.sqlite3') as db:
            self.assertEqual(db.execute('SELECT payload FROM outbox').fetchone()[0], '[source URL removed]')
        records = json.loads((self.output / 'outboxes.json').read_text())
        self.assertEqual(records[0]['queued_events'], 1)

    def test_draft_release_is_found_when_tag_endpoint_returns_not_found(self):
        github = Mock()
        draft = {'id': 123, 'tag_name': 'handoff-test', 'draft': True}
        github.request.side_effect = [None, [draft]]
        self.assertEqual(publisher.find_release(github, 'handoff-test'), draft)

    def test_empty_delete_response_is_supported(self):
        response = Mock(status=204)
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        with patch.object(publisher.urllib.request, 'urlopen', return_value=response):
            self.assertEqual(publisher.GitHub('test-token', 'owner/repo').request('DELETE', '/test'), {})

    def test_recovery_only_deletes_eligible_starter_without_digest(self):
        github = Mock()
        uploaded = {
            'failed.part001': {'id': 1, 'state': 'starter', 'size': 268435456, 'digest': None},
            'valid.part001': {'id': 2, 'state': 'uploaded', 'size': 100, 'digest': 'sha256:ok'},
            'other.part001': {'id': 3, 'state': 'starter', 'size': 0, 'digest': None},
            'nonempty.part001': {'id': 4, 'state': 'uploaded', 'size': 100, 'digest': None},
            'digested.part001': {'id': 5, 'state': 'starter', 'size': 100, 'digest': 'sha256:ok'},
        }
        with contextlib.redirect_stdout(io.StringIO()):
            publisher.recover_starter_assets(github, {'draft': True}, uploaded,
                                            {'failed.part001', 'valid.part001', 'nonempty.part001',
                                             'digested.part001'})
        github.request.assert_called_once_with('DELETE', '/releases/assets/1')
        self.assertEqual(set(uploaded), {'valid.part001', 'other.part001', 'nonempty.part001',
                                        'digested.part001'})

    def test_recovery_refuses_published_release(self):
        with self.assertRaisesRegex(ValueError, 'unpublished draft'):
            publisher.recover_starter_assets(Mock(), {'draft': False}, {}, set())

    def test_finish_requires_every_bundle_manifest(self):
        self.output.mkdir()
        for name in finisher.REQUIRED_BUNDLES - {'benchmark-data'}:
            (self.output / f'{name}.manifest.json').write_text('{}')
        self.assertFalse(finisher.complete_inventory([self.output]))
        (self.output / 'benchmark-data.manifest.json').write_text('{}')
        self.assertTrue(finisher.complete_inventory([self.output]))

    def test_outbox_restore_refuses_overwrite(self):
        source = self.source / 'queue.sqlite3'
        with sqlite3.connect(source) as db:
            db.execute('CREATE TABLE outbox (event_id TEXT,payload TEXT)')
            db.execute('INSERT INTO outbox VALUES (?,?)', ('one', '{}'))
        destination = self.root / 'restored.sqlite3'
        with contextlib.redirect_stdout(io.StringIO()):
            events.restore_outbox(source, destination)
        with self.assertRaisesRegex(ValueError, 'Existing outbox preserved'):
            events.restore_outbox(source, destination)
        with sqlite3.connect(destination) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM outbox').fetchone()[0], 1)

    def test_finalized_wal_snapshot_uses_immutable_read(self):
        source = self.source / 'wal-snapshot.sqlite3'
        db = sqlite3.connect(source)
        try:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE outbox (event_id TEXT,payload TEXT)')
            db.execute('INSERT INTO outbox VALUES (?,?)', ('one', '{}'))
            db.commit()
        finally:
            db.close()
        destination = self.root / 'restored-wal.sqlite3'
        with patch.object(events.sqlite3, 'connect', wraps=sqlite3.connect) as connect:
            with contextlib.redirect_stdout(io.StringIO()):
                events.restore_outbox(source, destination)
            self.assertIn('mode=ro&immutable=1', connect.call_args_list[0].args[0])
        with sqlite3.connect(destination) as restored:
            self.assertEqual(restored.execute('SELECT COUNT(*) FROM outbox').fetchone()[0], 1)


if __name__ == '__main__':
    unittest.main()
