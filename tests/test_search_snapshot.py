from __future__ import annotations

import json
import os
import sqlite3
from types import SimpleNamespace

import pytest

from src import indexing_lock, search_snapshot as snapshots
from src.env_utils import environment_with_saved_dotenv
from scripts import publish_search_snapshot as publisher, run_searchable_index as runner


@pytest.fixture(autouse=True)
def clear_reader_environment(monkeypatch):
    for key in (*snapshots.PATH_KEYS, snapshots.ORIGINAL_ENV, snapshots.ACTIVE_ROOT):
        monkeypatch.setenv(key, os.environ.get(key, ""))
        monkeypatch.delenv(key)


def fake_snapshot(root):
    directory = root / 'data/search_snapshots/first'
    (directory / 'chroma').mkdir(parents=True)
    for name in ('chroma/chroma.sqlite3', 'chroma/embedder_config_v3.json', 'lexical_v3.sqlite3', 'relations.db'):
        (directory / name).touch()
    (directory / 'manifest_v3.json').write_text(json.dumps({'hnsw_validated': True}))
    record = {'directory': str(directory), 'generation': 'first', 'ready': True,
              'attachments': 2, 'published_at': '2026-10-08', 'audit_passed': False}
    (directory / 'snapshot.json').write_text(json.dumps(record))
    snapshots.pointer_path(root).write_text(json.dumps(record))
    return directory


def test_reader_paths_remain_separate_while_writer_is_live(tmp_path, monkeypatch):
    directory = fake_snapshot(tmp_path)
    active = str(tmp_path / 'data/chroma')
    monkeypatch.setenv('CHROMA_DIR', active)
    snapshots.apply_search_snapshot(tmp_path)
    assert os.environ['CHROMA_DIR'] == str(directory / 'chroma')
    snapshots.apply_search_snapshot(tmp_path)  # nested imports retain canonical originals
    with indexing_lock.held(indexing_lock.default_path(tmp_path)):
        with indexing_lock.native_read_access(snapshots.read_lock_path(tmp_path)):
            assert snapshots.load_snapshot(tmp_path)['attachments'] == 2
        with pytest.raises(RuntimeError):
            with indexing_lock.native_read_access(indexing_lock.default_path(tmp_path)):
                pass
        child = environment_with_saved_dotenv(tmp_path)
        assert child['CHROMA_DIR'] == active
        assert 'MANIFEST_PATH' not in child
        assert snapshots.ACTIVE_ROOT not in child
        assert snapshots.ORIGINAL_ENV not in child
    assert snapshots.public_status(tmp_path)['available']
    assert snapshots.public_status(tmp_path)['audit_passed'] is False
    monkeypatch.setenv('CHROMA_DIR', active)
    with pytest.raises(ValueError, match='changed'):
        snapshots.read_lock_path(tmp_path)


@pytest.mark.parametrize('damage', ['outside', 'not-ready', 'generation', 'manifest', 'inflight', 'pending', 'missing', 'symlink'])
def test_invalid_snapshot_never_falls_back_to_live_writer(tmp_path, damage):
    directory = fake_snapshot(tmp_path)
    if damage == 'outside':
        snapshots.pointer_path(tmp_path).write_text(json.dumps({'directory': str(tmp_path)}))
    elif damage in ('not-ready', 'generation'):
        (directory / 'snapshot.json').write_text(json.dumps({'ready': damage == 'generation', 'generation': 'wrong'}))
    elif damage in ('manifest', 'inflight', 'pending'):
        data = {'hnsw_validated': damage != 'manifest'}
        data[{'inflight': 'inflight_attachments', 'pending': 'post_index_pending'}.get(damage, 'unused')] = ['partial']
        (directory / 'manifest_v3.json').write_text(json.dumps(data))
    else:
        p = directory / 'relations.db'
        p.unlink()
        if damage == 'symlink':
            other = tmp_path / 'live.db'; other.touch(); p.symlink_to(other)
    with pytest.raises(ValueError):
        snapshots.apply_search_snapshot(tmp_path)
    assert snapshots.public_status(tmp_path)['available'] is False


def test_absent_snapshot_preserves_historical_lock(tmp_path, monkeypatch):
    assert snapshots.apply_search_snapshot(tmp_path) is None
    assert snapshots.public_status(tmp_path) == {'available': False}
    assert snapshots.read_lock_path(tmp_path) == indexing_lock.default_path(tmp_path)
    assert snapshots.writer_environment({'A': 'b'}) == {'A': 'b'}
    monkeypatch.setenv(snapshots.ACTIVE_ROOT, str(tmp_path))
    with pytest.raises(ValueError, match='disappeared'):
        snapshots.apply_search_snapshot(tmp_path)
    with pytest.raises(ValueError, match='canonical paths'):
        snapshots.writer_environment(os.environ)


@pytest.mark.parametrize('target,value', [('pointer', None), ('snapshot.json', None),
                                        ('manifest_v3.json', None), ('snapshot.json', {'ready': True, 'generation': 'first'})])
def test_malformed_certificates_are_reported_without_breaking_admin(tmp_path, target, value):
    directory = fake_snapshot(tmp_path)
    path = snapshots.pointer_path(tmp_path) if target == 'pointer' else directory / target
    path.write_text(json.dumps(value))
    assert snapshots.public_status(tmp_path)['available'] is False


@pytest.mark.parametrize('value', ['[]', '{}', json.dumps({key: 7 for key in snapshots.PATH_KEYS})])
def test_writer_rejects_unrestorable_reader_paths(value):
    with pytest.raises(ValueError):
        snapshots.writer_environment({snapshots.ORIGINAL_ENV: value})


def native_source(root, monkeypatch):
    chromadb = pytest.importorskip('chromadb')
    directory = root / 'data/chroma'; directory.mkdir(parents=True)
    client = chromadb.PersistentClient(path=str(directory))
    try:
        col = client.create_collection('zotero_paragraphs_v3', embedding_function=None,
                                      configuration={'hnsw': {'space': 'cosine', 'num_threads': 1}})
        col.add(ids=['old1', 'old2'], embeddings=[[1., 0., 0.], [0., 1., 0.]], documents=['old first', 'old second'])
    finally:
        client.close()
    (directory / 'embedder_config_v3.json').write_text('{}')
    (root / 'data/manifest_v3.json').write_text(json.dumps({'hnsw_validated': True, 'files': {'old': {}}}))
    for name in ('lexical_v3.sqlite3', 'relations.db'):
        with sqlite3.connect(root / 'data' / name) as db:
            db.execute('create table records(value)'); db.execute("insert into records values('old')")
    monkeypatch.setenv('CHROMA_DIR', str(directory))
    monkeypatch.setenv('MANIFEST_PATH', str(root / 'data/manifest_v3.json'))
    monkeypatch.setenv('LEXICAL_DB_PATH', str(root / 'data/lexical_v3.sqlite3'))
    monkeypatch.setenv('RELATIONS_DB_PATH', str(root / 'data/relations.db'))
    return chromadb, directory


def test_native_queries_keep_old_generation_during_real_write(tmp_path, monkeypatch):
    chromadb, source = native_source(tmp_path, monkeypatch)
    record = publisher.publish(tmp_path, initialize=True)
    saved = snapshots.pointer_path(tmp_path).read_bytes()
    assert publisher.publish(tmp_path, initialize=True)['generation'] == record['generation']
    with pytest.raises(ValueError, match='full DB audit'):
        publisher.publish(tmp_path)
    snapshots.apply_search_snapshot(tmp_path)
    with indexing_lock.held(indexing_lock.default_path(tmp_path)):
        writer = chromadb.PersistentClient(path=str(source))
        reader = chromadb.PersistentClient(path=os.environ['CHROMA_DIR'])
        try:
            writer.get_collection('zotero_paragraphs_v3', embedding_function=None).add(
                ids=['new'], embeddings=[[0., 0., 1.]], documents=['new document'])
            col = reader.get_collection('zotero_paragraphs_v3', embedding_function=None)
            assert col.count() == 2
            assert col.query(query_embeddings=[[1., 0., 0.]], n_results=1)['ids'][0] == ['old1']
            assert writer.get_collection('zotero_paragraphs_v3', embedding_function=None).count() == 3
        finally:
            reader.close(); writer.close()
    assert snapshots.pointer_path(tmp_path).read_bytes() == saved


def test_failed_refresh_keeps_previous_copy_and_requires_ready_source(tmp_path, monkeypatch):
    native_source(tmp_path, monkeypatch)
    old = publisher.publish(tmp_path, initialize=True)
    gate = tmp_path / 'data/quality'; gate.mkdir()
    (gate / 'server_database_gate.json').write_text(json.dumps({'gate': {'passed': True}}))
    def fail(*_):
        raise ValueError('native verification failed')
    monkeypatch.setattr(publisher, 'verify_native_copy', fail)
    with pytest.raises(ValueError, match='native verification'):
        publisher.publish(tmp_path)
    assert snapshots.load_snapshot(tmp_path)['generation'] == old['generation']
    (tmp_path / 'data/manifest_v3.json').write_text(json.dumps({'hnsw_validated': False}))
    with pytest.raises(ValueError, match='incomplete'):
        publisher.publish(tmp_path)


def test_index_wrapper_does_not_reenable_paid_features_or_write_reader_copy(tmp_path, monkeypatch):
    directory = fake_snapshot(tmp_path)
    (tmp_path / '.env').write_text('PDF_AI_TOC_FAST_PATH_ENABLE=1\n')
    monkeypatch.setenv('PDF_AI_TOC_FAST_PATH_ENABLE', '0')
    monkeypatch.setenv('CHROMA_DIR', str(tmp_path / 'data/chroma'))
    snapshots.apply_search_snapshot(tmp_path)
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner.sys, 'argv', ['run_searchable_index.py', '--progress'])
    calls = []
    monkeypatch.setattr(runner.subprocess, 'run', lambda command, **kwargs: calls.append((command, kwargs)) or SimpleNamespace(returncode=0))
    assert runner.main() == 0
    assert len(calls) == 1
    assert calls[0][1]['env']['PDF_AI_TOC_FAST_PATH_ENABLE'] == '0'
    assert calls[0][1]['env']['CHROMA_DIR'] == str(tmp_path / 'data/chroma')
    assert calls[0][1]['env']['CHROMA_DIR'] != str(directory / 'chroma')


def test_graph_reader_lease_uses_snapshot_during_canonical_write(tmp_path, monkeypatch):
    from src import item_vectors
    fake_snapshot(tmp_path)
    snapshots.apply_search_snapshot(tmp_path)
    monkeypatch.setattr(item_vectors, 'PROJECT_ROOT', tmp_path)
    with indexing_lock.held(indexing_lock.default_path(tmp_path)):
        with item_vectors._native_read_ready():
            pass


def test_import_index_child_restores_canonical_environment(tmp_path, monkeypatch):
    import asyncio
    monkeypatch.syspath_prepend(str(publisher.ROOT / 'src'))
    import rag_mcp_server as server
    for key in (*snapshots.PATH_KEYS, snapshots.ORIGINAL_ENV, snapshots.ACTIVE_ROOT):
        monkeypatch.delenv(key, raising=False)
    fake_snapshot(tmp_path)
    monkeypatch.setenv('CHROMA_DIR', str(tmp_path / 'data/chroma'))
    snapshots.apply_search_snapshot(tmp_path)
    monkeypatch.setattr(server, 'PROJECT_ROOT', tmp_path)
    calls = []
    class Process:
        returncode = 0
        async def communicate(self):
            return b'Done', None
    async def start(*args, **kwargs):
        calls.append(kwargs)
        return Process()
    monkeypatch.setattr(server.asyncio, 'create_subprocess_exec', start)
    assert asyncio.run(server._index_imported_item('example'))['status'] == 'indexed'
    assert calls[0]['env']['CHROMA_DIR'] == str(tmp_path / 'data/chroma')
    assert snapshots.ACTIVE_ROOT not in calls[0]['env']


@pytest.mark.parametrize('fail', [False, True])
def test_audit_refreshes_reader_only_after_success_and_lock_release(tmp_path, monkeypatch, fail):
    from scripts import run_db_audit as audit
    fake_snapshot(tmp_path)
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    paths = tuple(tmp_path / f'{name}.json' for name in ('gate', 'reconciliation', 'source', 'status'))
    monkeypatch.setattr(audit, '_validated_report_paths', lambda: paths)
    def check():
        if fail:
            raise audit.AuditFailure(2)
    monkeypatch.setattr(audit, '_audit_once', check)
    calls = []
    def run(*args, **kwargs):
        assert not indexing_lock.default_path(tmp_path).exists()
        calls.append(args)
    monkeypatch.setattr(audit.subprocess, 'run', run)
    if fail:
        with pytest.raises(audit.AuditFailure):
            audit.main()
        assert not calls
    else:
        audit.main()
        assert len(calls) == 1


def test_initial_service_stop_failure_restores_services_already_stopped(tmp_path, monkeypatch):
    native_source(tmp_path, monkeypatch)
    monkeypatch.setattr(publisher, 'loaded_services', lambda: ['first', 'second'])
    restored = []
    def stop(labels):
        if labels == ['second']:
            raise OSError('service stop failed')
    monkeypatch.setattr(publisher, 'stop_services', stop)
    monkeypatch.setattr(publisher, 'restore_services', restored.extend)
    with pytest.raises(OSError, match='stop failed'):
        publisher.publish(tmp_path, initialize=True, restart_services=True)
    assert restored == ['first']
    assert not snapshots.pointer_path(tmp_path).exists()


def test_native_fts_nul_compatibility_requires_matching_document_contents(tmp_path, monkeypatch):
    chromadb, source = native_source(tmp_path, monkeypatch)
    client = chromadb.PersistentClient(path=str(source))
    try:
        client.get_collection('zotero_paragraphs_v3', embedding_function=None).update(
            ids=['old1'], documents=['abc\x00def ghijkl mnopqr'], embeddings=[[1., 0., 0.]])
    finally:
        client.close()
    with sqlite3.connect(source / 'chroma.sqlite3') as db:
        errors = db.execute('PRAGMA quick_check').fetchall()
    if errors == [('ok',)]:
        pytest.skip('Python SQLite and native Chroma use compatible NUL tokenization')
    assert errors == [(publisher.FTS_COMPAT_ERROR,)]
    record = publisher.publish(tmp_path, initialize=True)
    assert record['sqlite_compatibility']
    with sqlite3.connect(source / 'chroma.sqlite3') as db:
        db.execute("UPDATE embedding_fulltext_search_content SET c0=c0 || ' tampered' WHERE instr(c0,char(0))>0")
        assert publisher.verified_fts_compatibility(db) is False
    with pytest.raises(ValueError, match='SQLite snapshot verification failed'):
        publisher.sqlite_copy(source / 'chroma.sqlite3', tmp_path / 'bad-copy.sqlite3', chroma=True)


def test_failed_sqlite_copy_removes_unpublished_stage(tmp_path, monkeypatch):
    native_source(tmp_path, monkeypatch)
    old = publisher.publish(tmp_path, initialize=True)
    gate = tmp_path / 'data/quality'
    gate.mkdir()
    (gate / 'server_database_gate.json').write_text(json.dumps({'gate': {'passed': True}}))
    def fail(*args, **kwargs):
        raise sqlite3.OperationalError('database or disk is full')
    monkeypatch.setattr(publisher, 'sqlite_copy', fail)
    with pytest.raises(sqlite3.OperationalError, match='disk is full'):
        publisher.publish(tmp_path)
    assert not list((tmp_path / 'data/search_snapshots').glob('.partial-*'))
    assert snapshots.load_snapshot(tmp_path)['generation'] == old['generation']
    assert (tmp_path / 'data/chroma/chroma.sqlite3').exists()


@pytest.mark.parametrize('platform', ['darwin', 'linux'])
def test_snapshot_space_check_fails_before_copy(tmp_path, monkeypatch, platform):
    source = tmp_path / 'chroma'
    source.mkdir()
    for name in ['chroma.sqlite3', 'chroma.sqlite3-wal']:
        (source / name).write_bytes(b'source')
    lexical = tmp_path / 'lexical.db'
    lexical.write_bytes(b'lexical')
    monkeypatch.setattr(publisher, 'lexical_path', lambda root: lexical)
    monkeypatch.setattr(publisher.sys, 'platform', platform)
    monkeypatch.setattr(publisher.shutil, 'disk_usage', lambda path: SimpleNamespace(free=0))
    with pytest.raises(RuntimeError, match='空き容量'):
        publisher.require_copy_space(tmp_path, source, tmp_path)
    monkeypatch.setattr(publisher.shutil, 'disk_usage', lambda path: SimpleNamespace(free=10 * 1024 ** 3))
    publisher.require_copy_space(tmp_path, source, tmp_path)


def test_pruning_retains_current_previous_open_and_unknown_copies(tmp_path, monkeypatch):
    current = fake_snapshot(tmp_path)
    home = current.parent
    for name in ('previous', 'unused', 'open', 'unknown'):
        directory = home / name
        directory.mkdir()
        if name != 'unknown':
            (directory / 'snapshot.json').write_text(json.dumps({'ready': True, 'generation': name}))
    record = snapshots.load_snapshot(tmp_path)
    record['previous'] = str(home / 'previous')
    monkeypatch.setattr(publisher, 'load_snapshot', lambda root: record)
    monkeypatch.setattr(publisher.shutil, 'which', lambda name: '/usr/bin/lsof')
    monkeypatch.setattr(publisher.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=f'n{home / "open" / "db"}\n'))
    assert publisher.prune_unused_copies(tmp_path) == ['unused']
    assert {p.name for p in home.iterdir()} == {'first', 'previous', 'open', 'unknown'}
    monkeypatch.setattr(publisher.shutil, 'which', lambda name: None)
    assert publisher.prune_unused_copies(tmp_path) == []


def test_pruning_does_not_delete_when_open_file_inspection_fails(tmp_path, monkeypatch):
    import subprocess
    current = fake_snapshot(tmp_path)
    unused = current.parent / 'unused'
    unused.mkdir()
    (unused / 'snapshot.json').write_text(json.dumps({'ready': True, 'generation': 'unused'}))
    monkeypatch.setattr(publisher.shutil, 'which', lambda name: '/usr/bin/lsof')
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr(publisher.subprocess, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        publisher.prune_unused_copies(tmp_path)
    assert unused.exists()
