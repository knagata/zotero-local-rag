import json
from scripts.update_status_cache import cached_checks, signature


def test_unchanged_inputs_reuse_and_wal_write_invalidates(tmp_path):
    database = tmp_path / 'relations.db'
    database.write_bytes(b'db')
    wal = tmp_path / 'relations.db-wal'
    report = tmp_path / 'report.json'
    calls = []
    def compute():
        calls.append(1)
        return {'pending': 1}, {'pending': 2}
    structure, summaries, stamp, reused = cached_checks(report, [database, wal], compute)
    assert not reused
    report.write_text(json.dumps(dict(structure=structure, summaries=summaries, local_checks_signature=stamp)))
    assert cached_checks(report, [database, wal], compute)[3] is True
    assert len(calls) == 1
    wal.write_bytes(b'new transaction')
    assert cached_checks(report, [database, wal], compute)[3] is False
    assert len(calls) == 2


def test_concurrent_changes_never_certify_cache(tmp_path):
    source = tmp_path / 'input'; source.write_text('before')
    def compute():
        source.write_text('after')
        return {}, {}
    assert cached_checks(tmp_path / 'report', [source], compute)[2] is None


def test_invalid_or_incomplete_previous_report_recomputes(tmp_path):
    source = tmp_path / 'input'
    report = tmp_path / 'report'
    for previous in ['invalid', '[]', json.dumps({'local_checks_signature': signature([source])})]:
        report.write_text(previous)
        assert cached_checks(report, [source], lambda: ({}, {}))[3] is False


def test_lightweight_reports_missing_failed_and_stale_without_chroma(tmp_path):
    import sqlite3
    from scripts.update_status_cache import lightweight_checks
    database = tmp_path / 'relations.db'
    with sqlite3.connect(database) as c:
        c.execute('CREATE TABLE document_structures(item_key, source_fingerprint, status)')
        c.execute('CREATE TABLE artifact_processing_status(item_key, artifact_type, status)')
        c.execute('CREATE TABLE document_node_summaries(item_key, source_fingerprint)')
        c.executemany('INSERT INTO document_structures VALUES (?,?,?)', [('GOOD', 'v1', 'ok'), ('STALE', 'v2', 'ok')])
        c.executemany('INSERT INTO artifact_processing_status VALUES (?,?,?)', [('GOOD', 'summary', 'success'), ('STALE', 'summary', 'failed')])
        c.execute('INSERT INTO document_node_summaries VALUES (?,?)', ('STALE', 'v1'))
    structure, summary = lightweight_checks(database, {'GOOD', 'STALE', 'NEW'})
    assert structure['sample_keys'] == ['NEW']
    assert summary['sample_keys'] == ['STALE']
    assert summary['failed'] == 1
    assert 'audit_passed' not in summary
