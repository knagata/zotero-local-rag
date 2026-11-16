from __future__ import annotations

import copy
import json
from unittest.mock import patch

import pytest

from src.partial_source_audit import source_exclusions, exclude_source_items
from src.database_gate import external_audit_attestation, _validate_external_audits
from scripts import run_db_audit
from tests.test_run_db_audit import _env


def report():
    return {'passed': False, 'summary': {'orphan_chunks': 0, 'dangling_node_chunks': 0,
        'unretrievable_items': 0, 'unreadable_source_documents': 0,
        'documents_with_lost_pages': 1, 'lost_pages_total': 2},
        'documents': [{'attachment_key': 'BADATT', 'lost_pages': [1, 2]}]}


def test_only_document_local_losses_can_continue():
    valid = report()
    valid['documents'].append({'attachment_key': 'GOODATT', 'lost_pages': []})
    assert source_exclusions(valid) == {'BADATT'}
    for field in ('orphan_chunks', 'dangling_node_chunks', 'unretrievable_items',
                  'unreadable_source_documents', 'lost_pages_total', 'documents_with_lost_pages'):
        bad = report()
        bad['summary'][field] += 1
        with pytest.raises(RuntimeError):
            source_exclusions(bad)
    for key in ('', None):
        bad = report()
        bad['documents'][0]['attachment_key'] = key
        with pytest.raises(RuntimeError):
            source_exclusions(bad)
    bad = report()
    bad['documents'] *= 2
    with pytest.raises(RuntimeError):
        source_exclusions(bad)
    with pytest.raises(RuntimeError):
        source_exclusions({'passed': False, 'summary': report()['summary'], 'documents': []})


def test_exclusions_apply_to_whole_item_and_require_unique_ownership():
    comparisons = [{'item_key': 'GOOD', 'passed': True}, {'item_key': 'BAD', 'passed': True}]
    rows = [{'metadata': {'attachmentKey': 'BADATT', 'itemKey': 'BAD'}}]
    assert exclude_source_items(comparisons, rows, [], {'BADATT'}) == {'BAD'}
    assert comparisons[0]['passed'] and not comparisons[1]['passed']
    assert exclude_source_items([], [], [{'attachment_key': 'BADATT', 'item_key': 'BAD'}], {'BADATT'}) == {'BAD'}
    for statuses in ([], [{'attachment_key': 'BADATT', 'item_key': 'OTHER'}]):
        with pytest.raises(RuntimeError):
            exclude_source_items([], rows if statuses else [], statuses, {'BADATT'})
    assert exclude_source_items([], [], [], set()) == set()


def test_failed_source_attestation_never_authorizes_full_gate_and_is_byte_bound(tmp_path):
    source = tmp_path / 'source.json'
    source.write_text(json.dumps(report()))
    zotero = tmp_path / 'zotero.json'
    zotero.write_text('{"passed": true}')
    attestations = {'zotero_reconciliation': external_audit_attestation(zotero),
                    'source_verification': external_audit_attestation(source, allow_partial_source=True)}
    with pytest.raises(RuntimeError):
        external_audit_attestation(source)
    with pytest.raises(RuntimeError):
        _validate_external_audits(attestations)
    _validate_external_audits(attestations, allow_partial_source=True)
    altered = copy.deepcopy(attestations)
    altered['source_verification']['excluded_attachments'] = []
    with pytest.raises(RuntimeError, match='exclusions changed'):
        _validate_external_audits(altered, allow_partial_source=True)
    source.write_text(json.dumps(report()) + '\n')
    with pytest.raises(RuntimeError, match='changed after'):
        _validate_external_audits(attestations, allow_partial_source=True)


@pytest.mark.parametrize("partial", [True, False])
def test_source_failure_reaches_cutover_and_partial_publication(tmp_path, partial):
    calls = []
    def child(command, cwd=None):
        calls.append(command)
        if 'scripts/verify_against_source.py' in command:
            output = command[command.index('--output') + 1]
            from pathlib import Path
            Path(output).write_text(json.dumps(report()))
            return type('Result', (), {'returncode': 2})()
        if 'scripts/audit_v3_cutover.py' in command:
            output = command[command.index('--output') + 1]
            from pathlib import Path
            Path(output).write_text(json.dumps({'gate': {'passed': False}, 'indexed_gate': {
                'passed': True, 'eligible_item_keys': ['GOOD'], 'excluded_item_keys': ['BAD']}}))
            return type('Result', (), {'returncode': 2})()
        return type('Result', (), {'returncode': 0})()
    with patch.object(run_db_audit, 'ROOT', tmp_path), \
         patch.dict('os.environ', _env(tmp_path), clear=True), \
         patch('subprocess.run', side_effect=child), \
         patch('src.audit_repair.repair_derived_data', return_value={'repaired': False}), \
         patch.object(run_db_audit, '_refresh_search_snapshot') as publish:
        if partial:
            run_db_audit.main(['--allow-partial-index'])
        else:
            with pytest.raises(SystemExit) as failure:
                run_db_audit.main([])
            assert failure.value.code == 2
    if partial:
        publish.assert_called_once_with(partial=True)
    else:
        publish.assert_not_called()
    assert len(calls) == 3
    assert json.loads((tmp_path / 'data/quality/server_database_audit_status.json').read_text())['status'] == ('completed_partial' if partial else 'failed')


def test_inventory_excludes_lost_page_items_even_if_they_have_indexed_chunks(monkeypatch):
    from scripts import audit_v3_cutover as cutover
    from src.source_coverage import make_source_coverage
    from src import db_relations
    monkeypatch.setattr(db_relations, 'get_artifact_processing_statuses', lambda **kw: [])
    complete = make_source_coverage(unit_kind='page', expected_units=[1], attempted_units=[1], text_units=[1])
    files = {'GOODATT': {'quality': {'source_coverage': complete}}, 'BADATT': {}}
    comparisons = [{'item_key': 'GOOD', 'passed': True}, {'item_key': 'BAD', 'passed': True}]
    rows = [{'metadata': {'attachmentKey': 'BADATT', 'itemKey': 'BAD'}}]
    failures = ['incomplete_source_coverage']
    result = cutover._inventory_reports(files, set(files), comparisons, failures, report(), rows)
    assert result['indexed_gate']['passed']
    assert result['indexed_gate']['eligible_item_keys'] == ['GOOD']
    assert result['indexed_gate']['excluded_item_keys'] == ['BAD']
    assert not comparisons[1]['passed']
    assert 'source_verification_failed' in failures
    with pytest.raises(RuntimeError, match='manifest'):
        cutover._inventory_reports({}, set(), [], [], report(), rows)
