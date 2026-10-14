import pytest
from src.indexed_gate import indexed_gate, summary_scope


def fixture():
    files = {'WAIT': {}, 'OK': {'quality': {'source_coverage': {'schema_version': 'unknown'}}}}
    statuses = [{'attachment_key': 'WAIT', 'item_key': 'WAIT_ITEM', 'artifact_type': 'extraction',
                 'status': 'blocked', 'reason_code': 'awaiting_mistral_ocr_batch'}]
    return files, statuses


def test_partial_gate_requires_complete_indexed_coverage(monkeypatch):
    import src.indexed_gate as module
    files, statuses = fixture()
    monkeypatch.setattr(module, 'validate_source_coverage', lambda c: {'passed': c == 'complete'})
    monkeypatch.setattr(module, 'coverage_gap_is_adoptable', lambda c: False)
    files['OK']['quality']['source_coverage'] = 'complete'
    comparisons = [{'item_key': 'GOOD', 'passed': True}, {'item_key': 'BAD', 'passed': False},
                   {'item_key': 'WAIT_ITEM', 'passed': True}]
    result = indexed_gate(files, {'OK'}, statuses, comparisons,
                          ['manifest_chroma_attachment_mismatch', 'incomplete_source_coverage'])
    assert result['passed'] and result['eligible_item_keys'] == ['GOOD']
    assert result['pending_ocr_attachments'] == ['WAIT']
    report = {'gate': {'passed': False}, 'indexed_gate': result}
    assert summary_scope(report, ['BAD', 'WAIT_ITEM', 'GOOD']) == ['GOOD']
    assert summary_scope({'gate': {'passed': True}}, ['ALL']) == ['ALL']
    for extra in ['lexical_content_mismatch', 'chroma_lexical_id_mismatch', 'hnsw_not_validated', 'inflight_attachments', 'post_index_pending', 'missing_v3_pipeline_provenance']:
        assert not indexed_gate(files, {'OK'}, statuses, comparisons, [extra])['passed']
    assert not indexed_gate(files, {'OK', 'EXTRA'}, statuses, comparisons, [])['passed']
    assert not indexed_gate(files, {'OK'}, [], comparisons, [])['passed']
    files['OK']['quality']['source_coverage'] = None
    assert not indexed_gate(files, {'OK'}, statuses, comparisons, [])['passed']
    with pytest.raises(RuntimeError):
        summary_scope({'indexed_gate': {'passed': False}}, ['GOOD'])


def test_real_coverage_contract_and_empty_eligible_set():
    from src.source_coverage import make_source_coverage
    complete = make_source_coverage(unit_kind='page', expected_units=[1], attempted_units=[1], text_units=[1])
    files = {'OK': {'quality': {'source_coverage': complete}}, 'WAIT': {}}
    statuses = [{'attachment_key':'WAIT', 'item_key':'PENDING', 'artifact_type':'extraction',
                 'status':'blocked', 'reason_code':'awaiting_mistral_ocr_batch'}]
    rows = [{'item_key':'GOOD', 'passed':True}]
    assert indexed_gate(files, {'OK'}, statuses, rows, ['incomplete_source_coverage'])['passed']
    assert not indexed_gate(files, {'OK'}, statuses, [], ['incomplete_source_coverage'])['passed']
    files['OK']['quality']['source_coverage']['text_units'] = []
    assert not indexed_gate(files, {'OK'}, statuses, rows, ['incomplete_source_coverage'])['passed']
