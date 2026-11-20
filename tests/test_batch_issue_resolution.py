"""Past OCR failures need a source-bound, audited replacement before resolution."""
import hashlib
import json

import pytest

from citation_graph import admin_jobs


@pytest.fixture
def completed_ocr(tmp_path):
    root = tmp_path
    data = root / 'data'
    published = data / 'search_snapshots' / 'approved'
    (published / 'chroma').mkdir(parents=True)
    source = root / 'completed.pdf'
    source.write_bytes(b'Existing OCR body')
    stat = source.stat()
    entry = {'pdf_path': str(source), 'mtime': stat.st_mtime, 'size': stat.st_size,
             'content_signature': 'sha256:' + hashlib.sha256(source.read_bytes()).hexdigest(),
             'pipeline_fingerprint': 'approved-pipeline', 'quality': {'ocr_policy': 'ocr-complete'}}
    manifest = {'hnsw_validated': True, 'files': {'SOURCE01': entry}}
    for path in (data / 'manifest_v3.json', published / 'manifest_v3.json'):
        path.write_text(json.dumps(manifest))
    proof = {'generation': 'approved', 'ready': True, 'attachments': 1,
             'published_at': '2026-11-21T00:00:00+00:00', 'audit_passed': True}
    (published / 'snapshot.json').write_text(json.dumps(proof))
    (data / 'search_snapshot.json').write_text(json.dumps({**proof, 'directory': str(published)}))
    for name in ('chroma/chroma.sqlite3', 'chroma/embedder_config_v3.json', 'lexical_v3.sqlite3', 'relations.db'):
        (published / name).touch()
    batch = {'phase': 'collected', 'candidates': [{'attachment_key': 'SOURCE01', 'title': 'Recovered book'}],
             'review_required_attachments': ['SOURCE01'],
             'reports': [{'attachment_key': 'SOURCE01', 'ok': False, 'status_code': 504, 'retryable': True}]}
    batch_path = data / 'mistral_ocr_batch_state.json'
    batch_path.write_text(json.dumps(batch))
    return root, source, published, batch_path


def edit_json(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def test_audited_completed_source_moves_to_history_without_changing_batch(completed_ocr):
    root, _, _, batch_path = completed_ocr
    original = batch_path.read_bytes()
    status = admin_jobs._mistral_batch_status(root)
    assert status['issues'] == []
    assert status['resolved_issues'][0]['status_code'] == 504
    assert status['resolved_issues'][0]['title'] == 'Recovered book'
    assert batch_path.read_bytes() == original  # including the paid-retry review guard


@pytest.mark.parametrize('field', ['mtime', 'size', 'content_signature', 'pipeline_fingerprint'])
def test_changed_source_identity_does_not_hide_failure(completed_ocr, field):
    root, _, _, _ = completed_ocr
    edit_json(root / 'data/manifest_v3.json', lambda m: m['files']['SOURCE01'].__setitem__(field, 'changed'))
    assert admin_jobs._mistral_batch_status(root)['resolved_issues'] == []


@pytest.mark.parametrize('case', ['no_snapshot', 'unverified', 'bad_certificate', 'no_live_entry',
                                 'no_published_entry', 'no_policy', 'no_signature', 'missing_field',
                                 'changed_file', 'missing_file', 'bad_path', 'bad_manifest', 'bad_files'])
def test_incomplete_or_unavailable_evidence_keeps_warning(completed_ocr, case):
    root, source, published, _ = completed_ocr
    live = root / 'data/manifest_v3.json'
    if case == 'no_snapshot':
        (root / 'data/search_snapshot.json').unlink()
    elif case == 'unverified':
        edit_json(published / 'snapshot.json', lambda p: p.update(audit_passed=False))
    elif case == 'bad_certificate':
        (published / 'snapshot.json').write_text('{')
    elif case == 'no_live_entry':
        edit_json(live, lambda m: m['files'].clear())
    elif case == 'no_published_entry':
        edit_json(published / 'manifest_v3.json', lambda m: m['files'].clear())
    elif case == 'no_policy':
        edit_json(live, lambda m: m['files']['SOURCE01'].update(quality=[]))
    elif case == 'no_signature':
        edit_json(live, lambda m: m['files']['SOURCE01'].update(content_signature=None))
    elif case == 'missing_field':
        edit_json(live, lambda m: m['files']['SOURCE01'].pop('pipeline_fingerprint'))
    elif case == 'changed_file':
        source.write_bytes(b'A different replacement PDF with different length')
    elif case == 'missing_file':
        source.unlink()
    elif case == 'bad_path':
        edit_json(live, lambda m: m['files']['SOURCE01'].update(pdf_path=None))
    elif case == 'bad_manifest':
        live.write_text('[]')
    elif case == 'bad_files':
        edit_json(live, lambda m: m.update(files=[]))
    status = admin_jobs._mistral_batch_status(root)
    assert len(status['issues']) == 1
    assert status['resolved_issues'] == []


def test_approved_subset_also_resolves_its_included_source(completed_ocr):
    root, _, published, _ = completed_ocr
    edit_json(published / 'snapshot.json', lambda p: p.update(
        audit_passed=False, partial=True, eligible_items=1, excluded_items=1, pending_ocr_attachments=1))
    edit_json(published / 'manifest_v3.json', lambda m: m.update(partial_publication=True))
    assert admin_jobs._mistral_batch_status(root)['issues'] == []


def test_relative_source_path_and_independent_unresolved_failure(completed_ocr):
    root, _, _, batch_path = completed_ocr
    edit_json(root / 'data/manifest_v3.json', lambda m: m['files']['SOURCE01'].update(pdf_path='completed.pdf'))
    edit_json(batch_path, lambda s: s['reports'].append({'attachment_key': 'OTHER001', 'ok': False}))
    status = admin_jobs._mistral_batch_status(root)
    assert [i['attachment_key'] for i in status['resolved_issues']] == ['SOURCE01']
    assert [i['attachment_key'] for i in status['issues']] == ['OTHER001']


def test_same_generation_without_completed_ocr_policy_is_not_enough(completed_ocr):
    root, _, published, _ = completed_ocr
    for path in (root / 'data/manifest_v3.json', published / 'manifest_v3.json'):
        edit_json(path, lambda m: m['files']['SOURCE01'].update(quality={'parser': 'another-engine'}))
    assert len(admin_jobs._mistral_batch_status(root)['issues']) == 1


def test_empty_issues_avoid_snapshot_reads(monkeypatch, tmp_path):
    import src.search_snapshot
    monkeypatch.setattr(src.search_snapshot, 'load_snapshot', lambda root: pytest.fail('unneeded snapshot read'))
    assert admin_jobs._resolved_batch_issue_keys(tmp_path, []) == set()
