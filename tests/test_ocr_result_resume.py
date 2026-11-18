from __future__ import annotations

import json

import pytest

from scripts import run_mistral_ocr_maintenance as maintenance
from scripts import run_searchable_index as runner


@pytest.mark.parametrize('phase', ['', 'prepared', 'uploaded', 'failed', 'cancelled', 'timeout_exceeded'])
def test_collect_only_never_submits_or_retries(phase, monkeypatch):
    monkeypatch.setattr(maintenance, '_load', lambda: {'phase': phase})
    monkeypatch.setattr(maintenance, '_run', lambda *a, **k: pytest.fail('unexpected external command'))
    assert maintenance.main(collect_only=True) == 0


def test_collect_only_preserves_retryable_errors_without_resubmission(monkeypatch):
    state = {'phase': 'collected', 'adoption_applied_at': 'earlier',
             'reports': [{'item_key': 'BAD', 'retryable': True}]}
    monkeypatch.setattr(maintenance, '_load', lambda: state)
    monkeypatch.setattr(maintenance, '_run', lambda *a, **k: pytest.fail('must not submit'))
    assert maintenance.main(collect_only=True) == 0


def test_completed_results_are_adopted_without_paid_features(tmp_path, monkeypatch):
    queue = tmp_path / 'queue.json'
    queue.write_text(json.dumps({'candidates': [{'item_key': 'GOOD'}]}))
    state = {'phase': 'collected', 'adoptable_count': 1, 'adoption_queue': str(queue)}
    calls, saved = [], []
    monkeypatch.setattr(maintenance, '_load', lambda: state)
    monkeypatch.setattr(maintenance, '_save', lambda value: saved.append(value.copy()))
    monkeypatch.setattr(maintenance, '_run', lambda *a, **k: calls.append((a, k)))
    assert maintenance.main(collect_only=True) == 0
    assert len(calls) == 2
    assert '--reocr-candidates' in calls[0][0]
    assert calls[1][0] == (str(maintenance.STRUCTURE), '--item', 'GOOD')
    for _, kwargs in calls:
        assert kwargs['environment']['PDF_MISTRAL_TOC_QUEUE_ENABLE'] == '0'
        assert kwargs['environment']['PDF_AI_TOC_FAST_PATH_ENABLE'] == '0'
        assert kwargs['environment']['OCR_LAYER_AUDIT_ENABLE'] == '0'
    assert saved[-1]['adoption_applied_at']


def test_collect_only_handles_remote_terminal_error_without_blocking_other_items(monkeypatch):
    states = iter([{'phase': 'submitted'}, {'phase': 'failed'}])
    calls = []
    monkeypatch.setattr(maintenance, '_load', lambda: next(states))
    monkeypatch.setattr(maintenance, '_run', lambda *a: calls.append(a))
    assert maintenance.main(collect_only=True) == 0
    assert len(calls) == 1 and '--status' in calls[0]


def test_index_wrapper_resumes_existing_results_before_indexing(tmp_path, monkeypatch):
    path = tmp_path / 'data/mistral_ocr_batch_state.json'
    path.parent.mkdir()
    path.write_text('{"phase":"success"}')
    calls = []
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **k: calls.append((a,k)))
    runner.resume_existing_ocr({'CHROMA_DIR': str(tmp_path/'data/chroma')})
    assert '--collect-only' in calls[0][0][0]
    assert calls[0][1]['check'] is True
    assert calls[0][1]['env']['CHROMA_DIR'] == str(tmp_path/'data/chroma')
    path.write_text('{"phase":"prepared"}')
    runner.resume_existing_ocr({})
    assert len(calls) == 1


def test_quality_rejected_result_is_not_automatically_bought_again(monkeypatch):
    state = {'phase': 'collected', 'adoption_applied_at': 'earlier',
             'candidates': [{'attachment_key': 'REJECTED'}],
             'reports': [{'attachment_key': 'REJECTED', 'item_key': 'BAD', 'retryable': False}]}
    monkeypatch.setattr(maintenance, '_load', lambda: state)
    excluded = []
    monkeypatch.setattr(maintenance, '_queued_candidate_count',
                        lambda **kw: excluded.append(kw['exclude_attachments']) or 0)
    monkeypatch.setattr(maintenance, '_run', lambda *a, **kw: pytest.fail('must not buy rejected OCR again'))
    assert maintenance.main() == 0
    assert excluded == [{'REJECTED'}]


def test_local_environment_is_forwarded_to_child(monkeypatch):
    calls = []
    monkeypatch.setattr(maintenance.subprocess, 'run', lambda *a, **k: calls.append(k))
    maintenance._run('script.py', environment={'PDF_MISTRAL_TOC_QUEUE_ENABLE': '0'})
    assert calls[0]['env']['PDF_MISTRAL_TOC_QUEUE_ENABLE'] == '0'
    assert calls[0]['check'] is True


def test_state_replaced_during_adoption_is_not_marked_applied(tmp_path, monkeypatch):
    queue = tmp_path / 'queue.json'
    queue.write_text('{"candidates":[{"item_key":"GOOD"}]}')
    state = {'job_id': 'original', 'adoptable_count': 1, 'adoption_queue': str(queue)}
    monkeypatch.setattr(maintenance, '_run', lambda *a, **kw: None)
    monkeypatch.setattr(maintenance, '_load', lambda: {'job_id': 'different'})
    monkeypatch.setattr(maintenance, '_save', lambda *a: pytest.fail('must preserve new job'))
    with pytest.raises(RuntimeError, match='state changed'):
        maintenance._adopt(state, local_only=True)


def test_new_queue_count_excludes_processed_results(tmp_path, monkeypatch):
    import sqlite3
    database = tmp_path / 'relations.db'
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE artifact_processing_status (attachment_key, artifact_type, status, reason_code)')
        connection.executemany('INSERT INTO artifact_processing_status VALUES (?, ?, ?, ?)', [
            ('OLD', 'extraction', 'blocked', 'awaiting_mistral_ocr_batch'),
            ('NEW', 'extraction', 'blocked', 'awaiting_mistral_ocr_batch'),
            ('NEW', 'extraction', 'blocked', 'awaiting_mistral_ocr_batch'),
            ('DONE', 'extraction', 'success', 'complete'),
        ])
    monkeypatch.setattr(maintenance, 'RELATIONS_DB', database)
    assert maintenance._queued_candidate_count() == 2
    assert maintenance._queued_candidate_count(exclude_attachments={'OLD'}) == 1


@pytest.mark.parametrize('phase', ['submitted', 'success'])
def test_failed_remote_read_does_not_stop_unrelated_local_indexing(monkeypatch, phase):
    import subprocess
    monkeypatch.setattr(maintenance, '_load', lambda: {'phase': phase})
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args)
    monkeypatch.setattr(maintenance, '_run', fail)
    assert maintenance.main(collect_only=True) == 0
    with pytest.raises(subprocess.CalledProcessError):
        maintenance.main()


def test_adoption_write_failure_still_stops_pipeline(tmp_path, monkeypatch):
    import subprocess
    queue = tmp_path / 'queue.json'
    queue.write_text('{"candidates":[{"item_key":"GOOD"}]}')
    monkeypatch.setattr(maintenance, '_load', lambda: {
        'phase':'collected', 'adoptable_count':1, 'adoption_queue':str(queue)})
    monkeypatch.setattr(maintenance, '_run', lambda *a, **k: (_ for _ in ()).throw(subprocess.CalledProcessError(1,a)))
    with pytest.raises(subprocess.CalledProcessError):
        maintenance.main(collect_only=True)


@pytest.mark.parametrize("payload", ["not json", "[]", "null"])
def test_invalid_optional_batch_state_is_reported_without_blocking_normal_updates(tmp_path, monkeypatch, capsys, payload):
    path = tmp_path / 'data/mistral_ocr_batch_state.json'
    path.parent.mkdir()
    path.write_text(payload)
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    runner.resume_existing_ocr({})
    assert '要確認' in capsys.readouterr().out


def test_new_batch_does_not_include_old_quality_rejection(monkeypatch):
    calls = []
    monkeypatch.setattr(maintenance, '_load', lambda: {
        'phase':'collected', 'adoption_applied_at':'earlier',
        'candidates':[{'attachment_key':'OLD'}]})
    monkeypatch.setattr(maintenance, '_queued_candidate_count', lambda **kw: 1)
    monkeypatch.setattr(maintenance, '_run', lambda *a, **kw: calls.append(a))
    maintenance.main()
    assert '--exclude-attachment' in calls[0] and calls[0][-1] == 'OLD'


def test_preparation_filters_exact_attachment_even_with_same_parent(tmp_path, monkeypatch):
    import argparse
    from scripts import run_mistral_ocr_batch as batch
    pdf = tmp_path / 'source.pdf'
    pdf.write_bytes(b'pdf fixture bytes')
    async def resolved(items):
        return [({'attachment_key': key, 'item_key': 'SAMEPARENT'}, pdf) for key in ('OLD','NEW')]
    monkeypatch.setattr(batch, 'resolve_candidates', resolved)
    def write(rows, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
        return len(rows)
    monkeypatch.setattr(batch, 'write_batch_jsonl', write)
    args = argparse.Namespace(item=None, exclude_attachment=['OLD'], max_input_bytes=100000,
                              work_dir=tmp_path/'work', state=tmp_path/'state.json', model='model')
    state = batch.prepare(args)
    assert [row['attachment_key'] for row in state['candidates']] == ['NEW']


def test_retry_does_not_resend_quality_rejection_under_same_parent(monkeypatch):
    calls = []
    monkeypatch.setattr(maintenance, '_run', lambda *a: calls.append(a))
    assert maintenance._submit_retryable({'reports':[
        {'item_key':'PARENT','attachment_key':'TIMEOUT','retryable':True},
        {'item_key':'PARENT','attachment_key':'REJECTED','retryable':False},
    ]})
    assert calls[0][-2:] == ('--exclude-attachment', 'REJECTED')


def test_alternative_plane_does_not_mark_live_ocr_as_applied(tmp_path, monkeypatch):
    path = tmp_path / 'data/mistral_ocr_batch_state.json'
    path.parent.mkdir()
    path.write_text('{"phase":"collected","adoptable_count":1}')
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    calls = []
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **k: calls.append(k))
    runner.resume_existing_ocr({'CHROMA_DIR': str(tmp_path/'test-plane/chroma')})
    assert not calls
    separate = tmp_path / 'test-state.json'
    separate.write_text('{"phase":"collected"}')
    runner.resume_existing_ocr({'CHROMA_DIR':str(tmp_path/'test-plane/chroma'),
                                'MISTRAL_BATCH_STATE_PATH':str(separate)})
    assert len(calls) == 1


def test_review_rejections_survive_later_batch_preparation(tmp_path, monkeypatch):
    import argparse
    from scripts import run_mistral_ocr_batch as batch
    state_path = tmp_path / 'state.json'
    state_path.write_text(json.dumps({'reports':[
        {'attachment_key':'OLD','ok':True,'gate':{'passed':False}},
        {'attachment_key':'TRANSIENT','ok':False,'retryable':True},
        {'attachment_key':'ACCEPTED','ok':True,'gate':{'passed':True}},
    ]}))
    async def no_candidates(items):
        return []
    monkeypatch.setattr(batch,'resolve_candidates',no_candidates)
    args = argparse.Namespace(item=None, max_input_bytes=100000, work_dir=tmp_path/'work',
                              state=state_path,model='model')
    first = batch.prepare(args)
    assert first['review_required_attachments'] == ['OLD']
    second = batch.prepare(args)
    assert second['review_required_attachments'] == ['OLD']
    args.retry_reviewed_attachment = ['OLD']
    assert batch.prepare(args)['review_required_attachments'] == []


def test_old_review_protection_does_not_trigger_new_paid_batch(monkeypatch):
    monkeypatch.setattr(maintenance,'_load',lambda:{'phase':'collected',
        'adoption_applied_at':'earlier','review_required_attachments':['OLD']})
    monkeypatch.setattr(maintenance,'_queued_candidate_count',lambda **kw: 0 if kw['exclude_attachments']=={'OLD'} else 1)
    monkeypatch.setattr(maintenance,'_run',lambda *a,**k:pytest.fail('must not repurchase old rejected OCR'))
    assert maintenance.main() == 0
