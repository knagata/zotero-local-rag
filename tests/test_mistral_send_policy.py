import asyncio
from types import SimpleNamespace

import httpx
import pytest
from src import mistral_send_policy as policy, mistral_ocr_extract as extract
from scripts import run_mistral_ocr_batch as batch


def client(monkeypatch, attachment=None, parent=None, status=200):
    calls = []
    factory = httpx.Client
    def handler(request):
        calls.append(request.url.path)
        body = attachment if len(calls) == 1 else parent
        return httpx.Response(status, json={'data': body or {}}, request=request)
    monkeypatch.setattr(policy.httpx, 'Client', lambda **kwargs: factory(transport=httpx.MockTransport(handler), **kwargs))
    return calls


@pytest.mark.parametrize('groups,expected', [
    (([],), False), ((['other'],), False), (([' NO-MISTRAL-OCR '],), True),
    (([], [{'tag': 'no-mistral-ocr'}]), True), (([42, {}],), False),
])
def test_tag_exclusion_is_specific(groups, expected):
    assert policy.has_send_exclusion(*groups) is expected


@pytest.mark.parametrize('attachment_tags,parent_tags', [(['no-mistral-ocr'], []), ([], ['no-mistral-ocr'])])
def test_fresh_attachment_or_parent_tag_blocks_send(monkeypatch, attachment_tags, parent_tags):
    calls = client(monkeypatch, {'itemType': 'attachment', 'parentItem': 'PARENT01', 'tags': attachment_tags}, {'tags': parent_tags})
    with pytest.raises(RuntimeError, match='no-mistral-ocr'):
        policy.assert_mistral_send_allowed('SOURCE01')
    assert len(calls) == 2


def test_untagged_attachment_allowed_and_bad_response_fails_closed(monkeypatch):
    calls = client(monkeypatch, {'itemType': 'attachment'})
    policy.assert_mistral_send_allowed('SOURCE01')
    assert len(calls) == 1
    with pytest.raises(ValueError, match='添付ID'):
        policy.assert_mistral_send_allowed('../bad')


@pytest.mark.parametrize('body,match', [({'itemType': 'book'}, '添付では'), ({'itemType': 'attachment', 'parentItem': '../bad'}, '親アイテムID')])
def test_invalid_item_or_parent_fails_closed(monkeypatch, body, match):
    client(monkeypatch, body)
    with pytest.raises(ValueError, match=match):
        policy.assert_mistral_send_allowed('SOURCE01')


def test_zotero_failure_cannot_be_permission_to_pay(monkeypatch):
    client(monkeypatch, status=503)
    with pytest.raises(httpx.HTTPStatusError):
        policy.assert_mistral_send_allowed('SOURCE01')


def test_preparation_skips_excluded_parent_without_cloud_calls(monkeypatch):
    monkeypatch.setattr(batch, 'get_artifact_processing_statuses', lambda **kwargs: [])
    monkeypatch.setattr(batch, 'build_queue_rows', lambda rows: [{'attachment_key': 'SOURCE01', 'item_key': 'PARENT01'}])
    class API:
        async def list_normalized_attachments(self, **kwargs):
            return [SimpleNamespace(attachmentKey='SOURCE01', parentTags=('no-mistral-ocr',), tags=())]
    monkeypatch.setattr(batch, 'ZoteroLocalAPI', API)
    monkeypatch.setattr(batch, 'mistral_ocr_available', lambda: pytest.fail('excluded source must not reach cloud policy'))
    assert asyncio.run(batch.resolve_candidates(None)) == []


def test_already_uploaded_batch_checks_tags_before_any_vendor_call(monkeypatch, tmp_path):
    source = tmp_path/'paper.pdf';source.write_bytes(b'pdf')
    input_path = tmp_path/'input.jsonl';input_path.write_text('{}')
    monkeypatch.setattr(batch, 'source_matches', lambda *args: (True, ''))
    monkeypatch.setattr(batch, 'mistral_ocr_available', lambda: (True, ''))
    def blocked(key):
        raise RuntimeError('no-mistral-ocr')
    monkeypatch.setattr(batch, 'assert_mistral_send_allowed', blocked)
    monkeypatch.setattr(batch, 'client_from_env', lambda: pytest.fail('must not contact vendor'))
    state = {'phase': 'uploaded', 'candidate_count': 1, 'input_files': [{'input_path': str(input_path), 'input_file_id': 'old-upload'}], 'candidates': [{'attachment_key': 'SOURCE01', 'source_path': str(source)}]}
    with pytest.raises(RuntimeError, match='no-mistral-ocr'):
        batch.submit(SimpleNamespace(), state)


def test_direct_refetch_is_blocked_but_archived_replay_is_allowed(monkeypatch, tmp_path):
    pdf = tmp_path/'paper.pdf';pdf.write_bytes(b'fake-source')
    monkeypatch.setenv('MISTRAL_OCR_API_KEY', 'test')
    monkeypatch.setattr(extract, '_resolve_cache_paths', lambda root: ('model', tmp_path))
    def blocked(key):
        raise RuntimeError('no-mistral-ocr')
    monkeypatch.setattr(extract, 'assert_mistral_send_allowed', blocked)
    monkeypatch.setattr(extract, '_call_mistral_ocr_api', lambda *a, **k: pytest.fail('must not send'))
    with pytest.raises(RuntimeError, match='no-mistral-ocr'):
        extract.extract_chunks_from_pdf_with_mistral_ocr(pdf, 'SOURCE01', {}, use_cache=False)
    monkeypatch.setattr(extract, '_find_cached_result', lambda *a, **k: ('digest', 'model', {}))
    monkeypatch.setattr(extract, 'extract_chunks_from_mistral_ocr_result', lambda *a, **k: ([], {'is_corrupted': True}))
    _, quality = extract.extract_chunks_from_pdf_with_mistral_ocr(pdf, 'SOURCE01', {})
    assert quality['ocr_cache'] == 'hit' and quality['is_corrupted'] is True


def test_direct_entrypoint_import_uses_same_policy(monkeypatch):
    import importlib.util
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(policy.__file__).parent))
    spec = importlib.util.spec_from_file_location('mistral_send_policy_direct_test', policy.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.has_send_exclusion(['no-mistral-ocr'])


def test_tag_added_during_upload_stops_job_creation(monkeypatch, tmp_path):
    source = tmp_path/'paper.pdf';source.write_bytes(b'pdf')
    payload = tmp_path/'input.jsonl';payload.write_text('{}')
    uploaded = []
    class Client:
        def upload(self, path):
            uploaded.append(path)
            return 'uploaded-id'
        def create_job(self, *args, **kwargs):
            pytest.fail('tag added during upload must stop billed job creation')
    def gate(key):
        if uploaded:
            raise RuntimeError('no-mistral-ocr')
    monkeypatch.setattr(batch, 'assert_mistral_send_allowed', gate)
    monkeypatch.setattr(batch, 'source_matches', lambda *args: (True, ''))
    monkeypatch.setattr(batch, 'mistral_ocr_available', lambda: (True, ''))
    monkeypatch.setattr(batch, 'client_from_env', Client)
    monkeypatch.setattr(batch, 'save_json_atomic', lambda *args: None)
    state = {'phase': 'prepared', 'candidate_count': 1, 'model': 'model', 'input_files': [{'input_path': str(payload)}], 'candidates': [{'attachment_key': 'SOURCE01', 'source_path': str(source)}]}
    with pytest.raises(RuntimeError, match='no-mistral-ocr'):
        batch.submit(SimpleNamespace(upload_workers=1, state=tmp_path/'state.json'), state)
    assert state['phase'] == 'uploaded'
