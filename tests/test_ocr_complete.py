from types import SimpleNamespace
from unittest.mock import Mock
import sys
from pathlib import Path

import fitz
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import index_from_zotero as ingest
import pdf_extract
from src import mistral_send_policy as policy
from tests.test_mistral_send_policy import client


@pytest.mark.parametrize('tags,parent,expected', [
    ([], [], False), (['ocr-complete'], [], True),
    ([], [{'tag': ' OCR-COMPLETE '}], True),
    (['no-mistral-ocr', 42, {}], [], False),
])
def test_complete_tag_is_distinct(tags, parent, expected):
    assert policy.attachment_ocr_complete(SimpleNamespace(tags=tags, parentTags=parent)) is expected


@pytest.mark.parametrize('tags,parent', [(['ocr-complete'], []), ([], ['ocr-complete'])])
def test_complete_prevents_fresh_paid_send(monkeypatch, tags, parent):
    client(monkeypatch, {'itemType': 'attachment', 'parentItem': 'PARENT01', 'tags': tags}, {'tags': parent})
    with pytest.raises(RuntimeError, match='ocr-complete'):
        policy.assert_mistral_send_allowed('SOURCE01')


@pytest.mark.parametrize('chunks', [[], [('id', 'existing text', {})]])
def test_complete_uses_text_layer_even_with_forced_local_overrides(monkeypatch, chunks):
    quality = {'is_corrupted': True, 'empty_pages': [2]}
    extract = Mock(return_value=(chunks, quality))
    monkeypatch.setattr(ingest, 'extract_chunks_from_pdf', extract)
    for name in ['_extract_pdf_override', '_prepare_pdf_for_structure_gate',
                 '_recover_pdf_outline_with_ai_toc', '_dispatch_pdf_gate_action']:
        monkeypatch.setattr(ingest, name, Mock(side_effect=AssertionError('must not reprocess')))
    result = ingest._extract_pdf_chunks(
        a=SimpleNamespace(attachmentKey='SOURCE01', parentTags=('ocr-complete',)),
        args=SimpleNamespace(use_docling=True), col=None, docling_worker=None,
        file_path=Path('source.pdf'), files_manifest={}, force_docling=True,
        force_ndlocr=True, granite_worker=None, manifest={}, meta_base={},
        mtime=1, prev=None, scope_item_key='PARENT01', show_progress=False,
        size=1, source_metadata={}, stored_signature=None, structure_recovery=True,
        v3_pipeline_fingerprint='test',
    )
    assert result.chunks == chunks
    assert result.quality['is_corrupted'] is True
    assert result.quality['empty_pages'] == [2]
    assert result.quality['ocr_policy'] == 'ocr-complete'
    assert extract.call_args.args[2]['ocr_complete'] is True
    assert not result.deferred


def test_text_layer_extractor_never_calls_tesseract_when_complete(monkeypatch, tmp_path):
    path = tmp_path / 'source.pdf'
    with fitz.open() as doc:
        doc.new_page().insert_text((40, 40), 'Existing OCR text remains usable for search.')
        doc.save(path)
    monkeypatch.setattr(pdf_extract, 'PDF_OCR_FALLBACK', True)
    monkeypatch.setattr(pdf_extract, 'PDF_POPPLER_TEXT_FALLBACK', False)
    analyze = pdf_extract.analyze_text_quality
    def degraded(text):
        return {**analyze(text), 'corruption_type': 'extraction_failure', 'extraction_failure_score': 0.9, 'is_corrupted': True, 'is_scanned': False}
    monkeypatch.setattr(pdf_extract, 'analyze_text_quality', degraded)
    ocr = Mock(side_effect=AssertionError('Tesseract must not run'))
    monkeypatch.setattr(pdf_extract, '_ocr_page_with_tesseract', ocr)
    chunks, quality = pdf_extract.extract_chunks_from_pdf(path, 'SOURCE01', {'ocr_complete': True})
    assert chunks
    assert any('Existing OCR text' in row[1] for row in chunks)
    assert quality['extraction_failure_pages'] == [1]
    ocr.assert_not_called()


def test_explicit_mistral_route_cannot_override_complete_tag():
    args = SimpleNamespace(use_docling=False, reparse_corrupted=False)
    route = {'target_engine': 'mistral_ocr'}
    tagged = SimpleNamespace(tags=('ocr-complete',))
    assert not ingest._reparse_decision(args, source_type='pdf', previous=None, reocr_route=route, attachment=tagged).force_mistral
    assert ingest._reparse_decision(args, source_type='pdf', previous=None, reocr_route=route).force_mistral


def test_complete_is_removed_from_batch_candidates(monkeypatch):
    from scripts import run_mistral_ocr_batch as batch
    import asyncio
    monkeypatch.setattr(batch, 'get_artifact_processing_statuses', lambda **kwargs: [])
    monkeypatch.setattr(batch, 'build_queue_rows', lambda rows: [{'attachment_key': 'SOURCE01', 'item_key': 'PARENT01'}])
    class API:
        async def list_normalized_attachments(self, **kwargs):
            return [SimpleNamespace(attachmentKey='SOURCE01', parentTags=('ocr-complete',), tags=())]
    monkeypatch.setattr(batch, 'ZoteroLocalAPI', API)
    monkeypatch.setattr(batch, 'mistral_ocr_available', lambda: pytest.fail('must not prepare external OCR'))
    assert asyncio.run(batch.resolve_candidates(None)) == []
