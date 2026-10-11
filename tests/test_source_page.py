import hashlib
import io
import json

import fitz
import pytest
from PIL import Image
from src.source_page import render_source_page


@pytest.fixture
def paper(tmp_path):
    path = tmp_path/'paper.pdf'
    with fitz.open() as doc:
        page = doc.new_page(width=600, height=800)
        page.insert_text((40, 50), 'Original page - OCR verification', fontsize=18)
        page.draw_rect(fitz.Rect(40, 90, 560, 300), color=(0, 0, 0))
        page.insert_text((60, 125), 'Variety       Region       Year', fontsize=16)
        page.insert_text((60, 175), 'Sample A      North        1930', fontsize=16)
        page.insert_text((60, 225), 'Sample B      South        1931', fontsize=16)
        doc.new_page(width=600, height=800).insert_text((40, 50), 'Second page')
        doc.set_page_labels([{'startpage': 0, 'prefix': '', 'style': 'r', 'firstpagenum': 1}])
        doc.save(path)
    return path


@pytest.mark.parametrize('rotation', [0, 90, 180, 270])
def test_render_returns_bounded_image_and_exact_provenance(paper, rotation):
    metadata, content = render_source_page(paper, 1, rotation=rotation)
    image = Image.open(io.BytesIO(content))
    assert image.format == 'JPEG'
    assert image.size == (metadata['width'], metadata['height'])
    assert max(image.size) == 1800
    assert metadata['pdf_page'] == 1 and metadata['page_label'] == 'i'
    assert metadata['total_pages'] == 2 and metadata['writes_performed'] is False
    assert metadata['image_sha256'] == hashlib.sha256(content).hexdigest()
    assert not (paper.parent/'paper.jpg').exists()


def test_crop_magnifies_table_and_keeps_source_unchanged(paper):
    before = paper.read_bytes()
    metadata, _ = render_source_page(paper, 1, crop=[0, 0, 1, .5])
    assert (metadata['width'], metadata['height']) == (1800, 1200)
    assert paper.read_bytes() == before


@pytest.mark.parametrize('page,crop,rotation,match', [
    (0, None, 0, '1-based'), (True, None, 0, '1-based'), (1.5, None, 0, '1-based'),
    (3, None, 0, 'exceeds'), (1, None, 45, 'rotation'),
    (1, [0, 0, 1], 0, 'crop'), (1, [0, 0, 2, 1], 0, 'crop'),
    (1, [0, 0, float('nan'), 1], 0, 'crop'), (1, [0, 0, '1', 1], 0, 'crop'),
    (1, [0, 0, 0, 1], 0, 'crop'), (1, [0, 0, .0001, 1], 0, 'too small'),
])
def test_invalid_requests_fail_with_guidance(paper, page, crop, rotation, match):
    with pytest.raises(ValueError, match=match):
        render_source_page(paper, page, crop=crop, rotation=rotation)


def test_non_pdf_and_encrypted_pdf_are_rejected(tmp_path):
    path = tmp_path/'image.png'
    Image.new('RGB', (30, 30)).save(path)
    with pytest.raises(ValueError, match='not a PDF'):
        render_source_page(path, 1)
    path = tmp_path/'encrypted.pdf'
    with fitz.open() as doc:
        doc.new_page()
        doc.save(path, encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw='owner', user_pw='secret')
    with pytest.raises(ValueError, match='encrypted'):
        render_source_page(path, 1)


def test_mcp_tool_returns_image_and_chunk_provenance(paper, monkeypatch):
    import asyncio
    from tests.test_mcp_app_factory import server
    class API:
        async def get_item(self, key):
            return {'key': key, 'data': {'itemType': 'attachment', 'parentItem': 'PARENT01'}}
        @staticmethod
        def _unwrap_item(raw):
            return raw['key'], raw['data']
        @staticmethod
        def resolve_pdf_path_from_attachment(*args):
            return str(paper)
    monkeypatch.setattr(server, '_z_api', lambda: API())
    monkeypatch.setattr(server, '_chunk_by_id', lambda key: {'metadata': {'attachmentKey': 'SOURCE01', 'page': 2}})
    blocks = asyncio.run(server.get_source_page(chunk_id='chunk-1'))
    assert blocks[0].type == 'text' and blocks[1].type == 'image'
    assert blocks[1].mimeType == 'image/jpeg'
    result = asyncio.run(server.create_mcp().call_tool('get_source_page', {'chunk_id': 'chunk-1'}))
    assert any(block.type == 'image' and block.mimeType == 'image/jpeg' for block in result.content)
    metadata = json.loads(blocks[0].text)
    assert metadata['attachment_key'] == 'SOURCE01' and metadata['pdf_page'] == 2
    assert metadata['item_key'] == 'PARENT01'
    with pytest.raises(ValueError, match='does not match'):
        asyncio.run(server.get_source_page(chunk_id='chunk-1', attachment_key='OTHER001'))


@pytest.mark.parametrize('chunk,key,page,match', [
    ('missing', '', None, 'not found'), ('', '../file', 1, 'attachment_key'),
    ('', 'SOURCE01', None, 'pdf_page'), ('no-page', '', None, 'pdf_page'),
])
def test_tool_rejects_unresolvable_requests_before_api(monkeypatch, chunk, key, page, match):
    import asyncio
    from tests.test_mcp_app_factory import server
    monkeypatch.setattr(server, '_chunk_by_id', lambda key: None if key == 'missing' else {'metadata': {'attachmentKey': 'SOURCE01'}})
    monkeypatch.setattr(server, '_z_api', lambda: pytest.fail('invalid source must not contact Zotero'))
    with pytest.raises(ValueError, match=match):
        asyncio.run(server.get_source_page(chunk_id=chunk, attachment_key=key, pdf_page=page))


@pytest.mark.parametrize('item_type,path,match', [('book', 'unused', 'attachment'), ('attachment', None, '同期')])
def test_tool_reports_non_attachment_and_unsynced_source(monkeypatch, item_type, path, match):
    import asyncio
    from tests.test_mcp_app_factory import server
    class API:
        async def get_item(self, key):
            return {'key': key, 'data': {'itemType': item_type}}
        @staticmethod
        def _unwrap_item(raw):
            return raw['key'], raw['data']
        @staticmethod
        def resolve_pdf_path_from_attachment(*args):
            return path
    monkeypatch.setattr(server, '_z_api', lambda: API())
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        asyncio.run(server.get_source_page(attachment_key='SOURCE01', pdf_page=1))


def test_oversized_image_is_rejected(paper, monkeypatch):
    monkeypatch.setattr(fitz.Pixmap, 'tobytes', lambda *args, **kwargs: b'x'*(6*1024**2+1))
    with pytest.raises(ValueError, match='too large'):
        render_source_page(paper, 1)
