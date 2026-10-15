from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock,Mock

import pytest
from fastapi import Request

from citation_graph.debug_pdf import ReviewDocuments,byte_range,index_stamp,pdf_response
from citation_graph.debug_reader import source_stamp,verify_indexed_source
from citation_graph.debug_store import ReviewConflict


@pytest.fixture(autouse=True)
def isolated_review_paths(tmp_path,monkeypatch):
    monkeypatch.setenv('CHROMA_DIR',str(tmp_path/'data/chroma'))
    monkeypatch.setenv('MANIFEST_PATH',str(tmp_path/'data/manifest_v3.json'))


def test_single_ranges_are_bounded_and_reject_multipart():
    for value,expected in [(None,(0,99)),('bytes=0-15',(0,15)),('bytes=90-',(90,99)),('bytes=-10',(90,99)),('bytes=-1000',(0,99)),('bytes=0-1000',(0,99))]:
        assert byte_range(value,100)==expected
    for value in ['abc','bytes=','bytes=-','bytes=0-1,3-4','bytes=100-','bytes=10-1','bytes=abc-','bytes=-0','bytes=0-x','bytes=1']:
        with pytest.raises(ValueError):byte_range(value,100)


def test_pdf_routes_require_auth_and_support_get_head_ranges(debug_client):
    client,_=debug_client
    base='/admin/debug/api/documents/ABCD1234';headers={'Authorization':'test'}
    for suffix in ['/pdf?version=x','/state?version=x']:
        assert client.get(base+suffix).status_code==401
    version=client.get(base+'/pages/1',headers=headers).json()['version']
    url=base+'/pdf?version='+version
    full=client.get(url,headers=headers)
    assert full.status_code==200 and full.content.startswith(b'%PDF-')
    assert full.headers['content-type']=='application/pdf' and full.headers['accept-ranges']=='bytes'
    assert full.headers['cache-control']=='no-store'
    partial=client.get(url,headers={**headers,'Range':'bytes=0-15'})
    assert partial.status_code==206 and partial.content==full.content[:16]
    assert partial.headers['content-range']==f'bytes 0-15/{len(full.content)}'
    assert client.get(url,headers={**headers,'Range':'bytes=-10'}).content==full.content[-10:]
    head=client.head(url,headers=headers)
    assert head.status_code==200 and not head.content and int(head.headers['content-length'])==len(full.content)
    invalid=client.get(url,headers={**headers,'Range':'bytes=9000000-'})
    assert invalid.status_code==416 and invalid.headers['content-range']==f'bytes */{len(full.content)}'
    assert client.get(base+'/state?version='+version,headers=headers).status_code==200
    assert client.get(base+'/pdf?version=old',headers=headers).status_code==409
    assert client.get(base+'/state?version=old',headers=headers).status_code==409


def test_document_generation_guards_without_repeated_chunk_scan(tmp_path):
    path=tmp_path/'source.pdf';path.write_bytes(b'PDF')
    reader=Mock();reader.resolve=AsyncMock(return_value=(path,'ITEM'));reader.inspect.return_value={'version':'v'}
    documents=ReviewDocuments(tmp_path,reader)
    assert asyncio.run(documents.validate('ABCD1234','v'))[0]==path
    assert asyncio.run(documents.validate('ABCD1234','v'))[0]==path
    reader.inspect.assert_called_once()
    with pytest.raises(ReviewConflict):asyncio.run(documents.validate('ABCD1234','old'))
    (tmp_path/'data/chroma').mkdir(parents=True)
    db=tmp_path/'data/chroma/chroma.sqlite3';db.write_bytes(b'changed')
    assert index_stamp(tmp_path)[0] is not None
    with pytest.raises(ReviewConflict):asyncio.run(documents.validate('ABCD1234','v'))
    for n in range(5):asyncio.run(documents.inspect(f'AAAA000{n}',1))
    assert len(documents.sessions)==4 and 'AAAA0000' not in documents.sessions
    path.write_bytes(b'changed')
    with pytest.raises(ReviewConflict):asyncio.run(documents.validate('AAAA0004','v'))


def test_inspection_cannot_adopt_a_generation_changed_during_read(tmp_path):
    path=tmp_path/'source.pdf';path.write_bytes(b'PDF')
    reader=Mock();reader.resolve=AsyncMock(return_value=(path,'ITEM'))
    def inspect(*a):path.write_bytes(b'changed');return {'version':'v'}
    reader.inspect.side_effect=inspect
    with pytest.raises(ReviewConflict):asyncio.run(ReviewDocuments(tmp_path,reader).inspect('ABCD1234',1))


def test_source_replacement_waits_for_reindexing(tmp_path):
    from src.manifest import content_signature
    path=tmp_path/'source.pdf';path.write_bytes(b'PDF')
    stat=path.stat();entry={'size':stat.st_size,'mtime':stat.st_mtime,'content_signature':content_signature(path,stat.st_size)}
    manifest={'files':{'KEY':entry}}
    verify_indexed_source(path,'KEY',manifest,[{}])
    verify_indexed_source(path,'KEY',manifest,[])
    with pytest.raises(ReviewConflict):verify_indexed_source(path,'KEY',{'inflight_attachments':['KEY']},[])
    entry['content_signature']='wrong'
    with pytest.raises(ReviewConflict):verify_indexed_source(path,'KEY',manifest,[{}])
    entry.pop('content_signature');path.write_bytes(b'longer')
    with pytest.raises(ReviewConflict):verify_indexed_source(path,'KEY',manifest,[{}])


def test_stream_detects_replacement_before_and_during_transfer(tmp_path):
    path=tmp_path/'source.pdf';path.write_bytes(b'PDF')
    stamp=source_stamp(path);request=Request({'type':'http','method':'GET','headers':[]})
    path.write_bytes(b'changed')
    with pytest.raises(ReviewConflict):asyncio.run(pdf_response(path,stamp,request))
    async def transfer():
        response=await pdf_response(path,source_stamp(path),request)
        path.write_bytes(b'replaced')
        return [block async for block in response.body_iterator]
    with pytest.raises(ReviewConflict):asyncio.run(transfer())


def test_pdfjs_assets_are_pinned_complete_and_authenticated(debug_client,tmp_path):
    client,_=debug_client
    # Fixture serves its own fixed asset list; production assets are checked independently.
    static=Path(__file__).resolve().parents[1]/'citation_graph/static/pdfjs'
    manifest=json.loads((static/'assets.json').read_text())
    assert manifest['version']=='6.4.299' and 'LICENSE' in manifest['files']
    for name,digest in manifest['files'].items():
        assert hashlib.sha256((static/name).read_bytes()).hexdigest()==digest
    assert client.get('/admin/pdfjs/pdf.mjs').status_code==401

from tests.test_debug_routes import debug_client  # noqa: E402,F401


def test_pdfjs_fixed_assets_serve_only_manifest_entries(debug_client,tmp_path):
    client,_=debug_client;headers={'Authorization':'test'}
    static=tmp_path/'citation_graph/static/pdfjs';static.mkdir()
    assets={'pdf.mjs':b'js','pdf_viewer.css':b'css','wasm/decode.wasm':b'wasm','LICENSE':b'license'}
    for name,data in assets.items():
        path=static/name;path.parent.mkdir(exist_ok=True);path.write_bytes(data)
    (static/'assets.json').write_text(json.dumps({'files':list(assets)}))
    for name in assets:
        response=client.get('/admin/pdfjs/'+name,headers=headers)
        assert response.status_code==200 and response.content==assets[name]
    assert client.get('/admin/pdfjs/unknown',headers=headers).status_code==404


def test_partial_stream_rechecks_after_read(tmp_path,monkeypatch):
    import citation_graph.debug_pdf as module
    path=tmp_path/'file.pdf';path.write_bytes(b'PDF')
    request=Request({'type':'http','method':'GET','headers':[]})
    async def altered_read(method,size):
        data=method(size);path.write_bytes(b'changed');return data
    async def transfer():
        response=await pdf_response(path,source_stamp(path),request)
        monkeypatch.setattr(module.asyncio,'to_thread',altered_read)
        return [data async for data in response.body_iterator]
    with pytest.raises(ReviewConflict):asyncio.run(transfer())


def test_pdf_renderer_interactions():
    import shutil
    import subprocess
    node=shutil.which('node')
    if node is None:pytest.skip('Node unavailable; PDF DOM harness also runs in supplied runtime')
    subprocess.run([node,'tests/pdf_review_dom.cjs'],check=True)
