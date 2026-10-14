from __future__ import annotations
import json
import sqlite3
from unittest.mock import Mock, AsyncMock
import fitz
import pytest
from citation_graph.debug_store import DraftStore, ReviewConflict, validate_draft
from citation_graph.debug_changes import corrected_rows
from citation_graph.debug_journal import write_journal,snapshot_relations,restore_relations
from citation_graph.debug_reader import DebugReader,current_toc,source_hash


@pytest.fixture
def review(tmp_path, monkeypatch):
    import citation_graph.debug_reader as reader
    path=tmp_path/'source.pdf'
    with fitz.open() as doc:
        doc.new_page().insert_text((40,40),'Original text')
        doc.new_page()
        doc.save(path)
    rows=[{'id':'ATT_1','text':'Original text','metadata':{'attachmentKey':'ABCD1234','itemKey':'ITEM1234','page':1,'title':'Book','structure_path':'["Chapter"]'}},
          {'id':'ATT_2','text':'Second','metadata':{'attachmentKey':'ABCD1234','itemKey':'ITEM1234','page':2}}]
    manifest=tmp_path/'manifest_v3.json';manifest.write_text(json.dumps({'files':{'ABCD1234':{'pdf_path':str(path),'title':'Book','quality':{'source_coverage_adopted':True}},'HTML0001':{'pdf_path':'page.html'}}}))
    db=tmp_path/'parents.db'
    with sqlite3.connect(db) as c:
        c.execute('CREATE TABLE document_nodes(item_key,attachment_key,node_type)')
        c.execute("INSERT INTO document_nodes VALUES('ITEM1234','ABCD1234','attachment_root')")
    monkeypatch.setattr(reader,'get_db_connection',lambda:sqlite3.connect(db))
    monkeypatch.setattr(reader,'manifest_path',lambda root:manifest)
    monkeypatch.setattr(reader,'chroma_dir',lambda root:tmp_path/'chroma')
    monkeypatch.setattr(reader,'get_item_chunks',lambda *a,**kw:rows)
    monkeypatch.setattr(reader,'get_artifact_processing_statuses',lambda:[{'artifact_type':'structure','status':'degraded','item_key':'ITEM1234','reason_code':'flat_fallback'}, {'artifact_type':'extraction','status':'failed','attachment_key':'ABCD1234'}])
    monkeypatch.setattr(reader,'get_chunk_quality_reports',lambda:[{'attachment_key':'ABCD1234'}])
    return DebugReader(tmp_path),path,rows,manifest


def test_store_versions_conflicts_and_no_canonical_writes(tmp_path):
    store=DraftStore(tmp_path/'draft.sqlite3')
    assert store.read('ATT')['revision']==0
    store.path.touch()
    assert store.read('ATT')['revision']==0
    payload={'version':'v','edits':{'A':'corrected'},'toc':[]}
    assert store.save('ATT',0,payload,'reviewer')['revision']==1
    assert store.read('ATT')['edits']==payload['edits']
    assert store.read('OTHER')['revision']==0
    with pytest.raises(ReviewConflict):store.save('ATT',0,payload,'other')
    assert store.save('ATT',1,payload,'reviewer')['revision']==2
    with pytest.raises(ValueError):store.save('ATT',2,{'large':'x'*(2*1024**2)},'reviewer')


def test_draft_validation_and_transformation(review):
    _,_,rows,_=review
    toc=[{'chunk_id':'ATT_1','level':1,'title':'Fixed heading'},{'chunk_id':'ATT_2','level':2,'title':'Section'}]
    changed=corrected_rows(rows,{'ATT_1':'Corrected text'},toc)
    assert rows[0]['text']=='Original text'
    assert changed[0]['text']=='Corrected text'
    assert json.loads(changed[1]['metadata']['structure_path'])==['Fixed heading','Section']
    assert corrected_rows(rows,{},[])==rows
    edits=[{'UNKNOWN':'x'},{'ATT_1':''},{'ATT_1':12},{'ATT_1':'x'*30001},{str(n):'x' for n in range(501)}]
    for edit in edits:
        with pytest.raises(ValueError):validate_draft(rows,edit,[])
    cases=[[{'chunk_id':'ATT_2','level':1,'title':'second'},{'chunk_id':'ATT_1','level':1,'title':'first'}],
           [{'chunk_id':'UNKNOWN','level':1,'title':'Title'}],toc*251]
    for level in [True,0,3,9,'1']:
        cases.append([{'chunk_id':'ATT_1','level':level,'title':'Title'}])
    for title in ['',None,'x'*301]:cases.append([{'chunk_id':'ATT_1','level':1,'title':title}])
    for case in cases:
        with pytest.raises(ValueError):validate_draft(rows,{},case)


def test_inspection_inventory_draft_conflict_and_toc(review):
    reader,path,rows,manifest=review
    inventory=reader.inventory()
    assert len(inventory)==1 and 'flat_fallback' in inventory[0]['issues']
    doc=reader.inspect('ABCD1234',path,'ITEM1234',1)
    assert doc['total_pages']==2 and len(doc['chunks'])==1
    assert doc['current_toc'][0]['title']=='Chapter'
    saved=reader.save(doc,0,doc['version'],{'ATT_1':'Corrected'},[], 'user')
    assert saved['revision']==1
    assert not reader.inspect('ABCD1234',path,'ITEM1234',2)['draft_stale']
    with pytest.raises(ReviewConflict):reader.save(doc,1,'wrong',{},[],'user')
    rows[0]['text']='Changed in index'
    changed=reader.inspect('ABCD1234',path,'ITEM1234',1)
    assert changed['draft_stale']
    m=json.loads(manifest.read_text());m['files']['ABCD1234']['manual_review']={'revision':1,'version':changed['version']};manifest.write_text(json.dumps(m))
    assert reader.inspect('ABCD1234',path,'ITEM1234',1)['draft_applied']
    for page in [0,3]:
        with pytest.raises(ValueError):reader.inspect('ABCD1234',path,'ITEM1234',page)
    assert current_toc([{'id':'A','metadata':{'structure_path':'plain heading'}},{'id':'B','metadata':{'structure_path':12}}])[0]['title']=='plain heading'
    with pytest.raises(ReviewConflict):source_hash(path,(0,))


def test_source_resolver_refuses_invalid_deleted_and_missing(review,monkeypatch):
    import asyncio
    reader,path,_,_=review
    import citation_graph.debug_reader as module
    data={'itemType':'attachment','parentItem':'ITEM1234'}
    api=Mock();api.get_item=AsyncMock(return_value={});api._unwrap_item.return_value=('ABCD1234',data);api.resolve_pdf_path_from_attachment.return_value=str(path)
    monkeypatch.setattr(module,'ZoteroLocalAPI',lambda:api)
    assert asyncio.run(reader.resolve('ABCD1234'))==(path,'ITEM1234')
    for key in ['../bad','short']:
        with pytest.raises(ValueError):asyncio.run(reader.resolve(key))
    data['deleted']=True
    with pytest.raises(ValueError):asyncio.run(reader.resolve('ABCD1234'))
    data.pop('deleted');api.resolve_pdf_path_from_attachment.return_value=''
    with pytest.raises(FileNotFoundError):asyncio.run(reader.resolve('ABCD1234'))


def test_small_journal_restores_only_selected_item(tmp_path):
    c=sqlite3.connect(':memory:')
    c.execute('CREATE TABLE document_nodes(node_id PRIMARY KEY,item_key)')
    c.execute('CREATE TABLE document_node_chunks(node_id,chunk_id)')
    c.executemany('INSERT INTO document_nodes VALUES(?,?)',[('N','ITEM'),('OTHER','OTHER')])
    c.execute("INSERT INTO document_node_chunks VALUES('N','CHUNK')");c.commit()
    snapshot=snapshot_relations(c,'ITEM')
    c.execute("UPDATE document_node_chunks SET chunk_id='BROKEN'");c.commit()
    restore_relations(c,'ITEM',snapshot)
    assert c.execute('SELECT chunk_id FROM document_node_chunks').fetchone()[0]=='CHUNK'
    assert c.execute('SELECT count(*) FROM document_nodes').fetchone()[0]==2
    path=tmp_path/'journal/current.json.gz';write_journal(path,{'relations':snapshot})
    import gzip
    assert json.loads(gzip.decompress(path.read_bytes()))['relations']==snapshot


def test_history_survives_restore_and_is_paginated(tmp_path):
    store=DraftStore(tmp_path/'history.sqlite3')
    assert store.history('ATT')==[]
    with pytest.raises(FileNotFoundError):store.revision('ATT',1)
    store.path.touch()
    assert store.history('ATT')==[]
    with pytest.raises(FileNotFoundError):store.revision('ATT',1)
    for n in range(52):
        store.save('ATT',n,{'edits':{'C':str(n)},'toc':[]},'reviewer')
    first=store.history('ATT')
    assert len(first)==50 and first[0]['revision']==52
    assert [x['revision'] for x in store.history('ATT',first[-1]['revision'])]==[2,1]
    assert store.revision('ATT',1)['edits']=={'C':'0'}
    with pytest.raises(FileNotFoundError):store.revision('ATT',53)
    assert store.history('OTHER')==[]


def test_empty_toc_explicitly_clears_headings(review):
    _,_,rows,_=review
    cleared=corrected_rows(rows,{},[],toc_changed=True)
    assert json.loads(cleared[0]['metadata']['structure_path'])==[]
    assert cleared[0]['metadata']['chapter']==''
    assert rows[0]['metadata']['structure_path']=='["Chapter"]'


def test_source_api_errors_are_actionable(review,monkeypatch):
    import asyncio
    import httpx
    import citation_graph.debug_reader as module
    reader,_,_,_=review
    api=Mock()
    monkeypatch.setattr(module,'ZoteroLocalAPI',lambda:api)
    for status,exception in [(404,FileNotFoundError),(500,RuntimeError)]:
        response=httpx.Response(status,request=httpx.Request('GET','http://localhost/'))
        api.get_item=AsyncMock(side_effect=httpx.HTTPStatusError('error',request=response.request,response=response))
        with pytest.raises(exception):asyncio.run(reader.resolve('ABCD1234'))
    api.get_item=AsyncMock(side_effect=httpx.ConnectError('offline'))
    with pytest.raises(RuntimeError,match='起動'):asyncio.run(reader.resolve('ABCD1234'))


def test_journal_size_and_verification_fail_before_replacing(tmp_path,monkeypatch):
    import citation_graph.debug_journal as journal
    path=tmp_path/'current.json.gz';path.write_bytes(b'previous')
    monkeypatch.setattr(journal,'MAX_JOURNAL_BYTES',2)
    with pytest.raises(ValueError):write_journal(path,{'rows':[]})
    monkeypatch.setattr(journal,'MAX_JOURNAL_BYTES',1000)
    monkeypatch.setattr(journal.gzip,'decompress',lambda raw:b'corrupt')
    with pytest.raises(RuntimeError):write_journal(path,{'rows':[]})
    assert path.read_bytes()==b'previous'
    assert list(tmp_path.iterdir())==[path]


def test_fixed_debug_plan_rejects_arbitrary_commands(tmp_path):
    from citation_graph.debug_jobs import build_debug_job
    from citation_graph.admin_jobs import job_definitions
    python=tmp_path/'.venv/bin/python';python.parent.mkdir(parents=True);python.touch()
    definitions=job_definitions(tmp_path)
    for selection in [None,{}, {'attachment_key':'../x','revision':1}, {'attachment_key':'ABCD1234','revision':True}, {'attachment_key':'ABCD1234','revision':0}, {'attachment_key':'ABCD1234','revision':'1'}, {'attachment_key':1,'revision':1}]:
        with pytest.raises(ValueError):build_debug_job(tmp_path,definitions,selection)
    job=build_debug_job(tmp_path,definitions,{'attachment_key':'ABCD1234','revision':2})
    assert not job.paid and len(job.steps)==2
    assert job.steps[0][1][-4:]==('--attachment','ABCD1234','--revision','2')
    assert job.steps[1]==definitions['database_audit'].steps[0]


def test_password_protected_pdf_is_rejected(review):
    reader,path,_,_=review
    encrypted=path.with_name('encrypted.pdf')
    with fitz.open(path) as doc:
        doc.save(encrypted,encryption=fitz.PDF_ENCRYPT_AES_256,owner_pw='owner',user_pw='secret')
    with pytest.raises(ValueError,match='暗号化'):reader.inspect('ABCD1234',encrypted,'ITEM1234',1)


def test_debug_review_browser_interactions():
    import shutil
    import subprocess
    node=shutil.which('node')
    if node is None:
        pytest.skip('Node unavailable; DOM harness also runs with the supplied Node runtime')
    subprocess.run([node,'tests/debug_review_dom.cjs'],check=True)
