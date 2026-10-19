from __future__ import annotations
import copy
import sqlite3
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock,AsyncMock
import pytest
from citation_graph.debug_apply import apply_with_runtime,Runtime,apply_review
from citation_graph.debug_store import ReviewConflict
from src.document_structure import build_document_structure,attach_structure_metadata


class FakeRuntime:
    def __init__(self,fail=None):
        self.old=[{'id':'A1','text':'Original','metadata':{'itemKey':'ITEM1234','attachmentKey':'ABCD1234','page':1,'source_type':'pdf','zone':'body'}},
                  {'id':'NOTE','text':'Note','metadata':{'itemKey':'ITEM1234','noteKey':'N','source_type':'note'}}]
        self.current=copy.deepcopy(self.old);self.saved=[];self.fail=fail;self.restored=False
        self.build=build_document_structure;self.attach=attach_structure_metadata
        self.close=Mock();self.verify_source=Mock();self.invalidate_references=Mock()
    def rows(self,item):return self.current
    def manifest(self):return {'files':{'ABCD1234':{}},'hnsw_validated':True}
    def snapshot(self,item):return {}
    def vectors(self,ids):return {key:[1.,0.] for key in ids}
    def invalidate(self):pass
    def save_manifest(self,value):self.saved.append(copy.deepcopy(value))
    def write(self,rows,edited):
        for row in rows:
            self.current=[copy.deepcopy(row) if old['id']==row['id'] else old for old in self.current]
        if self.fail=='write':raise RuntimeError('write failed')
    def structure(self,*args):
        if self.fail=='structure':raise RuntimeError('structure failed')
    def verify(self,*args):
        if self.fail=='verify':raise RuntimeError('query failed')
    def restore(self,journal):self.current=copy.deepcopy(journal['rows']);self.restored=True


def data(runtime):
    document={'item_key':'ITEM1234','attachment_key':'ABCD1234','all_chunks':runtime.old[:1], 'source_sha256':'source','version':'version'}
    draft={'revision':1,'version':'version','edits':{'A1':'Corrected'},'toc':[]}
    return document,draft


def test_manual_apply_changes_only_requested_text_and_keeps_a_journal(tmp_path):
    runtime=FakeRuntime();document,draft=data(runtime)
    apply_with_runtime(runtime,document,draft,tmp_path)
    assert runtime.current[0]['text']=='Corrected' and runtime.current[1]['text']=='Note'
    assert runtime.saved[-1]['hnsw_validated'] is True
    assert runtime.saved[-1]['inflight_attachments']==[]
    assert runtime.saved[-1]['files']['ABCD1234']['manual_review']['revision']==1
    assert (tmp_path/'data/debug_rollback/current.json.gz').exists()
    assert not runtime.restored


@pytest.mark.parametrize('phase',['write','structure','verify'])
def test_manual_apply_compensates_failures(tmp_path,phase):
    runtime=FakeRuntime(phase);document,draft=data(runtime)
    with pytest.raises(RuntimeError):apply_with_runtime(runtime,document,draft,tmp_path)
    assert runtime.current==runtime.old and runtime.restored


def test_manual_apply_preflight_has_no_destructive_writes(tmp_path):
    runtime=FakeRuntime();document,draft=data(runtime)
    document['all_chunks']=[]
    with pytest.raises(ValueError):apply_with_runtime(runtime,document,draft,tmp_path)
    document['all_chunks']=runtime.old[:1]
    runtime.build=lambda *a:{'diagnostics':{'valid':False}}
    with pytest.raises(ValueError):apply_with_runtime(runtime,document,draft,tmp_path)
    assert not runtime.saved and not (tmp_path/'data/debug_rollback').exists()


@pytest.fixture
def runtime(tmp_path,monkeypatch):
    from src import embedder,db_relations,lexical_index,v3_runtime,v3_data_plane
    collection=Mock();collection._zotero_embedder_config={};collection._chroma_client=Mock()
    monkeypatch.setattr(embedder,'get_collection',lambda **kw:collection)
    pipeline=tmp_path/'pipeline.json';pipeline.write_text('{"pipeline_fingerprint":"OK"}')
    monkeypatch.setattr(v3_data_plane,'pipeline_config_path',lambda root:pipeline)
    monkeypatch.setattr(v3_runtime,'pipeline_payload',lambda *a,**kw:{'pipeline_fingerprint':'OK'})
    monkeypatch.setattr(v3_data_plane,'manifest_path',lambda root:tmp_path/'manifest.json')
    db=tmp_path/'relations.db'
    monkeypatch.setattr(db_relations,'get_db_connection',lambda:sqlite3.connect(db))
    monkeypatch.setattr(db_relations,'replace_document_structure',Mock())
    monkeypatch.setattr(db_relations,'mark_artifact_status',Mock())
    monkeypatch.setattr(db_relations,'update_item_citation_status',Mock())
    monkeypatch.setattr(lexical_index,'upsert_chunks',Mock())
    value=Runtime(tmp_path)
    monkeypatch.setattr(value,'verify',Mock())
    return value,pipeline,collection


def test_runtime_updates_metadata_without_reembedding_unchanged_chunks(runtime,monkeypatch):
    value,pipeline,col=runtime
    old=FakeRuntime().old
    value.save_manifest({'files':{}});assert value.manifest()['files']=={}
    assert value.snapshot('ITEM')=={}
    assert value.vectors([])=={}
    col.get.return_value={'ids':['A1'],'embeddings':[SimpleNamespace(tolist=lambda:[1.,0.])]}
    assert value.vectors(['A1'])=={'A1':[1.,0.]}
    col.get.return_value={'ids':[],'embeddings':[]}
    with pytest.raises(RuntimeError):value.vectors(['MISSING'])
    value.write(old,{'A1'})
    calls=col.update.call_args_list
    assert calls[0].kwargs['ids']==['A1','NOTE'] and 'documents' not in calls[0].kwargs
    assert calls[1].kwargs['documents']==['Original']
    value.structure('ITEM1234',build_document_structure('ITEM1234',old[:1]))
    col.get.return_value={'ids':['A1','NOTE'],'metadatas':[{'extra':'new'},{'extra':'new'}]}
    value.restore({'rows':old,'vectors':{'A1':[1.,0.]},'item_key':'ITEM1234','relations':{},'manifest':{'restored':True}})
    assert value.manifest()['restored'] is True
    (value.root/'data/quality').mkdir(parents=True)
    gate=value.root/'data/quality/server_database_gate.json';gate.write_text('success')
    value.invalidate();assert not gate.exists()
    value.invalidate_references('ITEM1234')
    value.db.update_item_citation_status.assert_called_once_with('ITEM1234','pending')
    value.close();col._chroma_client.close.assert_called_once()
    pipeline.write_text('{"pipeline_fingerprint":"WRONG"}')
    with pytest.raises(RuntimeError):Runtime(value.root)


def test_writer_revalidates_revision_and_source_before_mutating(tmp_path,monkeypatch):
    from citation_graph import debug_apply as module
    from src import indexing_lock,database_gate,v3_data_plane
    from citation_graph import debug_reader
    runtime=FakeRuntime();document,draft=data(runtime)
    reader=Mock();reader.resolve=AsyncMock(return_value=(tmp_path/'pdf','ITEM1234'));reader.inspect.return_value=document;reader.store.read.return_value=draft
    monkeypatch.setattr(module,'DebugReader',lambda root:reader)
    monkeypatch.setattr(module,'Runtime',lambda root:runtime)
    monkeypatch.setattr(indexing_lock,'held',lambda path:nullcontext())
    monkeypatch.setattr(v3_data_plane,'enforce_environment',lambda root:None)
    validator=Mock();monkeypatch.setattr(database_gate,'validate_database_gate',validator)
    monkeypatch.setattr(debug_reader,'source_stamp',lambda path:())
    monkeypatch.setattr(debug_reader,'source_hash',lambda *a:'source')
    apply_review(tmp_path,'ABCD1234',1);assert runtime.close.called and validator.called
    with pytest.raises(ReviewConflict):apply_review(tmp_path,'ABCD1234',2)
    draft['edits']={}
    with pytest.raises(ValueError):apply_review(tmp_path,'ABCD1234',1)
    draft['edits']={'A1':'Corrected'}
    monkeypatch.setattr(debug_reader,'source_hash',lambda *a:'changed')
    with pytest.raises(ReviewConflict):apply_review(tmp_path,'ABCD1234',1)


def test_native_rollback_restores_text_vectors_and_removes_added_metadata(tmp_path,monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"src"))
    import chromadb
    from src import embedder,v3_data_plane,v3_runtime,db_relations,lexical_index
    native=tmp_path/'chroma'
    client=chromadb.PersistentClient(path=str(native))
    col=client.get_or_create_collection('zotero_paragraphs_v3',embedding_function=None)
    class Embedding:
        def __call__(self,input):return [[float(len(text)),1.] for text in input]
    col._embedding_function=Embedding();col._chroma_client=client;col._zotero_embedder_config={}
    metadata={'itemKey':'ITEM1234','attachmentKey':'ABCD1234','page':1,'source_type':'pdf','zone':'body'}
    col.upsert(ids=['A1'],documents=['Original'],metadatas=[metadata],embeddings=[[1.,0.]])
    lexical=tmp_path/'lexical.sqlite3'
    upsert=lexical_index.upsert_chunks
    upsert(['A1'],['Original'],[metadata],path=lexical)
    monkeypatch.setattr(lexical_index,'upsert_chunks',lambda ids,texts,metas:upsert(ids,texts,metas,path=lexical))
    monkeypatch.setattr(embedder,'get_collection',lambda **kw:col)
    monkeypatch.setattr(v3_data_plane,'chroma_dir',lambda root:native)
    monkeypatch.setattr(v3_data_plane,'manifest_path',lambda root:tmp_path/'manifest.json')
    pipeline=tmp_path/'pipeline.json';pipeline.write_text('{"pipeline_fingerprint":"OK"}')
    monkeypatch.setattr(v3_data_plane,'pipeline_config_path',lambda root:pipeline)
    monkeypatch.setattr(v3_runtime,'pipeline_payload',lambda *a,**kw:{'pipeline_fingerprint':'OK'})
    monkeypatch.setattr(db_relations,'get_db_connection',lambda:sqlite3.connect(tmp_path/'relations.db'))
    (tmp_path/'manifest.json').write_text('{"files":{"ABCD1234":{}},"hnsw_validated":true}')
    runtime=Runtime(tmp_path);runtime.verify_source=Mock()
    original=runtime.rows('ITEM1234')
    def fail(*a):raise RuntimeError('structure write failed')
    monkeypatch.setattr(runtime,'structure',fail)
    document={'item_key':'ITEM1234','attachment_key':'ABCD1234','all_chunks':original,'source_sha256':'SHA'}
    draft={'revision':1,'edits':{'A1':'Corrected'},'toc':[]}
    try:
        with pytest.raises(RuntimeError,match='structure write failed'):
            apply_with_runtime(runtime,document,draft,tmp_path)
        result=col.get(ids=['A1'],include=['documents','metadatas','embeddings'])
        assert result['documents']==['Original'] and result['metadatas']==[metadata]
        assert result['embeddings'][0].tolist()==[1.,0.]
        with sqlite3.connect(lexical) as c:
            assert c.execute('SELECT body FROM chunks_fts').fetchone()[0]=='Original'
    finally:runtime.close()


def test_zone_only_correction_updates_structure_and_invalidates_references(tmp_path):
    runtime=FakeRuntime();document,draft=data(runtime)
    draft['edits']={};draft['zones']={'A1':'footnote'}
    apply_with_runtime(runtime,document,draft,tmp_path)
    assert runtime.current[0]['metadata']['zone']=='footnote'
    assert runtime.current[0]['text']=='Original'
    runtime.invalidate_references.assert_called_once_with('ITEM1234')
    runtime=FakeRuntime('verify');document,draft=data(runtime)
    runtime.old[0]['metadata']['zone']='endnote';runtime.current[0]['metadata']['zone']='endnote'
    with pytest.raises(RuntimeError):apply_with_runtime(runtime,document,draft,tmp_path)
    runtime.invalidate_references.assert_called_once_with('ITEM1234')
    assert runtime.current==runtime.old and runtime.restored


def test_manual_zone_is_never_silently_overwritten_by_structure_guard(tmp_path):
    runtime=FakeRuntime();document,draft=data(runtime)
    draft['edits']={};draft['zones']={'A1':'bibliography'}
    with pytest.raises(ValueError,match='分類'):apply_with_runtime(runtime,document,draft,tmp_path)
    assert runtime.current==runtime.old and not runtime.saved


def test_reference_refresh_uses_text_and_effective_zone_changes():
    from citation_graph.debug_apply import references_changed
    body=[{'id':'A','metadata':{'zone':'body'}}]
    bibliography=[{'id':'A','metadata':{'zone':'bibliography'}}]
    assert references_changed(body,bibliography,{})
    assert references_changed(bibliography,body,{})
    assert references_changed(bibliography,bibliography,{'A':'corrected'})
    assert not references_changed(body,body,{'A':'corrected'})
    assert not references_changed(bibliography,bibliography,{})
    assert references_changed(bibliography,[],{})
