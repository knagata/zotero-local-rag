from __future__ import annotations

import copy
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from citation_graph.debug_apply import Runtime,apply_with_runtime
from citation_graph.debug_reader import current_toc
from citation_graph.debug_segments import history_plan
from src.document_structure import build_document_structure,attach_structure_metadata


@pytest.fixture
def native_review(tmp_path):
    import chromadb
    from src import lexical_index
    client=chromadb.PersistentClient(path=str(tmp_path/'chroma'))
    col=client.get_or_create_collection('split_review',embedding_function=None)
    class Embedding:
        def __call__(self,input):return [[float(len(text)),1.] for text in input]
    col._embedding_function=Embedding();col._chroma_client=client
    meta={'itemKey':'ITEM','attachmentKey':'ATT','page':1,'zone':'body','source_type':'pdf'}
    col.add(ids=['ATT_1','OTHER_1','NOTE_1'],documents=['Before.\nInaugurations\nText.','Sibling','Note'],
            metadatas=[meta,{**meta,'attachmentKey':'OTHER'},{'itemKey':'ITEM','source_type':'note'}],embeddings=[[1.,0.],[0.,1.],[1.,1.]])
    runtime=Runtime.__new__(Runtime)
    runtime.root=tmp_path;runtime.collection=col;runtime.gate_path=tmp_path/'gate'
    runtime.manifest_path=tmp_path/'manifest.json'
    from src.manifest import load_manifest,save_manifest
    runtime.load=load_manifest;runtime.save=save_manifest
    runtime.save_manifest({'files':{'ATT':{},'OTHER':{}},'hnsw_validated':True})
    runtime.db=SimpleNamespace(get_db_connection=lambda:sqlite3.connect(tmp_path/'relations.sqlite3'),
        replace_document_structure=Mock(),mark_artifact_status=Mock(),update_item_citation_status=Mock())
    lexical=tmp_path/'lexical.sqlite3'
    runtime.lexical=SimpleNamespace(
        upsert_chunks=lambda ids,texts,metas:lexical_index.upsert_chunks(ids,texts,metas,path=lexical),
        delete_by_chunk_ids=lambda ids:lexical_index.delete_by_chunk_ids(ids,path=lexical))
    def rows(item):
        result=col.get(where={'itemKey':item},include=['documents','metadatas'])
        from src.chunk_store import natural_chunk_key
        return sorted([{'id':key,'text':text,'metadata':md} for key,text,md in zip(result['ids'],result['documents'],result['metadatas'])],key=lambda row:natural_chunk_key(row['id']))
    runtime.rows=rows;runtime.build=build_document_structure;runtime.attach=attach_structure_metadata
    runtime.verify=lambda sample:col.query(query_embeddings=[[1.,0.]],n_results=1)
    runtime.verify_source=Mock()
    original=rows('ITEM')
    runtime.lexical.upsert_chunks([r['id'] for r in original],[r['text'] for r in original],[r['metadata'] for r in original])
    try:yield runtime,original,lexical
    finally:runtime.close()


def document(source):
    return {'item_key':'ITEM','attachment_key':'ATT','source_sha256':'sha','all_chunks':[r for r in source if r['metadata'].get('attachmentKey')=='ATT']}


def draft(source):
    row=source[0];offset=row['text'].index('Inaugurations')
    return {'revision':1,'edits':{},'toc_changed':True,'toc':[{'chunk_id':row['id'],'offset':offset,'anchor_text':row['text'],'page':1,'title':'Inaugurations','level':1}]}


def lexical_rows(path):
    with sqlite3.connect(path) as connection:
        return dict(connection.execute('SELECT chunk_id,body FROM chunks_fts'))


def test_native_split_and_merge_preserve_source_siblings_notes_vectors_and_fts(native_review):
    runtime,original,lexical=native_review
    apply_with_runtime(runtime,document(original),draft(original),runtime.root)
    live=runtime.rows('ITEM');target=document(live)['all_chunks']
    assert [r['text'] for r in target]==['Before.\n','Inaugurations\nText.']
    assert target[0]['metadata']['chapter']=='' and target[1]['metadata']['chapter']=='Inaugurations'
    assert ''.join(r['text'] for r in target)==original[0]['text']
    assert {r['id']:r['text'] for r in live}==lexical_rows(lexical)
    assert next(r for r in live if r['id']=='OTHER_1')['text']=='Sibling'
    assert next(r for r in live if r['id']=='NOTE_1')==next(r for r in original if r['id']=='NOTE_1')
    group=[r['id'] for r in target]
    undo={'revision':2,'edits':{'ATT_1':original[0]['text']},'toc':[],'toc_changed':True,'merges':[group]}
    apply_with_runtime(runtime,document(live),undo,runtime.root)
    restored=runtime.rows('ITEM')
    assert document(restored)['all_chunks'][0]['text']==original[0]['text']
    assert len(restored)==len(original)
    assert {r['id']:r['text'] for r in restored}==lexical_rows(lexical)
    assert runtime.manifest()['hnsw_validated']


@pytest.mark.parametrize('phase',['structure','verify_source','delete'])
def test_native_split_and_merge_failures_restore_exact_rows_vectors_and_fts(native_review,monkeypatch,phase):
    runtime,original,lexical=native_review
    if phase=='delete':
        apply_with_runtime(runtime,document(original),draft(original),runtime.root)
        original=runtime.rows('ITEM')
        group=[r['id'] for r in document(original)['all_chunks']]
        change={'revision':2,'edits':{},'toc':[],'toc_changed':True,'merges':[group]}
        delete=runtime.collection.delete
        def fail_delete(*a,**kw):
            delete(*a,**kw)
            raise RuntimeError('delete failed')
        monkeypatch.setattr(runtime.collection,'delete',fail_delete)
    else:
        change=draft(original)
        monkeypatch.setattr(runtime,phase,Mock(side_effect=RuntimeError('phase failed')))
    before=runtime.collection.get(include=['documents','metadatas','embeddings'])
    before_vectors={key:v.tolist() for key,v in zip(before['ids'],before['embeddings'])}
    manifest=copy.deepcopy(runtime.manifest())
    with pytest.raises(RuntimeError):apply_with_runtime(runtime,document(original),change,runtime.root)
    assert runtime.rows('ITEM')==original
    after=runtime.collection.get(include=['embeddings'])
    assert {key:v.tolist() for key,v in zip(after['ids'],after['embeddings'])}==before_vectors
    assert {r['id']:r['text'] for r in original}==lexical_rows(lexical)
    assert runtime.manifest()==manifest


def test_new_ids_collisions_fail_before_any_write(native_review):
    runtime,original,_=native_review
    with pytest.raises(ValueError,match='識別子'):runtime.check_new_ids({'ATT_1'})
    runtime.check_new_ids({'UNUSED'})


def test_history_can_undo_and_redo_a_merge_of_a_previous_split():
    from tests.test_debug_segments import rows,toc
    from citation_graph.debug_changes import corrected_rows
    source=rows();split=corrected_rows(source,{},[toc(source[0],8)])
    group=[row['id'] for row in split[:-1]]
    history={'edits':{},'originals':{'ATT_1':source[0]['text']},'toc':[],'toc_before':current_toc(split),
             'merges':[group],'merge_before':[split[:-1]],'split_groups':[]}
    merged=corrected_rows(split,{},[],merges=[group],toc_changed=True)
    edits,headings,merges,cuts=history_plan(merged,history,True)
    redo=corrected_rows(merged,edits,headings,merges=merges,cuts=cuts,toc_changed=True)
    assert [(r['id'],r['text'],r['metadata']['structure_path']) for r in redo]==[(r['id'],r['text'],r['metadata']['structure_path']) for r in split]
    edits,headings,merges,cuts=history_plan(redo,history,False)
    undo=corrected_rows(redo,edits,headings,merges=merges,cuts=cuts,toc_changed=True)
    assert len(undo)==len(source) and undo[0]['text']==source[0]['text']


def test_mcp_context_includes_split_parts_in_document_order(native_review,monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'src'))
    from src import rag_mcp_server as mcp
    runtime,original,_=native_review
    # Use the production ID and paragraph metadata shape for real Chroma filtering.
    runtime.collection.delete(ids=['ATT_1'])
    root='ABCDEFGH:p1:para0:part0'
    md={**original[0]['metadata'],'attachmentKey':'ABCDEFGH','para_index':0}
    runtime.collection.add(ids=[root,root+'!review1',root+'!review2'],documents=['Before.','Heading.','Later.'],
        metadatas=[md,md,md],embeddings=[[1.,0.]]*3)
    assert mcp.parse_id(root+'!review1')==('ABCDEFGH','pdf',1,0,0)
    assert mcp._context_chunks(runtime.collection,'invalid',1)['ids']==[]
    assert mcp._context_chunks(runtime.collection,root,0)['ids']==[]
    context=mcp._neighbor_context(runtime.collection,root,1)
    assert [r['id'] for r in context]==[root+'!review1',root+'!review2']
    monkeypatch.setattr(mcp,'_check_indexing_lock',lambda:(False,''))
    monkeypatch.setattr(mcp,'_col',lambda:runtime.collection)
    result=mcp.get_chunk_context(root+'!review1',1)
    assert result['context_text']=='Before.\n\nHeading.\n\nLater.'
    fake=Mock();fake.get.side_effect=[{'ids':[]},{'ids':['fallback']}]
    assert mcp._context_chunks(fake,root,1)['ids']==['fallback']
    fake.get.side_effect=None;fake.get.return_value={'ids':['html']}
    assert mcp._context_chunks(fake,'ABCDEFGH:html:para0:part0',1)['ids']==['html']
