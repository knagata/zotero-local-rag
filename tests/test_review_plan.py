from __future__ import annotations
import copy
import pytest
from citation_graph.review_plan import build_review_plan,public_plan,restore_snapshot,validated_plan,layout_generation
from citation_graph.review_rechunk import digest
from citation_graph.debug_store import ReviewConflict
from tests.test_review_rechunk import source,heading


def test_preview_hash_and_saved_plan_guard():
    rows=source();row=next(r for r in rows if 'New section' in r['text'])
    draft={'edits':{},'toc':[heading(row)],'toc_changed':True,'rechunk':True}
    plan=build_review_plan(rows,draft)
    draft['plan_hash']=plan['plan_hash']
    assert validated_plan(rows,draft)==plan
    shown=public_plan(plan,rows)
    assert shown['before_count']==len(rows) and shown['verified_overlap_chars']>0
    assert shown['changes'] and shown['plan_hash']==plan['plan_hash']
    with pytest.raises(ReviewConflict):validated_plan(rows,{**draft,'plan_hash':'old'})
    with pytest.raises(ValueError):build_review_plan(rows,{**draft,'merges':[['A','B']]})
    plain=build_review_plan(rows,{'edits':{},'toc':[]})
    assert plain['version']==0
    assert layout_generation(rows)!=layout_generation(plan['rows'])


def test_history_snapshot_recreates_exact_layout_and_rejects_tampering():
    rows=source('Original')
    data={'version':'v','source_sha256':'sha','all_chunks':rows}
    historical={'rechunk_before':rows,'rechunk_after':source('After')}
    undo=restore_snapshot(data,historical,True,'v')
    redo=restore_snapshot(data,historical,False,'v')
    assert build_review_plan(rows,undo)['rows']==rows
    assert build_review_plan(rows,redo)['rows']==historical['rechunk_after']
    with pytest.raises(ReviewConflict):restore_snapshot(data,historical,True,'stale')
    for target in [[],[rows[0],rows[0]],[{**rows[0],'metadata':{'attachmentKey':'OTHER'}}]]:
        with pytest.raises(ReviewConflict):build_review_plan(rows,{**undo,'layout_target':target,'layout_hash':digest(target)})
    with pytest.raises(ReviewConflict):build_review_plan(rows,{**undo,'layout_hash':'tampered'})


def test_preview_is_bounded_but_hash_covers_the_complete_plan():
    rows=source('Short')*201
    rows=[{**copy.deepcopy(row),'id':str(n)} for n,row in enumerate(rows)]
    plan={'rows':rows,'version':1,'plan_hash':digest(rows),'blocks':[{'verified_overlap_chars':1}]*201}
    shown=public_plan(plan,[])
    assert len(shown['changes'])==len(shown['blocks'])==200
    assert shown['changed_count']==shown['block_count']==201 and shown['after_count']==201


def test_structured_native_apply_and_exact_layout_restore(native_review):
    from citation_graph.debug_apply import apply_with_runtime
    from tests.test_debug_split_apply import document,lexical_rows
    runtime,original,lexical=native_review
    runtime.collection.delete(ids=['ATT_1']);runtime.lexical.delete_by_chunk_ids(['ATT_1'])
    rows=source()
    for row in rows:row['metadata']['itemKey']='ITEM'
    runtime.collection.add(ids=[r['id'] for r in rows],documents=[r['text'] for r in rows],metadatas=[r['metadata'] for r in rows],embeddings=[[1.,0.]]*len(rows))
    runtime.lexical.upsert_chunks([r['id'] for r in rows],[r['text'] for r in rows],[r['metadata'] for r in rows])
    target=next(r for r in rows if 'New section' in r['text'])
    draft={'revision':1,'edits':{},'toc':[heading(target)],'toc_changed':True,'rechunk':True}
    draft['plan_hash']=build_review_plan(rows,draft)['plan_hash']
    before=runtime.rows('ITEM')
    apply_with_runtime(runtime,document(before),draft,runtime.root)
    live=runtime.rows('ITEM')
    assert {r['id']:r['text'] for r in live}==lexical_rows(lexical)
    changed=document(live)['all_chunks']
    assert next(r for r in changed if r['metadata'].get('chapter')=='New section')['text'].startswith('New section')
    restore={'revision':2,'edits':{},'toc':[],'toc_changed':True,'layout_target':rows,'layout_hash':digest(rows)}
    apply_with_runtime(runtime,document(live),restore,runtime.root)
    assert [(r['id'],r['text']) for r in runtime.rows('ITEM')]==[(r['id'],r['text']) for r in before]
    assert all('review_rechunk_version' not in r['metadata'] and 'review_layout_hash' not in r['metadata'] for r in runtime.rows('ITEM'))
    assert layout_generation(document(runtime.rows('ITEM'))['all_chunks'])==layout_generation(rows)
    assert {r['id']:r['text'] for r in runtime.rows('ITEM')}==lexical_rows(lexical)


@pytest.mark.parametrize('phase',['structure','verify_source'])
def test_structured_native_fault_rolls_back_rows_and_vectors(native_review,monkeypatch,phase):
    from citation_graph.debug_apply import apply_with_runtime
    from tests.test_debug_split_apply import document,lexical_rows
    from unittest.mock import Mock
    runtime,_,lexical=native_review
    rows=document(runtime.rows('ITEM'))['all_chunks']
    draft={'revision':1,'edits':{},'toc':[{'chunk_id':'ATT_1','page':1,'level':1,'title':'Inaugurations','offset':8,'anchor_text':rows[0]['text']}],'toc_changed':True,'rechunk':True}
    draft['plan_hash']=build_review_plan(rows,draft)['plan_hash']
    before=runtime.rows('ITEM');vectors=runtime.collection.get(include=['embeddings'])
    saved={key:v.tolist() for key,v in zip(vectors['ids'],vectors['embeddings'])}
    monkeypatch.setattr(runtime,phase,Mock(side_effect=RuntimeError('injected')))
    with pytest.raises(RuntimeError):apply_with_runtime(runtime,document(before),draft,runtime.root)
    assert runtime.rows('ITEM')==before
    after=runtime.collection.get(include=['embeddings'])
    assert {key:v.tolist() for key,v in zip(after['ids'],after['embeddings'])}==saved
    assert {r['id']:r['text'] for r in before}==lexical_rows(lexical)


# Shared fixture always creates its Chroma/FTS/journal under pytest's temporary directory.
from tests.test_debug_split_apply import native_review  # noqa: F401,E402


def test_body_only_edits_do_not_require_a_structural_plan_hash():
    rows=source('Body')
    assert validated_plan(rows,{'edits':{},'toc':[],'rechunk':True})['version']==0


def test_unsupported_source_saves_the_draft_with_a_reason(tmp_path):
    from citation_graph.debug_reader import DebugReader
    reader=DebugReader(tmp_path)
    doc={'version':'v','all_chunks':[],'attachment_key':'ATT','source_sha256':'sha','total_pages':2}
    saved=reader.save(doc,0,'v',{},[{'page':2,'level':1,'title':'Missing'}],'tester',True,rechunk=True)
    assert saved['revision']==1 and '未抽出' in saved['rechunk_error']
    assert saved['toc'][0]['title']=='Missing'
    with pytest.raises(ValueError,match='未抽出'):build_review_plan([],saved)


def test_explicit_zones_follow_source_groups_when_an_old_id_is_retired():
    from citation_graph.review_plan import validate_explicit_zones
    plan={'blocks':[{'source_ids':['OLD_A','OLD_B']}]}
    rows=[{'id':'OLD_A','metadata':{'zone':'footnote','review_source_group':'OLD_A'}}]
    validate_explicit_zones(rows,{'OLD_B':'footnote'},plan)
    with pytest.raises(ValueError,match='分類'):validate_explicit_zones(rows,{'OLD_B':'body'},plan)
    with pytest.raises(ValueError):validate_explicit_zones([],{'OLD_B':'body'},plan)
    validate_explicit_zones(rows,{'OLD_A':'footnote'},{'blocks':[]})
    with pytest.raises(ValueError):validate_explicit_zones(rows,{'MISSING':'body'},{'blocks':[]})
