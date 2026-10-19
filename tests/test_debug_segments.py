from __future__ import annotations

import copy
import pytest
from citation_graph.debug_segments import split_rows,merge_rows,history_merges
from citation_graph.debug_changes import corrected_rows
from citation_graph.debug_store import validate_draft
from src.chunk_store import natural_chunk_key
from src.document_structure import build_document_structure


def rows():
    return [{'id':'ATT_1','text':'Before.\nInaugurations\nNew text.\nNext\nLast.','metadata':{'attachmentKey':'ATT','page':1,'zone':'body'}},
            {'id':'ATT_2','text':'Following','metadata':{'attachmentKey':'ATT','page':2,'zone':'body'}}]


def toc(source,offset,title='Inaugurations',level=1):
    return {'chunk_id':source['id'],'page':source['metadata']['page'],'offset':offset,'anchor_text':source['text'],'title':title,'level':level}


def test_lossless_split_separates_the_old_section_and_allows_multiple_headings():
    source=rows();original=copy.deepcopy(source);text=source[0]['text']
    headings=[toc(source[0],text.index('Inaugurations')),toc(source[0],text.index('Next'),'Next')]
    validate_draft(source,{},headings)
    split,resolved,groups=split_rows(source,{},headings)
    assert ''.join(row['text'] for row in split[:-1])==text
    assert [row['text'] for row in split[:-1]]==['Before.\n','Inaugurations\nNew text.\n','Next\nLast.']
    assert len(set(row['id'] for row in split))==4
    assert split==sorted(split,key=lambda row:natural_chunk_key(row['id']))
    assert groups==[[row['id'] for row in split[:-1]]]
    assert all('offset' not in entry and 'anchor_text' not in entry for entry in resolved)
    changed=corrected_rows(source,{},headings)
    assert changed[0]['metadata']['chapter']==''
    assert changed[1]['metadata']['chapter']=='Inaugurations'
    assert changed[2]['metadata']['chapter']=='Next'
    assert source==original
    assert merge_rows(split,groups)==source
    assert build_document_structure('ITEM',changed)['diagnostics']['valid']
    # Split the retained prefix again: its new child must precede the earlier child.
    second=[toc(split[0],3,'Earlier')]
    again,_,_=split_rows(split,{},second)
    assert again==sorted(again,key=lambda row:natural_chunk_key(row['id']))
    assert ''.join(row['text'] for row in again[:-1])==text
    latest=[['ATT_1',again[1]['id']]]
    assert merge_rows(again,latest)==split
    with pytest.raises(ValueError,match='再分割'):merge_rows(again,groups)


@pytest.mark.parametrize('offset',[-1,True,1.5,'2',999])
def test_invalid_offsets_are_rejected(offset):
    source=rows()
    with pytest.raises(ValueError,match='位置'):validate_draft(source,{},[toc(source[0],offset)])


def test_changed_text_duplicate_and_backward_offsets_are_rejected():
    source=rows();entry=toc(source[0],8)
    for entries in [[entry,entry],[toc(source[0],15),entry]]:
        with pytest.raises(ValueError,match='本文順'):validate_draft(source,{},entries)
    with pytest.raises(ValueError,match='再指定'):validate_draft(source,{'ATT_1':'Changed. '+source[0]['text']},[entry])
    with pytest.raises(ValueError,match='再指定'):validate_draft(source,{},[{**entry,'chunk_id':'MISSING'}])
    with pytest.raises(ValueError,match='空'):corrected_rows(source,{},[toc(source[0],7),toc(source[0],8,'Second')])
    unusual=[{'id':'A','text':'ab','metadata':{}},{'id':'A!','text':'cd','metadata':{}}]
    with pytest.raises(ValueError,match='順序'):split_rows(unusual,{},[toc({**unusual[0],'metadata':{'page':1}},1)])


def test_offsets_use_unicode_characters_and_edited_text():
    source=rows();source[0]['text']='𠮷😀\n見出し\n本文'
    entry=toc(source[0],3,'見出し')
    changed=corrected_rows(source,{},[entry])
    assert changed[0]['text']=='𠮷😀\n' and changed[1]['text']=='見出し\n本文'
    entry['anchor_text']='前文\n改訂見出し\n本文';entry['offset']=3
    changed=corrected_rows(source,{'ATT_1':entry['anchor_text']},[entry],zones={'ATT_1':'footnote'})
    assert [row['text'] for row in changed[:-1]]==['前文\n','改訂見出し\n本文']
    assert all(row['metadata']['zone']=='footnote' for row in changed[:-1])


def test_merge_guards_and_history_resolution():
    source=rows();split,_,groups=split_rows(source,{},[toc(source[0],8)])
    assert history_merges(source,{'split_groups':groups})==[]
    assert history_merges(split,{'split_groups':groups})==groups
    nested=copy.deepcopy(split[1]);nested['id']+='!review1';nested['metadata']['review_parent']=split[1]['id']
    with pytest.raises(ValueError,match='再分割'):merge_rows(split+[nested],groups)
    with pytest.raises(ValueError,match='一部'):history_merges(split[1:],{'split_groups':groups})
    for group in [[],['ATT_1'],['ATT_1','MISSING'],['ATT_1','ATT_1']]:
        with pytest.raises(ValueError):merge_rows(split,[group])
    with pytest.raises(ValueError):merge_rows(split,[groups[0],groups[0]])
    with pytest.raises(ValueError):merge_rows(split,[['ATT_1','ATT_2']])
    wrong=copy.deepcopy(split);wrong[1]['metadata']['page']=3
    with pytest.raises(ValueError,match='ページ'):merge_rows(wrong,groups)
    with pytest.raises(ValueError,match='500'):merge_rows(source,[['ATT_1','ATT_2']]*501)
    assert corrected_rows(split,{},[],toc_changed=True,merges=groups)[0]['text']==source[0]['text']


@pytest.mark.parametrize('cuts',[{'MISSING':[1]},{'ATT_1':True},{'ATT_1':[1]*501},{'ATT_1':[True]},{'ATT_1':[0]},{'ATT_1':[999]}])
def test_explicit_history_cuts_are_strictly_bounded(cuts):
    with pytest.raises(ValueError):split_rows(rows(),{},[],cuts)


def test_history_deduplicates_groups_and_refuses_text_restores_across_new_splits():
    from citation_graph.debug_segments import history_plan
    source=rows();split,_,groups=split_rows(source,{},[toc(source[0],8)])
    assert history_merges(split,{'split_groups':groups+groups})==groups
    old={'edits':{'ATT_1':'Old edit'},'originals':{'ATT_1':source[0]['text']},'toc':[],'toc_before':[]}
    for before in [True,False]:
        with pytest.raises(ValueError,match='分割状態'):history_plan(split,old,before)
