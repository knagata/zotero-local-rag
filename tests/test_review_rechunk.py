from __future__ import annotations
import copy
import pytest
from citation_graph.review_rechunk import chunk_block,restore_block,structured_review,source_position
from src.document_structure import build_document_structure


def source(text=None):
    text=text or ('First argument explains a previous policy. '*40+'New section '+ 'The next argument concerns community power. '*50)
    return [{'id':f'ATT:p1:para0:part{n}','text':part,'metadata':{'attachmentKey':'ATT','page':1,'para_index':0,'part_index':n,'zone':'body'}} for n,part in enumerate(chunk_block(text))]


def heading(row,needle='New section'):
    return {'chunk_id':row['id'],'offset':row['text'].index(needle),'anchor_text':row['text'],'page':1,'title':needle,'level':1}


def test_verified_blocks_are_rechunked_after_structure_is_attached():
    rows=source();original=copy.deepcopy(rows);target=next(r for r in rows if 'New section' in r['text'])
    plan=structured_review(rows,{},[heading(target)])
    new=plan['rows'];starts=[r for r in new if r['metadata']['chapter']=='New section']
    assert starts[0]['text'].startswith('New section')
    assert all('previous policy' not in r['text'] for r in starts)
    assert plan['blocks'][0]['verified_overlap_chars']>0
    assert build_document_structure('ITEM',new)['diagnostics']['valid']
    assert rows==original
    assert plan==structured_review(rows,{},[heading(target)])
    raw,locations,_=restore_block(rows,{})
    assert raw.count('New section')==1
    assert source_position(locations[target['id']],target['text'].index('New section'))==raw.index('New section')


def test_non_overlapping_and_repetitive_short_source_is_kept():
    rows=source('稲 稲 稲 稲 稲')
    assert structured_review(rows,{},[])['rows'][0]['text']=='稲 稲 稲 稲 稲'
    assert chunk_block('見出し')==['見出し']
    assert source_position((10,5,1),2)==12
    assert source_position((10,5,1),5)==16


def test_unknown_or_changed_overlap_is_not_guessed():
    rows=source()
    wrong=copy.deepcopy(rows);wrong[1]['text']='No matching prefix '+wrong[1]['text']
    with pytest.raises(ValueError,match='重なり'):restore_block(wrong,{})
    wrong=copy.deepcopy(rows);wrong[-1]['text']+=' Extra tail'
    with pytest.raises(ValueError,match='再現'):restore_block(wrong,{})
    with pytest.raises(ValueError,match='校訂'):restore_block(rows,{rows[1]['id']:'Changed '+rows[1]['text']})
    edited={rows[0]['id']:rows[0]['text'].replace('First argument','Revised argument',1)}
    assert restore_block(rows,edited)[0].startswith('Revised argument')


def test_provenance_duplicate_boundaries_empty_text_and_mixed_zones_are_rejected():
    rows=source();target=next(r for r in rows if 'New section' in r['text'])
    wrong=copy.deepcopy(rows);wrong[1]['metadata']['part_index']=99
    with pytest.raises(ValueError,match='part'):structured_review(wrong,{},[])
    with pytest.raises(ValueError,match='重複'):structured_review(rows,{},[heading(target),heading(target)])
    with pytest.raises(ValueError,match='分類'):structured_review(rows,{},[],{rows[0]['id']:'footnote'})
    wrong=source('Short');wrong[0]['text']='  '
    with pytest.raises(ValueError,match='空'):structured_review(wrong,{},[])


def test_short_front_and_end_of_sections_stay_on_the_original_page():
    row=source('Front. New section '+ 'Body. '*100)[0]
    plan=structured_review([row],{},[heading(row)])
    assert plan['rows'][0]['text']=='Front.'
    assert plan['rows'][1]['text'].startswith('New section')
    assert all(r['metadata']['page']==1 for r in plan['rows'])


def test_invalid_rechunk_id_order_is_rejected():
    rows=[{'id':'Z','text':'Short one','metadata':{'page':1,'para_index':1}},
          {'id':'A','text':'Short two','metadata':{'page':1,'para_index':2}}]
    with pytest.raises(ValueError,match='識別子'):structured_review(rows,{},[])
