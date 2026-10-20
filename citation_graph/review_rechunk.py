"""Replay verified PDF paragraph parts, then feed manual structure to the shared chunker."""
from __future__ import annotations

import copy
import hashlib
import json
from itertools import groupby
from src.text_utils import (
    split_long_paragraph, merge_short_chunk_records, joiner_for_text, overlap_for_text,
    is_no_space_language_document, MAX_CHARS, TARGET_CHARS, MAX_CHARS_CJK,
    TARGET_CHARS_CJK, MIN_CHUNK_CHARS, MIN_CHUNK_CHARS_NO_SPACE,
)


def limits(text: str) -> tuple[int,int,int]:
    return ((MAX_CHARS_CJK,TARGET_CHARS_CJK,MIN_CHUNK_CHARS_NO_SPACE)
            if is_no_space_language_document(text) else (MAX_CHARS,TARGET_CHARS,MIN_CHUNK_CHARS))


def chunk_block(text: str) -> list[str]:
    maximum,target,minimum = limits(text)
    parts = split_long_paragraph(text,max_chars=maximum,target_chars=target)
    return [row[1] for row in merge_short_chunk_records(
        [(str(n),part,{}) for n,part in enumerate(parts)],min_chars=minimum,
        max_chars=maximum,hard_min_chars=0)]


def restore_block(rows: list[dict], edits: dict) -> tuple[str,dict,list[int]]:
    """Only remove overlaps if exact replay reproduces ALL stored paragraph parts."""
    first = rows[0]
    original = first['text']
    overlaps = [0]
    for row in rows[1:]:
        size = overlap_for_text(row['text'])
        candidates = [n for n in range(max(1,size-2),size+1)
                      if original[-n:] == row['text'][:n]]
        if not candidates:
            raise ValueError('元段落の重なりを確認できません。単純連結せず、紙面と抽出結果を確認してください。')
        overlap = max(candidates)
        original += joiner_for_text(original)+row['text'][overlap:]
        overlaps.append(overlap)
    if len(rows)>1 and chunk_block(original) != [row['text'] for row in rows]:
        raise ValueError('元のチャンク生成を再現できません。校訂を保持して自動再編を保留します。')
    text = edits.get(first['id'],first['text'])
    locations = {first['id']:(0,0,0)}
    for row,overlap in zip(rows[1:],overlaps[1:]):
        current = edits.get(row['id'],row['text'])
        if text[-overlap:] != current[:overlap]:
            raise ValueError('重なった本文の校訂が一致しません。片方の校訂を黙って捨てず、確認してください。')
        separator = joiner_for_text(text)
        locations[row['id']] = (len(text)-overlap,overlap,len(separator))
        text += separator+current[overlap:]
    return text,locations,overlaps


def source_position(location: tuple[int,int,int], offset: int) -> int:
    start,overlap,separator = location
    return start+offset+(separator if offset>=overlap and overlap else 0)


def digest(rows: list[dict]) -> str:
    return hashlib.sha256(json.dumps(rows,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def structured_review(rows: list[dict], edits: dict, toc: list[dict], zones: dict | None = None) -> dict:
    zones = zones or {}
    result, blocks, previous_path = [], [], []
    heading_map = {}
    for entry in toc:
        heading_map.setdefault(entry['chunk_id'],[]).append(entry)
    def group_key(row):
        md = row['metadata']
        # Missing source paragraph provenance is never inferred from repeated text.
        return (md.get('attachmentKey'),md.get('page'),md.get('para_index',row['id']),
                md.get('structure_path',''),md.get('zone','body'))
    for _, values in groupby(rows,key=group_key):
        group = list(values)
        if len(group)>1:
            parts = [row['metadata'].get('part_index') for row in group]
            if any(type(part) is not int for part in parts) or parts != list(range(parts[0],parts[0]+len(parts))):
                raise ValueError('元段落のpart順序が不明です。自動再編を保留します。')
        text,locations,overlaps = restore_block(group,edits)
        events = []
        for row in group:
            for heading in heading_map.get(row['id'],[]):
                events.append((source_position(locations[row['id']],heading.get('offset',0)),heading))
        if [position for position,_ in events] != sorted({position for position,_ in events}):
            raise ValueError('見出しの元本文上の位置が重複・逆順です。開始位置を再指定してください。')
        supplied_zones = {zones.get(row['id'],row['metadata'].get('zone','body')) for row in group}
        if len(supplied_zones)!=1:
            raise ValueError('同じ元段落の分類が混在しています。分類範囲を確認してください。')
        zone = supplied_zones.pop()
        boundaries = [0,*[position for position,_ in events if position>0],len(text)]
        event_map = dict(events)
        layout_hash = digest([{'text':text,'events':[(position,entry['title'],entry['level']) for position,entry in events]}])
        generated = []
        for start,end in zip(boundaries,boundaries[1:]):
            if start in event_map:
                heading = event_map[start]
                previous_path = previous_path[:heading['level']-1]+[heading['title'].strip()]
            if not text[start:end].strip():
                raise ValueError('節の境界に空の本文があります。開始位置を確認してください。')
            for part in chunk_block(text[start:end]):
                md = copy.deepcopy(group[0]['metadata'])
                md.update(structure_path=json.dumps(previous_path,ensure_ascii=False),
                          structure_roles=json.dumps(['chapter']*len(previous_path)),
                          chapter=previous_path[0] if previous_path else '',
                          section=previous_path[-1] if len(previous_path)>1 else '',zone=zone,
                          review_rechunk_version=1,review_layout_hash=layout_hash,review_source_group=group[0]['id'],
                          review_source_hash=hashlib.sha256(text.encode()).hexdigest())
                generated.append({'id':'','text':part,'metadata':md})
        for n,row in enumerate(generated):
            row['id'] = group[n]['id'] if n<len(group) else group[-1]['id']+f'!review{n-len(group)+1}'
            row['metadata']['part_index'] = n
        result.extend(generated)
        blocks.append({'page':group[0]['metadata'].get('page'),'source_ids':[row['id'] for row in group],
                       'source_sha256':hashlib.sha256(text.encode()).hexdigest(),
                       'verified_overlap_chars':sum(overlaps),'before':len(group),'after':len(generated),
                       'headings':[entry['title'] for _,entry in events]})
    from src.chunk_store import natural_chunk_key
    if result != sorted(result,key=lambda row:natural_chunk_key(row['id'])) or len({row['id'] for row in result})!=len(result):
        raise ValueError('再編後の識別子と本文順を確定できません。')
    return {'rows':result,'blocks':blocks,'plan_hash':digest(result),'version':1}
