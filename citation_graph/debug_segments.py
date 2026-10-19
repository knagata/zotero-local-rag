"""Lossless, ordered chunk splits and guarded inverses for review history."""
from __future__ import annotations

import copy
from src.chunk_store import natural_chunk_key


def merge_rows(rows: list[dict], merges: list[list[str]] | None = None) -> list[dict]:
    result = copy.deepcopy(rows)
    used: set[str] = set()
    if merges and (len(merges) > 500 or sum(map(len, merges)) > 1000):
        raise ValueError('分割の復元は一度に500箇所・1000チャンクまでです。')
    for group in merges or []:
        indexed = {row['id']: n for n, row in enumerate(result)}
        if len(group) < 2 or len(set(group)) != len(group) or used.intersection(group) or any(key not in indexed for key in group):
            raise ValueError('復元対象の分割チャンクが現在の索引と一致しません。')
        start = indexed[group[0]]
        members = result[start:start+len(group)]
        if any(row['metadata'].get('review_parent') in group[1:] for row in result if row['id'] not in group):
            raise ValueError('後から再分割されています。新しい分割の履歴から順に戻してください。')
        if [row['id'] for row in members] != group or any(row['metadata'].get('review_parent') != group[0] for row in members[1:]):
            raise ValueError('連続した同じ分割元だけを復元できます。後から再分割された場合は新しい履歴から戻してください。')
        if len({(row['metadata'].get('attachmentKey'),row['metadata'].get('page')) for row in members}) != 1:
            raise ValueError('異なる添付・ページのチャンクは結合できません。')
        merged = copy.deepcopy(members[0])
        merged['text'] = ''.join(row['text'] for row in members)
        result[start:start+len(group)] = [merged]
        used.update(group)
    return result


def split_rows(rows: list[dict], edits: dict, toc: list[dict], cuts: dict | None = None) -> tuple[list[dict], list[dict], list[list[str]]]:
    """Keep the original ID for the prefix; allocate ordered children for suffixes."""
    result, resolved, groups = [], copy.deepcopy(toc), []
    ids = {row['id'] for row in rows}
    cuts = cuts or {}
    if len(cuts)>500 or any(key not in ids or not isinstance(values,list) or len(values)>500 for key,values in cuts.items()) or sum(map(len,cuts.values()))>500:
        raise ValueError('分割位置は現在のチャンクへ500箇所以内で指定してください。')
    for original in rows:
        row = copy.deepcopy(original)
        row['text'] = edits.get(row['id'], row['text'])
        entries = [entry for entry in resolved if entry.get('chunk_id') == row['id']]
        positions = cuts.get(row['id'],[])
        if any(isinstance(value,bool) or not isinstance(value,int) or not 0<value<len(row['text']) for value in positions):
            raise ValueError('分割位置は本文の途中を指定してください。')
        boundaries = sorted(set(positions) | {entry.get('offset',0) for entry in entries if entry.get('offset',0)})
        ends = [0, *boundaries, len(row['text'])]
        if boundaries and any(not row['text'][a:b].strip() for a,b in zip(ends, ends[1:])):
            raise ValueError('分割すると空のチャンクができます。見出しの先頭を指定してください。')
        keys = [row['id']]
        if boundaries:
            tails = [key[len(row['id']):] for key in ids if key.startswith(row['id'])]
            depth = 1 + max(len(tail)-len(tail.lstrip('!')) for tail in tails)
            keys += [row['id']+'!'*depth+f'review{n}' for n in range(1,len(boundaries)+1)]
        for n, (start,end) in enumerate(zip(ends, ends[1:])):
            part = copy.deepcopy(row)
            part['id'], part['text'] = keys[n], row['text'][start:end]
            if n:
                part['metadata']['review_parent'] = row['id']
            result.append(part)
        for entry in entries:
            entry['chunk_id'] = keys[ends.index(entry.get('offset',0))]
            entry.pop('offset',None)
            entry.pop('anchor_text',None)
        if boundaries:
            groups.append(keys)
    ordered = sorted(result,key=lambda row:natural_chunk_key(row['id']))
    if [row['id'] for row in ordered] != [row['id'] for row in result] or len({row['id'] for row in result}) != len(result):
        raise ValueError('分割後のチャンク順序・識別子を安全に確定できません。DBは変更しません。')
    return result, resolved, groups


def history_merges(rows: list[dict], historical: dict) -> list[list[str]]:
    ids = {row['id'] for row in rows}
    groups = []
    for group in historical.get('split_groups',[]):
        present = ids.intersection(group[1:])
        if present and group not in groups:
            if not set(group).issubset(ids):
                raise ValueError('分割履歴の一部が現在の索引にありません。履歴と本文を確認してください。')
            groups.append(group)
    merge_rows(rows,groups)
    return groups


def history_plan(rows: list[dict], historical: dict, before: bool) -> tuple[dict,list,list,dict]:
    """Rebase stored offsets onto the live topology, preserving immutable history."""
    groups = historical.get('split_groups',[])+historical.get('merges',[])
    merges = history_merges(rows,{'split_groups':groups})
    edits = dict(historical.get('originals',{}) if before else historical['edits'])
    toc = copy.deepcopy(historical.get('toc_before',[]) if before else historical['toc'])
    cuts = {} if before else copy.deepcopy(historical.get('cuts',{}))
    if not before:
        for group in historical.get('split_groups',[]):
            edits.setdefault(group[0],historical['originals'][group[0]])
    if before:
        locations = {}
        for members in historical.get('merge_before',[]):
            root, offset = members[0]['id'], 0
            edits[root] = ''.join(row['text'] for row in members)
            for row in members:
                locations[row['id']] = (root,offset)
                if offset:
                    cuts.setdefault(root,[]).append(offset)
                offset += len(row['text'])
        for entry in toc:
            if entry.get('chunk_id') in locations:
                root,offset = locations[entry['chunk_id']]
                entry['chunk_id'] = root
                if offset:
                    entry['offset'],entry['anchor_text'] = offset,edits[root]
    covered = {key for group in groups for key in group[1:]}
    for key in edits:
        live = {row['id'] for row in rows if row['metadata'].get('review_parent') == key}-covered
        known = set(historical.get('existing_children',{}).get(key,[]))-covered
        if live != known:
            raise ValueError('修正対象の分割状態が変わっています。新しい分割の履歴から戻してください。')
    return edits,toc,merges,cuts
