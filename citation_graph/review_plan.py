"""One deterministic preview/apply plan, with immutable layout restore snapshots."""
from __future__ import annotations
from citation_graph.debug_changes import corrected_rows
from citation_graph.debug_store import ReviewConflict
from citation_graph.review_rechunk import structured_review,digest


def layout_generation(rows: list[dict]) -> str:
    return digest([{'id':row['id'],'layout':row['metadata'].get('review_layout_hash'),
                    'parent':row['metadata'].get('review_parent')} for row in rows])


def build_review_plan(rows: list[dict], draft: dict) -> dict:
    if 'layout_target' in draft:
        target = draft['layout_target']
        if (digest(target) != draft.get('layout_hash') or not target
                or len({r['id'] for r in target}) != len(target)
                or {r['metadata'].get('attachmentKey') for r in target} != {r['metadata'].get('attachmentKey') for r in rows}):
            raise ReviewConflict('履歴の復元レイアウトを検証できません。DBは変更しません。')
        return {'rows':target,'plan_hash':digest(target),'version':1,'blocks':[],'restoring':True}
    if draft.get('rechunk') and (draft.get('toc_changed') or draft['toc']):
        if draft.get('merges') or draft.get('cuts'):
            raise ValueError('分割履歴の復元と構造からの再編は別々に反映してください。')
        if any(not entry.get('chunk_id') for entry in draft['toc']):
            raise ValueError('未抽出ページの開始位置は下書きに保存できますが、再編前に位置を確認してください。')
        return structured_review(rows,draft['edits'],draft['toc'],draft.get('zones'))
    target = corrected_rows(rows,draft['edits'],draft['toc'],toc_changed=draft.get('toc_changed',False),
                            zones=draft.get('zones'),merges=draft.get('merges'),cuts=draft.get('cuts'))
    return {'rows':target,'plan_hash':digest(target),'version':0,'blocks':[]}


def public_plan(plan: dict, before: list[dict]) -> dict:
    original = {row['id']:row for row in before}
    after = {row['id']:row for row in plan['rows']}
    changed = [row for row in plan['rows'] if row['id'] not in original or row != original[row['id']]]
    return {'version':plan['version'],'plan_hash':plan['plan_hash'],'before_count':len(before),'after_count':len(after),
            'removed_count':len(set(original)-set(after)),'blocks':plan['blocks'][:200],'block_count':len(plan['blocks']),
            'verified_overlap_chars':sum(b['verified_overlap_chars'] for b in plan['blocks']),
            'changes':[{'id':row['id'],'page':row['metadata'].get('page'),
                        'before':original.get(row['id'],{}).get('text'),'after':row['text'],
                        'path':row['metadata'].get('structure_path')} for row in changed[:200]],'changed_count':len(changed),
            'notice':'章節に基づく再編です。参考文献・注・図版の分類範囲も別途確認してください。'}


def restore_snapshot(data: dict,historical: dict,before: bool,version: str) -> dict:
    if version != data['version']:
        raise ReviewConflict('現在の索引が変更されています。再読込してください。')
    target = historical['rechunk_before' if before else 'rechunk_after']
    return {'version':version,'source_sha256':data['source_sha256'],
            'edits':{},'toc':[],'toc_changed':True,'zones':{},'rechunk':False,
            'layout_target':target,'layout_hash':digest(target),
            'rechunk_before':data['all_chunks'],'rechunk_after':target,
            'layout_generation':layout_generation(data['all_chunks'])}


def validated_plan(rows: list[dict],draft: dict) -> dict:
    plan = build_review_plan(rows,draft)
    if draft.get('rechunk') and plan['version'] and plan['plan_hash'] != draft.get('plan_hash'):
        raise ReviewConflict('再編プレビューが保存時と一致しません。再確認して下書きを保存してください。')
    return plan


def validate_explicit_zones(rows: list[dict],zones: dict,plan: dict) -> None:
    roots = {key:block['source_ids'][0] for block in plan['blocks'] for key in block['source_ids']}
    indexed = {row['id']:row for row in rows}
    for key,zone in zones.items():
        targets = ([row for row in rows if row['metadata'].get('review_source_group')==roots[key]]
                   if key in roots else [indexed[key]] if key in indexed else [])
        if not targets or any(row['metadata'].get('zone') != zone for row in targets):
            raise ValueError('指定した分類が再編後の文書構造と一致しません。分類範囲と見出しを確認してください。')
