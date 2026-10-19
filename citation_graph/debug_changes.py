"""Pure, reviewable transformation of one attachment's canonical chunks."""
from __future__ import annotations

import json

from citation_graph.debug_store import validate_draft
from citation_graph.debug_annotations import resolve_toc, validate_zones
from citation_graph.debug_segments import merge_rows, split_rows


def corrected_rows(rows: list[dict], edits: dict, toc: list[dict], *, toc_changed: bool = False, zones: dict | None = None, merges: list | None = None, cuts: dict | None = None) -> list[dict]:
    rows = merge_rows(rows, merges)
    toc = resolve_toc(rows, toc)
    zones = zones or {}
    validate_draft(rows, edits, toc)
    validate_zones(rows, zones)
    result, toc, _ = split_rows(rows, edits, toc, cuts)
    original_ids = {row['id'] for row in rows}
    boundaries = {entry['chunk_id']: entry for entry in toc}
    path: list[str] = []
    for row in result:
        origin = row['id'] if row['id'] in zones else (row['metadata'].get('review_parent') if row['id'] not in original_ids else None)
        if origin in zones:
            row['metadata']['zone'] = zones[origin]
        heading = boundaries.get(row['id'])
        if heading:
            path = path[:heading['level']-1] + [heading['title'].strip()]
        if toc or toc_changed:
            md = row['metadata']
            md['structure_path'] = json.dumps(path, ensure_ascii=False)
            md['structure_roles'] = json.dumps(['chapter'] * len(path))
            md['chapter'] = path[0] if path else ''
            md['section'] = path[-1] if len(path)>1 else ''
    return result
