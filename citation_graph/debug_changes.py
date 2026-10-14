"""Pure, reviewable transformation of one attachment's canonical chunks."""
from __future__ import annotations

import copy
import json

from citation_graph.debug_store import validate_draft
from citation_graph.debug_annotations import resolve_toc, validate_zones


def corrected_rows(rows: list[dict], edits: dict, toc: list[dict], *, toc_changed: bool = False, zones: dict | None = None) -> list[dict]:
    toc = resolve_toc(rows, toc)
    zones = zones or {}
    validate_draft(rows, edits, toc)
    validate_zones(rows, zones)
    result = copy.deepcopy(rows)
    boundaries = {entry['chunk_id']: entry for entry in toc}
    path: list[str] = []
    for row in result:
        if row['id'] in edits:
            row['text'] = edits[row['id']]
        if row['id'] in zones:
            row['metadata']['zone'] = zones[row['id']]
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
