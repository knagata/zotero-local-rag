"""Pure, reviewable transformation of one attachment's canonical chunks."""
from __future__ import annotations

import copy
import json

from citation_graph.debug_store import validate_draft


def corrected_rows(rows: list[dict], edits: dict, toc: list[dict], *, toc_changed: bool = False) -> list[dict]:
    validate_draft(rows, edits, toc)
    result = copy.deepcopy(rows)
    boundaries = {entry['chunk_id']: entry for entry in toc}
    path: list[str] = []
    for row in result:
        if row['id'] in edits:
            row['text'] = edits[row['id']]
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
