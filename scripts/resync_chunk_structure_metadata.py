#!/usr/bin/env python3
"""Re-stamp chunk metadata from the canonical structure.

Run after any structure rebuild. `replace_document_structure` renames nodes
(their ids are derived from content) and nothing updated the copy of `node_id`
carried in Chroma, but retrieval filters on that copy -- so the leaf route, the
highest-weighted of the three fused routes, matched nothing and search degraded
without a word. The database is the authority here: `document_node_chunks`
records which chunk belongs to which leaf.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_utils import load_dotenv_native  # noqa: E402

load_dotenv_native(ROOT)

from src.chunk_store import active_collection_name  # noqa: E402
from src.db_relations import get_db_connection  # noqa: E402
from src.structure_metadata_sync import (  # noqa: E402
    desired_chunk_metadata, orphaned_chunk_ids, stale_chunk_updates,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item", action="append", help="Limit to item keys; repeatable.")
    parser.add_argument("--apply", action="store_true", help="Write. Without it, report only.")
    parser.add_argument("--collection", default=None)
    args = parser.parse_args()

    import chromadb

    name = args.collection or active_collection_name()
    collection = chromadb.PersistentClient(path=str(ROOT / "data" / "chroma")).get_collection(name)
    connection = get_db_connection()

    where = ""
    params: list[str] = []
    if args.item:
        where = f" WHERE n.item_key IN ({','.join('?' * len(args.item))})"
        params = list(args.item)

    nodes = [
        dict(row) for row in connection.execute(
            "SELECT node_id, item_key, zone, summary_policy, retrieval_policy, citation_policy "
            "FROM document_nodes n" + where, params)
    ]
    mapping: dict[str, list[str]] = defaultdict(list)
    for node_id, chunk_id in connection.execute(
        "SELECT c.node_id, c.chunk_id FROM document_node_chunks c "
        "JOIN document_nodes n ON n.node_id = c.node_id" + where, params
    ):
        mapping[str(node_id)].append(str(chunk_id))

    desired = desired_chunk_metadata(nodes, mapping)
    if not desired:
        print(json.dumps({"chunks_considered": 0}, ensure_ascii=False))
        return 0

    ids = list(desired)
    current: dict[str, dict] = {}
    for start in range(0, len(ids), 500):
        batch = ids[start:start + 500]
        got = collection.get(ids=batch, include=["metadatas"])
        for chunk_id, metadata in zip(got.get("ids") or [], got.get("metadatas") or []):
            current[str(chunk_id)] = dict(metadata or {})

    updates = stale_chunk_updates(current, desired)
    report = {
        "collection": name,
        "chunks_considered": len(desired),
        "present_in_index": len(current),
        "stale": len(updates),
        "structure_claims_not_indexed": len(desired) - len(current),
        "applied": False,
    }
    if args.apply and updates:
        keys = list(updates)
        for start in range(0, len(keys), 500):
            batch = keys[start:start + 500]
            collection.update(ids=batch, metadatas=[updates[key] for key in batch])
        report["applied"] = True
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
