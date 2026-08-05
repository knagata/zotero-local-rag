# src/structure_metadata_sync.py
"""Keep chunk metadata in step with the canonical structure after a rebuild.

``replace_document_structure`` writes new nodes to the database, and node ids
are derived from content, so a rebuild renames them. Nothing then updated the
copy of ``node_id`` carried in each chunk's Chroma metadata -- only the ingest
path stamped that, via ``attach_structure_metadata``. Rebuilding the structure
therefore left the two stores disagreeing.

That disagreement is not cosmetic, because retrieval reads the metadata copy:
``rag_mcp_server`` filters candidate chunks with ``{"node_id": {"$in": ...}}``
using ids taken from the database. After a rebuild those ids match nothing, so
the leaf path -- the highest-weighted of the three fused routes -- silently
returns an empty set and search quietly degrades to the item and direct routes.
On 2026-07-28, 213,748 chunks (44.7%) pointed at nodes that no longer existed,
and 51 items had no live leaf at all.

The database is the authority: ``document_node_chunks`` records which chunk
belongs to which leaf. This module copies that mapping, and the policies that
travel with it, back onto the chunks.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence

#: Metadata the retrieval layer reads from the chunk rather than the database.
#: Each must be re-stamped together: a chunk carrying a fresh node_id beside a
#: stale retrieval_policy would be routed correctly and then filtered out.
SYNCED_KEYS = ("node_id", "zone", "summary_policy", "retrieval_policy", "citation_policy")


def desired_chunk_metadata(
    nodes: Sequence[Mapping[str, Any]], chunk_ids_by_node: Mapping[str, Sequence[str]],
) -> Dict[str, Dict[str, Any]]:
    """Map each chunk id to the metadata its current leaf implies."""
    by_id = {str(node.get("node_id")): node for node in nodes}
    desired: Dict[str, Dict[str, Any]] = {}
    for node_id, chunk_ids in chunk_ids_by_node.items():
        node = by_id.get(str(node_id))
        if node is None:
            continue
        payload = {
            "node_id": str(node_id),
            "zone": str(node.get("zone") or "body"),
            "summary_policy": str(node.get("summary_policy") or "include"),
            "retrieval_policy": str(node.get("retrieval_policy") or "normal"),
            "citation_policy": str(node.get("citation_policy") or "none"),
        }
        for chunk_id in chunk_ids:
            desired[str(chunk_id)] = dict(payload)
    return desired


def stale_chunk_updates(
    current: Mapping[str, Mapping[str, Any]], desired: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Chunks whose stored metadata disagrees with the structure, and their fix.

    Only the disagreeing chunks are returned. Rewriting every chunk on every
    rebuild would make the repair itself the most expensive part of a rebuild,
    and would hide how much had actually drifted.
    """
    updates: Dict[str, Dict[str, Any]] = {}
    for chunk_id, wanted in desired.items():
        stored = current.get(chunk_id)
        if stored is None:
            continue
        # Callers may extend the canonical routing payload with refreshed
        # source fields (structure_path/chapter/etc.). Comparing only the five
        # base routing keys silently discarded those metadata-only repairs.
        if any(
            str(stored.get(key) or "") != str(value or "")
            for key, value in wanted.items()
        ):
            merged = dict(stored)
            merged.update(wanted)
            updates[chunk_id] = merged
    return updates


def orphaned_chunk_ids(
    current: Iterable[str], desired: Mapping[str, Mapping[str, Any]],
) -> List[str]:
    """Indexed chunks the current structure does not account for.

    A chunk the structure no longer claims keeps whatever leaf it was last
    stamped with, so it stays routable through a node that may since have been
    reused for different text. Reported rather than silently cleared, because
    the cause is upstream -- extraction produced chunks the structure builder
    did not see -- and blanking the metadata would hide it.
    """
    return sorted(set(str(value) for value in current) - set(desired))


__all__ = [
    "SYNCED_KEYS", "desired_chunk_metadata", "orphaned_chunk_ids", "stale_chunk_updates",
]
