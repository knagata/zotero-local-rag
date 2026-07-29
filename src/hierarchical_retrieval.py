"""Deterministic policy and fusion primitives for hierarchical retrieval V3."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence


RRF_K = 60
PATH_WEIGHTS = {"leaf": 1.7, "same_item": 1.2, "direct": 1.0}
_EXPLICIT_NOTE_INTENT = re.compile(
    r"(?:脚注|巻末注|注釈|出典|引用元|典拠|参考文献|文献リスト|footnotes?|endnotes?|notes?|citations?|sources?|references?)",
    re.IGNORECASE,
)


def partition_leaf_ids(values: Sequence[str], *, batch_size: int = 100) -> List[List[str]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    unique = list(dict.fromkeys(str(value) for value in values if value))
    return [unique[start:start + batch_size] for start in range(0, len(unique), batch_size)]


def explicit_note_intent(queries: str | Sequence[str]) -> bool:
    values = [queries] if isinstance(queries, str) else list(queries)
    return any(_EXPLICIT_NOTE_INTENT.search(str(value or "")) for value in values)


def retrieval_policy_allowed(
    metadata: Dict[str, Any], *, allow_explicit: bool, include_corrupted: bool = False,
) -> bool:
    """Whether a chunk may be returned for this query.

    ``include_corrupted`` opens up ``zone="corrupted"`` chunks -- OCR noise and
    pages that resisted repair, retained rather than discarded (note 79, U3) so
    they remain reachable on purpose. They stay out of ordinary results because
    their text is known to be unreliable, not because it is uninteresting.
    """
    if include_corrupted and str(metadata.get("zone") or "") == "corrupted":
        return True
    policy = str(metadata.get("retrieval_policy") or "normal")
    if policy == "exclude":
        return False
    if policy == "explicit_only":
        return allow_explicit
    return policy == "normal"


def fuse_retrieval_paths(
    rankings: Sequence[tuple[str, Sequence[Dict[str, Any]]]],
    *, routed_nodes_by_chunk: Dict[str, Sequence[str]] | None = None,
    rrf_k: int = RRF_K,
) -> List[Dict[str, Any]]:
    """Fuse ranked source hits and retain every contributing retrieval path."""
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")
    fused: Dict[str, Dict[str, Any]] = {}
    scores: Dict[str, float] = {}
    paths: Dict[str, List[str]] = {}
    for path, hits in rankings:
        if path not in PATH_WEIGHTS:
            raise ValueError(f"unknown retrieval path: {path}")
        for rank, hit in enumerate(hits, start=1):
            chunk_id = str(hit.get("id") or hit.get("chunk_id") or "")
            if not chunk_id:
                continue
            fused.setdefault(chunk_id, dict(hit))
            scores[chunk_id] = scores.get(chunk_id, 0.0) + PATH_WEIGHTS[path] / (rrf_k + rank)
            if path not in paths.setdefault(chunk_id, []):
                paths[chunk_id].append(path)
    output = []
    node_map = routed_nodes_by_chunk or {}
    for chunk_id in sorted(fused, key=lambda value: (-scores[value], value)):
        hit = dict(fused[chunk_id])
        hit["rrf_score"] = scores[chunk_id]
        hit["retrieval_paths"] = paths[chunk_id]
        hit["routed_by_node_ids"] = list(dict.fromkeys(str(value) for value in node_map.get(chunk_id, []) if value))
        output.append(hit)
    return output


__all__ = [
    "PATH_WEIGHTS", "RRF_K", "explicit_note_intent", "fuse_retrieval_paths",
    "partition_leaf_ids", "retrieval_policy_allowed",
]
