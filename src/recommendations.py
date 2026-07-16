"""Citation-structure and semantic recommendations for owned Zotero items."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

try:
    from .db_relations import get_cocitation_pairs, get_coupling_pairs, get_network_item_keys
    from .item_vectors import get_item_meta, get_item_vectors
except ImportError:  # pragma: no cover - direct script execution
    from db_relations import get_cocitation_pairs, get_coupling_pairs, get_network_item_keys
    from item_vectors import get_item_meta, get_item_vectors


RRF_K = 60
VALID_METHODS = {"coupling", "cocitation", "semantic", "hybrid"}


def _semantic_pairs(item_key: str, candidate_keys: list[str]) -> list[dict[str, Any]]:
    keys = list(dict.fromkeys([item_key, *candidate_keys]))
    vectors = get_item_vectors(keys)
    target = vectors.get(item_key)
    if target is None:
        return []
    target_array = np.asarray(target, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for key in candidate_keys:
        vector = vectors.get(key)
        if vector is None:
            continue
        similarity = float(target_array @ np.asarray(vector, dtype=np.float32))
        if np.isfinite(similarity):
            rows.append({"item_key": key, "cosine_similarity": similarity})
    return sorted(rows, key=lambda row: (-row["cosine_similarity"], row["item_key"]))


def related_items(item_key: str, *, method: str = "hybrid", k: int = 10) -> list[dict[str, Any]]:
    """Return related owned items with human-readable evidence."""
    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of: {', '.join(sorted(VALID_METHODS))}.")
    if k <= 0:
        return []
    internal_k = max(k * 5, 50)
    candidates = [key for key in get_network_item_keys() if key != item_key]
    rankings: dict[str, list[dict[str, Any]]] = {}
    if method in {"coupling", "hybrid"}:
        rankings["coupling"] = get_coupling_pairs(item_key, internal_k)
    if method in {"cocitation", "hybrid"}:
        rankings["cocitation"] = get_cocitation_pairs(item_key, internal_k)
    if method in {"semantic", "hybrid"}:
        rankings["semantic"] = _semantic_pairs(item_key, candidates)[:internal_k]

    scores: dict[str, float] = defaultdict(float)
    evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source, rows in rankings.items():
        for rank, row in enumerate(rows, start=1):
            key = row["item_key"]
            scores[key] += 1.0 / (RRF_K + rank)
            if source == "coupling":
                detail = {
                    "method": source,
                    "shared_references": row["shared_reference_count"],
                    "description": f"共有参照 {row['shared_reference_count']}件",
                }
            elif source == "cocitation":
                detail = {
                    "method": source,
                    "shared_citers": row["shared_citer_count"],
                    "description": f"共引用元 {row['shared_citer_count']}件",
                }
            else:
                detail = {
                    "method": source,
                    "cosine_similarity": round(row["cosine_similarity"], 6),
                    "description": f"内容類似度 {row['cosine_similarity']:.3f}",
                }
            evidence[key].append(detail)

    ordered = sorted(scores, key=lambda key: (-scores[key], key))[:k]
    metadata = get_item_meta(ordered)
    return [
        {
            "item_key": key,
            "title": metadata.get(key, {}).get("title", ""),
            "creators": metadata.get(key, {}).get("creators", ""),
            "year": metadata.get(key, {}).get("year"),
            "rrf_score": scores[key],
            "evidence": evidence[key],
        }
        for key in ordered
    ]


__all__ = ["related_items"]
