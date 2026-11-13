"""Small, dependency-free helpers shared by retrieval entry points."""
from __future__ import annotations

from typing import Any


MIN_MINOR_LANG_HITS = 2


def language_balanced_order(
    ranked_hits: list[tuple[str, dict[str, Any]]],
    k: int,
    *,
    min_per_language: int = MIN_MINOR_LANG_HITS,
) -> list[tuple[str, dict[str, Any]]]:
    """Ensure Japanese, English, and Thai hits appear when multiple are available.

    The caller supplies a relevance-ranked candidate pool. The returned first
    ``k`` entries contain a small quota from each language, while preserving
    the original relevance order within the selected set. Remaining entries
    are appended unchanged so callers can continue filtering if necessary.
    """
    if k <= 1 or min_per_language <= 0:
        return ranked_hits

    positions: dict[str, list[int]] = {"ja": [], "en": [], "th": []}
    for index, (_, hit) in enumerate(ranked_hits):
        metadata = hit.get("metadata") if isinstance(hit, dict) else None
        lang = metadata.get("lang") if isinstance(metadata, dict) else None
        if lang in positions:
            positions[lang].append(index)

    available = sorted(
        (indexes for indexes in positions.values() if indexes), key=lambda indexes: indexes[0],
    )
    if len(available) < 2:
        return ranked_hits

    available = available[:k]
    quota = min(min_per_language, max(1, k // len(available)))
    selected = {index for indexes in available for index in indexes[:quota]}
    for index in range(len(ranked_hits)):
        if len(selected) >= min(k, len(ranked_hits)):
            break
        selected.add(index)

    top = [hit for index, hit in enumerate(ranked_hits) if index in selected]
    remainder = [hit for index, hit in enumerate(ranked_hits) if index not in selected]
    return top + remainder
