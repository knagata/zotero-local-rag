"""Canonical, source-order-preserving document structure v2.

The builder deliberately does not infer relationships from semantic similarity.
It consumes extractor metadata in document order, preserves every contiguous
heading run, and uses deterministic contiguous segments only when no structure
is available.  This keeps the tree safe to use as a retrieval routing map.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence

try:
    from .chunk_store import natural_chunk_key
except ImportError:  # pragma: no cover - direct src entrypoint
    from chunk_store import natural_chunk_key


STRUCTURE_VERSION = "2"
TARGET_SEGMENT_CHARS = 18_000
MAX_SEGMENT_CHARS = 30_000


def source_fingerprint(chunks: Sequence[Dict[str, Any]]) -> str:
    """Return a stable fingerprint for ordered chunk text and source metadata."""
    digest = hashlib.sha256()
    for chunk in sorted(chunks, key=lambda value: natural_chunk_key(str(value.get("id") or ""))):
        metadata = chunk.get("metadata") or {}
        payload = {
            "id": str(chunk.get("id") or ""),
            "attachment": str(metadata.get("attachmentKey") or ""),
            "locator": str(metadata.get("locator") or ""),
            "text": str(chunk.get("text") or ""),
        }
        digest.update(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return f"sha256:{digest.hexdigest()}"


def _normalise_title(value: Any) -> str:
    return " ".join(str(value or "").split())


def _node_id(*, item_key: str, kind: str, locator: Dict[str, Any]) -> str:
    payload = json.dumps(
        {"item_key": item_key, "kind": kind, "locator": locator},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return "dn:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _metadata_path(metadata: Dict[str, Any]) -> List[str]:
    """Read the richest available heading path without guessing hierarchy."""
    raw_path = metadata.get("structure_path") or metadata.get("heading_path")
    if isinstance(raw_path, str):
        try:
            raw_path = json.loads(raw_path)
        except (TypeError, ValueError):
            raw_path = [raw_path]
    if isinstance(raw_path, (list, tuple)):
        values: List[str] = []
        for value in raw_path:
            if isinstance(value, dict):
                value = value.get("title") or value.get("text")
            title = _normalise_title(value)
            if title:
                values.append(title)
        if values:
            return values
    values = [_normalise_title(metadata.get("chapter")), _normalise_title(metadata.get("section"))]
    return [value for value in values if value]


def _heading_type(depth: int) -> str:
    if depth == 1:
        return "chapter"
    if depth == 2:
        return "section"
    return "subsection"


def _attachment_groups(chunks: Sequence[Dict[str, Any]]) -> List[tuple[str, List[Dict[str, Any]]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for chunk in sorted(chunks, key=lambda value: natural_chunk_key(str(value.get("id") or ""))):
        metadata = chunk.get("metadata") or {}
        key = str(metadata.get("attachmentKey") or "")
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(chunk)
    return [(key, groups[key]) for key in order]


def _contiguous_segments(chunks: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Deterministically split a flat run while never reordering or clustering it."""
    segments: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_chars = 0
    for chunk in chunks:
        text_len = len(str(chunk.get("text") or ""))
        if current and (
            current_chars + text_len > MAX_SEGMENT_CHARS
            or current_chars >= TARGET_SEGMENT_CHARS
        ):
            segments.append(current)
            current = []
            current_chars = 0
        current.append(chunk)
        current_chars += text_len
    if current:
        segments.append(current)
    return segments


def _node_payload(
    *, item_key: str, parent_node_id: str | None, node_type: str, depth: int, ordinal: int,
    title: str | None, source_kind: str, locator: Dict[str, Any], confidence: float,
) -> Dict[str, Any]:
    node_id = _node_id(item_key=item_key, kind=node_type, locator=locator)
    return {
        "node_id": node_id,
        "item_key": item_key,
        "attachment_key": locator.get("attachment_key"),
        "parent_node_id": parent_node_id,
        "node_type": node_type,
        "depth": depth,
        "ordinal": ordinal,
        "title": title,
        "normalized_title": _normalise_title(title).casefold() if title else None,
        "source_kind": source_kind,
        "source_locator": locator,
        "confidence": confidence,
        "content_chars": 0,
        "first_chunk_id": None,
        "last_chunk_id": None,
        "chunks": [],
    }


def _set_leaf_chunks(node: Dict[str, Any], chunks: Sequence[Dict[str, Any]]) -> None:
    ids = [str(chunk.get("id") or "") for chunk in chunks]
    if not ids or any(not value for value in ids):
        raise ValueError("every source chunk needs a non-empty id")
    node["chunks"] = [{"chunk_id": value, "ordinal": index} for index, value in enumerate(ids)]
    node["first_chunk_id"] = ids[0]
    node["last_chunk_id"] = ids[-1]
    node["content_chars"] = sum(len(str(chunk.get("text") or "")) for chunk in chunks)


def _roll_up_content(nodes: List[Dict[str, Any]]) -> None:
    by_id = {node["node_id"]: node for node in nodes}
    for node in reversed(nodes):
        parent_id = node.get("parent_node_id")
        if not parent_id:
            continue
        parent = by_id[parent_id]
        parent["content_chars"] += int(node.get("content_chars") or 0)
        if parent.get("first_chunk_id") is None and node.get("first_chunk_id"):
            parent["first_chunk_id"] = node["first_chunk_id"]
        if node.get("last_chunk_id"):
            parent["last_chunk_id"] = node["last_chunk_id"]


def validate_structure(nodes: Sequence[Dict[str, Any]], chunks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate coverage, parent ordering, and leaf-only source assignment."""
    expected = [str(chunk.get("id") or "") for chunk in chunks]
    expected_set = set(expected)
    seen_nodes: set[str] = set()
    assigned: List[str] = []
    errors: List[str] = []
    for node in nodes:
        node_id = str(node.get("node_id") or "")
        if not node_id or node_id in seen_nodes:
            errors.append("duplicate_or_missing_node_id")
            continue
        parent = node.get("parent_node_id")
        if parent and parent not in seen_nodes:
            errors.append(f"parent_not_before_child:{node_id}")
        seen_nodes.add(node_id)
        leaf_chunks = node.get("chunks") or []
        if leaf_chunks and any(child.get("parent_node_id") == node_id for child in nodes):
            errors.append(f"non_leaf_has_chunks:{node_id}")
        assigned.extend(str(value.get("chunk_id") if isinstance(value, dict) else value) for value in leaf_chunks)
    if len(assigned) != len(set(assigned)):
        errors.append("duplicate_chunk_assignment")
    if set(assigned) != expected_set:
        errors.append("chunk_coverage_mismatch")
    if expected != sorted(expected, key=natural_chunk_key):
        errors.append("input_chunks_not_in_document_order")
    return {
        "valid": not errors,
        "errors": errors,
        "expected_chunk_count": len(expected),
        "assigned_chunk_count": len(assigned),
        "node_count": len(nodes),
        "leaf_count": sum(1 for node in nodes if node.get("chunks")),
    }


def build_document_structure(item_key: str, chunks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a canonical tree from ordered extractor chunks.

    The function is pure and intentionally accepts Chroma-style ``{"id",
    "text", "metadata"}`` rows, making it safe for audit dry-runs and tests.
    """
    if not item_key:
        raise ValueError("item_key is required")
    ordered_chunks = sorted(chunks, key=lambda value: natural_chunk_key(str(value.get("id") or "")))
    fingerprint = source_fingerprint(ordered_chunks)
    if not ordered_chunks:
        return {
            "item_key": item_key, "source_fingerprint": fingerprint,
            "structure_version": STRUCTURE_VERSION, "status": "unavailable", "confidence": 0.0,
            "nodes": [], "diagnostics": {"reason": "no_chunks", "valid": True},
        }

    nodes: List[Dict[str, Any]] = []
    sibling_ordinals: Dict[str | None, int] = defaultdict(int)

    def add_node(**kwargs: Any) -> Dict[str, Any]:
        parent = kwargs.get("parent_node_id")
        ordinal = sibling_ordinals[parent]
        sibling_ordinals[parent] += 1
        node = _node_payload(ordinal=ordinal, **kwargs)
        nodes.append(node)
        return node

    item_root = add_node(
        item_key=item_key, parent_node_id=None, node_type="item_root", depth=0,
        title=None, source_kind="item", locator={"item_key": item_key}, confidence=1.0,
    )
    explicit_runs = 0
    fallback_runs = 0
    attachment_count = 0

    for attachment_key, attachment_chunks in _attachment_groups(ordered_chunks):
        attachment_count += 1
        first_meta = attachment_chunks[0].get("metadata") or {}
        attachment_title = _normalise_title(first_meta.get("filename") or first_meta.get("title")) or None
        attachment_root = add_node(
            item_key=item_key, parent_node_id=item_root["node_id"], node_type="attachment_root", depth=1,
            title=attachment_title, source_kind="attachment",
            locator={"attachment_key": attachment_key, "first_chunk_id": attachment_chunks[0]["id"]},
            confidence=1.0,
        )
        active_path: List[str] = []
        active_nodes: List[Dict[str, Any]] = []
        index = 0
        while index < len(attachment_chunks):
            chunk = attachment_chunks[index]
            path = _metadata_path(chunk.get("metadata") or {})
            run = [chunk]
            index += 1
            while index < len(attachment_chunks):
                next_path = _metadata_path(attachment_chunks[index].get("metadata") or {})
                if next_path != path:
                    break
                run.append(attachment_chunks[index])
                index += 1

            if path:
                explicit_runs += 1
                common = 0
                while common < len(path) and common < len(active_path) and path[common] == active_path[common]:
                    common += 1
                active_path = active_path[:common]
                active_nodes = active_nodes[:common]
                parent = active_nodes[-1] if active_nodes else attachment_root
                for path_index in range(common, len(path)):
                    title = path[path_index]
                    heading = add_node(
                        item_key=item_key, parent_node_id=parent["node_id"],
                        node_type=_heading_type(path_index + 1), depth=parent["depth"] + 1,
                        title=title, source_kind="metadata_heading",
                        locator={
                            "attachment_key": attachment_key, "path": path[: path_index + 1],
                            "first_chunk_id": run[0]["id"],
                        }, confidence=0.85,
                    )
                    active_path.append(title)
                    active_nodes.append(heading)
                    parent = heading
                parent = active_nodes[-1]
                leaf = add_node(
                    item_key=item_key, parent_node_id=parent["node_id"], node_type="semantic_segment",
                    depth=parent["depth"] + 1, title=None, source_kind="metadata_heading",
                    locator={"attachment_key": attachment_key, "first_chunk_id": run[0]["id"], "path": path},
                    confidence=0.85,
                )
                _set_leaf_chunks(leaf, run)
            else:
                fallback_runs += 1
                parent = active_nodes[-1] if active_nodes else attachment_root
                for segment_index, segment in enumerate(_contiguous_segments(run)):
                    leaf = add_node(
                        item_key=item_key, parent_node_id=parent["node_id"], node_type="semantic_segment",
                        depth=parent["depth"] + 1, title=None, source_kind="semantic_fallback",
                        locator={
                            "attachment_key": attachment_key, "first_chunk_id": segment[0]["id"],
                            "segment_index": segment_index,
                        }, confidence=0.5,
                    )
                    _set_leaf_chunks(leaf, segment)

    _roll_up_content(nodes)
    diagnostics = validate_structure(nodes, ordered_chunks)
    diagnostics.update({
        "attachment_count": attachment_count,
        "explicit_runs": explicit_runs,
        "fallback_runs": fallback_runs,
    })
    if not diagnostics["valid"]:
        raise ValueError(f"invalid generated document structure: {diagnostics['errors']}")
    if explicit_runs and not fallback_runs:
        status, confidence = "exact", 0.85
    elif explicit_runs:
        status, confidence = "recovered", 0.7
    else:
        status, confidence = "flat_fallback", 0.5
    return {
        "item_key": item_key, "source_fingerprint": fingerprint,
        "structure_version": STRUCTURE_VERSION, "status": status, "confidence": confidence,
        "nodes": nodes, "diagnostics": diagnostics,
    }


__all__ = [
    "MAX_SEGMENT_CHARS", "STRUCTURE_VERSION", "TARGET_SEGMENT_CHARS",
    "build_document_structure", "source_fingerprint", "validate_structure",
]
