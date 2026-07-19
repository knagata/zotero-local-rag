"""Bottom-up summaries for canonical document-structure nodes.

This is intentionally separate from the legacy item/section pipeline.  It can
be backfilled and evaluated behind a feature flag before changing retrieval.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List
from pathlib import Path
import os

try:
    from .build_summaries import (
        _excluded_from_llm, _extractive_section, _llm_summary_only_item, _llm_summary_only_section,
        classify_section_content, is_meta_summary,
    )
    from .chunk_store import get_item_chunks
    from .db_relations import (
        get_all_document_node_summaries, get_document_nodes, get_document_structure, mark_artifact_status,
        replace_document_node_summary_parts, save_document_node_summary,
    )
    from .embedder import create_embedding_function, open_chroma_collection, resolve_embedder_settings
    from .chunk_store import active_collection_name
    from .llm_client import LLMError, RateLimitReached, get_llm_spec
except ImportError:  # pragma: no cover
    from build_summaries import _excluded_from_llm, _extractive_section, _llm_summary_only_item, _llm_summary_only_section, classify_section_content, is_meta_summary
    from chunk_store import get_item_chunks
    from db_relations import get_all_document_node_summaries, get_document_nodes, get_document_structure, mark_artifact_status, replace_document_node_summary_parts, save_document_node_summary
    from embedder import create_embedding_function, open_chroma_collection, resolve_embedder_settings
    from chunk_store import active_collection_name
    from llm_client import LLMError, RateLimitReached, get_llm_spec


PROMPT_VERSION = "structure-v2-1"
MAX_PARENT_INPUT_CHARS = 30_000
ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", ROOT / "data" / "chroma"))


def _nearest_title(node: Dict[str, Any], by_id: Dict[str, Dict[str, Any]]) -> str:
    current = node
    while current:
        if current.get("title"):
            return str(current["title"])
        parent_id = current.get("parent_node_id")
        current = by_id.get(str(parent_id)) if parent_id else None
    return "資料"


def _groups(rows: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    size = 0
    for row in rows:
        row_size = len(str(row.get("summary") or "")) + len(str(row.get("title") or "")) + 8
        if current and size + row_size > MAX_PARENT_INPUT_CHARS:
            groups.append(current)
            current, size = [], 0
        current.append(row)
        size += row_size
    if current:
        groups.append(current)
    return groups


def _deepseek_model(role: str) -> str:
    """Resolve the configured role while keeping this API-only pipeline explicit."""
    spec = get_llm_spec(role)
    provider, separator, model = spec.partition(":")
    if provider != "deepseek" or not separator or not model:
        raise LLMError(f"Structure summaries require a DeepSeek API role, got {spec!r}.")
    return model


def _parent_summary(
    title: str, children: List[Dict[str, Any]], *, use_llm: bool,
) -> tuple[str, str, str, List[Dict[str, Any]]]:
    """Return summary, kind, model.  Child order is always document order."""
    if not children:
        return "", "extractive", "", []
    if not use_llm:
        return "\n\n".join(str(row["summary"]) for row in children)[:4000], "extractive", "extractive", []
    try:
        groups = _groups(children)
        reduced: List[Dict[str, Any]] = []
        parts: List[Dict[str, Any]] = []
        for index, group in enumerate(groups):
            result, model = _llm_summary_only_item(title, [
                {"section_id": row["node_id"], "summary": row["summary"]} for row in group
            ], model=_deepseek_model("cheap"), reasoning="disabled")
            summary = str(result.get("summary") or "").strip()
            if not summary or is_meta_summary(summary):
                raise LLMError("node summary was empty or meta")
            reduced.append({"node_id": f"part-{index}", "summary": summary})
            parts.append({"child_node_ids": [row["node_id"] for row in group], "summary": summary, "model": model})
        if len(reduced) == 1:
            return reduced[0]["summary"], "llm", model, parts
        result, model = _llm_summary_only_item(title, [
            {"section_id": row["node_id"], "summary": row["summary"]} for row in reduced
        ], model=_deepseek_model("standard"), reasoning="disabled")
        summary = str(result.get("summary") or "").strip()
        if not summary or is_meta_summary(summary):
            raise LLMError("node summary was empty or meta")
        return summary, "llm", model, parts
    except RateLimitReached:
        raise
    except LLMError:
        return "\n\n".join(str(row["summary"]) for row in children)[:4000], "extractive", "extractive", []


def build_structure_summaries(item_key: str, *, mode: str = "extractive", force: bool = False) -> Dict[str, Any]:
    """Build leaf then parent summaries from a persisted v2 document tree."""
    structure = get_document_structure(item_key)
    nodes = get_document_nodes(item_key, include_chunks=True)
    if structure is None or not nodes:
        mark_artifact_status(
            item_key, "summary", "blocked", reason_code="structure_missing",
            message="Build document structure v2 before building node summaries.",
        )
        return {"item_key": item_key, "status": "blocked", "reason": "structure_missing"}
    chunks = {str(chunk.get("id") or ""): chunk for chunk in get_item_chunks(item_key)}
    if not chunks:
        mark_artifact_status(item_key, "summary", "blocked", reason_code="no_chunks")
        return {"item_key": item_key, "status": "blocked", "reason": "no_chunks"}
    requested_llm = mode == "llm"
    excluded, exclusion_reason = _excluded_from_llm(item_key) if requested_llm else (False, None)
    use_llm = requested_llm and not excluded
    mark_artifact_status(
        item_key, "summary", "running", source_fingerprint=structure["source_fingerprint"],
        processor_version=PROMPT_VERSION,
    )
    by_id = {str(node["node_id"]): node for node in nodes}
    children: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        if node.get("parent_node_id"):
            children[str(node["parent_node_id"])].append(node)
    generated: Dict[str, Dict[str, Any]] = {}
    skipped = 0
    try:
        for node in sorted(nodes, key=lambda row: (int(row["depth"]), str(row.get("first_chunk_id") or "")), reverse=True):
            node_id = str(node["node_id"])
            direct = node.get("chunks") or []
            title = _nearest_title(node, by_id)
            if direct:
                source_chunks = [chunks[row["chunk_id"]] for row in direct if row["chunk_id"] in chunks]
                if not source_chunks:
                    continue
                section = {"section_id": node_id, "chapter": title, "chunks": source_chunks}
                if classify_section_content(section) == "non_content":
                    skipped += 1
                    continue
                if use_llm:
                    try:
                        result, model_name = _llm_summary_only_section(
                            section, model=_deepseek_model("cheap"), reasoning="disabled",
                        )
                        summary = str(result.get("summary") or "").strip()
                        if not summary or is_meta_summary(summary):
                            raise LLMError("leaf summary was empty or meta")
                        kind, quality = "llm", "accepted"
                    except RateLimitReached:
                        raise
                    except LLMError:
                        result, model_name = _extractive_section(section), "extractive"
                        summary, kind, quality = str(result.get("summary") or ""), "extractive", "degraded"
                else:
                    result, model_name = _extractive_section(section), "extractive"
                    summary, kind, quality = str(result.get("summary") or ""), "extractive", "degraded"
                if not summary:
                    continue
                generated[node_id] = {
                    "node_id": node_id, "summary": summary, "kind": kind, "model": model_name,
                    "source_chunk_count": len(source_chunks),
                }
                save_document_node_summary(
                    node_id, item_key, summary, summary_kind=kind, model=model_name,
                    prompt_version=PROMPT_VERSION, source_fingerprint=structure["source_fingerprint"],
                    source_chunk_count=len(source_chunks), source_chars=sum(len(str(row.get("text") or "")) for row in source_chunks),
                    quality_status=quality,
                )
                replace_document_node_summary_parts(
                    node_id, [], prompt_version=PROMPT_VERSION,
                    source_fingerprint=structure["source_fingerprint"],
                )
                continue
            child_rows = [generated[str(child["node_id"])] for child in children.get(node_id, []) if str(child["node_id"]) in generated]
            if not child_rows:
                continue
            summary, kind, model_name, parts = _parent_summary(title, child_rows, use_llm=use_llm)
            if not summary:
                continue
            quality = "accepted" if kind == "llm" else "degraded"
            source_chunk_count = sum(int(row.get("source_chunk_count") or 0) for row in child_rows)
            generated[node_id] = {
                "node_id": node_id, "summary": summary, "kind": kind, "model": model_name,
                "source_chunk_count": source_chunk_count,
            }
            save_document_node_summary(
                node_id, item_key, summary, summary_kind=kind, model=model_name,
                prompt_version=PROMPT_VERSION, source_fingerprint=structure["source_fingerprint"],
                source_chunk_count=source_chunk_count, source_chars=int(node.get("content_chars") or 0),
                quality_status=quality,
            )
            replace_document_node_summary_parts(
                node_id, parts, prompt_version=PROMPT_VERSION,
                source_fingerprint=structure["source_fingerprint"],
            )
        llm_count = sum(1 for row in generated.values() if row["kind"] == "llm")
        artifact_status = "success" if llm_count else ("empty" if not generated else "degraded")
        mark_artifact_status(
            item_key, "summary", artifact_status,
            reason_code=(exclusion_reason if excluded else ("no_summary_content" if not generated else "no_llm_summary")) if not llm_count else None,
            source_fingerprint=structure["source_fingerprint"], processor_version=PROMPT_VERSION,
            counts={"nodes": len(generated), "llm": llm_count, "extractive": len(generated) - llm_count, "skipped": skipped},
            fallback_kind="extractive_outline" if not llm_count else None,
        )
        return {"item_key": item_key, "status": artifact_status, "nodes": len(generated), "llm": llm_count, "skipped": skipped}
    except RateLimitReached:
        mark_artifact_status(item_key, "summary", "failed", reason_code="rate_limited", retryable=True)
        raise
    except Exception as exc:
        mark_artifact_status(item_key, "summary", "failed", reason_code="summary_build_failed", message=str(exc)[:1000], retryable=True)
        raise


def embed_structure_summaries(*, item_keys: set[str] | None = None) -> Dict[str, int]:
    """Build the searchable LLM-only node-summary collection."""
    base = active_collection_name(chroma_dir=CHROMA_DIR)
    if not base:
        raise RuntimeError("No active paragraph collection was found.")
    cfg = resolve_embedder_settings(ROOT)
    embedding_function = create_embedding_function(cfg)
    collection = open_chroma_collection(CHROMA_DIR, f"{base}__sum_node", embedding_function)
    rows = [
        row for row in get_all_document_node_summaries(searchable_only=True)
        if item_keys is None or row["item_key"] in item_keys
    ]
    if item_keys is None:
        existing_ids = set(collection.get(include=[]).get("ids") or [])
        expected_ids = {f"sum:node:{row['node_id']}" for row in rows}
        stale = sorted(existing_ids - expected_ids)
        for start in range(0, len(stale), 1000):
            collection.delete(ids=stale[start:start + 1000])
    else:
        for item_key in item_keys:
            try:
                collection.delete(where={"itemKey": item_key})
            except Exception:
                pass
    documents = []
    metadatas = []
    ids = []
    for row in rows:
        ids.append(f"sum:node:{row['node_id']}")
        documents.append("\n".join(filter(None, [str(row.get("title") or ""), str(row["summary"])]))[:4000])
        metadatas.append({
            "itemKey": str(row["item_key"]), "attachmentKey": str(row.get("attachment_key") or ""),
            "node_id": str(row["node_id"]), "parent_node_id": str(row.get("parent_node_id") or ""),
            "node_type": str(row.get("node_type") or ""), "depth": int(row.get("depth") or 0),
            "title": str(row.get("title") or ""), "summary_kind": "llm",
        })
    batch_size = max(1, int(os.environ.get("SUMMARY_EMBED_BATCH_SIZE", "16")))
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start:start + batch_size]
        batch_docs = documents[start:start + batch_size]
        collection.upsert(
            ids=batch_ids, documents=batch_docs, metadatas=metadatas[start:start + batch_size],
            embeddings=embedding_function(batch_docs),
        )
    client = getattr(collection, "_chroma_client", None)
    if client:
        client.close()
    return {"nodes": len(ids)}


__all__ = ["PROMPT_VERSION", "build_structure_summaries", "embed_structure_summaries"]
