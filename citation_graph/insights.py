"""Read-only data service for Citation Graph structure and summaries."""
from __future__ import annotations

from typing import Any

from src import db_relations
from src.chunk_store import get_item_chunks, natural_chunk_key


MAX_PAGE_SIZE = 100


def _item_key(value: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError("item key is required.")
    return normalized


def _page(values: list[dict[str, Any]], cursor: str | None, limit: int) -> dict[str, Any]:
    try:
        offset = int(cursor or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("cursor must be a non-negative integer.") from exc
    if offset < 0:
        raise ValueError("cursor must be a non-negative integer.")
    size = max(1, min(int(limit), MAX_PAGE_SIZE))
    rows = values[offset : offset + size]
    next_offset = offset + len(rows)
    return {
        "items": rows,
        "total": len(values),
        "next_cursor": str(next_offset) if next_offset < len(values) else None,
    }


def _generation_status(
    conn: Any, item_key: str, row_count: int,
) -> dict[str, Any]:
    if row_count:
        return {"status": "available", "count": row_count}
    row = conn.execute(
        "SELECT status, updated_at FROM artifact_processing_status "
        "WHERE item_key = ? AND artifact_type = 'summary' AND attachment_key = ''",
        (item_key,),
    ).fetchone()
    if row:
        status = "processed_empty" if row["status"] == "empty" else row["status"]
        return {"status": status, "count": 0, "updated_at": row["updated_at"]}
    return {"status": "not_processed", "count": 0}


def _current_summary_report_status(
    conn: Any, item_key: str, section_id: str, summary: str, model: str | None,
) -> str | None:
    summary_hash = db_relations._summary_fingerprint(summary, model)
    row = conn.execute(
        "SELECT status FROM summary_quality_reports "
        "WHERE item_key = ? AND section_id = ? AND summary_hash = ?",
        (item_key, section_id, summary_hash),
    ).fetchone()
    return str(row["status"]) if row else None


def get_processing_overview(item_key: str) -> dict[str, Any]:
    """Return explicit per-artifact processing state for Citation Insights."""
    key = _item_key(item_key)
    rows = db_relations.get_item_processing_status(key)
    structure = db_relations.get_document_structure(key)
    statuses = {str(row["status"]) for row in rows}
    if statuses.intersection({"blocked", "failed"}):
        overall = "needs_attention"
    elif statuses.intersection({"pending", "running", "stale"}):
        overall = "pending"
    elif "degraded" in statuses:
        overall = "degraded"
    elif statuses:
        overall = "complete"
    else:
        overall = "not_processed"
    return {
        "item_key": key,
        "overall": overall,
        "artifacts": rows,
        "structure": structure,
    }


def get_document_outline(item_key: str) -> dict[str, Any]:
    """Return the persisted v2 outline with any derived node summaries."""
    key = _item_key(item_key)
    structure = db_relations.get_document_structure(key)
    if structure is None:
        return {"item_key": key, "structure": None, "nodes": []}
    summaries = {
        str(row["node_id"]): row
        for row in db_relations.get_document_node_summaries(key)
        if row.get("quality_status") != "disabled"
    }
    nodes = []
    for node in db_relations.get_document_nodes(key):
        summary = summaries.get(str(node["node_id"]))
        nodes.append({
            "node_id": node["node_id"], "parent_node_id": node.get("parent_node_id"),
            "node_type": node["node_type"], "depth": node["depth"],
            "title": node.get("title") or "", "content_chars": node.get("content_chars") or 0,
            "first_chunk_id": node.get("first_chunk_id") or "",
            "last_chunk_id": node.get("last_chunk_id") or "",
            "source_kind": node.get("source_kind") or "",
            "confidence": node.get("confidence"),
            "summary": summary.get("summary") if summary else None,
            "summary_kind": summary.get("summary_kind") if summary else None,
            "quality_status": summary.get("quality_status") if summary else None,
            "summary_parts": db_relations.get_document_node_summary_parts(str(node["node_id"])) if summary else [],
        })
    return {"item_key": key, "structure": structure, "nodes": nodes}


def get_item_insights(item_key: str) -> dict[str, Any]:
    """Return the lightweight overview and tab counts for one Zotero item."""
    key = _item_key(item_key)
    conn = db_relations.get_db_connection()
    try:
        abstract_row = conn.execute(
            "SELECT abstract FROM item_citation_status WHERE item_key = ?", (key,),
        ).fetchone()
        summary = db_relations.get_item_root_summary(key, searchable_only=False)
        if summary and summary.get("quality_status") == "disabled":
            summary = None
        section_count = int(conn.execute('''
            SELECT COUNT(*)
            FROM document_node_summaries s
            JOIN document_nodes n ON n.node_id = s.node_id
            WHERE s.item_key = ? AND n.node_type != 'item_root'
              AND s.quality_status != 'disabled'
        ''', (key,)).fetchone()[0])
        if summary:
            summary["kind"] = summary.get("summary_kind") or "llm"
            summary["report_status"] = _current_summary_report_status(
                conn, key, "", summary["summary"], summary.get("model"),
            )
        sections = _generation_status(conn, key, section_count)
        return {
            "item_key": key,
            "abstract": abstract_row["abstract"] if abstract_row else None,
            "summary": summary,
            "sections": sections,
            "processing": get_processing_overview(key),
        }
    finally:
        conn.close()


def list_sections(
    item_key: str, *, query: str = "", cursor: str | None = None, limit: int = 50,
) -> dict[str, Any]:
    key = _item_key(item_key)
    needle = str(query or "").strip().casefold()
    conn = db_relations.get_db_connection()
    try:
        rows = [dict(row) for row in conn.execute('''
            SELECT s.*, n.node_type, n.depth, n.title, n.first_chunk_id
            FROM document_node_summaries s
            JOIN document_nodes n ON n.node_id = s.node_id
            WHERE s.item_key = ? AND n.node_type != 'item_root'
              AND s.quality_status != 'disabled'
            ORDER BY n.first_chunk_id, n.depth, n.ordinal, n.node_id
        ''', (key,)).fetchall()]
        rows.sort(key=lambda row: natural_chunk_key(str(row.get("first_chunk_id") or "")))
        result = []
        for row in rows:
            if needle and needle not in "\n".join((
                str(row.get("title") or ""), str(row.get("summary") or ""),
            )).casefold():
                continue
            result.append({
                "section_id": row["node_id"],
                "chapter": row.get("title") or "",
                "summary": row["summary"],
                "model": row.get("model") or "",
                "chunk_count": row.get("source_chunk_count"),
                "chapter_authors": "",
                "first_publication_note": "",
                "updated_at": row.get("updated_at"),
                "report_status": _current_summary_report_status(
                    conn, key, str(row["node_id"]), row["summary"], row.get("model"),
                ),
            })
        return _page(result, cursor, limit)
    finally:
        conn.close()


def get_section_source(item_key: str, section_id: str) -> dict[str, Any]:
    key = _item_key(item_key)
    section_key = str(section_id or "").strip()
    if not section_key:
        raise ValueError("section_id is required.")
    node = next((
        row for row in db_relations.get_document_nodes(key)
        if str(row.get("node_id")) == section_key
    ), None)
    if node is None or node.get("node_type") == "item_root":
        raise KeyError("section source was not found.")
    chunk_ids = set(db_relations.get_node_descendant_chunks([section_key]))
    chunks = [
        chunk for chunk in get_item_chunks(key)
        if str(chunk.get("id") or "") in chunk_ids
    ]
    chunks.sort(key=lambda row: natural_chunk_key(str(row.get("id") or "")))
    return {
        "item_key": key,
        "section_id": section_key,
        "chapter": node.get("title") or "",
        "chunks": [{
            "chunk_id": str(chunk.get("id") or ""),
            "text": str(chunk.get("text") or ""),
            "page": chunk.get("metadata", {}).get("page")
                or chunk.get("metadata", {}).get("pageNumber"),
        } for chunk in chunks if chunk.get("text")],
    }


__all__ = [
    "get_item_insights", "get_section_source", "list_sections",
]
