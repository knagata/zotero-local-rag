"""Read-only data service for Citation Graph summaries and structured cases."""
from __future__ import annotations

from typing import Any, Iterable

from . import db_relations
from .chunk_store import get_item_chunks, natural_chunk_key


CASE_STATUSES = ("confirmed", "partial", "candidate")
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
    conn: Any, item_key: str, kind: str, row_count: int,
) -> dict[str, Any]:
    if row_count:
        return {"status": "available", "count": row_count}
    row = conn.execute(
        "SELECT status, row_count, updated_at FROM insight_generation_status "
        "WHERE item_key = ? AND kind = ?",
        (item_key, kind),
    ).fetchone()
    if row:
        return {"status": row["status"], "count": 0, "updated_at": row["updated_at"]}
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


def _case_report_statuses(
    conn: Any, rows: list[dict[str, Any]],
) -> dict[int, str]:
    if not rows:
        return {}
    current_hashes = {
        int(row["case_id"]): db_relations._case_fingerprint(row) for row in rows
    }
    placeholders = ",".join("?" for _ in rows)
    reports = conn.execute(
        f"SELECT case_id, case_hash, status FROM case_quality_reports "
        f"WHERE case_id IN ({placeholders})",
        tuple(current_hashes),
    ).fetchall()
    return {
        int(report["case_id"]): str(report["status"])
        for report in reports
        if current_hashes.get(int(report["case_id"])) == report["case_hash"]
    }


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
        str(row["node_id"]): row for row in db_relations.get_document_node_summaries(key)
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
        summary_row = conn.execute(
            "SELECT summary, model, updated_at FROM item_summaries WHERE item_key = ?", (key,),
        ).fetchone()
        section_count = int(conn.execute(
            "SELECT COUNT(*) FROM section_summaries WHERE item_key = ?", (key,),
        ).fetchone()[0])
        case_rows = [dict(row) for row in conn.execute(
            "SELECT * FROM case_annotations WHERE item_key = ? ORDER BY case_id", (key,),
        ).fetchall()]
        report_statuses = _case_report_statuses(conn, case_rows)
        counts = {status: 0 for status in CASE_STATUSES}
        for row in case_rows:
            if report_statuses.get(int(row["case_id"])) == "disabled":
                continue
            status = str(row.get("quality_status") or "confirmed")
            if status in counts:
                counts[status] += 1
        visible_case_count = sum(counts.values())
        summary = dict(summary_row) if summary_row else None
        if summary:
            summary["kind"] = "extractive" if summary.get("model") == "extractive" else "llm"
            summary["report_status"] = _current_summary_report_status(
                conn, key, "", summary["summary"], summary.get("model"),
            )
        sections = _generation_status(conn, key, "sections", section_count)
        cases = _generation_status(conn, key, "cases", visible_case_count)
        cases["counts"] = counts
        return {
            "item_key": key,
            "abstract": abstract_row["abstract"] if abstract_row else None,
            "summary": summary,
            "sections": sections,
            "cases": cases,
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
        rows = [dict(row) for row in conn.execute(
            "SELECT * FROM section_summaries WHERE item_key = ?", (key,),
        ).fetchall()]
        rows.sort(key=lambda row: natural_chunk_key(str(row.get("section_id") or "")))
        result = []
        for row in rows:
            if needle and needle not in "\n".join((
                str(row.get("chapter") or ""), str(row.get("summary") or ""),
            )).casefold():
                continue
            result.append({
                "section_id": row["section_id"],
                "chapter": row.get("chapter") or "",
                "summary": row["summary"],
                "model": row.get("model") or "",
                "chunk_count": row.get("chunk_count"),
                "chapter_authors": row.get("chapter_authors") or "",
                "first_publication_note": row.get("first_publication_note") or "",
                "updated_at": row.get("updated_at"),
                "report_status": _current_summary_report_status(
                    conn, key, str(row["section_id"]), row["summary"], row.get("model"),
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
    # Use the same deterministic grouping routine as summary generation so IDs
    # cannot drift between the stored summary and its displayed source chunks.
    from .build_summaries import split_sections

    sections = {
        str(section["section_id"]): section
        for section in split_sections(get_item_chunks(key))
    }
    section = sections.get(section_key)
    if section is None:
        raise KeyError("section source was not found.")
    return {
        "item_key": key,
        "section_id": section_key,
        "chapter": section.get("chapter") or "",
        "chunks": [{
            "chunk_id": str(chunk.get("id") or ""),
            "text": str(chunk.get("text") or ""),
            "page": chunk.get("metadata", {}).get("page")
                or chunk.get("metadata", {}).get("pageNumber"),
        } for chunk in section.get("chunks") or [] if chunk.get("text")],
    }


def _normalize_statuses(statuses: Iterable[str] | None) -> set[str]:
    values = {
        str(value or "").strip().lower() for value in (statuses or CASE_STATUSES)
        if str(value or "").strip()
    }
    unknown = values.difference(CASE_STATUSES)
    if unknown:
        raise ValueError("statuses must contain only confirmed, partial, or candidate.")
    if not values:
        raise ValueError("at least one case status is required.")
    return values


def list_cases(
    item_key: str, *, query: str = "", statuses: Iterable[str] | None = None,
    section_id: str = "", cursor: str | None = None, limit: int = 20,
) -> dict[str, Any]:
    key = _item_key(item_key)
    selected = _normalize_statuses(statuses)
    needle = str(query or "").strip().casefold()
    section_key = str(section_id or "").strip()
    conn = db_relations.get_db_connection()
    try:
        rows = [dict(row) for row in conn.execute('''
            SELECT c.*, COUNT(e.evidence_id) AS evidence_count
            FROM case_annotations c
            LEFT JOIN case_evidence e ON e.case_id = c.case_id
            WHERE c.item_key = ?
            GROUP BY c.case_id ORDER BY c.case_id
        ''', (key,)).fetchall()]
        report_statuses = _case_report_statuses(conn, rows)
        result = []
        for row in rows:
            report_status = report_statuses.get(int(row["case_id"]))
            if report_status == "disabled":
                continue
            quality_status = str(row.get("quality_status") or "confirmed")
            if quality_status not in selected:
                continue
            if section_key and str(row.get("section_id") or "") != section_key:
                continue
            searchable = "\n".join(str(row.get(field) or "") for field in (
                "description", "region", "grp", "period", "practices", "phenomena",
            )).casefold()
            if needle and needle not in searchable:
                continue
            result.append({
                "case_id": int(row["case_id"]),
                "section_id": row.get("section_id") or "",
                "node_id": row.get("node_id") or "",
                "title": row.get("title") or "",
                "description": row["description"],
                "case_type": row.get("case_type") or "other",
                "region": row.get("region") or "",
                "group": row.get("grp") or "",
                "practices": row.get("practices") or "",
                "phenomena": row.get("phenomena") or "",
                "period": row.get("period") or "",
                "source_kind": row.get("source_kind") or "",
                "model": row.get("model") or "",
                "quality_status": quality_status,
                "confidence": row.get("confidence"),
                "normalization_version": row.get("normalization_version") or "v1_legacy",
                "updated_at": row.get("updated_at"),
                "evidence_count": int(row.get("evidence_count") or 0),
                "report_status": report_status,
            })
        return _page(result, cursor, limit)
    finally:
        conn.close()


def get_case_evidence(case_id: int) -> dict[str, Any]:
    try:
        normalized_id = int(case_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("case_id must be an integer.") from exc
    conn = db_relations.get_db_connection()
    try:
        case_row = conn.execute(
            "SELECT * FROM case_annotations WHERE case_id = ?", (normalized_id,),
        ).fetchone()
        if case_row is None:
            raise KeyError("case was not found.")
        current = dict(case_row)
        if _case_report_statuses(conn, [current]).get(normalized_id) == "disabled":
            raise KeyError("case was not found.")
        rows = conn.execute(
            "SELECT field_name, chunk_id, evidence_quote FROM case_evidence "
            "WHERE case_id = ? ORDER BY evidence_id", (normalized_id,),
        ).fetchall()
        return {
            "case_id": normalized_id,
            "item_key": current["item_key"],
            "evidence": [dict(row) for row in rows],
        }
    finally:
        conn.close()


__all__ = [
    "CASE_STATUSES", "get_case_evidence", "get_item_insights", "get_section_source",
    "list_cases", "list_sections",
]
