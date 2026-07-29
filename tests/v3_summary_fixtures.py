from __future__ import annotations

from typing import Any


def seed_v3_summaries(
    db_relations: Any, *, item_key: str = "ITEM", root_summary: str = "item summary",
    sections: list[tuple[str, str, str, list[str]]] | None = None,
    model: str = "deepseek:flash",
) -> dict[str, Any]:
    """Create a minimal persisted V3 tree with accepted node summaries."""
    section_rows = sections or [
        ("w0", "Chapter", "section summary", ["chunk-1"]),
    ]
    root_id = f"node:{item_key}:root"
    fingerprint = f"fixture:{item_key}"
    conn = db_relations.get_db_connection()
    try:
        conn.execute(
            "INSERT INTO document_structures "
            "(item_key, source_fingerprint, structure_version, status, confidence, "
            " node_count, leaf_count, diagnostics_json) "
            "VALUES (?, ?, 'fixture-v1', 'exact', 1.0, ?, ?, '{}')",
            (item_key, fingerprint, len(section_rows) + 1, len(section_rows)),
        )
        conn.execute(
            "INSERT INTO document_nodes "
            "(node_id, item_key, parent_node_id, node_type, depth, ordinal, title, "
            " source_kind, content_chars, first_chunk_id, last_chunk_id) "
            "VALUES (?, ?, NULL, 'item_root', 0, 0, NULL, 'item', 0, ?, ?)",
            (root_id, item_key, section_rows[0][3][0], section_rows[-1][3][-1]),
        )
        section_ids: dict[str, str] = {}
        for ordinal, (name, title, _summary, chunk_ids) in enumerate(section_rows):
            node_id = f"node:{item_key}:{name}"
            section_ids[name] = node_id
            conn.execute(
                "INSERT INTO document_nodes "
                "(node_id, item_key, parent_node_id, node_type, depth, ordinal, title, "
                " source_kind, content_chars, first_chunk_id, last_chunk_id) "
                "VALUES (?, ?, ?, 'section', 1, ?, ?, 'fixture', 0, ?, ?)",
                (node_id, item_key, root_id, ordinal, title, chunk_ids[0], chunk_ids[-1]),
            )
            for chunk_ordinal, chunk_id in enumerate(chunk_ids):
                conn.execute(
                    "INSERT INTO document_node_chunks (node_id, chunk_id, ordinal) "
                    "VALUES (?, ?, ?)",
                    (node_id, chunk_id, chunk_ordinal),
                )
        conn.commit()
    finally:
        conn.close()

    all_chunks = [chunk for row in section_rows for chunk in row[3]]
    db_relations.save_document_node_summary(
        root_id, item_key, root_summary, summary_kind="llm", model=model,
        prompt_version="fixture-v1", source_fingerprint=fingerprint,
        source_chunk_count=len(all_chunks), source_chars=len(root_summary),
        quality_status="accepted",
    )
    for name, _title, summary, chunk_ids in section_rows:
        db_relations.save_document_node_summary(
            section_ids[name], item_key, summary, summary_kind="llm", model=model,
            prompt_version="fixture-v1", source_fingerprint=fingerprint,
            source_chunk_count=len(chunk_ids), source_chars=len(summary),
            quality_status="accepted",
        )
    return {"root_id": root_id, "section_ids": section_ids}
