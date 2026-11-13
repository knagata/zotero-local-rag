"""Reduce a copied query generation to documents approved by an indexed audit."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


ITEM_COLUMNS = {"item_key", "citing_item_key", "cited_item_key", "zotero_item_key"}


def _chroma_rows(path: Path, collection_name: str, eligible_items: set[str], *, limit: int = 5000):
    import chromadb

    client = chromadb.PersistentClient(path=str(path))
    try:
        collection = client.get_collection(collection_name, embedding_function=None)
        total = collection.count()
        excluded: list[str] = []
        kept: list[str] = []
        attachment_keys: set[str] = set()
        for offset in range(0, total, limit):
            batch = collection.get(limit=limit, offset=offset, include=["metadatas"])
            ids = batch.get("ids") or []
            metadatas = batch.get("metadatas") or []
            if len(ids) != len(metadatas):
                raise ValueError("Partial snapshot cannot validate Chroma metadata rows")
            for chunk_id, metadata in zip(ids, metadatas, strict=True):
                metadata = metadata if isinstance(metadata, dict) else {}
                item_key = str(metadata.get("itemKey") or "")
                if item_key in eligible_items:
                    kept.append(str(chunk_id))
                    attachment_key = str(metadata.get("attachmentKey") or "")
                    if attachment_key:
                        attachment_keys.add(attachment_key)
                else:
                    excluded.append(str(chunk_id))
        for offset in range(0, len(excluded), 1000):
            collection.delete(ids=excluded[offset:offset + 1000])
        if collection.count() != len(kept):
            raise ValueError("Partial snapshot Chroma row count changed unexpectedly")
        return set(kept), attachment_keys
    finally:
        client.close()


def _filter_lexical(path: Path, eligible_items: set[str]) -> set[str]:
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TEMP TABLE eligible_items (item_key TEXT PRIMARY KEY)")
        connection.executemany("INSERT INTO eligible_items VALUES (?)", [(key,) for key in sorted(eligible_items)])
        connection.execute("DELETE FROM chunks_fts WHERE item_key IS NULL OR item_key NOT IN (SELECT item_key FROM eligible_items)")
        ids = {str(row[0]) for row in connection.execute("SELECT chunk_id FROM chunks_fts")}
        connection.commit()
        return ids
    finally:
        connection.close()


def _filter_relations(path: Path, eligible_items: set[str], eligible_attachments: set[str]) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TEMP TABLE eligible_items (item_key TEXT PRIMARY KEY)")
        connection.execute("CREATE TEMP TABLE eligible_attachments (attachment_key TEXT PRIMARY KEY)")
        connection.executemany("INSERT INTO eligible_items VALUES (?)", [(key,) for key in sorted(eligible_items)])
        connection.executemany("INSERT INTO eligible_attachments VALUES (?)", [(key,) for key in sorted(eligible_attachments)])
        tables = [str(row[0]) for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'fts_%'")]
        for table in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            columns = {str(row[1]) for row in connection.execute(f"PRAGMA table_info({quoted})")}
            conditions = [
                f'({column} IS NOT NULL AND {column} NOT IN (SELECT item_key FROM eligible_items))'
                for column in sorted(columns & ITEM_COLUMNS)
            ]
            if "attachment_key" in columns:
                conditions.append("(attachment_key IS NOT NULL AND attachment_key NOT IN (SELECT attachment_key FROM eligible_attachments))")
            if conditions:
                connection.execute(f"DELETE FROM {quoted} WHERE " + " OR ".join(conditions))
        if "document_nodes" in tables:
            for table in ("document_node_chunks", "document_node_summaries"):
                if table in tables:
                    connection.execute(
                        f'DELETE FROM "{table}" WHERE node_id NOT IN (SELECT node_id FROM document_nodes)')
        connection.commit()
    finally:
        connection.close()


def filter_query_copy(chroma_path: Path, collection_name: str, lexical_path: Path,
                      manifest_path: Path, relations_path: Path,
                      eligible_items: list[str]) -> dict[str, Any]:
    """Keep only passing item rows in a newly copied generation; never touch source DBs."""
    selected_items = set(eligible_items)
    if not selected_items or len(selected_items) != len(eligible_items):
        raise ValueError("Partial snapshot requires a nonempty unique eligible item list")
    before_ids, before_attachments = _chroma_rows(chroma_path, collection_name, selected_items)
    lexical_ids = _filter_lexical(lexical_path, selected_items)
    if before_ids != lexical_ids:
        raise ValueError("Partial snapshot Chroma and lexical IDs differ after filtering")
    _filter_relations(relations_path, selected_items, before_attachments)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise TypeError("Partial snapshot manifest has no file inventory")
    manifest["files"] = {key: value for key, value in files.items() if key in before_attachments}
    manifest["partial_publication"] = True
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"chunks": len(before_ids), "attachments": len(before_attachments),
            "eligible_items": len(selected_items)}
