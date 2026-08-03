"""Read paragraph chunks from Chroma's SQLite metadata store without loading HNSW."""
from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from typing import Any

try:
    from .v3_data_plane import collection_name as v3_collection_name
except ImportError:
    from v3_data_plane import collection_name as v3_collection_name

ROOT = Path(__file__).resolve().parents[1]
# .expanduser(): v3_data_plane.resolve_configured_path is "the one place this
# resolution rule is written" (its own docstring) precisely because this used
# to be a second, independent copy that missed .expanduser() -- a ~-prefixed
# CHROMA_DIR would silently resolve to a literal "~" subdirectory here while
# every other V3 entry point expanded it correctly (found in code review,
# fixed 2026-07-30). Callers that already resolved CHROMA_DIR themselves
# (e.g. via v3_data_plane.chroma_dir()/enforce_environment()) should still
# pass chroma_dir= explicitly rather than rely on this default, since this
# module has no way to know a caller's working directory differs from ROOT.
DEFAULT_CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", ROOT / "data" / "chroma")).expanduser()


def natural_chunk_key(chunk_id: str) -> list[Any]:
    return [int(token) if token.isdigit() else token for token in re.split(r"(\d+)", chunk_id)]


def active_collection_name(chroma_dir: Path = DEFAULT_CHROMA_DIR) -> str | None:
    del chroma_dir  # the active collection is no longer inferred from database contents
    return v3_collection_name()


def _db_path(chroma_dir: Path) -> Path:
    return chroma_dir / "chroma.sqlite3"


def list_item_keys(
    *, chroma_dir: Path = DEFAULT_CHROMA_DIR, collection_name: str | None = None,
) -> list[str]:
    db_path = _db_path(chroma_dir)
    name = collection_name or active_collection_name(chroma_dir)
    if not db_path.exists() or not name:
        return []
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
    try:
        rows = connection.execute('''
            SELECT DISTINCT item.string_value
            FROM collections c JOIN segments s ON s.collection = c.id AND s.scope = 'METADATA'
            JOIN embeddings e ON e.segment_id = s.id
            JOIN embedding_metadata item ON item.id = e.id AND item.key = 'itemKey'
            WHERE c.name = ? AND item.string_value IS NOT NULL
            ORDER BY item.string_value
        ''', (name,)).fetchall()
        return [str(row[0]) for row in rows]
    finally:
        connection.close()


def list_chunk_ids(
    *, chroma_dir: Path = DEFAULT_CHROMA_DIR, collection_name: str | None = None,
) -> list[str]:
    """Return every chunk ID for one collection without loading HNSW."""
    db_path = _db_path(chroma_dir)
    name = collection_name or active_collection_name(chroma_dir)
    if not db_path.exists() or not name:
        return []
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    try:
        rows = connection.execute('''
            SELECT e.embedding_id
            FROM collections c JOIN segments s ON s.collection = c.id AND s.scope = 'METADATA'
            JOIN embeddings e ON e.segment_id = s.id
            WHERE c.name = ? ORDER BY e.embedding_id
        ''', (name,)).fetchall()
        return [str(row[0]) for row in rows]
    finally:
        connection.close()


def list_attachment_keys(
    *, chroma_dir: Path = DEFAULT_CHROMA_DIR, collection_name: str | None = None,
) -> list[str]:
    """Return non-empty attachment keys present in one collection."""
    db_path = _db_path(chroma_dir)
    name = collection_name or active_collection_name(chroma_dir)
    if not db_path.exists() or not name:
        return []
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    try:
        rows = connection.execute('''
            SELECT DISTINCT attachment.string_value
            FROM collections c JOIN segments s ON s.collection = c.id AND s.scope = 'METADATA'
            JOIN embeddings e ON e.segment_id = s.id
            JOIN embedding_metadata attachment ON attachment.id = e.id
                AND attachment.key = 'attachmentKey'
            WHERE c.name = ? AND COALESCE(attachment.string_value, '') <> ''
            ORDER BY attachment.string_value
        ''', (name,)).fetchall()
        return [str(row[0]) for row in rows]
    finally:
        connection.close()


def list_chunk_ids_without_item(
    *, chroma_dir: Path = DEFAULT_CHROMA_DIR, collection_name: str | None = None,
) -> list[str]:
    """Chunk ids that have no non-empty itemKey.

    audit_v3_cutover.py's per-item comparison iterates legacy item keys, so a
    chunk with no itemKey is never examined by it -- 1,774 such chunks
    answered vector search while every gate passed (2026-07-28). A LEFT JOIN
    against embedding_metadata is required rather than a WHERE on that table
    directly: a chunk missing the itemKey key entirely has no row there at
    all, not a row with an empty string, so an inner join would silently skip
    it same as the per-item audit does.
    """
    db_path = _db_path(chroma_dir)
    name = collection_name or active_collection_name(chroma_dir)
    if not db_path.exists() or not name:
        return []
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    try:
        rows = connection.execute('''
            SELECT e.embedding_id
            FROM collections c JOIN segments s ON s.collection = c.id AND s.scope = 'METADATA'
            JOIN embeddings e ON e.segment_id = s.id
            LEFT JOIN embedding_metadata item ON item.id = e.id AND item.key = 'itemKey'
            WHERE c.name = ? AND COALESCE(item.string_value, '') = ''
            ORDER BY e.embedding_id
        ''', (name,)).fetchall()
        return [str(row[0]) for row in rows]
    finally:
        connection.close()


def list_chunk_ids_without_attachment(
    *, chroma_dir: Path = DEFAULT_CHROMA_DIR, collection_name: str | None = None,
) -> list[str]:
    """Return chunks whose attachmentKey metadata is absent or empty.

    Zotero notes are written with ``attachmentKey=None``/``noteKey=<key>`` by
    design (note_extract.py) and are excluded from the canonical document
    tree elsewhere (orphan_cleanup.note_only_item) -- they are not attachment
    chunks that lost their key, so they must not be counted here either
    (2026-08-03).
    """
    db_path = _db_path(chroma_dir)
    name = collection_name or active_collection_name(chroma_dir)
    if not db_path.exists() or not name:
        return []
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    try:
        rows = connection.execute('''
            SELECT e.embedding_id
            FROM collections c JOIN segments s ON s.collection = c.id AND s.scope = 'METADATA'
            JOIN embeddings e ON e.segment_id = s.id
            LEFT JOIN embedding_metadata attachment
                ON attachment.id = e.id AND attachment.key = 'attachmentKey'
            LEFT JOIN embedding_metadata source_type
                ON source_type.id = e.id AND source_type.key = 'source_type'
            WHERE c.name = ? AND COALESCE(attachment.string_value, '') = ''
                AND COALESCE(source_type.string_value, '') <> 'note'
            ORDER BY e.embedding_id
        ''', (name,)).fetchall()
        return [str(row[0]) for row in rows]
    finally:
        connection.close()


def get_item_chunks(
    item_key: str,
    *,
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
    collection_name: str | None = None,
) -> list[dict[str, Any]]:
    """Return one item's chunks in document order with all stored metadata."""
    db_path = _db_path(chroma_dir)
    name = collection_name or active_collection_name(chroma_dir)
    if not item_key or not db_path.exists() or not name:
        return []
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute('''
            SELECT e.embedding_id, em.key, em.string_value, em.int_value,
                   em.float_value, em.bool_value
            FROM collections c JOIN segments s ON s.collection = c.id AND s.scope = 'METADATA'
            JOIN embeddings e ON e.segment_id = s.id
            JOIN embedding_metadata item ON item.id = e.id
                AND item.key = 'itemKey' AND item.string_value = ?
            JOIN embedding_metadata em ON em.id = e.id
            WHERE c.name = ?
        ''', (item_key, name)).fetchall()
    finally:
        connection.close()

    chunks: dict[str, dict[str, Any]] = {}
    for row in rows:
        chunk = chunks.setdefault(row["embedding_id"], {"id": row["embedding_id"], "text": "", "metadata": {}})
        if row["string_value"] is not None:
            value: Any = row["string_value"]
        elif row["int_value"] is not None:
            value = row["int_value"]
        elif row["float_value"] is not None:
            value = row["float_value"]
        else:
            value = bool(row["bool_value"]) if row["bool_value"] is not None else None
        if row["key"] == "chroma:document":
            chunk["text"] = value or ""
        else:
            chunk["metadata"][row["key"]] = value
    return sorted(chunks.values(), key=lambda chunk: natural_chunk_key(chunk["id"]))


def get_collection_chunks(
    *,
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
    collection_name: str | None = None,
) -> list[dict[str, Any]]:
    """Return every chunk in one collection with one SQLite scan."""
    db_path = _db_path(chroma_dir)
    name = collection_name or active_collection_name(chroma_dir)
    if not db_path.exists() or not name:
        return []
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute('''
            SELECT e.embedding_id, em.key, em.string_value, em.int_value,
                   em.float_value, em.bool_value
            FROM collections c JOIN segments s ON s.collection = c.id AND s.scope = 'METADATA'
            JOIN embeddings e ON e.segment_id = s.id
            JOIN embedding_metadata em ON em.id = e.id
            WHERE c.name = ?
            ORDER BY e.embedding_id
        ''', (name,)).fetchall()
    finally:
        connection.close()

    chunks: dict[str, dict[str, Any]] = {}
    for row in rows:
        chunk = chunks.setdefault(
            row["embedding_id"],
            {"id": row["embedding_id"], "text": "", "metadata": {}},
        )
        if row["string_value"] is not None:
            value: Any = row["string_value"]
        elif row["int_value"] is not None:
            value = row["int_value"]
        elif row["float_value"] is not None:
            value = row["float_value"]
        else:
            value = bool(row["bool_value"]) if row["bool_value"] is not None else None
        if row["key"] == "chroma:document":
            chunk["text"] = value or ""
        else:
            chunk["metadata"][row["key"]] = value
    return list(chunks.values())


def get_item_text(item_key: str, *, max_chars: int | None = None, **kwargs: Any) -> str:
    text = "\n\n".join(chunk["text"] for chunk in get_item_chunks(item_key, **kwargs) if chunk["text"])
    return text[:max_chars] if max_chars is not None else text


__all__ = [
    "active_collection_name", "get_collection_chunks", "get_item_chunks", "get_item_text",
    "list_attachment_keys", "list_chunk_ids", "list_chunk_ids_without_attachment",
    "list_chunk_ids_without_item", "list_item_keys", "natural_chunk_key",
]
