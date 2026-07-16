"""Read paragraph chunks from Chroma's SQLite metadata store without loading HNSW."""
from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", ROOT / "data" / "chroma"))


def natural_chunk_key(chunk_id: str) -> list[Any]:
    return [int(token) if token.isdigit() else token for token in re.split(r"(\d+)", chunk_id)]


def active_collection_name(chroma_dir: Path = DEFAULT_CHROMA_DIR) -> str | None:
    explicit = (os.environ.get("CHROMA_COLLECTION") or "").strip()
    if explicit:
        return explicit
    try:
        config = json.loads((chroma_dir / "embedder_config.json").read_text(encoding="utf-8"))
        name = str(config.get("collection") or "").strip()
        if name:
            return name
    except (OSError, ValueError, TypeError):
        pass
    db_path = chroma_dir / "chroma.sqlite3"
    if not db_path.exists():
        return None
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    try:
        row = connection.execute('''
            SELECT c.name, COUNT(e.id) AS n FROM collections c
            LEFT JOIN segments s ON s.collection = c.id AND s.scope = 'METADATA'
            LEFT JOIN embeddings e ON e.segment_id = s.id
            WHERE c.name NOT LIKE '%\_\_sum\_%' ESCAPE '\'
              AND c.name NOT LIKE '%\_\_cases' ESCAPE '\'
            GROUP BY c.id ORDER BY n DESC, c.name LIMIT 1
        ''').fetchone()
        return str(row[0]) if row else None
    finally:
        connection.close()


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


def get_item_text(item_key: str, *, max_chars: int | None = None, **kwargs: Any) -> str:
    text = "\n\n".join(chunk["text"] for chunk in get_item_chunks(item_key, **kwargs) if chunk["text"])
    return text[:max_chars] if max_chars is not None else text


__all__ = ["active_collection_name", "get_item_chunks", "get_item_text", "list_item_keys", "natural_chunk_key"]
