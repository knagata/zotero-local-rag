"""SQLite FTS5 trigram index synchronized with Chroma paragraph chunks."""
from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "lexical.sqlite3"


def _path(path: Path | None = None) -> Path:
    return path or Path(os.environ.get("LEXICAL_DB_PATH", DEFAULT_DB_PATH))


def _connect(path: Path | None = None) -> sqlite3.Connection:
    db_path = _path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path), timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            chunk_id UNINDEXED,
            item_key UNINDEXED,
            lang UNINDEXED,
            source_type UNINDEXED,
            attachment_key UNINDEXED,
            note_key UNINDEXED,
            body,
            title,
            creators,
            tokenize = 'trigram'
        )
    ''')
    return connection


def upsert_chunks(
    ids: list[str], documents: list[str], metadatas: list[dict[str, Any]],
    *, path: Path | None = None, replace: bool = True,
) -> None:
    """Replace FTS rows for the supplied Chroma chunk IDs."""
    rows = []
    for chunk_id, body, metadata in zip(ids, documents, metadatas):
        md = metadata if isinstance(metadata, dict) else {}
        rows.append((
            chunk_id,
            md.get("itemKey") or "",
            md.get("lang") or "other",
            md.get("source_type") or "",
            md.get("attachmentKey") or "",
            md.get("noteKey") or "",
            body or "",
            md.get("title") or "",
            md.get("creators") or "",
        ))
    if not rows:
        return
    connection = _connect(path)
    try:
        if replace:
            connection.executemany("DELETE FROM chunks_fts WHERE chunk_id = ?", [(row[0],) for row in rows])
        connection.executemany(
            """INSERT INTO chunks_fts
               (chunk_id, item_key, lang, source_type, attachment_key, note_key,
                body, title, creators) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        connection.commit()
    finally:
        connection.close()


def delete_by_attachment_keys(keys: Iterable[str], *, path: Path | None = None) -> None:
    values = [(key,) for key in set(keys) if key]
    if not values:
        return
    connection = _connect(path)
    try:
        connection.executemany("DELETE FROM chunks_fts WHERE attachment_key = ?", values)
        connection.commit()
    finally:
        connection.close()


def delete_by_note_key(note_key: str, *, path: Path | None = None) -> None:
    if not note_key:
        return
    connection = _connect(path)
    try:
        connection.execute("DELETE FROM chunks_fts WHERE note_key = ?", (note_key,))
        connection.commit()
    finally:
        connection.close()


def search_chunks(
    query: str,
    *,
    k: int = 25,
    include_notes: bool = False,
    item_keys: list[str] | None = None,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return lexical matches ordered by FTS5 BM25 (lower score is better)."""
    query = " ".join((query or "").split()).strip()
    if not query or k <= 0 or not _path(path).exists():
        return []
    short_query = len(query) < 3
    if short_query:
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{escaped}%"
        where = ["(body LIKE ? ESCAPE '\\' OR title LIKE ? ESCAPE '\\' OR creators LIKE ? ESCAPE '\\')"]
        params: list[Any] = [pattern, pattern, pattern]
    else:
        phrase = '"' + query.replace('"', '""') + '"'
        where = ["chunks_fts MATCH ?"]
        params = [phrase]
    if not include_notes:
        where.append("source_type <> 'note'")
    if item_keys:
        placeholders = ",".join("?" for _ in item_keys)
        where.append(f"item_key IN ({placeholders})")
        params.extend(item_keys)
    params.append(k)
    connection = _connect(path)
    try:
        score_expression = "0.0" if short_query else "bm25(chunks_fts)"
        rows = connection.execute(
            f"""SELECT chunk_id, item_key, lang, source_type, {score_expression} AS bm25_score
                FROM chunks_fts WHERE {' AND '.join(where)}
                ORDER BY bm25_score ASC, chunk_id ASC LIMIT ?""",
            params,
        ).fetchall()
        return [dict(row) for row in rows]
    except sqlite3.OperationalError:
        return []
    finally:
        connection.close()


def rebuild_from_chroma(*, path: Path | None = None, batch_size: int = 5000) -> int:
    """Rebuild the lexical DB from the active Chroma collection."""
    try:
        from .item_vectors import _open_collection
    except ImportError:  # pragma: no cover
        from item_vectors import _open_collection

    db_path = _path(path)
    if db_path.exists():
        db_path.unlink()
    client, collection = _open_collection()
    total = 0
    try:
        count = collection.count()
        for offset in range(0, count, batch_size):
            result = collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            ids = result.get("ids") or []
            documents = result.get("documents") or []
            metadatas = result.get("metadatas") or []
            upsert_chunks(ids, documents, metadatas, path=db_path, replace=False)
            total += len(ids)
    finally:
        client.close()
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage the lexical FTS5 chunk index.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild from the active Chroma collection.")
    args = parser.parse_args()
    if not args.rebuild:
        parser.error("--rebuild is required")
    print(f"Rebuilt lexical index: {rebuild_from_chroma():,} chunks")


if __name__ == "__main__":
    main()
