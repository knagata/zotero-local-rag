"""SQLite FTS5 trigram index synchronized with Chroma paragraph chunks."""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Any, Iterable

try:
    from .v3_data_plane import lexical_path as v3_lexical_path
except ImportError:
    from v3_data_plane import lexical_path as v3_lexical_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "lexical_v3.sqlite3"


def _path(path: Path | None = None) -> Path:
    if path is not None:
        return path
    return v3_lexical_path(PROJECT_ROOT)


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
    for chunk_id, body, metadata in zip(ids, documents, metadatas, strict=False):
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


def delete_by_chunk_ids(ids: Iterable[str], *, path: Path | None = None) -> int:
    """Delete exact chunk IDs (used by deterministic index repair).

    Returns the row count. A caller in purge_orphans.py added the old ``None``
    return to a running total with ``or 0``, so its reported fts-rows-removed
    count was always zero regardless of what was actually deleted -- harmless
    to the deletion itself, but a report that always lies about one of its own
    numbers (2026-07-28, found in code review).
    """
    values = [(str(chunk_id),) for chunk_id in set(ids) if chunk_id]
    if not values:
        return 0
    connection = _connect(path)
    try:
        cursor = connection.executemany("DELETE FROM chunks_fts WHERE chunk_id = ?", values)
        connection.commit()
        return cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else len(values)
    finally:
        connection.close()


def list_chunk_ids(*, path: Path | None = None) -> list[str]:
    """Return all lexical chunk IDs through a strictly read-only connection."""
    db_path = _path(path)
    if not db_path.exists():
        return []
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    try:
        rows = connection.execute("SELECT chunk_id FROM chunks_fts ORDER BY chunk_id").fetchall()
        return [str(row[0]) for row in rows]
    finally:
        connection.close()


def chunk_ids_by_attachment_keys(
    keys: Iterable[str], *, path: Path | None = None,
) -> dict[str, set[str]]:
    """Read the exact lexical ID set for selected attachments."""
    selected = sorted({str(key) for key in keys if key})
    result = {key: set() for key in selected}
    if not selected or not _path(path).exists():
        return result
    placeholders = ",".join("?" for _ in selected)
    connection = sqlite3.connect(f"file:{_path(path)}?mode=ro", uri=True, timeout=30)
    try:
        rows = connection.execute(
            f"SELECT attachment_key, chunk_id FROM chunks_fts WHERE attachment_key IN ({placeholders})",
            selected,
        ).fetchall()
        for attachment_key, chunk_id in rows:
            result[str(attachment_key)].add(str(chunk_id))
        return result
    finally:
        connection.close()


#: FTS5's trigram tokenizer indexes three-character sequences, so a shorter
#: token has no trigram to match and cannot be found through MATCH at all.
TRIGRAM_MIN_CHARS = 3


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
    # A trigram index cannot represent a token shorter than three characters,
    # so such a token matches nothing -- and ANDing it into the MATCH made the
    # whole query return nothing. The length test used to be applied to the
    # entire string, so "移民" found rows and "移民 国家" found none, as did
    # "landscape of power" and "a landscape": one short word anywhere silenced
    # the search (2026-07-28). Two-character words carry meaning in Japanese
    # and English articles and prepositions are unavoidable, so multi-word
    # search was broken in both languages.
    #
    # Short tokens are therefore matched with LIKE and long ones with MATCH,
    # in the same conjunction. Dropping them instead would silently widen the
    # query beyond what was asked for.
    tokens = query.split()
    indexable = [token for token in tokens if len(token) >= TRIGRAM_MIN_CHARS]
    short_tokens = [token for token in tokens if len(token) < TRIGRAM_MIN_CHARS]
    where = []
    params: list[Any] = []
    if indexable:
        # Whitespace-separated terms stay independent operands: quoting the
        # whole query would demand adjacency and collapse recall.
        where.append("chunks_fts MATCH ?")
        params.append(" AND ".join(
            '"' + token.replace('"', '""') + '"' for token in indexable
        ))
    for token in short_tokens:
        escaped = token.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{escaped}%"
        where.append(
            "(body LIKE ? ESCAPE '\\' OR title LIKE ? ESCAPE '\\' OR creators LIKE ? ESCAPE '\\')"
        )
        params.extend([pattern, pattern, pattern])
    if not where:
        return []
    short_query = not indexable
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
