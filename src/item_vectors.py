"""Shared item-level semantic vectors derived from Chroma chunk embeddings."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .v3_data_plane import (
        chroma_dir as v3_chroma_dir, collection_name as v3_collection_name,
    )
except ImportError:
    from v3_data_plane import (
        chroma_dir as v3_chroma_dir, collection_name as v3_collection_name,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_PATH = PROJECT_ROOT / "data" / "item_vectors_cache.json"
_META_KEY = "__meta__"


def _open_collection():
    import chromadb

    directory = v3_chroma_dir(PROJECT_ROOT)
    client = chromadb.PersistentClient(path=str(directory))
    return client, client.get_collection(v3_collection_name())


def get_item_vectors(
    item_keys: list[str], *, cache_path: Path | None = None
) -> dict[str, list[float]]:
    """Return normalized means of each item's chunk embeddings.

    Unknown items are computed incrementally. A collection or embedding-dimension
    change invalidates the whole cache. Individual malformed vectors are ignored.

    A per-item entry also carries the chunk count Chroma reported when it was
    computed ("n"). Re-ingesting one item (e.g. via ``--force-reparse``) does
    not change the collection name or dimension, so the wholesale invalidation
    above never fires for it -- without this per-item check the stale vector
    would be served indefinitely and ``related_items()`` would silently never
    reflect the re-extraction (found in code review, fixed 2026-07-30).
    """
    requested = list(dict.fromkeys(key for key in item_keys if key))
    if not requested:
        return {}
    cache_path = cache_path or DEFAULT_CACHE_PATH
    vectors: dict[str, list[float]] = {}
    chunk_counts: dict[str, int] = {}
    metadata: dict[str, Any] = {}
    if cache_path.exists():
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                metadata = raw.get(_META_KEY, {}) if isinstance(raw.get(_META_KEY), dict) else {}
                for key, value in raw.items():
                    if key == _META_KEY:
                        continue
                    if isinstance(value, list):
                        # Legacy entries (written before per-item staleness
                        # tracking) carry no chunk count -- leave them out of
                        # chunk_counts so the freshness check below always
                        # recomputes them once, migrating them to the new
                        # format.
                        vectors[key] = value
                    elif (
                        isinstance(value, dict)
                        and isinstance(value.get("v"), list)
                        and isinstance(value.get("n"), int)
                    ):
                        vectors[key] = value["v"]
                        chunk_counts[key] = value["n"]
        except (OSError, json.JSONDecodeError):
            vectors = {}
            chunk_counts = {}

    client = None
    try:
        client, collection = _open_collection()
        peek = collection.peek(1)
        embeddings = peek.get("embeddings") if peek else None
        current_dim = None
        if embeddings is not None and len(embeddings):
            current_dim = int(np.asarray(embeddings).shape[-1])
        collection_name = collection.name
        if (
            metadata.get("embedding_dim") not in (None, current_dim)
            or metadata.get("collection") not in (None, collection_name)
        ):
            vectors = {}
            chunk_counts = {}

        stale_keys: set[str] = set()
        for key in (key for key in requested if key in vectors):
            try:
                current_count = collection.get(where={"itemKey": key}, include=[])
                current_count = len(current_count.get("ids") or [])
            except Exception:
                continue
            if chunk_counts.get(key) != current_count:
                stale_keys.add(key)
        for key in stale_keys:
            vectors.pop(key, None)
            chunk_counts.pop(key, None)

        for key in (key for key in requested if key not in vectors):
            try:
                result = collection.get(where={"itemKey": key}, include=["embeddings"])
                item_embeddings = result.get("embeddings")
                if item_embeddings is None or len(item_embeddings) == 0:
                    continue
                array = np.asarray(item_embeddings, dtype=np.float32)
                array = array[np.isfinite(array).all(axis=1)]
                if not len(array):
                    continue
                vector = array.mean(axis=0)
                norm = float(np.linalg.norm(vector))
                if norm <= 0:
                    continue
                vector /= norm
                vectors[key] = [round(float(value), 6) for value in vector]
                chunk_counts[key] = len(item_embeddings)
                current_dim = current_dim or len(vector)
            except Exception:
                continue

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            _META_KEY: {
                "embedding_dim": current_dim,
                "collection": collection_name,
            },
            **{
                key: {"v": value, "n": chunk_counts[key]}
                for key, value in vectors.items()
                if key in chunk_counts
            },
        }
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(cache_path)
    except Exception:
        pass
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
    return {key: vectors[key] for key in requested if key in vectors}


def get_item_meta(
    item_keys: list[str], *, chroma_db: Path | None = None
) -> dict[str, dict[str, Any]]:
    """Read one title/creator/year tuple per item directly from Chroma SQLite."""
    requested = list(dict.fromkeys(key for key in item_keys if key))
    if not requested:
        return {}
    chroma_db = chroma_db or Path(
        v3_chroma_dir(PROJECT_ROOT)
    ) / "chroma.sqlite3"
    if not chroma_db.exists():
        return {}
    placeholders = ",".join("?" for _ in requested)
    try:
        connection = sqlite3.connect(str(chroma_db), timeout=5)
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            f"""
            SELECT ikey.string_value AS item_key,
                   MAX(CASE WHEN em.key = 'title' THEN em.string_value END) AS title,
                   MAX(CASE WHEN em.key = 'creators' THEN em.string_value END) AS creators,
                   COALESCE(
                       MAX(CASE WHEN em.key = 'year' THEN em.int_value END),
                       MAX(CASE WHEN em.key = 'year' THEN em.string_value END)
                   ) AS year
            FROM collections c
            JOIN segments s ON s.collection = c.id AND s.scope = 'METADATA'
            JOIN embeddings e ON e.segment_id = s.id
            JOIN embedding_metadata ikey ON ikey.id = e.id
            JOIN embedding_metadata em ON em.id = ikey.id
            WHERE c.name = ? AND ikey.key = 'itemKey'
              AND ikey.string_value IN ({placeholders})
            GROUP BY ikey.string_value
            """,
            [v3_collection_name(), *requested],
        ).fetchall()
        connection.close()
        return {
            row["item_key"]: {
                "title": row["title"] or "",
                "creators": row["creators"] or "",
                "year": row["year"],
            }
            for row in rows
        }
    except (sqlite3.Error, OSError):
        return {}


__all__ = ["get_item_meta", "get_item_vectors"]
