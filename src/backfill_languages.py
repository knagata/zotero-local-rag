"""Add deterministic language metadata to an existing Chroma collection in place."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import chromadb

try:
    from .chunk_store import active_collection_name
    from .lexical_index import rebuild_from_chroma
    from .text_utils import detect_lang
    from .v3_data_plane import chroma_dir
except ImportError:  # pragma: no cover
    from chunk_store import active_collection_name
    from lexical_index import rebuild_from_chroma
    from text_utils import detect_lang
    from v3_data_plane import chroma_dir


ROOT = Path(__file__).resolve().parents[1]
# Asked of v3_data_plane rather than read from the environment here: it
# owns both the default location and the rule that expands ~ and resolves
# a relative value against the project root. Copies of that rule have
# drifted twice, each time leaving one configuration naming two databases.
CHROMA_DIR = chroma_dir(ROOT)


def annotate_batch(
    documents: list[str], metadatas: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    counts: dict[str, int] = {"ja": 0, "zh": 0, "en": 0, "other": 0}
    updated: list[dict[str, Any]] = []
    for index, document in enumerate(documents):
        metadata = dict(metadatas[index]) if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
        lang = detect_lang(document, metadata.get("language"))
        metadata["lang"] = lang
        counts[lang] += 1
        updated.append(metadata)
    return updated, counts


def backfill_languages(*, batch_size: int = 1000, sync_lexical: bool = True) -> dict[str, int]:
    collection_name = active_collection_name(CHROMA_DIR)
    if not collection_name:
        raise RuntimeError("No active paragraph collection was found.")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(collection_name)
    totals = {"ja": 0, "zh": 0, "en": 0, "other": 0, "chunks": 0}
    try:
        count = collection.count()
        for offset in range(0, count, batch_size):
            result = collection.get(
                limit=batch_size, offset=offset, include=["documents", "metadatas"],
            )
            ids = result.get("ids") or []
            documents = result.get("documents") or []
            metadatas, counts = annotate_batch(documents, result.get("metadatas") or [])
            if ids:
                # Full metadata dictionaries are supplied so existing fields are preserved.
                collection.update(ids=ids, metadatas=metadatas)
            totals["chunks"] += len(ids)
            for lang in ("ja", "zh", "en", "other"):
                totals[lang] += counts[lang]
            print(f"Language metadata: {totals['chunks']:,}/{count:,}", flush=True)
    finally:
        client.close()
    if sync_lexical:
        totals["lexical_chunks"] = rebuild_from_chroma()
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    totals = backfill_languages(batch_size=args.batch_size)
    print(
        "Updated {chunks:,} chunks (ja={ja:,}, zh={zh:,}, en={en:,}, other={other:,}).".format(**totals)
    )


if __name__ == "__main__":
    main()
