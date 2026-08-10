#!/usr/bin/env python3
"""Create and verify a rollback snapshot of the current V3 generation.

The snapshot is taken while holding the same indexing lock as ingestion. Chroma
is copied as one quiescent directory; SQLite-backed ledgers use SQLite's online
backup API so WAL state is included in a consistent destination database.
Nothing in the active generation is modified.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import indexing_lock  # noqa: E402
from src.env_utils import load_dotenv_native  # noqa: E402
from src.v3_data_plane import (  # noqa: E402
    V3_COLLECTION,
    chroma_dir,
    enforce_environment,
    lexical_path,
    manifest_path,
    pipeline_config_path,
    resolve_configured_path,
)


@dataclass(frozen=True)
class GenerationSources:
    chroma: Path
    manifest: Path
    lexical: Path
    relations: Path
    pipeline_config: Path
    lock: Path


def resolve_sources(root: Path = ROOT) -> GenerationSources:
    load_dotenv_native(root)
    enforce_environment(root)
    return GenerationSources(
        chroma=chroma_dir(root),
        manifest=manifest_path(root),
        lexical=lexical_path(root),
        relations=resolve_configured_path(
            root, os.environ.get("RELATIONS_DB_PATH", root / "data" / "relations.db")
        ),
        pipeline_config=pipeline_config_path(root),
        lock=indexing_lock.default_path(root),
    )


def _directory_bytes(path: Path) -> int:
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def _sqlite_read(path: Path, sql: str) -> Any:
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=30)
    try:
        return connection.execute(sql).fetchone()[0]
    finally:
        connection.close()


def _backup_sqlite(source: Path, destination: Path) -> None:
    source_connection = sqlite3.connect(
        f"{source.resolve().as_uri()}?mode=ro", uri=True, timeout=30
    )
    destination_connection = sqlite3.connect(destination)
    try:
        source_connection.backup(destination_connection)
    finally:
        destination_connection.close()
        source_connection.close()


def _chroma_count(path: Path) -> int:
    import chromadb

    client = chromadb.PersistentClient(path=str(path))
    try:
        collection = client.get_collection(V3_COLLECTION)
        count = collection.count()
        if count:
            sample = collection.peek(limit=1)
            if len(sample.get("ids") or []) != 1:
                raise RuntimeError("Chroma count is nonzero but a one-row read failed")
        return count
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def verify_backup(destination: Path, *, _during_creation: bool = False) -> dict[str, Any]:
    if (destination / "INCOMPLETE").exists() and not _during_creation:
        raise RuntimeError(f"backup is marked incomplete: {destination}")
    paths = {
        "chroma": destination / "chroma",
        "manifest": destination / "manifest_v3.json",
        "lexical": destination / "lexical_v3.sqlite3",
        "relations": destination / "relations.db",
        "pipeline": destination / "chroma" / "embedder_config_v3.json",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"backup is incomplete; missing: {', '.join(missing)}")

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    pipeline = json.loads(paths["pipeline"].read_text(encoding="utf-8"))
    embedder = pipeline.get("embedder") if isinstance(pipeline.get("embedder"), dict) else {}
    model_value = str(embedder.get("emb_model") or "")
    lexical_quick_check = _sqlite_read(paths["lexical"], "PRAGMA quick_check")
    relations_quick_check = _sqlite_read(paths["relations"], "PRAGMA quick_check")
    if lexical_quick_check != "ok" or relations_quick_check != "ok":
        raise RuntimeError(
            "backup SQLite verification failed: "
            f"lexical={lexical_quick_check!r}, relations={relations_quick_check!r}"
        )

    return {
        "verified": True,
        "collection": V3_COLLECTION,
        "chroma_count": _chroma_count(paths["chroma"]),
        "lexical_count": int(_sqlite_read(paths["lexical"], "SELECT COUNT(*) FROM chunks_fts")),
        "manifest_attachments": len(manifest.get("files") or {}),
        "pipeline": {
            "model_name": Path(model_value).name if model_value else None,
            "dimension": pipeline.get("embedding_dim") or embedder.get("embedding_dim"),
            "normalize_embeddings": pipeline.get("normalize_embeddings"),
            "model_state_fingerprint": pipeline.get("model_state_fingerprint"),
        },
        "lexical_quick_check": lexical_quick_check,
        "relations_quick_check": relations_quick_check,
        "backup_bytes": _directory_bytes(destination),
    }


def backup_generation(
    destination: Path,
    *,
    root: Path = ROOT,
    sources: GenerationSources | None = None,
) -> dict[str, Any]:
    source = sources or resolve_sources(root)
    destination_resolved = destination.expanduser().resolve()
    source_chroma_resolved = source.chroma.expanduser().resolve()
    if (
        destination_resolved == source_chroma_resolved
        or source_chroma_resolved in destination_resolved.parents
    ):
        raise ValueError(
            "backup destination must be outside the source Chroma directory: "
            f"{destination_resolved}"
        )
    if destination.exists():
        raise FileExistsError(f"backup destination already exists: {destination}")
    required = (source.chroma, source.manifest, source.lexical, source.relations)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"active V3 generation is incomplete: {missing}")
    if not source.pipeline_config.exists():
        raise FileNotFoundError(f"pipeline config is missing: {source.pipeline_config}")

    with indexing_lock.held(source.lock):
        source_chroma_count = _chroma_count(source.chroma)
        source_lexical_count = int(
            _sqlite_read(source.lexical, "SELECT COUNT(*) FROM chunks_fts")
        )
        source_bytes = (
            _directory_bytes(source.chroma)
            + source.manifest.stat().st_size
            + source.lexical.stat().st_size
            + source.relations.stat().st_size
        )
        destination.mkdir(parents=True, exist_ok=False)
        incomplete = destination / "INCOMPLETE"
        incomplete.write_text("backup has not completed\n", encoding="utf-8")
        shutil.copytree(source.chroma, destination / "chroma", copy_function=shutil.copy2)
        shutil.copy2(source.manifest, destination / "manifest_v3.json")
        _backup_sqlite(source.lexical, destination / "lexical_v3.sqlite3")
        _backup_sqlite(source.relations, destination / "relations.db")

        verification = verify_backup(destination, _during_creation=True)
        if verification["chroma_count"] != source_chroma_count:
            raise RuntimeError(
                "Chroma count changed in backup: "
                f"{source_chroma_count} -> {verification['chroma_count']}"
            )
        if verification["lexical_count"] != source_lexical_count:
            raise RuntimeError(
                "lexical count changed in backup: "
                f"{source_lexical_count} -> {verification['lexical_count']}"
            )

        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True,
        ).stdout.strip()
        report = {
            **verification,
            "source_commit": commit,
            "source_bytes": source_bytes,
            "destination": str(destination),
        }
        (destination / "backup-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        incomplete.unlink()
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--verify-only", action="store_true", help="Read and verify an existing snapshot."
    )
    args = parser.parse_args()

    report = verify_backup(args.destination) if args.verify_only else backup_generation(
        args.destination
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
