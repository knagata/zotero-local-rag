#!/usr/bin/env python3
"""Read-only Phase 6 audit for the legacy RAG stores.

The V3 data plane is the only active one.  This program intentionally has no
retire/delete switch: it records the old data plane's observable state and can
compare a later maintenance run against that baseline.  A Phase 6 deletion is
allowed only after this audit reports no legacy writes and the separate backup
restore check succeeds.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
LEGACY_TABLES = ("item_summaries", "section_summaries", "insight_generation_status")
LEGACY_COLLECTIONS = (
    "zotero_paragraphs",
    "zotero_paragraphs__sum_item",
    "zotero_paragraphs__sum_section",
)


def _sha256_bytes(parts: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return f"sha256:{digest.hexdigest()}"


def file_fingerprint(path: Path, *, hash_contents: bool = True) -> dict[str, Any]:
    """Return a non-mutating file identity, or an explicit missing marker."""
    if not path.exists():
        return {"exists": False}
    stat = path.stat()
    result: dict[str, Any] = {
        "exists": True,
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if hash_contents:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result["sha256"] = f"sha256:{digest.hexdigest()}"
    return result


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,),
    ).fetchone() is not None


def _sqlite_table_snapshot(conn: sqlite3.Connection, table: str) -> dict[str, Any]:
    if not _table_exists(conn, table):
        return {"exists": False, "count": 0, "fingerprint": "sha256:"}
    columns = [str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')]
    has_updated_at = "updated_at" in columns
    rows = conn.execute(f'SELECT * FROM "{table}" ORDER BY rowid').fetchall()
    fingerprint = _sha256_bytes(
        (json.dumps(dict(row), ensure_ascii=False, sort_keys=True, default=str).encode("utf-8") + b"\n")
        for row in rows
    )
    result: dict[str, Any] = {
        "exists": True,
        "count": len(rows),
        "fingerprint": fingerprint,
        "schema": columns,
    }
    if has_updated_at:
        result["last_updated_at"] = conn.execute(
            f'SELECT MAX(updated_at) FROM "{table}"',
        ).fetchone()[0]
    return result


def sqlite_snapshot(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        return {"exists": False, "tables": {name: {"exists": False, "count": 0} for name in LEGACY_TABLES}}
    # mode=ro ensures a status audit cannot initialize or migrate relations.db.
    conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return {
            "exists": True,
            "tables": {name: _sqlite_table_snapshot(conn, name) for name in LEGACY_TABLES},
        }
    finally:
        conn.close()


def fts_snapshot(path: Path) -> dict[str, Any]:
    result = file_fingerprint(path, hash_contents=False)
    if not result.get("exists"):
        return result
    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        exists = _table_exists(conn, "chunks_fts")
        result["chunks_fts_exists"] = exists
        if exists:
            count, items, attachments = conn.execute(
                "SELECT COUNT(*), COUNT(DISTINCT item_key), COUNT(DISTINCT attachment_key) FROM chunks_fts",
            ).fetchone()
            result.update({
                "chunks": int(count), "items": int(items), "attachments": int(attachments),
                "schema": conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks_fts'",
                ).fetchone()[0],
            })
        return result
    finally:
        conn.close()


def manifest_snapshot(path: Path) -> dict[str, Any]:
    result = file_fingerprint(path)
    if not result.get("exists"):
        return result
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["parse_error"] = str(exc)
        return result
    files = value.get("files") if isinstance(value.get("files"), dict) else {}
    notes = value.get("notes") if isinstance(value.get("notes"), dict) else {}
    result.update({
        "files": len(files), "notes": len(notes),
        "pipeline_fingerprint": value.get("pipeline_fingerprint"),
    })
    return result


def chroma_snapshot(path: Path) -> dict[str, Any]:
    """Inspect only the legacy collection metadata/counts; never create one."""
    result: dict[str, Any] = {"exists": path.exists(), "collections": {}}
    if not path.exists():
        return result
    import chromadb

    client = chromadb.PersistentClient(path=str(path))
    try:
        known = {entry.name: entry for entry in client.list_collections()}
        for name in LEGACY_COLLECTIONS:
            collection = known.get(name)
            if collection is None:
                result["collections"][name] = {"exists": False, "count": 0}
                continue
            metadata = collection.metadata or {}
            result["collections"][name] = {
                "exists": True,
                "count": int(collection.count()),
                "metadata": metadata,
                "metadata_fingerprint": _sha256_bytes([
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode("utf-8"),
                ]),
            }
    finally:
        try:
            client.close()
        except Exception:  # pragma: no cover - older Chroma clients have no close()
            pass
    return result


def snapshot(
    *, db_path: Path, chroma_path: Path, manifest_path: Path, fts_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "legacy-retirement-audit-1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "relations": sqlite_snapshot(db_path),
        "legacy_manifest": manifest_snapshot(manifest_path),
        "legacy_fts": fts_snapshot(fts_path),
        "legacy_chroma": chroma_snapshot(chroma_path),
    }


def _changed(before: Any, after: Any) -> bool:
    return before != after


def compare(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Report every observable legacy mutation, including same-count upserts."""
    failures: list[str] = []
    for table in LEGACY_TABLES:
        previous = baseline.get("relations", {}).get("tables", {}).get(table, {})
        observed = current.get("relations", {}).get("tables", {}).get(table, {})
        if int(observed.get("count") or 0) > int(previous.get("count") or 0):
            failures.append(f"legacy_table_row_growth:{table}")
        if _changed(previous.get("fingerprint"), observed.get("fingerprint")):
            failures.append(f"legacy_table_content_changed:{table}")
    for name in LEGACY_COLLECTIONS:
        previous = baseline.get("legacy_chroma", {}).get("collections", {}).get(name, {})
        observed = current.get("legacy_chroma", {}).get("collections", {}).get(name, {})
        if int(observed.get("count") or 0) > int(previous.get("count") or 0):
            failures.append(f"legacy_collection_row_growth:{name}")
        if _changed(previous.get("metadata_fingerprint"), observed.get("metadata_fingerprint")):
            failures.append(f"legacy_collection_metadata_changed:{name}")
    previous_fts = baseline.get("legacy_fts", {})
    observed_fts = current.get("legacy_fts", {})
    if int(observed_fts.get("chunks") or 0) > int(previous_fts.get("chunks") or 0):
        failures.append("legacy_fts_row_growth")
    # V3 uses its own FTS file, so this mtime change is a direct legacy-write signal.
    if previous_fts.get("mtime_ns") != observed_fts.get("mtime_ns"):
        failures.append("legacy_fts_file_changed")
    if baseline.get("legacy_manifest", {}).get("sha256") != current.get("legacy_manifest", {}).get("sha256"):
        failures.append("legacy_manifest_changed")
    return {"passed": not failures, "failures": failures}


def compare_chroma_backup(current: dict[str, Any], backup: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    for name in LEGACY_COLLECTIONS:
        live = current.get("collections", {}).get(name, {})
        saved = backup.get("collections", {}).get(name, {})
        if live.get("count") != saved.get("count"):
            failures.append(f"backup_collection_count_mismatch:{name}")
        if live.get("metadata_fingerprint") != saved.get("metadata_fingerprint"):
            failures.append(f"backup_collection_metadata_mismatch:{name}")
    return {"passed": not failures, "failures": failures}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "relations.db")
    parser.add_argument("--chroma", type=Path, default=ROOT / "data" / "chroma")
    parser.add_argument("--legacy-manifest", type=Path, default=ROOT / "data" / "manifest.json")
    parser.add_argument("--legacy-fts", type=Path, default=ROOT / "data" / "lexical.sqlite3")
    parser.add_argument("--baseline", type=Path, help="Earlier JSON from this script to compare against.")
    parser.add_argument(
        "--backup-chroma", type=Path,
        help="Existing full Chroma backup to verify by legacy collection count and metadata.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path (the only write this tool performs).")
    args = parser.parse_args(argv)

    report = snapshot(
        db_path=args.db, chroma_path=args.chroma,
        manifest_path=args.legacy_manifest, fts_path=args.legacy_fts,
    )
    exit_code = 0
    if args.baseline:
        try:
            baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"cannot read baseline: {exc}")
        report["baseline"] = str(args.baseline)
        report["write_zero_gate"] = compare(baseline, report)
        exit_code = 2 if not report["write_zero_gate"]["passed"] else 0
    if args.backup_chroma:
        backup = chroma_snapshot(args.backup_chroma)
        report["backup_chroma"] = str(args.backup_chroma)
        report["backup_chroma_gate"] = compare_chroma_backup(report["legacy_chroma"], backup)
        if not report["backup_chroma_gate"]["passed"]:
            exit_code = 2
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
