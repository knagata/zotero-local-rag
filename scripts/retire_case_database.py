#!/usr/bin/env python3
"""Back up and retire the removed structured-case database (dry-run by default)."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "relations.db"
DEFAULT_CHROMA = ROOT / "data" / "chroma"
DEFAULT_BACKUPS = ROOT / "data" / "backups"
CASE_TABLES = ("case_annotations", "case_evidence", "case_quality_reports")


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,),
    ).fetchone() is not None


def _rows(conn: sqlite3.Connection, table: str, where: str = "", params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    if not _table_exists(conn, table):
        return []
    conn.row_factory = sqlite3.Row
    return [dict(row) for row in conn.execute(f'SELECT * FROM "{table}" {where}', params)]


def inspect_sqlite(db_path: Path) -> dict[str, int]:
    if not db_path.exists():
        return {table: 0 for table in (*CASE_TABLES, "artifact_processing_status", "artifact_processing_events")}
    with sqlite3.connect(db_path) as conn:
        counts = {table: len(_rows(conn, table)) for table in CASE_TABLES}
        counts["artifact_processing_status"] = len(_rows(
            conn, "artifact_processing_status", "WHERE artifact_type='cases'",
        ))
        counts["artifact_processing_events"] = len(_rows(
            conn, "artifact_processing_events", "WHERE artifact_type='cases'",
        ))
        return counts


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bytes):
        return "X'" + value.hex() + "'"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def _dump_table_sql(conn: sqlite3.Connection, table: str, rows: list[dict[str, Any]]) -> list[str]:
    if not _table_exists(conn, table):
        return []
    schema = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,),
    ).fetchone()
    output = [f'DROP TABLE IF EXISTS "{table}";', f"{schema[0]};"] if schema and schema[0] else []
    for row in rows:
        columns = ", ".join(f'"{name}"' for name in row)
        values = ", ".join(_sql_literal(value) for value in row.values())
        output.append(f'INSERT INTO "{table}" ({columns}) VALUES ({values});')
    return output


def backup_sqlite(db_path: Path, backup_dir: Path) -> dict[str, int]:
    backup_dir.mkdir(parents=True, exist_ok=False)
    counts: dict[str, int] = {}
    jsonl_path = backup_dir / "cases.sqlite.jsonl"
    sql_path = backup_dir / "cases.sqlite.sql"
    with sqlite3.connect(db_path) as conn, jsonl_path.open("w", encoding="utf-8") as jsonl:
        conn.row_factory = sqlite3.Row
        table_rows: list[tuple[str, list[dict[str, Any]]]] = []
        for table in CASE_TABLES:
            rows = _rows(conn, table)
            table_rows.append((table, rows))
        for table in ("artifact_processing_status", "artifact_processing_events"):
            rows = _rows(conn, table, "WHERE artifact_type='cases'")
            table_rows.append((table, rows))
        sql_lines = ["PRAGMA foreign_keys=OFF;", "BEGIN TRANSACTION;"]
        for table, rows in table_rows:
            counts[table] = len(rows)
            for row in rows:
                jsonl.write(json.dumps({"table": table, "row": row}, ensure_ascii=False) + "\n")
            if table in CASE_TABLES:
                sql_lines.extend(_dump_table_sql(conn, table, rows))
            # Shared-ledger rows remain in JSONL for audit. They cannot be
            # inserted into the post-retirement CHECK constraints by SQL restore.
        sql_lines.extend(["COMMIT;", "PRAGMA foreign_keys=ON;"])
        sql_path.write_text("\n".join(sql_lines) + "\n", encoding="utf-8")
    return counts


def _collection_names(client: Any) -> list[str]:
    output = []
    for value in client.list_collections():
        output.append(str(value if isinstance(value, str) else value.name))
    return sorted(name for name in output if name.endswith("__cases"))


def backup_chroma(chroma_path: Path, backup_dir: Path) -> dict[str, int]:
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_path))
    counts: dict[str, int] = {}
    manifest: dict[str, Any] = {"collections": []}
    with (backup_dir / "cases.chroma.jsonl").open("w", encoding="utf-8") as handle:
        for name in _collection_names(client):
            collection = client.get_collection(name)
            manifest["collections"].append({"name": name, "metadata": collection.metadata or {}})
            offset = 0
            count = 0
            while True:
                batch = collection.get(
                    limit=500, offset=offset,
                    include=["documents", "metadatas", "embeddings"],
                )
                ids = list(batch.get("ids") or [])
                if not ids:
                    break
                embeddings = batch.get("embeddings")
                if hasattr(embeddings, "tolist"):
                    embeddings = embeddings.tolist()
                documents = list(batch.get("documents") or [None] * len(ids))
                metadatas = list(batch.get("metadatas") or [None] * len(ids))
                embeddings = list(embeddings or [None] * len(ids))
                for index, record_id in enumerate(ids):
                    handle.write(json.dumps({
                        "collection": name, "id": record_id,
                        "document": documents[index], "metadata": metadatas[index],
                        "embedding": embeddings[index],
                    }, ensure_ascii=False) + "\n")
                count += len(ids)
                offset += len(ids)
            counts[name] = count
    (backup_dir / "cases.chroma.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return counts


def _rebuild_status_tables(conn: sqlite3.Connection) -> None:
    if _table_exists(conn, "artifact_processing_status"):
        conn.execute("DELETE FROM artifact_processing_status WHERE artifact_type='cases'")
        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='artifact_processing_status'",
        ).fetchone()[0]
        if "'cases'" in sql:
            conn.execute("ALTER TABLE artifact_processing_status RENAME TO artifact_processing_status_case_legacy")
            conn.execute('''
                CREATE TABLE artifact_processing_status (
                    item_key TEXT NOT NULL, attachment_key TEXT NOT NULL DEFAULT '',
                    artifact_type TEXT NOT NULL CHECK(artifact_type IN
                        ('extraction','structure','summary','references','embeddings','summary_index')),
                    status TEXT NOT NULL CHECK(status IN
                        ('pending','running','success','empty','degraded','blocked','failed','stale','excluded')),
                    reason_code TEXT, message TEXT, retryable INTEGER NOT NULL DEFAULT 0,
                    attempt_count INTEGER NOT NULL DEFAULT 0, source_fingerprint TEXT,
                    processor_version TEXT, model TEXT, counts_json TEXT, fallback_kind TEXT,
                    started_at TIMESTAMP, finished_at TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(item_key, attachment_key, artifact_type)
                )
            ''')
            conn.execute('''
                INSERT INTO artifact_processing_status SELECT *
                FROM artifact_processing_status_case_legacy WHERE artifact_type <> 'cases'
            ''')
            conn.execute("DROP TABLE artifact_processing_status_case_legacy")
            conn.execute("CREATE INDEX idx_artifact_processing_item ON artifact_processing_status(item_key, artifact_type)")
    if _table_exists(conn, "artifact_processing_events"):
        conn.execute("DELETE FROM artifact_processing_events WHERE artifact_type='cases'")
    if _table_exists(conn, "insight_generation_status"):
        conn.execute("DELETE FROM insight_generation_status WHERE kind='cases'")
        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='insight_generation_status'",
        ).fetchone()[0]
        if "'cases'" in sql:
            conn.execute("ALTER TABLE insight_generation_status RENAME TO insight_generation_status_case_legacy")
            conn.execute('''
                CREATE TABLE insight_generation_status (
                    item_key TEXT NOT NULL,
                    kind TEXT NOT NULL CHECK(kind = 'sections'),
                    status TEXT NOT NULL CHECK(status IN ('processed_empty', 'available')),
                    row_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(item_key, kind)
                )
            ''')
            legacy_columns = {
                str(row[1]) for row in conn.execute(
                    "PRAGMA table_info(insight_generation_status_case_legacy)"
                )
            }
            columns = [
                name for name in ("item_key", "kind", "status", "row_count", "updated_at")
                if name in legacy_columns
            ]
            column_sql = ", ".join(f'"{name}"' for name in columns)
            conn.execute(
                f"INSERT INTO insight_generation_status ({column_sql}) "
                f"SELECT {column_sql} FROM insight_generation_status_case_legacy "
                "WHERE kind = 'sections'"
            )
            conn.execute("DROP TABLE insight_generation_status_case_legacy")


def retire_sqlite(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("BEGIN IMMEDIATE")
        _rebuild_status_tables(conn)
        for table in ("case_quality_reports", "case_evidence", "case_annotations"):
            conn.execute(f'DROP TABLE IF EXISTS "{table}"')
        conn.commit()


def retire_chroma(chroma_path: Path, names: Iterable[str]) -> None:
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_path))
    existing = set(_collection_names(client))
    for name in names:
        if name in existing:
            client.delete_collection(name)


def run(*, db_path: Path, chroma_path: Path, backups_path: Path, apply: bool) -> dict[str, Any]:
    sqlite_counts = inspect_sqlite(db_path)
    if not apply:
        return {
            "mode": "dry-run", "sqlite": sqlite_counts,
            "message": "No files, tables, rows, or Chroma collections were changed.",
        }
    stamp = datetime.now(timezone.utc).strftime("cases-%Y%m%dT%H%M%S.%fZ")
    backup_dir = backups_path / stamp
    sqlite_backup = backup_sqlite(db_path, backup_dir)
    chroma_backup = backup_chroma(chroma_path, backup_dir)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_db": str(db_path), "source_chroma": str(chroma_path),
        "sqlite": sqlite_backup, "chroma": chroma_backup,
        "restore": "Load cases.sqlite.sql, then recreate each manifest collection and add cases.chroma.jsonl records with their stored embeddings.",
    }
    (backup_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    # Destructive operations are deliberately last: a complete manifest and all
    # exports must already exist before anything is retired.
    retire_sqlite(db_path)
    retire_chroma(chroma_path, chroma_backup)
    return {"mode": "apply", "backup_dir": str(backup_dir), **manifest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path(os.environ.get("RELATIONS_DB_PATH", DEFAULT_DB)))
    parser.add_argument("--chroma", type=Path, default=Path(os.environ.get("CHROMA_DIR", DEFAULT_CHROMA)))
    parser.add_argument("--backups", type=Path, default=DEFAULT_BACKUPS)
    parser.add_argument("--apply", action="store_true", help="Back up, then remove case artifacts.")
    args = parser.parse_args(argv)
    print(json.dumps(run(
        db_path=args.db, chroma_path=args.chroma, backups_path=args.backups, apply=args.apply,
    ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
