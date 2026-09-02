#!/usr/bin/env python3
"""Inspect pending daily maintenance work without modifying canonical data."""
from __future__ import annotations

import asyncio
import json
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from src.env_utils import load_dotenv_native

load_dotenv_native(ROOT)

from index_from_zotero import (  # noqa: E402
    _apply_rag_tag_policy,
    _ready_preferred_pdfs,
    paths,
)
from zotero_source_localapi import ZoteroLocalAPI  # noqa: E402
from update_citations import _unwrap_item, get_all_items  # noqa: E402

REPORT_PATH = ROOT / "data" / "admin_update_status.json"
TERMINAL_CITATION_STATUSES = frozenset({"mapped", "not_found", "limited"})


def _brief(keys: list[str], limit: int = 20) -> list[str]:
    return sorted(dict.fromkeys(keys))[:limit]


def _index_freshness(
    attachments: list[Any], manifest: dict[str, Any], *, excluded_keys: set[str],
) -> dict[str, Any]:
    files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    expected = {str(row.attachmentKey) for row in attachments}
    pending: list[str] = []
    missing_source: list[str] = []
    pipeline = str(manifest.get("pipeline_fingerprint") or "")
    for row in attachments:
        key = str(row.attachmentKey)
        previous = files.get(key)
        try:
            stat = Path(str(row.pdf_path)).stat()
        except OSError:
            missing_source.append(key)
            continue
        if not previous or (
            float(previous.get("mtime", -1)) != float(stat.st_mtime)
            or int(previous.get("size", -1)) != int(stat.st_size)
            or str(previous.get("pipeline_fingerprint") or "") != pipeline
        ):
            pending.append(key)
    retired = set(str(key) for key in files) - expected - excluded_keys
    excluded_tracked = set(str(key) for key in files).intersection(excluded_keys)
    all_pending = pending + list(retired) + list(excluded_tracked)
    return {
        "pending": len(all_pending),
        "new_or_changed": len(pending),
        "retired": len(retired),
        "excluded_tracked": len(excluded_tracked),
        "missing_source": len(missing_source),
        "sample_keys": _brief(all_pending + missing_source),
    }


def _note_freshness(notes: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    stored = manifest.get("notes") if isinstance(manifest.get("notes"), dict) else {}
    current = {
        str(row["noteKey"]): row.get("version")
        for row in notes
        if isinstance(row, dict) and row.get("noteKey")
    }
    changed = sorted(
        key for key, version in current.items()
        if key not in stored or (stored.get(key) or {}).get("version") != version
    )
    retired = sorted(set(str(key) for key in stored) - set(current))
    return {
        "pending": len(changed) + len(retired),
        "new_or_changed": len(changed),
        "retired": len(retired),
        "sample_keys": _brief(changed + retired),
    }


def _citation_freshness(items: list[dict[str, Any]], database: Path) -> dict[str, Any]:
    statuses: dict[str, dict[str, str]] = {}
    with sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT item_key, s2_status, doi, isbn FROM item_citation_status"
        ).fetchall()
        statuses = {
            str(key): {
                "status": str(status or ""), "doi": str(doi or ""), "isbn": str(isbn or ""),
            }
            for key, status, doi, isbn in rows
        }
    current: dict[str, dict[str, Any]] = {}
    for raw in items:
        key, data = _unwrap_item(raw)
        if key:
            current[str(key)] = data
    pending = sorted(
        key for key in current
        if statuses.get(key, {}).get("status") not in TERMINAL_CITATION_STATUSES
    )
    errors = sorted(key for key in current if statuses.get(key, {}).get("status") == "error")
    metadata_changed = sorted(
        key for key, data in current.items()
        if statuses.get(key, {}).get("status") in TERMINAL_CITATION_STATUSES
        and any(
            str(data.get(field) or "") != statuses.get(key, {}).get(column, "")
            for field, column in (("DOI", "doi"), ("ISBN", "isbn"))
        )
    )
    work = sorted(set(pending).union(metadata_changed))
    return {
        "pending": len(work),
        "unprocessed": len(pending),
        "metadata_changed": len(metadata_changed),
        "errors": len(errors),
        "sample_keys": _brief(errors + work),
    }


def _structure_freshness() -> dict[str, Any]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "rebuild_document_structure.py"),
        "--all",
        "--dry-run",
        "--no-source-refresh",
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:] or f"structure check exited {result.returncode}")
    payload = json.loads(result.stdout)
    items = payload.get("items") or []
    changed = [str(row.get("item_key")) for row in items if row.get("changed")]
    failed = [str(row.get("item_key")) for row in items if row.get("status") == "failed"]
    refresh_failed = [
        str(row.get("item_key")) for row in items if row.get("source_refresh_error")
    ]
    return {
        "pending": len(changed),
        "failed": len(failed),
        "source_refresh_failed": len(refresh_failed),
        "sample_keys": _brief(failed + refresh_failed + changed),
    }


async def inspect() -> dict[str, Any]:
    ingest_paths = paths()
    manifest = json.loads(ingest_paths.manifest_path.read_text(encoding="utf-8"))
    attachment_api = ZoteroLocalAPI()
    note_api = ZoteroLocalAPI()
    zotero_dir = ingest_paths.zotero_data_dir if ingest_paths.zotero_data_dir else None
    rows, notes, citation_items = await asyncio.gather(
        attachment_api.list_normalized_attachments(
            zotero_data_dir=zotero_dir,
            pdf_cache_dir=str(ingest_paths.pdf_cache_dir),
            collection_key=None,
            require_complete=False,
        ),
        note_api.list_notes(collection_key=None),
        asyncio.to_thread(get_all_items),
    )
    inventory = [row for row in rows if getattr(row, "pdf_path", None)]
    included, excluded, preferred = _apply_rag_tag_policy(inventory)
    ready = _ready_preferred_pdfs(preferred, inventory, manifest.get("files") or {})
    ready_keys = {str(row.attachmentKey) for row in ready}
    included = [row for row in included if str(row.attachmentKey) not in ready_keys]
    excluded_keys = {str(row.attachmentKey) for row in excluded}.union(ready_keys)
    attachment_status = _index_freshness(included, manifest, excluded_keys=excluded_keys)
    note_status = _note_freshness(notes, manifest)
    index_status = {
        **attachment_status,
        "pending": int(attachment_status["pending"]) + int(note_status["pending"]),
        "attention": int(attachment_status["missing_source"]),
        "notes": note_status,
        "sample_keys": _brief(
            list(attachment_status["sample_keys"]) + list(note_status["sample_keys"])
        ),
    }
    structure_status = _structure_freshness()
    structure_status["attention"] = (
        int(structure_status["failed"]) + int(structure_status["source_refresh_failed"])
    )
    citation_status = _citation_freshness(citation_items, ROOT / "data" / "relations.db")
    citation_status["attention"] = int(citation_status["errors"])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "index": index_status,
        "structure": structure_status,
        "citations": citation_status,
    }


def _write_report(report: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=REPORT_PATH.parent, delete=False) as stream:
        stream.write(encoded)
        stream.flush()
        candidate = Path(stream.name)
    candidate.chmod(0o600)
    candidate.replace(REPORT_PATH)


def main() -> int:
    report = asyncio.run(inspect())
    _write_report(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
