#!/usr/bin/env python3
"""Restartable, one-item-at-a-time rollout of note 79's PDF quality gates.

The old bulk scan run shared one Python process with RapidOCR and was observed
to leak a multiprocessing semaphore.  This runner deliberately gives every
parent item a fresh process, persists progress after every result, and runs the
same V3 manifest/Chroma/FTS audit that is used for cutover.  It is safe to
restart: a source fingerprint change makes an item pending again, while a
completed unchanged item is left alone.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_utils import load_dotenv_native

load_dotenv_native(ROOT)

from src.manifest import load_manifest
from src.pdf_provenance import SCAN_DERIVED_CLASSES, classify_pdf_source
from src.zotero_source_localapi import ZoteroLocalAPI


DEFAULT_QUEUE = ROOT / "data" / "quality" / "u5_scanned_reingest_queue.json"
DEFAULT_LOG_DIR = ROOT / "data" / "quality" / "u5_scanned_reingest_logs"
DEFAULT_AUDIT_DIR = ROOT / "data" / "quality" / "u5_scanned_reingest_audits"
MANIFEST = ROOT / "data" / "manifest_v3.json"
QUEUE_SCHEMA_VERSION = "u5-scanned-reingest-2"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


async def _parent_keys() -> dict[str, str]:
    zotero_dir = os.environ.get("ZOTERO_DATA_DIR")
    if not zotero_dir:
        raise RuntimeError("ZOTERO_DATA_DIR is required to resolve parent item keys")
    attachments = await ZoteroLocalAPI().list_normalized_attachments(
        zotero_data_dir=zotero_dir,
        pdf_cache_dir=str(ROOT / "data" / "pdf_cache"),
    )
    return {
        attachment.attachmentKey: str(attachment.parentItemKey or attachment.attachmentKey)
        for attachment in attachments
        if attachment.source_type == "pdf"
    }


def prepare_queue(queue_path: Path) -> dict[str, Any]:
    """Classify all current V3 PDF sources and preserve current progress."""
    prior = _read_json(queue_path)
    prior_is_current = prior.get("schema_version") == QUEUE_SCHEMA_VERSION
    prior_rows = {
        str(row.get("attachment_key")): row
        for row in prior.get("items", []) if isinstance(row, dict) and row.get("attachment_key")
    }
    manifest = load_manifest(MANIFEST)
    files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    parent_by_attachment = asyncio.run(_parent_keys())
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for attachment_key, entry in sorted(files.items()):
        if not isinstance(entry, dict):
            continue
        path = Path(str(entry.get("pdf_path") or ""))
        if path.suffix.lower() != ".pdf":
            continue
        if not path.exists():
            missing.append(str(attachment_key))
            continue
        source = classify_pdf_source(path)
        if source.kind not in SCAN_DERIVED_CLASSES:
            continue
        stat = path.stat()
        fingerprint = f"stat:{stat.st_mtime}:{stat.st_size}"
        old = prior_rows.get(str(attachment_key), {})
        unchanged = prior_is_current and old.get("source_fingerprint") == fingerprint
        old_status = str(old.get("status") or "pending")
        # A new routing/scope contract invalidates every old terminal state.
        # For a same-fingerprint v2 row, preserve resumable state except a
        # process interrupted while marked running.
        status = old_status if unchanged and old_status != "running" else "pending"
        attempts = int(old.get("attempts") or 0) if unchanged else 0
        rows.append({
            "attachment_key": str(attachment_key),
            "item_key": parent_by_attachment.get(str(attachment_key), str(attachment_key)),
            "title": entry.get("title"),
            "source_path": str(path),
            "source_fingerprint": fingerprint,
            "source_class": source.kind,
            "source_classification": source.as_metadata(),
            "status": status,
            "attempts": attempts,
            **({"last_result": old["last_result"]} if "last_result" in old else {}),
            **({"recovered_interrupted_at": _now()} if unchanged and old_status == "running" else {}),
            **({"migrated_from_schema": prior.get("schema_version")} if not prior_is_current and prior else {}),
        })
    payload = {
        "schema_version": QUEUE_SCHEMA_VERSION,
        "generated_at": _now(),
        "manifest": str(MANIFEST),
        "items": rows,
        "classification_missing_paths": missing,
    }
    _write_json(queue_path, payload)
    return payload


def _result_event(output: str) -> dict[str, Any] | None:
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("event") == "index_batch_result":
            return value
    return None


def _manifest_outcome(attachment_key: str, fingerprint: str) -> dict[str, Any]:
    entry = (load_manifest(MANIFEST).get("files") or {}).get(attachment_key)
    if not isinstance(entry, dict):
        return {"verified": False, "reason": "manifest_entry_missing"}
    actual = f"stat:{entry.get('mtime')}:{entry.get('size')}"
    quality = entry.get("quality") if isinstance(entry.get("quality"), dict) else {}
    reason = str(quality.get("ocr_layer_audit_reason") or "")
    source_class = str(quality.get("source_class") or "")
    verified = (
        actual == fingerprint
        and source_class in SCAN_DERIVED_CLASSES
        # "cloud_not_allowed:no-cloud" was removed with the flag itself
        # (2026-07-27): anything in the library is already sent to the cloud as
        # chunks by the RAG server, so a per-item no-cloud reason never applied.
        and reason in {"measured", "insufficient_sample", "not_applicable_no_ocr_layer"}
    )
    return {
        "verified": verified, "reason": reason or "audit_not_persisted",
        "source_class": source_class, "ocr_layer_quality": quality.get("ocr_layer_quality"),
        "ocr_layer_error_rate": quality.get("ocr_layer_error_rate"),
        "text_defects": quality.get("text_defects") or [],
    }


def _audit_item(item_key: str, output_path: Path) -> dict[str, Any]:
    command = [
        str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "scripts" / "audit_v3_cutover.py"),
        "--item", item_key, "--output", str(output_path),
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    payload = _read_json(output_path)
    comparison = (payload.get("items") or [{}])[0] if isinstance(payload.get("items"), list) else {}
    return {
        "returncode": completed.returncode,
        "item_passed": bool(comparison.get("passed")),
        "item_failures": comparison.get("failures") or [],
        "global_failures": (payload.get("gate") or {}).get("global_failures") or [],
        "output_tail": completed.stdout[-4000:],
    }


def run_queue(
    queue_path: Path, log_dir: Path, audit_dir: Path, max_attempts: int, limit: int,
    attachment_keys: set[str] | None = None,
) -> dict[str, Any]:
    queue = _read_json(queue_path)
    rows = queue.get("items") if isinstance(queue.get("items"), list) else []
    processed = 0
    for row in rows:
        if not isinstance(row, dict) or row.get("status") in {
            "completed", "completed_with_preexisting_audit_failure", "deferred", "blocked",
        }:
            continue
        if attachment_keys and str(row.get("attachment_key") or "") not in attachment_keys:
            continue
        if int(row.get("attempts") or 0) >= max_attempts:
            row["status"] = "failed"
            continue
        if limit and processed >= limit:
            break
        item_key = str(row["item_key"])
        attachment_key = str(row["attachment_key"])
        row["status"] = "running"
        row["started_at"] = _now()
        _write_json(queue_path, queue)
        command = [
            str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "src" / "index_from_zotero.py"),
            "--force-reparse", "--item", item_key, "--attachment", attachment_key,
            "--source-type", "pdf", "--require-data-dir", "--progress",
        ]
        completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log_path = log_dir / f"{attachment_key}.attempt-{int(row.get('attempts') or 0) + 1}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(completed.stdout, encoding="utf-8")
        event = _result_event(completed.stdout)
        outcome = _manifest_outcome(attachment_key, str(row["source_fingerprint"]))
        audit = _audit_item(item_key, audit_dir / f"{attachment_key}.json") if completed.returncode == 0 else {}
        row["attempts"] = int(row.get("attempts") or 0) + 1
        row["finished_at"] = _now()
        row["last_result"] = {
            "returncode": completed.returncode, "event": event, "manifest": outcome,
            "integrity_audit": audit, "log": str(log_path), "output_tail": completed.stdout[-4000:],
        }
        if completed.returncode == 0 and "deferred to Mistral OCR batch" in completed.stdout:
            row["status"] = "deferred"
        elif completed.returncode == 0 and event and not event.get("inflight_attachments") and outcome["verified"]:
            row["status"] = "completed" if audit.get("item_passed") else "completed_with_preexisting_audit_failure"
        elif row["attempts"] >= max_attempts:
            row["status"] = "failed"
        else:
            row["status"] = "pending"
        processed += 1
        _write_json(queue_path, queue)
        print(json.dumps({"attachment_key": attachment_key, "status": row["status"], "result": row["last_result"]}, ensure_ascii=False))
    counts: dict[str, int] = {}
    for row in rows:
        if isinstance(row, dict):
            status = str(row.get("status") or "pending")
            counts[status] = counts.get(status, 0) + 1
    queue["last_run_at"] = _now()
    queue["summary"] = counts
    _write_json(queue_path, queue)
    return {"processed": processed, "summary": counts, "queue": str(queue_path)}


def retry_deferred(queue_path: Path, attachment_keys: set[str] | None = None) -> int:
    """Make explicitly selected Mistral deferrals eligible for a fresh run."""
    queue = _read_json(queue_path)
    rows = queue.get("items") if isinstance(queue.get("items"), list) else []
    retried = 0
    for row in rows:
        if not isinstance(row, dict) or row.get("status") != "deferred":
            continue
        if attachment_keys and str(row.get("attachment_key") or "") not in attachment_keys:
            continue
        row["status"] = "pending"
        row["retry_requested_at"] = _now()
        retried += 1
    if retried:
        _write_json(queue_path, queue)
    return retried


def recover_interrupted(queue_path: Path) -> int:
    """Return rows left ``running`` by an interrupted supervisor to pending."""
    queue = _read_json(queue_path)
    rows = queue.get("items") if isinstance(queue.get("items"), list) else []
    recovered = 0
    for row in rows:
        if isinstance(row, dict) and row.get("status") == "running":
            row["status"] = "pending"
            row["recovered_interrupted_at"] = _now()
            recovered += 1
    if recovered:
        _write_json(queue_path, queue)
    return recovered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--prepare", action="store_true", help="Reclassify current PDFs and create/update the queue.")
    parser.add_argument("--run", action="store_true", help="Run pending queue entries sequentially.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum queue entries for this invocation.")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--attachment", action="append", help="Run only this attachment key; repeatable.")
    parser.add_argument("--retry-deferred", action="store_true", help="Explicitly retry deferred selected item(s).")
    parser.add_argument("--recover-interrupted", action="store_true", help="Reset stale running rows before a resume.")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    args = parser.parse_args()
    if not args.prepare and not args.run:
        parser.error("select --prepare, --run, or both")
    if args.max_attempts < 1 or args.limit < 0:
        parser.error("--max-attempts must be positive and --limit cannot be negative")
    if (args.retry_deferred or args.recover_interrupted) and not args.run:
        parser.error("--retry-deferred/--recover-interrupted requires --run")
    if args.prepare:
        prepared = prepare_queue(args.queue)
        print(json.dumps({"prepared": len(prepared["items"]), "missing_paths": len(prepared["classification_missing_paths"])}, ensure_ascii=False))
    if args.run:
        selected = {str(value) for value in args.attachment or []} or None
        if args.recover_interrupted:
            print(json.dumps({"recovered_interrupted": recover_interrupted(args.queue)}, ensure_ascii=False))
        if args.retry_deferred:
            print(json.dumps({"retried_deferred": retry_deferred(args.queue, selected)}, ensure_ascii=False))
        print(json.dumps(run_queue(
            args.queue, args.log_dir, args.audit_dir, args.max_attempts, args.limit,
            selected,
        ), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
