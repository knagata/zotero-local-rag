#!/usr/bin/env python3
"""Export deferred AI-TOC failures as a deterministic Mistral OCR batch queue."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db_relations import get_artifact_processing_statuses

REASON = "awaiting_mistral_ocr_batch"


def build_queue_rows(statuses: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for status in statuses:
        if (
            status.get("artifact_type") != "extraction"
            or status.get("status") != "blocked"
            or status.get("reason_code") != REASON
            or not status.get("attachment_key")
        ):
            continue
        counts = status.get("counts") if isinstance(status.get("counts"), dict) else {}
        rows.append({
            "item_key": str(status.get("item_key") or ""),
            "attachment_key": str(status["attachment_key"]),
            "target_engine": "mistral_ocr",
            "recommendation": "mistral_ocr_batch_after_ai_toc_gate",
            "queue_reason": REASON,
            "ai_toc_reason": counts.get("ai_toc_reason"),
            "ai_toc_diagnostics": counts.get("ai_toc_diagnostics") or {},
            "total_pages": counts.get("total_pages"),
            "source_mtime": counts.get("source_mtime"),
            "source_size": counts.get("source_size"),
            "source_type": counts.get("source_type") or "pdf",
            "batch_document_path": counts.get("batch_document_path"),
            "epub_mapping_path": counts.get("epub_mapping_path"),
        })
    return sorted(rows, key=lambda row: (row["item_key"], row["attachment_key"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--item", action="append", help="Limit to itemKey; repeatable")
    args = parser.parse_args(argv)
    statuses = get_artifact_processing_statuses(
        artifact_type="extraction", reason_code=REASON,
    )
    rows = build_queue_rows(statuses)
    if args.item:
        allowed = {str(value) for value in args.item}
        rows = [row for row in rows if row["item_key"] in allowed]
    payload = {
        "schema_version": "mistral-toc-candidates-v1",
        "dry_run": True,
        "target_engine": "mistral_ocr",
        "candidates": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"candidates": len(rows), "dry_run": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
