#!/usr/bin/env python3
"""Run non-blocking GROBID reference enrichment without re-embedding documents."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db_relations import (
    get_item_processing_status, mark_artifact_status, stage_reference_candidates,
)
from src.env_utils import load_dotenv_native
from src.feature_gates import grobid_enrichment_enabled, verify_enabled_features
from src.grobid_enrichment import process_pdf, should_enrich
from src.text_utils import detect_lang
from src.zotero_source_localapi import ZoteroLocalAPI

load_dotenv_native(ROOT)
PROCESSOR = "grobid:0.9.0-crf"


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def already_processed(item_key: str, attachment_key: str, fingerprint: str) -> bool:
    return any(
        row.get("artifact_type") == "references"
        and str(row.get("attachment_key") or "") == attachment_key
        and row.get("status") in {"success", "empty"}
        and row.get("source_fingerprint") == fingerprint
        and row.get("processor_version") == PROCESSOR
        for row in get_item_processing_status(item_key)
    )


async def run(args: argparse.Namespace) -> dict[str, int]:
    api = ZoteroLocalAPI()
    attachments = await api.list_normalized_attachments(
        os.environ.get("ZOTERO_DATA_DIR"), str(ROOT / "data" / "pdf_cache"),
    )
    requested = {str(value) for value in (args.item or [])}
    counts = {"eligible": 0, "processed": 0, "skipped": 0, "failed": 0, "references": 0}
    for attachment in attachments:
        item_key = str(attachment.parentItemKey or attachment.attachmentKey)
        if requested and item_key not in requested:
            continue
        language = detect_lang("", attachment.language)
        if not should_enrich(
            item_type=attachment.parentItemType, language=language,
            source_type=attachment.source_type,
        ):
            continue
        counts["eligible"] += 1
        if args.limit and counts["processed"] >= args.limit:
            break
        pdf_path = Path(attachment.pdf_path)
        fingerprint = file_fingerprint(pdf_path)
        if already_processed(item_key, attachment.attachmentKey, fingerprint) and not args.force:
            counts["skipped"] += 1
            continue
        if args.dry_run:
            print(json.dumps({
                "item_key": item_key, "attachment_key": attachment.attachmentKey,
                "item_type": attachment.parentItemType, "language": language,
                "fingerprint": fingerprint, "action": "would_process",
            }, ensure_ascii=False))
            counts["processed"] += 1
            continue
        mark_artifact_status(
            item_key, "references", "running", attachment_key=attachment.attachmentKey,
            source_fingerprint=fingerprint, processor_version=PROCESSOR,
        )
        try:
            result = process_pdf(pdf_path)
            staged = stage_reference_candidates(item_key, PROCESSOR, result.references)
            status = "success" if result.references else "empty"
            mark_artifact_status(
                item_key, "references", status, attachment_key=attachment.attachmentKey,
                reason_code=None if result.references else "grobid_no_references",
                retryable=False, source_fingerprint=fingerprint, processor_version=PROCESSOR,
                counts={
                    "references": len(result.references), "staged": staged["staged"],
                    "updated": staged["updated"], "headings": len(result.headings),
                    "paragraphs": result.paragraph_count,
                    "citation_markers": result.citation_marker_count,
                    "linked_citations": result.linked_citation_count,
                },
            )
            counts["references"] += len(result.references)
        except Exception as exc:
            counts["failed"] += 1
            mark_artifact_status(
                item_key, "references", "failed", attachment_key=attachment.attachmentKey,
                reason_code="grobid_enrichment_failed", message=str(exc)[:1000],
                retryable=False, source_fingerprint=fingerprint, processor_version=PROCESSOR,
            )
        counts["processed"] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item", action="append", help="Limit to parent itemKey; repeatable")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--apply", action="store_true", help="Write only reference artifacts/review queue")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    args.dry_run = not args.apply
    # Previously this forced the flag on, which made the flag decorative and
    # meant "GROBID is off" and "GROBID is running" were never checked at the
    # one place that can act on them.  Enabling the feature is now the
    # operator's declaration, and a declaration without a service is an error
    # rather than a silent no-op (see src/feature_gates.py).
    if not grobid_enrichment_enabled():
        print(
            "GROBID enrichment is off. Set GROBID_ENRICHMENT_ENABLE=1 in .env "
            "and start a local GROBID service before running this worker.",
            file=sys.stderr,
        )
        return 2
    problems = verify_enabled_features()
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 2
    result = asyncio.run(run(args))
    print(json.dumps({**result, "dry_run": args.dry_run}, ensure_ascii=False))
    return 0 if result["failed"] == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
