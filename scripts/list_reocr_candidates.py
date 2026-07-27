#!/usr/bin/env python3
"""Rank V3 re-OCR candidates without modifying canonical data."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chunk_store import get_item_chunks, list_item_keys
from src.db_relations import (
    get_document_nodes, get_document_structure, get_item_processing_status,
    get_chunk_quality_reports,
    get_summary_quality_reports,
)
from src.extraction_engine import EngineRegistry
from src.manifest import load_manifest
from src.reocr_quality import candidate_assessment


def _engine_identity(
    attachment_key: str, chunks: list[dict[str, Any]], quality: dict[str, Any],
    nodes: list[dict[str, Any]], statuses: list[dict[str, Any]],
) -> tuple[str | None, str | None]:
    for chunk in chunks:
        md = chunk.get("metadata") or {}
        if md.get("extraction_engine"):
            return str(md["extraction_engine"]), str(md.get("extraction_version") or "") or None
    for node in nodes:
        if str(node.get("attachment_key") or "") == attachment_key and node.get("extraction_engine"):
            return str(node["extraction_engine"]), str(node.get("extraction_version") or "") or None
    engine = quality.get("extraction_engine") or quality.get("parser")
    version = quality.get("extraction_version") or quality.get("parser_version") or quality.get("version")
    if engine:
        return str(engine), str(version or "") or None
    extraction = next((
        row for row in statuses
        if row.get("artifact_type") == "extraction"
        and str(row.get("attachment_key") or "") == attachment_key
    ), None)
    return (
        (str(extraction.get("processor_version")) if extraction and extraction.get("processor_version") else None),
        None,
    )


def build_candidates(
    manifest: dict[str, Any], _legacy_report: dict[str, Any] | None = None, *,
    item_keys: Iterable[str] | None = None,
    chunk_loader: Callable[[str], list[dict[str, Any]]] = get_item_chunks,
    structure_loader: Callable[[str], dict[str, Any] | None] = get_document_structure,
    node_loader: Callable[[str], list[dict[str, Any]]] = get_document_nodes,
    status_loader: Callable[[str], list[dict[str, Any]]] = get_item_processing_status,
    summary_reports: list[dict[str, Any]] | None = None,
    chunk_reports: list[dict[str, Any]] | None = None,
    target_engine: str = "docling", target_version: str = "v3-adapter-1",
) -> list[dict[str, Any]]:
    """Build case-independent candidates from V3 extraction and structure data.

    ``_legacy_report`` is accepted but ignored so callers get a safe migration
    path; no grounding, case, quote, or legacy summary fields are consulted.
    """
    reports_by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in summary_reports or []:
        reports_by_item[str(report.get("item_key") or "")].append(report)
    # Reader-reported source-text damage, keyed by attachment so a report about
    # one PDF does not raise the score of its item's other attachments.
    chunk_reports_by_attachment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in chunk_reports or []:
        chunk_reports_by_attachment[str(report.get("attachment_key") or "")].append(report)
    keys = list(item_keys) if item_keys is not None else list_item_keys()
    rows: list[dict[str, Any]] = []
    files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    for item_key in sorted({str(key) for key in keys if key}):
        all_chunks = chunk_loader(item_key)
        by_attachment: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for chunk in all_chunks:
            attachment_key = str((chunk.get("metadata") or {}).get("attachmentKey") or "")
            if attachment_key:
                by_attachment[attachment_key].append(chunk)
        structure = structure_loader(item_key) or {}
        nodes = node_loader(item_key)
        statuses = status_loader(item_key)
        for attachment_key, chunks in sorted(by_attachment.items()):
            quality = dict((files.get(attachment_key) or {}).get("quality") or {})
            current_engine, current_version = _engine_identity(
                attachment_key, chunks, quality, nodes, statuses,
            )
            assessment = candidate_assessment(
                quality=quality, chunks=chunks,
                structure_status=str(structure.get("status") or "unavailable"),
                summary_reports=reports_by_item[item_key],
                chunk_reports=chunk_reports_by_attachment[attachment_key],
                current_engine=current_engine, current_version=current_version,
                target_engine=target_engine, target_version=target_version,
            )
            if not assessment["candidate"]:
                continue
            rows.append({
                "item_key": item_key, "attachment_key": attachment_key,
                **assessment,
                "recommendation": f"reocr_with_{target_engine}",
            })
    return sorted(rows, key=lambda row: (-int(row["score"]), row["item_key"], row["attachment_key"]))


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# 再OCR候補（V3）", "",
        "extraction品質・構造・要約品質報告・engine/version差だけで生成した非破壊の候補表。",
        "", "| rank | itemKey | attachmentKey | lang | current → target | reasons | score |",
        "|---:|---|---|---|---|---|---:|",
    ]
    for rank, row in enumerate(rows, 1):
        current = f"{row['current_engine']}@{row['current_version']}"
        target = f"{row['target_engine']}@{row['target_version']}"
        lines.append(
            f"| {rank} | {row['item_key']} | {row['attachment_key']} | {row['language']} | "
            f"{current} → {target} | {', '.join(row['reasons'])} | {row['score']} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest.json")
    parser.add_argument("--engine", default=os.environ.get("PDF_EXTRACTION_ENGINE", "docling"))
    parser.add_argument("--engine-version")
    parser.add_argument("--item", action="append", help="Limit to itemKey; repeatable")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args(argv)
    if not args.json_output and not args.markdown_output:
        parser.error("at least one of --json-output or --markdown-output is required")
    engine = EngineRegistry().get(args.engine)
    target_version = args.engine_version or engine.version
    rows = build_candidates(
        load_manifest(args.manifest), item_keys=args.item or list_item_keys(),
        summary_reports=get_summary_quality_reports(None),
        chunk_reports=get_chunk_quality_reports("pending"),
        target_engine=engine.name, target_version=target_version,
    )
    payload = {
        "schema_version": "reocr-candidates-v3", "dry_run": True,
        "target_engine": engine.name, "target_version": target_version,
        "candidates": rows,
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(rows), encoding="utf-8")
    print(json.dumps({"candidates": len(rows), "dry_run": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
