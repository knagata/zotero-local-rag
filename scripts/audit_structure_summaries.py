#!/usr/bin/env python3
"""Audit hierarchical summaries and their searchable Chroma collection."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chromadb

from src.build_structure_summaries import (
    CHROMA_DIR, PROMPT_VERSION, _select_searchable_summary_rows, inherited_title,
)
from src.db_relations import get_all_document_node_summaries, get_db_connection
from src.summary_core import is_meta_summary
from src.v3_data_plane import V3_COLLECTION


def build_report(*, collection_name: str) -> dict:
    connection = get_db_connection()
    try:
        structures = {
            str(row[0]): str(row[1])
            for row in connection.execute(
                "SELECT item_key, source_fingerprint FROM document_structures"
            ).fetchall()
        }
        statuses = {
            str(row[0]): str(row[1])
            for row in connection.execute(
                "SELECT item_key, status FROM artifact_processing_status "
                "WHERE artifact_type='summary'"
            ).fetchall()
        }
    finally:
        connection.close()

    rows = get_all_document_node_summaries()
    failures: list[str] = []
    missing_status = sorted(set(structures) - set(statuses))
    failed_status = sorted(
        key for key, status in statuses.items()
        if key in structures and status in {"running", "stale", "failed", "blocked", "degraded"}
    )
    stale_rows = sorted({
        str(row["item_key"]) for row in rows
        if str(row.get("source_fingerprint") or "") != structures.get(str(row["item_key"]), "")
    })
    stale_prompt_rows = sorted({
        str(row["node_id"]) for row in rows
        if str(row.get("prompt_version") or "") != PROMPT_VERSION
    })
    invalid_text_rows = sorted({
        str(row["node_id"]) for row in rows
        if not str(row.get("summary") or "").strip() or is_meta_summary(str(row.get("summary") or ""))
    })
    invalid_searchable_rows = sorted({
        str(row["node_id"]) for row in rows
        if bool(row.get("searchable")) and (
            str(row.get("summary_kind") or "") != "llm"
            or str(row.get("quality_status") or "") not in {"accepted", "candidate"}
        )
    })
    if missing_status:
        failures.append("missing_summary_status")
    if failed_status:
        failures.append("nonterminal_or_degraded_summary_status")
    if stale_rows:
        failures.append("stale_summary_source_fingerprint")
    if stale_prompt_rows:
        failures.append("stale_summary_prompt")
    if invalid_text_rows:
        failures.append("empty_or_meta_summary")
    if invalid_searchable_rows:
        failures.append("invalid_searchable_summary")

    all_searchable = [row for row in rows if bool(row.get("searchable"))]
    searchable = _select_searchable_summary_rows(all_searchable)
    expected_ids = {f"sum:node:{row['node_id']}" for row in searchable}
    titles = {str(row["node_id"]): str(row.get("title") or "") for row in all_searchable}
    parents = {
        str(row["node_id"]): str(row.get("parent_node_id") or "")
        for row in all_searchable
    }
    expected_records = {}
    for row in searchable:
        node_id = str(row["node_id"])
        heading = inherited_title(node_id, titles, parents)
        expected_records[f"sum:node:{node_id}"] = {
            "document": "\n".join(filter(None, [heading, str(row["summary"])]))[:4000],
            "metadata": {
                "itemKey": str(row["item_key"]),
                "attachmentKey": str(row.get("attachment_key") or ""),
                "node_id": node_id,
                "parent_node_id": str(row.get("parent_node_id") or ""),
                "node_type": str(row.get("node_type") or ""),
                "depth": int(row.get("depth") or 0),
                "title": heading,
                "summary_kind": "llm",
                "structure_version": "3",
                "source_fingerprint": str(row.get("source_fingerprint") or ""),
            },
        }
    actual_records = {}
    try:
        collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(
            f"{collection_name}__sum_node"
        )
        indexed = collection.get(include=["documents", "metadatas"])
        actual_ids = set(indexed.get("ids") or [])
        for summary_id, document, metadata in zip(
            indexed.get("ids") or [],
            indexed.get("documents") or [],
            indexed.get("metadatas") or [],
        ):
            actual_records[str(summary_id)] = {
                "document": str(document or ""),
                "metadata": dict(metadata or {}),
            }
    except Exception:
        actual_ids = set()
        if expected_ids:
            failures.append("summary_collection_missing")
    if actual_ids != expected_ids:
        failures.append("summary_index_id_mismatch")
    content_mismatches = []
    for summary_id in sorted(expected_ids & actual_ids):
        expected = expected_records[summary_id]
        actual = actual_records.get(summary_id) or {}
        actual_metadata = actual.get("metadata") or {}
        if (
            actual.get("document") != expected["document"]
            or any(
                actual_metadata.get(key) != value
                for key, value in expected["metadata"].items()
            )
        ):
            content_mismatches.append(summary_id)
    if content_mismatches:
        failures.append("summary_index_content_mismatch")

    return {
        "schema_version": "structure-summary-audit-1",
        "collection": collection_name,
        "passed": not failures,
        "failures": list(dict.fromkeys(failures)),
        "counts": {
            "structures": len(structures),
            "summary_statuses": len(statuses),
            "summary_rows": len(rows),
            "expected_index_rows": len(expected_ids),
            "actual_index_rows": len(actual_ids),
        },
        "details": {
            "missing_status_items": missing_status,
            "failed_status_items": failed_status,
            "stale_source_items": stale_rows,
            "stale_prompt_nodes": stale_prompt_rows,
            "invalid_text_nodes": invalid_text_rows,
            "invalid_searchable_nodes": invalid_searchable_rows,
            "missing_index_ids": sorted(expected_ids - actual_ids),
            "unexpected_index_ids": sorted(actual_ids - expected_ids),
            "content_mismatch_index_ids": content_mismatches,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default="zotero_paragraphs_v3")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.collection != V3_COLLECTION:
        parser.error(
            f"--collection must be {V3_COLLECTION!r}; the legacy data plane is retired"
        )
    report = build_report(collection_name=args.collection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], **report["counts"]}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
