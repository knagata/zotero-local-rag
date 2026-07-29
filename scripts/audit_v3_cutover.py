#!/usr/bin/env python3
"""Deterministically compare legacy and V3 paragraph collections for cutover."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence, Set

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_utils import load_dotenv_native
load_dotenv_native(ROOT)

from src.chunk_store import (
    get_item_chunks, list_attachment_keys, list_chunk_ids,
    list_chunk_ids_without_attachment, list_chunk_ids_without_item, list_item_keys,
)
from src.db_relations import get_document_structure
from src.source_verification import dangling_node_ids, unretrievable_documents
from src.document_structure import STRUCTURE_VERSION, source_fingerprint
from src.lexical_index import list_chunk_ids as list_lexical_chunk_ids
from src.manifest import load_manifest
from src.source_coverage import validate_source_coverage
from src.database_gate import database_state_fingerprint, external_audit_attestation
from src.v3_data_plane import V3_COLLECTION


def source_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Notes are searchable annotations, not canonical attachment structure."""
    return [
        row for row in rows
        if str((row.get("metadata") or {}).get("source_type") or "") != "note"
    ]


def item_metrics(
    rows: Sequence[Mapping[str, Any]], live_node_ids: Set[str] | None = None,
) -> dict[str, Any]:
    rows = source_rows(rows)
    ids = [str(row.get("id") or "") for row in rows]
    zones = Counter(str((row.get("metadata") or {}).get("zone") or "missing") for row in rows)
    # Coverage must mean the node exists, not that the string is non-empty.
    # Counting the string let 53 items score 1.0 while every one of their
    # chunks named a node that had been replaced by a rebuild -- the gate's
    # most substantive check passing on nothing at all (2026-07-28). Uses the
    # single definition in source_verification rather than a second copy here:
    # two copies of "does this node_id exist" is exactly the kind of split
    # that let the string-only version go unnoticed as long as it did.
    dangling = set(dangling_node_ids(rows, live_node_ids or set())) if live_node_ids is not None else set()
    assigned = sum(
        bool((row.get("metadata") or {}).get("node_id")) and str(row.get("id") or "") not in dangling
        for row in rows
    )
    return {
        "chunks": len(rows),
        "characters": sum(len(str(row.get("text") or "")) for row in rows),
        "duplicate_chunk_ids": len(ids) - len(set(ids)),
        "node_assigned_chunks": assigned,
        "node_coverage": round(assigned / len(rows), 6) if rows else 0.0,
        "zones": dict(sorted(zones.items())),
    }


def compare_item(
    item_key: str, old_rows: Sequence[Mapping[str, Any]], new_rows: Sequence[Mapping[str, Any]],
    live_node_ids: Set[str] | None = None,
) -> dict[str, Any]:
    old_source_rows = source_rows(old_rows)
    new_source_rows = source_rows(new_rows)
    old = item_metrics(old_rows)
    new = item_metrics(new_rows, live_node_ids)
    old_chars = int(old["characters"])
    is_new_item = not bool(old_source_rows)
    ratio = (int(new["characters"]) / old_chars) if old_chars else None
    failures = []
    if old_source_rows and not new_source_rows:
        failures.append("missing_v3_item")
    if new["duplicate_chunk_ids"]:
        failures.append("duplicate_v3_chunk_ids")
    if new_source_rows and new["node_coverage"] != 1.0:
        failures.append("incomplete_node_coverage")
    # zone was reported but never asserted, and retrieval_policy appeared in
    # this file not at all -- which is how a 65,000-character essay marked
    # zone="index" throughout left search entirely while the gate passed
    # (ND49KK4N, 2026-07-28). Same reasoning as above: one definition, in
    # source_verification, used by both the audit and the standalone check.
    if unretrievable_documents({item_key: new_source_rows}):
        failures.append("no_retrievable_chunks")
    return {
        "item_key": item_key, "old": old, "new": new,
        "delta": {
            "chunks": int(new["chunks"]) - int(old["chunks"]),
            "characters": int(new["characters"]) - int(old["characters"]),
            "new_item": is_new_item,
            "character_ratio": round(ratio, 6) if ratio is not None else None,
            "character_ratio_outlier": bool(ratio is not None and (ratio < 0.8 or ratio > 1.5)),
        },
        "passed": not failures, "failures": failures,
    }


def structure_failures(
    structure: Mapping[str, Any] | None, new_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    rows = source_rows(new_rows)
    if not rows:
        return []
    if not structure:
        return ["missing_v3_structure"]
    failures = []
    if str(structure.get("structure_version") or "") != STRUCTURE_VERSION:
        failures.append("stale_structure_version")
    expected = source_fingerprint(list(rows))
    if str(structure.get("source_fingerprint") or "") != expected:
        failures.append("stale_structure_fingerprint")
    if str(structure.get("status") or "") == "unavailable":
        failures.append("unavailable_v3_structure")
    expected_attachments = {
        str((row.get("metadata") or {}).get("attachmentKey") or "")
        for row in rows
        if str((row.get("metadata") or {}).get("attachmentKey") or "")
    }
    nodes = structure.get("nodes") if isinstance(structure.get("nodes"), list) else []
    structured_attachments = {
        str(node.get("attachment_key") or "")
        for node in nodes
        if isinstance(node, Mapping)
        and str(node.get("node_type") or "") == "attachment_root"
        and str(node.get("attachment_key") or "")
    }
    if expected_attachments and structured_attachments != expected_attachments:
        failures.append("structure_attachment_coverage_mismatch")
    return failures


def global_gate_failures(
    *, manifest: Mapping[str, Any], manifest_attachment_keys: set[str],
    chroma_attachment_keys: set[str], chroma_ids: set[str], lexical_ids: set[str],
    pipeline_config_exists: bool, chunks_without_item_count: int = 0,
    chunks_without_attachment_count: int = 0,
) -> list[str]:
    failures = []
    if manifest_attachment_keys != chroma_attachment_keys:
        failures.append("manifest_chroma_attachment_mismatch")
    if chroma_ids != lexical_ids:
        failures.append("chroma_lexical_id_mismatch")
    # The per-item comparison above iterates legacy item keys, so a chunk with
    # no itemKey is never examined by it -- 1,774 such chunks answered vector
    # search while every per-item gate passed (2026-07-28). This is the only
    # check in the gate that looks at the whole collection rather than at one
    # item at a time, and is exactly what catches that population.
    if chunks_without_item_count:
        failures.append("chunks_without_item")
    if chunks_without_attachment_count:
        failures.append("chunks_without_attachment")
    if not manifest.get("pipeline_fingerprint") or not pipeline_config_exists:
        failures.append("missing_v3_pipeline_provenance")
    if manifest.get("hnsw_validated") is not True:
        failures.append("hnsw_not_validated")
    if manifest.get("post_index_pending"):
        failures.append("post_index_pending")
    if manifest.get("inflight_attachments"):
        failures.append("inflight_attachments")
    files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    if any(
        not isinstance(entry, Mapping)
        or not isinstance(entry.get("quality"), Mapping)
        or not validate_source_coverage(entry["quality"].get("source_coverage"))["passed"]
        for entry in files.values()
    ):
        failures.append("incomplete_source_coverage")
    # A worker may report its timeout guard as having fired after it has
    # nevertheless returned every expected page.  Full page coverage is not a
    # partial extraction and must not block cutover merely due to that stale
    # diagnostic flag.
    if any(
        isinstance(entry, dict)
        and isinstance(entry.get("quality"), dict)
        and bool(entry["quality"].get("truncated_by_timeout"))
        and not bool(entry["quality"].get("truncated_approved"))
        and int(entry["quality"].get("processed_pages") or 0)
            < int(entry["quality"].get("expected_pages") or entry["quality"].get("total_pages") or 0)
        for entry in files.values()
    ):
        failures.append("unapproved_truncated_pdf")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-collection", default="zotero_paragraphs")
    parser.add_argument("--new-collection", default="zotero_paragraphs_v3")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest_v3.json")
    parser.add_argument("--lexical-db", type=Path, default=ROOT / "data" / "lexical_v3.sqlite3")
    parser.add_argument(
        "--pipeline-config", type=Path,
        default=ROOT / "data" / "chroma" / "embedder_config_v3.json",
    )
    parser.add_argument(
        "--zotero-report", type=Path,
        help="Passing full Zotero-to-manifest reconciliation report (required with --new-only).",
    )
    parser.add_argument(
        "--source-report", type=Path,
        help="Passing original-source verification report (required with --new-only).",
    )
    parser.add_argument("--item", action="append")
    parser.add_argument(
        "--new-only", action="store_true",
        help="Audit the rebuilt target as a standalone database and emit a summary gate.",
    )
    parser.add_argument(
        "--exclude-item", action="append", default=[],
        help="Exclude a legacy item intentionally removed from the current source.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.new_collection != V3_COLLECTION:
        parser.error(
            f"--new-collection must be {V3_COLLECTION!r}; the legacy data plane is retired"
        )
    external_audits: dict[str, dict[str, str]] = {}
    if args.new_only:
        if not args.zotero_report or not args.source_report:
            parser.error("--new-only requires --zotero-report and --source-report")
        try:
            zotero_report = json.loads(args.zotero_report.read_text(encoding="utf-8"))
            source_report = json.loads(args.source_report.read_text(encoding="utf-8"))
            if not isinstance(zotero_report, Mapping) or not isinstance(source_report, Mapping):
                raise RuntimeError("External audit reports must be JSON objects")
            expected_manifest = str(args.manifest.resolve())
            if str(zotero_report.get("manifest_path") or "") != expected_manifest:
                raise RuntimeError("Zotero reconciliation report targets a different manifest")
            if str(source_report.get("manifest_path") or "") != expected_manifest:
                raise RuntimeError("Source verification report targets a different manifest")
            if str(source_report.get("collection") or "") != args.new_collection:
                raise RuntimeError("Source verification report targets a different collection")
            external_audits = {
                "zotero_reconciliation": external_audit_attestation(args.zotero_report),
                "source_verification": external_audit_attestation(args.source_report),
            }
        except (OSError, ValueError, RuntimeError) as exc:
            parser.error(str(exc))
    excluded = {str(value) for value in args.exclude_item if value}
    old_keys = [] if args.new_only else [
        key for key in list_item_keys(collection_name=args.old_collection)
        if key not in excluded
    ]
    new_keys = [
        key for key in list_item_keys(collection_name=args.new_collection)
        if key not in excluded
    ]
    audited_keys = sorted(set(old_keys) | set(new_keys))
    selected = list(dict.fromkeys(args.item or audited_keys))
    if args.limit > 0:
        selected = selected[:args.limit]
    comparisons = []
    attested_chroma_rows: list[Mapping[str, Any]] = []
    structure_statuses: Counter[str] = Counter()
    zones: Counter[str] = Counter()
    # The set every chunk's node_id is checked against. Read once: without it
    # node coverage only asserts that a string is present, which is how 53
    # items scored 1.0 with no live structure behind them.
    from src.db_relations import get_db_connection

    connection = get_db_connection()
    try:
        live_node_ids = {
            str(row[0])
            for row in connection.execute("SELECT node_id FROM document_nodes")
        }
        structure_rows = [
            ("structure", *tuple(row)) for row in connection.execute(
                "SELECT item_key, source_fingerprint, structure_version, status, "
                "COALESCE(confidence, ''), node_count, leaf_count, "
                "COALESCE(diagnostics_json, '') "
                "FROM document_structures"
            ).fetchall()
        ]
        structure_rows.extend(
            ("node", *tuple(row)) for row in connection.execute(
                "SELECT item_key, node_id, COALESCE(parent_node_id, ''), node_type, "
                "depth, ordinal, COALESCE(title, ''), COALESCE(normalized_title, ''), "
                "COALESCE(attachment_key, ''), source_kind, "
                "COALESCE(source_locator_json, ''), COALESCE(confidence, ''), "
                "content_chars, zone, summary_policy, retrieval_policy, citation_policy, "
                "COALESCE(first_chunk_id, ''), COALESCE(last_chunk_id, '') "
                ", COALESCE(extraction_engine, ''), COALESCE(extraction_version, '') "
                "FROM document_nodes"
            ).fetchall()
        )
        structure_rows.extend(
            ("mapping", *tuple(row)) for row in connection.execute(
                "SELECT n.item_key, c.node_id, c.chunk_id, c.ordinal "
                "FROM document_node_chunks c "
                "JOIN document_nodes n ON n.node_id = c.node_id"
            ).fetchall()
        )
    finally:
        connection.close()
    for item_key in selected:
        old_rows = (
            [] if args.new_only
            else get_item_chunks(item_key, collection_name=args.old_collection)
        )
        new_rows = get_item_chunks(item_key, collection_name=args.new_collection)
        attested_chroma_rows.extend(new_rows)
        row = compare_item(item_key, old_rows, new_rows, live_node_ids)
        structure = get_document_structure(item_key)
        row["structure"] = {
            "status": str((structure or {}).get("status") or "missing"),
            "source_fingerprint": str((structure or {}).get("source_fingerprint") or ""),
        }
        structure_statuses[row["structure"]["status"]] += 1
        row["failures"].extend(structure_failures(structure, new_rows))
        row["failures"] = list(dict.fromkeys(row["failures"]))
        row["passed"] = not row["failures"]
        zones.update(row["new"]["zones"])
        comparisons.append(row)
    manifest = load_manifest(args.manifest)
    manifest_files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    orphaned_chunk_ids = list_chunk_ids_without_item(collection_name=args.new_collection)
    attachmentless_chunk_ids = list_chunk_ids_without_attachment(
        collection_name=args.new_collection,
    )
    chroma_ids = set(list_chunk_ids(collection_name=args.new_collection))
    lexical_ids = set(list_lexical_chunk_ids(path=args.lexical_db))
    global_failures = global_gate_failures(
        manifest=manifest,
        manifest_attachment_keys=set(str(value) for value in manifest_files),
        chroma_attachment_keys=set(list_attachment_keys(collection_name=args.new_collection)),
        chroma_ids=chroma_ids,
        lexical_ids=lexical_ids,
        pipeline_config_exists=args.pipeline_config.exists(),
        chunks_without_item_count=len(orphaned_chunk_ids),
        chunks_without_attachment_count=len(attachmentless_chunk_ids),
    )
    failed = [row for row in comparisons if not row["passed"]]
    outliers = [row for row in comparisons if row["delta"]["character_ratio_outlier"]]
    report = {
        "schema_version": "v3-cutover-audit-1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "old_collection": args.old_collection, "new_collection": args.new_collection,
        "new_only": bool(args.new_only),
        "gate": {
            "passed": not failed and not global_failures and len(comparisons) == len(audited_keys),
            "selected_items": len(comparisons), "legacy_items": len(old_keys),
            "new_items": len(new_keys), "audited_items": len(audited_keys),
            "failed_items": len(failed), "character_ratio_outliers": len(outliers),
            "coverage_required": 1.0,
            "note": "Character-ratio outliers require review but do not alone fail the deterministic gate.",
            "global_failures": global_failures,
            "chunks_without_item": len(orphaned_chunk_ids),
            "chunks_without_attachment": len(attachmentless_chunk_ids),
        },
        "structure_statuses": dict(sorted(structure_statuses.items())),
        "v3_zones": dict(sorted(zones.items())),
        "items": comparisons,
        "external_audits": external_audits,
        "database_state": {
            "fingerprint": database_state_fingerprint(
                manifest_path=args.manifest,
                pipeline_config_path=args.pipeline_config,
                chroma_rows=attested_chroma_rows,
                lexical_ids=lexical_ids,
                structure_rows=structure_rows,
            ),
            "manifest_path": str(args.manifest.resolve()),
            "lexical_db_path": str(args.lexical_db.resolve()),
            "pipeline_config_path": str(args.pipeline_config.resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["gate"], ensure_ascii=False))
    if failed or global_failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
