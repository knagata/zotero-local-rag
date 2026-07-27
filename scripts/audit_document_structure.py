#!/usr/bin/env python3
"""Inspect a persisted document structure and verify source-chunk coverage."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chunk_store import get_item_chunks
from src.db_relations import get_document_nodes, get_document_structure
from src.document_structure import build_document_structure, validate_structure


def _tree_lines(nodes: list[dict]) -> list[str]:
    lines = []
    for node in nodes:
        indent = "  " * int(node.get("depth") or 0)
        title = node.get("title") or "(content)"
        chunks = node.get("chunks") or []
        suffix = f" [{len(chunks)} chunks]" if chunks else ""
        zone = f" zone={node.get('zone')}" if chunks else ""
        lines.append(f"{indent}- {node.get('node_type')}: {title}{suffix}{zone}")
    return lines


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not value.get("id"):
            raise ValueError(f"invalid chunk row at {path}:{line_number}")
        rows.append(value)
    return rows


def audit_metrics(nodes: list[dict], chunks: list[dict], *, baseline: list[dict] | None = None) -> dict:
    audit = validate_structure(nodes, chunks)
    zone_chunk_counts: dict[str, int] = {}
    zone_chars: dict[str, int] = {}
    chunk_by_id = {str(row.get("id") or ""): row for row in chunks}
    for node in nodes:
        if not node.get("chunks"):
            continue
        zone = str(node.get("zone") or "body")
        ids = [str(row.get("chunk_id") if isinstance(row, dict) else row) for row in node["chunks"]]
        zone_chunk_counts[zone] = zone_chunk_counts.get(zone, 0) + len(ids)
        zone_chars[zone] = zone_chars.get(zone, 0) + sum(
            len(str((chunk_by_id.get(chunk_id) or {}).get("text") or "")) for chunk_id in ids
        )
    total_chars = sum(len(str(row.get("text") or "")) for row in chunks)
    metrics = {
        **audit, "chunk_count": len(chunks), "text_chars": total_chars,
        "zone_chunk_counts": dict(sorted(zone_chunk_counts.items())),
        "zone_chars": dict(sorted(zone_chars.items())),
    }
    if baseline is not None:
        baseline_chars = sum(len(str(row.get("text") or "")) for row in baseline)
        metrics["comparison"] = {
            "baseline_chunks": len(baseline), "candidate_chunks": len(chunks),
            "chunk_delta": len(chunks) - len(baseline),
            "baseline_chars": baseline_chars, "candidate_chars": total_chars,
            "char_delta": total_chars - baseline_chars,
            "char_ratio": (total_chars / baseline_chars) if baseline_chars else None,
        }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item", required=True)
    parser.add_argument("--format", choices=("tree", "json"), default="tree")
    parser.add_argument("--check-coverage", action="store_true")
    parser.add_argument(
        "--candidate-jsonl", type=Path,
        help="Audit candidate V3 chunks without writing DB; compare them with the active chunks.",
    )
    args = parser.parse_args()
    baseline = get_item_chunks(args.item)
    if args.candidate_jsonl:
        chunks = _load_jsonl(args.candidate_jsonl)
        built = build_document_structure(args.item, chunks)
        structure, nodes = {key: value for key, value in built.items() if key != "nodes"}, built["nodes"]
        audit = audit_metrics(nodes, chunks, baseline=baseline)
    else:
        structure = get_document_structure(args.item)
        nodes = get_document_nodes(args.item, include_chunks=True)
        if structure is None:
            print(json.dumps({"item_key": args.item, "error": "structure_not_found"}, ensure_ascii=False))
            raise SystemExit(2)
        chunks = baseline
        audit = audit_metrics(nodes, chunks)
    payload = {"structure": structure, "audit": audit, "nodes": nodes}
    if args.format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"{args.item}: {structure['status']} / {structure['node_count']} nodes")
        print("\n".join(_tree_lines(nodes)))
        print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.check_coverage and not audit["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
