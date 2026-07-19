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
from src.document_structure import validate_structure


def _tree_lines(nodes: list[dict]) -> list[str]:
    lines = []
    for node in nodes:
        indent = "  " * int(node.get("depth") or 0)
        title = node.get("title") or "(content)"
        chunks = node.get("chunks") or []
        suffix = f" [{len(chunks)} chunks]" if chunks else ""
        lines.append(f"{indent}- {node.get('node_type')}: {title}{suffix}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item", required=True)
    parser.add_argument("--format", choices=("tree", "json"), default="tree")
    parser.add_argument("--check-coverage", action="store_true")
    args = parser.parse_args()
    structure = get_document_structure(args.item)
    nodes = get_document_nodes(args.item, include_chunks=True)
    if structure is None:
        print(json.dumps({"item_key": args.item, "error": "structure_not_found"}, ensure_ascii=False))
        raise SystemExit(2)
    audit = validate_structure(nodes, get_item_chunks(args.item))
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
