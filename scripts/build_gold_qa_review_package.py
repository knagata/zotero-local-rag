#!/usr/bin/env python3
"""Build an evidence-backed, no-cloud-safe Gold QA worksheet for Claude."""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chunk_store import get_item_chunks
from src.env_utils import load_dotenv_native
from src.reference_agent import _item_excluded

DEFAULT_DB = ROOT / "data" / "relations.db"
DEFAULT_OUTPUT = ROOT / "dev-notes" / "gold_qa_review_package.json"


def choose_cases(rows: list[dict[str, Any]], *, count: int, seed: int, per_item: int = 2) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["item_key"])].append(row)
    rng = random.Random(seed)
    for candidates in grouped.values():
        rng.shuffle(candidates)
    item_keys = sorted(grouped)
    rng.shuffle(item_keys)
    selected = []
    for round_index in range(per_item):
        for item_key in item_keys:
            if len(selected) >= count:
                return selected
            if len(grouped[item_key]) > round_index:
                selected.append(grouped[item_key][round_index])
    return selected


def build_candidates(
    db_path: Path, *, count: int, seed: int,
    policy: Callable[[str], tuple[bool, str | None]],
) -> tuple[list[dict[str, Any]], list[dict[str, str | None]]]:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = [dict(row) for row in connection.execute('''
            SELECT case_id, item_key, section_id, description, evidence_quote, chunk_id, model
            FROM case_annotations
            WHERE evidence_quote IS NOT NULL AND evidence_quote != ''
              AND chunk_id IS NOT NULL AND chunk_id != ''
              AND model LIKE '%luna%'
            ORDER BY case_id
        ''').fetchall()]
    finally:
        connection.close()
    decisions = {key: policy(key) for key in sorted({str(row["item_key"]) for row in rows})}
    allowed = [row for row in rows if not decisions[str(row["item_key"])][0]]
    excluded = [
        {"item_key": key, "reason": reason}
        for key, (blocked, reason) in decisions.items() if blocked
    ]
    enriched = []
    chunks_by_item: dict[str, list[dict[str, Any]]] = {}
    for row in allowed:
        item_key = str(row["item_key"])
        chunks = chunks_by_item.setdefault(item_key, get_item_chunks(item_key))
        index = next((i for i, chunk in enumerate(chunks) if chunk["id"] == row["chunk_id"]), None)
        if index is None:
            continue
        evidence_chunks = chunks[index:index + 2]
        source = "\n\n".join(str(chunk.get("text") or "") for chunk in evidence_chunks)
        quote = str(row["evidence_quote"])
        normalized_source = " ".join(source.split())
        if " ".join(quote.split()) not in normalized_source:
            continue
        metadata = chunks[0].get("metadata", {}) if chunks else {}
        enriched.append({
            **row, "candidate_id": f"case-{row['case_id']}",
            "item_title": metadata.get("title"), "language": metadata.get("lang"),
            "evidence_chunk_ids": [chunk["id"] for chunk in evidence_chunks],
            "source_excerpt": source[:8000],
        })
    return choose_cases(enriched, count=count, seed=seed), excluded


def write_package(
    candidates: list[dict[str, Any]], excluded: list[dict[str, str | None]],
    *, output: Path, seed: int,
) -> dict[str, Any]:
    payload = {
        "purpose": "検索評価用Gold QA候補の独立レビュー",
        "seed": seed, "candidate_count": len(candidates), "excluded_items": excluded,
        "instructions": [
            "source_excerptとevidence_quoteを読み、descriptionが原文に支持されるか確認する。",
            "支持される場合、書名や長い原文句をそのまま使わず、利用者が検索で尋ねそうな自然なqueryを1つ作る。",
            "query単独でこの事例を探す意味が薄い、曖昧すぎる、または根拠が不十分ならexcludeする。",
            "expected_item_keysは候補のitem_keyだけ、evidence_chunk_idsには指定された先頭chunk_idを含める。",
            "同じ論点・ほぼ同じqueryを重複採用しない。最終的に20〜30件をincludeする。",
        ],
        "response_format": {"gold_qa": [{
            "candidate_id": "case-N", "decision": "include|exclude", "query": "string|null",
            "expected_item_keys": ["ITEMKEY"], "evidence_chunk_ids": ["CHUNKID"], "note": "string",
        }]},
        "candidates": candidates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output), "candidates": len(candidates), "excluded_items": len(excluded)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    load_dotenv_native(ROOT)
    candidates, excluded = build_candidates(
        args.db, count=args.count, seed=args.seed, policy=_item_excluded,
    )
    print(json.dumps(write_package(candidates, excluded, output=args.output, seed=args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
