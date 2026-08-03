#!/usr/bin/env python3
"""Plan, evaluate, and explicitly adopt a per-item V3 re-OCR result.

The default is always dry-run.  ``--adopt`` is deliberately limited to one
item and consumes a previously prepared result after the deterministic gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_utils import load_dotenv_native

# 他のエントリポイントと同じく、環境変数を読む前に .env を取り込む。これが
# 無いと CHROMA_DIR / MANIFEST_PATH / LEXICAL_DB_PATH は常に未設定に見え、
# 設定に関係なく既定値だけで動く。
load_dotenv_native(ROOT)

from src.chunk_store import get_item_chunks  # noqa: E402
from src.db_relations import mark_artifact_status  # noqa: E402
from src.embedder import get_collection  # noqa: E402
from src.reocr_adoption import adopt_prepared_reocr  # noqa: E402
from src.reocr_quality import (  # noqa: E402
    evaluate_adoption_gate, majority_language, text_metrics,
)
from src.v3_data_plane import resolve_configured_path  # noqa: E402


def selected_rows(path: Path, limit: int, item_key: str | None = None) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("candidates")
    if not isinstance(rows, list):
        raise ValueError("candidate JSON must contain a candidates array")
    selected = [row for row in rows if isinstance(row, dict)]
    if item_key:
        selected = [row for row in selected if str(row.get("item_key") or "") == item_key]
    return selected[:limit]


def _prepared_results(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError("result JSON must contain a results array")
    return {
        (str(row.get("item_key") or ""), str(row.get("attachment_key") or "")): row
        for row in rows if isinstance(row, dict)
    }


def compare_result(
    row: dict[str, Any], prepared: dict[str, Any], *,
    chunk_loader=get_item_chunks, min_character_ratio: float = 0.8,
    max_character_ratio: float = 1.5,
) -> dict[str, Any]:
    item_key = str(row.get("item_key") or "")
    attachment_key = str(row.get("attachment_key") or "")
    old_chunks = [
        chunk for chunk in chunk_loader(item_key)
        if str((chunk.get("metadata") or {}).get("attachmentKey") or "") == attachment_key
    ]
    blocks = prepared.get("blocks")
    if not isinstance(blocks, list):
        raise ValueError(f"prepared result for {attachment_key} must contain blocks")
    quality = prepared.get("quality") if isinstance(prepared.get("quality"), dict) else {}
    expected_pages = int(
        quality.get("total_pages") or row.get("quality", {}).get("total_pages") or 0
    )
    before = text_metrics(old_chunks, total_pages=expected_pages)
    after = text_metrics(blocks, total_pages=expected_pages)
    if quality.get("page_coverage") is not None:
        after["page_coverage"] = float(quality["page_coverage"])
    gate = evaluate_adoption_gate(
        before, after, min_character_ratio=min_character_ratio,
        max_character_ratio=max_character_ratio,
    )
    before_language = majority_language(old_chunks)
    after_language = majority_language(blocks)
    return {
        "item_key": item_key, "attachment_key": attachment_key,
        "engine": prepared.get("engine") or row.get("target_engine"),
        "version": prepared.get("version") or row.get("target_version"),
        "before": before, "after": after,
        "language": {
            "before": before_language, "after": after_language,
            "matches": before_language == "unknown" or after_language == before_language,
        },
        "structure": {
            "before": row.get("structure_status") or "unavailable",
            "after": prepared.get("structure_status") or "unavailable",
            "flat_fallback_resolved": (
                row.get("structure_status") == "flat_fallback"
                and prepared.get("structure_status") in {"exact", "recovered"}
            ),
            "heading_count_before": int(row.get("heading_count") or 0),
            "heading_count_after": int(prepared.get("heading_count") or 0),
        },
        "quality_gate": gate,
    }


def validate_adoption_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.force_adopt and not args.item:
        parser.error("--force-adopt requires exactly one --item KEY")
    if args.force_adopt:
        args.adopt = True
    if args.adopt and not args.item:
        parser.error("--adopt requires exactly one --item KEY")
    if args.adopt and args.limit != 1:
        parser.error("--adopt requires --limit 1")
    if args.adopt and args.results is None:
        parser.error("--adopt requires --results from a completed dry-run extraction")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--results", type=Path, help="Prepared, non-canonical V3 extraction results")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--item", help="Process exactly one itemKey")
    parser.add_argument("--adopt", action="store_true", help="Adopt one gate-passing result into the V3 indexes")
    parser.add_argument("--force-adopt", action="store_true", help="Human override; requires --item and --limit 1")
    parser.add_argument("--min-character-ratio", type=float, default=0.8)
    parser.add_argument("--max-character-ratio", type=float, default=1.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be positive")
    if not 0 < args.min_character_ratio <= args.max_character_ratio:
        parser.error("character ratio thresholds are invalid")
    validate_adoption_args(args, parser)
    rows = selected_rows(args.candidates, args.limit, args.item)
    if not rows:
        parser.error("candidate queue is empty for the selected scope")
    prepared = _prepared_results(args.results)
    comparisons = []
    for row in rows:
        key = (str(row.get("item_key") or ""), str(row.get("attachment_key") or ""))
        if key not in prepared:
            comparisons.append({
                "item_key": key[0], "attachment_key": key[1],
                "status": "awaiting_dry_run_extraction", "quality_gate": None,
            })
            continue
        comparison = compare_result(
            row, prepared[key], min_character_ratio=args.min_character_ratio,
            max_character_ratio=args.max_character_ratio,
        )
        comparison["status"] = "evaluated"
        comparisons.append(comparison)

    failures = [
        row for row in comparisons
        if row.get("quality_gate") is not None and not row["quality_gate"]["passed"]
    ]
    awaiting = [row for row in comparisons if row.get("quality_gate") is None]
    authorized = bool(args.adopt and not awaiting and (args.force_adopt or not failures))
    report = {
        "schema_version": "reocr-comparison-v3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": not bool(args.adopt),
        "force_adopt": bool(args.force_adopt),
        "adoption_authorized": authorized,
        "canonical_data_modified": False,
        "comparisons": comparisons,
    }
    if authorized:
        comparison = comparisons[0]
        selected = rows[0]
        key = (str(selected.get("item_key") or ""), str(selected.get("attachment_key") or ""))
        collection_name = os.environ.get("CHROMA_COLLECTION_V3", "zotero_paragraphs_v3")
        # 変数名は MANIFEST_PATH / LEXICAL_DB_PATH。以前ここだけ
        # MANIFEST_V3_PATH / LEXICAL_V3_DB_PATH というリポジトリ内のどこにも
        # 存在しない名前を読んでおり、設定を常に無視して既定値へ落ちていた
        # （2026-08-04）。解決は resolve_configured_path に委ねる。
        chroma_dir = resolve_configured_path(
            ROOT, os.environ.get("CHROMA_DIR") or ROOT / "data" / "chroma")
        manifest_path = resolve_configured_path(
            ROOT, os.environ.get("MANIFEST_PATH") or ROOT / "data" / "manifest_v3.json")
        lexical_path = resolve_configured_path(
            ROOT, os.environ.get("LEXICAL_DB_PATH") or ROOT / "data" / "lexical_v3.sqlite3")
        collection = get_collection(
            chroma_dir=chroma_dir, project_root=ROOT,
            chroma_collection_env=collection_name,
            chroma_collection_default="zotero_paragraphs_v3",
            persist_active_config=False,
        )
        old_chunks = get_item_chunks(key[0], chroma_dir=chroma_dir, collection_name=collection_name)
        report["adoption"] = adopt_prepared_reocr(
            item_key=key[0], attachment_key=key[1], prepared=prepared[key],
            collection=collection, old_item_chunks=old_chunks,
            manifest_path=manifest_path, lexical_path=lexical_path,
            force=bool(args.force_adopt),
            gate_passed=bool(comparison["quality_gate"]["passed"]),
        )
        report["canonical_data_modified"] = True
    elif args.adopt and failures:
        failure = failures[0]
        mark_artifact_status(
            str(failure.get("item_key") or ""), "extraction", "degraded",
            attachment_key=str(failure.get("attachment_key") or ""),
            reason_code="ocr_below_threshold", counts=failure.get("quality_gate"),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "dry_run": report["dry_run"], "evaluated": len(comparisons) - len(awaiting),
        "awaiting": len(awaiting), "failed": len(failures), "adoption_authorized": authorized,
    }, ensure_ascii=False))
    if args.adopt and not authorized:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
