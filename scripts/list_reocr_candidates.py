#!/usr/bin/env python3
"""Rank re-OCR candidates without modifying the chunk store or relations DB."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chunk_store import get_item_chunks
from src.manifest import load_manifest


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").split())


def _majority_language(chunks: list[dict[str, Any]]) -> str:
    languages = [
        str(chunk.get("metadata", {}).get("lang") or "").strip().casefold()
        for chunk in chunks
    ]
    languages = [language for language in languages if language]
    return Counter(languages).most_common(1)[0][0] if languages else "unknown"


def _quote_location(quote: str, chunks: list[dict[str, Any]]) -> str:
    needle = _normalize(quote)
    if not needle:
        return "missing"
    texts = [_normalize(chunk.get("text")) for chunk in chunks]
    if any(needle in text for text in texts):
        return "single"
    if any(needle in f"{first} {second}" for first, second in zip(texts, texts[1:])):
        return "adjacent_pair"
    return "missing"


def _grounding_stats(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "total_generated": 0, "total_discarded": 0, "suspicious_sections": 0,
        "evidence_not_in_chunk": 0, "cross_chunk_evidence": 0,
    })
    for item in report.get("items", []):
        item_key = str(item.get("item_key") or "")
        for section in item.get("sections", []):
            verification = section.get("verification") or {}
            current = stats[item_key]
            current["total_generated"] += int(verification.get("total_generated") or 0)
            current["total_discarded"] += int(verification.get("total_discarded") or 0)
            current["suspicious_sections"] += int(bool(verification.get("suspicious_section")))
            for bucket_name in ("cases", "chapter_authors", "first_publication_note"):
                reasons = (verification.get(bucket_name) or {}).get("reasons") or {}
                current["evidence_not_in_chunk"] += int(reasons.get("evidence_not_in_chunk") or 0)
    return stats


def build_candidates(
    manifest: dict[str, Any], report: dict[str, Any],
    *, chunk_loader: Callable[[str], list[dict[str, Any]]] = get_item_chunks,
) -> list[dict[str, Any]]:
    """Combine manifest quality, grounding failures, and chunk language."""
    grounding = _grounding_stats(report)
    rows: list[dict[str, Any]] = []
    item_keys = sorted({
        str(item.get("item_key") or "") for item in report.get("items", []) if item.get("item_key")
    })
    for item_key in item_keys:
        chunks = chunk_loader(item_key)
        by_attachment: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for chunk in chunks:
            attachment_key = str(chunk.get("metadata", {}).get("attachmentKey") or "unknown")
            by_attachment[attachment_key].append(chunk)

        # Quotes crossing a boundary are useful OCR/layout signals even when the
        # relaxed grounding verifier can retain them safely.
        item_report = next(
            (item for item in report.get("items", []) if item.get("item_key") == item_key), {},
        )
        cases = [
            case for section in item_report.get("sections", []) for case in section.get("cases", [])
        ]
        primary_attachment = max(
            by_attachment, key=lambda key: len(by_attachment[key]), default="unknown",
        )
        crossed_by_attachment = Counter()
        missing_from_all = 0
        for case in cases:
            locations = {
                attachment_key: _quote_location(
                    str(case.get("evidence_quote") or ""), attachment_chunks,
                )
                for attachment_key, attachment_chunks in by_attachment.items()
            }
            for attachment_key, location in locations.items():
                if location == "adjacent_pair":
                    crossed_by_attachment[attachment_key] += 1
            if locations and all(location == "missing" for location in locations.values()):
                missing_from_all += 1
        for attachment_key, attachment_chunks in by_attachment.items():
            crossed = crossed_by_attachment[attachment_key]
            quality = (manifest.get("files", {}).get(attachment_key, {}).get("quality") or {})
            current = dict(grounding.get(item_key, {}))
            current["cross_chunk_evidence"] = crossed
            if attachment_key == primary_attachment:
                current["evidence_not_in_chunk"] = int(
                    current.get("evidence_not_in_chunk") or 0
                ) + missing_from_all
                generated = int(current.get("total_generated") or 0)
                discarded = int(current.get("total_discarded") or 0)
            else:
                current["evidence_not_in_chunk"] = 0
                current["suspicious_sections"] = 0
                generated = discarded = 0
            lang = _majority_language(attachment_chunks)
            is_scanned = bool(quality.get("is_scanned"))
            is_corrupted = bool(quality.get("is_corrupted"))
            signals = (
                is_scanned or is_corrupted or current.get("suspicious_sections")
                or current.get("evidence_not_in_chunk") or crossed
            )
            if not signals:
                continue
            if lang == "ja":
                recommendation = "benchmark_ja_ocr"
            elif is_scanned or is_corrupted or current.get("evidence_not_in_chunk"):
                recommendation = "docling_reparse"
            else:
                recommendation = "inspect_grounding"
            score = (
                6 * int(is_corrupted) + 5 * int(is_scanned)
                + 4 * int(current.get("suspicious_sections") or 0)
                + 2 * int(current.get("evidence_not_in_chunk") or 0) + crossed
            )
            rows.append({
                "item_key": item_key, "attachment_key": attachment_key, "lang": lang,
                "parser": str(quality.get("parser") or "unknown"),
                "is_scanned": is_scanned, "is_corrupted": is_corrupted,
                "discard_rate": round(discarded / generated, 4) if generated else 0.0,
                "suspicious_sections": int(current.get("suspicious_sections") or 0),
                "evidence_not_in_chunk": int(current.get("evidence_not_in_chunk") or 0),
                "cross_chunk_evidence": crossed, "score": score,
                "recommendation": recommendation,
            })
    return sorted(rows, key=lambda row: (-row["score"], row["item_key"], row["attachment_key"]))


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# 再OCR候補", "",
        "この一覧は品質フラグ、grounding破棄統計、言語、chunk跨ぎquoteを統合した読み取り専用の候補表。",
        "", "| rank | itemKey | attachmentKey | lang | parser | scanned | corrupted | discard | suspicious | missing chunk | cross chunk | score | 推奨処理 |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(rows, 1):
        lines.append(
            f"| {rank} | {row['item_key']} | {row['attachment_key']} | {row['lang']} | "
            f"{row['parser']} | {int(row['is_scanned'])} | {int(row['is_corrupted'])} | "
            f"{row['discard_rate']:.1%} | {row['suspicious_sections']} | "
            f"{row['evidence_not_in_chunk']} | {row['cross_chunk_evidence']} | "
            f"{row['score']} | {row['recommendation']} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest.json")
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    if not args.json_output and not args.markdown_output:
        parser.error("at least one of --json-output or --markdown-output is required")
    manifest = load_manifest(args.manifest)
    report = json.loads(args.comparison.read_text(encoding="utf-8"))
    rows = build_candidates(manifest, report)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps({"source": str(args.comparison), "candidates": rows}, ensure_ascii=False, indent=2)
            + "\n", encoding="utf-8",
        )
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(rows), encoding="utf-8")
    print(json.dumps({"candidates": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
