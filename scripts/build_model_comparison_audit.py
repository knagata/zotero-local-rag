#!/usr/bin/env python3
"""Build a blind A/B worksheet for summaries shared by two quality reports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _cell(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ").strip()


def _sections(report: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(item.get("item_key")), str(section.get("section_id"))): section
        for item in report.get("items", [])
        for section in item.get("sections", [])
        if section.get("status") == "generated"
    }


def build(
    baseline: dict[str, Any], candidate: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    left, right = _sections(baseline), _sections(candidate)
    shared = sorted(set(left) & set(right))
    lines = [
        "# 要約モデル盲検A/B監査", "",
        "各行について原文を確認し、`A` / `B` / `tie` / `both_fail` を記入する。",
        "評価軸は誤支持・捏造を最優先し、次に重要点の被覆、明瞭さを見る。",
        "モデル対応表は別ファイルにあり、判定完了まで開かない。", "",
        "| item | section | chapter | 要約A | 要約B | 判定 | 根拠・問題点 |",
        "|---|---|---|---|---|---|---|",
    ]
    assignments = []
    for key in shared:
        first, second = left[key], right[key]
        swap = hashlib.sha256(":".join(key).encode()).digest()[0] % 2 == 1
        a, b = (second, first) if swap else (first, second)
        a_model = candidate.get("llm") if swap else baseline.get("llm")
        b_model = baseline.get("llm") if swap else candidate.get("llm")
        chapter = first.get("chapter") or second.get("chapter")
        lines.append(
            f"| {_cell(key[0])} | {_cell(key[1])} | {_cell(chapter)} | "
            f"{_cell(a.get('llm_summary'))} | {_cell(b.get('llm_summary'))} | | |"
        )
        assignments.append({
            "item_key": key[0], "section_id": key[1],
            "A": a_model, "B": b_model,
        })
    key_file = {
        "baseline": baseline.get("llm"), "candidate": candidate.get("llm"),
        "shared_sections": len(shared), "assignments": assignments,
    }
    return "\n".join(lines) + "\n", key_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("worksheet", type=Path)
    parser.add_argument("answer_key", type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    worksheet, answer_key = build(baseline, candidate)
    args.worksheet.parent.mkdir(parents=True, exist_ok=True)
    args.answer_key.parent.mkdir(parents=True, exist_ok=True)
    args.worksheet.write_text(worksheet, encoding="utf-8")
    args.answer_key.write_text(
        json.dumps(answer_key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "worksheet": str(args.worksheet), "answer_key": str(args.answer_key),
        "shared_sections": answer_key["shared_sections"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
