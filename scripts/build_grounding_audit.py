#!/usr/bin/env python3
"""Render a complete human-audit worksheet from a no-write summary comparison."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _cell(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ").strip()


def render_audit(report: dict[str, Any]) -> str:
    sections = [section for item in report.get("items", []) for section in item.get("sections", [])]
    statuses = Counter(section.get("status", "unknown") for section in sections)
    lines = [
        "# Grounding再テスト監査資料", "",
        f"- 作成元: `{report.get('source_file', 'comparison JSON')}`",
        f"- モデル: `{report.get('llm', 'unknown')}`",
        f"- stop reason: `{report.get('stop_reason', 'unknown')}`",
        f"- generated: {statuses.get('generated', 0)} / skipped_non_content: "
        f"{statuses.get('skipped_non_content', 0)}", "",
        "判定欄は、人間が `OK` / `mis-support` / `fabrication` / `miss` を記入するためのもの。",
        "", "## 資料・要約散文", "",
        "| item | section | status | chapter | 要約散文 | 人間判定 |", "|---|---|---|---|---|---|",
    ]
    for item in report.get("items", []):
        for section in item.get("sections", []):
            lines.append(
                f"| {_cell(item.get('item_key'))} | {_cell(section.get('section_id'))} | "
                f"{_cell(section.get('status'))} | {_cell(section.get('chapter'))} | "
                f"{_cell(section.get('llm_summary'))} | |"
            )
    lines.extend(["", "## 保存候補となる構造化フィールド", ""])
    for item in report.get("items", []):
        for section in item.get("sections", []):
            if section.get("status") != "generated":
                continue
            lines.extend([
                f"### {_cell(item.get('item_key'))} / {_cell(section.get('section_id'))}", "",
                "| 種別 | 値 | evidence_quote | chunk_id | 人間判定 |", "|---|---|---|---|---|",
            ])
            for case in section.get("cases") or []:
                lines.append(
                    f"| case | {_cell(case.get('description'))} | {_cell(case.get('evidence_quote'))} | "
                    f"{_cell(case.get('chunk_id'))} | |"
                )
            for author in section.get("chapter_authors") or []:
                lines.append(
                    f"| chapter_author | {_cell(author.get('name'))} | "
                    f"{_cell(author.get('evidence_quote'))} | | |"
                )
            publication = section.get("first_publication_note")
            if publication:
                lines.append(
                    f"| first_publication | {_cell(publication.get('note'))} | "
                    f"{_cell(publication.get('evidence_quote'))} | | |"
                )
            verification = section.get("verification") or {}
            lines.extend([
                "", f"破棄率: {verification.get('discard_rate', 0):.1%} / "
                f"suspicious: `{bool(verification.get('suspicious_section'))}`", "",
            ])
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    report["source_file"] = str(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_audit(report), encoding="utf-8")
    print(json.dumps({"output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
