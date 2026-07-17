#!/usr/bin/env python3
"""Build a reproducible random-section worksheet for the Gate 3 human audit."""
from __future__ import annotations

import argparse
import random
import sqlite3
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.build_summaries import _section_source_text, split_sections
from src.chunk_store import get_item_chunks


def choose_rows(rows: list[dict[str, Any]], *, count: int, seed: int) -> list[dict[str, Any]]:
    if count > len(rows):
        raise ValueError(f"requested {count} sections from a population of {len(rows)}")
    chosen = random.Random(seed).sample(rows, count)
    return sorted(chosen, key=lambda row: (row["item_key"], row["section_id"]))


def _cell(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item", action="append", required=True)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    connection = sqlite3.connect(ROOT / "data" / "relations.db")
    connection.row_factory = sqlite3.Row
    population: list[dict[str, Any]] = []
    for item_key in dict.fromkeys(args.item):
        sections = {
            section["section_id"]: section for section in split_sections(get_item_chunks(item_key))
        }
        for row in connection.execute(
            "SELECT * FROM section_summaries WHERE item_key = ? ORDER BY section_id", (item_key,),
        ):
            section = sections.get(row["section_id"])
            if section is None:
                continue
            population.append({
                **dict(row), "source": _section_source_text(section),
                "cases": [dict(case) for case in connection.execute(
                    "SELECT * FROM case_annotations WHERE item_key = ? AND section_id = ? ORDER BY case_id",
                    (item_key, row["section_id"]),
                )],
            })
    chosen = choose_rows(population, count=args.count, seed=args.seed)

    source_dir = args.output.parent / "gate3_sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ゲート3 独立監査資料", "",
        f"- 乱数seed: `{args.seed}`", f"- 母集団: {len(population)}節 / 抽出: {len(chosen)}節",
        f"- 対象item: `{', '.join(dict.fromkeys(args.item))}`", "- モデル: `codex_cli:gpt-5.6-luna`",
        "- 判定: 要約散文は `OK / mis-support / fabrication`、保存フィールドは同じ基準で記入。",
        "- 各sourceファイルは、その節のLuna入力と同じ先頭最大30,000文字。", "",
        "## 要約散文", "", "| item | section | chapter | source | 要約 | 人間判定 |",
        "|---|---|---|---|---|---|",
    ]
    for row in chosen:
        name = f"{row['item_key']}-{row['section_id']}.md"
        source_path = source_dir / name
        source_path.write_text(
            f"# {row['item_key']} / {row['section_id']} 原文\n\n{row['source']}\n", encoding="utf-8",
        )
        relative = source_path.relative_to(args.output.parent)
        lines.append(
            f"| {_cell(row['item_key'])} | {_cell(row['section_id'])} | {_cell(row['chapter'])} | "
            f"[{name}]({relative.as_posix()}) | {_cell(row['summary'])} | |"
        )
    lines.extend(["", "## 保存済み構造化フィールド", ""])
    for row in chosen:
        lines.extend([
            f"### {_cell(row['item_key'])} / {_cell(row['section_id'])}", "",
            "| 種別 | 値 | evidence_quote | chunk_id | containment | 人間判定 |",
            "|---|---|---|---|---|---|",
        ])
        for case in row["cases"]:
            quote = str(case.get("evidence_quote") or "")
            contained = bool(quote and quote in row["source"])
            lines.append(
                f"| case | {_cell(case.get('description'))} | {_cell(quote)} | "
                f"{_cell(case.get('chunk_id'))} | {'OK' if contained else 'NG'} | |"
            )
        if not row["cases"]:
            lines.append("| case | （保存なし） | | | - | |")
        if row.get("chapter_authors"):
            lines.append(
                f"| chapter_authors | {_cell(row['chapter_authors'])} | （DBにはquote非保存） | | - | |"
            )
        if row.get("first_publication_note"):
            lines.append(
                f"| first_publication | {_cell(row['first_publication_note'])} | （DBにはquote非保存） | | - | |"
            )
        lines.append("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    connection.close()
    print(f"population={len(population)} sampled={len(chosen)} seed={args.seed}")


if __name__ == "__main__":
    main()
