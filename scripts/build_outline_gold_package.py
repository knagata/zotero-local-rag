#!/usr/bin/env python3
"""Build reviewable CSV skeletons for the outline gold-set pilot."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

import fitz

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_ai_outline_inference import normalize_title


SELECTION = (
    ("ML9LQDHL", "english_book_standard", "latour_first20.json"),
    ("VYQRJK74", "english_article", "szulecki_all14.json"),
    ("E4PJ686A", "japanese_long_form", None),
    ("83Z5HCN9", "no_outline", None),
    ("IMBFR28G", "japanese_scanned_two_column_article", None),
)

FIELDS = (
    "id", "title", "level", "parent_id", "printed_page", "pdf_page",
    "end_pdf_page", "kind", "include_in_summary", "bookmark_level",
    "bookmark_pdf_page", "ai_level", "ai_page_hint", "candidate_sources",
    "review_status", "notes",
)


def candidate_rows(
    bookmarks: list[list[Any]], ai_headings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ai_by_title = {normalize_title(row.get("title", "")): row for row in ai_headings}
    used_ai: set[str] = set()
    for index, (level, title, page) in enumerate(bookmarks, start=1):
        normalized = normalize_title(str(title))
        ai = ai_by_title.get(normalized)
        if ai:
            used_ai.add(normalized)
        rows.append({
            "id": f"n{index:04d}", "title": str(title).strip(), "level": "",
            "parent_id": "", "printed_page": "", "pdf_page": "",
            "end_pdf_page": "", "kind": "", "include_in_summary": "",
            "bookmark_level": level, "bookmark_pdf_page": page,
            "ai_level": ai.get("level", "") if ai else "",
            "ai_page_hint": ai.get("page_hint", "") if ai else "",
            "candidate_sources": "bookmark+ai" if ai else "bookmark",
            "review_status": "pending", "notes": "",
        })
    for ai in ai_headings:
        normalized = normalize_title(ai.get("title", ""))
        if normalized in used_ai:
            continue
        rows.append({
            "id": f"n{len(rows) + 1:04d}", "title": ai.get("title", ""),
            "level": "", "parent_id": "", "printed_page": "", "pdf_page": "",
            "end_pdf_page": "", "kind": "", "include_in_summary": "",
            "bookmark_level": "", "bookmark_pdf_page": "",
            "ai_level": ai.get("level", ""), "ai_page_hint": ai.get("page_hint", ""),
            "candidate_sources": "ai", "review_status": "pending", "notes": "",
        })
    return rows


def main() -> None:
    output = ROOT / "evaluations" / "outline_gold_v1"
    output.mkdir(parents=True, exist_ok=True)
    manifest_data = json.loads((ROOT / "data" / "manifest_v3.json").read_text(encoding="utf-8"))
    files = manifest_data["files"]
    package: list[dict[str, Any]] = []
    for key, role, ai_filename in SELECTION:
        entry = files[key]
        path = Path(entry["pdf_path"])
        with fitz.open(path) as document:
            bookmarks = document.get_toc(simple=True)
            page_count = document.page_count
        ai_headings: list[dict[str, Any]] = []
        if ai_filename:
            ai_path = ROOT / "tmp" / "ai_outline_benchmark" / ai_filename
            if ai_path.exists():
                ai_headings = json.loads(ai_path.read_text(encoding="utf-8")).get("prediction", [])
        rows = candidate_rows(bookmarks, ai_headings)
        csv_name = f"{key}.csv"
        with (output / csv_name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        package.append({
            "attachment_key": key, "role": role, "title": entry.get("title"),
            "pdf_path": str(path), "page_count": page_count,
            "bookmark_count": len(bookmarks), "ai_candidate_count": len(ai_headings),
            "review_csv": csv_name,
        })
    payload = {"schema_version": "outline-gold-package-1", "documents": package}
    (output / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
