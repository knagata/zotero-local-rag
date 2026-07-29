#!/usr/bin/env python3
"""Rank owned PDFs as candidates for the eight OCR bake-off categories."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.manifest import load_manifest


JA_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
REFERENCE_RE = re.compile(r"references?|bibliograph|参考文献|引用文献|巻末注|注\s*$", re.I | re.M)
MATH_RE = re.compile(r"[∑∫√≈≠≤≥±∞α-ωΑ-Ω]|\b(?:table|equation|theorem|表\s*\d)\b", re.I)
CATEGORIES = (
    "en_two_column", "tables_math", "ja_horizontal", "ja_vertical",
    "notes_bibliography_book", "embedded_text_pair", "scanned_pair", "no_outline",
)


def score_profile(profile: Mapping[str, Any], category: str) -> float:
    pages = float(profile.get("pages") or 0)
    chars_per_page = float(profile.get("chars_per_sampled_page") or 0)
    japanese = float(profile.get("japanese_ratio") or 0)
    scanned = bool(profile.get("scanned"))
    outline = int(profile.get("outline_entries") or 0)
    columns = float(profile.get("two_column_ratio") or 0)
    references = bool(profile.get("reference_signal"))
    math = int(profile.get("math_signals") or 0)
    if category == "en_two_column":
        return 5 * columns + min(chars_per_page / 1000, 2) + 2 * (japanese < 0.05)
    if category == "tables_math":
        return min(math, 10) + min(float(profile.get("blocks_per_page") or 0) / 10, 3)
    if category == "ja_horizontal":
        return 6 * japanese + min(chars_per_page / 1000, 2) + 2 * (not scanned)
    if category == "ja_vertical":
        return 6 * japanese + 4 * scanned + float(profile.get("vertical_line_ratio") or 0) * 5
    if category == "notes_bibliography_book":
        return 4 * references + min(pages / 100, 4) + min(outline / 20, 2)
    if category == "embedded_text_pair":
        return min(chars_per_page / 800, 5) + 2 * (not scanned)
    if category == "scanned_pair":
        return 6 * scanned + max(0, 2 - chars_per_page / 500)
    if category == "no_outline":
        return 4 * (outline == 0) + min(pages / 50, 3) + min(chars_per_page / 1000, 1)
    raise ValueError(f"unknown category: {category}")


def profile_pdf(path: Path, quality: Mapping[str, Any]) -> dict[str, Any]:
    import fitz

    document = fitz.open(str(path))
    try:
        pages = document.page_count
        indices = sorted(set(
            [0, max(0, pages // 4), max(0, pages // 2), max(0, 3 * pages // 4), max(0, pages - 1)]
            + list(range(max(0, pages - 10), pages))
        ))
        total_chars = japanese_chars = blocks = column_pages = vertical_lines = lines = 0
        tail_text = []
        math_signals = 0
        for index in indices:
            page = document[index]
            text = page.get_text("text") or ""
            total_chars += len(text)
            japanese_chars += len(JA_RE.findall(text))
            math_signals += len(MATH_RE.findall(text))
            if index >= max(0, pages - 10):
                tail_text.append(text)
            page_blocks = [row for row in page.get_text("blocks") if str(row[4]).strip()]
            blocks += len(page_blocks)
            midpoint = page.rect.width / 2
            left = [row for row in page_blocks if (row[0] + row[2]) / 2 < midpoint * 0.9]
            right = [row for row in page_blocks if (row[0] + row[2]) / 2 > midpoint * 1.1]
            if len(left) >= 2 and len(right) >= 2:
                column_pages += 1
            raw = page.get_text("dict")
            for block in raw.get("blocks", []):
                for line in block.get("lines", []):
                    lines += 1
                    direction = line.get("dir") or (1, 0)
                    if abs(float(direction[1])) > abs(float(direction[0])):
                        vertical_lines += 1
        sampled = max(1, len(indices))
        scanned = bool(
            quality.get("is_scanned") or quality.get("scanned_pages")
            or float(quality.get("scanned_ratio") or 0) > 0
        )
        return {
            "pages": pages, "sampled_pages": sampled,
            "chars_per_sampled_page": round(total_chars / sampled, 2),
            "japanese_ratio": round(japanese_chars / max(1, total_chars), 6),
            "blocks_per_page": round(blocks / sampled, 2),
            "two_column_ratio": round(column_pages / sampled, 6),
            "vertical_line_ratio": round(vertical_lines / max(1, lines), 6),
            "outline_entries": len(document.get_toc(simple=True)),
            "reference_signal": bool(REFERENCE_RE.search("\n".join(tail_text))),
            "math_signals": math_signals, "scanned": scanned,
        }
    finally:
        document.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest_v3.json")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = load_manifest(args.manifest).get("files") or {}
    profiles = []
    for attachment_key, entry in sorted(files.items()):
        if not isinstance(entry, dict):
            continue
        path = Path(str(entry.get("pdf_path") or ""))
        if path.suffix.casefold() != ".pdf" or not path.is_file():
            continue
        try:
            profile = profile_pdf(path, entry.get("quality") or {})
        except Exception as exc:
            profile = {"error": str(exc)[:300]}
        profiles.append({
            "attachment_key": attachment_key, "title": entry.get("title") or path.stem,
            "path": str(path), **profile,
        })
        if args.max_files and len(profiles) >= args.max_files:
            break
    suggestions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for category in CATEGORIES:
        ranked = sorted(
            (row for row in profiles if not row.get("error")),
            key=lambda row: (-score_profile(row, category), str(row["attachment_key"])),
        )[: max(1, args.top)]
        suggestions[category] = [
            {**row, "score": round(score_profile(row, category), 4)} for row in ranked
        ]
    report = {"schema_version": "ocr-bakeoff-suggestions-1", "profiled": len(profiles), "categories": suggestions}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"profiled": len(profiles), "categories": len(suggestions), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
