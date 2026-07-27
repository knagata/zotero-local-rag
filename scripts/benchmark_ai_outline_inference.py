#!/usr/bin/env python3
"""Measure how well an LLM can infer a PDF outline from sampled page text.

The PDF's embedded outline is used only as hidden evaluation truth.  The model
receives extracted text from the first N pages, with page boundaries retained.
"""
from __future__ import annotations

import argparse
import csv
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import sys
from typing import Any

import fitz

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm_client import get_llm


OUTLINE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "headings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "level": {"type": "integer"},
                    "page_hint": {"type": ["integer", "null"]},
                },
                "required": ["title", "level", "page_hint"],
                "additionalProperties": False,
            },
        },
        "evidence": {"type": "string"},
    },
    "required": ["headings", "evidence"],
    "additionalProperties": False,
}


def normalize_title(value: str) -> str:
    value = value.casefold().replace("\u3000", " ")
    value = re.sub(r"[^\w\u3040-\u30ff\u3400-\u9fff]+", " ", value)
    return " ".join(value.split())


def match_outlines(
    truth: list[dict[str, Any]], prediction: list[dict[str, Any]], *, threshold: float = 0.78,
) -> dict[str, Any]:
    """Greedily match predicted headings to truth in title-similarity order."""
    candidates: list[tuple[float, int, int]] = []
    for ti, expected in enumerate(truth):
        left = normalize_title(str(expected["title"]))
        for pi, actual in enumerate(prediction):
            right = normalize_title(str(actual.get("title") or ""))
            ratio = SequenceMatcher(None, left, right).ratio() if left and right else 0.0
            if ratio >= threshold:
                candidates.append((ratio, ti, pi))
    used_truth: set[int] = set()
    used_prediction: set[int] = set()
    matches: list[dict[str, Any]] = []
    for ratio, ti, pi in sorted(candidates, reverse=True):
        if ti in used_truth or pi in used_prediction:
            continue
        used_truth.add(ti)
        used_prediction.add(pi)
        matches.append({
            "truth_index": ti, "prediction_index": pi, "title_similarity": round(ratio, 4),
            "level_match": int(truth[ti]["level"]) == int(prediction[pi].get("level") or 0),
        })
    precision = len(matches) / max(1, len(prediction))
    recall = len(matches) / max(1, len(truth))
    return {
        "truth_count": len(truth), "prediction_count": len(prediction),
        "matched": len(matches), "title_precision": round(precision, 4),
        "title_recall": round(recall, 4),
        "title_f1": round(2 * precision * recall / max(1e-12, precision + recall), 4),
        "level_accuracy_on_matched": round(
            sum(int(row["level_match"]) for row in matches) / max(1, len(matches)), 4
        ),
        "matches": sorted(matches, key=lambda row: row["truth_index"]),
    }


def extract_sample(path: Path, pages: int, max_chars_per_page: int) -> tuple[str, list[dict[str, Any]]]:
    with fitz.open(path) as document:
        truth = [
            {"level": int(level), "title": str(title).strip(), "page": int(page)}
            for level, title, page in document.get_toc(simple=True)
            if str(title).strip()
        ]
        sections = []
        for index in range(min(pages, document.page_count)):
            text = (document[index].get_text("text") or "").strip()
            sections.append(f"\n--- PDF PAGE {index + 1} ---\n{text[:max_chars_per_page]}")
    return "".join(sections), truth


def load_gold_csv(path: Path) -> list[dict[str, Any]]:
    """Load reviewed outline rows, ignoring empty spreadsheet tail rows."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "id": row.get("id", "").strip(),
            "title": row.get("title", "").strip(),
            "level": int(row.get("level") or 0),
            "page": int(row.get("pdf_page") or 0),
            "parent_id": row.get("parent_id", "").strip(),
            "kind": row.get("kind", "").strip(),
        }
        for row in rows
        if row.get("title", "").strip() and int(row.get("level") or 0) > 0
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--pages", type=int, default=20)
    parser.add_argument("--max-chars-per-page", type=int, default=6000)
    parser.add_argument("--llm-role", default="extract")
    parser.add_argument("--gold-csv", type=Path, help="Reviewed canonical outline CSV.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    sample, embedded_truth = extract_sample(args.pdf, args.pages, args.max_chars_per_page)
    truth = load_gold_csv(args.gold_csv) if args.gold_csv else embedded_truth
    if not truth:
        raise SystemExit("No outline evaluation truth is available.")
    prompt = f"""You are reconstructing the complete semantic outline of a document.
Use only the sampled page text below. A table of contents may describe headings
beyond the sampled pages; include those headings when visible. Include explicit
title/front-matter and functional regions such as abstract, acknowledgements,
contents, references, notes, appendices, and index. Exclude only running headers,
page numbers, and elements that are not semantic document regions.
Preserve source order and wording. Set level 1 for parts or top-level sections,
level 2 for chapters/sections under them, and so on. A document-title entry is
metadata at level 1, not a parent/root: including it must never shift major
content headings from level 1 to level 2. Front- and back-matter functional
regions are also level 1 unless the sampled contents explicitly nests them.
page_hint is the PDF page
suggested by the sample, or null when it cannot be inferred. Do not invent entries.

SAMPLED TEXT ({args.pages} first PDF pages):
{sample}
"""
    result = get_llm(args.llm_role).generate_json(prompt, schema=OUTLINE_SCHEMA, timeout=180.0)
    excluded = set() if args.gold_csv else {"cover", "contents", "table of contents", "title page", "copyright"}
    filtered_truth = [row for row in truth if normalize_title(row["title"]) not in excluded]
    score = match_outlines(filtered_truth, result["headings"])
    report = {
        "schema_version": "ai-outline-inference-benchmark-1",
        "pdf": str(args.pdf), "sample_pages": args.pages,
        "gold_csv": str(args.gold_csv) if args.gold_csv else None,
        "truth": filtered_truth, "prediction": result["headings"],
        "model_evidence": result["evidence"], "score": score,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
