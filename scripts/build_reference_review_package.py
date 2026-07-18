#!/usr/bin/env python3
"""Build no-cloud-safe, batched reference-review material for Claude."""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db_relations import get_reference_review_candidates
from src.env_utils import load_dotenv_native
from src.reference_agent import _item_excluded

DEFAULT_OUTPUT = ROOT / "dev-notes" / "reference_review_batches"
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:a-z0-9]+", re.IGNORECASE)
ISBN_RE = re.compile(
    r"\bISBN(?:-1[03])?\s*:?\s*((?:97[89][\s-]?)?(?:\d[\s-]?){9}[\dXx])",
    re.IGNORECASE,
)
YEAR_RE = re.compile(r"(?<!\d)(?:18|19|20)\d{2}(?!\d)")

INSTRUCTIONS = [
    "各候補をraw_referenceだけから判定し、外部知識で書誌情報を補わない。",
    "approvedはraw_reference中に同一のDOIまたはISBNが文字どおり存在する場合だけ。",
    "有効な引用でも安定識別子がなければrejectedとし、noteをunresolved: insufficient stable identifierで始める。",
    "複数文献を含むrawは一行のまま承認せず、rejected / compound referenceとする。",
    "ノイズや書誌でない行はrejected / not a bibliographic referenceとする。",
    "title・year・DOI・ISBNはrawに存在する表記だけを返す。判断不能なら推測しない。",
]


def classify_candidate(row: dict[str, Any]) -> dict[str, Any]:
    raw = unicodedata.normalize("NFKC", str(row.get("raw_reference") or ""))
    dois = sorted(set(match.rstrip(".,;)") for match in DOI_RE.findall(raw)))
    isbns = sorted(set(re.sub(r"[^0-9Xx]", "", match) for match in ISBN_RE.findall(raw)))
    years = sorted(set(YEAR_RE.findall(raw)))
    compound = len(years) > 1 or raw.count(";") >= 2
    flags = []
    if not row.get("title"):
        flags.append("missing_title")
    if compound:
        flags.append("compound_reference")
    if dois:
        flags.append("literal_doi")
    if isbns:
        flags.append("literal_isbn")
    return {
        "review_id": row["review_id"], "item_key": row["item_key"],
        "source_kind": row.get("source_kind") or "legacy",
        "raw_reference": raw, "source_context": row.get("source_context"),
        "existing_metadata": {
            "title": row.get("title"), "authors": row.get("authors") or [],
            "year": row.get("year"), "doi": row.get("doi"), "isbn": row.get("isbn"),
            "container": row.get("container"), "work_type": row.get("work_type"),
        },
        "literal_identifiers": {"dois": dois, "isbns": isbns},
        "years_in_raw": years, "flags": flags,
        "recommended_action": (
            "reject_compound" if compound else
            "verify_identifier" if dois or isbns else "review_then_unresolved_or_reject"
        ),
    }


def build_package(
    rows: list[dict[str, Any]], *, policy: Callable[[str], tuple[bool, str | None]],
) -> tuple[list[dict[str, Any]], list[dict[str, str | None]]]:
    decisions: dict[str, tuple[bool, str | None]] = {}
    for item_key in sorted({str(row["item_key"]) for row in rows}):
        decisions[item_key] = policy(item_key)
    included = [classify_candidate(row) for row in rows if not decisions[str(row["item_key"])][0]]
    excluded = [
        {"item_key": item_key, "reason": reason}
        for item_key, (blocked, reason) in decisions.items() if blocked
    ]
    return included, excluded


def write_batches(
    entries: list[dict[str, Any]], excluded: list[dict[str, str | None]],
    *, output_dir: Path, batch_size: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for start in range(0, len(entries), batch_size):
        batch = entries[start:start + batch_size]
        number = start // batch_size + 1
        path = output_dir / f"batch-{number:03d}.json"
        path.write_text(json.dumps({
            "batch": number, "instructions": INSTRUCTIONS,
            "response_format": {
                "decisions": [{
                    "review_id": "integer", "status": "approved|rejected|pending",
                    "title": "string|null", "authors": ["string"], "year": "integer|null",
                    "doi": "string|null", "isbn": "string|null", "note": "string",
                }],
            },
            "candidates": batch,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        files.append(path.name)
    manifest = {
        "candidates": len(entries), "batch_size": batch_size, "batches": len(files),
        "files": files, "excluded_items": excluded,
        "source_counts": dict(Counter(entry["source_kind"] for entry in entries)),
        "flag_counts": dict(Counter(flag for entry in entries for flag in entry["flags"])),
        "apply_command": (
            "uv run python scripts/review_references.py apply-decisions RESPONSE.json "
            "--expected-batch INPUT_BATCH.json"
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=25)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    load_dotenv_native(ROOT)
    rows = get_reference_review_candidates("pending")
    entries, excluded = build_package(rows, policy=_item_excluded)
    manifest = write_batches(entries, excluded, output_dir=args.output_dir, batch_size=args.batch_size)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
