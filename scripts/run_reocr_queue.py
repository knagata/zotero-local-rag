#!/usr/bin/env python3
"""Run a bounded re-OCR queue, regenerate summaries, and write a quality report."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chunk_store import get_item_chunks
from src.manifest import load_manifest


REPEAT_ARTIFACT_RE = re.compile(r"(.)\1{19,}")


def _selected_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("candidates")
    if not isinstance(rows, list):
        raise ValueError("candidate JSON must contain a candidates array")
    return [row for row in rows[:limit] if isinstance(row, dict)]


def _relation_counts(item_key: str) -> dict[str, int]:
    connection = sqlite3.connect(ROOT / "data" / "relations.db")
    try:
        return {
            "sections": connection.execute(
                "SELECT COUNT(*) FROM section_summaries WHERE item_key = ?", (item_key,),
            ).fetchone()[0],
            "cases": connection.execute(
                "SELECT COUNT(*) FROM case_annotations WHERE item_key = ?", (item_key,),
            ).fetchone()[0],
        }
    finally:
        connection.close()


def snapshot(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest = load_manifest(ROOT / "data" / "manifest.json")
    result = []
    for row in rows:
        item_key = str(row.get("item_key") or "")
        attachment_key = str(row.get("attachment_key") or "")
        chunks = [
            chunk for chunk in get_item_chunks(item_key)
            if chunk.get("metadata", {}).get("attachmentKey") == attachment_key
        ]
        text = "\n".join(str(chunk.get("text") or "") for chunk in chunks)
        quality = (manifest.get("files", {}).get(attachment_key, {}).get("quality") or {})
        result.append({
            "item_key": item_key,
            "attachment_key": attachment_key,
            "language": row.get("lang") or "unknown",
            "chunks": len(chunks),
            "characters": len(text),
            "pages": len({chunk.get("metadata", {}).get("page") for chunk in chunks}),
            "repeat_artifacts": len(REPEAT_ARTIFACT_RE.findall(text)),
            "quality": {
                "parser": quality.get("parser") or "unknown",
                "is_scanned": bool(quality.get("is_scanned")),
                "is_corrupted": bool(quality.get("is_corrupted")),
            },
            "relations": _relation_counts(item_key),
        })
    return result


def _back_up() -> list[str]:
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    backup_dir = ROOT / "data" / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in ("relations.db", "lexical.sqlite3", "manifest.json"):
        source = ROOT / "data" / name
        if not source.exists():
            continue
        target = backup_dir / f"{source.stem}-before-reocr-{stamp}{source.suffix}"
        shutil.copy2(source, target)
        paths.append(str(target.relative_to(ROOT)))
    return paths


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--llm", default="codex_cli:gpt-5.6-luna")
    parser.add_argument("--max-discard-rate", type=float, default=0.25)
    parser.add_argument("--min-character-ratio", type=float, default=0.75)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    rows = _selected_rows(args.candidates, args.limit)
    if not rows:
        parser.error("candidate queue is empty")

    before = snapshot(rows)
    backups = _back_up()
    _run([
        sys.executable, "src/index_from_zotero.py", "--progress",
        "--reocr-candidates", str(args.candidates), "--reocr-limit", str(args.limit),
    ])

    audits = []
    audit_dir = ROOT / "data" / "quality" / "reocr-audits"
    for item_key in dict.fromkeys(str(row.get("item_key") or "") for row in rows):
        if not item_key:
            continue
        audit_path = audit_dir / f"{item_key}.json"
        _run([
            sys.executable, "-m", "src.build_summaries", "--mode", "llm", "--force",
            "--item", item_key, "--llm", args.llm, "--audit-output", str(audit_path),
        ])
        audits.extend(json.loads(audit_path.read_text(encoding="utf-8")).get("items", []))

    after = snapshot(rows)
    before_by_attachment = {row["attachment_key"]: row for row in before}
    failures = []
    for current in after:
        previous = before_by_attachment[current["attachment_key"]]
        baseline = max(1, int(previous["characters"]))
        current["character_ratio"] = round(current["characters"] / baseline, 4)
        if current["character_ratio"] < args.min_character_ratio:
            failures.append(f"{current['attachment_key']}: character ratio below threshold")
        if current["repeat_artifacts"]:
            failures.append(f"{current['attachment_key']}: pathological repeat artifact")
    for audit in audits:
        verification = audit.get("verification") or {}
        if float(verification.get("discard_rate") or 0) > args.max_discard_rate:
            failures.append(f"{audit.get('item_key')}: grounding discard rate above threshold")
        if verification.get("suspicious_sections"):
            failures.append(f"{audit.get('item_key')}: suspicious sections remain")

    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "candidates": str(args.candidates),
        "limit": args.limit,
        "llm": args.llm,
        "thresholds": {
            "max_discard_rate": args.max_discard_rate,
            "min_character_ratio": args.min_character_ratio,
        },
        "backups": backups,
        "before": before,
        "after": after,
        "summary_audits": audits,
        "quality_gate": {"passed": not failures, "failures": failures},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"quality_gate": report["quality_gate"]}, ensure_ascii=False))
    if failures:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
