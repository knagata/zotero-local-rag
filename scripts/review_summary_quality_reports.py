#!/usr/bin/env python3
"""Review only summary-quality reports that automated triage could not decide."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from db_relations import get_summary_quality_reports, resolve_summary_quality_report  # noqa: E402


def _clip(value: object, limit: int = 400) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def review_pending(input_fn: Callable[[str], str] = input) -> dict[str, int]:
    reports = [
        report for report in get_summary_quality_reports("pending")
        if report.get("triage_status") == "uncertain"
    ]
    if not reports:
        print("人間確認が必要な要約品質レポートはありません。")
        return {"pending": 0, "disabled": 0, "kept": 0, "skipped": 0}
    totals = {"pending": len(reports), "disabled": 0, "kept": 0, "skipped": 0}
    for report in reports:
        print("\n" + "-" * 72)
        print(f"報告 #{report['report_id']}  {report['summary_key']}")
        print(f"  model   : {report.get('summary_model')}")
        print(f"  reason  : {report.get('reason')}")
        print(f"  details : {_clip(report.get('details'))}")
        print(f"  AI判定  : {_clip(report.get('triage_evidence'))}")
        answer = input_fn("[d] Disable  [k] Keep  [Enter] Skip  [q] Quit: ").strip().lower()
        if answer in {"q", "quit"}:
            totals["skipped"] += len(reports) - totals["disabled"] - totals["kept"]
            break
        if answer in {"d", "disable"}:
            resolve_summary_quality_report(
                report["report_id"], "disable", reviewer_note="Human exception review.",
            )
            totals["disabled"] += 1
        elif answer in {"k", "keep"}:
            resolve_summary_quality_report(
                report["report_id"], "keep", reviewer_note="Human exception review.",
            )
            totals["kept"] += 1
        else:
            totals["skipped"] += 1
    return totals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        print(json.dumps([
            row for row in get_summary_quality_reports("pending")
            if row.get("triage_status") == "uncertain"
        ], ensure_ascii=False, indent=2))
        return 0
    print(json.dumps(review_pending(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
