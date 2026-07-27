#!/usr/bin/env python3
"""Review only structured-case reports that automated triage could not decide."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from db_relations import get_case_quality_reports, resolve_case_quality_report  # noqa: E402


def review_pending(input_fn=input) -> dict[str, int]:
    reports = [row for row in get_case_quality_reports("pending") if row.get("triage_status") == "uncertain"]
    totals = {"pending": len(reports), "disabled": 0, "kept": 0, "skipped": 0}
    if not reports:
        print("人間確認が必要な事例品質レポートはありません。")
        return totals
    for index, report in enumerate(reports):
        print("\n" + "-" * 72)
        print(f"報告 #{report['report_id']}  case:{report['case_id']}  item:{report['item_key']}")
        print(f"  reason  : {report.get('reason')}")
        print(f"  details : {report.get('details')}")
        print(f"  AI判定  : {report.get('triage_evidence')}")
        answer = input_fn("[d] Disable  [k] Keep  [Enter] Skip  [q] Quit: ").strip().lower()
        if answer in {"q", "quit"}:
            totals["skipped"] += len(reports) - index
            break
        if answer in {"d", "disable"}:
            resolve_case_quality_report(report["report_id"], "disable", reviewer_note="Human exception review.")
            totals["disabled"] += 1
        elif answer in {"k", "keep"}:
            resolve_case_quality_report(report["report_id"], "keep", reviewer_note="Human exception review.")
            totals["kept"] += 1
        else:
            totals["skipped"] += 1
    return totals


if __name__ == "__main__":
    print(json.dumps(review_pending(), ensure_ascii=False))
