#!/usr/bin/env python3
"""Review citation/reference relations reported from Citation Graph or MCP."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from db_relations import get_relation_reports, review_relation_report  # noqa: E402


def _clip(value: object, limit: int = 300) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def format_report(report: dict) -> str:
    direction = "参照先（所蔵 → 外部）" if report["direction"] == "references" else "被引用元（外部 → 所蔵）"
    lines = [
        f"報告 #{report['report_id']}  {direction}",
        f"  relation_key : {report['relation_key']}",
        f"  Zotero key   : {report['item_key']}",
        f"  外部資料      : {_clip(report.get('relation_title')) or '(タイトルなし)'}",
        f"  理由          : {report['reason']}",
        f"  報告内容      : {_clip(report.get('details')) or '(詳細なし)'}",
        f"  登録行 / 文脈 : {report.get('record_count', 0)} / {report.get('context_count', 0)}",
    ]
    if report.get("sample_raw_reference"):
        lines.append(f"  参考文献原文  : {_clip(report['sample_raw_reference'])}")
    if report.get("sample_context"):
        lines.append(f"  文脈例        : {_clip(report['sample_context'])}")
    return "\n".join(lines)


def review_pending(input_fn: Callable[[str], str] = input) -> dict[str, int]:
    reports = get_relation_reports("pending")
    if not reports:
        print("未確認の引用関係レポートはありません。")
        return {"pending": 0, "disabled": 0, "kept": 0, "skipped": 0}

    totals = {"pending": len(reports), "disabled": 0, "kept": 0, "skipped": 0}
    print(f"未確認の引用関係レポート: {len(reports)}件")
    print("誤りを確認できた場合だけDisableしてください。Enterは保留です。")
    for report in reports:
        print("\n" + "-" * 72)
        print(format_report(report))
        while True:
            answer = input_fn("[d] Disable  [k] Keep  [Enter] Skip  [q] Quit: ").strip().lower()
            if answer in {"", "s", "skip"}:
                totals["skipped"] += 1
                break
            if answer in {"q", "quit"}:
                totals["skipped"] += len(reports) - sum(
                    totals[key] for key in ("disabled", "kept", "skipped")
                )
                return totals
            if answer in {"d", "disable"}:
                note = input_fn("確認メモ（任意）: ").strip()
                review_relation_report(report["report_id"], "disable", note)
                totals["disabled"] += 1
                print("→ Disableしました。以後の検索・グラフから除外されます。")
                break
            if answer in {"k", "keep"}:
                note = input_fn("確認メモ（任意）: ").strip()
                review_relation_report(report["report_id"], "keep", note)
                totals["kept"] += 1
                print("→ Keepしました。関係は引き続き利用されます。")
                break
            print("d、k、Enter、q のいずれかを入力してください。")
    return totals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="未確認レポートをJSON表示して終了")
    parser.add_argument("--disable", type=int, metavar="REPORT_ID", help="指定レポートをDisable")
    parser.add_argument("--keep", type=int, metavar="REPORT_ID", help="指定レポートをKeep")
    parser.add_argument("--note", default="", help="--disable/--keepの確認メモ")
    args = parser.parse_args(argv)

    if args.list:
        print(json.dumps(get_relation_reports("pending"), ensure_ascii=False, indent=2))
        return 0
    if args.disable is not None and args.keep is not None:
        parser.error("--disable and --keep cannot be used together")
    if args.disable is not None:
        if not review_relation_report(args.disable, "disable", args.note):
            print(f"Report not found: {args.disable}", file=sys.stderr)
            return 1
        return 0
    if args.keep is not None:
        if not review_relation_report(args.keep, "keep", args.note):
            print(f"Report not found: {args.keep}", file=sys.stderr)
            return 1
        return 0

    totals = review_pending()
    print(
        "\n確認結果: "
        f"Disable {totals['disabled']} / Keep {totals['kept']} / "
        f"保留 {totals['skipped']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
