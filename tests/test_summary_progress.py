from __future__ import annotations

import io
from contextlib import redirect_stdout

from scripts.build_structure_summaries import SummaryProgressReporter


def test_summary_progress_reports_api_and_item_counters() -> None:
    output = io.StringIO()
    reporter = SummaryProgressReporter(total_items=3)
    with redirect_stdout(output):
        reporter.api_event("request", {"kind": "section"})
        reporter.api_event("response", {"kind": "section", "verification": {
            "kept_sentences": 2, "generated_sentences": 3,
        }})
        reporter.api_event("request", {"kind": "reduction"})
        reporter.api_event("error", {"kind": "reduction", "error": "LLMError"})
        reporter.item_completed({"status": "success"})
    text = output.getvalue()
    assert "API送信 1 / 応答 0 / 失敗 0 / 処理中 1" in text
    assert "API応答 1/1 / 失敗 0 / 処理中 0 / 根拠確認 2/3文" in text
    assert "API失敗 1 / 送信 2 / 応答 1 / 処理中 0" in text
    assert "itemバッチ 1/3 完了 (success=1)" in text
