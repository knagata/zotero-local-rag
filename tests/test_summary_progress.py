from __future__ import annotations

import io
from contextlib import redirect_stdout

from scripts import build_structure_summaries as summary_cli
from scripts.build_structure_summaries import SummaryProgressReporter, _embedding_item_keys


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


def test_limited_current_batch_does_not_request_any_embedding() -> None:
    scope = _embedding_item_keys(
        all_items=True, limit=10,
        results=[{"item_key": f"I{index}", "status": "skipped_current"} for index in range(10)],
    )

    assert scope == set()


def test_limited_batch_embeds_only_items_whose_summaries_changed() -> None:
    scope = _embedding_item_keys(
        all_items=True, limit=10, results=[
            {"item_key": "CURRENT", "status": "skipped_current"},
            {"item_key": "CHANGED", "status": "success"},
            {"item_key": "EMPTY", "status": "empty"},
            {"item_key": "FAILED", "status": "failed"},
        ],
    )

    assert scope == {"CHANGED", "EMPTY"}


def test_unlimited_all_retains_explicit_full_index_reconciliation() -> None:
    scope = _embedding_item_keys(
        all_items=True, limit=0,
        results=[{"item_key": "CURRENT", "status": "skipped_current"}],
    )

    assert scope is None


def test_summary_progress_reports_embedding_count() -> None:
    output = io.StringIO()
    reporter = SummaryProgressReporter(total_items=1)
    with redirect_stdout(output):
        reporter.embedding_progress(16, 40)

    assert "要約索引 16/40 件を埋め込み済み" in output.getvalue()


def test_limited_all_selects_pending_items_before_applying_limit(monkeypatch) -> None:
    current = {"CURRENT-1", "CURRENT-2"}
    monkeypatch.setattr(
        summary_cli, "structure_summaries_are_current",
        lambda key, *, mode: key in current,
    )

    selected = summary_cli._select_batch_keys(
        ["CURRENT-1", "CURRENT-2", "STALE-1", "STALE-2", "STALE-3"],
        all_items=True, limit=2, force=False, retry_failed=False, mode="llm",
    )

    assert selected == ["STALE-1", "STALE-2"]


def test_explicit_full_run_does_not_prefilter_current_items(monkeypatch) -> None:
    monkeypatch.setattr(
        summary_cli, "structure_summaries_are_current",
        lambda key, *, mode: True,
    )

    selected = summary_cli._select_batch_keys(
        ["CURRENT-1", "CURRENT-2"], all_items=True, limit=0,
        force=False, retry_failed=False, mode="llm",
    )

    assert selected == ["CURRENT-1", "CURRENT-2"]
