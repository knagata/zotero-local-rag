from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import build_deepseek_summaries


class DeepSeekSummaryBatchTests(unittest.TestCase):
    def test_old_strict_gate_failure_ledger_is_retired(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failure-ledger.json"
            path.write_text(
                '{"version": 1, "items": {"OLD": {"attempts": 2}}}',
                encoding="utf-8",
            )
            ledger = build_deepseek_summaries._load_failure_ledger(path)
        self.assertEqual(ledger, {
            "version": build_deepseek_summaries.FAILURE_LEDGER_VERSION,
            "items": {},
        })

    def test_failure_ledger_quarantines_after_two_attempts_and_clears_on_success(self):
        ledger = {"version": 1, "items": {}}
        failure = {"item_key": "ITEM", "status": "section_failure"}
        build_deepseek_summaries._record_failure_attempt(ledger, failure)
        build_deepseek_summaries._record_failure_attempt(ledger, failure)
        self.assertEqual(ledger["items"]["ITEM"]["attempts"], 2)
        build_deepseek_summaries._record_failure_attempt(
            ledger, {"item_key": "ITEM", "status": "updated"},
        )
        self.assertNotIn("ITEM", ledger["items"])

    def test_no_content_is_quarantined_immediately(self):
        ledger = {"version": 1, "items": {}}
        build_deepseek_summaries._record_failure_attempt(
            ledger, {"item_key": "ITEM", "status": "no_content"},
        )
        self.assertEqual(ledger["items"]["ITEM"]["attempts"], 2)

    def test_provider_failure_is_not_recorded_as_item_failure(self):
        ledger = {"version": 1, "items": {}}
        build_deepseek_summaries._record_failure_attempt(
            ledger, {"item_key": "ITEM", "status": "provider_failure"},
        )
        self.assertNotIn("ITEM", ledger["items"])

    def test_section_provider_failure_is_propagated_to_item(self):
        chunks = [{"text": "source", "metadata": {"title": "Title"}}]
        sections = [{
            "section_id": "section-1", "chapter": None, "chunks": chunks,
        }]
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(build_deepseek_summaries, "get_item_chunks", return_value=chunks),
                patch.object(build_deepseek_summaries.build_summaries, "split_sections", return_value=sections),
                patch.object(build_deepseek_summaries, "_generate_section", return_value={
                    "status": "provider_failure", "error": "402 Payment Required",
                }),
            ):
                result = build_deepseek_summaries._process_item(
                    "ITEM", model="deepseek-v4-pro", workers=1,
                    checkpoint_dir=Path(directory),
                )
        self.assertEqual(result["status"], "provider_failure")

    def test_quality_failure_falls_back_but_provider_failure_does_not(self):
        accepted = {"status": "accepted", "model": "deepseek:pro", "summary": "ok"}
        with patch.object(
            build_deepseek_summaries, "_retry_summary",
            side_effect=[{"status": "quality_failure"}, accepted],
        ) as retry:
            result = build_deepseek_summaries._retry_with_quality_fallback(
                lambda model: model, "flash", "pro",
            )
        self.assertEqual(result["model"], "deepseek:pro")
        self.assertEqual(result["fallback_from"], "flash")
        self.assertEqual(retry.call_count, 2)

        with patch.object(
            build_deepseek_summaries, "_retry_summary",
            return_value={"status": "provider_failure"},
        ) as retry:
            result = build_deepseek_summaries._retry_with_quality_fallback(
                lambda model: model, "flash", "pro",
            )
        self.assertEqual(result["status"], "provider_failure")
        self.assertEqual(retry.call_count, 1)


if __name__ == "__main__":
    unittest.main()
