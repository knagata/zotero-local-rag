from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import build_deepseek_summaries


class DeepSeekSummaryBatchTests(unittest.TestCase):
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
                patch.object(build_deepseek_summaries.build_summaries, "_excluded_from_llm", return_value=(False, None)),
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


if __name__ == "__main__":
    unittest.main()
