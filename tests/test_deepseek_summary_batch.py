from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
