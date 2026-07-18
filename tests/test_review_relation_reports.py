from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from scripts import review_relation_reports


SAMPLE = {
    "report_id": 7,
    "direction": "references",
    "relation_key": "references:ITEM:S2",
    "item_key": "ITEM",
    "relation_title": "Suspicious work",
    "reason": "not_in_source",
    "details": "Absent from the bibliography.",
    "record_count": 1,
    "context_count": 0,
    "sample_raw_reference": None,
    "sample_context": None,
}


class ReviewRelationReportTests(unittest.TestCase):
    def test_enter_safely_skips(self):
        with patch.object(review_relation_reports, "get_relation_reports", return_value=[SAMPLE]), patch.object(
            review_relation_reports, "review_relation_report",
        ) as review:
            with redirect_stdout(StringIO()):
                totals = review_relation_reports.review_pending(lambda _prompt: "")
        self.assertEqual(totals["skipped"], 1)
        review.assert_not_called()

    def test_disable_requires_explicit_choice(self):
        answers = iter(["d", "confirmed against source"])
        with patch.object(review_relation_reports, "get_relation_reports", return_value=[SAMPLE]), patch.object(
            review_relation_reports, "review_relation_report", return_value=True,
        ) as review:
            with redirect_stdout(StringIO()):
                totals = review_relation_reports.review_pending(lambda _prompt: next(answers))
        self.assertEqual(totals["disabled"], 1)
        review.assert_called_once_with(7, "disable", "confirmed against source")

    def test_keep_is_recorded(self):
        answers = iter(["k", "verified as correct"])
        with patch.object(review_relation_reports, "get_relation_reports", return_value=[SAMPLE]), patch.object(
            review_relation_reports, "review_relation_report", return_value=True,
        ) as review:
            with redirect_stdout(StringIO()):
                totals = review_relation_reports.review_pending(lambda _prompt: next(answers))
        self.assertEqual(totals["kept"], 1)
        review.assert_called_once_with(7, "keep", "verified as correct")


if __name__ == "__main__":
    unittest.main()
