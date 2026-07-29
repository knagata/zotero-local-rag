from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts import triage_quality_reports


class QualityReportTriageTests(unittest.TestCase):
    def test_summary_source_uses_only_reported_chunk_ids(self):
        report = {
            "item_key": "ITEM",
            "evidence_chunk_ids": ["chunk-2"],
            "section_id": "",
        }
        chunks = [
            {"id": "chunk-1", "text": "unselected"},
            {"id": "chunk-2", "text": "selected source"},
        ]
        with patch.object(
            triage_quality_reports, "get_item_chunks", return_value=chunks,
        ):
            source = triage_quality_reports._summary_source(report)
        self.assertEqual(source, ("selected source", "reported_chunks"))

    def test_decisive_judgment_requires_exact_source_quote(self):
        valid = triage_quality_reports._validated_judgment({
            "decision": "confirmed",
            "explanation": "The reported number directly conflicts with the source.",
            "evidence_quote": "the source says twelve participants",
        }, "In the study, the source says twelve participants in total.")
        self.assertEqual(valid["decision"], "confirmed")

        invalid = triage_quality_reports._validated_judgment({
            "decision": "confirmed",
            "explanation": "The reported number directly conflicts with the source.",
            "evidence_quote": "a fabricated quotation",
        }, "The source says twelve participants.")
        self.assertEqual(invalid["decision"], "uncertain")

    def test_confirmed_summary_report_is_reversibly_disabled(self):
        report = {
            "report_id": 7, "item_key": "ITEM", "section_id": "w0",
            "summary_hash": "expected", "reason": "wrong_number",
            "details": "The summary says 21 but the source says 12.",
        }
        current = {"summary": "There were 21 participants.", "model": "flash"}
        judgment = {
            "decision": "confirmed", "model": "deepseek:flash:quality-triage",
            "explanation": "The source gives a different count.",
            "evidence_quote": "There were 12 participants.",
        }
        with patch.object(
            triage_quality_reports, "_current_reported_summary", return_value=current,
        ), patch.object(
            triage_quality_reports, "_summary_fingerprint", return_value="expected",
        ), patch.object(
            triage_quality_reports, "_summary_source",
            return_value=("There were 12 participants.", "reported_chunks"),
        ), patch.object(
            triage_quality_reports, "_judge_with_fallback", return_value=judgment,
        ), patch.object(
            triage_quality_reports, "resolve_summary_quality_report",
        ) as resolve:
            decision = triage_quality_reports.triage_summary_report(report)
        self.assertEqual(decision, "confirmed")
        self.assertEqual(resolve.call_args.args[:2], (7, "disable"))

    def test_relation_without_local_evidence_is_escalated_without_llm(self):
        report = {
            "report_id": 8, "item_key": "ITEM", "sample_raw_reference": None,
            "sample_context": None,
        }
        with patch.object(
            triage_quality_reports, "mark_relation_report_uncertain",
        ) as uncertain, patch.object(
            triage_quality_reports, "_judge_with_fallback",
        ) as judge:
            decision = triage_quality_reports.triage_relation_report(report)
        self.assertEqual(decision, "uncertain")
        uncertain.assert_called_once()
        judge.assert_not_called()


if __name__ == "__main__":
    unittest.main()
