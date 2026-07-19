from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts import build_deepseek_cases


class FakeClient:
    provider = "deepseek"
    model = "test"

    def __init__(self, response):
        self.response = response

    def generate_json(self, *_args, **_kwargs):
        return self.response


class DeepSeekCaseBatchTests(unittest.TestCase):
    def test_obvious_lowercase_boundary_continuation_keeps_both_evidence_pieces(self):
        units = [
            {"unit_id": "u1", "chunk_id": "c1", "text": "The first resistant"},
            {"unit_id": "u2", "chunk_id": "c2", "text": "weeds appeared in 1996. More text."},
        ]
        text, evidence = build_deepseek_cases._extend_boundary_evidence(
            "u1", units, "The first resistant",
        )
        self.assertEqual(text, "The first resistant weeds appeared in 1996.")
        self.assertEqual([row["chunk_id"] for row in evidence], ["c1", "c2"])

    def test_union_candidates_are_tiered_instead_of_hard_rejected(self):
        units = [
            {"unit_id": "u1", "chunk_id": "c1", "text": "Concrete grounded event."},
            {"unit_id": "u2", "chunk_id": "c2", "text": "Borderline grounded example."},
        ]
        cheap = FakeClient({"cases": [
            {"evidence_unit_id": "u1"}, {"evidence_unit_id": "u2"},
        ]})
        standard = FakeClient({"decisions": [
            {"evidence_unit_id": "u1", "is_empirical_case": True, "priority": 1},
            {"evidence_unit_id": "u2", "is_empirical_case": False, "priority": 0},
        ]})

        def hydrate(selected, _units, _section):
            unit_id = selected["cases"][0]["evidence_unit_id"]
            text = next(row["text"] for row in units if row["unit_id"] == unit_id)
            return {"cases": [{
                "description": text, "evidence_quote": text, "region": None,
                "group": None, "period": None, "practices": [], "phenomena": [],
            }]}, {"accepted": True}

        with patch.object(
            build_deepseek_cases.build_summaries, "_section_evidence_units", return_value=units,
        ), patch.object(
            build_deepseek_cases.build_summaries, "_hydrate_selector_result", side_effect=hydrate,
        ), patch.object(
            build_deepseek_cases, "get_llm", side_effect=lambda role: cheap if role == "cheap" else standard,
        ):
            cases, stats = build_deepseek_cases._extract_section(
                {"section_id": "w0", "chunks": []}, samples=2,
            )
        self.assertEqual(stats["saved"], 2)
        self.assertEqual(cases[0]["quality_status"], "partial")
        self.assertEqual(cases[1]["quality_status"], "candidate")
        self.assertEqual(cases[1]["chunk_id"], "c2")


if __name__ == "__main__":
    unittest.main()
