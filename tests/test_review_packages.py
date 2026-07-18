from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.apply_gold_qa_review import apply_review
from scripts.build_gold_qa_review_package import choose_cases
from scripts.build_reference_review_package import build_package, classify_candidate
from scripts.review_references import validate_decision_coverage


class ReferenceReviewPackageTests(unittest.TestCase):
    def test_classifies_literal_identifier_and_compound_reference(self):
        row = {
            "review_id": 1, "item_key": "ITEM", "raw_reference":
            "Author (2020). First. doi:10.1234/example; Other (2021). Second; Third.",
            "authors": [],
        }
        result = classify_candidate(row)
        self.assertIn("literal_doi", result["flags"])
        self.assertIn("compound_reference", result["flags"])
        self.assertEqual(result["recommended_action"], "reject_compound")

    def test_isbn_requires_an_isbn_label_and_does_not_treat_pages_as_identifier(self):
        false_row = {
            "review_id": 1, "item_key": "ITEM", "authors": [],
            "raw_reference": "前掲書 pp.951-953102 1931年の記録",
        }
        true_row = {
            "review_id": 2, "item_key": "ITEM", "authors": [],
            "raw_reference": "Title. ISBN 978-92-76-29788-8.",
        }
        self.assertNotIn("literal_isbn", classify_candidate(false_row)["flags"])
        self.assertIn("literal_isbn", classify_candidate(true_row)["flags"])

    def test_detects_zwsp_obfuscated_doi_without_false_compound_year(self):
        row = {
            "review_id": 3, "item_key": "ITEM", "authors": [],
            "raw_reference": (
                "Author (2011). Title. https://\u200bdoi.\u200borg/\u200b"
                "10.\u200b1016/\u200bj.\u200bgiq.\u200b2010.\u200b06.\u200b010"
            ),
        }
        result = classify_candidate(row)
        self.assertEqual(result["literal_identifiers"]["dois"], ["10.1016/j.giq.2010.06.010"])
        self.assertNotIn("compound_reference", result["flags"])

    def test_package_fails_closed_for_blocked_item(self):
        rows = [
            {"review_id": 1, "item_key": "ALLOW", "raw_reference": "Author (2020). Title.", "authors": []},
            {"review_id": 2, "item_key": "BLOCK", "raw_reference": "Secret (2020). Title.", "authors": []},
        ]
        entries, excluded = build_package(
            rows, policy=lambda key: (key == "BLOCK", "no-cloud" if key == "BLOCK" else None),
        )
        self.assertEqual([row["item_key"] for row in entries], ["ALLOW"])
        self.assertEqual(excluded, [{"item_key": "BLOCK", "reason": "no-cloud"}])

    def test_reference_response_must_cover_the_input_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            batch = Path(directory) / "batch.json"
            batch.write_text(json.dumps({"candidates": [
                {"review_id": 1}, {"review_id": 2},
            ]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "coverage mismatch"):
                validate_decision_coverage([{"review_id": 1}], batch)


class GoldQAPackageTests(unittest.TestCase):
    def test_case_selection_is_reproducible_and_limits_each_item(self):
        rows = [
            {"item_key": item, "case_id": index}
            for item in ("A", "B", "C") for index in range(4)
        ]
        first = choose_cases([dict(row) for row in rows], count=6, seed=7)
        second = choose_cases([dict(row) for row in rows], count=6, seed=7)
        self.assertEqual(first, second)
        self.assertTrue(all(sum(row["item_key"] == key for row in first) <= 2 for key in "ABC"))

    def test_applies_validated_claude_gold_response(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "package.json"
            response = root / "response.json"
            output = root / "gold.jsonl"
            candidates = []
            decisions = []
            for index in range(2):
                candidate_id = f"case-{index}"
                candidates.append({
                    "candidate_id": candidate_id, "item_key": f"ITEM{index}",
                    "chunk_id": f"CHUNK{index}", "language": "ja",
                })
                decisions.append({
                    "candidate_id": candidate_id, "decision": "include",
                    "query": f"十分に長い検索質問 {index}",
                    "expected_item_keys": [f"ITEM{index}"],
                    "evidence_chunk_ids": [f"CHUNK{index}"], "note": "verified",
                })
            package.write_text(json.dumps({"candidates": candidates}), encoding="utf-8")
            response.write_text(json.dumps({"gold_qa": decisions}), encoding="utf-8")
            result = apply_review(package, response, output, minimum=2)
            self.assertEqual(result["included"], 2)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 2)

    def test_rejects_changed_expected_item(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "p.json"
            response = root / "r.json"
            package.write_text(json.dumps({"candidates": [{
                "candidate_id": "case-1", "item_key": "RIGHT", "chunk_id": "CHUNK",
            }]}), encoding="utf-8")
            response.write_text(json.dumps({"gold_qa": [{
                "candidate_id": "case-1", "decision": "include", "query": "long enough query",
                "expected_item_keys": ["WRONG"], "evidence_chunk_ids": ["CHUNK"],
            }]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected_item_keys changed"):
                apply_review(package, response, root / "out.jsonl", minimum=1)

    def test_gold_response_must_cover_every_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "p.json"
            response = root / "r.json"
            package.write_text(json.dumps({"candidates": [
                {"candidate_id": "case-1", "item_key": "A", "chunk_id": "C1"},
                {"candidate_id": "case-2", "item_key": "B", "chunk_id": "C2"},
            ]}), encoding="utf-8")
            response.write_text(json.dumps({"gold_qa": [{
                "candidate_id": "case-1", "decision": "exclude",
            }]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cover every candidate"):
                apply_review(package, response, root / "out.jsonl", minimum=0)


if __name__ == "__main__":
    unittest.main()
