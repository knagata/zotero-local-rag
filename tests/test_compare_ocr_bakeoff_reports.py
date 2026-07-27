from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts import compare_ocr_bakeoff_reports as compare


def _run(sample, engine, score, *, duration=10, rss=100, status="completed", **extra):
    metrics = {name: score for name in compare.METRIC_ORDER}
    return {
        "sample_id": sample, "engine": engine, "status": status,
        "duration_seconds": duration, "process_peak_rss_mb": rss,
        "score": {"total_score": score, "metrics": metrics, "tree_errors": ["private"]},
        "blocks": [{"text": "must not leak"}], "pdf_path": "/Users/private/source.pdf",
        **extra,
    }


class CompareOcrBakeoffReportsTests(unittest.TestCase):
    def setUp(self):
        self.categories = {"s1": "horizontal", "s2": "vertical"}

    def test_whitelists_fields_and_calculates_winners_and_engine_means(self):
        result = compare.aggregate_reports([{"runs": [
            _run("s1", "beta", 0.8, duration=8, rss=120),
            _run("s1", "alpha", 0.8, duration=7, rss=130),
            _run("s2", "alpha", 0.6, duration=13, rss=150),
            _run("s2", "beta", 0.4, duration=12, rss=140),
        ]}], self.categories)
        self.assertEqual(result["category_winners"], [
            {"category": "horizontal", "winner": "alpha", "mean_score": 0.8, "completed_runs": 1},
            {"category": "vertical", "winner": "alpha", "mean_score": 0.6, "completed_runs": 1},
        ])
        alpha = next(row for row in result["engine_averages"] if row["engine"] == "alpha")
        self.assertEqual(alpha["mean_score"], 0.7)
        serialized = json.dumps(result)
        self.assertNotIn("must not leak", serialized)
        self.assertNotIn("/Users/private", serialized)
        self.assertNotIn("tree_errors", serialized)

    def test_result_is_independent_of_report_order_and_duplicate_order(self):
        older = {"runs": [_run("s1", "alpha", 0.5, duration=20)]}
        better = {"runs": [_run("s1", "alpha", 0.7, duration=30)]}
        first = compare.aggregate_reports([older, better], self.categories)
        second = compare.aggregate_reports([better, older], self.categories)
        self.assertEqual(first, second)
        self.assertEqual(first["runs"][0]["score"], 0.7)

    def test_exact_tie_falls_back_to_engine_name(self):
        result = compare.aggregate_reports([{"runs": [
            _run("s1", "zeta", 0.8, duration=5, rss=90),
            _run("s1", "alpha", 0.8, duration=5, rss=90),
        ]}], self.categories)
        winner = next(row for row in result["category_winners"] if row["category"] == "horizontal")
        self.assertEqual(winner["winner"], "alpha")

    def test_cli_writes_path_free_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            report = root / "report.json"
            output = root / "out"
            manifest.write_text(json.dumps({"samples": [
                {"id": "s1", "category": "horizontal", "path_env": "PRIVATE_PDF"},
            ]}), encoding="utf-8")
            report.write_text(json.dumps({"runs": [
                _run("s1", "alpha", 0.9),
            ]}), encoding="utf-8")
            self.assertEqual(compare.main([
                str(report), "--manifest", str(manifest), "--output", str(output),
            ]), 0)
            payload = (output / "comparison.json").read_text(encoding="utf-8")
            markdown = (output / "comparison.md").read_text(encoding="utf-8")
        self.assertNotIn("PRIVATE_PDF", payload)
        self.assertNotIn("source.pdf", payload)
        self.assertIn("同点規則", markdown)
        self.assertIn("| s1 | horizontal | alpha |", markdown)


if __name__ == "__main__":
    unittest.main()
