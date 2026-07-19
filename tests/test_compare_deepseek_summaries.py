from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import compare_deepseek_summaries


class DeepSeekSummaryComparisonTests(unittest.TestCase):
    def test_only_extractive_item_keys_are_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "relations.db"
            connection = sqlite3.connect(path)
            connection.execute("CREATE TABLE item_summaries (item_key TEXT, model TEXT)")
            connection.executemany(
                "INSERT INTO item_summaries VALUES (?, ?)",
                [("B", "extractive"), ("A", "extractive"), ("L", "codex_cli:gpt-5.6-luna")],
            )
            connection.commit()
            connection.close()
            self.assertEqual(compare_deepseek_summaries._extractive_keys(path), ["A", "B"])

    def test_non_content_section_never_calls_deepseek(self):
        section = {
            "section_id": "c0", "chapter": "Contents",
            "chunks": [{"id": "a", "text": "entry " * 100, "metadata": {}}],
        }
        with patch.object(
            compare_deepseek_summaries.build_summaries, "classify_section_content",
            return_value="non_content",
        ), patch.object(compare_deepseek_summaries, "_generate_mode") as generate:
            result = compare_deepseek_summaries._compare_section(section, "deepseek-v4-pro")
        generate.assert_not_called()
        self.assertEqual(result["status"], "non_content")

    def test_quality_failure_is_retried_up_to_attempt_limit(self):
        rejected = ({
            "summary": "bad", "sentences": [],
            "_verification": {"accepted": False},
            "_usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }, "deepseek:test")
        accepted = ({
            "summary": "good", "sentences": [],
            "_verification": {"accepted": True},
            "_usage": {"prompt_tokens": 10, "completion_tokens": 3},
        }, "deepseek:test")
        with patch.object(
            compare_deepseek_summaries.build_summaries,
            "_llm_summary_only_section", side_effect=[rejected, accepted],
        ) as generate:
            result = compare_deepseek_summaries._generate_mode(
                {"chunks": []}, "deepseek-v4-flash", "disabled", attempts=3,
            )
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["attempt_count"], 2)
        self.assertEqual(result["usage"]["prompt_tokens"], 20)
        self.assertEqual(generate.call_count, 2)


if __name__ == "__main__":
    unittest.main()
