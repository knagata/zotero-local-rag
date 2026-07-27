from __future__ import annotations

import unittest

from src.summary_prompts import cited_item_summary_prompt, cited_section_summary_prompt


class SummaryPromptTests(unittest.TestCase):
    def test_section_prompt_keeps_source_and_grounding_contract(self):
        prompt = cited_section_summary_prompt("[e0001]\nsource")
        self.assertIn("evidence_unit_ids", prompt)
        self.assertTrue(prompt.endswith("[e0001]\nsource"))

    def test_item_prompt_keeps_title_and_verified_source(self):
        prompt = cited_item_summary_prompt("Title", "[s0001]\nsummary")
        self.assertIn("Title", prompt)
        self.assertIn("検証済み節要約", prompt)
        self.assertTrue(prompt.endswith("[s0001]\nsummary"))


if __name__ == "__main__":
    unittest.main()
