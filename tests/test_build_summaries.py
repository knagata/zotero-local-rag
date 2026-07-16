from __future__ import annotations

import unittest
import os
from unittest.mock import patch

from src import build_summaries
from src.build_summaries import SECTION_WINDOW, split_sections
from src.embedder import resolve_collection_name
from src.llm_client import RateLimitReached


class SummaryPipelineTests(unittest.TestCase):
    def test_codex_schemas_are_strict_objects(self):
        def assert_strict(schema):
            if schema.get("type") == "object":
                self.assertFalse(schema.get("additionalProperties", True))
                self.assertEqual(set(schema.get("required", [])), set(schema.get("properties", {})))
                for child in schema.get("properties", {}).values():
                    assert_strict(child)
            if schema.get("type") == "array":
                assert_strict(schema["items"])

        assert_strict(build_summaries.SECTION_SCHEMA)
        assert_strict(build_summaries.ITEM_SCHEMA)

    def test_chapter_groups_and_fallback_windows(self):
        chunks = [
            {"id": "a", "text": "a", "metadata": {"chapter": "One"}},
            {"id": "b", "text": "b", "metadata": {"chapter": "One"}},
        ] + [
            {"id": f"u{i}", "text": "x", "metadata": {}}
            for i in range(SECTION_WINDOW + 1)
        ]
        sections = split_sections(chunks)
        self.assertEqual([(row["section_id"], len(row["chunks"])) for row in sections], [
            ("c0", 2), ("w0", SECTION_WINDOW), ("w1", 1),
        ])

    def test_collection_suffix_preserves_explicit_base(self):
        self.assertEqual(
            resolve_collection_name(lambda _: [], env_value="paragraphs", suffix="sum_item"),
            "paragraphs__sum_item",
        )

    def test_llm_mode_does_not_treat_extractively_summarized_item_as_unchanged(self):
        chunks = [{"id": "a", "text": "body", "metadata": {}}]
        existing = {"chunk_count": 1, "source_mtime": 0.0, "model": "extractive"}
        with patch.object(build_summaries, "get_item_chunks", return_value=chunks), patch.object(
            build_summaries, "load_manifest", return_value={}
        ), patch.object(build_summaries, "get_item_summary", return_value=existing), patch.object(
            build_summaries, "_source_mtime", return_value=0.0
        ), patch.object(build_summaries, "_excluded_from_llm", return_value=(True, "policy")):
            result = build_summaries.build_item("ITEM", mode="llm")
        self.assertEqual(result["status"], "excluded")

    def test_summary_exclusion_fails_closed_when_tags_cannot_be_checked(self):
        with patch.dict(os.environ, {"SUMMARY_EXCLUDE_TAGS": "private"}, clear=False), patch.object(
            build_summaries.httpx, "get", side_effect=RuntimeError("offline")
        ):
            excluded, reason = build_summaries._excluded_from_llm("ITEM")
        self.assertTrue(excluded)
        self.assertIn("could not verify", reason)

    def test_summary_exclusion_requires_an_explicit_cloud_policy(self):
        with patch.dict(os.environ, {}, clear=True):
            excluded, reason = build_summaries._excluded_from_llm("ITEM")
        self.assertTrue(excluded)
        self.assertIn("not configured", reason)

    def test_rate_limit_propagates_for_resumable_nightly_stop(self):
        chunks = [{"id": "a", "text": "body", "metadata": {}}]
        with patch.object(build_summaries, "get_item_chunks", return_value=chunks), patch.object(
            build_summaries, "load_manifest", return_value={}
        ), patch.object(build_summaries, "get_item_summary", return_value=None), patch.object(
            build_summaries, "_source_mtime", return_value=0.0
        ), patch.object(build_summaries, "_excluded_from_llm", return_value=(False, None)), patch.object(
            build_summaries, "_llm_section", side_effect=RateLimitReached("quota")
        ):
            with self.assertRaises(RateLimitReached):
                build_summaries.build_item("ITEM", mode="llm")


if __name__ == "__main__":
    unittest.main()
