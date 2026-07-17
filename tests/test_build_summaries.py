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
            for variant in schema.get("anyOf", []):
                assert_strict(variant)

        assert_strict(build_summaries.SECTION_SCHEMA)
        assert_strict(build_summaries.ITEM_SCHEMA)

    def test_front_matter_and_toc_are_non_content(self):
        fixtures = [
            {"chapter": "目次", "text": "章題 " * 100},
            {"chapter": "Title Page", "text": "publication data " * 50},
            {"chapter": "Acknowledgments", "text": "Many people helped with this book. " * 30},
            {"chapter": "About Virginia Heffernan", "text": "Biographical information. " * 30},
            {"chapter": "", "text": "序章 概要 5\n第一章 調査 20\n第二章 分析 40\n第三章 結論 70\n" * 10},
            {"chapter": "", "text": (
                "Thank you for downloading this ebook.\n"
                "CONTENTS PREFACE 1. DESIGN 2. TEXT 3. IMAGES 4. VIDEO\n"
                "INDEX A note about the index and print page numbers.\n"
                + "Knausgaard, 79 Kindle, 241 Korea, 196 " * 30
            )},
            {"chapter": "", "text": "short copyright notice"},
        ]
        for index, fixture in enumerate(fixtures):
            section = {
                "section_id": str(index), "chapter": fixture["chapter"],
                "chunks": [{"id": "x", "text": fixture["text"], "metadata": {}}],
            }
            self.assertEqual(build_summaries.classify_section_content(section), "non_content")

    def test_normal_prose_is_content(self):
        text = "Fieldwork participants described reciprocal exchange in the village. " * 20
        section = {
            "section_id": "c1", "chapter": "Ethnography",
            "chunks": [{"id": "chunk-1", "text": text, "metadata": {}}],
        }
        self.assertEqual(build_summaries.classify_section_content(section), "content")

    def test_structured_fields_require_exact_evidence_and_direct_chunk(self):
        source = "Alice Smith conducted fieldwork in Fiji in 2020. Published in 2021."
        result = {
            "summary": "要約",
            "cases": [{
                "description": "フィジーでの調査", "region": "Fiji", "group": None,
                "practices": ["fieldwork"], "phenomena": [], "period": "2020",
                "locator_hint": None, "source_kind": "primary",
                "evidence_quote": "Alice Smith conducted fieldwork in Fiji in 2020.",
            }, {
                "description": "捏造", "region": None, "group": None, "practices": [],
                "phenomena": [], "period": None, "locator_hint": None,
                "source_kind": "primary", "evidence_quote": "Not in the input.",
            }],
            "chapter_authors": [
                {"name": "Alice Smith", "evidence_quote": "Alice Smith conducted fieldwork"},
                {"name": "Bob Jones", "evidence_quote": "Alice Smith conducted fieldwork"},
            ],
            "first_publication_note": {
                "note": "Published in 2021.", "evidence_quote": "Published in 2021.",
            },
        }
        verified, stats = build_summaries._verify_section_result(
            result, source, [{"id": "chunk-1", "text": source}],
        )
        self.assertEqual(len(verified["cases"]), 1)
        self.assertEqual(verified["cases"][0]["chunk_id"], "chunk-1")
        self.assertEqual([row["name"] for row in verified["chapter_authors"]], ["Alice Smith"])
        self.assertEqual(verified["first_publication_note"]["note"], "Published in 2021.")
        self.assertEqual(stats["total_generated"], 5)
        self.assertEqual(stats["total_discarded"], 2)

    def test_publication_value_outside_evidence_is_discarded(self):
        result = {
            "summary": "要約", "cases": [], "chapter_authors": [],
            "first_publication_note": {
                "note": "Published online in 2025 with DOI 10.1234/fake.",
                "evidence_quote": "The paper was published online.",
            },
        }
        verified, stats = build_summaries._verify_section_result(
            result, "The paper was published online.",
        )
        self.assertIsNone(verified["first_publication_note"])
        self.assertEqual(
            stats["first_publication_note"]["reasons"]["value_not_in_evidence"], 1,
        )

    def test_case_date_outside_evidence_is_discarded(self):
        result = {
            "summary": "要約", "chapter_authors": [], "first_publication_note": None,
            "cases": [{
                "description": "1992年に父から批判された。", "region": None, "group": None,
                "practices": [], "phenomena": [], "period": "1992",
                "locator_hint": None, "source_kind": "primary",
                "evidence_quote": "Her father criticized the prose.",
            }],
        }
        verified, stats = build_summaries._verify_section_result(
            result, "Her father criticized the prose.",
        )
        self.assertEqual(verified["cases"], [])
        self.assertEqual(stats["cases"]["reasons"]["value_not_in_evidence"], 1)

    def test_case_evidence_may_span_two_adjacent_chunks(self):
        result = {
            "summary": "要約", "chapter_authors": [], "first_publication_note": None,
            "cases": [{
                "description": "村で交換が行われた。", "region": None, "group": None,
                "practices": ["交換"], "phenomena": [], "period": None,
                "locator_hint": None, "source_kind": "primary",
                "evidence_quote": "村で 交換が行われた。",
            }],
        }
        verified, stats = build_summaries._verify_section_result(
            result,
            "村で\n\n交換が行われた。",
            [{"id": "chunk-1", "text": "村で"}, {"id": "chunk-2", "text": "交換が行われた。"}],
        )
        self.assertEqual(len(verified["cases"]), 1)
        self.assertEqual(verified["cases"][0]["chunk_id"], "chunk-1")
        self.assertEqual(stats["total_discarded"], 0)

    def test_case_evidence_spanning_three_chunks_is_discarded(self):
        result = {
            "summary": "要約", "chapter_authors": [], "first_publication_note": None,
            "cases": [{
                "description": "村で交換が行われた。", "region": None, "group": None,
                "practices": ["交換"], "phenomena": [], "period": None,
                "locator_hint": None, "source_kind": "primary",
                "evidence_quote": "村で 盛んに 交換が行われた。",
            }],
        }
        verified, stats = build_summaries._verify_section_result(
            result,
            "村で\n\n盛んに\n\n交換が行われた。",
            [
                {"id": "chunk-1", "text": "村で"},
                {"id": "chunk-2", "text": "盛んに"},
                {"id": "chunk-3", "text": "交換が行われた。"},
            ],
        )
        self.assertEqual(verified["cases"], [])
        self.assertEqual(stats["cases"]["reasons"]["evidence_not_in_chunk"], 1)

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
        chunks = [{"id": "a", "text": "substantive body " * 40, "metadata": {}}]
        with patch.object(build_summaries, "get_item_chunks", return_value=chunks), patch.object(
            build_summaries, "load_manifest", return_value={}
        ), patch.object(build_summaries, "get_item_summary", return_value=None), patch.object(
            build_summaries, "_source_mtime", return_value=0.0
        ), patch.object(build_summaries, "_excluded_from_llm", return_value=(False, None)), patch.object(
            build_summaries, "_llm_section", side_effect=RateLimitReached("quota")
        ):
            with self.assertRaises(RateLimitReached):
                build_summaries.build_item("ITEM", mode="llm")

    def test_llm_build_skips_non_content_and_removes_old_section(self):
        chunks = [{
            "id": "a", "text": "chapter listing " * 50,
            "metadata": {"chapter": "目次"},
        }]
        with patch.object(build_summaries, "get_item_chunks", return_value=chunks), patch.object(
            build_summaries, "load_manifest", return_value={}
        ), patch.object(build_summaries, "get_item_summary", return_value=None), patch.object(
            build_summaries, "_source_mtime", return_value=0.0
        ), patch.object(build_summaries, "_excluded_from_llm", return_value=(False, None)), patch.object(
            build_summaries, "delete_section_summary"
        ) as delete, patch.object(build_summaries, "save_item_summary"), patch.object(
            build_summaries, "_llm_section"
        ) as llm:
            audit_sections = []
            result = build_summaries.build_item(
                "ITEM", mode="llm", audit_sections=audit_sections,
            )
        delete.assert_called_once_with("ITEM", "c0")
        llm.assert_not_called()
        self.assertEqual(result["skipped_non_content"], 1)
        self.assertEqual(result["sections"], 0)
        self.assertEqual(audit_sections[0]["status"], "skipped_non_content")


if __name__ == "__main__":
    unittest.main()
