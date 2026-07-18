from __future__ import annotations

import io
import unittest
import os
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

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
        assert_strict(build_summaries.SELECTOR_SECTION_SCHEMA)
        assert_strict(build_summaries.SELECTOR_CASE_JUDGE_SCHEMA)
        assert_strict(build_summaries.ITEM_SCHEMA)

    def test_selector_units_preserve_exact_ocr_text_and_chunk_identity(self):
        section = {
            "section_id": "c0", "chapter": "",
            "chunks": [
                {"id": "chunk-a", "text": "組谷では交換した。\n次の文。", "metadata": {}},
                {"id": "chunk-b", "text": "Second chunk remains exact.", "metadata": {}},
            ],
        }
        units = build_summaries._section_evidence_units(section)
        self.assertEqual([row["chunk_id"] for row in units], ["chunk-a", "chunk-a", "chunk-b"])
        self.assertEqual(units[0]["text"], "組谷では交換した。")
        self.assertNotIn("村", units[0]["text"])

    def test_selector_units_match_section_source_budget(self):
        section = {
            "section_id": "c0", "chapter": "",
            "chunks": [
                {"id": "a", "text": "A" * 20000, "metadata": {}},
                {"id": "b", "text": "B" * 20000, "metadata": {}},
            ],
        }
        units = build_summaries._section_evidence_units(section)
        source = build_summaries._section_source_text(section)
        self.assertEqual(len(source), 30000)
        self.assertTrue(all(unit["text"] in source for unit in units))
        self.assertEqual(sum(len(unit["text"]) for unit in units), 29998)

    def test_selector_hydrates_quote_locally_and_rejects_unknown_id(self):
        section = {
            "section_id": "c0", "chapter": "",
            "chunks": [{
                "id": "chunk-a", "text": "Alice conducted fieldwork in Fiji in 2020.",
                "metadata": {},
            }],
        }
        units = build_summaries._section_evidence_units(section)
        generated = {
            "summary": "要約", "chapter_authors": [], "first_publication_note": None,
            "cases": [{
                "description": "2020年にフィジーで調査した。", "region": "Fiji",
                "group": None, "practices": ["fieldwork"], "phenomena": [],
                "period": "2020", "locator_hint": None, "source_kind": "primary",
                "evidence_unit_id": units[0]["unit_id"],
            }, {
                "description": "invalid", "region": None, "group": None,
                "practices": [], "phenomena": [], "period": None,
                "locator_hint": None, "source_kind": "primary",
                "evidence_unit_id": "u9999",
            }],
        }
        verified, stats = build_summaries._hydrate_selector_result(generated, units, section)
        self.assertEqual(len(verified["cases"]), 1)
        self.assertEqual(
            verified["cases"][0]["evidence_quote"],
            "Alice conducted fieldwork in Fiji in 2020.",
        )
        self.assertEqual(
            verified["cases"][0]["description"],
            "Alice conducted fieldwork in Fiji in 2020.",
        )
        self.assertEqual(verified["cases"][0]["region"], "Fiji")
        self.assertEqual(verified["cases"][0]["practices"], ["fieldwork"])
        self.assertEqual(verified["cases"][0]["chunk_id"], "chunk-a")
        self.assertEqual(stats["invalid_evidence_unit_ids"], 1)

    def test_selector_consensus_requires_distinct_sample_votes(self):
        section = {
            "section_id": "c0", "chapter": "",
            "chunks": [{"id": "chunk-a", "text": "People exchanged gifts.", "metadata": {}}],
        }
        units = build_summaries._section_evidence_units(section)
        case = {
            "description": "贈与交換を行った。", "region": None, "group": None,
            "practices": ["gift exchange"], "phenomena": [], "period": None,
            "locator_hint": None, "source_kind": "primary",
            "evidence_unit_id": units[0]["unit_id"],
        }
        base = {
            "summary": "要約", "chapter_authors": [], "first_publication_note": None,
        }
        verified, stats = build_summaries._selector_consensus(
            [{**base, "cases": [case, case]}, {**base, "cases": [case]}, {**base, "cases": []}],
            units, section, min_votes=2,
        )
        self.assertEqual(len(verified["cases"]), 1)
        self.assertEqual(stats["selector"]["consensus_cases"], 1)
        self.assertEqual(stats["selector"]["case_selections"], 3)

    def test_selector_case_judge_uses_majority_and_ignores_unknown_ids(self):
        class Client:
            def __init__(self):
                self.responses = iter([
                    {"decisions": [
                        {"evidence_unit_id": "u0001", "is_empirical_case": True},
                        {"evidence_unit_id": "unknown", "is_empirical_case": True},
                    ]},
                    {"decisions": [
                        {"evidence_unit_id": "u0001", "is_empirical_case": True},
                        {"evidence_unit_id": "u0002", "is_empirical_case": True},
                    ]},
                    {"decisions": [
                        {"evidence_unit_id": "u0001", "is_empirical_case": False},
                        {"evidence_unit_id": "u0002", "is_empirical_case": False},
                    ]},
                ])

            def generate_json(self, *_args, **_kwargs):
                return next(self.responses)

        accepted, stats = build_summaries._judge_selector_case_ids(
            Client(), {"u0001", "u0002"}, [
                {"unit_id": "u0001", "chunk_id": "a", "text": "A concrete event."},
                {"unit_id": "u0002", "chunk_id": "a", "text": "An abstract claim."},
            ], samples=3, min_votes=2,
        )
        self.assertEqual(accepted, {"u0001"})
        self.assertEqual(stats["votes"], {"u0001": 2, "u0002": 1})

    def test_selector_rejects_obvious_fragments_before_judging(self):
        self.assertFalse(build_summaries._is_self_contained_evidence("roken sentence ending."))
        self.assertFalse(build_summaries._is_self_contained_evidence("A" * 900))
        self.assertTrue(build_summaries._is_self_contained_evidence("A complete event happened."))
        self.assertTrue(build_summaries._is_self_contained_evidence("村で祭礼が行われた。"))

    def test_front_matter_and_toc_are_non_content(self):
        fixtures = [
            {"chapter": "目次", "text": "章題 " * 100},
            {"chapter": "Title Page", "text": "publication data " * 50},
            {"chapter": "Acknowledgments", "text": "Many people helped with this book. " * 30},
            {"chapter": "About Virginia Heffernan", "text": "Biographical information. " * 30},
            {"chapter": "事項一覧", "text": "用語と参照ページ " * 50},
            {"chapter": "Glossary", "text": "Term definitions and page references. " * 30},
            {"chapter": "List of Abbreviations", "text": "Abbreviation entries. " * 40},
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

    def test_detects_model_meta_responses(self):
        fixtures = [
            "入力には要約対象となる本文が含まれていません。本文をご提示ください。",
            "この内容を要約することはできません。対象テキストをご提示ください。",
            "I cannot provide a summary because the input does not contain the passage.",
            (
                "入力原文は索引の断片であり、経験的事例の叙述を含んでいません。"
                "具体的な事例を単一の引用によって裏付けることはできません。"
                "根拠のある抽出結果は空としました。"
            ),
        ]
        for value in fixtures:
            self.assertTrue(build_summaries.is_meta_summary(value))
        self.assertFalse(build_summaries.is_meta_summary(
            "本章は、入力資料に含まれる複数の民族誌事例を比較して論じる。"
        ))

    def test_contributors_heading_is_non_content(self):
        section = {
            "section_id": "c1", "chapter": "Contributors",
            "chunks": [{"id": "x", "text": "Biographical information. " * 30, "metadata": {}}],
        }
        self.assertEqual(build_summaries.classify_section_content(section), "non_content")

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

    def test_ellipsis_composite_quotes_are_rejected_before_containment(self):
        for ellipsis in ("…", "...", "‥"):
            quote = f"Alice observed exchange{ellipsis}in Fiji."
            result = {
                "summary": "要約", "chapter_authors": [], "first_publication_note": None,
                "cases": [{
                    "description": "exchange in Fiji", "region": "Fiji", "group": None,
                    "practices": ["exchange"], "phenomena": [], "period": None,
                    "locator_hint": None, "source_kind": "primary",
                    "evidence_quote": quote,
                }],
            }
            verified, stats = build_summaries._verify_section_result(result, quote)
            self.assertEqual(verified["cases"], [])
            self.assertEqual(stats["cases"]["reasons"]["composite_quote"], 1)

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

    def test_force_does_not_replace_luna_with_deepseek_without_explicit_override(self):
        chunks = [{"id": "a", "text": "body", "metadata": {}}]
        existing = {
            "chunk_count": 1, "source_mtime": 0.0,
            "model": "codex_cli:gpt-5.6-luna",
        }
        deepseek = Mock(provider="deepseek", model="deepseek-v4-pro")
        with patch.object(build_summaries, "get_item_chunks", return_value=chunks), patch.object(
            build_summaries, "load_manifest", return_value={}
        ), patch.object(build_summaries, "get_item_summary", return_value=existing), patch.object(
            build_summaries, "_source_mtime", return_value=0.0
        ), patch.object(build_summaries, "get_llm", return_value=deepseek), patch.object(
            build_summaries, "_excluded_from_llm"
        ) as exclusion:
            result = build_summaries.build_item("ITEM", mode="llm", force=True)
        self.assertEqual(result["status"], "protected_existing")
        exclusion.assert_not_called()

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

    def test_max_items_counts_updates_not_unchanged_scans(self):
        results = [
            {"item_key": "A", "status": "unchanged"},
            {"item_key": "B", "status": "updated"},
            {"item_key": "C", "status": "unchanged"},
            {"item_key": "D", "status": "updated"},
        ]
        output = io.StringIO()
        with patch.object(build_summaries, "list_item_keys", return_value=list("ABCDE")), patch.object(
            build_summaries, "build_item", side_effect=results
        ) as build, patch.object(
            sys, "argv", ["build_summaries", "--max-items", "2", "--no-embed"]
        ), redirect_stdout(output):
            build_summaries.main()
        self.assertEqual([call.args[0] for call in build.call_args_list], list("ABCD"))
        self.assertIn('"stop_reason": "max_items"', output.getvalue())

    def test_stop_file_prevents_starting_the_next_item(self):
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            stop_file = Path(tmp) / "nightly.stop"
            stop_file.touch()
            with patch.object(build_summaries, "list_item_keys", return_value=["A", "B"]), patch.object(
                build_summaries, "build_item"
            ) as build, patch.object(
                sys, "argv", [
                    "build_summaries", "--stop-file", str(stop_file), "--no-embed",
                ],
            ), redirect_stdout(output):
                build_summaries.main()
        build.assert_not_called()
        self.assertIn('"stop_reason": "stop_requested"', output.getvalue())

    def test_quota_guard_runs_before_each_llm_request(self):
        chunks = [{"id": "a", "text": "substantive body " * 40, "metadata": {}}]
        generated = {
            "summary": "本文の要約", "cases": [], "chapter_authors": [],
            "first_publication_note": None,
            "_verification": {"total_generated": 0, "total_discarded": 0,
                              "suspicious_section": False},
        }
        item_result = {"summary": "全体要約", "summary_en": "Summary", "keywords": ["test"]}
        guard = Mock()
        with patch.object(build_summaries, "get_item_chunks", return_value=chunks), patch.object(
            build_summaries, "load_manifest", return_value={}
        ), patch.object(build_summaries, "get_item_summary", return_value=None), patch.object(
            build_summaries, "_source_mtime", return_value=0.0
        ), patch.object(build_summaries, "_excluded_from_llm", return_value=(False, None)), patch.object(
            build_summaries, "_llm_section", return_value=(generated, "codex_cli:test")
        ), patch.object(
            build_summaries, "_llm_item", return_value=(item_result, "codex_cli:test")
        ), patch.object(build_summaries, "save_section_summary"), patch.object(
            build_summaries, "replace_case_annotations"
        ), patch.object(build_summaries, "save_item_summary"):
            build_summaries.build_item("ITEM", mode="llm", quota_guard=guard)
        self.assertEqual(guard.call_count, 2)

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

    def test_llm_build_discards_meta_response_and_removes_old_section(self):
        chunks = [{
            "id": "a", "text": "substantive source text " * 50,
            "metadata": {"chapter": "Terms"},
        }]
        generated = {
            "summary": "入力には要約対象の本文が含まれていません。ご提示ください。",
            "cases": [], "chapter_authors": [], "first_publication_note": None,
            "_verification": {"total_generated": 0, "total_discarded": 0,
                              "suspicious_section": False},
        }
        with patch.object(build_summaries, "get_item_chunks", return_value=chunks), patch.object(
            build_summaries, "load_manifest", return_value={}
        ), patch.object(build_summaries, "get_item_summary", return_value=None), patch.object(
            build_summaries, "_source_mtime", return_value=0.0
        ), patch.object(build_summaries, "_excluded_from_llm", return_value=(False, None)), patch.object(
            build_summaries, "_llm_section", return_value=(generated, "codex_cli:gpt-5.6-luna")
        ), patch.object(build_summaries, "delete_section_summary") as delete, patch.object(
            build_summaries, "save_section_summary"
        ) as save_section, patch.object(build_summaries, "save_item_summary"):
            audit_sections = []
            result = build_summaries.build_item(
                "ITEM", mode="llm", audit_sections=audit_sections,
            )
        delete.assert_called_once_with("ITEM", "c0")
        save_section.assert_not_called()
        self.assertEqual(result["sections"], 0)
        self.assertEqual(result["skipped_non_content"], 1)
        self.assertEqual(audit_sections[0]["skip_reason"], "meta_response")


if __name__ == "__main__":
    unittest.main()
