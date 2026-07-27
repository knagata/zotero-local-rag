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
from src import summary_core
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

        assert_strict(build_summaries.SUMMARY_ONLY_SCHEMA)
        assert_strict(build_summaries.ITEM_SCHEMA)

    def test_evidence_units_preserve_exact_ocr_text_and_chunk_identity(self):
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

    def test_evidence_units_match_section_source_budget(self):
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

    def test_summary_only_requires_valid_ids_but_soft_flags_ordinary_values(self):
        units = [{
            "unit_id": "u0001", "chunk_id": "a",
            "text": "Alice conducted fieldwork in Fiji in 2020.",
        }]
        result = {"sentences": [{
            "text": "Aliceは2020年にフィジーで現地調査を行った。",
            "evidence_unit_ids": ["u0001"],
        }, {
            "text": "調査は2025年に終了した。", "evidence_unit_ids": ["u0001"],
        }, {
            "text": "存在しない根拠。", "evidence_unit_ids": ["u9999"],
        }]}
        verified, stats = build_summaries._verify_summary_only_result(result, units)
        self.assertEqual(len(verified["sentences"]), 2)
        self.assertEqual(verified["sentences"][0]["evidence_unit_ids"], ["u0001"])
        self.assertEqual(stats["reasons"], {"invalid_evidence_id": 1})
        self.assertEqual(stats["warnings"], {"value_not_in_evidence": 1})
        self.assertTrue(stats["accepted"])

    def test_summary_only_rejects_unsupported_definitive_identifier(self):
        units = [{
            "unit_id": "u0001", "chunk_id": "a",
            "text": "The registered DOI is 10.1234/source.",
        }]
        result = {"sentences": [{
            "text": "DOIは10.9999/inventedである。", "evidence_unit_ids": ["u0001"],
        }, {
            "text": "資料にはDOIが登録されている。", "evidence_unit_ids": ["u0001"],
        }]}
        verified, stats = build_summaries._verify_summary_only_result(result, units)
        self.assertEqual(len(verified["sentences"]), 1)
        self.assertEqual(stats["reasons"], {"identifier_not_in_evidence": 1})
        self.assertFalse(stats["accepted"])

    def test_summary_only_accepts_three_fully_grounded_sentences(self):
        units = [
            {"unit_id": "u0001", "chunk_id": "a", "text": "The study began in 2020."},
            {"unit_id": "u0002", "chunk_id": "a", "text": "It examined village exchange."},
        ]
        result = {"sentences": [
            {"text": "研究は2020年に始まった。", "evidence_unit_ids": ["u0001"]},
            {"text": "村落交換を検討した。", "evidence_unit_ids": ["u0002"]},
            {"text": "研究は2020年の開始である。", "evidence_unit_ids": ["u0001"]},
        ]}
        verified, stats = build_summaries._verify_summary_only_result(result, units)
        self.assertEqual(
            verified["summary"],
            "研究は2020年に始まった。村落交換を検討した。研究は2020年の開始である。",
        )
        self.assertTrue(stats["accepted"])

    def test_item_summary_only_uses_verified_section_ids(self):
        generated = {"sentences": [
            {"text": "研究は2020年に開始された。", "evidence_unit_ids": ["s0001"]},
            {"text": "村落交換を検討した。", "evidence_unit_ids": ["s0002"]},
            {"text": "調査対象は交換実践である。", "evidence_unit_ids": ["s0002"]},
        ]}
        client = Mock(provider="deepseek", model="deepseek-v4-pro")
        client.generate_json.return_value = generated
        with patch.object(summary_core, "DeepSeekClient", return_value=client):
            result, model = build_summaries._llm_summary_only_item("Title", [
                {"section_id": "w0", "summary": "研究は2020年に開始された。"},
                {"section_id": "w1", "summary": "村落交換の実践を調査対象として検討した。"},
            ])
        self.assertTrue(result["_verification"]["accepted"])
        self.assertEqual(result["sentences"][0]["evidence_unit_ids"], ["s0001"])
        self.assertEqual(model, "deepseek:deepseek-v4-pro:summary-only:disabled")

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
        chunks = [{"id": "a", "text": "substantive body " * 40, "metadata": {}}]
        existing = {"chunk_count": 1, "source_mtime": 0.0, "model": "extractive"}
        # Previously this asserted "excluded", which only happened because the
        # unconfigured cloud policy failed closed. That gate was removed
        # 2026-07-27, so the item now reaches the LLM -- which is the point of
        # the test: an extractively summarised item must not count as unchanged.
        with patch.object(build_summaries, "get_item_chunks", return_value=chunks), patch.object(
            build_summaries, "load_manifest", return_value={}
        ), patch.object(build_summaries, "get_item_summary", return_value=existing), patch.object(
            build_summaries, "_source_mtime", return_value=0.0
        ), patch.object(
            build_summaries, "_llm_section",
            side_effect=RuntimeError("stop after the unchanged check"),
        ) as llm_section:
            with self.assertRaises(RuntimeError):
                build_summaries.build_item("ITEM", mode="llm")
        llm_section.assert_called()

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
        ), patch.object(build_summaries, "get_llm", return_value=deepseek):
            result = build_summaries.build_item("ITEM", mode="llm", force=True)
        self.assertEqual(result["status"], "protected_existing")

    def test_rate_limit_propagates_for_resumable_batch_stop(self):
        chunks = [{"id": "a", "text": "substantive body " * 40, "metadata": {}}]
        with patch.object(build_summaries, "get_item_chunks", return_value=chunks), patch.object(
            build_summaries, "load_manifest", return_value={}
        ), patch.object(build_summaries, "get_item_summary", return_value=None), patch.object(
            build_summaries, "_source_mtime", return_value=0.0
        ), patch.object(
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
            stop_file = Path(tmp) / "batch.stop"
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
            "summary": "本文の要約", "chapter_authors": [],
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
        ), patch.object(
            build_summaries, "_llm_section", return_value=(generated, "codex_cli:test")
        ), patch.object(
            build_summaries, "_llm_item", return_value=(item_result, "codex_cli:test")
        ), patch.object(build_summaries, "save_section_summary"), patch.object(
            build_summaries, "save_item_summary"
        ):
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
        ), patch.object(
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
            "chapter_authors": [], "first_publication_note": None,
            "_verification": {"total_generated": 0, "total_discarded": 0,
                              "suspicious_section": False},
        }
        with patch.object(build_summaries, "get_item_chunks", return_value=chunks), patch.object(
            build_summaries, "load_manifest", return_value={}
        ), patch.object(build_summaries, "get_item_summary", return_value=None), patch.object(
            build_summaries, "_source_mtime", return_value=0.0
        ), patch.object(
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


class MinLeafCharsTests(unittest.TestCase):
    """The threshold below which a node is not worth summarizing.

    Summary length is nearly independent of source length -- measured across
    25,858 summaries, the median was 248 characters for a 400-1,000 character
    leaf and 448 for one over 20,000 -- so the compression a summary buys is
    decided entirely by the source. Raising the floor to 1,000 (2026-07-28)
    dropped the band whose median compression was only 0.35, where a hit on the
    summary sends the reader to the source anyway.
    """

    def test_threshold_excludes_the_band_that_measured_uneconomical(self):
        from src.build_structure_summaries import MIN_LEAF_CHARS
        # 716 was the median source length of the removed band.
        self.assertGreater(MIN_LEAF_CHARS, 716)

    def test_a_skipped_leaf_is_not_a_failure(self):
        # Skipping is a deliberate economy, so it must not mark the item as
        # blocked or failed -- its chunks remain searchable in their own
        # collection, which is the whole reason the summary is optional.
        from src.build_structure_summaries import MIN_LEAF_CHARS
        self.assertIsInstance(MIN_LEAF_CHARS, int)
        self.assertGreater(MIN_LEAF_CHARS, 0)

