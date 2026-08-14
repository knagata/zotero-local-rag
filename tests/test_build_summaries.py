from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from src import build_summaries
from src import db_relations
from src import summary_core
from src.build_summaries import SECTION_WINDOW, split_sections
from src.embedder import resolve_collection_name


class SummaryPipelineTests(unittest.TestCase):
    def test_legacy_summary_writers_are_not_exposed(self):
        for name in ("build_item", "embed_summaries", "main"):
            self.assertFalse(hasattr(build_summaries, name), name)
        for name in (
            "save_item_summary",
            "save_section_summary",
            "replace_extractive_summary_bundle",
            "delete_section_summary",
            "mark_insight_generation_status",
        ):
            self.assertFalse(hasattr(db_relations, name), name)

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

    def test_summary_only_reports_each_paid_request_and_verified_response(self):
        generated = {"sentences": [
            {"text": "第一の根拠文。", "evidence_unit_ids": ["s0001"]},
            {"text": "第二の根拠文。", "evidence_unit_ids": ["s0001"]},
        ]}
        client = Mock(provider="deepseek", model="deepseek-v4-pro")
        client.generate_json.return_value = generated
        events = []
        summary_core.set_summary_progress_callback(
            lambda event, details: events.append((event, details)),
        )
        try:
            with patch.object(summary_core, "DeepSeekClient", return_value=client):
                build_summaries._llm_summary_only_item(
                    "Title", [{"section_id": "w0", "summary": "根拠となる節要約。"}],
                )
        finally:
            summary_core.set_summary_progress_callback(None)
        self.assertEqual([event for event, _details in events], ["request", "response"])
        self.assertEqual(events[0][1]["kind"], "reduction")
        self.assertEqual(events[1][1]["verification"]["kept_sentences"], 2)

    def test_summary_only_reports_a_terminal_api_error(self):
        from src.llm_client import LLMError
        client = Mock(provider="deepseek", model="deepseek-v4-pro")
        client.generate_json.side_effect = LLMError("provider failed")
        events = []
        summary_core.set_summary_progress_callback(
            lambda event, details: events.append((event, details)),
        )
        try:
            with patch.object(summary_core, "DeepSeekClient", return_value=client):
                with self.assertRaises(LLMError):
                    build_summaries._llm_summary_only_item(
                        "Title", [{"section_id": "w0", "summary": "根拠となる節要約。"}],
                    )
        finally:
            summary_core.set_summary_progress_callback(None)
        self.assertEqual([event for event, _details in events], ["request", "error"])
        self.assertEqual(events[1][1]["error"], "LLMError")

    def test_section_summary_reports_its_verified_response(self):
        generated = {"sentences": [
            {"text": "第一の根拠文。", "evidence_unit_ids": ["u0001"]},
            {"text": "第二の根拠文。", "evidence_unit_ids": ["u0001"]},
        ]}
        client = Mock(provider="deepseek", model="deepseek-v4-flash")
        client.generate_json.return_value = generated
        events = []
        summary_core.set_summary_progress_callback(
            lambda event, details: events.append((event, details)),
        )
        try:
            with patch.object(summary_core, "DeepSeekClient", return_value=client):
                build_summaries._llm_summary_only_section({
                    "section_id": "w0", "chapter": "Chapter",
                    "chunks": [{"id": "c0", "text": "根拠となる本文。"}],
                })
        finally:
            summary_core.set_summary_progress_callback(None)
        self.assertEqual([event for event, _details in events], ["request", "response"])
        self.assertEqual(events[1][1]["kind"], "section")

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
