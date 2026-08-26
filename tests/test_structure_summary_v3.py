from __future__ import annotations

import unittest
import tempfile
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

from src import db_relations
from src.build_structure_summaries import (
    MAX_PARENT_INPUT_CHARS, _chunk_groups, _llm_leaf_summary, _llm_quality,
    _select_searchable_summary_rows,
    _chapter_summary_targets, adds_nothing_over_its_children, build_structure_summaries,
    embed_structure_summaries,
    structure_summaries_are_current,
)
from src.document_structure import build_document_structure


class StructureSummaryV3Tests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(db_relations, "DB_PATH", str(Path(self.tempdir.name) / "relations.db"))
        self.db_patch.start()
        db_relations._db_initialized = False

    def tearDown(self):
        self.db_patch.stop()
        db_relations._db_initialized = False
        self.tempdir.cleanup()

    @patch("src.build_structure_summaries.get_item_processing_status")
    @patch("src.build_structure_summaries.get_document_structure", return_value=None)
    def test_terminal_structureless_summary_does_not_clog_limited_batches(
        self, _structure, statuses,
    ):
        statuses.return_value = [{
            "artifact_type": "summary", "attachment_key": None, "status": "excluded",
        }]

        self.assertTrue(structure_summaries_are_current("ITEM", mode="llm"))

        statuses.return_value = []
        self.assertFalse(structure_summaries_are_current("ITEM", mode="llm"))

    @patch("src.build_structure_summaries.get_item_processing_status")
    @patch("src.build_structure_summaries.get_document_structure")
    def test_current_empty_summary_requires_matching_source_and_processor(
        self, structure, statuses,
    ):
        from src.build_structure_summaries import PROMPT_VERSION
        structure.return_value = {"source_fingerprint": "source"}
        statuses.return_value = [{
            "artifact_type": "summary", "attachment_key": None, "status": "empty",
            "source_fingerprint": "source", "processor_version": PROMPT_VERSION,
        }]
        self.assertTrue(structure_summaries_are_current("ITEM", mode="llm"))

        statuses.return_value[0]["source_fingerprint"] = "old"
        self.assertFalse(structure_summaries_are_current("ITEM", mode="llm"))

    @patch("src.build_structure_summaries._summaries_are_current", return_value=True)
    @patch("src.build_structure_summaries.get_item_processing_status", return_value=[])
    @patch("src.build_structure_summaries.get_document_structure", return_value={
        "source_fingerprint": "source",
    })
    def test_nonempty_summary_delegates_to_full_current_contract(
        self, _structure, _statuses, current,
    ):
        self.assertTrue(structure_summaries_are_current("ITEM", mode="llm"))
        current.assert_called_once_with("ITEM", {"source_fingerprint": "source"}, "llm")

    @patch("src.build_structure_summaries.resolve_embedder_settings", return_value={})
    @patch("src.build_structure_summaries.create_embedding_function")
    @patch("src.build_structure_summaries.open_chroma_collection")
    @patch("src.build_structure_summaries.get_all_document_node_summaries")
    def test_summary_embedding_reports_completed_rows(
        self, rows, open_collection, create_embedding, _settings,
    ):
        collection = unittest.mock.Mock()
        collection._chroma_client = None
        open_collection.return_value = collection
        create_embedding.return_value = lambda documents: [[0.0] for _ in documents]
        rows.return_value = [{
            "item_key": "ITEM", "attachment_key": "A", "node_id": "node",
            "parent_node_id": "", "node_type": "chapter", "depth": 1,
            "title": "Chapter", "summary": "Summary", "source_fingerprint": "source",
        }]
        progress = []

        result = embed_structure_summaries(
            item_keys={"ITEM"}, base_collection_name="base",
            progress_callback=lambda completed, total: progress.append((completed, total)),
        )

        self.assertEqual(result, {"nodes": 1})
        self.assertEqual(progress, [(1, 1)])

    def test_leaf_input_groups_are_ordered_and_bounded_at_chunk_boundaries(self):
        rows = [{"id": f"c{index}", "text": letter * 16_000} for index, letter in enumerate("abc")]
        groups = _chunk_groups(rows)
        self.assertEqual([[row["id"] for row in group] for group in groups], [["c0"], ["c1"], ["c2"]])
        self.assertTrue(all(sum(len(row["text"]) for row in group) <= MAX_PARENT_INPUT_CHARS for group in groups))

    def test_llm_quality_uses_sentence_verification_without_rejecting_legacy_results(self):
        self.assertEqual(_llm_quality({}), "accepted")
        self.assertEqual(_llm_quality({"accepted": True, "kept_sentences": 2}), "accepted")
        self.assertEqual(_llm_quality({"accepted": False, "kept_sentences": 1}), "candidate")

    @patch("src.build_structure_summaries._deepseek_model", return_value="flash")
    @patch("src.build_structure_summaries._llm_summary_only_section")
    def test_leaf_summary_retains_partial_sentence_verification(self, summarize, _model):
        verification = {
            "accepted": False, "generated_sentences": 3,
            "kept_sentences": 1, "discarded_sentences": 2,
        }
        summarize.return_value = ({"summary": "根拠付きの一文。", "_verification": verification}, "deepseek:test")
        result = _llm_leaf_summary("leaf", "Chapter", [{"id": "c1", "text": "x" * 500}])
        self.assertEqual(result[0], "根拠付きの一文。")
        self.assertEqual(result[3], verification)
        self.assertEqual(result[4], "candidate")

    @patch("src.build_structure_summaries._deepseek_model", return_value="flash")
    @patch("src.build_structure_summaries._llm_summary_only_section")
    def test_leaf_summary_rejects_an_empty_verified_result(self, summarize, _model):
        from src.build_structure_summaries import LLMError
        summarize.return_value = ({"summary": "", "_verification": {}}, "deepseek:test")
        with self.assertRaisesRegex(LLMError, "empty or meta"):
            _llm_leaf_summary("leaf", "Chapter", [{"id": "c1", "text": "x" * 500}])

    def test_japanese_part_distinguishes_direct_sections_from_labelled_chapters(self):
        chunks = [
            {"id": "A:p1", "text": "one", "metadata": {"attachmentKey": "A",
             "structure_path": ["第Ⅰ部 交換様式", "１ 生産から交換へ"], "zone": "body"}},
            {"id": "A:p2", "text": "two", "metadata": {"attachmentKey": "A",
             "structure_path": ["第Ⅱ部 世界帝国", "１章 共同体と国家", "１ 未開社会"], "zone": "body"}},
        ]
        built = build_document_structure("ITEM", chunks)
        children = defaultdict(list)
        for node in built["nodes"]:
            if node.get("parent_node_id"):
                children[str(node["parent_node_id"])].append(node)
        targets = _chapter_summary_targets(built["nodes"], children)
        titles = {node.get("title") for node in built["nodes"] if node["node_id"] in targets}
        self.assertIn("第Ⅰ部 交換様式", titles)
        self.assertIn("１章 共同体と国家", titles)
        self.assertNotIn("１ 生産から交換へ", titles)

    def test_role_annotated_attachment_does_not_disable_unannotated_attachment(self):
        chunks = [
            {"id": "A:p1", "text": "a", "metadata": {"attachmentKey": "A",
             "structure_path": ["Chapter A"], "structure_roles": ["chapter"], "zone": "body"}},
            {"id": "B:p1", "text": "b", "metadata": {"attachmentKey": "B",
             "structure_path": ["Chapter B"], "zone": "body"}},
        ]
        built = build_document_structure("ITEM", chunks)
        children = defaultdict(list)
        for node in built["nodes"]:
            if node.get("parent_node_id"):
                children[str(node["parent_node_id"])].append(node)
        targets = _chapter_summary_targets(built["nodes"], children)
        titles = {node.get("title") for node in built["nodes"] if node["node_id"] in targets}
        self.assertEqual(titles, {"Chapter A", "Chapter B"})

    def test_short_pdf_uses_attachment_summary_without_flattening_outline(self):
        chunks = [
            {"id": f"A:p{index}", "text": "argument " * 100,
             "metadata": {"attachmentKey": "A", "source_type": "pdf",
                          "structure_path": [f"Section {index}"],
                          "structure_roles": ["chapter"], "zone": "body"}}
            for index in range(1, 5)
        ]
        built = build_document_structure("ITEM", chunks)
        children = defaultdict(list)
        for node in built["nodes"]:
            if node.get("parent_node_id"):
                children[str(node["parent_node_id"])].append(node)
        chunk_map = {str(row["id"]): row for row in chunks}
        targets = _chapter_summary_targets(built["nodes"], children, chunk_map)
        target_nodes = [node for node in built["nodes"] if node["node_id"] in targets]
        self.assertEqual([node["node_type"] for node in target_nodes], ["attachment_root"])
        self.assertTrue(any(node.get("title") == "Section 1" for node in built["nodes"]))

    def test_short_legacy_pdf_recovers_attachment_key_from_chunk_id(self):
        chunks = [
            {"id": f"ABCD1234:p{index}", "text": "argument " * 100,
             "metadata": {"source_type": "pdf", "structure_path": [f"Section {index}"],
                          "structure_roles": ["chapter"], "zone": "body"}}
            for index in range(1, 5)
        ]
        built = build_document_structure("ITEM", chunks)
        children = defaultdict(list)
        for node in built["nodes"]:
            if node.get("parent_node_id"):
                children[str(node["parent_node_id"])].append(node)
        targets = _chapter_summary_targets(
            built["nodes"], children, {str(row["id"]): row for row in chunks},
        )
        target_nodes = [node for node in built["nodes"] if node["node_id"] in targets]
        self.assertEqual([node["node_type"] for node in target_nodes], ["attachment_root"])
        self.assertEqual(target_nodes[0]["attachment_key"], "ABCD1234")

    def test_short_pdf_aggregation_does_not_cross_repeated_attachment_runs(self):
        chunks = [
            {"id": "01:first", "text": "a" * 90_000,
             "metadata": {"attachmentKey": "SAMEKEY1", "source_type": "pdf",
                          "structure_path": ["First run"],
                          "structure_roles": ["chapter"], "zone": "body"}},
            {"id": "02:middle", "text": "middle",
             "metadata": {"attachmentKey": "OTHER001", "source_type": "html",
                          "structure_path": ["Intervening document"], "zone": "body"}},
            {"id": "03:last", "text": "b" * 90_000,
             "metadata": {"attachmentKey": "SAMEKEY1", "source_type": "pdf",
                          "structure_path": ["Second run"],
                          "structure_roles": ["chapter"], "zone": "body"}},
        ]
        built = build_document_structure("ITEM", chunks)
        children = defaultdict(list)
        for node in built["nodes"]:
            if node.get("parent_node_id"):
                children[str(node["parent_node_id"])].append(node)
        targets = _chapter_summary_targets(
            built["nodes"], children, {str(row["id"]): row for row in chunks},
        )
        repeated_roots = [
            node for node in built["nodes"]
            if node.get("node_type") == "attachment_root"
            and node.get("attachment_key") == "SAMEKEY1"
        ]
        self.assertEqual(len(repeated_roots), 2)
        self.assertTrue(all(node["node_id"] in targets for node in repeated_roots))

    def test_many_page_pdf_keeps_chapter_summary_boundaries(self):
        chunks = [
            {"id": f"A:p{page}", "text": "argument ",
             "metadata": {"attachmentKey": "A", "source_type": "pdf", "page": page,
                          "structure_path": [f"Chapter {1 if page <= 20 else 2}"],
                          "structure_roles": ["chapter"], "zone": "body"}}
            for page in range(1, 41)
        ]
        built = build_document_structure("ITEM", chunks)
        children = defaultdict(list)
        for node in built["nodes"]:
            if node.get("parent_node_id"):
                children[str(node["parent_node_id"])].append(node)
        targets = _chapter_summary_targets(
            built["nodes"], children, {str(row["id"]): row for row in chunks},
        )
        titles = {node.get("title") for node in built["nodes"] if node["node_id"] in targets}
        self.assertEqual(titles, {"Chapter 1", "Chapter 2"})

    def test_chapter_scope_excludes_notes_but_includes_short_body_leaves(self):
        chunks = [
            # Must clear MIN_LEAF_CHARS (1,000): this case is about which
            # leaves are excluded and why, not about the size threshold.
            {"id": "A:p1", "text": "body argument " * 100,
             "metadata": {"attachmentKey": "A", "structure_path": ["Chapter"], "zone": "body"}},
            {"id": "A:p2", "text": "footnote source " * 40,
             "metadata": {"attachmentKey": "A", "structure_path": ["Chapter"], "zone": "footnote"}},
            {"id": "A:p3", "text": "short",
             "metadata": {"attachmentKey": "A", "structure_path": ["Chapter"], "zone": "body"}},
        ]
        built = build_document_structure("ITEM", chunks)
        db_relations.replace_document_structure(
            "ITEM", source_fingerprint=built["source_fingerprint"], structure_version=built["structure_version"],
            status=built["status"], confidence=built["confidence"], nodes=built["nodes"],
            diagnostics=built["diagnostics"],
        )
        with patch("src.build_structure_summaries.get_item_chunks", return_value=chunks):
            build_structure_summaries("ITEM", mode="extractive")
        chapter = next(row for row in db_relations.get_document_node_summaries("ITEM") if row.get("title") == "Chapter")
        reasons = {row["reason"] for row in chapter["input_scope"]["excluded"]}
        self.assertEqual(reasons, {"summary_policy_exclude"})
        self.assertEqual(chapter["source_chars"], len(chunks[0]["text"]) + len(chunks[2]["text"]))

    def test_short_sections_are_summarized_once_as_one_chapter(self):
        chunks = [
            {"id": f"A:p{index}", "text": letter * 600,
             "metadata": {"attachmentKey": "A", "structure_path": ["Chapter", f"Section {index}"],
                          "zone": "body"}}
            for index, letter in enumerate("abc", start=1)
        ]
        built = build_document_structure("ITEM", chunks)
        db_relations.replace_document_structure(
            "ITEM", source_fingerprint=built["source_fingerprint"],
            structure_version=built["structure_version"], status=built["status"],
            confidence=built["confidence"], nodes=built["nodes"], diagnostics=built["diagnostics"],
        )
        calls = []

        def fake_section(section, **kwargs):
            calls.append([row["id"] for row in section["chunks"]])
            return {"summary": "one chapter summary"}, "deepseek:cheap"

        with patch("src.build_structure_summaries.get_item_chunks", return_value=chunks), \
                patch("src.build_structure_summaries._deepseek_model", return_value="deepseek-chat"), \
                patch("src.build_structure_summaries._llm_summary_only_section", side_effect=fake_section):
            result = build_structure_summaries("ITEM", mode="llm")

        self.assertEqual(result["status"], "success")
        self.assertEqual(calls, [["A:p1", "A:p2", "A:p3"]])
        summaries = db_relations.get_document_node_summaries("ITEM")
        chapter = next(row for row in summaries if row.get("title") == "Chapter")
        self.assertEqual(chapter["source_chars"], 1_800)
        self.assertEqual(
            chapter["input_scope"]["included_chunk_ids"], ["A:p1", "A:p2", "A:p3"],
        )
        self.assertFalse(any(str(row.get("title") or "").startswith("Section") for row in summaries))

    def test_search_index_suppresses_one_child_parent_duplicate(self):
        rows = [
            {"node_id": "root", "parent_node_id": None},
            {"node_id": "chapter", "parent_node_id": "root"},
            {"node_id": "a", "parent_node_id": "chapter"},
            {"node_id": "b", "parent_node_id": "chapter"},
        ]
        selected = _select_searchable_summary_rows(rows)
        self.assertEqual([row["node_id"] for row in selected], ["chapter", "a", "b"])

    def _persist_structure(self, item_key, chunks):
        built = build_document_structure(item_key, chunks)
        db_relations.replace_document_structure(
            item_key, source_fingerprint=built["source_fingerprint"],
            structure_version=built["structure_version"], status=built["status"],
            confidence=built["confidence"], nodes=built["nodes"], diagnostics=built["diagnostics"],
        )

    def _llm_patches(self):
        def fake_section(section, **kwargs):
            return {"summary": "chapter summary text"}, "deepseek:cheap"

        def fake_item(title, rows, **kwargs):
            return {"summary": "parent summary text"}, "deepseek:standard"

        return (
            patch("src.build_structure_summaries._deepseek_model", return_value="deepseek-chat"),
            patch("src.build_structure_summaries._llm_summary_only_section", side_effect=fake_section),
            patch("src.build_structure_summaries._llm_summary_only_item", side_effect=fake_item),
        )

    def test_second_run_with_same_fingerprint_skips_without_llm(self):
        chunks = [{
            "id": "A:p1", "text": "body argument " * 80,
            "metadata": {"attachmentKey": "A", "structure_path": ["Chapter"], "zone": "body"},
        }]
        self._persist_structure("ITEM", chunks)
        model_p, section_p, item_p = self._llm_patches()
        with patch("src.build_structure_summaries.get_item_chunks", return_value=chunks), \
                model_p, section_p as section, item_p as item:
            first = build_structure_summaries("ITEM", mode="llm")
            self.assertEqual(first["status"], "success")
            calls_after_first = section.call_count + item.call_count
            self.assertGreater(calls_after_first, 0)
            second = build_structure_summaries("ITEM", mode="llm")
            self.assertEqual(second["status"], "skipped_current")
            self.assertEqual(section.call_count + item.call_count, calls_after_first)

    def test_structure_replacement_reuses_identical_llm_inputs_without_api_calls(self):
        """The destructive tree replace must retain validated V3 summaries."""
        chunks = [{
            "id": "A:p1", "text": "body argument " * 80,
            "metadata": {"attachmentKey": "A", "structure_path": ["Chapter"], "zone": "body"},
        }]
        self._persist_structure("ITEM", chunks)
        model_p, section_p, item_p = self._llm_patches()
        with patch("src.build_structure_summaries.get_item_chunks", return_value=chunks), \
                model_p, section_p as section, item_p as item:
            self.assertEqual(build_structure_summaries("ITEM", mode="llm")["status"], "success")
            self.assertGreater(section.call_count + item.call_count, 0)

        # This is what a STRUCTURE_VERSION rebuild does: the old nodes (and
        # their FK-bound summaries) are replaced, while the reuse cache keeps
        # only the audit candidates outside the live retrieval tables.
        built = build_document_structure("ITEM", chunks)
        db_relations.replace_document_structure(
            "ITEM", source_fingerprint=built["source_fingerprint"],
            structure_version=built["structure_version"], status=built["status"],
            confidence=built["confidence"], nodes=built["nodes"], diagnostics=built["diagnostics"],
        )
        self.assertEqual(db_relations.get_document_node_summaries("ITEM"), [])
        self.assertTrue(db_relations.get_document_node_summary_reuse_cache("ITEM"))

        model_p, section_p, item_p = self._llm_patches()
        with patch("src.build_structure_summaries.get_item_chunks", return_value=chunks), \
                model_p, section_p as section, item_p as item:
            result = build_structure_summaries("ITEM", mode="llm")
        self.assertEqual(result["status"], "success")
        self.assertGreater(result["reused"], 0)
        self.assertEqual(section.call_count + item.call_count, 0)

    def test_changed_leaf_input_is_not_reused_after_structure_replacement(self):
        original = [{
            "id": "A:p1", "text": "original body argument " * 80,
            "metadata": {"attachmentKey": "A", "structure_path": ["Chapter"], "zone": "body"},
        }]
        self._persist_structure("ITEM", original)
        model_p, section_p, item_p = self._llm_patches()
        with patch("src.build_structure_summaries.get_item_chunks", return_value=original), \
                model_p, section_p, item_p:
            build_structure_summaries("ITEM", mode="llm")

        changed = [{**original[0], "text": "changed body argument " * 80}]
        built = build_document_structure("ITEM", changed)
        db_relations.replace_document_structure(
            "ITEM", source_fingerprint=built["source_fingerprint"],
            structure_version=built["structure_version"], status=built["status"],
            confidence=built["confidence"], nodes=built["nodes"], diagnostics=built["diagnostics"],
        )
        model_p, section_p, item_p = self._llm_patches()
        with patch("src.build_structure_summaries.get_item_chunks", return_value=changed), \
                model_p, section_p as section, item_p:
            result = build_structure_summaries("ITEM", mode="llm")
        self.assertGreater(section.call_count, 0)
        self.assertLess(result["reused"], result["nodes"])

    def test_extractive_existing_regenerates_under_llm_mode(self):
        chunks = [{
            "id": "A:p1", "text": "body argument " * 80,
            "metadata": {"attachmentKey": "A", "structure_path": ["Chapter"], "zone": "body"},
        }]
        self._persist_structure("ITEM", chunks)
        with patch("src.build_structure_summaries.get_item_chunks", return_value=chunks):
            build_structure_summaries("ITEM", mode="extractive")
        model_p, section_p, item_p = self._llm_patches()
        with patch("src.build_structure_summaries.get_item_chunks", return_value=chunks), \
                model_p, section_p as section, item_p:
            result = build_structure_summaries("ITEM", mode="llm")
        self.assertNotEqual(result["status"], "skipped_current")
        self.assertGreater(section.call_count, 0)

    def test_parent_node_reduction_uses_standard_role(self):
        chunks = [
            {"id": "A:p1", "text": "chapter one body " * 60,
             "metadata": {"attachmentKey": "A", "structure_path": ["Chapter One"], "zone": "body"}},
            {"id": "A:p2", "text": "chapter two body " * 60,
             "metadata": {"attachmentKey": "A", "structure_path": ["Chapter Two"], "zone": "body"}},
        ]
        self._persist_structure("ITEM", chunks)
        roles: list[str] = []

        def record_role(role):
            roles.append(role)
            return "deepseek-model"

        _model_p, section_p, item_p = self._llm_patches()
        with patch("src.build_structure_summaries.get_item_chunks", return_value=chunks), \
                \
                patch("src.build_structure_summaries._deepseek_model", side_effect=record_role), \
                section_p, item_p:
            result = build_structure_summaries("ITEM", mode="llm")
        self.assertEqual(result["status"], "success")
        # Leaves summarize with the cheap role; the parent (item root) reduction
        # over the two chapter summaries uses the standard role (spec §5.1).
        self.assertIn("cheap", roles)
        self.assertIn("standard", roles)

    def test_summary_reads_the_explicit_parallel_collection(self):
        chunks = [{
            "id": "A:p1", "text": "body argument " * 50,
            "metadata": {"attachmentKey": "A", "structure_path": ["Chapter"], "zone": "body"},
        }]
        built = build_document_structure("ITEM", chunks)
        db_relations.replace_document_structure(
            "ITEM", source_fingerprint=built["source_fingerprint"],
            structure_version=built["structure_version"], status=built["status"],
            confidence=built["confidence"], nodes=built["nodes"], diagnostics=built["diagnostics"],
        )
        with patch("src.build_structure_summaries.get_item_chunks", return_value=chunks) as loader:
            build_structure_summaries(
                "ITEM", mode="extractive", collection_name="zotero_paragraphs_v3",
            )
        loader.assert_called_once_with("ITEM", collection_name="zotero_paragraphs_v3")
    def test_a_single_child_parent_adopts_its_child_without_an_llm_call(self):
        """Its text *is* the child's text, so reducing it is a summary of a
        summary -- and `_select_searchable_summary_rows` then suppresses it at
        embed time anyway. A full rebuild produced 9,623 such parents, 98%
        matching the child's source length to within 5%, accounting for 42% of
        that run's LLM calls: generated, paid for, discarded (2026-07-28).
        """
        chunks = [{
            "id": "A:p1", "text": "the only body paragraph " * 80,
            "metadata": {"attachmentKey": "A", "structure_path": ["Only Chapter"], "zone": "body"},
        }]
        self._persist_structure("ITEM", chunks)
        parent_calls = []

        def fake_section(section, **kwargs):
            return {"summary": "leaf summary text"}, "deepseek:cheap"

        def fake_item(title, rows, **kwargs):
            parent_calls.append(title)
            return {"summary": "parent summary text"}, "deepseek:standard"

        with patch("src.build_structure_summaries._deepseek_model", return_value="deepseek-chat"), \
             patch("src.build_structure_summaries._llm_summary_only_section", side_effect=fake_section), \
             patch("src.build_structure_summaries._llm_summary_only_item", side_effect=fake_item), \
             patch("src.build_structure_summaries.get_item_chunks", return_value=chunks):
            build_structure_summaries("ITEM", mode="llm")

        self.assertEqual(parent_calls, [], "a one-child parent must not be reduced")
        summaries = {
            row["node_id"]: row["summary"]
            for row in db_relations.get_all_document_node_summaries()
            if row["item_key"] == "ITEM"
        }
        self.assertTrue(summaries, "the chain must still be populated")
        self.assertNotIn("parent summary text", set(summaries.values()))

    def test_paying_and_keeping_consult_the_same_rule(self):
        """One definition, not two agreeing copies.

        The rule lived only at the index, which discarded 9,766 parents that
        generation had already paid an LLM to produce. Two copies would fix
        that until one of them drifted -- and the direction of drift is
        spending. Redefining the rule must therefore move both sites at once:
        here it is inverted, and both must follow it.
        """
        chunks = [{
            "id": "A:p1", "text": "the only body paragraph " * 80,
            "metadata": {"attachmentKey": "A", "structure_path": ["Only Chapter"], "zone": "body"},
        }]
        self._persist_structure("ITEM", chunks)
        parent_calls = []

        def fake_item(title, rows, **kwargs):
            parent_calls.append(title)
            return {"summary": "parent summary text"}, "deepseek:standard"

        # Inverted: a one-child parent now *does* add something.
        with patch("src.build_structure_summaries.adds_nothing_over_its_children",
                   side_effect=lambda count: count != 1), \
             patch("src.build_structure_summaries._deepseek_model", return_value="deepseek-chat"), \
             patch("src.build_structure_summaries._llm_summary_only_section",
                   side_effect=lambda section, **kw: ({"summary": "leaf summary text"}, "deepseek:cheap")), \
             patch("src.build_structure_summaries._llm_summary_only_item", side_effect=fake_item), \
             patch("src.build_structure_summaries.get_item_chunks", return_value=chunks):
            build_structure_summaries("ITEM", mode="llm")
            # Generation followed the redefinition and paid for the parent.
            # The chapter is now summarized directly from its raw descendants;
            # only attachment_root and item_root are parent reductions.
            self.assertEqual(len(parent_calls), 2)
            # The index must follow the very same redefinition and keep it.
            rows = [
                {"node_id": "root", "parent_node_id": None},
                {"node_id": "chapter", "parent_node_id": "root"},
                {"node_id": "leaf", "parent_node_id": "chapter"},
            ]
            kept = {row["node_id"] for row in _select_searchable_summary_rows(rows)}
        self.assertIn("chapter", kept)

    def test_a_parent_with_one_extractive_and_one_llm_child_is_not_paid_for(self):
        """Generation and the embed-time selector must count children the same way.

        generated{} held every child regardless of kind, so a parent with one
        LLM child and one extractive-fallback child looked like a two-child
        parent during generation (paid for its own reduction) but looked like
        a one-child parent at embed time, once get_all_document_node_summaries
        (searchable_only=True) filtered the extractive sibling out -- and was
        then discarded as a duplicate (2026-07-28, found in code review).
        """
        from src.llm_client import LLMError

        chunks = [
            {"id": "A:p1", "text": "the llm-summarized paragraph text here " * 40,
             "metadata": {"attachmentKey": "A", "structure_path": ["Chapter", "SubA"], "zone": "body"}},
            {"id": "A:p2", "text": "the extractive-fallback paragraph text here " * 40,
             "metadata": {"attachmentKey": "A", "structure_path": ["Chapter", "SubB"], "zone": "body"}},
        ]
        self._persist_structure("ITEM", chunks)
        parent_calls = []

        def fake_section(section, **kwargs):
            if section.get("chapter") == "SubB":
                raise LLMError("simulated failure")
            return {"summary": "leaf summary text"}, "deepseek:cheap"

        def fake_item(title, rows, **kwargs):
            parent_calls.append(title)
            return {"summary": "parent summary text"}, "deepseek:standard"

        with patch("src.build_structure_summaries._deepseek_model", return_value="deepseek-chat"), \
             patch("src.build_structure_summaries._llm_summary_only_section", side_effect=fake_section), \
             patch("src.build_structure_summaries._llm_summary_only_item", side_effect=fake_item), \
             patch("src.build_structure_summaries.get_item_chunks", return_value=chunks):
            build_structure_summaries("ITEM", mode="llm")

        self.assertEqual(
            parent_calls, [],
            "the parent has exactly one searchable (llm) child and must adopt it, not be reduced",
        )
        rows = [
            row for row in db_relations.get_all_document_node_summaries()
            if row["item_key"] == "ITEM"
        ]
        summaries = {row["summary"] for row in rows}
        self.assertNotIn("parent summary text", summaries)

    def test_a_single_non_searchable_child_is_still_adopted_without_payment(self):
        """A genuine single child must stay cheap regardless of searchability.

        Restricting the single-child check to searchable children (the fix
        above) must not regress the ordinary case: a parent whose only child
        happens to be an extractive fallback still has nothing to gain from a
        paid reduction over it, and must be adopted exactly as before.
        """
        from src.llm_client import LLMError

        chunks = [{
            "id": "A:p1", "text": "the only extractive paragraph text here " * 40,
            "metadata": {"attachmentKey": "A", "structure_path": ["Chapter", "SubB"], "zone": "body"},
        }]
        self._persist_structure("ITEM", chunks)
        parent_calls = []

        def fake_section(section, **kwargs):
            raise LLMError("simulated failure")

        def fake_item(title, rows, **kwargs):
            parent_calls.append(title)
            return {"summary": "parent summary text"}, "deepseek:standard"

        with patch("src.build_structure_summaries._deepseek_model", return_value="deepseek-chat"), \
             patch("src.build_structure_summaries._llm_summary_only_section", side_effect=fake_section), \
             patch("src.build_structure_summaries._llm_summary_only_item", side_effect=fake_item), \
             patch("src.build_structure_summaries.get_item_chunks", return_value=chunks):
            build_structure_summaries("ITEM", mode="llm")

        self.assertEqual(parent_calls, [], "a single child, searchable or not, must be adopted for free")

    def test_the_shipped_rule_suppresses_exactly_the_one_child_case(self):
        self.assertTrue(adds_nothing_over_its_children(1))
        self.assertFalse(adds_nothing_over_its_children(2))
        self.assertFalse(adds_nothing_over_its_children(0))


if __name__ == "__main__":
    unittest.main()


class InheritedTitleTests(unittest.TestCase):
    """A chapter heading must reach the summary index.

    Leaves are semantic_segments with no title, and the titled parent above
    them is usually dropped as a one-child duplicate, so the embedded document
    -- title plus summary -- carried no heading at all: 11,262 of 12,280 rows
    had an empty title, and for 9,160 the heading existed in no searchable
    field. Searching for a chapter by name could not find it (2026-07-28).
    """

    def test_a_leaf_inherits_the_chapter_heading(self):
        from src.build_structure_summaries import inherited_title
        titles = {"root": "", "chap": "Chapter One: Method", "leaf": ""}
        parents = {"leaf": "chap", "chap": "root", "root": ""}
        self.assertEqual(inherited_title("leaf", titles, parents), "Chapter One: Method")

    def test_a_node_with_its_own_title_keeps_it(self):
        from src.build_structure_summaries import inherited_title
        self.assertEqual(
            inherited_title("chap", {"chap": "Own", "root": "Ancestor"}, {"chap": "root"}), "Own")

    def test_an_untitled_tree_yields_nothing_rather_than_guessing(self):
        from src.build_structure_summaries import inherited_title
        self.assertEqual(inherited_title("leaf", {"leaf": "", "root": ""}, {"leaf": "root"}), "")

    def test_a_cycle_terminates(self):
        # Node ids are content-derived and a malformed tree must not hang the
        # embedding pass.
        from src.build_structure_summaries import inherited_title
        self.assertEqual(inherited_title("a", {"a": "", "b": ""}, {"a": "b", "b": "a"}), "")

    def test_resolution_uses_the_unsuppressed_tree(self):
        # The titled parent is exactly the node that one-child suppression
        # removes, so resolving from the surviving rows alone would find nothing.
        from src.build_structure_summaries import (
            _select_searchable_summary_rows, inherited_title,
        )
        all_rows = [
            {"node_id": "chap", "parent_node_id": "root", "title": "Chapter One"},
            {"node_id": "leaf", "parent_node_id": "chap", "title": None},
        ]
        titles = {str(r["node_id"]): str(r.get("title") or "") for r in all_rows}
        parents = {str(r["node_id"]): str(r.get("parent_node_id") or "") for r in all_rows}
        surviving = _select_searchable_summary_rows(all_rows)
        self.assertEqual([r["node_id"] for r in surviving], ["leaf"])
        self.assertEqual(inherited_title("leaf", titles, parents), "Chapter One")
