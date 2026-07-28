from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from src import db_relations
from src.build_structure_summaries import (
    MAX_PARENT_INPUT_CHARS, _chunk_groups, _select_searchable_summary_rows,
    adds_nothing_over_its_children, build_structure_summaries,
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

    def test_leaf_input_groups_are_ordered_and_bounded_at_chunk_boundaries(self):
        rows = [{"id": f"c{index}", "text": letter * 16_000} for index, letter in enumerate("abc")]
        groups = _chunk_groups(rows)
        self.assertEqual([[row["id"] for row in group] for group in groups], [["c0"], ["c1"], ["c2"]])
        self.assertTrue(all(sum(len(row["text"]) for row in group) <= MAX_PARENT_INPUT_CHARS for group in groups))

    def test_parent_scope_excludes_notes_and_short_display_only_leaves(self):
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
        self.assertEqual(reasons, {"summary_policy_exclude", "below_min_chars"})
        self.assertEqual(chapter["source_chars"], len(chunks[0]["text"]))

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


if __name__ == "__main__":
    unittest.main()

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
            self.assertEqual(len(parent_calls), 1)
            # The index must follow the very same redefinition and keep it.
            rows = [
                {"node_id": "root", "parent_node_id": None},
                {"node_id": "chapter", "parent_node_id": "root"},
                {"node_id": "leaf", "parent_node_id": "chapter"},
            ]
            kept = {row["node_id"] for row in _select_searchable_summary_rows(rows)}
        self.assertIn("chapter", kept)

    def test_the_shipped_rule_suppresses_exactly_the_one_child_case(self):
        self.assertTrue(adds_nothing_over_its_children(1))
        self.assertFalse(adds_nothing_over_its_children(2))
        self.assertFalse(adds_nothing_over_its_children(0))



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
