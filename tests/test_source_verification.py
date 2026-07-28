"""Tests for verification against the original files.

Each case here is one of the four failures that were live in the index on
2026-07-28 while every existing gate reported success. If a case stops failing
without the corresponding defect being fixed, this check has lost the property
it was built for.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from source_verification import (  # noqa: E402
    chunks_without_item, compare_document, dangling_node_ids, indexed_page_chars,
    unretrievable_documents,
)


def _chunk(chunk_id, text, **metadata):
    return {"id": chunk_id, "text": text, "metadata": metadata}


class LostPageTests(unittest.TestCase):
    """A page the source can read must have text in the index."""

    def test_a_page_of_source_text_with_nothing_indexed_is_a_loss(self):
        # The shape of the endnote-section loss: 40 pages readable in the PDF,
        # nothing in the index, and a character ratio of 0.81 that sat inside
        # the outlier threshold and so raised nothing.
        source = {n: 3000 for n in range(1, 41)}
        indexed = {n: 3000 for n in range(1, 21)}
        verdict = compare_document("ATT", source, indexed)
        self.assertEqual(verdict.lost_pages, list(range(21, 41)))
        self.assertTrue(verdict.failed)

    def test_a_fully_indexed_document_passes(self):
        source = {1: 2000, 2: 2100}
        verdict = compare_document("ATT", source, {1: 1900, 2: 2000})
        self.assertEqual(verdict.lost_pages, [])
        self.assertFalse(verdict.failed)

    def test_indexed_text_absent_from_the_source_layer_is_not_a_failure(self):
        # A scanned page has no text layer; OCR supplies it. The check is
        # deliberately one-directional.
        verdict = compare_document("ATT", {1: 0, 2: 0}, {1: 4000, 2: 3800})
        self.assertEqual(verdict.lost_pages, [])
        self.assertFalse(verdict.failed)

    def test_a_scanned_page_nobody_covered_is_reported_separately(self):
        # Neither a loss (there was no text to lose) nor a success. Folding it
        # into either bucket would hide the population OCR is meant to reach.
        verdict = compare_document("ATT", {1: 0, 2: 5000}, {2: 5000})
        self.assertEqual(verdict.lost_pages, [])
        self.assertEqual(verdict.unreadable_pages, [1])

    def test_a_running_head_alone_does_not_count_as_page_text(self):
        # Presence must mean text, not a folio number, or every blank verso
        # becomes a false alarm.
        verdict = compare_document("ATT", {1: 8}, {})
        self.assertEqual(verdict.lost_pages, [])

    def test_the_short_page_threshold_is_not_a_loss_tolerance(self):
        # 19 characters is not evidence of text; 20 is, and its absence fails
        # regardless of how small a fraction of the document it represents.
        self.assertEqual(compare_document("A", {1: 19}, {}).lost_pages, [])
        self.assertEqual(compare_document("A", {1: 20}, {}).lost_pages, [1])


class IndexedPageCharsTests(unittest.TestCase):
    def test_pages_accumulate_across_chunks(self):
        rows = [
            _chunk("a", "one", page=1), _chunk("b", "two", page=1), _chunk("c", "three", page=2),
        ]
        self.assertEqual(indexed_page_chars(rows), {1: 6, 2: 5})

    def test_rows_without_a_page_are_ignored_rather_than_miscounted(self):
        rows = [_chunk("a", "text", page=0), _chunk("b", "text"), _chunk("c", "text", page="x")]
        self.assertEqual(indexed_page_chars(rows), {})


class CompanionInvariantTests(unittest.TestCase):
    def test_a_chunk_with_no_item_is_reported(self):
        rows = [_chunk("a", "t", itemKey="ITEM"), _chunk("b", "t"), _chunk("c", "t", itemKey="  ")]
        self.assertEqual(chunks_without_item(rows), ["b", "c"])

    def test_a_node_id_naming_nothing_is_reported(self):
        rows = [_chunk("a", "t", node_id="dn:live"), _chunk("b", "t", node_id="dn:gone")]
        self.assertEqual(dangling_node_ids(rows, ["dn:live"]), ["b"])

    def test_a_chunk_with_no_node_id_is_not_dangling(self):
        # Absent is a different problem from wrong, and conflating them would
        # bury the 45% that point at nodes which no longer exist.
        self.assertEqual(dangling_node_ids([_chunk("a", "t")], ["dn:live"]), [])

    def test_a_document_no_query_can_reach_is_reported(self):
        # The shape of the zone-erased essay: text present, every chunk excluded.
        rows_by_item = {
            "GONE": [_chunk("a", "essay text", retrieval_policy="exclude")],
            "FINE": [_chunk("b", "essay text", retrieval_policy="normal")],
        }
        self.assertEqual(unretrievable_documents(rows_by_item), ["GONE"])

    def test_a_partially_reachable_document_passes(self):
        rows_by_item = {"OK": [
            _chunk("a", "body", retrieval_policy="normal"),
            _chunk("b", "index entry", retrieval_policy="exclude"),
        ]}
        self.assertEqual(unretrievable_documents(rows_by_item), [])

    def test_an_empty_document_is_not_reported_as_unreachable(self):
        # Nothing to reach is a different finding, and belongs to the page check.
        self.assertEqual(unretrievable_documents({"EMPTY": [_chunk("a", "   ")]}), [])

    def test_a_chunk_with_no_policy_counts_as_reachable(self):
        # The retrieval filter defaults absent policy to "normal", so this must
        # agree with it -- otherwise the check disagrees with the system it audits.
        self.assertEqual(unretrievable_documents({"OK": [_chunk("a", "body")]}), [])



class BlankPageNoticeTests(unittest.TestCase):
    """A page that says it is blank is not a page that was lost.

    193 of the first run's 539 "lost" pages were deliberately blank. A check
    that is wrong about a third of its output teaches its reader to discount
    all of it, which costs more than the finding is worth.
    """

    def test_the_common_notices_are_recognised(self):
        from source_verification import is_blank_page_notice
        for text in ("This page intentionally left blank.",
                     "THIS PAGE INTENTIONALLY LEFT BLANK",
                     "  ", "", "白紙"):
            self.assertTrue(is_blank_page_notice(text), text)

    def test_real_text_is_not_mistaken_for_a_notice(self):
        from source_verification import is_blank_page_notice
        self.assertFalse(is_blank_page_notice("Notes INTRODUCTION Epigraph. Nietzsche 2001, 120."))
        self.assertFalse(is_blank_page_notice("A short but real page of body text."))

    def test_a_long_page_merely_mentioning_the_phrase_is_not_a_notice(self):
        # A page discussing blank pages is a page of text.
        long_text = ("The convention of printing 'this page intentionally left blank' "
                     "originated with military manuals, where an unmarked page invited "
                     "the suspicion that content had been removed. " * 2)
        from source_verification import is_blank_page_notice
        self.assertFalse(is_blank_page_notice(long_text))

if __name__ == "__main__":
    unittest.main()
