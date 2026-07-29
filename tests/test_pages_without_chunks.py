"""P2-13 (2026-07-29): extraction=success is a per-attachment binary.

An attachment can hold this status while individual pages inside it produced
zero chunks, and nothing in the ledger said which -- 583 attachments recorded
success while 539 pages across them had no indexed text, found only by
comparing against the original source files.
"""
from __future__ import annotations

import sys
import typing
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from index_from_zotero import (  # noqa: E402
    _pages_without_chunks,
    _source_content_unchanged,
)


class PagesWithoutChunksTests(unittest.TestCase):
    def test_a_gap_in_the_middle_is_reported(self):
        chunks = [
            ("A:p1:para0", "text", {"page": 1}),
            ("A:p3:para0", "text", {"page": 3}),
        ]
        self.assertEqual(_pages_without_chunks(chunks, 3), [2])

    def test_full_coverage_reports_nothing(self):
        chunks = [("A:p1:para0", "text", {"page": 1}), ("A:p2:para0", "text", {"page": 2})]
        self.assertEqual(_pages_without_chunks(chunks, 2), [])

    def test_page_as_string_digit_still_counts(self):
        chunks = [("A:p1:para0", "text", {"page": "1"})]
        self.assertEqual(_pages_without_chunks(chunks, 2), [2])

    def test_unknown_expected_page_count_returns_empty_rather_than_guessing(self):
        chunks = [("A:p1:para0", "text", {"page": 1})]
        self.assertEqual(_pages_without_chunks(chunks, None), [])
        self.assertEqual(_pages_without_chunks(chunks, 0), [])

    def test_zero_chunks_reports_every_expected_page(self):
        self.assertEqual(_pages_without_chunks([], 3), [1, 2, 3])

    def test_public_helper_annotations_resolve(self):
        typing.get_type_hints(_pages_without_chunks)
        typing.get_type_hints(_source_content_unchanged)


if __name__ == "__main__":
    unittest.main()
