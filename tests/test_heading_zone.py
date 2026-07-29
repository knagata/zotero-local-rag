"""Tests for the single heading-to-zone classifier.

Three extractors each carried their own copy of this vocabulary, and the main
PDF path (pdf_extract.py) carried none at all -- an estimated 10.6 million
characters across 217 items were classified as body text while sitting under
headings like "Sources" or "Notes" (2026-07-28).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heading_zone import classify_heading, classify_heading_path  # noqa: E402


class ClassifyHeadingTests(unittest.TestCase):
    def test_bibliography_vocabulary(self):
        for title in ("References", "Bibliography", "Select Bibliography",
                      "Further Reading", "Works Cited", "参考文献", "参考資料"):
            self.assertEqual(classify_heading(title), "bibliography", title)

    def test_endnote_vocabulary_including_the_per_chapter_form(self):
        for title in ("Notes", "Endnotes", "Notes to Chapter 3", "巻末注", "注記"):
            self.assertEqual(classify_heading(title), "endnote", title)

    def test_footnote_vocabulary(self):
        self.assertEqual(classify_heading("Footnotes"), "footnote")
        self.assertEqual(classify_heading("脚注"), "footnote")

    def test_index_vocabulary(self):
        for title in ("Index", "General Index", "地名索引", "索引"):
            self.assertEqual(classify_heading(title), "index", title)

    def test_back_matter_vocabulary(self):
        for title in ("Glossary", "Appendix", "あとがき"):
            self.assertEqual(classify_heading(title), "back_matter", title)

    def test_ordinary_chapter_titles_stay_body(self):
        for title in ("Introduction", "Chapter One", "The Politics of Images"):
            self.assertEqual(classify_heading(title), "body", title)

    def test_a_title_merely_starting_with_paratext_vocabulary_stays_body(self):
        # The trailing-qualifier allowance is restricted to "to"/"for" so it
        # cannot absorb an unrelated chapter title.
        self.assertEqual(classify_heading("Sources of Power"), "body")
        self.assertEqual(classify_heading("Notes on the State of Virginia"), "body")

    def test_empty_or_missing_title_is_body(self):
        self.assertEqual(classify_heading(""), "body")
        self.assertEqual(classify_heading(None), "body")

    def test_pdf_outline_format_controls_and_section_numbers_are_ignored(self):
        self.assertEqual(classify_heading("\ufeffReferences"), "bibliography")
        self.assertEqual(classify_heading("6 References"), "bibliography")
        self.assertEqual(classify_heading("6. References"), "bibliography")
        self.assertEqual(classify_heading("IV Notes"), "endnote")


class ClassifyHeadingPathTests(unittest.TestCase):
    def test_a_paratext_ancestor_governs_an_unmarked_descendant(self):
        # "Chapter 3" carries no vocabulary of its own; everything under
        # "Notes" is a note regardless. A leaf-only check misses this.
        self.assertEqual(classify_heading_path(["Notes", "Chapter 3"]), "endnote")

    def test_an_ordinary_chapter_with_no_paratext_ancestor_stays_body(self):
        self.assertEqual(classify_heading_path(["Part One", "Chapter 3"]), "body")

    def test_outermost_match_wins_over_a_closer_one(self):
        # Not load-bearing in practice (nesting bibliography-under-bibliography
        # is unusual), but the contract is explicit: check top-down, return
        # the first match.
        self.assertEqual(
            classify_heading_path(["Bibliography", "Notes"]), "bibliography")

    def test_an_empty_path_is_body(self):
        self.assertEqual(classify_heading_path([]), "body")

    def test_a_single_element_path_behaves_like_classify_heading(self):
        self.assertEqual(classify_heading_path(["Index"]), "index")


if __name__ == "__main__":
    unittest.main()
