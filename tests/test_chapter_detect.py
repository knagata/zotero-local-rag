from __future__ import annotations

import unittest

from src.chapter_detect import build_epub_toc_tree, build_pdf_outline_tree


class _Link:
    def __init__(self, title: str, href: str):
        self.title = title
        self.href = href


class ChapterDetectTests(unittest.TestCase):
    def test_pdf_outline_keeps_all_levels(self):
        tree = build_pdf_outline_tree([
            (1, "Part I", 1), (2, "Chapter 1", 3), (3, "Methods", 5),
            (2, "Chapter 2", 9), (1, "Part II", 20),
        ])
        self.assertEqual([node["title"] for node in tree], ["Part I", "Part II"])
        self.assertEqual(tree[0]["children"][0]["children"][0]["title"], "Methods")
        self.assertEqual(tree[0]["children"][1]["title"], "Chapter 2")

    def test_epub_toc_keeps_href_fragment_and_nested_children(self):
        tree = build_epub_toc_tree([
            (_Link("Part I", "part.xhtml"), [
                _Link("Chapter 1", "chapter.xhtml#start"),
            ]),
        ])
        self.assertEqual(tree[0]["href"], "part.xhtml")
        child = tree[0]["children"][0]
        self.assertEqual(child["href"], "chapter.xhtml")
        self.assertEqual(child["fragment"], "start")
        self.assertEqual(child["depth"], 2)


if __name__ == "__main__":
    unittest.main()
