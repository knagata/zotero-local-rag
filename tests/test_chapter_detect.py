from __future__ import annotations

import unittest
from unittest.mock import patch

from src.chapter_detect import (
    _is_usable_pdf_toc,
    _recover_flat_pdf_toc,
    _recover_flat_toc_paths,
    _toc_entries_by_href,
    build_epub_toc_tree,
    build_pdf_page_chapter_lookup,
    build_pdf_page_structure_path_lookup,
    build_pdf_outline_tree,
    get_pdf_toc,
    get_epub_chapter_index_to_toc_entries,
    infer_structure_roles,
)


class _Link:
    def __init__(self, title: str, href: str):
        self.title = title
        self.href = href


class _EpubItem:
    def __init__(self, name: str):
        self._name = name

    def get_name(self):
        return self._name


class ChapterDetectTests(unittest.TestCase):
    def test_generated_page_bookmarks_are_not_a_document_outline(self):
        toc = [
            (1, f"{page:05d}___{'a' * 32}", page + 1)
            for page in range(174)
        ]

        self.assertFalse(_is_usable_pdf_toc(toc))
        self.assertIsNone(build_pdf_page_chapter_lookup(toc))
        self.assertIsNone(build_pdf_page_structure_path_lookup(toc))

    def test_generated_page_bookmarks_are_rejected_at_the_pdf_reader_boundary(self):
        class Document:
            def get_toc(self):
                return [[1, f"{page:05d}___{'a' * 32}", page + 1] for page in range(4)]

            def close(self):
                pass

        with patch("src.chapter_detect.fitz.open", return_value=Document()):
            self.assertEqual(get_pdf_toc("generated.pdf"), [])

    def test_one_machine_bookmark_does_not_discard_real_chapters(self):
        toc = [
            (1, f"00000___{'a' * 32}", 1),
            (1, "In Plato's Cave", 10),
            (1, "Melancholy Objects", 48),
        ]

        self.assertTrue(_is_usable_pdf_toc(toc))

    def test_filename_navigation_is_not_a_document_outline(self):
        toc = [(1, "scan-001.jpg", 1), (1, "scan-002.jpg", 2)]

        self.assertFalse(_is_usable_pdf_toc(toc))

    def test_epub_toc_mapping_uses_opf_spine_indices_not_all_documents(self):
        class Book:
            toc = [_Link("Chapter", "Text/chapter.xhtml")]
            spine = [("cover", "yes"), ("chapter", "yes")]

            def get_item_with_id(self, idref):
                return {
                    "cover": _EpubItem("Text/cover.xhtml"),
                    "chapter": _EpubItem("Text/chapter.xhtml"),
                }.get(idref)

        with patch("src.chapter_detect._ebooklib_epub.read_epub", return_value=Book()):
            entries = get_epub_chapter_index_to_toc_entries("book.epub")

        self.assertNotIn(0, entries)
        self.assertEqual(entries[1][0]["path"], ["Chapter"])

    def test_flat_pdf_parts_recover_chapter_levels(self):
        recovered = _recover_flat_pdf_toc([
            (1, "Introduction", 1),
            (1, "Part I. Foundations", 10),
            (1, "1. First Essay", 10),
            (1, "2. Second Essay", 20),
            (1, "III. Contemporary Positions", 30),
            (1, "9. Later Essay", 30),
            (1, "Bibliography", 40),
        ])
        self.assertEqual([level for level, _title, _page in recovered], [1, 1, 2, 2, 1, 2, 1])

    def test_flat_pdf_recovers_standalone_roman_container(self):
        recovered = _recover_flat_pdf_toc([
            (1, "III. Contemporary Positions", 30),
            (1, "9. Later Essay", 30),
            (1, "10. Final Essay", 40),
            (1, "Bibliography", 50),
        ])
        self.assertEqual([level for level, _title, _page in recovered], [1, 2, 2, 1])

    def test_flat_pdf_does_not_promote_unpunctuated_roman_chapter(self):
        recovered = _recover_flat_pdf_toc([
            (1, "I Introduction", 1),
            (1, "1. Background", 2),
        ])
        self.assertEqual([level for level, _title, _page in recovered], [1, 1])

    def test_pdf_roles_distinguish_part_chapter_and_section(self):
        self.assertEqual(
            infer_structure_roles(["Part One", "Chapter 1", "Methods"]),
            ["part", "chapter", "section"],
        )
        self.assertEqual(
            infer_structure_roles(["III. Contemporary Positions", "9. Later Essay"]),
            ["part", "chapter"],
        )
        self.assertEqual(
            infer_structure_roles(["PART 1 Inventing media software", "1 Alan Kay"]),
            ["part", "chapter"],
        )

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

    def test_part_and_chapter_root_siblings_recover_with_nested_sections(self):
        tree = build_epub_toc_tree([
            _Link("PART 1 Inventing media software", "part.xhtml"),
            (_Link("1 Alan Kay", "chapter.xhtml"), [
                _Link("Appearance versus function", "chapter.xhtml#appearance"),
            ]),
            _Link("Conclusion", "conclusion.xhtml"),
        ])
        entries = _toc_entries_by_href(tree)
        self.assertEqual(entries["chapter.xhtml"][0]["path"], [
            "PART 1 Inventing media software", "1 Alan Kay",
        ])
        self.assertEqual(entries["chapter.xhtml"][0]["roles"], ["part", "chapter"])
        self.assertEqual(entries["chapter.xhtml"][1]["path"], [
            "PART 1 Inventing media software", "1 Alan Kay", "Appearance versus function",
        ])
        self.assertEqual(entries["conclusion.xhtml"][0]["path"], ["Conclusion"])

    def test_flat_japanese_toc_recovers_only_explicit_chapter_sections(self):
        tree = build_epub_toc_tree([
            _Link("序章", "intro.xhtml"),
            _Link("一 本書の主題", "intro.xhtml#a"),
            _Link("二 方法", "intro.xhtml#b"),
            _Link("第１章 歴史", "one.xhtml"),
            _Link("一 出発点", "one.xhtml#a"),
            _Link("注", "notes.xhtml"),
            _Link("［第１章］", "notes.xhtml#one"),
            _Link("Notes", "english-notes.xhtml"),
            _Link("About the Author", "author.xhtml"),
            _Link("参考文献", "refs.xhtml"),
        ])
        paths = _recover_flat_toc_paths(tree)
        self.assertEqual(paths[id(tree[1])], ["序章", "一 本書の主題"])
        self.assertEqual(paths[id(tree[4])], ["第１章 歴史", "一 出発点"])
        self.assertEqual(paths[id(tree[6])], ["注", "［第１章］"])
        self.assertEqual(paths[id(tree[7])], ["Notes"])
        self.assertEqual(paths[id(tree[8])], ["About the Author"])
        self.assertEqual(paths[id(tree[9])], ["参考文献"])

    def test_same_epub_document_keeps_each_nested_fragment_path(self):
        tree = build_epub_toc_tree([(
            _Link("I Studying Culture at Scale", "part.xhtml"),
            [(
                _Link("2 The Science of Culture?", "chapter.xhtml#chapter"),
                [_Link("Analyzing and Visualizing", "chapter.xhtml#section")],
            )],
        )])
        entries = _toc_entries_by_href(tree)["chapter.xhtml"]
        self.assertEqual(entries, [
            {
                "fragment": "chapter",
                "path": ["I Studying Culture at Scale", "2 The Science of Culture?"],
                "roles": ["part", "chapter"],
            },
            {
                "fragment": "section",
                "path": [
                    "I Studying Culture at Scale", "2 The Science of Culture?",
                    "Analyzing and Visualizing",
                ],
                "roles": ["part", "chapter", "section"],
            },
        ])

    def test_nested_chapter_with_subsections_is_not_mislabelled_as_part(self):
        tree = build_epub_toc_tree([(
            _Link("Chapter 1", "chapter.xhtml#chapter"),
            [(
                _Link("Section A", "chapter.xhtml#section"),
                [_Link("Detail", "chapter.xhtml#detail")],
            )],
        )])
        entries = _toc_entries_by_href(tree)["chapter.xhtml"]
        self.assertEqual(entries[0]["roles"], ["chapter"])
        self.assertEqual(entries[1]["roles"], ["chapter", "section"])
        self.assertEqual(entries[2]["roles"], ["chapter", "section", "subsection"])

    def test_kanji_numbered_direct_part_child_is_a_section_role(self):
        tree = build_epub_toc_tree([
            _Link("第Ⅰ部 交換様式", "part.xhtml"),
            _Link("一 生産から交換へ", "part.xhtml#one"),
        ])
        entries = _toc_entries_by_href(tree)["part.xhtml"]
        self.assertEqual(entries[1]["roles"], ["part", "section"])

    def test_roman_numbered_root_chapters_are_not_automatically_parts(self):
        tree = build_epub_toc_tree([
            (_Link("I Introduction", "one.xhtml"), [
                _Link("Background", "one.xhtml#background"),
            ]),
            (_Link("II Methods", "two.xhtml"), [
                _Link("Sampling", "two.xhtml#sampling"),
            ]),
        ])
        entries = _toc_entries_by_href(tree)
        self.assertEqual(entries["one.xhtml"][0]["roles"], ["chapter"])
        self.assertEqual(entries["one.xhtml"][1]["roles"], ["chapter", "section"])

    def test_flat_toc_recovers_part_chapter_section_patterns(self):
        tree = build_epub_toc_tree([
            _Link("第Ⅱ部 世界帝国", "part.xhtml"),
            _Link("１章 共同体と国家", "chapter.xhtml"),
            _Link("１ 未開社会と戦争", "chapter.xhtml#one"),
            _Link("２ 国家の誕生", "chapter.xhtml#two"),
            _Link("第１講 青銅器文化", "lecture.xhtml"),
            _Link("１ 東南アジア地域の特徴", "lecture.xhtml#one"),
        ])
        paths = _recover_flat_toc_paths(tree)
        self.assertEqual(paths[id(tree[1])], ["第Ⅱ部 世界帝国", "１章 共同体と国家"])
        self.assertEqual(
            paths[id(tree[2])],
            ["第Ⅱ部 世界帝国", "１章 共同体と国家", "１ 未開社会と戦争"],
        )
        self.assertEqual(
            paths[id(tree[5])],
            ["第１講 青銅器文化", "１ 東南アジア地域の特徴"],
        )

    def test_flat_toc_recovers_explicit_sections_and_unnumbered_children(self):
        tree = build_epub_toc_tree([
            _Link("第一章 渋谷に誕生した盛り", "chapter.xhtml"),
            _Link("第一節 渋谷に生まれたコミュニティ", "section.xhtml"),
            _Link("茶髪になった同級生", "section.xhtml#a"),
            _Link("学校の枠を超えたコミュニティ", "section.xhtml#b"),
            _Link("第二節 雑誌が作った空間", "section2.xhtml"),
            _Link("外見を極める女の子", "section2.xhtml#a"),
        ])
        paths = _recover_flat_toc_paths(tree)
        self.assertEqual(
            paths[id(tree[2])],
            [
                "第一章 渋谷に誕生した盛り",
                "第一節 渋谷に生まれたコミュニティ",
                "茶髪になった同級生",
            ],
        )
        self.assertEqual(paths[id(tree[5])][-2:], ["第二節 雑誌が作った空間", "外見を極める女の子"])

    def test_flat_english_toc_uses_numbered_followers_to_find_chapters(self):
        tree = build_epub_toc_tree([
            _Link("II The Layers", "part.xhtml"),
            _Link("Earth Layer", "earth.xhtml"),
            _Link("16. Discovering Computation", "earth.xhtml#s16"),
            _Link("17. Digestion", "earth.xhtml#s17"),
            _Link("Cloud Layer", "cloud.xhtml"),
            _Link("24. Platform Geography", "cloud.xhtml#s24"),
        ])
        paths = _recover_flat_toc_paths(tree)
        self.assertEqual(paths[id(tree[1])], ["II The Layers", "Earth Layer"])
        self.assertEqual(
            paths[id(tree[2])],
            ["II The Layers", "Earth Layer", "16. Discovering Computation"],
        )
        self.assertEqual(
            paths[id(tree[5])],
            ["II The Layers", "Cloud Layer", "24. Platform Geography"],
        )


if __name__ == "__main__":
    unittest.main()
