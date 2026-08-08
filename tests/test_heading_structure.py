"""The structural heading vocabulary, in one place and in both languages.

heading_zone.py consolidated the *functional* vocabulary after three
extractors disagreed about it. The structural vocabulary was left split and
drifted the same way: chapter_detect knew "PART ONE" as well as "第一部",
while source_structure_refresh -- the module that rebuilds a flat document's
tree -- knew only the Japanese. All 84 flat PDFs awaiting recovery were
English, so none of them was ever examined.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import heading_structure as hs  # noqa: E402


class PartAndChapterTests(unittest.TestCase):
    def test_the_same_level_is_recognised_in_either_language(self):
        for japanese, english in (
            ("第一部 脱中心化", "PART ONE: Decentring"),
            ("第一章 物理的メディア", "CHAPTER 1: Physical Media"),
            ("第一節 基礎", "SECTION 2 Foundations"),
        ):
            with self.subTest(pair=(japanese, english)):
                jp_level = [
                    name for name, pattern in (
                        ("part", hs.PART_RE), ("chapter", hs.CHAPTER_RE),
                        ("section", hs.SECTION_RE),
                    ) if pattern.match(hs.normalize(japanese))
                ]
                en_level = [
                    name for name, pattern in (
                        ("part", hs.PART_RE), ("chapter", hs.CHAPTER_RE),
                        ("section", hs.SECTION_RE),
                    ) if pattern.match(hs.normalize(english))
                ]
                self.assertEqual(jp_level, en_level)

    def test_english_ordinals_are_accepted_as_words_and_digits(self):
        for heading in ("CHAPTER ONE", "CHAPTER 1", "Chapter I", "chapter twelve"):
            with self.subTest(heading=heading):
                self.assertTrue(hs.CHAPTER_RE.match(hs.normalize(heading)))

    def test_unnumbered_japanese_openers_are_still_chapters(self):
        for heading in ("序章", "終章", "はじめに", "おわりに", "結論"):
            with self.subTest(heading=heading):
                self.assertTrue(hs.CHAPTER_RE.match(hs.normalize(heading)))

    def test_a_bare_opening_marker_is_a_chapter_but_a_word_starting_with_it_is_not(self):
        # chapter_detect accepted a bare "序" as an opener; the consolidated
        # pattern dropped it, so a Japanese book opening on 序 lost its first
        # boundary. 序説/序文 must stay out: those are functional front matter.
        self.assertTrue(hs.CHAPTER_RE.match(hs.normalize("序 メディアの条件")))
        self.assertTrue(hs.CHAPTER_RE.match(hs.normalize("序")))
        self.assertFalse(hs.CHAPTER_RE.match(hs.normalize("序説の話")))

    def test_note_headings_are_left_to_the_functional_vocabulary(self):
        # chapter_detect's chapter pattern also matched 注/註. They are notes,
        # not chapters -- heading_zone places them in the endnote zone, and
        # admitting them here would open a chapter at every note section.
        from heading_zone import classify_heading_path

        for heading in ("注", "註"):
            with self.subTest(heading=heading):
                self.assertFalse(hs.CHAPTER_RE.match(hs.normalize(heading)))
                self.assertIn(classify_heading_path([heading]), {"endnote", "footnote"})

    def test_a_part_is_not_also_a_chapter(self):
        self.assertTrue(hs.PART_RE.match(hs.normalize("PART III")))
        self.assertFalse(hs.CHAPTER_RE.match(hs.normalize("PART III")))


class BareRomanNumeralTests(unittest.TestCase):
    """A Roman numeral with no PART/CHAPTER word in front of it to vouch for it."""

    #: Every one of these is an ordinary English word spelled entirely from
    #: IVXLCDM. /usr/share/dict/words holds 37 of them; none is spelled from
    #: I, V and X alone, which is why the bare patterns stop at those three.
    PROSE = ("DID THE COMMITTEE ever meet again?", "MIX THE FLOUR and water.",
             "CIVIL THE tone remained.", "LIVID THE crowd became.",
             "MILD weather returned.", "Did you have to work at Cootamundra?")

    def test_prose_spelled_from_roman_letters_is_not_a_part(self):
        # source_structure_refresh admits a PART match from any block type, so
        # one such sentence became a top-level part with the chapters after it
        # nested underneath; two of them made the part numbering non-contiguous
        # and the whole document was rejected.
        for text in self.PROSE:
            with self.subTest(text=text):
                self.assertFalse(hs.PART_RE.match(hs.normalize(text)))
                self.assertFalse(hs.ROMAN_SUBHEADING_RE.match(hs.normalize(text)))
                self.assertIsNone(hs.ordinal(text))

    def test_a_real_bare_numeral_still_opens_a_part(self):
        for text, expected in (("I THE BEGINNING", 1), ("IV THE END", 4),
                               ("XII THE LAST", 12)):
            with self.subTest(text=text):
                self.assertTrue(hs.PART_RE.match(hs.normalize(text)))
                self.assertEqual(hs.ordinal(text), expected)

    def test_a_roman_subheading_is_still_recognised(self):
        for text in ("III 展開", "Ⅳ 展開", "VII", "XXIV Method"):
            with self.subTest(text=text):
                self.assertTrue(hs.ROMAN_SUBHEADING_RE.match(hs.normalize(text)))


class NormalizeTests(unittest.TestCase):
    def test_layout_decoration_is_stripped_from_the_front(self):
        # Headings arrive from OCR and layout extraction as "■ ■ ■ ■ ■ CHAPTER
        # ONE"; anchored patterns match nothing until the run is removed.
        self.assertEqual(hs.normalize("■ ■ ■ ■ ■ CHAPTER ONE"), "CHAPTER ONE")
        self.assertTrue(hs.CHAPTER_RE.match(hs.normalize("■ ■ ■ ■ ■ CHAPTER ONE")))

    def test_full_width_forms_fold_to_their_ascii_twins(self):
        self.assertTrue(hs.CHAPTER_RE.match(hs.normalize("第１章 序説")))

    def test_leading_japanese_text_is_not_stripped(self):
        self.assertEqual(hs.normalize("第一章 物理的メディア"), "第一章 物理的メディア")

    def test_markdown_hashes_and_runs_of_space_collapse(self):
        self.assertEqual(hs.normalize("##  PART   TWO "), "PART TWO")


class OrdinalTests(unittest.TestCase):
    def test_numbers_are_read_from_every_written_form(self):
        for heading, expected in (
            ("CHAPTER ONE", 1), ("CHAPTER 12", 12), ("Chapter IV", 4),
            ("第三章", 3), ("第十二章", 12), ("PART IV", 4),
            ("■ ■ CHAPTER SIX", 6),
        ):
            with self.subTest(heading=heading):
                self.assertEqual(hs.ordinal(heading), expected)

    def test_prose_is_not_read_as_a_roman_numeral(self):
        # The leading letter of an ordinary word is a valid Roman numeral:
        # "Introduction" scored 1, "CONTENTS" 100 and "Method" 1000, which
        # would make a contiguity check pass or fail on prose.
        for heading in ("序章", "Introduction", "NOTES", "CONTENTS", "Method",
                        "Ical Practice", "Ди"):
            with self.subTest(heading=heading):
                self.assertIsNone(hs.ordinal(heading))

    def test_a_gap_in_a_run_is_detectable(self):
        # An extractor that drops "CHAPTER TWO" leaves 1, 3, 4, 6. Storing that
        # would assert a six-chapter book has four chapters.
        found = [hs.ordinal(h) for h in
                 ("CHAPTER ONE", "CHAPTER THREE", "CHAPTER FOUR", "CHAPTER SIX")]
        self.assertEqual(found, [1, 3, 4, 6])
        self.assertNotEqual(found, list(range(1, len(found) + 1)))


class IndexGroupTests(unittest.TestCase):
    def test_a_kana_row_marker_is_recognised(self):
        for heading in ("ア行", "か行", "サ 行"):
            with self.subTest(heading=heading):
                self.assertTrue(hs.INDEX_GROUP_RE.match(hs.normalize(heading)))

    def test_an_all_kanji_word_ending_in_the_same_character_is_not(self):
        # 五行 and 銀行 are ordinary prose; treating one as an index marker
        # moved a body section into the excluded index zone.
        for heading in ("五行", "銀行", "旅行"):
            with self.subTest(heading=heading):
                self.assertFalse(hs.INDEX_GROUP_RE.match(hs.normalize(heading)))


class CoversTheVocabularyItReplacesTests(unittest.TestCase):
    """Nothing the split definitions matched may stop matching here."""

    SAMPLES = (
        "第一部 脱中心化", "第一章 物理的メディア", "第一節 基礎", "序章", "終章",
        "はじめに", "おわりに", "結論", "PART ONE", "PART III", "CHAPTER 12",
        "III 展開", "ア行", "3. Method", "一 序説",
    )

    #: The Japanese-only definitions source_structure_refresh carried before the
    #: vocabulary was consolidated, kept verbatim so their coverage stays pinned
    #: after the originals were deleted.
    RETIRED_JAPANESE = (
        ("part", r"^第[一二三四五六七八九十百]+部"),
        ("chapter", r"^(?:第[一二三四五六七八九十百]+章|序章|結論)(?:\s|$)"),
        ("section", r"^第[一二三四五六七八九十百]+節"),
        ("roman", r"^(?:[IVXLCDM]+|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+)(?:\s|$)"),
        ("index", r"^[ァ-ヶぁ-ん]+\s*行$"),
    )

    def test_every_pattern_the_japanese_module_had_is_covered(self):
        import re

        patterns = {
            "part": hs.PART_RE, "chapter": hs.CHAPTER_RE, "section": hs.SECTION_RE,
            "roman": hs.ROMAN_SUBHEADING_RE, "index": hs.INDEX_GROUP_RE,
        }
        for label, source in self.RETIRED_JAPANESE:
            old, new = re.compile(source, re.IGNORECASE), patterns[label]
            for sample in self.SAMPLES:
                if old.match(sample):
                    with self.subTest(pattern=label, sample=sample):
                        self.assertTrue(new.match(hs.normalize(sample)))

    def test_every_pattern_the_bilingual_module_had_is_covered(self):
        import chapter_detect as cd

        for label, old, new in (
            ("part", cd._PART_RE, hs.PART_RE),
            ("chapter", cd._CHAPTER_RE, hs.CHAPTER_RE),
            ("section", cd._EXPLICIT_SECTION_RE, hs.SECTION_RE),
        ):
            for sample in self.SAMPLES:
                if old.match(sample):
                    with self.subTest(pattern=label, sample=sample):
                        self.assertTrue(new.match(hs.normalize(sample)))


if __name__ == "__main__":
    unittest.main()
