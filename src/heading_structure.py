# src/heading_structure.py
"""Where a heading sits in a document's hierarchy. The single definition.

``heading_zone.py`` answers what a heading *means* -- notes, bibliography,
index -- and says in its own docstring why that vocabulary had to be
consolidated: three extractors each carried a copy and they disagreed about
coverage. The structural vocabulary was never given the same treatment and
drifted in exactly the same way. ``chapter_detect`` recognised "PART ONE"
alongside "第一部", while ``source_structure_refresh`` -- the module that
actually rebuilds a flat document's tree -- knew only the Japanese forms, so
an English book's body headings were never even examined. Of 84 flat PDFs
awaiting recovery, every one was English.

A scholarly book has the same shape in any language: parts contain chapters,
chapters contain sections, and both are followed by their notes. Only the
words differ, so the words live here as data and the code that consumes them
stays language-neutral. Adding a language should mean extending a pattern,
never adding a branch.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

#: Ordinals spelled as words, which English chapter headings use as often as
#: digits ("CHAPTER ONE" is not rarer than "CHAPTER 1").
_ORDINAL_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}
_KANJI_DIGITS = {
    "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9,
}
_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}

#: Any ordinal form a part or chapter number can take, in either language.
_ORDINAL = (
    r"[0-9]+|[IVXLCDM]+|[一二三四五六七八九十百〇]+|"
    + "|".join(_ORDINAL_WORDS)
)

#: Decorative glyphs that layout and OCR put in front of a heading. Measured on
#: this library, headings arrive as "■ ■ ■ ■ ■ CHAPTER ONE" often enough that a
#: pattern anchored at the start matches nothing without stripping them first.
_LEADING_DECORATION_RE = re.compile(r"^[^\w぀-ヿ㐀-鿿]+")


def normalize(text: str) -> str:
    """Fold a heading to the form the patterns below expect.

    NFKC so full-width digits and Roman numerals compare with their ASCII
    twins, then the decorative run stripped from the front.
    """
    folded = unicodedata.normalize("NFKC", str(text or ""))
    folded = " ".join(folded.replace("#", " ").split())
    return _LEADING_DECORATION_RE.sub("", folded).strip()


#: A Roman numeral standing on its own, with no PART/CHAPTER word in front of
#: it to vouch for it. Deliberately narrower than _ORDINAL: 37 English words are
#: spelled entirely from IVXLCDM -- did, dim, mix, mild, civil, livid, mimic --
#: so a bare [IVXLCDM]+ reads ordinary prose as a part number ("DID THE
#: COMMITTEE..." scored 999). No English word is spelled from I, V and X alone,
#: and I..XXXIX is further than any book's parts or subheadings run.
_BARE_ROMAN = r"[IVX]{1,6}"

PART_RE = re.compile(
    rf"^(?:第\s*(?:{_ORDINAL})\s*[部編]|PART\s+(?:{_ORDINAL})\b|"
    rf"{_BARE_ROMAN}\s+THE\s+)",
    re.IGNORECASE,
)
CHAPTER_RE = re.compile(
    rf"^(?:第?\s*(?:{_ORDINAL})\s*(?:章|講)|CHAPTER\s+(?:{_ORDINAL})\b|"
    r"序章|終章|序論|結論|はじめに|おわりに|補考(?:\s|$)|序(?:\s|$))",
    re.IGNORECASE,
)
SECTION_RE = re.compile(
    rf"^(?:第?\s*(?:{_ORDINAL})\s*節(?:\s|$)|SECTION\s+(?:{_ORDINAL})\b)",
    re.IGNORECASE,
)
#: A bare numeric prefix ("3. Method", "2.1 Results"). Weaker evidence than the
#: named forms above: it also matches list items and OCR record numbers, so
#: callers pair it with a heading block type rather than trusting it alone.
NUMBERED_RE = re.compile(
    r"^(?:[0-9]{1,2}(?:\.[0-9]{1,2}){0,3}[.)．。:]?|[一二三四五六七八九十百]+)(?:\s|　)",
)
#: A Roman numeral standing alone as a subheading ("III", "Ⅳ 展開"). Restricted
#: to _BARE_ROMAN for the same reason PART_RE is: nothing precedes it to confirm
#: it is a numeral, so "DID THE ..." and "MILD ..." would both open one.
ROMAN_SUBHEADING_RE = re.compile(
    rf"^(?:{_BARE_ROMAN}|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+)(?:\s|$)", re.IGNORECASE,
)
#: A kana group heading inside a Japanese index ("ア行"). Deliberately kana
#: only: an all-kanji word ending in 行 (銀行, 五行) is ordinary prose, and
#: treating one as an index marker moved a body section into the excluded
#: index zone.
INDEX_GROUP_RE = re.compile(r"^[ァ-ヶぁ-ん]+\s*行$")

_ORDINAL_AFTER_RE = re.compile(
    rf"^(?:第\s*|PART\s+|CHAPTER\s+|SECTION\s+)?({_ORDINAL})", re.IGNORECASE,
)


def _roman_to_int(token: str) -> Optional[int]:
    total = previous = 0
    for char in reversed(token.lower()):
        value = _ROMAN_VALUES.get(char)
        if value is None:
            return None
        total = total - value if value < previous else total + value
        previous = max(previous, value)
    return total or None


def _kanji_to_int(token: str) -> Optional[int]:
    if not token or any(c not in _KANJI_DIGITS and c not in "十百" for c in token):
        return None
    total, current = 0, 0
    for char in token:
        if char in _KANJI_DIGITS:
            current = _KANJI_DIGITS[char]
        elif char == "十":
            total += (current or 1) * 10
            current = 0
        elif char == "百":
            total += (current or 1) * 100
            current = 0
    return (total + current) or None


def ordinal(text: str) -> Optional[int]:
    """The number a part, chapter or section heading carries, or None.

    Callers use this to check that a recovered run of chapters is contiguous.
    An extractor that drops "CHAPTER TWO" leaves a book looking like it goes
    1, 3, 4, 6 -- storing that as the tree would assert a six-chapter book has
    four chapters, which is worse than leaving it flat.

    The heading has to be structural first. Reading a number out of any text
    at all treats the leading letter of an ordinary word as a Roman numeral --
    "Introduction" scored 1, "CONTENTS" 100 and "Method" 1000 -- which would
    then make a contiguity check pass or fail on prose.
    """
    folded = normalize(text)
    if not (PART_RE.match(folded) or CHAPTER_RE.match(folded) or SECTION_RE.match(folded)):
        return None
    match = _ORDINAL_AFTER_RE.match(folded)
    if not match:
        return None
    token = match.group(1)
    if token.isdigit():
        return int(token)
    word = _ORDINAL_WORDS.get(token.lower())
    if word:
        return word
    return _kanji_to_int(token) or _roman_to_int(token)


__all__ = [
    "CHAPTER_RE", "INDEX_GROUP_RE", "NUMBERED_RE", "PART_RE",
    "ROMAN_SUBHEADING_RE", "SECTION_RE", "normalize", "ordinal",
]
