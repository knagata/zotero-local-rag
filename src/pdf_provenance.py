# src/pdf_provenance.py
"""Classify a PDF's text-layer provenance, and detect extraction defects.

Two independent concerns, both established empirically in
``dev-notes/current/79_embedding_gates.md``:

**1. Source class (``classify_pdf_source``).** A scanned PDF can carry a text
layer -- OCR puts one there -- so "does it have text?" does not tell us what
kind of document it is. What matters is *how much the text layer can be
trusted*: a born-digital PDF's text layer **is** the typeset result, while a
scan's is a lossy derivative of the page image. The same observation ("this
page has no text") therefore means *figure* in the first case and *OCR
failure* in the second, which is why the class has to be decided before any
page-level repair runs.

The discriminator is whether pure-text pages exist at all. A scan images its
front matter and colophon along with everything else, whereas a born-digital
document -- even a plate-heavy art book -- always typesets its preface and
imprint. Measured over 80 documents from the library, scans had 0-2 pure-text
pages out of a 19-page sample while born-digital documents had 4-18, with no
overlap. The sample must be **front-loaded**: front matter only exists at the
start, so an evenly spread sample misses it and misfiles art books as scans.

**2. Text defects (``detect_text_defects``).** Deterministic detectors for the
two extraction artefacts that the LLM sampling audit (stage 2) kept surfacing
but which need no model at all. Both were measured to separate cleanly:

- *letter spacing* (``p r o b l e m s``): a PyMuPDF text-layer artefact.
  Latin-dominant documents in the library sit at 0.6-3.1%; the two affected
  ones at 7.4% and 38.3%.
- *dropped ligatures* (``sofware``, ``afer``): the ft/fi/fl glyph was dropped
  outright rather than mapped, so -- unlike a *preserved* ligature codepoint
  (U+FB00-) which ``clean_extracted_text``'s NFKC pass already decomposes --
  no normalisation can recover it. Detected via a closed list of the words
  this produces, which essentially never occur legitimately: 226 hits in the
  one affected document, 0-6 elsewhere.

Neither defect is an OCR problem, so the remedy is re-extraction, not re-OCR.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

# --- source classification ------------------------------------------------

# A single image covering this share of the page area means the page *is* an
# image rather than a page carrying one.
FULL_PAGE_IMAGE_AREA_RATIO = 0.6
# Pages with at least this much text count as carrying real text.
PURE_TEXT_MIN_CHARS = 200
# Below this share of pure-text pages the document is a scan.  Measured gap:
# scans 0-10.5%, born-digital 21-95%.
PURE_TEXT_PAGE_RATIO_MAX = 0.15
# Above this share of full-page-image pages the document looks imaged.
FULL_PAGE_IMAGE_RATIO_MIN = 0.6
# A scan whose pages mostly carry text has an OCR layer.
OCR_LAYER_TEXT_PAGE_RATIO_MIN = 0.5

BORN_DIGITAL = "born_digital"
SCANNED_OCR_LAYER = "scanned_ocr_layer"
SCANNED_NO_TEXT = "scanned_no_text"

#: Source classes whose text layer is a derivative of the page image and must
#: therefore be validated before it is trusted (stage 2).
SCAN_DERIVED_CLASSES = frozenset({SCANNED_OCR_LAYER, SCANNED_NO_TEXT})


@dataclass
class SourceClass:
    """Result of ``classify_pdf_source``."""

    kind: str
    full_page_image_ratio: float
    pure_text_page_ratio: float
    text_page_ratio: float
    sampled_pages: List[int] = field(default_factory=list)
    producer: str = ""

    @property
    def text_layer_is_derived(self) -> bool:
        """True when the text layer came from OCR rather than typesetting."""
        return self.kind in SCAN_DERIVED_CLASSES

    def as_metadata(self) -> Dict[str, Any]:
        return {
            "source_class": self.kind,
            "source_class_full_page_image_ratio": round(self.full_page_image_ratio, 3),
            "source_class_pure_text_page_ratio": round(self.pure_text_page_ratio, 3),
            "source_class_text_page_ratio": round(self.text_page_ratio, 3),
            "source_class_sampled_pages": list(self.sampled_pages),
            "pdf_producer": self.producer,
        }


def _sample_page_indices(page_count: int, front: int = 12, spread: int = 8) -> List[int]:
    """Front-loaded sample: the first ``front`` pages plus ``spread`` across.

    Front matter is the whole point of the pure-text test and only exists at
    the start, so it must be sampled densely; the spread pages keep the
    full-page-image ratio representative of the body.
    """
    if page_count <= 0:
        return []
    indices = set(range(min(front, page_count)))
    if page_count > 1:
        indices.update(
            int(i * (page_count - 1) / max(1, spread - 1)) for i in range(spread)
        )
    return sorted(i for i in indices if 0 <= i < page_count)


def _page_is_full_image(page: Any) -> bool:
    page_area = abs(page.rect.get_area()) or 1.0
    try:
        images = page.get_images(full=True)
    except Exception:
        return False
    for image in images:
        try:
            rects = page.get_image_rects(image[0]) or []
        except Exception:
            continue
        for rect in rects:
            if abs(rect.get_area()) / page_area > FULL_PAGE_IMAGE_AREA_RATIO:
                return True
    return False


def classify_pdf_source(pdf_path: Path) -> SourceClass:
    """Decide whether a PDF is born-digital or a scan (with or without OCR)."""
    import fitz

    with fitz.open(str(pdf_path)) as document:
        page_count = document.page_count
        producer = str((document.metadata or {}).get("producer") or "")
        indices = _sample_page_indices(page_count)
        full_image = pure_text = with_text = 0
        for index in indices:
            page = document[index]
            is_image = _page_is_full_image(page)
            try:
                text = (page.get_text("text") or "").strip()
            except Exception:
                text = ""
            has_text = len(text) >= PURE_TEXT_MIN_CHARS
            if is_image:
                full_image += 1
            elif has_text:
                pure_text += 1
            if has_text:
                with_text += 1

    sampled = max(1, len(indices))
    image_ratio = full_image / sampled
    pure_ratio = pure_text / sampled
    text_ratio = with_text / sampled

    if image_ratio >= FULL_PAGE_IMAGE_RATIO_MIN and pure_ratio <= PURE_TEXT_PAGE_RATIO_MAX:
        kind = (
            SCANNED_OCR_LAYER if text_ratio >= OCR_LAYER_TEXT_PAGE_RATIO_MIN
            else SCANNED_NO_TEXT
        )
    else:
        kind = BORN_DIGITAL

    return SourceClass(
        kind=kind, full_page_image_ratio=image_ratio, pure_text_page_ratio=pure_ratio,
        text_page_ratio=text_ratio, sampled_pages=[i + 1 for i in indices],
        producer=producer,
    )


# --- deterministic text-defect detection ----------------------------------

_LATIN_WORD = re.compile(r"[A-Za-z]+")
_LATIN_CHAR = re.compile(r"[A-Za-z]")
_NON_SPACE = re.compile(r"\S")

# Single letters that are legitimate standalone words and must not count as
# letter-spacing evidence.
_STANDALONE_LETTERS = frozenset({"a", "i"})

# Words produced by dropping an ft/fi/fl/ff ligature outright.  Deliberately a
# closed list rather than a lexicon check: these exact strings essentially
# never occur in real text, so precision is high with no dictionary to ship.
# Short or ambiguous fragments (e.g. "dra") are excluded -- they produced false
# positives during calibration.
_DROPPED_LIGATURE_WORDS = (
    "afer", "ofen", "sofware", "thef", "frst", "fnd", "fnal", "feld", "ofce",
    "beneft", "diferent", "diference", "efect", "efort", "sufcient", "specifc",
    "signifcant", "confrm", "defne", "identifed", "classifed", "fgure",
    "profle", "artifcial", "confict", "infuence", "fexible", "briefy",
    "chiefy", "diffcult", "efciency", "suffcient", "justifed", "modifed",
    "scientifc", "certifcate", "notifed", "satisfed", "unifed",
)
_DROPPED_LIGATURE = re.compile(
    r"\b(?:" + "|".join(_DROPPED_LIGATURE_WORDS) + r")\b", re.IGNORECASE
)


def letter_spacing_ratio_max() -> float:
    return float(os.environ.get("PDF_LETTER_SPACING_RATIO_MAX", "0.05"))


def dropped_ligature_ratio_max() -> float:
    return float(os.environ.get("PDF_DROPPED_LIGATURE_RATIO_MAX", "0.0005"))


def _latin_dominant(text: str, latin_words: List[str]) -> bool:
    """Whether the letter-spacing test applies to this text at all.

    In a CJK document the few Latin tokens are mostly initials and
    abbreviations, so the isolated-single-letter ratio is meaningless there --
    two Japanese documents measured 38.8% and 84.5% off 242 and 207 tokens.
    Requiring both a real token count and Latin dominance excludes them.
    """
    if len(latin_words) < 2000:
        return False
    non_space = len(_NON_SPACE.findall(text)) or 1
    return len(_LATIN_CHAR.findall(text)) / non_space > 0.5


def detect_text_defects(text: str) -> Dict[str, Any]:
    """Report deterministic extraction defects found in ``text``.

    Returns measured ratios plus a ``defects`` list naming whatever crossed its
    threshold.  A defect here means the *extraction* is wrong, not the OCR, so
    the caller should re-extract rather than re-OCR.
    """
    latin_words = _LATIN_WORD.findall(text)
    applicable = _latin_dominant(text, latin_words)

    isolated = sum(
        1 for word in latin_words
        if len(word) == 1 and word.lower() not in _STANDALONE_LETTERS
    )
    spacing_ratio = (isolated / len(latin_words)) if latin_words else 0.0
    dropped = len(_DROPPED_LIGATURE.findall(text))
    dropped_ratio = (dropped / len(latin_words)) if latin_words else 0.0

    defects: List[str] = []
    if applicable and spacing_ratio > letter_spacing_ratio_max():
        defects.append("letter_spacing")
    if applicable and dropped_ratio > dropped_ligature_ratio_max():
        defects.append("dropped_ligature")

    return {
        "text_defects": defects,
        "letter_spacing_ratio": round(spacing_ratio, 4),
        "dropped_ligature_count": dropped,
        "dropped_ligature_ratio": round(dropped_ratio, 5),
        "text_defect_scope_applicable": applicable,
        "latin_word_count": len(latin_words),
    }


__all__ = [
    "BORN_DIGITAL", "SCANNED_NO_TEXT", "SCANNED_OCR_LAYER", "SCAN_DERIVED_CLASSES",
    "SourceClass", "classify_pdf_source", "detect_text_defects",
    "dropped_ligature_ratio_max", "letter_spacing_ratio_max",
]
