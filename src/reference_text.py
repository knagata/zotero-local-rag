"""Conservative classification helpers for extracted bibliography text."""
from __future__ import annotations

import re
import unicodedata
from typing import Any


SHORT_FORM_REFERENCE_RE = re.compile(
    r"\b(?:ibid|op\.?\s*cit|loc\.?\s*cit)\b|同書|同前|前掲",
    re.IGNORECASE,
)


def is_short_form_reference(value: str | None) -> bool:
    """Return True when a reference depends on a preceding bibliographic entry."""
    return bool(SHORT_FORM_REFERENCE_RE.search(value or ""))


YEAR_RE = re.compile(r"(?<!\d)(?:18|19|20)\d{2}(?!\d)")
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:a-z0-9]+", re.IGNORECASE)
REVIEW_RE = re.compile(r"\b(?:book\s+review|review\s+of)\b", re.IGNORECASE)


def normalize_reference_text(value: str | None) -> str:
    """Normalize bibliographic text while retaining Unicode letters and words."""
    text = unicodedata.normalize("NFKD", value or "").casefold()
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(re.findall(r"\w+", text, flags=re.UNICODE))


def _author_surnames(authors: Any) -> list[str]:
    if isinstance(authors, str):
        names = authors.split(",")
    else:
        names = [author.get("name", "") for author in (authors or [])]
    surnames: list[str] = []
    for name in names:
        tokens = normalize_reference_text(name).split()
        if tokens and len(tokens[-1]) >= 3:
            surnames.append(tokens[-1])
    return surnames


def s2_candidate_is_supported(raw_reference: str | None, candidate: dict[str, Any]) -> bool:
    """Accept an S2 hit only when the raw citation contains decisive evidence.

    A DOI match is definitive. Otherwise the complete normalized S2 title must
    occur in the citation, any stated year must agree, and an available S2
    author surname must occur. For legacy results without author metadata, an
    exact title plus matching year remains acceptable.
    """
    raw = raw_reference or ""
    if not raw or is_short_form_reference(raw):
        return False

    external_ids = candidate.get("externalIds") or {}
    candidate_doi = normalize_reference_text(external_ids.get("DOI"))
    raw_dois = {normalize_reference_text(value) for value in DOI_RE.findall(raw)}
    if candidate_doi and candidate_doi in raw_dois:
        return True

    title = candidate.get("title") or ""
    normalized_title = normalize_reference_text(title)
    normalized_raw = normalize_reference_text(raw)
    # Very short/generic titles are not safe identifiers by themselves.
    if len(normalized_title.replace(" ", "")) < 8 or normalized_title not in normalized_raw:
        return False
    if REVIEW_RE.search(title) and not REVIEW_RE.search(raw):
        return False

    raw_years = set(YEAR_RE.findall(raw))
    candidate_year = candidate.get("year")
    if raw_years and (candidate_year is None or str(candidate_year) not in raw_years):
        return False

    surnames = _author_surnames(candidate.get("authors"))
    if surnames:
        raw_tokens = set(normalized_raw.split())
        if not any(surname in raw_tokens for surname in surnames):
            return False
        return True

    # Older rows were fetched without authors. Do not accept them unless both
    # exact title and matching year are present in the source text.
    return bool(raw_years and candidate_year is not None)
