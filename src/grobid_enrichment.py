"""Optional, non-canonical GROBID enrichment for English scholarly PDFs."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import httpx

try:
    from .feature_gates import DEFAULT_GROBID_URL, grobid_enrichment_enabled
except ImportError:  # direct execution with src/ on sys.path
    from feature_gates import DEFAULT_GROBID_URL, grobid_enrichment_enabled

NS = {"tei": "http://www.tei-c.org/ns/1.0"}
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
SUPPORTED_ITEM_TYPES = {"journalArticle", "conferencePaper", "preprint"}


@dataclass(frozen=True)
class GrobidResult:
    references: list[dict[str, Any]]
    headings: list[str]
    title: str | None
    paragraph_count: int
    citation_marker_count: int
    linked_citation_count: int
    tei_xml: str


def should_enrich(*, item_type: str | None, language: str | None, source_type: str) -> bool:
    return (
        grobid_enrichment_enabled()
        and source_type == "pdf"
        and str(item_type or "") in SUPPORTED_ITEM_TYPES
        and str(language or "").casefold().split("-")[0] == "en"
    )


def _text(element: ET.Element | None) -> str:
    return " ".join("".join(element.itertext()).split()) if element is not None else ""


def _first(element: ET.Element, path: str) -> str | None:
    value = _text(element.find(path, NS)).strip()
    return value or None


def parse_tei(xml_text: str) -> GrobidResult:
    root = ET.fromstring(xml_text)
    bibliography = root.findall(".//tei:listBibl/tei:biblStruct", NS)
    contexts: dict[str, list[str]] = {}
    citation_markers = root.findall('.//tei:ref[@type="bibr"]', NS)
    for marker in citation_markers:
        target = str(marker.get("target") or "").lstrip("#")
        if not target:
            continue
        parent_text = _text(marker)
        contexts.setdefault(target, []).append(parent_text)
    references: list[dict[str, Any]] = []
    known_ids: set[str] = set()
    for ordinal, entry in enumerate(bibliography, 1):
        reference_id = str(entry.get(XML_ID) or "")
        if reference_id:
            known_ids.add(reference_id)
        raw = _first(entry, './/tei:note[@type="raw_reference"]') or _text(entry)
        authors = []
        for author in entry.findall(".//tei:author", NS):
            name = _text(author)
            if name:
                authors.append(name)
        date = entry.find(".//tei:date", NS)
        year_value = (date.get("when") if date is not None else None) or _text(date)
        year = None
        if year_value:
            digits = "".join(ch for ch in str(year_value) if ch.isdigit())
            year = int(digits[:4]) if len(digits) >= 4 else None
        doi = None
        for identifier in entry.findall(".//tei:idno", NS):
            if str(identifier.get("type") or "").casefold() == "doi":
                doi = _text(identifier) or None
                break
        references.append({
            "raw": raw, "title": _first(entry, ".//tei:title[@level='a']")
            or _first(entry, ".//tei:title"),
            "authors": authors, "year": year, "doi": doi,
            "container": _first(entry, ".//tei:title[@level='j']"),
            "type": "scholarly_reference", "lang": "en",
            "source_reference_id": ordinal,
            "source_context": json.dumps({
                "tei_id": reference_id, "markers": contexts.get(reference_id, []),
            }, ensure_ascii=False),
            "source_kind": "grobid_bibliography",
        })
    linked = sum(
        1 for marker in citation_markers
        if str(marker.get("target") or "").lstrip("#") in known_ids
    )
    headings = [_text(node) for node in root.findall(".//tei:body//tei:head", NS)]
    headings = [heading for heading in headings if heading]
    title = _first(root, ".//tei:titleStmt/tei:title")
    return GrobidResult(
        references=references, headings=headings, title=title,
        paragraph_count=len(root.findall(".//tei:body//tei:p", NS)),
        citation_marker_count=len(citation_markers), linked_citation_count=linked,
        tei_xml=xml_text,
    )


def process_pdf(pdf_path: Path, *, timeout: float | None = None) -> GrobidResult:
    base = (os.environ.get("GROBID_URL") or DEFAULT_GROBID_URL).rstrip("/")
    request_timeout = timeout or float(os.environ.get("GROBID_TIMEOUT_SEC", "120"))
    with pdf_path.open("rb") as handle:
        response = httpx.post(
            f"{base}/api/processFulltextDocument",
            files={"input": (pdf_path.name, handle, "application/pdf")},
            data={
                "consolidateHeader": "0", "consolidateCitations": "0",
                "includeRawCitations": "1",
            },
            timeout=request_timeout,
        )
    response.raise_for_status()
    result = parse_tei(response.text)
    if result.paragraph_count < 1:
        raise ValueError("GROBID TEI contains no body paragraphs")
    return result


__all__ = ["GrobidResult", "parse_tei", "process_pdf", "should_enrich"]
