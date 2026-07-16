"""National Diet Library Search SRU client.

Official specification: https://ndlsearch.ndl.go.jp/help/api/specifications
"""
from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any


BASE_URL = "https://ndlsearch.ndl.go.jp/api/sru"
_last_call = 0.0


def _escape_cql(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _values(element: ET.Element, local_name: str) -> list[str]:
    return [
        (node.text or "").strip()
        for node in element.iter()
        if node.tag.rsplit("}", 1)[-1] == local_name and (node.text or "").strip()
    ]


def _normalize_record(record: ET.Element) -> dict[str, Any] | None:
    titles = _values(record, "title")
    if not titles:
        return None
    identifiers = _values(record, "identifier")
    joined = " ".join(identifiers)
    doi_match = re.search(r"(?:doi:\s*|https?://doi\.org/)?(10\.\d{4,9}/\S+)", joined, re.I)
    isbn_match = re.search(r"(?:97[89])?\d{9}[\dXx]", re.sub(r"[-\s]", "", joined))
    dates = _values(record, "date") + _values(record, "issued")
    year_match = re.search(r"(?:18|19|20)\d{2}", " ".join(dates))
    ndl_identifier = next((value for value in identifiers if "ndl" in value.casefold()), identifiers[0] if identifiers else "")
    alternatives = _values(record, "alternative")
    return {
        "title": titles[0], "alternative_titles": alternatives,
        "authors": "; ".join(_values(record, "creator")),
        "year": int(year_match.group()) if year_match else None,
        "publisher": "; ".join(_values(record, "publisher")),
        "doi": doi_match.group(1).rstrip(".,;)") if doi_match else None,
        "isbn": isbn_match.group() if isbn_match else None,
        "ndl_bibid": ndl_identifier.rstrip("/").rsplit("/", 1)[-1] if ndl_identifier else None,
        "identifiers": identifiers, "source": "ndl",
    }


def search_ndl(
    title: str, *, author: str = "", year: int | None = None,
    maximum_records: int = 5, timeout: float = 15.0,
) -> list[dict[str, Any]]:
    """Search NDL Search through SRU 1.2 and normalize DC-NDL records."""
    global _last_call
    if not title.strip() or maximum_records <= 0:
        return []
    clauses = [f'title="{_escape_cql(title.strip())}"']
    if author:
        clauses.append(f'creator="{_escape_cql(author.strip())}"')
    if year:
        clauses.append(f'from="{year}" AND until="{year}"')
    params = {
        "operation": "searchRetrieve", "version": "1.2",
        "recordSchema": "dcndl", "maximumRecords": min(maximum_records, 20),
        "query": " AND ".join(clauses),
    }
    elapsed = time.monotonic() - _last_call
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)
    request = urllib.request.Request(
        f"{BASE_URL}?{urllib.parse.urlencode(params)}",
        headers={"Accept": "application/xml", "User-Agent": "ZoteroLocalRAG/1.0"},
    )
    _last_call = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        root = ET.fromstring(response.read())
    records = [node for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "recordData"]
    return [normalized for record in records if (normalized := _normalize_record(record))]


__all__ = ["search_ndl"]
