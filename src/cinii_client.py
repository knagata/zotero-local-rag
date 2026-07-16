"""Small CiNii Research OpenSearch v2 client.

Official specification: https://support.nii.ac.jp/en/cir/r_opensearch
CiNii currently requires an application ID for v2; set ``CINII_APP_ID``.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from typing import Any


BASE_URL = "https://cir.nii.ac.jp/opensearch/v2/all"
_last_call = 0.0


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return _text(value.get("@value") or value.get("name") or value.get("value"))
    if isinstance(value, list):
        return "; ".join(filter(None, (_text(item) for item in value)))
    return ""


def _normalize_item(item: dict[str, Any]) -> dict[str, Any] | None:
    title = _text(item.get("title") or item.get("name") or item.get("dc:title"))
    if not title:
        return None
    identifier = _text(item.get("@id") or item.get("id") or item.get("identifier"))
    raw_identifiers = _text(item.get("identifier") or item.get("dc:identifier"))
    doi = next((token for token in raw_identifiers.replace(";", " ").split() if token.lower().startswith("10.")), None)
    date = _text(item.get("date") or item.get("publicationDate") or item.get("dc:date"))
    year = int(date[:4]) if len(date) >= 4 and date[:4].isdigit() else None
    return {
        "title": title,
        "authors": _text(item.get("creator") or item.get("author") or item.get("dc:creator")),
        "year": year,
        "doi": doi,
        "cinii_crid": identifier.rstrip("/").rsplit("/", 1)[-1] if identifier else None,
        "url": identifier or None,
        "source": "cinii",
    }


def search_cinii(
    title: str, *, author: str = "", year: int | None = None,
    count: int = 5, timeout: float = 15.0,
) -> list[dict[str, Any]]:
    """Search CiNii Research; return an empty list when no app ID is configured."""
    global _last_call
    app_id = os.environ.get("CINII_APP_ID", "").strip()
    if not app_id or not title.strip() or count <= 0:
        return []
    elapsed = time.monotonic() - _last_call
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)
    params: dict[str, Any] = {
        "appid": app_id, "format": "json", "title": title.strip(),
        "count": min(count, 20), "lang": "en", "sortorder": 1,
    }
    if author:
        params["creator"] = author
    if year:
        params["from"] = year
        params["until"] = year
    request = urllib.request.Request(
        f"{BASE_URL}?{urllib.parse.urlencode(params)}",
        headers={"Accept": "application/ld+json, application/json", "User-Agent": "ZoteroLocalRAG/1.0"},
    )
    _last_call = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    candidates: list[Any]
    if isinstance(payload, list):
        candidates = payload
    elif isinstance(payload, dict):
        candidates = payload.get("items") or payload.get("@graph") or payload.get("results") or []
    else:
        candidates = []
    return [normalized for item in candidates if isinstance(item, dict) and (normalized := _normalize_item(item))]


__all__ = ["search_cinii"]
