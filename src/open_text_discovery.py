"""Read-only discovery adapters for externally hosted textual resources."""
from __future__ import annotations

import asyncio
import os
import re
import time
from typing import Any
from urllib.parse import urlparse
from xml.etree import ElementTree

import httpx

SUPPORTED_SOURCES = ("ndl", "cinii", "jstage", "openlibrary")
_NON_TEXT_TYPES = re.compile(r"(?:写真|絵画|地図|音声|映像|動画|music|sound|image|map)", re.I)
_HEADERS = {"User-Agent": "zotero-local-rag/1.0 (personal research discovery)"}
_SEARCH_MODES = frozenset({"auto", "all", "any", "phrase"})
_CACHE_TTL_SECONDS = 300.0
_CACHE: dict[tuple[str, str, int], tuple[float, list[dict[str, Any]]]] = {}
_SOURCE_LOCKS = {source: asyncio.Lock() for source in SUPPORTED_SOURCES}


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].casefold()


def _texts(element: ElementTree.Element, *names: str) -> list[str]:
    wanted = {name.casefold() for name in names}
    values: list[str] = []
    for child in element.iter():
        if _local_name(child.tag) in wanted and child.text and child.text.strip():
            value = child.text.strip()
            if value not in values:
                values.append(value)
    return values


def _first(element: ElementTree.Element, *names: str) -> str:
    values = _texts(element, *names)
    return values[0] if values else ""


def _identifier(url: str) -> str:
    return urlparse(url).path.rstrip("/").rsplit("/", 1)[-1]


def _jstage_identifier(url: str) -> str:
    path = urlparse(url).path
    article_path = path.split("/_article", 1)[0].strip("/")
    return article_path.replace("/", ":") if article_path else _identifier(url)


def _candidate(**values: Any) -> dict[str, Any]:
    result = {
        "source": "", "record_id": "", "title": "", "creators": [],
        "date": "", "resource_type": "text", "container_title": "",
        "publisher": "", "volume": "", "issue": "", "pages": "",
        "identifiers": {}, "landing_url": "",
        "fulltext_status": "unknown", "download_urls": [],
        "textual_resource": True,
    }
    result.update(values)
    return result


def _page_range(start: Any, end: Any, explicit: Any = "") -> str:
    page_range = str(explicit or "").strip()
    if page_range:
        return page_range
    first = str(start or "").strip()
    last = str(end or "").strip()
    if first and last and first != last:
        return f"{first}-{last}"
    return first or last


def _cinii_identifiers(item: dict[str, Any], record_id: str) -> dict[str, Any]:
    identifiers: dict[str, Any] = {"crid": record_id}
    for value in item.get("dc:identifier") or []:
        if not isinstance(value, dict) or not value.get("@value"):
            continue
        kind = str(value.get("@type") or "identifier").rsplit(":", 1)[-1].casefold()
        existing = identifiers.get(kind)
        if existing is None:
            identifiers[kind] = str(value["@value"])
        elif isinstance(existing, list):
            existing.append(str(value["@value"]))
        else:
            identifiers[kind] = [existing, str(value["@value"])]
    return identifiers


def _jstage_pdf_url(article_url: str) -> str:
    return f"{article_url.split('/_article', 1)[0]}/_pdf/-char/ja" if "/_article" in article_url else ""


def parse_ndl(text: str) -> list[dict[str, Any]]:
    root = ElementTree.fromstring(text)
    rows: list[dict[str, Any]] = []
    for item in (node for node in root.iter() if _local_name(node.tag) == "item"):
        categories = _texts(item, "category", "type")
        if any(_NON_TEXT_TYPES.search(value) for value in categories):
            continue
        url = _first(item, "link", "guid")
        rows.append(_candidate(
            source="ndl", record_id=_identifier(url), title=_first(item, "title"),
            creators=_texts(item, "creator", "author"), date=_first(item, "issued", "date"),
            resource_type=categories[0] if categories else "text",
            publisher=_first(item, "publisher"),
            volume=_first(item, "volume"), issue=_first(item, "number", "issue"),
            pages=_page_range(
                _first(item, "startingpage"), _first(item, "endingpage"),
                _first(item, "pagerange"),
            ),
            landing_url=url,
            identifiers={"ndl_url": url},
            fulltext_status="unknown",
        ))
    return [row for row in rows if row["title"]]


def parse_jstage(text: str) -> list[dict[str, Any]]:
    root = ElementTree.fromstring(text)
    rows: list[dict[str, Any]] = []
    for entry in (node for node in root.iter() if _local_name(node.tag) == "entry"):
        link_node = next((node for node in entry if _local_name(node.tag) == "article_link"), None)
        links = _texts(link_node, "ja", "en") if link_node is not None else []
        url = next((value for value in links if "/article/" in value), links[0] if links else "")
        issn = _first(entry, "issn")
        eissn = _first(entry, "eissn")
        identifiers = {key: value for key, value in (
            ("doi", _first(entry, "doi")), ("issn", issn), ("eissn", eissn),
        ) if value}
        pdf_url = _jstage_pdf_url(url)
        rows.append(_candidate(
            source="jstage", record_id=_jstage_identifier(url),
            title=_first(entry, "ja", "en") if _first(entry, "article_title") == "" else _first(entry, "article_title"),
            creators=_texts(entry, "name"),
            date=_first(entry, "pubyear", "publicationdate", "date", "year"),
            resource_type="journalArticle", container_title=_first(entry, "material_title"),
            volume=_first(entry, "volume"), issue=_first(entry, "number"),
            pages=_page_range(
                _first(entry, "startingpage"), _first(entry, "endingpage"),
            ),
            identifiers=identifiers, landing_url=url,
            fulltext_status="pdf_link_available_unverified" if pdf_url else "landing_page_available",
            download_urls=[pdf_url] if pdf_url else [],
        ))
    # article_title/material_title are containers whose actual text is in ja/en.
    for entry, row in zip((n for n in root.iter() if _local_name(n.tag) == "entry"), rows):
        title_node = next((n for n in entry if _local_name(n.tag) == "article_title"), None)
        material_node = next((n for n in entry if _local_name(n.tag) == "material_title"), None)
        publisher_node = next((n for n in entry if _local_name(n.tag) == "publisher"), None)
        if title_node is not None:
            row["title"] = _first(title_node, "ja", "en")
        if material_node is not None:
            row["container_title"] = _first(material_node, "ja", "en")
        if publisher_node is not None:
            row["publisher"] = _first(publisher_node, "ja", "en", "name")
    return [row for row in rows if row["title"]]


def parse_cinii(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in payload.get("items") or []:
        if not isinstance(item, dict):
            continue
        resource_type = str(item.get("dc:type") or "article")
        if _NON_TEXT_TYPES.search(resource_type):
            continue
        link = item.get("link") if isinstance(item.get("link"), dict) else {}
        url = str(link.get("@id") or item.get("@id") or "")
        creators = item.get("dc:creator") or []
        if isinstance(creators, str):
            creators = [creators]
        record_id = _identifier(url)
        rows.append(_candidate(
            source="cinii", record_id=_identifier(url), title=str(item.get("title") or ""),
            creators=[str(value) for value in creators],
            date=str(item.get("prism:publicationDate") or item.get("dc:date") or ""),
            resource_type=resource_type,
            container_title=str(item.get("prism:publicationName") or ""),
            publisher=str(item.get("dc:publisher") or ""),
            volume=str(item.get("prism:volume") or ""),
            issue=str(item.get("prism:number") or ""),
            pages=_page_range(
                item.get("prism:startingPage"), item.get("prism:endingPage"),
                item.get("prism:pageRange"),
            ),
            landing_url=url,
            identifiers=_cinii_identifiers(item, record_id), fulltext_status="unknown",
        ))
    return [row for row in rows if row["title"]]


def parse_openlibrary(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in payload.get("docs") or []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "")
        public = bool(item.get("public_scan_b")) or item.get("ebook_access") == "public"
        archive_ids = [str(value) for value in item.get("ia") or []]
        downloads = [f"https://archive.org/details/{value}" for value in archive_ids[:1]] if public else []
        rows.append(_candidate(
            source="openlibrary", record_id=_identifier(key), title=str(item.get("title") or ""),
            creators=[str(value) for value in item.get("author_name") or []],
            date=str(item.get("first_publish_year") or ""), resource_type="book",
            publisher=str((item.get("publisher") or [""])[0]),
            identifiers={"openlibrary": key, "isbn": [str(v) for v in (item.get("isbn") or [])[:5]]},
            landing_url=f"https://openlibrary.org{key}",
            fulltext_status="public" if public else "restricted_or_unavailable",
            download_urls=downloads,
        ))
    return [row for row in rows if row["title"]]


async def _fetch_source(client: httpx.AsyncClient, source: str, query: str, limit: int) -> list[dict[str, Any]]:
    if source == "ndl":
        response = await client.get("https://ndlsearch.ndl.go.jp/api/opensearch", params={"any": query, "cnt": limit})
        response.raise_for_status()
        return parse_ndl(response.text)[:limit]
    if source == "jstage":
        response = await client.get(
            "https://api.jstage.jst.go.jp/searchapi/do",
            params={"service": 3, "article": query, "count": limit},
        )
        response.raise_for_status()
        return parse_jstage(response.text)[:limit]
    if source == "cinii":
        app_id = os.environ.get("CINII_APP_ID", "").strip()
        if not app_id:
            raise RuntimeError("CINII_APP_ID is not configured")
        response = await client.get(
            "https://cir.nii.ac.jp/opensearch/articles",
            params={"q": query, "count": limit, "format": "json", "appid": app_id},
        )
        response.raise_for_status()
        return parse_cinii(response.json())[:limit]
    if source == "openlibrary":
        fields = "key,title,author_name,first_publish_year,public_scan_b,ebook_access,ia,publisher,isbn"
        response = await client.get(
            "https://openlibrary.org/search.json",
            params={"q": query, "limit": limit, "fields": fields},
        )
        response.raise_for_status()
        return parse_openlibrary(response.json())[:limit]
    raise ValueError(f"unsupported source: {source}")


def _cache_key(source: str, query: str, limit: int) -> tuple[str, str, int]:
    return source, " ".join(query.casefold().split()), limit


async def _fetch_cached(
    client: httpx.AsyncClient, source: str, query: str, limit: int,
) -> list[dict[str, Any]]:
    key = _cache_key(source, query, limit)
    cached = _CACHE.get(key)
    now = time.monotonic()
    if cached and now - cached[0] < _CACHE_TTL_SECONDS:
        return [dict(row) for row in cached[1]]
    async with _SOURCE_LOCKS[source]:
        cached = _CACHE.get(key)
        if cached and time.monotonic() - cached[0] < _CACHE_TTL_SECONDS:
            return [dict(row) for row in cached[1]]
        attempts = 3 if source == "ndl" else 1
        for attempt in range(attempts):
            try:
                rows = await _fetch_source(client, source, query, limit)
                _CACHE[key] = (time.monotonic(), rows)
                return [dict(row) for row in rows]
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 429 or attempt + 1 == attempts:
                    raise
                retry_after = exc.response.headers.get("Retry-After", "")
                try:
                    delay = max(0.5, min(float(retry_after), 10.0))
                except ValueError:
                    delay = float(2 ** attempt)
                await asyncio.sleep(delay)


def _query_terms(query: str) -> list[str]:
    return list(dict.fromkeys(term for term in re.split(r"[\s　]+", query.strip()) if term))


def _candidate_identity(candidate: dict[str, Any]) -> tuple[str, str]:
    stable = str(candidate.get("record_id") or candidate.get("landing_url") or "")
    return str(candidate.get("source") or ""), stable or str(candidate.get("title") or "").casefold()


async def _search_source(
    client: httpx.AsyncClient, source: str, query: str, limit: int, mode: str,
) -> list[dict[str, Any]]:
    initial = await _fetch_cached(client, source, query, limit)
    if mode == "all" or (mode == "auto" and initial):
        return [{**row, "search_strategy": "source_native", "matched_query": query} for row in initial]
    if mode == "phrase":
        phrase = "".join(query.casefold().split())
        return [{**row, "search_strategy": "local_phrase_filter", "matched_query": query}
                for row in initial if phrase in "".join(
                    f"{row.get('title', '')}{row.get('container_title', '')}".casefold().split()
                )]
    terms = _query_terms(query)
    if len(terms) < 2:
        return initial
    strategy = "any_term" if mode == "any" else "relaxed_any_term"
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    if mode == "any":
        merged.update({
            _candidate_identity(row): {**row, "search_strategy": strategy, "matched_query": query}
            for row in initial
        })
    for term in terms:
        for row in await _fetch_cached(client, source, term, limit):
            merged.setdefault(
                _candidate_identity(row),
                {**row, "search_strategy": strategy, "matched_query": term},
            )
            if len(merged) >= limit:
                break
        if len(merged) >= limit:
            break
    return list(merged.values())[:limit]


async def search_external_texts(
    query: str, sources: list[str], limit: int, search_mode: str = "auto",
) -> dict[str, Any]:
    selected = list(dict.fromkeys(source.casefold() for source in sources))
    unknown = sorted(set(selected) - set(SUPPORTED_SOURCES))
    if unknown:
        raise ValueError(f"unsupported sources: {', '.join(unknown)}")
    search_mode = search_mode.casefold()
    if search_mode not in _SEARCH_MODES:
        raise ValueError(f"unsupported search_mode: {search_mode}")
    async with httpx.AsyncClient(headers=_HEADERS, timeout=30, follow_redirects=True) as client:
        results = await asyncio.gather(
            *(_search_source(client, source, query, limit, search_mode) for source in selected),
            return_exceptions=True,
        )
    candidates: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    for source, result in zip(selected, results):
        if isinstance(result, BaseException):
            if isinstance(result, httpx.HTTPStatusError):
                errors[source] = f"HTTP {result.response.status_code} returned by {source}"
            elif isinstance(result, httpx.RequestError):
                errors[source] = f"Network request to {source} failed ({type(result).__name__})"
            else:
                errors[source] = str(result)
        else:
            candidates.extend(result)
    return {
        "query": query, "sources": selected, "search_mode": search_mode,
        "candidates": candidates,
        "source_errors": errors, "writes_performed": False,
    }


def screening_observations(candidate: dict[str, Any], duplicates: list[dict[str, Any]]) -> dict[str, Any]:
    missing = [field for field in ("title", "creators", "date", "landing_url") if not candidate.get(field)]
    cautions: list[str] = []
    if candidate.get("fulltext_status") != "public":
        cautions.append("本文の公開・取得条件を確認できていません。")
    if missing:
        cautions.append(f"書誌項目が不足しています: {', '.join(missing)}")
    resource_type = str(candidate.get("resource_type") or "").casefold()
    if any(word in resource_type for word in ("thesis", "dissertation", "紀要", "論文")):
        cautions.append("資料種別だけでは研究の質を判断できません。本文の根拠と方法を確認してください。")
    return {
        **candidate,
        "zotero_duplicates": duplicates,
        "bibliographic_missing": missing,
        "cautions": cautions,
        "citation_evidence": "unknown_not_penalized",
        "eligible_for_proposal": bool(candidate.get("title") and candidate.get("landing_url")),
        "writes_performed": False,
    }
