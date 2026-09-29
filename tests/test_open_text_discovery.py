from __future__ import annotations

import asyncio

import pytest
import httpx

from src import open_text_discovery as discovery
from src.open_text_discovery import (
    parse_cinii, parse_jstage, parse_ndl, parse_openlibrary, screening_observations,
)


@pytest.fixture(autouse=True)
def clear_discovery_cache():
    discovery._CACHE.clear()


def test_ndl_parser_keeps_text_and_rejects_non_text_resources():
    xml = """<rss xmlns:dc='http://purl.org/dc/elements/1.1/'><channel>
      <item><title>文字資料</title><link>https://ndl.example/books/ID1</link>
        <author>著者</author><category>図書</category><dc:date>1900</dc:date></item>
      <item><title>古地図</title><link>https://ndl.example/books/ID2</link><category>地図</category></item>
    </channel></rss>"""
    rows = parse_ndl(xml)
    assert [row["title"] for row in rows] == ["文字資料"]
    assert rows[0]["record_id"] == "ID1"


def test_jstage_parser_reads_nested_japanese_fields():
    xml = """<feed xmlns='http://www.w3.org/2005/Atom' xmlns:prism='urn:prism'>
      <entry><article_title><ja>論文名</ja><en>Article</en></article_title>
        <article_link><ja>https://www.jstage.jst.go.jp/article/a/1/1/1/_article/-char/ja/</ja></article_link>
        <author><ja><name>著者名</name></ja></author>
        <material_title><ja>雑誌名</ja></material_title><prism:issn>1234-5678</prism:issn>
        <publisher><name><ja>植物学会</ja><en>Botanical Society</en></name></publisher>
        <prism:volume>19</prism:volume><prism:number>216</prism:number>
        <prism:startingPage>1b</prism:startingPage><prism:endingPage>3b</prism:endingPage>
        <pubyear>1935</pubyear><doi>10.1234/example</doi>
      </entry></feed>"""
    row = parse_jstage(xml)[0]
    assert row["title"] == "論文名"
    assert row["creators"] == ["著者名"]
    assert row["container_title"] == "雑誌名"
    assert row["publisher"] == "植物学会"
    assert row["record_id"] == "article:a:1:1:1"
    assert row["date"] == "1935"
    assert (row["volume"], row["issue"], row["pages"]) == ("19", "216", "1b-3b")
    assert row["identifiers"]["doi"] == "10.1234/example"
    assert row["download_urls"] == ["https://www.jstage.jst.go.jp/article/a/1/1/1/_pdf/-char/ja"]
    assert discovery._jstage_identifier("") == ""


def test_cinii_parser_does_not_treat_missing_citations_as_negative():
    rows = parse_cinii({"items": [None, {
        "title": "研究", "link": {"@id": "https://cir.nii.ac.jp/crid/123"},
        "dc:creator": "著者", "dc:type": "journal article", "prism:publicationDate": "2011",
        "prism:volume": "7", "prism:number": "2", "prism:startingPage": "11",
        "prism:endingPage": "29",
        "dc:identifier": [None, {"@type": "cir:NAID", "@value": "4001"},
                          {"@type": "cir:NAID", "@value": "4002"},
                          {"@type": "cir:NAID", "@value": "4003"}],
    }, {
        "title": "地図", "@id": "https://cir.nii.ac.jp/crid/map", "dc:type": "map",
    }]})
    screened = screening_observations(rows[0], [])
    assert screened["citation_evidence"] == "unknown_not_penalized"
    assert screened["eligible_for_proposal"] is True
    assert rows[0]["date"] == "2011"
    assert (rows[0]["volume"], rows[0]["issue"], rows[0]["pages"]) == ("7", "2", "11-29")
    assert rows[0]["identifiers"]["naid"] == ["4001", "4002", "4003"]


def test_explicit_page_range_wins_over_start_and_end():
    assert discovery._page_range("11", "29", "S1-S4") == "S1-S4"


def test_openlibrary_marks_only_public_scans_as_downloadable():
    rows = parse_openlibrary({"docs": [None,
        {"key": "/works/OPEN", "title": "Open", "public_scan_b": True, "ia": ["scan1"]},
        {"key": "/works/LOCKED", "title": "Locked", "ebook_access": "borrowable", "ia": ["scan2"]},
    ]})
    assert rows[0]["fulltext_status"] == "public"
    assert rows[0]["download_urls"] == ["https://archive.org/details/scan1"]
    assert rows[1]["fulltext_status"] == "restricted_or_unavailable"
    assert rows[1]["download_urls"] == []


def test_screening_reports_observations_instead_of_quality_score():
    result = screening_observations(
        {"title": "Thesis", "resource_type": "dissertation", "landing_url": "https://example.test"},
        [{"key": "DUP1"}],
    )
    assert "quality_score" not in result
    assert result["zotero_duplicates"] == [{"key": "DUP1"}]
    assert any("質を判断できません" in value for value in result["cautions"])


class _Response:
    def __init__(self, *, text="", payload=None):
        self.text = text
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _Client:
    async def get(self, url, params):
        if "ndlsearch" in url:
            return _Response(text="<rss><channel><item><title>NDL</title><link>https://x/ndl</link></item></channel></rss>")
        if "jstage" in url:
            return _Response(text="""<feed><entry><article_title><ja>J-STAGE</ja></article_title>
                <article_link><ja>https://x/article/jstage</ja></article_link></entry></feed>""")
        if "cir.nii" in url:
            return _Response(payload={"items": [{"title": "CiNii", "link": {"@id": "https://x/cinii"}}]})
        return _Response(payload={"docs": [{"key": "/works/OL1", "title": "OL"}]})


def test_all_source_fetchers_use_normalized_candidate_contract(monkeypatch):
    monkeypatch.setenv("CINII_APP_ID", "test-app")

    async def run():
        return {
            source: await discovery._fetch_source(_Client(), source, "query", 2)
            for source in discovery.SUPPORTED_SOURCES
        }

    results = asyncio.run(run())
    assert {source: rows[0]["source"] for source, rows in results.items()} == {
        source: source for source in discovery.SUPPORTED_SOURCES
    }
    with pytest.raises(ValueError, match="unsupported"):
        asyncio.run(discovery._fetch_source(_Client(), "unknown", "query", 2))


def test_cinii_without_app_id_is_a_source_error(monkeypatch):
    monkeypatch.delenv("CINII_APP_ID", raising=False)
    with pytest.raises(RuntimeError, match="CINII_APP_ID"):
        asyncio.run(discovery._fetch_source(_Client(), "cinii", "query", 1))


def test_search_orchestrator_deduplicates_sources_and_reports_partial_errors(monkeypatch):
    class ContextClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    async def fake_fetch(client, source, query, limit):
        if source == "jstage":
            raise RuntimeError("temporary failure")
        return [{"source": source, "title": query}]

    monkeypatch.setattr(discovery.httpx, "AsyncClient", ContextClient)
    monkeypatch.setattr(discovery, "_fetch_source", fake_fetch)
    result = asyncio.run(discovery.search_external_texts("topic", ["NDL", "ndl", "jstage"], 2))
    assert result["sources"] == ["ndl", "jstage"]
    assert [(row["source"], row["title"]) for row in result["candidates"]] == [("ndl", "topic")]
    assert result["candidates"][0]["search_strategy"] == "source_native"
    assert result["source_errors"] == {"jstage": "temporary failure"}
    assert result["search_mode"] == "auto"
    with pytest.raises(ValueError, match="unsupported sources"):
        asyncio.run(discovery.search_external_texts("topic", ["unknown"], 2))
    with pytest.raises(ValueError, match="unsupported search_mode"):
        asyncio.run(discovery.search_external_texts("topic", ["ndl"], 2, "loose"))


def test_auto_search_relaxes_zero_result_source_and_cache_avoids_repeat(monkeypatch):
    calls = []

    async def fake_fetch(client, source, query, limit):
        calls.append(query)
        return [] if " " in query else [{"source": source, "record_id": query, "title": query}]

    monkeypatch.setattr(discovery, "_fetch_source", fake_fetch)

    async def run():
        async with httpx.AsyncClient() as client:
            first = await discovery._search_source(client, "ndl", "alpha beta", 5, "auto")
            second = await discovery._search_source(client, "ndl", "alpha beta", 5, "auto")
            phrase = await discovery._search_source(client, "ndl", "alpha beta", 5, "phrase")
            return first, second, phrase

    first, second, phrase = asyncio.run(run())
    assert [row["title"] for row in first] == ["alpha", "beta"]
    assert [row["matched_query"] for row in first] == ["alpha", "beta"]
    assert all(row["search_strategy"] == "relaxed_any_term" for row in first)
    assert second == first
    assert phrase == []
    assert calls == ["alpha beta", "alpha", "beta"]


def test_ndl_429_retries_then_caches(monkeypatch):
    attempts = 0

    async def fake_fetch(client, source, query, limit):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            request = httpx.Request("GET", "https://ndl.example/search")
            response = httpx.Response(429, request=request, headers={"Retry-After": "0"})
            raise httpx.HTTPStatusError("limited", request=request, response=response)
        return [{"source": source, "record_id": "1", "title": "result"}]

    async def no_sleep(delay):
        return None

    monkeypatch.setattr(discovery, "_fetch_source", fake_fetch)
    monkeypatch.setattr(discovery.asyncio, "sleep", no_sleep)

    async def run():
        async with httpx.AsyncClient() as client:
            first = await discovery._fetch_cached(client, "ndl", "query", 1)
            second = await discovery._fetch_cached(client, "ndl", "query", 1)
            return first, second

    first, second = asyncio.run(run())
    assert first == second
    assert attempts == 2


def test_retry_uses_backoff_when_retry_after_is_invalid(monkeypatch):
    attempts = 0
    delays = []

    async def fake_fetch(client, source, query, limit):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            request = httpx.Request("GET", "https://ndl.example/search")
            response = httpx.Response(429, request=request, headers={"Retry-After": "invalid"})
            raise httpx.HTTPStatusError("limited", request=request, response=response)
        return []

    async def remember_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(discovery, "_fetch_source", fake_fetch)
    monkeypatch.setattr(discovery.asyncio, "sleep", remember_sleep)
    asyncio.run(discovery._fetch_cached(_Client(), "ndl", "query", 1))
    assert delays == [1.0]


def test_any_mode_and_single_term_paths(monkeypatch):
    async def fake_fetch(client, source, query, limit):
        if query == "alpha beta":
            return [{"source": source, "record_id": "both", "title": "both"}]
        return [{"source": source, "record_id": query, "title": query}]

    monkeypatch.setattr(discovery, "_fetch_source", fake_fetch)

    async def run():
        client = _Client()
        any_rows = await discovery._search_source(client, "jstage", "alpha beta", 2, "any")
        single = await discovery._search_source(client, "jstage", "alpha", 2, "any")
        return any_rows, single

    any_rows, single = asyncio.run(run())
    assert len(any_rows) == 2
    assert all(row["search_strategy"] == "any_term" for row in any_rows)
    assert single[0]["title"] == "alpha"


def test_cache_filled_while_waiting_for_source_lock(monkeypatch):
    key = discovery._cache_key("ndl", "query", 1)

    class FillingLock:
        async def __aenter__(self):
            discovery._CACHE[key] = (discovery.time.monotonic(), [{"title": "cached"}])

        async def __aexit__(self, *args):
            return None

    monkeypatch.setitem(discovery._SOURCE_LOCKS, "ndl", FillingLock())
    result = asyncio.run(discovery._fetch_cached(_Client(), "ndl", "query", 1))
    assert result == [{"title": "cached"}]


def test_search_errors_never_expose_query_credentials(monkeypatch):
    class ContextClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    request = httpx.Request("GET", "https://example.test/search?appid=secret-value")
    response = httpx.Response(404, request=request)

    async def fail(client, source, query, limit):
        if source == "ndl":
            raise httpx.ConnectError("network down", request=request)
        raise httpx.HTTPStatusError("credential-bearing URL", request=request, response=response)

    monkeypatch.setattr(discovery.httpx, "AsyncClient", ContextClient)
    monkeypatch.setattr(discovery, "_fetch_source", fail)
    result = asyncio.run(discovery.search_external_texts("topic", ["cinii", "ndl"], 1))
    assert result["source_errors"] == {
        "cinii": "HTTP 404 returned by cinii",
        "ndl": "Network request to ndl failed (ConnectError)",
    }
    assert "secret-value" not in str(result)
