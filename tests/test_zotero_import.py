from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from src import zotero_import


def candidate(**overrides):
    row = {
        "source": "jstage",
        "record_id": "article-1",
        "title": "Rice research",
        "creators": ["Example Author"],
        "date": "2026",
        "resource_type": "journalArticle",
        "landing_url": "https://www.jstage.jst.go.jp/article/example",
        "download_urls": ["https://www.jstage.jst.go.jp/article/example/_pdf"],
        "identifiers": {"doi": "10.1234/example"},
    }
    row.update(overrides)
    return row


def test_proposal_is_read_only_single_use_and_phrase_bound(tmp_path):
    path = tmp_path / "proposals.json"
    proposal = zotero_import.create_import_proposal(candidate(), [], import_mode="pdf", path=path)

    assert proposal["writes_performed"] is False
    assert proposal["pdf_url"].endswith("/_pdf")
    stored = json.loads(path.read_text())["proposals"][proposal["proposal_id"]]
    assert proposal["approval_phrase"] not in json.dumps(stored)

    with pytest.raises(ValueError, match="does not match"):
        zotero_import.claim_import_proposal(proposal["proposal_id"], "IMPORT wrong", path=path)

    claimed = zotero_import.claim_import_proposal(
        proposal["proposal_id"], proposal["approval_phrase"], path=path,
    )
    assert claimed["status"] == "executing"
    with pytest.raises(ValueError, match="already executing"):
        zotero_import.claim_import_proposal(
            proposal["proposal_id"], proposal["approval_phrase"], path=path,
        )


def test_proposal_blocks_duplicates_unless_reviewed_and_rejects_bad_pdf_url(tmp_path):
    duplicate = [{"key": "DUP1", "title": "Rice research"}]
    with pytest.raises(ValueError, match="duplicate"):
        zotero_import.create_import_proposal(candidate(), duplicate, path=tmp_path / "one.json")
    reviewed = zotero_import.create_import_proposal(
        candidate(), duplicate, allow_duplicate=True, path=tmp_path / "two.json",
    )
    assert reviewed["duplicates"] == duplicate

    unsafe = candidate(download_urls=["https://example.test/file.pdf"])
    with pytest.raises(ValueError, match="allowlisted"):
        zotero_import.create_import_proposal(
            unsafe, [], import_mode="pdf", path=tmp_path / "three.json",
        )


def test_expired_proposal_cannot_be_claimed(tmp_path, monkeypatch):
    path = tmp_path / "proposals.json"
    monkeypatch.setattr(zotero_import.time, "time", lambda: 100)
    proposal = zotero_import.create_import_proposal(candidate(), [], path=path)
    monkeypatch.setattr(zotero_import.time, "time", lambda: 100 + zotero_import.PROPOSAL_TTL_SECONDS + 1)

    with pytest.raises(ValueError, match="expired"):
        zotero_import.claim_import_proposal(
            proposal["proposal_id"], proposal["approval_phrase"], path=path,
        )


def test_local_writer_uses_native_authorization_and_three_phase_upload(tmp_path, monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL_API_BASE", "http://zotero.test/api")
    monkeypatch.setenv("ZOTERO_LOCAL_API_PREFIX", "users/0")
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/":
            return httpx.Response(200, headers={"Zotero-Server-ID": "server-1"})
        if request.url.path == "/api/local/authorize":
            return httpx.Response(200, json={"key": "write-key", "remember": True})
        if request.url.path == "/api/users/0/items":
            assert request.headers["Zotero-API-Key"] == "write-key"
            return httpx.Response(200, json={"successful": {"0": {}}})
        if request.url.path == "/api/users/0/items/ATTACH01/file":
            if b"upload=" in request.content:
                return httpx.Response(204)
            return httpx.Response(200, json={
                "url": "/storage/upload", "uploadKey": "upload-1",
                "contentType": "application/pdf",
            })
        if request.url.path == "/storage/upload":
            assert request.content.startswith(b"%PDF-")
            return httpx.Response(201)
        return httpx.Response(404)

    writer = zotero_import.ZoteroLocalWriter(transport=httpx.MockTransport(handler))
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF-1.7\nexample")
    asyncio.run(writer.create_items([{"itemType": "book", "title": "Example"}]))
    asyncio.run(writer.upload_file("ATTACH01", pdf))

    paths = [request.url.path for request in requests]
    assert paths.count("/api/local/authorize") == 1
    assert paths[-3:] == [
        "/api/users/0/items/ATTACH01/file", "/storage/upload",
        "/api/users/0/items/ATTACH01/file",
    ]


def test_download_rejects_cross_origin_redirect(tmp_path, monkeypatch):
    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url):
            return httpx.Response(
                302,
                headers={"Location": "https://evil.example/file.pdf"},
                request=httpx.Request("GET", url),
            )

    monkeypatch.setattr(zotero_import.httpx, "AsyncClient", lambda **kwargs: Client())
    row = {"candidate": candidate(), "pdf_url": candidate()["download_urls"][0]}
    with pytest.raises(ValueError, match="outside"):
        asyncio.run(zotero_import.download_candidate_pdf(row, tmp_path / "file.pdf"))


def test_finish_records_terminal_result(tmp_path):
    path = tmp_path / "proposals.json"
    proposal = zotero_import.create_import_proposal(candidate(), [], path=path)
    zotero_import.finish_import_proposal(
        proposal["proposal_id"], "failed", {"error": "denied"}, path=path,
    )
    row = json.loads(path.read_text())["proposals"][proposal["proposal_id"]]
    assert row["status"] == "failed"
    assert row["result"] == {"error": "denied"}
