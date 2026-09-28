from __future__ import annotations

import asyncio
import io
import json
import zipfile

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
    assert proposal["download_url"].endswith("/_pdf")
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


def test_proposal_blocks_duplicates_unless_reviewed_and_rejects_non_https_pdf(tmp_path):
    duplicate = [{"key": "DUP1", "title": "Rice research"}]
    with pytest.raises(ValueError, match="duplicate"):
        zotero_import.create_import_proposal(candidate(), duplicate, path=tmp_path / "one.json")
    reviewed = zotero_import.create_import_proposal(
        candidate(), duplicate, allow_duplicate=True, path=tmp_path / "two.json",
    )
    assert reviewed["duplicates"] == duplicate

    unsafe = candidate(download_urls=["http://example.test/file.pdf"])
    with pytest.raises(ValueError, match="direct HTTPS"):
        zotero_import.create_import_proposal(
            unsafe, [], import_mode="pdf", path=tmp_path / "three.json",
        )


def test_epub_proposal_and_structure_validation(tmp_path):
    epub_candidate = candidate(
        source="publisher-site",
        landing_url="https://publisher.example/books/one",
        download_urls=["https://cdn.example/files/book.epub"],
        resource_type="book",
    )
    proposal = zotero_import.create_import_proposal(
        epub_candidate, [], import_mode="epub", path=tmp_path / "epub.json",
    )
    assert proposal["download_url"].endswith(".epub")

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        archive.writestr("META-INF/container.xml", "<container/>")
    zotero_import._validate_download(payload.getvalue(), "epub")

    with pytest.raises(ValueError, match="not an EPUB"):
        zotero_import._validate_download(b"PK-not-really-a-zip", "epub")


def test_new_parent_and_attachment_have_no_preassigned_identity():
    row = {
        "candidate": zotero_import._candidate_payload(candidate()),
        "collection_key": "",
        "import_mode": "pdf",
        "download_url": candidate()["download_urls"][0],
    }

    parent, attachment = zotero_import._zotero_items(row)

    assert "key" not in parent and "version" not in parent
    assert "key" not in attachment and "version" not in attachment
    assert "parentItem" not in attachment


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
    monkeypatch.setenv("ZOTERO_LOCAL_AUTH_PATH", str(tmp_path / "auth.json"))
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
    auth_path = tmp_path / "auth.json"
    assert json.loads(auth_path.read_text()) == {
        "server_id": "server-1", "key": "write-key",
    }
    assert auth_path.stat().st_mode & 0o777 == 0o600


def test_local_writer_reuses_remembered_authorization(tmp_path, monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL_API_BASE", "http://zotero.test/api")
    monkeypatch.setenv("ZOTERO_LOCAL_API_PREFIX", "users/0")
    auth_path = tmp_path / "auth.json"
    monkeypatch.setenv("ZOTERO_LOCAL_AUTH_PATH", str(auth_path))
    auth_path.write_text(json.dumps({"server_id": "server-1", "key": "saved-key"}))
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/":
            return httpx.Response(200, headers={"Zotero-Server-ID": "server-1"})
        assert request.url.path == "/api/users/0/items"
        assert request.headers["Zotero-API-Key"] == "saved-key"
        return httpx.Response(200, json={"successful": {"0": {}}})

    writer = zotero_import.ZoteroLocalWriter(transport=httpx.MockTransport(handler))
    asyncio.run(writer.create_items([{"itemType": "book", "title": "Example"}]))

    assert [request.url.path for request in requests] == ["/api/", "/api/users/0/items"]


def test_local_writer_clears_rejected_remembered_authorization(tmp_path, monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL_API_BASE", "http://zotero.test/api")
    monkeypatch.setenv("ZOTERO_LOCAL_API_PREFIX", "users/0")
    auth_path = tmp_path / "auth.json"
    monkeypatch.setenv("ZOTERO_LOCAL_AUTH_PATH", str(auth_path))
    auth_path.write_text(json.dumps({"server_id": "server-1", "key": "old-key"}))
    writes = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal writes
        if request.url.path == "/api/":
            return httpx.Response(200, headers={"Zotero-Server-ID": "server-1"})
        if request.url.path == "/api/local/authorize":
            assert not auth_path.exists()
            return httpx.Response(200, json={"key": "new-key", "remember": True})
        writes += 1
        if writes == 1:
            assert request.headers["Zotero-API-Key"] == "old-key"
            return httpx.Response(401)
        assert request.headers["Zotero-API-Key"] == "new-key"
        return httpx.Response(200, json={"successful": {"0": {}}})

    writer = zotero_import.ZoteroLocalWriter(transport=httpx.MockTransport(handler))
    asyncio.run(writer.create_items([{"itemType": "book", "title": "Example"}]))

    assert json.loads(auth_path.read_text())["key"] == "new-key"


def test_local_authorization_timeout_has_unlock_instructions(tmp_path, monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL_API_BASE", "http://zotero.test/api")
    monkeypatch.setenv("ZOTERO_LOCAL_AUTH_PATH", str(tmp_path / "auth.json"))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/":
            return httpx.Response(200, headers={"Zotero-Server-ID": "server-1"})
        raise httpx.ReadTimeout("permission dialog was not answered", request=request)

    writer = zotero_import.ZoteroLocalWriter(transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="Unlock this Mac.*Always Allow"):
        asyncio.run(writer.authorize())


def test_zotero_9_uses_configured_web_api_fallback(monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL_API_BASE", "http://zotero.test/api")
    monkeypatch.setenv("ZOTERO_USER_ID", "12345")
    monkeypatch.setenv("ZOTERO_API_KEY", "web-write-key")
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "zotero.test":
            return httpx.Response(200, headers={"X-Zotero-Version": "9.0.6"})
        assert request.url == "https://api.zotero.org/users/12345/items"
        assert request.headers["Zotero-API-Key"] == "web-write-key"
        assert "Zotero-Server-ID" not in request.headers
        return httpx.Response(200, json={"successful": {"0": {}}})

    writer = zotero_import.ZoteroLocalWriter(transport=httpx.MockTransport(handler))
    asyncio.run(writer.create_items([{"itemType": "book", "title": "Example"}]))

    assert len(requests) == 2
    assert writer.web_api is True


def test_zotero_9_without_web_credentials_reports_actionable_error(monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL_API_BASE", "http://zotero.test/api")
    monkeypatch.delenv("ZOTERO_USER_ID", raising=False)
    monkeypatch.delenv("ZOTERO_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"X-Zotero-Version": "9.0.6"})

    writer = zotero_import.ZoteroLocalWriter(transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match=r"Zotero 9\.0\.6.*Zotero 10\+ required"):
        asyncio.run(writer.create_items([{"itemType": "book", "title": "Example"}]))


def test_download_allows_cross_origin_https_redirect(tmp_path, monkeypatch):
    calls = []

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url):
            calls.append(url)
            if len(calls) > 1:
                return httpx.Response(
                    200, content=b"%PDF-1.7\nredirected",
                    request=httpx.Request("GET", url),
                )
            return httpx.Response(
                302,
                headers={"Location": "https://evil.example/file.pdf"},
                request=httpx.Request("GET", url),
            )

    monkeypatch.setattr(zotero_import.httpx, "AsyncClient", lambda **kwargs: Client())
    row = {
        "candidate": candidate(), "import_mode": "pdf",
        "download_url": candidate()["download_urls"][0],
    }
    destination = asyncio.run(
        zotero_import.download_candidate_file(row, tmp_path / "file.pdf"),
    )
    assert destination.read_bytes().startswith(b"%PDF-")
    assert calls == [
        "https://www.jstage.jst.go.jp/article/example/_pdf",
        "https://evil.example/file.pdf",
    ]


def test_execute_epub_import_uploads_and_removes_temporary_file(monkeypatch):
    paths = []
    created = []

    class Writer:
        async def create_items(self, items):
            created.append(items[0])
            if len(created) == 1:
                assert items[0]["tags"] == [{"tag": "AI-added"}]
                assert "parentItem" not in items[0]
                return {"successful": {"0": {"key": "PARENT01"}}}
            assert items[0]["contentType"] == "application/epub+zip"
            assert items[0]["filename"].endswith(".epub")
            assert items[0]["parentItem"] == "PARENT01"
            return {"successful": {"0": {"key": "ATTACH01"}}}

        async def upload_file(self, attachment_key, file_path):
            assert attachment_key == "ATTACH01"
            assert file_path.read_bytes() == b"validated epub"
            paths.append(file_path)

    async def fake_download(row, destination):
        assert row["import_mode"] == "epub"
        destination.write_bytes(b"validated epub")
        return destination

    monkeypatch.setattr(zotero_import, "download_candidate_file", fake_download)
    row = {
        "candidate": zotero_import._candidate_payload(candidate(
            source="publisher-site",
            landing_url="https://publisher.example/books/one",
            download_urls=["https://cdn.example/files/book.epub"],
            resource_type="book",
        )),
        "collection_key": "",
        "import_mode": "epub",
        "download_url": "https://cdn.example/files/book.epub",
    }

    result = asyncio.run(zotero_import.execute_import(row, writer=Writer()))

    assert result["import_mode"] == "epub"
    assert result["item_key"] == "PARENT01"
    assert result["attachment_key"] == "ATTACH01"
    assert result["file_uploaded"] is True
    assert len(created) == 2
    assert paths and not paths[0].exists()


def test_finish_records_terminal_result(tmp_path):
    path = tmp_path / "proposals.json"
    proposal = zotero_import.create_import_proposal(candidate(), [], path=path)
    zotero_import.finish_import_proposal(
        proposal["proposal_id"], "failed", {"error": "denied"}, path=path,
    )
    row = json.loads(path.read_text())["proposals"][proposal["proposal_id"]]
    assert row["status"] == "failed"
    assert row["result"] == {"error": "denied"}
