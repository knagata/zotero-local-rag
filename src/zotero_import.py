"""Explicitly approved imports from screened open-text candidates into Zotero."""
from __future__ import annotations

import hashlib
import io
import json
import mimetypes
import os
import secrets
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from collections.abc import Mapping
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

try:
    from .zotero_source_localapi import local_api_base, local_api_prefix, zotero_api_headers
except ImportError:
    from zotero_source_localapi import local_api_base, local_api_prefix, zotero_api_headers


PROPOSAL_TTL_SECONDS = 24 * 60 * 60
MAX_FULLTEXT_BYTES = 250 * 1024 * 1024
SUPPORTED_IMPORT_MODES = frozenset({"metadata", "pdf", "epub"})
AI_ADDED_TAG = "AI-added"
_PROPOSAL_LOCK = threading.Lock()
_AUTH_LOCK = threading.Lock()


def _proposal_path() -> Path:
    configured = os.environ.get("ZOTERO_IMPORT_PROPOSALS_PATH")
    return Path(configured).expanduser() if configured else (
        Path(__file__).resolve().parents[1] / "data" / "zotero_import_proposals.json"
    )


def _auth_path() -> Path:
    configured = os.environ.get("ZOTERO_LOCAL_AUTH_PATH")
    return Path(configured).expanduser() if configured else (
        Path(__file__).resolve().parents[1] / "data" / "zotero_local_write_auth.json"
    )


def _load_local_auth(server_id: str) -> str:
    with _AUTH_LOCK:
        try:
            payload = json.loads(_auth_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ""
    if payload.get("server_id") != server_id:
        return ""
    return str(payload.get("key") or "")


def _save_local_auth(server_id: str, key: str) -> None:
    target = _auth_path()
    with _AUTH_LOCK:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps({"server_id": server_id, "key": key}) + "\n", encoding="utf-8",
        )
        temporary.chmod(0o600)
        os.replace(temporary, target)
        target.chmod(0o600)


def _clear_local_auth() -> None:
    target = _auth_path()
    with _AUTH_LOCK:
        if target.exists():
            target.unlink()


def _load_proposals(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"version": 1, "proposals": {}}
    return payload if isinstance(payload, dict) and isinstance(payload.get("proposals"), dict) else {
        "version": 1, "proposals": {},
    }


def _save_proposals(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _candidate_payload(candidate: Mapping[str, Any]) -> dict[str, Any]:
    source = str(candidate.get("source") or "").strip().casefold()
    title = str(candidate.get("title") or "").strip()
    landing_url = str(candidate.get("landing_url") or "").strip()
    if not title or not landing_url:
        raise ValueError("candidate title and landing_url are required")
    parsed = urlparse(landing_url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("candidate landing_url must be HTTPS")
    return {
        "source": source,
        "record_id": str(candidate.get("record_id") or "").strip(),
        "title": title,
        "creators": [str(value).strip() for value in candidate.get("creators") or [] if str(value).strip()],
        "date": str(candidate.get("date") or "").strip(),
        "resource_type": str(candidate.get("resource_type") or "text").strip(),
        "container_title": str(candidate.get("container_title") or "").strip(),
        "publisher": str(candidate.get("publisher") or "").strip(),
        "volume": str(candidate.get("volume") or "").strip(),
        "issue": str(candidate.get("issue") or "").strip(),
        "pages": str(candidate.get("pages") or "").strip(),
        "identifiers": dict(candidate.get("identifiers") or {}),
        "landing_url": landing_url,
        "fulltext_status": str(candidate.get("fulltext_status") or "unknown"),
        "download_urls": [str(value).strip() for value in candidate.get("download_urls") or [] if str(value).strip()],
    }


def _direct_download_url(candidate: Mapping[str, Any], mode: str) -> str:
    for value in candidate.get("download_urls") or []:
        parsed = urlparse(str(value))
        path = parsed.path.casefold()
        is_requested_format = (
            mode == "pdf" and (path.endswith(".pdf") or "_pdf" in path)
        ) or (mode == "epub" and path.endswith(".epub"))
        if (
            is_requested_format
            and parsed.scheme == "https"
            and bool(parsed.hostname)
        ):
            return str(value)
    return ""


def create_import_proposal(
    candidate: Mapping[str, Any], duplicates: list[dict[str, Any]], *,
    import_mode: str = "metadata", collection_key: str = "", allow_duplicate: bool = False,
    path: Path | None = None,
) -> dict[str, Any]:
    """Persist an immutable, expiring proposal. This performs no external write."""
    normalized = _candidate_payload(candidate)
    mode = str(import_mode or "metadata").casefold()
    if mode not in SUPPORTED_IMPORT_MODES:
        raise ValueError(f"unsupported import_mode: {mode}")
    download_url = _direct_download_url(normalized, mode) if mode != "metadata" else ""
    if mode != "metadata" and not download_url:
        raise ValueError(f"candidate has no direct HTTPS {mode.upper()} URL; use metadata mode")
    if duplicates and not allow_duplicate:
        raise ValueError("Zotero duplicate candidates exist; set allow_duplicate only after user review")
    proposal_id = secrets.token_hex(8)
    approval_phrase = f"IMPORT {proposal_id}"
    now = int(time.time())
    row = {
        "proposal_id": proposal_id,
        "approval_phrase_sha256": hashlib.sha256(approval_phrase.encode()).hexdigest(),
        "created_at": now,
        "expires_at": now + PROPOSAL_TTL_SECONDS,
        "status": "pending",
        "candidate": normalized,
        "duplicates": duplicates,
        "allow_duplicate": bool(allow_duplicate),
        "import_mode": mode,
        "download_url": download_url,
        "collection_key": str(collection_key or "").strip(),
    }
    target = path or _proposal_path()
    with _PROPOSAL_LOCK:
        payload = _load_proposals(target)
        payload["proposals"][proposal_id] = row
        _save_proposals(target, payload)
    return {
        "proposal_id": proposal_id,
        "approval_phrase": approval_phrase,
        "expires_at": row["expires_at"],
        "candidate": normalized,
        "duplicates": duplicates,
        "import_mode": mode,
        "download_url": download_url or None,
        "writes_performed": False,
        "next_step": "Internal single-use execution token; do not expose it as a second approval step.",
    }


def claim_import_proposal(
    proposal_id: str, approval_phrase: str, *, path: Path | None = None,
) -> dict[str, Any]:
    """Atomically claim one approved proposal so retries cannot duplicate it."""
    target = path or _proposal_path()
    with _PROPOSAL_LOCK:
        payload = _load_proposals(target)
        row = payload["proposals"].get(str(proposal_id))
        if not isinstance(row, dict):
            raise KeyError("import proposal not found")
        if row.get("status") != "pending":
            raise ValueError(f"import proposal is already {row.get('status')}")
        if int(row.get("expires_at") or 0) < int(time.time()):
            row["status"] = "expired"
            _save_proposals(target, payload)
            raise ValueError("import proposal has expired")
        actual = hashlib.sha256(str(approval_phrase).encode()).hexdigest()
        if not secrets.compare_digest(actual, str(row.get("approval_phrase_sha256") or "")):
            raise ValueError("approval phrase does not match the proposal")
        row["status"] = "executing"
        row["execution_started_at"] = int(time.time())
        _save_proposals(target, payload)
        return dict(row)


def finish_import_proposal(
    proposal_id: str, status: str, result: Mapping[str, Any], *, path: Path | None = None,
) -> None:
    target = path or _proposal_path()
    with _PROPOSAL_LOCK:
        payload = _load_proposals(target)
        row = payload["proposals"].get(str(proposal_id))
        if isinstance(row, dict):
            row["status"] = status
            row["finished_at"] = int(time.time())
            row["result"] = dict(result)
            _save_proposals(target, payload)


def _item_type(candidate: Mapping[str, Any]) -> str:
    value = str(candidate.get("resource_type") or "").casefold()
    if any(token in value for token in ("journal", "article", "論文", "紀要")):
        return "journalArticle"
    if any(token in value for token in ("thesis", "dissertation", "学位")):
        return "thesis"
    if "conference" in value:
        return "conferencePaper"
    return "book" if any(token in value for token in ("book", "図書")) else "document"


def _identifier(candidate: Mapping[str, Any], name: str) -> str:
    value = (candidate.get("identifiers") or {}).get(name)
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value or "").strip()


def _zotero_items(row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = row["candidate"]
    collections = [row["collection_key"]] if row.get("collection_key") else []
    item_type = _item_type(candidate)
    parent = {
        "itemType": item_type,
        "title": candidate["title"],
        "creators": [{"creatorType": "author", "name": name} for name in candidate["creators"]],
        "date": candidate["date"],
        "url": candidate["landing_url"],
        "libraryCatalog": candidate["source"],
        "collections": collections,
        "tags": [{"tag": AI_ADDED_TAG}],
        "relations": {},
    }
    if item_type == "book":
        parent.update({
            "publisher": candidate["publisher"],
            "volume": str(candidate.get("volume") or ""),
            "ISBN": _identifier(candidate, "isbn"),
        })
    elif item_type == "journalArticle":
        parent.update({
            "publicationTitle": candidate["container_title"],
            "publisher": candidate["publisher"],
            # ``get`` keeps proposals created before these fields were added executable.
            "volume": str(candidate.get("volume") or ""),
            "issue": str(candidate.get("issue") or ""),
            "pages": str(candidate.get("pages") or ""),
            "DOI": _identifier(candidate, "doi"), "ISSN": _identifier(candidate, "issn"),
        })
    mode = str(row["import_mode"])
    has_file = mode in {"pdf", "epub"}
    extension = mode if has_file else ""
    content_type = {
        "pdf": "application/pdf", "epub": "application/epub+zip",
    }.get(mode, "text/html")
    attachment = {
        "itemType": "attachment",
        "linkMode": "imported_file" if has_file else "linked_url",
        "title": f"Full Text {mode.upper()}" if has_file else "Source URL",
        "url": row.get("download_url") or row.get("pdf_url") or candidate["landing_url"],
        "contentType": content_type,
        "filename": f"full-text.{extension}" if has_file else "",
        "tags": [], "collections": [], "relations": {},
    }
    return parent, attachment


def _created_key(payload: Mapping[str, Any]) -> str:
    successful = payload.get("successful") or {}
    item = successful.get("0") or successful.get(0) or {}
    key = str(item.get("key") or "") if isinstance(item, Mapping) else ""
    if not key:
        raise RuntimeError("Zotero created an item but did not return its key")
    return key


class ZoteroLocalWriter:
    """Prefer Zotero 10 local writes, with a configured Web API fallback."""

    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 120.0):
        self.base = local_api_base()
        self.prefix = local_api_prefix()
        self.transport = transport
        self.timeout = timeout
        self.server_id = ""
        self.write_key = ""
        self.remember_key = False
        self.web_api = False

    def _client(self, timeout: float | None = None) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=timeout or self.timeout, transport=self.transport)

    async def _server(self) -> None:
        async with self._client() as client:
            response = await client.get(f"{self.base}/", headers=zotero_api_headers())
        response.raise_for_status()
        self.server_id = str(response.headers.get("Zotero-Server-ID") or "")
        if not self.server_id:
            user_id = str(os.environ.get("ZOTERO_USER_ID") or "").strip()
            api_key = str(os.environ.get("ZOTERO_API_KEY") or "").strip()
            if user_id and api_key:
                self.base = "https://api.zotero.org"
                self.prefix = f"users/{user_id}"
                self.write_key = api_key
                self.remember_key = True
                self.web_api = True
                return
            version = str(response.headers.get("X-Zotero-Version") or "unknown")
            raise RuntimeError(
                f"Zotero {version} does not support Local API writes (Zotero 10+ required). "
                "Upgrade Zotero, or configure both ZOTERO_USER_ID and ZOTERO_API_KEY "
                "to use the Zotero Web API fallback."
            )
        self.write_key = _load_local_auth(self.server_id)
        self.remember_key = bool(self.write_key)

    async def _authorize(self) -> None:
        if not self.server_id:
            await self._server()
        if self.write_key:
            return
        if self.web_api:
            return
        try:
            async with self._client(timeout=min(self.timeout, 30.0)) as client:
                response = await client.post(
                    f"{self.base}/local/authorize",
                    headers=zotero_api_headers(**{
                        "Content-Type": "application/json", "Zotero-Server-ID": self.server_id,
                    }),
                    json={"appName": "zotero-local-rag"},
                )
        except httpx.ReadTimeout as exc:
            raise RuntimeError(
                "Zotero write authorization timed out. Unlock this Mac, run "
                "`uv run python scripts/authorize_zotero_local.py`, and choose Always Allow."
            ) from exc
        response.raise_for_status()
        payload = response.json()
        self.write_key = str(payload.get("key") or "")
        self.remember_key = bool(payload.get("remember"))
        if not self.write_key:
            raise RuntimeError("Zotero did not grant a local write key")
        if self.remember_key:
            _save_local_auth(self.server_id, self.write_key)

    async def authorize(self) -> dict[str, Any]:
        """Obtain or reuse local write authorization without changing the library."""
        if not self.server_id:
            await self._server()
        if not self.write_key:
            await self._authorize()
        return {
            "server_id": self.server_id,
            "remembered": self.remember_key,
            "web_api": self.web_api,
        }

    async def _write(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        extra_headers = dict(kwargs.pop("headers", {}))
        for attempt in range(2):
            previous_base = self.base
            if not self.write_key:
                await self._authorize()
            if self.base != previous_base and url.startswith(previous_base):
                url = self.base + url[len(previous_base):]
                url = url.replace(f"/{local_api_prefix()}/", f"/{self.prefix}/", 1)
            identity_headers = {"Zotero-Server-ID": self.server_id} if self.server_id else {}
            headers = zotero_api_headers(self.write_key, **identity_headers, **extra_headers)
            async with self._client() as client:
                response = await client.request(method, url, headers=headers, **kwargs)
            if response.status_code != 401 or attempt:
                response.raise_for_status()
                if not self.remember_key:
                    self.write_key = ""
                return response
            if not self.web_api:
                _clear_local_auth()
            self.write_key = ""
        raise RuntimeError("Zotero write authorization failed")

    async def create_items(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        token = secrets.token_hex(16)
        response = await self._write(
            "POST", f"{self.base}/{self.prefix}/items", json=items,
            headers={"Content-Type": "application/json", "Zotero-Write-Token": token},
        )
        payload = response.json()
        failed = payload.get("failed") or {}
        if failed:
            raise RuntimeError(f"Zotero rejected imported items: {failed}")
        return payload

    async def upload_file(self, attachment_key: str, file_path: Path) -> None:
        content = file_path.read_bytes()
        digest = hashlib.md5(content, usedforsecurity=False).hexdigest()
        endpoint = f"{self.base}/{self.prefix}/items/{attachment_key}/file"
        request = await self._write(
            "POST", endpoint,
            data={
                "md5": digest, "filename": file_path.name,
                "filesize": str(len(content)), "mtime": str(int(file_path.stat().st_mtime * 1000)),
            },
            headers={"If-None-Match": "*"},
        )
        payload = request.json()
        if payload.get("exists"):
            return
        upload_url = str(payload.get("url") or "")
        upload_key = str(payload.get("uploadKey") or "")
        if not upload_url or not upload_key:
            raise RuntimeError("Zotero file upload authorization was incomplete")
        prefix = str(payload.get("prefix") or "").encode()
        suffix = str(payload.get("suffix") or "").encode()
        async with self._client() as client:
            uploaded = await client.post(
                urljoin(self.base + "/", upload_url),
                content=prefix + content + suffix,
                headers={"Content-Type": payload.get("contentType") or "application/pdf"},
            )
        uploaded.raise_for_status()
        await self._write("POST", endpoint, data={"upload": upload_key}, headers={"If-None-Match": "*"})


def _validate_download(content: bytes, mode: str) -> None:
    if len(content) > MAX_FULLTEXT_BYTES:
        raise ValueError(f"candidate {mode.upper()} exceeds the import size limit")
    if mode == "pdf":
        if not content.startswith(b"%PDF-"):
            raise ValueError("candidate download is not a PDF")
        return
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            if archive.read("mimetype") != b"application/epub+zip":
                raise ValueError("candidate download has an invalid EPUB mimetype")
    except (KeyError, zipfile.BadZipFile) as exc:
        raise ValueError("candidate download is not an EPUB") from exc


async def download_candidate_file(row: Mapping[str, Any], destination: Path) -> Path:
    mode = str(row["import_mode"])
    url = str(row.get("download_url") or row.get("pdf_url") or "")
    for _hop in range(6):
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError(f"{mode.upper()} download URL and redirects must use HTTPS")
        async with httpx.AsyncClient(timeout=120, follow_redirects=False) as client:
            response = await client.get(url)
        if response.is_redirect:
            location = str(response.headers.get("location") or "")
            url = str(response.next_request.url) if response.next_request else urljoin(url, location)
            continue
        response.raise_for_status()
        _validate_download(response.content, mode)
        destination.write_bytes(response.content)
        return destination
    raise ValueError(f"candidate {mode.upper()} redirected too many times")


async def execute_import(row: Mapping[str, Any], *, writer: ZoteroLocalWriter | None = None) -> dict[str, Any]:
    parent, attachment = _zotero_items(row)
    active_writer = writer or ZoteroLocalWriter()
    temporary: Path | None = None
    try:
        if row["import_mode"] in {"pdf", "epub"}:
            content_type = (
                "application/pdf" if row["import_mode"] == "pdf" else "application/epub+zip"
            )
            suffix = mimetypes.guess_extension(content_type) or f".{row['import_mode']}"
            handle, name = tempfile.mkstemp(prefix="zotero-import-", suffix=suffix)
            os.close(handle)
            temporary = Path(name)
            await download_candidate_file(row, temporary)
            attachment["filename"] = temporary.name
        parent_key = _created_key(await active_writer.create_items([parent]))
        attachment["parentItem"] = parent_key
        attachment_key = _created_key(await active_writer.create_items([attachment]))
        if temporary is not None:
            await active_writer.upload_file(attachment_key, temporary)
        return {
            "item_key": parent_key,
            "attachment_key": attachment_key,
            "import_mode": row["import_mode"],
            "file_uploaded": temporary is not None,
            "writes_performed": True,
        }
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


__all__ = [
    "ZoteroLocalWriter", "claim_import_proposal", "create_import_proposal",
    "download_candidate_file", "execute_import", "finish_import_proposal",
]
