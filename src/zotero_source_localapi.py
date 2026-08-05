from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, AsyncIterator

import httpx


@dataclass(frozen=True)
class ZoteroAttachment:
    """Normalized attachment record for indexing.

    Note: `pdf_path` is kept for backward compatibility but may point to a non-PDF file
    (e.g., an HTML snapshot) when `source_type != "pdf"`.
    """

    attachmentKey: str
    parentItemKey: Optional[str]
    title: Optional[str]
    year: Optional[int]
    creators: Optional[List[str]]
    pdf_path: str
    source_type: str  # "pdf" | "html" | "epub"
    contentType: Optional[str] = None
    filename: Optional[str] = None
    language: Optional[str] = None
    parentItemType: Optional[str] = None
    tags: tuple[str, ...] = ()
    parentTags: tuple[str, ...] = ()


def _normalized_tags(value: Any) -> tuple[str, ...]:
    """Return Zotero tag names in a stable, case-insensitive form."""
    if not isinstance(value, list):
        return ()
    tags = {
        str(row.get("tag") or "").strip().casefold()
        for row in value
        if isinstance(row, dict) and str(row.get("tag") or "").strip()
    }
    return tuple(sorted(tags))


def classify_attachment_source_type(content_type: Optional[str], filename: Optional[str]) -> Optional[str]:
    """Which of pdf/html/epub an attachment's contentType/filename implies, or None.

    The single definition of "eligible attachment type" for this project.
    Extracted so a reconciliation check (comparing what Zotero has against
    what the manifest has) can classify eligibility identically to the
    ingestion path itself, rather than keeping a second copy that could
    silently drift from what actually gets indexed.
    """
    fn_l = (filename or "").lower()
    if (content_type == "application/pdf") or fn_l.endswith(".pdf"):
        return "pdf"
    if (content_type in ("text/html", "application/xhtml+xml")) or fn_l.endswith((".html", ".htm")):
        return "html"
    if (content_type == "application/epub+zip") or fn_l.endswith(".epub"):
        return "epub"
    return None


class ZoteroLocalAPI:
    """Minimal client for the Zotero Local HTTP API.

    Zotero local API base typically:
      http://127.0.0.1:23119/api/users/0/...

    This client only implements the endpoints needed by the indexer:
    - list items (paged)
    - read a single item
    - download attachment file (fallback)
    - list notes (paged)

    Environment variables:
    - ZOTERO_LOCAL_API_BASE: default http://127.0.0.1:23119/api
    - ZOTERO_LOCAL_API_PREFIX: default users/0
    - ZOTERO_API_KEY: optional
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get("ZOTERO_LOCAL_API_BASE")
            or "http://127.0.0.1:23119/api"
        ).rstrip("/")
        self.user_prefix = (os.environ.get("ZOTERO_LOCAL_API_PREFIX") or "users/0").strip("/")
        self.api_key = api_key or os.environ.get("ZOTERO_API_KEY")
        self.timeout = timeout

        self.headers: Dict[str, str] = {
            "Accept": "application/json",
            "Zotero-API-Version": os.environ.get("ZOTERO_API_VERSION", "3"),
        }
        if self.api_key:
            self.headers["Zotero-API-Key"] = self.api_key

        self._parent_cache: Dict[str, Dict[str, Any]] = {}
        self._last_total_results: Optional[int] = None

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{self.user_prefix}/{path.lstrip('/')}"

    async def _get_json(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        async with httpx.AsyncClient(timeout=timeout or self.timeout) as client:
            r = await client.get(self._url(path), headers=self.headers, params=params)
            r.raise_for_status()
            total = r.headers.get("Total-Results")
            try:
                self._last_total_results = int(total) if total is not None else None
            except (TypeError, ValueError):
                self._last_total_results = None
            return r.json()

    async def list_pdf_attachments(
        self, limit: int = 200, *, require_complete: bool = False,
    ) -> List[Dict[str, Any]]:
        """Fetch items and return a de-duplicated list (by Zotero item key).

        Note: despite the name, we use this to list *attachments* in general (pdf/html).
        """
        seen: Dict[str, Dict[str, Any]] = {}
        all_seen_keys: set[str] = set()
        expected_total: Optional[int] = None
        raw_count = 0
        start = 0
        use_itemtype_filter = True
        while True:
            try:
                batch = await self._get_json(
                    "items",
                    params={
                        "format": "json",
                        "limit": limit,
                        "start": start,
                        **({"itemType": "attachment"} if use_itemtype_filter else {}),
                    },
                )
            except httpx.HTTPStatusError as e:
                if use_itemtype_filter and getattr(e.response, "status_code", None) in (400, 404):
                    use_itemtype_filter = False
                    batch = await self._get_json(
                        "items",
                        params={
                            "format": "json",
                            "limit": limit,
                            "start": start,
                        },
                    )
                else:
                    raise

            page_total = self._last_total_results
            if page_total is not None:
                if expected_total is None:
                    expected_total = page_total
                elif page_total != expected_total:
                    raise RuntimeError(
                        f"Zotero attachment inventory total changed during pagination: "
                        f"{expected_total} -> {page_total}"
                    )
            if not isinstance(batch, list) or not batch:
                break

            raw_count += len(batch)
            for raw in batch:
                if not isinstance(raw, dict):
                    continue
                k = raw.get("key")
                if not k and isinstance(raw.get("data"), dict):
                    k = raw["data"].get("key")
                if not k:
                    continue
                all_seen_keys.add(str(k))
                seen[str(k)] = raw

            start += len(batch)
            if len(batch) < limit:
                break

        if require_complete:
            if expected_total is None:
                raise RuntimeError(
                    "Zotero attachment inventory has no Total-Results header; "
                    "refusing a clean rebuild without a completeness proof"
                )
            if raw_count != expected_total or len(all_seen_keys) != expected_total:
                raise RuntimeError(
                    "Incomplete Zotero attachment inventory: "
                    f"expected={expected_total} rows={raw_count} unique={len(all_seen_keys)}"
                )
        return list(seen.values())

    async def get_item(self, item_key: str) -> Dict[str, Any]:
        if item_key in self._parent_cache:
            return self._parent_cache[item_key]
        obj = await self._get_json(f"items/{item_key}")
        if isinstance(obj, dict):
            self._parent_cache[item_key] = obj
        return obj

    @staticmethod
    def _unwrap_item(raw: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        key = raw.get("key")
        data = raw.get("data") if isinstance(raw.get("data"), dict) else raw
        if not key and isinstance(data, dict):
            key = data.get("key")
        if not key:
            raise ValueError(f"Cannot determine item key from record: {raw.keys()}")
        return str(key), data

    async def fetch_attachment_file_to_cache(self, attachment_key: str, cache_path: str) -> str:
        """Fallback: GET /items/<attachmentKey>/file and write to cache_path."""
        url = self._url(f"items/{attachment_key}/file")
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.get(url, headers=self.headers)
            r.raise_for_status()
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                f.write(r.content)
        return cache_path

    @staticmethod
    def _extract_parent_meta(parent_raw: Dict[str, Any]) -> Tuple[Optional[str], Optional[int], Optional[List[str]]]:
        _, pd = ZoteroLocalAPI._unwrap_item(parent_raw)

        title = pd.get("title")

        year: Optional[int] = None
        date = pd.get("date")
        if isinstance(date, str) and len(date) >= 4 and date[:4].isdigit():
            year = int(date[:4])

        creators_out: List[str] = []
        creators = pd.get("creators")
        if isinstance(creators, list):
            for c in creators:
                if not isinstance(c, dict):
                    continue
                last = (c.get("lastName") or "").strip()
                first = (c.get("firstName") or "").strip()
                name = (f"{last}, {first}".strip(", ")).strip()
                if name:
                    creators_out.append(name)

        return title, year, creators_out or None

    @staticmethod
    def resolve_pdf_path_from_attachment(
        attachment_key: str,
        attachment_data: Dict[str, Any],
        zotero_data_dir: Optional[str],
    ) -> Optional[str]:
        """Resolve a local file path for a Zotero attachment.

        Supports:
        - absolute/~/ path in attachment_data['path']
        - storage:<filename> pattern (common for Zotero internal storage)
        - filename fallback under {ZOTERO_DATA_DIR}/storage/{attachmentKey}/
        """
        path_field = attachment_data.get("path")
        filename = attachment_data.get("filename")

        if isinstance(path_field, str) and (path_field.startswith("/") or path_field.startswith("~")):
            p = os.path.expanduser(path_field)
            if os.path.exists(p):
                return p

        if isinstance(path_field, str) and path_field.startswith("storage:") and zotero_data_dir:
            name = path_field.split("storage:", 1)[1].lstrip("/")
            if (not name) and isinstance(filename, str):
                name = filename
            if name:
                candidate = os.path.join(zotero_data_dir, "storage", attachment_key, name)
                if os.path.exists(candidate):
                    return candidate

        if zotero_data_dir and isinstance(filename, str) and filename:
            candidate = os.path.join(zotero_data_dir, "storage", attachment_key, filename)
            if os.path.exists(candidate):
                return candidate

        return None

    async def iter_normalized_attachments(
        self,
        zotero_data_dir: Optional[str],
        pdf_cache_dir: str,
        collection_key: Optional[str] = None,
        require_complete: bool = False,
    ) -> AsyncIterator[ZoteroAttachment]:
        """Yield normalized PDF/HTML snapshot attachments with resolved local path.

        NOTE: async generator; consume via `async for`.
        """
        raw_atts = await self.list_pdf_attachments(require_complete=require_complete)

        for raw in raw_atts:
            att_key, ad = self._unwrap_item(raw)

            if ad.get("itemType") != "attachment":
                continue

            ct = ad.get("contentType")
            fn = (ad.get("filename") or "")
            source_type = classify_attachment_source_type(ct, fn)
            if source_type is None:
                continue

            parent_key = ad.get("parentItem")

            parent_title = None
            parent_year = None
            parent_creators = None
            parent_language = None
            parent_item_type = None
            parent_tags: tuple[str, ...] = ()

            if isinstance(parent_key, str) and parent_key:
                parent_obj = await self.get_item(parent_key)
                parent_title, parent_year, parent_creators = self._extract_parent_meta(parent_obj)
                _, parent_data = self._unwrap_item(parent_obj)
                parent_language = parent_data.get("language")
                parent_item_type = parent_data.get("itemType")
                parent_tags = _normalized_tags(parent_data.get("tags"))

                if collection_key:
                    _, pd = self._unwrap_item(parent_obj)
                    cols = pd.get("collections")
                    if not (isinstance(cols, list) and collection_key in cols):
                        continue

            link_mode = str(ad.get("linkMode") or "")
            if link_mode == "linked_url":
                # A linked_url attachment is a pointer to an external URL --
                # Zotero never stores a local file for it, so both the local
                # path resolution and the Local API file download below were
                # guaranteed to fail every time, and did so silently (`continue`
                # inside a bare `except Exception`, printed only with
                # DEBUG_ZOTERO_LOCALAPI=1). One such attachment (75QYJJYK) was
                # the single Zotero-eligible attachment absent from the
                # manifest with no record anywhere of why (2026-07-28). It is
                # skipped explicitly here, with a visible reason, rather than
                # attempted and swallowed.
                print(
                    f"[WARN] Skipping linked_url attachment (no local file exists to index): "
                    f"attachment={att_key} parent={parent_key}",
                    file=sys.stderr,
                )
                continue

            resolved = self.resolve_pdf_path_from_attachment(att_key, ad, zotero_data_dir)
            if not resolved:
                # Local API file download fallback
                if source_type == "pdf":
                    cache_ext = ".pdf"
                elif source_type == "epub":
                    cache_ext = ".epub"
                else:
                    cache_ext = ".html"
                cache_path = os.path.join(pdf_cache_dir, f"{att_key}{cache_ext}")
                try:
                    resolved = await self.fetch_attachment_file_to_cache(att_key, cache_path)
                except Exception as e:
                    # Always visible, not gated behind a debug flag: a silent
                    # skip here is exactly how 75QYJJYK disappeared with no
                    # trace in the manifest, Chroma, or any log (2026-07-28).
                    print(
                        f"[WARN] Failed to resolve or download attachment file: "
                        f"attachment={att_key} parent={parent_key} err={e}",
                        file=sys.stderr,
                    )
                    raise RuntimeError(
                        f"Eligible attachment {att_key} has no readable local file"
                    ) from e

            yield ZoteroAttachment(
                attachmentKey=att_key,
                parentItemKey=parent_key if isinstance(parent_key, str) else None,
                title=parent_title,
                year=parent_year,
                creators=parent_creators,
                pdf_path=resolved,
                source_type=source_type or "pdf",
                contentType=ct if isinstance(ct, str) else None,
                filename=fn if isinstance(fn, str) else None,
                language=parent_language if isinstance(parent_language, str) else None,
                parentItemType=parent_item_type if isinstance(parent_item_type, str) else None,
                tags=_normalized_tags(ad.get("tags")),
                parentTags=parent_tags,
            )

    async def list_normalized_attachments(
        self,
        zotero_data_dir: Optional[str],
        pdf_cache_dir: str,
        collection_key: Optional[str] = None,
        require_complete: bool = False,
    ) -> List[ZoteroAttachment]:
        out: List[ZoteroAttachment] = []
        async for a in self.iter_normalized_attachments(
            zotero_data_dir=zotero_data_dir,
            pdf_cache_dir=pdf_cache_dir,
            collection_key=collection_key,
            require_complete=require_complete,
        ):
            out.append(a)
        return out

    async def list_notes(
        self, collection_key: Optional[str] = None, limit: int = 200, *,
        require_complete: bool = False,
    ) -> List[Dict[str, Any]]:
        """List note items (HTML) with parent bibliographic metadata."""
        seen: Dict[str, Dict[str, Any]] = {}
        all_seen_keys: set[str] = set()
        expected_total: Optional[int] = None
        raw_count = 0
        start = 0
        use_itemtype_filter = True
        while True:
            try:
                batch = await self._get_json(
                    "items",
                    params={
                        "format": "json",
                        "limit": limit,
                        "start": start,
                        **({"itemType": "note"} if use_itemtype_filter else {}),
                    },
                )
            except httpx.HTTPStatusError as e:
                if use_itemtype_filter and getattr(e.response, "status_code", None) in (400, 404):
                    use_itemtype_filter = False
                    batch = await self._get_json(
                        "items",
                        params={
                            "format": "json",
                            "limit": limit,
                            "start": start,
                        },
                    )
                else:
                    raise

            page_total = self._last_total_results
            if page_total is not None:
                if expected_total is None:
                    expected_total = page_total
                elif page_total != expected_total:
                    raise RuntimeError(
                        f"Zotero note inventory total changed during pagination: "
                        f"{expected_total} -> {page_total}"
                    )
            if not isinstance(batch, list) or not batch:
                break

            raw_count += len(batch)
            for raw in batch:
                if not isinstance(raw, dict):
                    continue
                data = raw.get("data") if isinstance(raw.get("data"), dict) else raw
                raw_key = raw.get("key") or (data.get("key") if isinstance(data, dict) else None)
                if raw_key:
                    all_seen_keys.add(str(raw_key))
                if not isinstance(data, dict) or data.get("itemType") != "note":
                    continue
                k = raw.get("key") or data.get("key")
                if not k:
                    continue
                seen[str(k)] = raw

            start += len(batch)
            if len(batch) < limit:
                break

        if require_complete:
            if expected_total is None:
                raise RuntimeError(
                    "Zotero note inventory has no Total-Results header; "
                    "refusing a clean rebuild without a completeness proof"
                )
            if raw_count != expected_total or len(all_seen_keys) != expected_total:
                raise RuntimeError(
                    "Incomplete Zotero note inventory: "
                    f"expected={expected_total} rows={raw_count} unique={len(all_seen_keys)}"
                )
        out: List[Dict[str, Any]] = []
        for raw in seen.values():
            note_key, nd = self._unwrap_item(raw)
            parent_key = nd.get("parentItem")

            parent_title = None
            parent_year = None
            parent_creators = None
            parent_language = None

            if isinstance(parent_key, str) and parent_key:
                parent_obj = await self.get_item(parent_key)
                parent_title, parent_year, parent_creators = self._extract_parent_meta(parent_obj)
                _, parent_data = self._unwrap_item(parent_obj)
                parent_language = parent_data.get("language")

                if collection_key:
                    _, pd = self._unwrap_item(parent_obj)
                    cols = pd.get("collections")
                    if not (isinstance(cols, list) and collection_key in cols):
                        continue

            note_html = nd.get("note") or ""
            ver = raw.get("version")
            if not isinstance(ver, int):
                ver = None

            out.append(
                {
                    "noteKey": note_key,
                    "parentItemKey": parent_key if isinstance(parent_key, str) else None,
                    "note_html": note_html if isinstance(note_html, str) else "",
                    "title": parent_title,
                    "year": parent_year,
                    "creators": parent_creators,
                    "language": parent_language if isinstance(parent_language, str) else None,
                    "version": ver,
                }
            )

        return out
