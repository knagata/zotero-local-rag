"""Conservative chapter-work promotion and translation-link detection."""
from __future__ import annotations

import os
import re
from typing import Any

import httpx

try:
    from .db_relations import get_db_connection, get_section_summaries, resolve_work, save_work_link
    from .ndl_client import search_ndl
except ImportError:  # pragma: no cover
    from db_relations import get_db_connection, get_section_summaries, resolve_work, save_work_link
    from ndl_client import search_ndl


ORIGINAL_TITLE_RE = re.compile(
    r"(?:original\s+title|original\s+work|原題|原書名)\s*[:：]\s*([^\n]+)", re.I,
)


def plan_chapter_promotion(item_key: str) -> dict[str, Any]:
    """Promote only volumes with at least two distinct explicit chapter authors."""
    sections = get_section_summaries(item_key)
    authored = [row for row in sections if str(row.get("chapter_authors") or "").strip()]
    distinct_authors = {str(row["chapter_authors"]).strip().casefold() for row in authored}
    eligible = len(authored) >= 2 and len(distinct_authors) >= 2
    return {
        "item_key": item_key, "eligible": eligible,
        "reason": "multiple distinct chapter authors" if eligible else "insufficient distinct chapter-author evidence",
        "chapters": [{
            "section_id": row["section_id"], "title": row.get("chapter") or row["section_id"],
            "authors": row.get("chapter_authors"),
            "first_publication_note": row.get("first_publication_note"),
        } for row in authored],
    }


def promote_chapters(item_key: str, *, dry_run: bool = True) -> dict[str, Any]:
    plan = plan_chapter_promotion(item_key)
    if dry_run or not plan["eligible"]:
        return {**plan, "status": "dry_run" if dry_run else "skipped"}
    connection = get_db_connection()
    try:
        parent_row = connection.execute(
            "SELECT * FROM works WHERE zotero_item_key = ? AND container_work_id IS NULL LIMIT 1",
            (item_key,),
        ).fetchone()
    finally:
        connection.close()
    parent_id = int(parent_row["work_id"]) if parent_row else resolve_work(zotero_item_key=item_key)
    work_ids = []
    for chapter in plan["chapters"]:
        work_ids.append(resolve_work(
            zotero_item_key=item_key, title=chapter["title"], authors=chapter["authors"],
            work_type="chapter", container_work_id=parent_id, section_id=chapter["section_id"],
        ))
    return {**plan, "status": "promoted", "parent_work_id": parent_id, "chapter_work_ids": work_ids}


def _zotero_item(item_key: str) -> dict[str, Any]:
    base = (os.environ.get("ZOTERO_LOCAL_API_BASE") or "http://127.0.0.1:23119/api").rstrip("/")
    prefix = (os.environ.get("ZOTERO_LOCAL_API_PREFIX") or "users/0").strip("/")
    headers = {"Zotero-API-Version": os.environ.get("ZOTERO_API_VERSION", "3")}
    if os.environ.get("ZOTERO_API_KEY"):
        headers["Zotero-API-Key"] = os.environ["ZOTERO_API_KEY"]
    response = httpx.get(f"{base}/{prefix}/items/{item_key}", headers=headers, timeout=10)
    response.raise_for_status()
    payload = response.json()
    return payload.get("data", payload)


def detect_translation(item_key: str, *, dry_run: bool = True) -> dict[str, Any]:
    """Find an explicit/NDL original title; never link an LLM-only guess."""
    data = _zotero_item(item_key)
    title = str(data.get("title") or "").strip()
    extra = str(data.get("extra") or "")
    match = ORIGINAL_TITLE_RE.search(extra)
    original_title = match.group(1).strip() if match else ""
    source = "zotero-extra" if original_title else None
    ndl_record = None
    if not original_title and title:
        try:
            records = search_ndl(title, maximum_records=5)
        except Exception:
            records = []
        for record in records:
            alternatives = record.get("alternative_titles") or []
            original_title = next((
                candidate for candidate in alternatives
                if sum(char.isascii() and char.isalpha() for char in candidate) >= max(4, len(candidate) // 3)
            ), "")
            if original_title:
                ndl_record, source = record, "ndl"
                break
    creators = data.get("creators") or []
    authors = "; ".join(
        " ".join(filter(None, [creator.get("firstName"), creator.get("lastName")])) or creator.get("name", "")
        for creator in creators if isinstance(creator, dict)
    )
    result = {
        "item_key": item_key, "title": title, "original_title": original_title or None,
        "source": source, "status": "candidate" if original_title else "not_found",
    }
    if dry_run or not original_title:
        return {**result, "dry_run": dry_run}
    derived_id = resolve_work(
        zotero_item_key=item_key, title=title, authors=authors,
        year=int(str(data.get("date") or "")[:4]) if str(data.get("date") or "")[:4].isdigit() else None,
        lang=data.get("language"),
    )
    original_id = resolve_work(
        title=original_title, authors=authors,
        ndl_bibid=ndl_record.get("ndl_bibid") if ndl_record else None,
        isbn=ndl_record.get("isbn") if ndl_record else None,
        year=ndl_record.get("year") if ndl_record else None,
    )
    link_id = save_work_link(
        derived_id, original_id, "translation_of",
        confidence=0.95 if source == "ndl" else 1.0, source=source,
    )
    return {**result, "status": "linked", "derived_work_id": derived_id, "original_work_id": original_id, "link_id": link_id}


__all__ = ["detect_translation", "plan_chapter_promotion", "promote_chapters"]
