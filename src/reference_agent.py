"""Reference-section detection, structured extraction, and canonical resolution."""
from __future__ import annotations

import hashlib
import json
import os
import re
from difflib import SequenceMatcher
from typing import Any

import httpx

try:
    from .cinii_client import search_cinii
    from .chunk_store import get_item_chunks
    from .db_relations import (
        get_canonical_work_id, get_resolver_cache, normalize_work_title, resolve_work, save_resolver_cache,
        save_work_edge,
    )
    from .llm_client import LLMError, get_llm
    from .ndl_client import search_ndl
except ImportError:  # pragma: no cover
    from cinii_client import search_cinii
    from chunk_store import get_item_chunks
    from db_relations import get_canonical_work_id, get_resolver_cache, normalize_work_title, resolve_work, save_resolver_cache, save_work_edge
    from llm_client import LLMError, get_llm
    from ndl_client import search_ndl


REFERENCE_HEADING = re.compile(
    r"^\s*(参考文献|引用文献|文献一覧|文献|references|bibliography|works cited|literatur)\s*$",
    re.I,
)
YEAR_RE = re.compile(r"(?:\(|（)?((?:18|19|20)\d{2})(?:\)|）)?")

REFERENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"references": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "raw": {"type": "string"}, "authors": {"type": "array", "items": {"type": "string"}},
            "title": {"type": "string"}, "container": {"type": ["string", "null"]},
            "year": {"type": ["integer", "null"]}, "volume": {"type": ["string", "null"]},
            "pages": {"type": ["string", "null"]}, "publisher": {"type": ["string", "null"]},
            "doi": {"type": ["string", "null"]}, "isbn": {"type": ["string", "null"]},
            "lang": {"type": ["string", "null"]}, "type": {"type": ["string", "null"]},
            "translation_note": {"type": ["string", "null"]},
        },
        "required": ["raw", "authors", "title"],
    }}},
    "required": ["references"],
}


def detect_reference_sections(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return likely bibliography chunks, preferring explicit headings."""
    starts: list[int] = []
    for index, chunk in enumerate(chunks):
        first_line = str(chunk.get("text") or "").strip().splitlines()[0] if chunk.get("text") else ""
        if REFERENCE_HEADING.match(first_line[:100]):
            starts.append(index)
    if starts:
        selected: list[dict[str, Any]] = []
        for start in starts:
            selected.extend(chunks[start : start + 200])
        seen: set[str] = set()
        return [chunk for chunk in selected if not (chunk["id"] in seen or seen.add(chunk["id"]))]
    tail_start = max(0, int(len(chunks) * 0.85))
    return chunks[tail_start:]


def parse_reference_lines(text: str) -> list[dict[str, Any]]:
    """Cheap fallback parser for one-reference-per-line bibliographies."""
    results: list[dict[str, Any]] = []
    for raw in text.splitlines():
        line = " ".join(raw.split()).strip()
        match = YEAR_RE.search(line)
        if not match or len(line) < 20:
            continue
        before = line[: match.start()].strip(" ,.;、。")
        after = line[match.end() :].strip(" ,.;、。")
        if not before or len(after) < 4:
            continue
        title = re.split(r"[。.;]", after, maxsplit=1)[0].strip() or after
        results.append({
            "raw": line, "authors": [before], "title": title,
            "year": int(match.group(1)), "doi": None, "isbn": None,
            "container": None, "lang": None, "type": None,
            "translation_note": None,
        })
    return results


def _validate_references(rows: list[Any], source_text: str) -> list[dict[str, Any]]:
    normalized_source = " ".join(source_text.split())
    valid: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw = " ".join(str(row.get("raw") or "").split())
        title = str(row.get("title") or "").strip()
        if not raw or not title or raw not in normalized_source:
            continue
        row = dict(row)
        row["raw"] = raw
        valid.append(row)
    return valid


def extract_references(text: str, *, use_llm: bool = True) -> tuple[list[dict[str, Any]], str]:
    if use_llm:
        prompt = (
            "以下の参考文献・注記候補から各文献をJSONで構造化してください。rawは入力に存在する"
            "原文をそのまま返し、推測で文献を追加しないでください。邦訳併記はtranslation_noteへ"
            "保存してください。\n\n" + text[:30000]
        )
        try:
            client = get_llm("extract")
            result = client.generate_json(prompt, schema=REFERENCE_SCHEMA, timeout=300)
            return _validate_references(result.get("references") or [], text), f"{client.provider}:{client.model}"
        except LLMError:
            pass
    return parse_reference_lines(text), "heuristic"


def _item_excluded(item_key: str) -> tuple[bool, str | None]:
    tags = {tag.strip().casefold() for tag in os.environ.get("EXTRACT_EXCLUDE_TAGS", "").split(",") if tag.strip()}
    if not tags:
        return False, None
    base = (os.environ.get("ZOTERO_LOCAL_API_BASE") or "http://127.0.0.1:23119/api").rstrip("/")
    prefix = (os.environ.get("ZOTERO_LOCAL_API_PREFIX") or "users/0").strip("/")
    headers = {"Zotero-API-Version": os.environ.get("ZOTERO_API_VERSION", "3")}
    if os.environ.get("ZOTERO_API_KEY"):
        headers["Zotero-API-Key"] = os.environ["ZOTERO_API_KEY"]
    try:
        response = httpx.get(f"{base}/{prefix}/items/{item_key}", headers=headers, timeout=5)
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", payload)
        item_tags = {str(tag.get("tag") or "").strip().casefold() for tag in data.get("tags", [])}
        matched = sorted(tags & item_tags)
        return bool(matched), ", ".join(matched) if matched else None
    except Exception as exc:
        return True, f"could not verify exclusion tags: {exc}"


def _candidate_score(reference: dict[str, Any], candidate: dict[str, Any]) -> float:
    title_score = SequenceMatcher(
        None, normalize_work_title(reference.get("title")), normalize_work_title(candidate.get("title"))
    ).ratio()
    ref_year, candidate_year = reference.get("year"), candidate.get("year")
    if ref_year and candidate_year and abs(int(ref_year) - int(candidate_year)) > 1:
        title_score *= 0.7
    ref_authors = reference.get("authors") or []
    if isinstance(ref_authors, list):
        ref_authors = "; ".join(str(value) for value in ref_authors)
    candidate_authors = str(candidate.get("authors") or "")
    if ref_authors and candidate_authors:
        ref_tokens = set(re.findall(r"[\w\u3040-\u30ff\u3400-\u9fff]+", str(ref_authors).casefold()))
        candidate_tokens = set(re.findall(r"[\w\u3040-\u30ff\u3400-\u9fff]+", candidate_authors.casefold()))
        token_score = len(ref_tokens & candidate_tokens) / max(len(ref_tokens), 1)
        sequence_score = SequenceMatcher(
            None, normalize_work_title(str(ref_authors)), normalize_work_title(candidate_authors)
        ).ratio()
        if max(token_score, sequence_score) < 0.45:
            title_score *= 0.7
    return title_score


def resolve_reference(reference: dict[str, Any]) -> dict[str, Any]:
    """Resolve through stable IDs, CiNii, NDL, then retain an unresolved work."""
    cache_key = hashlib.sha256(json.dumps(reference, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    cached = get_resolver_cache(cache_key, "cascade")
    if cached:
        result = json.loads(cached)
        manifestation_id = int(result.get("manifestation_work_id") or result["work_id"])
        result["work_id"] = get_canonical_work_id(manifestation_id)
        if result["work_id"] != manifestation_id:
            result["manifestation_work_id"] = manifestation_id
        return result
    authors = reference.get("authors") or []
    authors_text = "; ".join(authors) if isinstance(authors, list) else str(authors)
    base = {
        "title": reference.get("title"), "authors": authors_text,
        "year": reference.get("year"), "doi": reference.get("doi"),
        "isbn": reference.get("isbn"), "lang": reference.get("lang"),
        "container": reference.get("container"), "work_type": reference.get("type"),
    }
    external_error = False
    if reference.get("doi") or reference.get("isbn"):
        result = {"work_id": resolve_work(**base), "confidence": 1.0, "source": "identifier"}
    else:
        candidates: list[dict[str, Any]] = []
        for search in (search_cinii, search_ndl):
            try:
                candidates.extend(search(
                    str(reference.get("title") or ""), author=authors_text.split(";")[0],
                    year=reference.get("year"),
                ))
            except Exception:
                external_error = True
                continue
        scored = sorted(
            ((_candidate_score(reference, candidate), candidate) for candidate in candidates),
            key=lambda pair: pair[0], reverse=True,
        )
        if scored and scored[0][0] >= 0.85:
            confidence, candidate = scored[0]
            result = {
                "work_id": resolve_work(**{**base, **{
                    key: candidate.get(key) for key in ("doi", "isbn", "cinii_crid", "ndl_bibid")
                }}),
                "confidence": confidence, "source": candidate.get("source"),
            }
        else:
            result = {"work_id": resolve_work(**base), "confidence": 0.5, "source": "unresolved"}
    manifestation_id = int(result["work_id"])
    result["work_id"] = get_canonical_work_id(manifestation_id)
    if result["work_id"] != manifestation_id:
        result["manifestation_work_id"] = manifestation_id
    # Do not turn an outage/rate-limit result into a permanent unresolved cache hit.
    # A successful identifier or external match remains safe to cache even if another
    # provider in the cascade was unavailable.
    if result.get("source") != "unresolved" or not external_error:
        save_resolver_cache(cache_key, "cascade", json.dumps(result, ensure_ascii=False))
    return result


def extract_references_for_item(
    item_key: str, *, dry_run: bool = True, use_llm: bool = True,
) -> dict[str, Any]:
    chunks = get_item_chunks(item_key)
    if not chunks:
        return {"item_key": item_key, "status": "empty", "references": []}
    if use_llm:
        excluded, reason = _item_excluded(item_key)
        if excluded:
            return {"item_key": item_key, "status": "excluded", "reason": reason, "references": []}
    candidates = detect_reference_sections(chunks)
    source_text = "\n".join(chunk["text"] for chunk in candidates if chunk.get("text"))[:30000]
    references, model = extract_references(source_text, use_llm=use_llm)
    metadata = chunks[0].get("metadata", {})
    citing_work = None
    if not dry_run:
        citing_work = resolve_work(
            zotero_item_key=item_key, title=metadata.get("title"), authors=metadata.get("creators"),
            year=metadata.get("year"), lang=metadata.get("lang"),
        )
    output = []
    for reference in references:
        if dry_run:
            row = {**reference, "resolution": "pending_commit"}
        else:
            resolved = resolve_reference(reference)
            row = {**reference, **resolved}
            assert citing_work is not None
            row["edge_id"] = save_work_edge(
                citing_work, resolved["work_id"], source="llm-pdf" if model != "heuristic" else "heuristic-pdf",
                confidence=float(resolved["confidence"]), raw_reference=reference["raw"],
            )
        output.append(row)
    return {
        "item_key": item_key, "status": "dry_run" if dry_run else "saved",
        "model": model, "candidate_chunks": len(candidates), "references": output,
    }


__all__ = [
    "detect_reference_sections", "extract_references", "extract_references_for_item",
    "parse_reference_lines", "resolve_reference",
]
