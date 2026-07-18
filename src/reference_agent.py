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
    from .crossref_client import search_crossref
    from .chunk_store import get_item_chunks
    from .db_relations import (
        apply_reference_metadata_resolution, apply_reference_metadata_resolutions,
        get_canonical_work_id,
        get_reference_review_candidates, get_resolver_cache,
        mark_reference_review_committed, normalize_work_title, resolve_work, save_resolver_cache,
        save_work_edge,
    )
    from .llm_client import LLMError, get_llm
    from .ndl_client import search_ndl
    from .reference_text import (
        is_compound_reference, is_short_form_reference, normalize_reference_text,
        strip_unicode_format_characters,
    )
except ImportError:  # pragma: no cover
    from cinii_client import search_cinii
    from crossref_client import search_crossref
    from chunk_store import get_item_chunks
    from db_relations import apply_reference_metadata_resolution, apply_reference_metadata_resolutions, get_canonical_work_id, get_reference_review_candidates, get_resolver_cache, mark_reference_review_committed, normalize_work_title, resolve_work, save_resolver_cache, save_work_edge
    from llm_client import LLMError, get_llm
    from ndl_client import search_ndl
    from reference_text import is_compound_reference, is_short_form_reference, normalize_reference_text, strip_unicode_format_characters


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
            "contributors": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string", "enum": [
                        "author", "editor", "translator", "interviewer", "interviewee", "other",
                    ]},
                },
                "required": ["name", "role"],
                "additionalProperties": False,
            }},
        },
        "required": [
            "raw", "authors", "title", "container", "year", "volume", "pages",
            "publisher", "doi", "isbn", "lang", "type", "translation_note", "contributors",
        ],
        "additionalProperties": False,
    }}},
    "required": ["references"],
    "additionalProperties": False,
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


def extract_references(
    text: str, *, use_llm: bool = True, fallback_heuristic: bool = True,
) -> tuple[list[dict[str, Any]], str]:
    if use_llm:
        prompt = (
            "以下の参考文献・注記候補から各文献をJSONで構造化してください。rawは入力に存在する"
            "原文をそのまま返し、推測で文献を追加しないでください。邦訳併記はtranslation_noteへ"
            "保存してください。新聞・雑誌の欄名やコーナー名を著作タイトルにしないでください。"
            "著者・編者・翻訳者・聞き手・被面接者はcontributorsで役割を区別してください。"
            "――、同上等の著者略記は、入力中で直前の著者を確認できる場合だけ同一人物として"
            "展開してください。\n\n" + text[:30000]
        )
        try:
            client = get_llm("extract")
            result = client.generate_json(prompt, schema=REFERENCE_SCHEMA, timeout=300)
            return _validate_references(result.get("references") or [], text), f"{client.provider}:{client.model}"
        except LLMError:
            if not fallback_heuristic:
                raise
    return parse_reference_lines(text), "heuristic"


def _item_excluded(item_key: str) -> tuple[bool, str | None]:
    configured = os.environ.get("EXTRACT_EXCLUDE_TAGS", "")
    tags = {tag.strip().casefold() for tag in configured.split(",") if tag.strip()}
    if not tags:
        allow_all = os.environ.get("EXTRACT_ALLOW_CLOUD_ALL", "").strip().casefold() in {"1", "true", "yes"}
        if allow_all:
            return False, None
        return True, "EXTRACT_EXCLUDE_TAGS is not configured; cloud extraction is disabled"
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


STABLE_IDENTIFIER_FIELDS = (
    "doi", "isbn", "cinii_crid", "ndl_bibid", "openalex_id", "s2_paper_id",
)


def _stable_identity(candidate: dict[str, Any]) -> tuple[str, str] | None:
    for field in STABLE_IDENTIFIER_FIELDS:
        value = str(candidate.get(field) or "").strip()
        if value:
            return field, value.casefold()
    return None


def _candidate_title_in_raw(raw: str, candidate: dict[str, Any]) -> str | None:
    normalized_raw = normalize_reference_text(raw)
    titles = [candidate.get("title"), *(candidate.get("alternative_titles") or [])]
    for title in titles:
        normalized = normalize_reference_text(str(title or ""))
        if len(normalized.replace(" ", "")) >= 8 and normalized in normalized_raw:
            return str(title)
    return None


def _candidate_author_in_raw(raw: str, candidate: dict[str, Any]) -> bool:
    normalized_raw = normalize_reference_text(raw)
    raw_tokens = set(normalized_raw.split())
    authors = str(candidate.get("authors") or "").strip()
    if not authors:
        return False
    for author in re.split(r"\s*;\s*|\s*,\s*(?=[^,;]+(?:;|$))", authors):
        normalized = normalize_reference_text(author)
        if not normalized:
            continue
        if normalized in normalized_raw:
            return True
        tokens = normalized.split()
        if tokens and any(len(token) >= 3 and token in raw_tokens for token in (tokens[0], tokens[-1])):
            return True
    return False


def assess_metadata_candidate(
    reference: dict[str, Any], candidate: dict[str, Any], *, runner_up_score: float = 0.0,
) -> dict[str, Any]:
    """Return deterministic evidence for accepting an external bibliography record."""
    raw = str(reference.get("raw") or "")
    identity = _stable_identity(candidate)
    score = _candidate_score(reference, candidate)
    title_supported = _candidate_title_in_raw(raw, candidate)
    author_supported = _candidate_author_in_raw(raw, candidate)
    raw_years = {int(value) for value in YEAR_RE.findall(raw)}
    candidate_year = candidate.get("year")
    year_supported = bool(
        candidate_year is not None and raw_years and int(candidate_year) in raw_years
    )
    margin = score - runner_up_score
    accepted = bool(
        not is_short_form_reference(raw)
        and not is_compound_reference(raw)
        and identity
        and title_supported
        and author_supported
        and year_supported
        and score >= 0.90
        and margin >= 0.05
    )
    return {
        "accepted": accepted,
        "score": round(score, 6), "runner_up_score": round(runner_up_score, 6),
        "margin": round(margin, 6), "stable_identifier": identity,
        "title_supported": title_supported, "author_supported": author_supported,
        "year_supported": year_supported,
    }


def identify_reference_metadata(reference: dict[str, Any]) -> dict[str, Any]:
    """Identify a citation by metadata without requiring an identifier in its text."""
    title = str(reference.get("title") or "").strip()
    raw = str(reference.get("raw") or "").strip()
    if not title or not raw or is_short_form_reference(raw) or is_compound_reference(raw):
        return {
            "status": "insufficient_metadata", "candidate": None, "evidence": None,
            "provider_failures": 0,
        }
    authors = reference.get("authors") or []
    authors_text = "; ".join(authors) if isinstance(authors, list) else str(authors)
    candidates: list[dict[str, Any]] = []
    failures = 0
    searches = (
        lambda: search_crossref(raw, rows=5),
        lambda: search_cinii(title, author=authors_text.split(";")[0], year=reference.get("year")),
        lambda: search_ndl(title, author=authors_text.split(";")[0], year=reference.get("year")),
    )
    for search in searches:
        try:
            candidates.extend(search())
        except Exception:
            failures += 1
    deduplicated: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in candidates:
        identity = _stable_identity(candidate)
        if identity is not None:
            deduplicated.setdefault(identity, candidate)
    scored = sorted(
        ((_candidate_score(reference, candidate), candidate) for candidate in deduplicated.values()),
        key=lambda pair: pair[0], reverse=True,
    )
    if not scored:
        status = "provider_unavailable" if failures == len(searches) else "unresolved"
        return {
            "status": status, "candidate": None, "evidence": None,
            "provider_failures": failures,
        }
    top_score, candidate = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    evidence = assess_metadata_candidate(reference, candidate, runner_up_score=runner_up)
    return {
        "status": "matched" if evidence["accepted"] else "ambiguous",
        "candidate": candidate, "evidence": evidence, "provider_failures": failures,
    }


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
        "cinii_crid": reference.get("cinii_crid"),
        "ndl_bibid": reference.get("ndl_bibid"),
        "openalex_id": reference.get("openalex_id"),
        "s2_paper_id": reference.get("s2_paper_id"),
    }
    external_error = False
    if any(reference.get(field) for field in STABLE_IDENTIFIER_FIELDS):
        result = {"work_id": resolve_work(**base), "confidence": 1.0, "source": "identifier"}
    else:
        identified = identify_reference_metadata(reference)
        external_error = bool(identified.get("provider_failures"))
        if identified["status"] == "matched":
            candidate = identified["candidate"]
            confidence = float(identified["evidence"]["score"])
            result = {
                "work_id": resolve_work(**{**base, **{
                    key: candidate.get(key) for key in STABLE_IDENTIFIER_FIELDS
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
    strict_llm: bool = False,
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
    references, model = extract_references(
        source_text, use_llm=use_llm, fallback_heuristic=not strict_llm,
    )
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


def commit_approved_reference_candidates(*, limit: int = 100) -> dict[str, Any]:
    """Commit approved literal identifiers or deterministically matched metadata."""
    totals = {"examined": 0, "committed": 0, "already_committed": 0, "insufficient_evidence": 0}
    for row in get_reference_review_candidates("approved"):
        if totals["examined"] >= max(limit, 0):
            break
        totals["examined"] += 1
        if row.get("committed_edge_id"):
            totals["already_committed"] += 1
            continue
        raw = str(row.get("raw_reference") or "")
        raw_folded = strip_unicode_format_characters(raw).casefold()
        doi = str(row.get("doi") or "").strip()
        isbn = str(row.get("isbn") or "").strip()
        doi_norm = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", doi.casefold())
        isbn_norm = re.sub(r"[^0-9x]", "", isbn.casefold())
        raw_isbn = re.sub(r"[^0-9x]", "", raw_folded)
        doi_verified = bool(doi_norm and doi_norm in raw_folded)
        isbn_verified = bool(isbn_norm and len(isbn_norm) in {10, 13} and isbn_norm in raw_isbn)
        metadata = row.get("resolution_metadata") or {}
        stored_evidence = row.get("resolution_evidence") or {}
        metadata_verified = False
        if metadata and stored_evidence.get("accepted"):
            reference_for_check = {
                "raw": raw, "title": row.get("title"), "authors": row.get("authors") or [],
                "year": row.get("year"),
            }
            rechecked = assess_metadata_candidate(
                reference_for_check, metadata,
                runner_up_score=float(stored_evidence.get("runner_up_score") or 0),
            )
            metadata_verified = rechecked["accepted"] and _stable_identity(metadata) is not None
        if not (doi_verified or isbn_verified or metadata_verified):
            totals["insufficient_evidence"] += 1
            continue
        reference = {
            "raw": raw, "title": row.get("title"), "authors": row.get("authors") or [],
            "year": row.get("year"), "doi": doi if doi_verified else None,
            "isbn": isbn if isbn_verified else None, "container": row.get("container"),
            "lang": row.get("lang"), "type": row.get("work_type"),
        }
        if metadata_verified:
            for field in STABLE_IDENTIFIER_FIELDS:
                if metadata.get(field):
                    reference[field] = metadata[field]
        resolved = resolve_reference(reference)
        citing_work = resolve_work(zotero_item_key=row["item_key"])
        edge_id = save_work_edge(
            citing_work, resolved["work_id"],
            source=(
                f"metadata-resolved:{row.get('resolution_source')}"
                if metadata_verified else "review-approved"
            ),
            confidence=(
                float(row.get("resolution_confidence") or resolved["confidence"])
                if metadata_verified else 1.0
            ),
            raw_reference=raw,
        )
        if mark_reference_review_committed(int(row["review_id"]), edge_id):
            totals["committed"] += 1
    return totals


def resolve_reference_review_candidates(
    *, limit: int = 100, apply: bool = False,
    note_prefix: str = "unresolved: insufficient stable identifier",
) -> dict[str, Any]:
    """Try strict metadata identification for identifier-less reviewed citations."""
    totals: dict[str, Any] = {
        "examined": 0, "matched": 0, "ambiguous": 0, "unresolved": 0,
        "provider_unavailable": 0, "applied": 0, "results": [],
    }
    rows = get_reference_review_candidates("rejected")
    for row in rows:
        if totals["examined"] >= max(limit, 0):
            break
        if not str(row.get("reviewer_note") or "").startswith(note_prefix):
            continue
        totals["examined"] += 1
        reference = {
            "raw": row.get("raw_reference"), "title": row.get("title"),
            "authors": row.get("authors") or [], "year": row.get("year"),
            "container": row.get("container"), "type": row.get("work_type"),
        }
        resolution = identify_reference_metadata(reference)
        status = resolution["status"]
        totals[status] = totals.get(status, 0) + 1
        result_row = {
            "review_id": int(row["review_id"]), "status": status,
            "candidate": resolution.get("candidate"),
            "evidence": resolution.get("evidence"),
        }
        if apply and status == "matched":
            if apply_reference_metadata_resolution(
                int(row["review_id"]), resolution["candidate"], resolution["evidence"],
            ):
                totals["applied"] += 1
                result_row["applied"] = True
        totals["results"].append(result_row)
    return totals


def apply_reference_metadata_report(
    results: list[dict[str, Any]], *,
    note_prefix: str = "unresolved: insufficient stable identifier",
) -> dict[str, int]:
    """Revalidate and atomically apply all matched rows from a dry-run report."""
    current = {
        int(row["review_id"]): row for row in get_reference_review_candidates("rejected")
    }
    resolutions: list[dict[str, Any]] = []
    for result in results:
        if result.get("status") != "matched":
            continue
        review_id = int(result["review_id"])
        row = current.get(review_id)
        if row is None or not str(row.get("reviewer_note") or "").startswith(note_prefix):
            raise ValueError(f"review {review_id}: no longer eligible for metadata resolution")
        candidate = result.get("candidate") or {}
        stored_evidence = result.get("evidence") or {}
        reference = {
            "raw": row.get("raw_reference"), "title": row.get("title"),
            "authors": row.get("authors") or [], "year": row.get("year"),
        }
        evidence = assess_metadata_candidate(
            reference, candidate,
            runner_up_score=float(stored_evidence.get("runner_up_score") or 0),
        )
        if not evidence["accepted"]:
            raise ValueError(f"review {review_id}: report match failed deterministic recheck")
        resolutions.append({
            "review_id": review_id, "candidate": candidate, "evidence": evidence,
        })
    applied = apply_reference_metadata_resolutions(resolutions) if resolutions else 0
    return {"matched_in_report": len(resolutions), "applied": applied}


__all__ = [
    "detect_reference_sections", "extract_references", "extract_references_for_item",
    "commit_approved_reference_candidates", "identify_reference_metadata",
    "apply_reference_metadata_report", "parse_reference_lines", "resolve_reference",
    "resolve_reference_review_candidates",
]
