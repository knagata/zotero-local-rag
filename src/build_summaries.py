"""Build item/section summaries and their hierarchical-search collections."""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import httpx

try:
    from .chunk_store import active_collection_name, get_item_chunks, list_item_keys
    from .db_relations import (
        delete_section_summary, get_case_annotations, get_item_summary, get_section_summaries,
        replace_case_annotations, save_item_summary, save_section_summary, update_case_chunk,
    )
    from .embedder import create_embedding_function, open_chroma_collection, resolve_embedder_settings
    from .llm_client import LLMError, RateLimitReached, get_llm
    from .manifest import load_manifest
    from .env_utils import load_dotenv_native
    from .codex_quota import CodexQuotaError, CodexQuotaFloorReached, require_weekly_quota
except ImportError:  # pragma: no cover
    from chunk_store import active_collection_name, get_item_chunks, list_item_keys
    from db_relations import (
        delete_section_summary, get_case_annotations, get_item_summary, get_section_summaries,
        replace_case_annotations, save_item_summary, save_section_summary, update_case_chunk,
    )
    from embedder import create_embedding_function, open_chroma_collection, resolve_embedder_settings
    from llm_client import LLMError, RateLimitReached, get_llm
    from manifest import load_manifest
    from env_utils import load_dotenv_native
    from codex_quota import CodexQuotaError, CodexQuotaFloorReached, require_weekly_quota


ROOT = Path(__file__).resolve().parents[1]
load_dotenv_native(ROOT)
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", ROOT / "data" / "chroma"))
MANIFEST_PATH = Path(os.environ.get("MANIFEST_PATH", ROOT / "data" / "manifest.json"))
SECTION_WINDOW = 40


SECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "cases": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "description": {"type": "string"}, "region": {"type": ["string", "null"]},
                "group": {"type": ["string", "null"]},
                "practices": {"type": "array", "items": {"type": "string"}},
                "phenomena": {"type": "array", "items": {"type": "string"}},
                "period": {"type": ["string", "null"]},
                "locator_hint": {"type": ["string", "null"]},
                "source_kind": {"type": ["string", "null"]},
                "evidence_quote": {"type": "string"},
            },
            "required": [
                "description", "region", "group", "practices", "phenomena", "period",
                "locator_hint", "source_kind", "evidence_quote",
            ],
            "additionalProperties": False,
        }},
        "chapter_authors": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "evidence_quote": {"type": "string"},
            },
            "required": ["name", "evidence_quote"],
            "additionalProperties": False,
        }},
        "first_publication_note": {
            "anyOf": [{
                "type": "object",
                "properties": {
                    "note": {"type": "string"},
                    "evidence_quote": {"type": "string"},
                },
                "required": ["note", "evidence_quote"],
                "additionalProperties": False,
            }, {"type": "null"}],
        },
    },
    "required": ["summary", "cases", "chapter_authors", "first_publication_note"],
    "additionalProperties": False,
}

NON_CONTENT_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"目次|contents?|table\s+of\s+contents|図版一覧|表一覧|口絵|凡例|奥付|索引|index|"
    r"copyright|著作権(?:について|表示|注意)?|colophon|標題紙|title\s+page|half\s+title|"
    r"cover|本著作物について|list\s+of\s+(?:illustrations|figures|tables)|"
    r"series(?:\s+page|\s+list)?|シリーズ一覧|叢書一覧|acknowledg(?:e)?ments?|謝辞|"
    r"about\s+(?:the\s+)?author|about\s+[\w .'-]+|著者紹介|contributors?|執筆者紹介|"
    r"事項一覧|人名一覧|用語一覧|略語一覧|年表|glossary|chronology|timeline|"
    r"list\s+of\s+(?:terms|names|abbreviations)|"
    r"references?|bibliography|参考文献|引用文献"
    r")\s*$",
    re.I,
)
META_SUMMARY_RE = re.compile(
    r"(?:"
    r"要約(?:すること)?(?:が|は)?できません|要約できません|ご提示ください|"
    r"(?:入力|本文|テキスト|内容|原文)[^。\n]{0,80}(?:含(?:まれて|んで)い(?:ません|ない)|"
    r"提示されていません|見当たりません|ありません)|"
    r"裏付けること(?:が|は)?できません|"
    r"(?:抽出結果|結果|配列|フィールド)[^。\n]{0,40}空(?:と|に)しました|"
    r"cannot\s+(?:provide|produce|generate|write)\s+(?:a\s+)?summary|"
    r"please\s+(?:provide|share)\s+(?:the\s+)?(?:text|content|passage)|"
    r"(?:input|provided\s+(?:text|content))[^.\n]{0,50}(?:does\s+not|doesn't)\s+contain"
    r")",
    re.I,
)
TOC_ENTRY_RE = re.compile(r"(?:\.{3,}|…{2,}|・{3,})\s*\d+\s*$")
CHAPTER_MARKER_RE = re.compile(
    r"(?:第[一二三四五六七八九十百0-9]+章|序章|終章|chapter\s+\d+)", re.I,
)
SPECIFIC_VALUE_RE = re.compile(
    r"https?://\S+|10\.\d{4,9}/\S+|"
    r"(?:ISBN(?:-1[03])?[:\s]*)?[0-9Xx][0-9Xx\- ]{8,}|"
    r"(?:18|19|20)\d{2}|\d+(?:\.\d+)?",
    re.I,
)
QUOTED_VALUE_RE = re.compile(r"[『「“\"]([^』」”\"]{2,})[』」”\"]")

ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"}, "summary_en": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "summary_en", "keywords"],
    "additionalProperties": False,
}


def split_sections(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group by chapter; use deterministic 40-chunk windows when no chapter exists."""
    chapter_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ungrouped: list[dict[str, Any]] = []
    for chunk in chunks:
        chapter = str(chunk.get("metadata", {}).get("chapter") or "").strip()
        (chapter_groups[chapter] if chapter else ungrouped).append(chunk)
    sections: list[dict[str, Any]] = []
    for index, (chapter, grouped) in enumerate(chapter_groups.items()):
        sections.append({"section_id": f"c{index}", "chapter": chapter, "chunks": grouped})
    for start in range(0, len(ungrouped), SECTION_WINDOW):
        sections.append({
            "section_id": f"w{start // SECTION_WINDOW}", "chapter": "",
            "chunks": ungrouped[start : start + SECTION_WINDOW],
        })
    return sections


def _section_source_text(section: dict[str, Any]) -> str:
    return "\n\n".join(
        str(chunk.get("text") or "") for chunk in section["chunks"] if chunk.get("text")
    )[:30000]


def classify_section_content(section: dict[str, Any]) -> str:
    """Conservatively reject front matter and table-of-contents-like sections."""
    text = _section_source_text(section).strip()
    minimum = max(0, int(os.environ.get("SUMMARY_MIN_SECTION_CHARS", "400")))
    if len(text) < minimum:
        return "non_content"
    chapter = str(section.get("chapter") or "").strip()
    first_line = text.splitlines()[0].strip() if text else ""
    for heading in (chapter, first_line[:160]):
        if heading and NON_CONTENT_HEADING_RE.fullmatch(heading):
            return "non_content"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    early_lines = lines[:8]
    if any(re.match(r"^index\s+(?:a\s+note\s+about|[a-z].*\b\d+)", line, re.I) for line in early_lines):
        return "non_content"
    if any(line.casefold().startswith("contents ") for line in early_lines):
        return "non_content"
    page_entries = sum(bool(TOC_ENTRY_RE.search(line)) for line in lines)
    if len(lines) >= 5 and page_entries >= 3 and page_entries / len(lines) >= 0.3:
        return "non_content"
    # OCR often collapses a table of contents into long lines. Multiple chapter
    # markers in a short section are safer to skip than to let the model fill in.
    if len(text) < 5000 and len(CHAPTER_MARKER_RE.findall(text)) >= 3:
        return "non_content"
    return "content"


def is_meta_summary(summary: str) -> bool:
    """Detect model refusals/meta commentary that must never become an index entry."""
    return bool(META_SUMMARY_RE.search(str(summary or "")[:500]))


def _normalize_evidence(value: Any) -> str:
    return " ".join(str(value or "").split())


def _new_verification_stats() -> dict[str, Any]:
    return {
        "cases": {"generated": 0, "kept": 0, "discarded": 0, "reasons": {}},
        "chapter_authors": {"generated": 0, "kept": 0, "discarded": 0, "reasons": {}},
        "first_publication_note": {"generated": 0, "kept": 0, "discarded": 0, "reasons": {}},
        "total_generated": 0, "total_kept": 0, "total_discarded": 0,
        "discard_rate": 0.0, "suspicious_section": False,
    }


def _record_verification(stats: dict[str, Any], field: str, kept: bool, reason: str = "") -> None:
    bucket = stats[field]
    bucket["generated"] += 1
    stats["total_generated"] += 1
    if kept:
        bucket["kept"] += 1
        stats["total_kept"] += 1
    else:
        bucket["discarded"] += 1
        stats["total_discarded"] += 1
        bucket["reasons"][reason] = bucket["reasons"].get(reason, 0) + 1


def _note_values_supported(note: str, evidence_quote: str) -> bool:
    normalized_quote = _normalize_evidence(evidence_quote).casefold()
    specific = [match.group(0).rstrip(".,;:。、）)") for match in SPECIFIC_VALUE_RE.finditer(note)]
    quoted = QUOTED_VALUE_RE.findall(note)
    return all(_normalize_evidence(value).casefold() in normalized_quote for value in specific + quoted)


def _case_specific_values_supported(case: dict[str, Any], evidence_quote: str) -> bool:
    """Require dates, counts, identifiers, and other numerals to occur in the evidence."""
    normalized_quote = _normalize_evidence(evidence_quote).casefold()
    checked = " ".join(filter(None, [
        str(case.get("description") or ""), str(case.get("period") or ""),
        str(case.get("locator_hint") or ""),
    ]))
    values = [match.group(0).rstrip(".,;:。、）)") for match in SPECIFIC_VALUE_RE.finditer(checked)]
    return all(_normalize_evidence(value).casefold() in normalized_quote for value in values)


def _evidence_chunk_id(evidence_quote: str, chunks: list[dict[str, Any]]) -> str | None:
    """Locate evidence in one chunk or a pair of adjacent chunks.

    When a quote crosses one boundary, the first chunk ID is retained as the
    navigation anchor. Quotes spanning three or more chunks remain unsupported.
    """
    needle = _normalize_evidence(evidence_quote)
    if not needle:
        return None
    for chunk in chunks:
        if needle in _normalize_evidence(chunk.get("text")):
            value = chunk.get("id")
            return str(value) if value is not None else None
    for first, second in zip(chunks, chunks[1:]):
        joined = " ".join(filter(None, [
            _normalize_evidence(first.get("text")),
            _normalize_evidence(second.get("text")),
        ]))
        if needle in joined:
            value = first.get("id")
            return str(value) if value is not None else None
    return None


def _verify_section_result(
    result: dict[str, Any], source_text: str, chunks: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep only structured fields backed by an exact quote from the LLM input."""
    normalized_source = _normalize_evidence(source_text)
    verified: dict[str, Any] = {
        "summary": str(result.get("summary") or ""),
        "cases": [], "chapter_authors": [], "first_publication_note": None,
    }
    stats = _new_verification_stats()
    for case in result.get("cases") or []:
        if not isinstance(case, dict):
            _record_verification(stats, "cases", False, "invalid_shape")
            continue
        quote = _normalize_evidence(case.get("evidence_quote"))
        if not quote or quote not in normalized_source:
            _record_verification(stats, "cases", False, "evidence_not_in_source")
            continue
        chunk_id = _evidence_chunk_id(quote, chunks or []) if chunks is not None else None
        if chunks is not None and chunk_id is None:
            _record_verification(stats, "cases", False, "evidence_not_in_chunk")
            continue
        if not _case_specific_values_supported(case, quote):
            _record_verification(stats, "cases", False, "value_not_in_evidence")
            continue
        row = dict(case)
        row["evidence_quote"] = quote
        if chunks is not None:
            row["chunk_id"] = chunk_id
        verified["cases"].append(row)
        _record_verification(stats, "cases", True)
    for author in result.get("chapter_authors") or []:
        if not isinstance(author, dict):
            _record_verification(stats, "chapter_authors", False, "invalid_shape")
            continue
        name = _normalize_evidence(author.get("name"))
        quote = _normalize_evidence(author.get("evidence_quote"))
        if not quote or quote not in normalized_source:
            _record_verification(stats, "chapter_authors", False, "evidence_not_in_source")
            continue
        if not name or name.casefold() not in quote.casefold():
            _record_verification(stats, "chapter_authors", False, "name_not_in_evidence")
            continue
        verified["chapter_authors"].append({"name": name, "evidence_quote": quote})
        _record_verification(stats, "chapter_authors", True)
    publication = result.get("first_publication_note")
    if publication is not None:
        if not isinstance(publication, dict):
            _record_verification(stats, "first_publication_note", False, "invalid_shape")
        else:
            note = _normalize_evidence(publication.get("note"))
            quote = _normalize_evidence(publication.get("evidence_quote"))
            if not quote or quote not in normalized_source:
                _record_verification(
                    stats, "first_publication_note", False, "evidence_not_in_source",
                )
            elif not note or not _note_values_supported(note, quote):
                _record_verification(
                    stats, "first_publication_note", False, "value_not_in_evidence",
                )
            else:
                verified["first_publication_note"] = {"note": note, "evidence_quote": quote}
                _record_verification(stats, "first_publication_note", True)
    generated = stats["total_generated"]
    stats["discard_rate"] = round(stats["total_discarded"] / generated, 4) if generated else 0.0
    stats["suspicious_section"] = bool(generated and stats["discard_rate"] > 0.5)
    return verified, stats


def _metadata(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    return chunks[0].get("metadata", {}) if chunks else {}


def _extractive_section(section: dict[str, Any]) -> dict[str, Any]:
    chunks = section["chunks"]
    sampled = chunks[:2] + (chunks[-1:] if len(chunks) > 2 else [])
    text = " ".join(str(chunk.get("text") or "").strip() for chunk in sampled)
    return {
        "summary": text[:1200], "cases": [], "chapter_authors": [],
        "first_publication_note": None,
    }


def _keywords(text: str, limit: int = 10) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z-]{3,}|[\u3040-\u30ff\u3400-\u9fff]{2,12}", text.casefold())
    stop = {"this", "that", "with", "from", "について", "として", "ために", "または"}
    return [token for token, _ in Counter(token for token in tokens if token not in stop).most_common(limit)]


def _source_mtime(chunks: list[dict[str, Any]], manifest: dict[str, Any]) -> float:
    keys = {chunk.get("metadata", {}).get("attachmentKey") for chunk in chunks}
    values = [
        float(manifest.get("files", {}).get(key, {}).get("mtime") or 0)
        for key in keys if key
    ]
    return max(values, default=0.0)


def _excluded_from_llm(item_key: str) -> tuple[bool, str | None]:
    configured = {
        tag.strip().casefold() for tag in os.environ.get("SUMMARY_EXCLUDE_TAGS", "").split(",") if tag.strip()
    }
    if not configured:
        allow_all = os.environ.get("SUMMARY_ALLOW_CLOUD_ALL", "").strip().casefold() in {"1", "true", "yes"}
        if allow_all:
            return False, None
        return True, "SUMMARY_EXCLUDE_TAGS is not configured; cloud summarization is disabled"
    base = (os.environ.get("ZOTERO_LOCAL_API_BASE") or "http://127.0.0.1:23119/api").rstrip("/")
    prefix = (os.environ.get("ZOTERO_LOCAL_API_PREFIX") or "users/0").strip("/")
    headers = {"Zotero-API-Version": os.environ.get("ZOTERO_API_VERSION", "3")}
    if os.environ.get("ZOTERO_API_KEY"):
        headers["Zotero-API-Key"] = os.environ["ZOTERO_API_KEY"]
    try:
        response = httpx.get(f"{base}/{prefix}/items/{item_key}", headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json().get("data", response.json())
        tags = {str(tag.get("tag") or "").strip().casefold() for tag in data.get("tags", [])}
        matched = sorted(configured & tags)
        return bool(matched), ", ".join(matched) if matched else None
    except Exception as exc:
        # If a privacy exclusion policy exists, inability to inspect tags must fail closed.
        return True, f"could not verify exclusion tags: {exc}"


def _llm_section(section: dict[str, Any]) -> tuple[dict[str, Any], str]:
    client = get_llm("summary")
    text = _section_source_text(section)
    prompt = (
        "以下の学術資料の章を日本語200〜300字で要約し、民族誌的・経験的事例、章著者、"
        "初出情報をJSONで抽出してください。事例の二次言及も source_kind='secondary' として保持し、"
        "根拠がない値は空配列またはnullにしてください。cases、chapter_authors、"
        "first_publication_noteには、各値を直接裏付ける入力原文の連続した文字列を"
        "evidence_quoteとして一字も変えずにコピーしてください。入力に存在しない知識、"
        "日付、DOI、ISBN、著者名、誌名、URLを補完しないでください。各caseの"
        "evidence_quoteはdescription・region・group・periodに書くすべての具体的事実を"
        "その一つの引用だけで直接支持する必要があります。複数箇所を組み合わせないと"
        "支持できないcaseは出力しないでください。OCR誤字や空白も修正せずコピーしてください。\n\n" + text
    )
    generated = client.generate_json(prompt, schema=SECTION_SCHEMA, timeout=300)
    verified, stats = _verify_section_result(generated, text, section["chunks"])
    verified["_verification"] = stats
    return verified, f"{client.provider}:{client.model}"


def _llm_item(title: str, section_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    client = get_llm("summary")
    prompt = (
        "以下の章要約から資料全体について、400〜600字の日本語構造化要約、150語程度の英語要約、"
        "日英混合キーワード10個をJSONで返してください。\nタイトル: " + title + "\n\n" +
        "\n\n".join(row["summary"] for row in section_rows)
    )
    return client.generate_json(prompt, schema=ITEM_SCHEMA, timeout=300), f"{client.provider}:{client.model}"


def build_item(
    item_key: str, *, mode: str = "extractive", force: bool = False,
    audit_sections: list[dict[str, Any]] | None = None,
    quota_guard: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    chunks = get_item_chunks(item_key)
    if not chunks:
        return {"item_key": item_key, "status": "empty"}
    manifest = load_manifest(MANIFEST_PATH)
    source_mtime = _source_mtime(chunks, manifest)
    existing = get_item_summary(item_key)
    same_source = (
        existing
        and existing.get("chunk_count") == len(chunks)
        and float(existing.get("source_mtime") or 0) == source_mtime
    )
    upgrading_extract = mode == "llm" and existing and existing.get("model") == "extractive"
    if not force and same_source and not upgrading_extract:
        return {"item_key": item_key, "status": "unchanged"}
    if mode == "llm":
        excluded, reason = _excluded_from_llm(item_key)
        if excluded:
            return {"item_key": item_key, "status": "excluded", "reason": reason}

    section_rows: list[dict[str, Any]] = []
    model = "extractive"
    use_llm = mode == "llm"
    skipped_non_content = 0
    verification_by_section: list[dict[str, Any]] = []
    for section in split_sections(chunks):
        if mode == "llm" and classify_section_content(section) == "non_content":
            delete_section_summary(item_key, section["section_id"])
            skipped_non_content += 1
            if audit_sections is not None:
                audit_sections.append({
                    "section_id": section["section_id"], "chapter": section["chapter"],
                    "status": "skipped_non_content", "llm_summary": None, "cases": [],
                    "chapter_authors": [], "first_publication_note": None,
                    "verification": None,
                })
            continue
        if use_llm:
            try:
                if quota_guard is not None:
                    quota_guard()
                result, model = _llm_section(section)
            except RateLimitReached:
                raise
            except LLMError:
                result, model = _extractive_section(section), "extractive"
                use_llm = False
        else:
            result = _extractive_section(section)
        summary = str(result.get("summary") or "").strip()
        if model != "extractive" and is_meta_summary(summary):
            delete_section_summary(item_key, section["section_id"])
            skipped_non_content += 1
            if audit_sections is not None:
                audit_sections.append({
                    "section_id": section["section_id"], "chapter": section["chapter"],
                    "status": "skipped_non_content", "skip_reason": "meta_response",
                    "llm_summary": None, "cases": [], "chapter_authors": [],
                    "first_publication_note": None, "verification": None,
                })
            continue
        row = {"section_id": section["section_id"], "chapter": section["chapter"], "summary": summary}
        section_rows.append(row)
        if result.get("_verification"):
            verification_by_section.append({
                "section_id": section["section_id"], **result["_verification"],
            })
        if audit_sections is not None:
            audit_sections.append({
                "section_id": section["section_id"], "chapter": section["chapter"],
                "status": "generated" if model != "extractive" else "extractive_fallback",
                "model": model, "llm_summary": summary,
                "cases": result.get("cases") or [],
                "chapter_authors": result.get("chapter_authors") or [],
                "first_publication_note": result.get("first_publication_note"),
                "verification": result.get("_verification"),
            })
        authors = result.get("chapter_authors") or []
        author_names = [
            str(author.get("name") or "").strip() for author in authors if isinstance(author, dict)
        ] if isinstance(authors, list) else []
        publication = result.get("first_publication_note")
        publication_note = publication.get("note") if isinstance(publication, dict) else None
        save_section_summary(
            item_key, section["section_id"], summary, chapter=section["chapter"], model=model,
            chunk_count=len(section["chunks"]),
            chapter_authors="; ".join(filter(None, author_names)),
            first_publication_note=publication_note,
        )
        replace_case_annotations(item_key, section["section_id"], result.get("cases") or [], model=model)

    metadata = _metadata(chunks)
    title = str(metadata.get("title") or "")
    if section_rows and use_llm and model != "extractive":
        try:
            if quota_guard is not None:
                quota_guard()
            item_result, model = _llm_item(title, section_rows)
        except RateLimitReached:
            raise
        except LLMError:
            item_result = {}
    else:
        item_result = {}
    summary = str(item_result.get("summary") or "\n\n".join(row["summary"] for row in section_rows))[:4000]
    keywords = item_result.get("keywords") or _keywords(summary)
    save_item_summary(
        item_key, summary, model, summary_en=str(item_result.get("summary_en") or ""),
        keywords="; ".join(keywords), chunk_count=len(chunks), source_mtime=source_mtime,
    )
    total_generated = sum(row["total_generated"] for row in verification_by_section)
    total_discarded = sum(row["total_discarded"] for row in verification_by_section)
    return {
        "item_key": item_key, "status": "updated", "sections": len(section_rows), "model": model,
        "skipped_non_content": skipped_non_content,
        "verification": {
            "total_generated": total_generated,
            "total_discarded": total_discarded,
            "discard_rate": round(total_discarded / total_generated, 4) if total_generated else 0.0,
            "suspicious_sections": [
                row["section_id"] for row in verification_by_section if row["suspicious_section"]
            ],
            "sections": verification_by_section,
        },
    }


def embed_summaries(
    *, item_keys: set[str] | None = None, only_missing: bool = False,
) -> dict[str, int]:
    base = active_collection_name(CHROMA_DIR)
    if not base:
        raise RuntimeError("No active paragraph collection was found.")
    cfg = resolve_embedder_settings(ROOT)
    ef = create_embedding_function(cfg)
    item_collection = open_chroma_collection(CHROMA_DIR, f"{base}__sum_item", ef)
    section_collection = open_chroma_collection(CHROMA_DIR, f"{base}__sum_section", ef)
    case_collection = open_chroma_collection(CHROMA_DIR, f"{base}__cases", ef)
    paragraph_collection = getattr(item_collection, "_chroma_client").get_collection(base)
    item_rows: list[tuple[str, str, dict[str, Any]]] = []
    section_rows: list[tuple[str, str, dict[str, Any]]] = []
    case_rows: list[tuple[str, str, dict[str, Any]]] = []
    available_keys = list_item_keys(chroma_dir=CHROMA_DIR, collection_name=base)
    target_keys = [key for key in available_keys if item_keys is None or key in item_keys]
    for item_key in target_keys:
        summary = get_item_summary(item_key)
        chunks = get_item_chunks(item_key, chroma_dir=CHROMA_DIR, collection_name=base)
        metadata = _metadata(chunks)
        if summary:
            document = "\n".join(filter(None, [
                str(metadata.get("title") or ""), str(metadata.get("creators") or ""),
                str(summary.get("keywords") or ""), str(summary.get("summary") or ""),
                str(summary.get("summary_en") or ""),
            ]))[:1200]
            item_rows.append((f"sum:item:{item_key}", document, {
                "itemKey": item_key, "title": str(metadata.get("title") or ""),
                "creators": str(metadata.get("creators") or ""), "year": int(metadata.get("year") or 0),
            }))
        for section in get_section_summaries(item_key):
            section_rows.append((
                f"sum:sec:{item_key}:{section['section_id']}",
                "\n".join(filter(None, [
                    str(metadata.get("title") or ""), str(section.get("chapter") or ""), section["summary"]
                ]))[:900],
                {"itemKey": item_key, "title": str(metadata.get("title") or ""),
                 "chapter": str(section.get("chapter") or ""), "section_id": section["section_id"]},
            ))
    for case in get_case_annotations():
        if item_keys is not None and case["item_key"] not in item_keys:
            continue
        document = "\n".join(filter(None, [
            case.get("description"), case.get("region"), case.get("grp"),
            case.get("practices"), case.get("phenomena"),
        ]))
        if not case.get("chunk_id") and document:
            try:
                nearest = paragraph_collection.query(
                    query_embeddings=ef([document]), n_results=1,
                    where={"itemKey": case["item_key"]}, include=["distances"],
                )
                ids = nearest.get("ids") or []
                case["chunk_id"] = ids[0][0] if ids and ids[0] else None
                update_case_chunk(int(case["case_id"]), case.get("chunk_id"))
            except Exception:
                pass
        case_rows.append((f"case:{case['case_id']}", document, {
            "case_id": int(case["case_id"]), "itemKey": str(case["item_key"]),
            "section_id": str(case.get("section_id") or ""),
            "chunk_id": str(case.get("chunk_id") or ""),
            "region": str(case.get("region") or ""),
        }))
    batch_size = max(1, int(os.environ.get("SUMMARY_EMBED_BATCH_SIZE", "16")))
    for collection, rows in (
        (item_collection, item_rows), (section_collection, section_rows), (case_collection, case_rows)
    ):
        if item_keys is None and not only_missing:
            existing_ids = set(collection.get(include=[]).get("ids") or [])
            current_ids = {row[0] for row in rows}
            stale_ids = sorted(existing_ids - current_ids)
            for start in range(0, len(stale_ids), 1000):
                collection.delete(ids=stale_ids[start : start + 1000])
        if item_keys is not None and not only_missing:
            for item_key in item_keys:
                try:
                    collection.delete(where={"itemKey": item_key})
                except Exception:
                    pass
        if only_missing and rows:
            existing_ids = set(collection.get(include=[]).get("ids") or [])
            rows = [row for row in rows if row[0] not in existing_ids]
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            collection.upsert(
                ids=[row[0] for row in batch], documents=[row[1] for row in batch],
                metadatas=[row[2] for row in batch], embeddings=ef([row[1] for row in batch]),
            )
            gc.collect()
            try:
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except (ImportError, RuntimeError):
                pass
            if start == 0 or start + batch_size >= len(rows) or (start // batch_size) % 25 == 0:
                print(f"Embedding {collection.name}: {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
    for collection in (item_collection, section_collection, case_collection):
        client = getattr(collection, "_chroma_client", None)
        if client:
            client.close()
    return {"items": len(item_rows), "sections": len(section_rows), "cases": len(case_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--mode", choices=("extractive", "llm"), default="extractive")
    parser.add_argument("--no-embed", action="store_true")
    parser.add_argument("--embed-only", action="store_true")
    parser.add_argument("--resume-embed", action="store_true", help="Embed only IDs not already present.")
    parser.add_argument("--llm", help="Override LLM_SUMMARY, e.g. codex_cli:auto.")
    parser.add_argument(
        "--audit-output", type=Path,
        help="Write per-item grounding verification statistics as JSON.",
    )
    parser.add_argument("--max-items", type=int, help="Maximum items to process in this run.")
    parser.add_argument("--max-hours", type=float, help="Stop before starting another item after this many hours.")
    parser.add_argument(
        "--stop-on-rate-limit", action="store_true",
        help="Treat provider quota exhaustion as a resumable normal stop.",
    )
    parser.add_argument(
        "--min-weekly-remaining-percent", type=int,
        help="Stop safely before each Codex LLM request when weekly quota is at or below this percentage.",
    )
    args = parser.parse_args()
    if args.max_items is not None and args.max_items < 0:
        parser.error("--max-items must be non-negative")
    if args.max_hours is not None and args.max_hours <= 0:
        parser.error("--max-hours must be positive")
    if (
        args.min_weekly_remaining_percent is not None
        and not 0 <= args.min_weekly_remaining_percent <= 100
    ):
        parser.error("--min-weekly-remaining-percent must be between 0 and 100")
    if args.llm:
        os.environ["LLM_SUMMARY"] = args.llm
    counts = Counter()
    updated_keys: set[str] = set()
    started = time.monotonic()
    processed = 0
    stop_reason = "completed"
    audit_items: list[dict[str, Any]] = []
    if not args.embed_only:
        keys = [args.item] if args.item else list_item_keys()
        if args.max_items is not None:
            keys = keys[:args.max_items]
        for index, key in enumerate(keys, start=1):
            if args.max_hours is not None and time.monotonic() - started >= args.max_hours * 3600:
                stop_reason = "max_hours"
                break
            if args.min_weekly_remaining_percent is not None:
                try:
                    require_weekly_quota(args.min_weekly_remaining_percent)
                except CodexQuotaFloorReached as exc:
                    stop_reason = "weekly_quota_floor"
                    print(f"[{index}/{len(keys)}] {exc}; stopping safely", flush=True)
                    break
                except CodexQuotaError as exc:
                    stop_reason = "quota_unknown"
                    print(f"[{index}/{len(keys)}] quota_unknown ({exc}); stopping safely", flush=True)
                    break
            try:
                quota_guard = None
                if args.min_weekly_remaining_percent is not None:
                    quota_guard = lambda: require_weekly_quota(
                        args.min_weekly_remaining_percent,
                    )
                result = build_item(
                    key, mode=args.mode, force=args.force, quota_guard=quota_guard,
                )
            except CodexQuotaFloorReached as exc:
                stop_reason = "weekly_quota_floor"
                print(f"[{index}/{len(keys)}] {exc}; stopping safely", flush=True)
                break
            except CodexQuotaError as exc:
                stop_reason = "quota_unknown"
                print(f"[{index}/{len(keys)}] quota_unknown ({exc}); stopping safely", flush=True)
                break
            except RateLimitReached as exc:
                counts["rate_limited"] += 1
                stop_reason = "rate_limit"
                print(f"[{index}/{len(keys)}] {key}: rate_limit ({exc})", flush=True)
                if args.stop_on_rate_limit:
                    break
                raise
            processed += 1
            audit_items.append(result)
            counts[result["status"]] += 1
            if result["status"] == "updated":
                updated_keys.add(key)
            print(f"[{index}/{len(keys)}] {key}: {result['status']}", flush=True)
    if not args.no_embed and (args.embed_only or updated_keys or args.resume_embed):
        embed_keys = None if args.embed_only or (args.resume_embed and not updated_keys) else updated_keys
        embedded = embed_summaries(
            item_keys=embed_keys,
            only_missing=args.resume_embed,
        )
        print(
            f"Embedded {embedded['items']} item summaries, {embedded['sections']} section summaries, "
            f"and {embedded['cases']} cases."
        )
    elif not args.no_embed:
        print("No changed summaries to embed.")
    run_report = {
        "counts": dict(counts), "processed": processed, "stop_reason": stop_reason,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "items": audit_items,
    }
    if args.audit_output:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(
            json.dumps(run_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
    print(json.dumps(run_report, ensure_ascii=False))


if __name__ == "__main__":
    main()
