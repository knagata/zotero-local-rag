"""Cached multilingual and case-oriented query expansion."""
from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Iterable

try:
    from .db_relations import get_query_expansion, save_query_expansion
    from .llm_client import LLMError, get_llm
except ImportError:  # pragma: no cover - direct script execution
    from db_relations import get_query_expansion, save_query_expansion
    from llm_client import LLMError, get_llm


_PROMPT_VERSION = "query-expansion-v1"
_MAX_EXPANDED_QUERIES = 12
_CASE_KEYS = ("queries", "hyde", "broader", "narrower")
_DEFAULT_KEYS = ("queries",)
_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {"type": "array", "items": {"type": "string"}},
        "hyde": {"type": "array", "items": {"type": "string"}},
        "broader": {"type": "array", "items": {"type": "string"}},
        "narrower": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["queries", "hyde", "broader", "narrower"],
    "additionalProperties": False,
}


def _dedupe(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        normalized = " ".join(value.split()).strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


def _has_japanese(text: str) -> bool:
    return bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff]", text))


def _has_english(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]{3,}", text))


def _already_bilingual(queries: list[str]) -> bool:
    return any(_has_japanese(query) for query in queries) and any(
        _has_english(query) for query in queries
    )


def _cache_key(queries: list[str], mode: str, provider: str, model: str) -> str:
    payload = json.dumps(
        {
            "version": _PROMPT_VERSION,
            "mode": mode,
            "queries": queries,
            "provider": provider,
            "model": model,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prompt(queries: list[str], mode: str) -> str:
    common = f"""You expand search queries for an anthropology and sociology library.
Input queries: {json.dumps(queries, ensure_ascii=False)}
Return only the JSON object required by the schema. Preserve proper names and technical terms.
Add concise Japanese and English search formulations and useful synonyms. Do not answer the query.
"""
    if mode == "case":
        return common + """
This is case retrieval. Also generate:
- hyde: one Japanese and one English hypothetical ethnographic passage describing a concrete case;
- broader: a few higher-level concepts that could discuss the same case;
- narrower: concrete practices, institutions, places, or observable actions.
Each entry must work as a standalone semantic-search query. Keep the total concise.
"""
    return common + """
Use queries for the translated and synonymous search formulations.
Return empty arrays for hyde, broader, and narrower.
"""


def expand_queries(
    queries: list[str],
    *,
    mode: str = "default",
    timeout: float = 5.0,
    logger: logging.Logger | None = None,
) -> list[str]:
    """Return original plus cached/generated expansions.

    Any configuration, provider, timeout, or response failure degrades to the
    original queries so semantic search remains available offline.
    """
    original = _dedupe(queries)
    if not original:
        return []
    if mode not in {"default", "case"}:
        raise ValueError("mode must be 'default' or 'case'.")
    if mode == "default" and _already_bilingual(original):
        return original

    try:
        client = get_llm("expand")
        key = _cache_key(original, mode, client.provider, client.model)
        cached = get_query_expansion(key)
        if cached:
            payload = json.loads(cached)
        else:
            payload = client.generate_json(_prompt(original, mode), schema=_SCHEMA, timeout=timeout)
            save_query_expansion(key, json.dumps(payload, ensure_ascii=False))
        keys = _CASE_KEYS if mode == "case" else _DEFAULT_KEYS
        additions = [item for name in keys for item in payload.get(name, [])]
        return _dedupe([*original, *additions])[:_MAX_EXPANDED_QUERIES]
    except Exception as exc:
        if logger:
            logger.warning("Query expansion unavailable; using original queries: %s", exc)
        return original


__all__ = ["expand_queries"]
