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
from typing import Any

import httpx

try:
    from .chunk_store import active_collection_name, get_item_chunks, list_item_keys
    from .db_relations import (
        get_case_annotations, get_item_summary, get_section_summaries, replace_case_annotations,
        save_item_summary, save_section_summary, update_case_chunk,
    )
    from .embedder import create_embedding_function, open_chroma_collection, resolve_embedder_settings
    from .llm_client import LLMError, RateLimitReached, get_llm
    from .manifest import load_manifest
    from .env_utils import load_dotenv_native
except ImportError:  # pragma: no cover
    from chunk_store import active_collection_name, get_item_chunks, list_item_keys
    from db_relations import (
        get_case_annotations, get_item_summary, get_section_summaries, replace_case_annotations,
        save_item_summary, save_section_summary, update_case_chunk,
    )
    from embedder import create_embedding_function, open_chroma_collection, resolve_embedder_settings
    from llm_client import LLMError, RateLimitReached, get_llm
    from manifest import load_manifest
    from env_utils import load_dotenv_native


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
            },
            "required": [
                "description", "region", "group", "practices", "phenomena", "period",
                "locator_hint", "source_kind",
            ],
            "additionalProperties": False,
        }},
        "chapter_authors": {"type": "array", "items": {"type": "string"}},
        "first_publication_note": {"type": ["string", "null"]},
    },
    "required": ["summary", "cases", "chapter_authors", "first_publication_note"],
    "additionalProperties": False,
}

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
    text = "\n\n".join(chunk["text"] for chunk in section["chunks"] if chunk.get("text"))[:30000]
    prompt = (
        "以下の学術資料の章を日本語200〜300字で要約し、民族誌的・経験的事例、章著者、"
        "初出情報をJSONで抽出してください。事例の二次言及も source_kind='secondary' として保持し、"
        "根拠がない値は空配列またはnullにしてください。\n\n" + text
    )
    return client.generate_json(prompt, schema=SECTION_SCHEMA, timeout=300), f"{client.provider}:{client.model}"


def _llm_item(title: str, section_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    client = get_llm("summary")
    prompt = (
        "以下の章要約から資料全体について、400〜600字の日本語構造化要約、150語程度の英語要約、"
        "日英混合キーワード10個をJSONで返してください。\nタイトル: " + title + "\n\n" +
        "\n\n".join(row["summary"] for row in section_rows)
    )
    return client.generate_json(prompt, schema=ITEM_SCHEMA, timeout=300), f"{client.provider}:{client.model}"


def build_item(item_key: str, *, mode: str = "extractive", force: bool = False) -> dict[str, Any]:
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
    for section in split_sections(chunks):
        if use_llm:
            try:
                result, model = _llm_section(section)
            except RateLimitReached:
                raise
            except LLMError:
                result, model = _extractive_section(section), "extractive"
                use_llm = False
        else:
            result = _extractive_section(section)
        summary = str(result.get("summary") or "").strip()
        row = {"section_id": section["section_id"], "chapter": section["chapter"], "summary": summary}
        section_rows.append(row)
        authors = result.get("chapter_authors") or []
        save_section_summary(
            item_key, section["section_id"], summary, chapter=section["chapter"], model=model,
            chunk_count=len(section["chunks"]),
            chapter_authors="; ".join(authors) if isinstance(authors, list) else str(authors),
            first_publication_note=result.get("first_publication_note"),
        )
        replace_case_annotations(item_key, section["section_id"], result.get("cases") or [], model=model)

    metadata = _metadata(chunks)
    title = str(metadata.get("title") or "")
    if use_llm and model != "extractive":
        try:
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
    return {"item_key": item_key, "status": "updated", "sections": len(section_rows), "model": model}


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
    parser.add_argument("--max-items", type=int, help="Maximum items to process in this run.")
    parser.add_argument("--max-hours", type=float, help="Stop before starting another item after this many hours.")
    parser.add_argument(
        "--stop-on-rate-limit", action="store_true",
        help="Treat provider quota exhaustion as a resumable normal stop.",
    )
    args = parser.parse_args()
    if args.max_items is not None and args.max_items < 0:
        parser.error("--max-items must be non-negative")
    if args.max_hours is not None and args.max_hours <= 0:
        parser.error("--max-hours must be positive")
    if args.llm:
        os.environ["LLM_SUMMARY"] = args.llm
    counts = Counter()
    updated_keys: set[str] = set()
    started = time.monotonic()
    processed = 0
    stop_reason = "completed"
    if not args.embed_only:
        keys = [args.item] if args.item else list_item_keys()
        if args.max_items is not None:
            keys = keys[:args.max_items]
        for index, key in enumerate(keys, start=1):
            if args.max_hours is not None and time.monotonic() - started >= args.max_hours * 3600:
                stop_reason = "max_hours"
                break
            try:
                result = build_item(key, mode=args.mode, force=args.force)
            except RateLimitReached as exc:
                counts["rate_limited"] += 1
                stop_reason = "rate_limit"
                print(f"[{index}/{len(keys)}] {key}: rate_limit ({exc})", flush=True)
                if args.stop_on_rate_limit:
                    break
                raise
            processed += 1
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
    print(json.dumps({
        "counts": dict(counts), "processed": processed, "stop_reason": stop_reason,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
