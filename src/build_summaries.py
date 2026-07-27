"""Build item/section summaries and their hierarchical-search collections."""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import signal
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import httpx

try:
    from .chunk_store import active_collection_name, get_item_chunks, list_item_keys
    from .db_relations import (
        delete_section_summary, get_item_summary, get_section_summaries,
        mark_insight_generation_status, save_item_summary, save_section_summary,
    )
    from .embedder import create_embedding_function, open_chroma_collection, resolve_embedder_settings
    from .llm_client import DeepSeekClient, InvalidLLMResponse, LLMError, RateLimitReached, get_llm
    from .manifest import load_manifest
    from .env_utils import load_dotenv_native
    from .codex_quota import CodexQuotaError, CodexQuotaFloorReached, require_weekly_quota
    from .summary_prompts import cited_item_summary_prompt, cited_section_summary_prompt
    # Shared primitives moved to summary_core (re-exported here for compatibility).
    from .summary_core import (
        CHAPTER_MARKER_RE, DEFINITIVE_IDENTIFIER_RE, META_SUMMARY_RE, NON_CONTENT_HEADING_RE,
        QUOTED_VALUE_RE, SPECIFIC_VALUE_RE, SUMMARY_ONLY_SCHEMA, TOC_ENTRY_RE,
        _extractive_section, _llm_summary_only_item, _llm_summary_only_section,
        _normalize_evidence, _note_values_supported, _section_evidence_units, _section_source_text,
        _split_exact_units, _verify_summary_only_result, classify_section_content, is_meta_summary,
    )
except ImportError:  # pragma: no cover
    from chunk_store import active_collection_name, get_item_chunks, list_item_keys
    from db_relations import (
        delete_section_summary, get_item_summary, get_section_summaries,
        mark_insight_generation_status, save_item_summary, save_section_summary,
    )
    from embedder import create_embedding_function, open_chroma_collection, resolve_embedder_settings
    from llm_client import DeepSeekClient, InvalidLLMResponse, LLMError, RateLimitReached, get_llm
    from manifest import load_manifest
    from env_utils import load_dotenv_native
    from codex_quota import CodexQuotaError, CodexQuotaFloorReached, require_weekly_quota
    from summary_prompts import cited_item_summary_prompt, cited_section_summary_prompt
    from summary_core import (
        CHAPTER_MARKER_RE, DEFINITIVE_IDENTIFIER_RE, META_SUMMARY_RE, NON_CONTENT_HEADING_RE,
        QUOTED_VALUE_RE, SPECIFIC_VALUE_RE, SUMMARY_ONLY_SCHEMA, TOC_ENTRY_RE,
        _extractive_section, _llm_summary_only_item, _llm_summary_only_section,
        _normalize_evidence, _note_values_supported, _section_evidence_units, _section_source_text,
        _split_exact_units, _verify_summary_only_result, classify_section_content, is_meta_summary,
    )


ROOT = Path(__file__).resolve().parents[1]
load_dotenv_native(ROOT)
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", ROOT / "data" / "chroma"))
MANIFEST_PATH = Path(os.environ.get("MANIFEST_PATH", ROOT / "data" / "manifest.json"))
SECTION_WINDOW = 40


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


def _llm_section(section: dict[str, Any]) -> tuple[dict[str, Any], str]:
    result, model = _llm_summary_only_section(section)
    # Legacy callers still display these two metadata fields.  They
    # remain empty until the canonical node-summary pipeline owns them.
    result.setdefault("chapter_authors", [])
    result.setdefault("first_publication_note", None)
    return result, model


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
    allow_model_migration: bool = False,
) -> dict[str, Any]:
    chunks = get_item_chunks(item_key)
    if not chunks:
        return {"item_key": item_key, "status": "empty"}
    manifest = load_manifest(MANIFEST_PATH)
    source_mtime = _source_mtime(chunks, manifest)
    existing = get_item_summary(item_key)
    if mode == "llm" and force and existing and not allow_model_migration:
        existing_model = str(existing.get("model") or "").casefold()
        target_client = get_llm("summary")
        target_spec = f"{target_client.provider}:{target_client.model}".casefold()
        if "luna" in existing_model and "deepseek" in target_spec:
            return {
                "item_key": item_key, "status": "protected_existing",
                "reason": "explicit --allow-model-migration is required to replace a Luna result",
            }
    same_source = (
        existing
        and existing.get("chunk_count") == len(chunks)
        and float(existing.get("source_mtime") or 0) == source_mtime
    )
    upgrading_extract = mode == "llm" and existing and existing.get("model") == "extractive"
    if not force and same_source and not upgrading_extract:
        return {"item_key": item_key, "status": "unchanged"}
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
                    "status": "skipped_non_content", "llm_summary": None,
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
                    "llm_summary": None, "chapter_authors": [],
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
    mark_insight_generation_status(item_key, "sections", len(section_rows))
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
    item_rows: list[tuple[str, str, dict[str, Any]]] = []
    section_rows: list[tuple[str, str, dict[str, Any]]] = []
    available_keys = list_item_keys(chroma_dir=CHROMA_DIR, collection_name=base)
    target_keys = [key for key in available_keys if item_keys is None or key in item_keys]
    for item_key in target_keys:
        summary = get_item_summary(item_key)
        chunks = get_item_chunks(item_key, chroma_dir=CHROMA_DIR, collection_name=base)
        metadata = _metadata(chunks)
        if summary:
            summary_model = str(summary.get("model") or "")
            document = "\n".join(filter(None, [
                str(metadata.get("title") or ""), str(metadata.get("creators") or ""),
                str(summary.get("keywords") or ""), str(summary.get("summary") or ""),
                str(summary.get("summary_en") or ""),
            ]))[:1200]
            item_rows.append((f"sum:item:{item_key}", document, {
                "itemKey": item_key, "title": str(metadata.get("title") or ""),
                "creators": str(metadata.get("creators") or ""), "year": int(metadata.get("year") or 0),
                "summary_model": summary_model,
                "summary_kind": "extractive" if summary_model == "extractive" else "llm",
            }))
        for section in get_section_summaries(item_key):
            section_model = str(section.get("model") or "")
            section_rows.append((
                f"sum:sec:{item_key}:{section['section_id']}",
                "\n".join(filter(None, [
                    str(metadata.get("title") or ""), str(section.get("chapter") or ""), section["summary"]
                ]))[:900],
                {"itemKey": item_key, "title": str(metadata.get("title") or ""),
                 "chapter": str(section.get("chapter") or ""), "section_id": section["section_id"],
                 "summary_model": section_model,
                 "summary_kind": "extractive" if section_model == "extractive" else "llm"},
            ))
    batch_size = max(1, int(os.environ.get("SUMMARY_EMBED_BATCH_SIZE", "16")))
    for collection, rows in ((item_collection, item_rows), (section_collection, section_rows)):
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
    for collection in (item_collection, section_collection):
        client = getattr(collection, "_chroma_client", None)
        if client:
            client.close()
    return {"items": len(item_rows), "sections": len(section_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-model-migration", action="store_true",
        help="Allow --force to replace an existing result produced by another model family.",
    )
    parser.add_argument("--mode", choices=("extractive", "llm"), default="extractive")
    parser.add_argument("--no-embed", action="store_true")
    parser.add_argument("--embed-only", action="store_true")
    parser.add_argument("--resume-embed", action="store_true", help="Embed only IDs not already present.")
    parser.add_argument("--llm", help="Override LLM_CHEAP for this run, e.g. deepseek:model.")
    parser.add_argument(
        "--audit-output", type=Path,
        help="Write per-item grounding verification statistics as JSON.",
    )
    parser.add_argument("--max-items", type=int, help="Maximum items to update in this run.")
    parser.add_argument("--max-hours", type=float, help="Stop before starting another item after this many hours.")
    parser.add_argument(
        "--stop-file", type=Path,
        help="Finish the current item, then stop while this file exists.",
    )
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
        os.environ["LLM_CHEAP"] = args.llm
    counts = Counter()
    updated_keys: set[str] = set()
    started = time.monotonic()
    processed = 0
    stop_reason = "completed"
    signal_state = {"requested": False}

    def request_stop(signum: int, _frame: Any) -> None:
        signal_state["requested"] = True
        print(f"Received signal {signum}; stopping after the current item.", flush=True)

    previous_handlers = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    audit_items: list[dict[str, Any]] = []
    try:
        if not args.embed_only:
            keys = [args.item] if args.item else list_item_keys()
            for index, key in enumerate(keys, start=1):
                if signal_state["requested"] or (
                    args.stop_file is not None and args.stop_file.exists()
                ):
                    stop_reason = "stop_requested"
                    break
                if args.max_items is not None and counts["updated"] >= args.max_items:
                    stop_reason = "max_items"
                    break
                if args.max_hours is not None and time.monotonic() - started >= args.max_hours * 3600:
                    stop_reason = "max_hours"
                    break
                try:
                    quota_guard = None
                    if args.min_weekly_remaining_percent is not None:
                        quota_guard = lambda: require_weekly_quota(
                            args.min_weekly_remaining_percent,
                        )
                    result = build_item(
                        key, mode=args.mode, force=args.force, quota_guard=quota_guard,
                        allow_model_migration=args.allow_model_migration,
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
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    if not args.no_embed and (args.embed_only or updated_keys or args.resume_embed):
        embed_keys = None if args.embed_only or (args.resume_embed and not updated_keys) else updated_keys
        embedded = embed_summaries(
            item_keys=embed_keys,
            only_missing=args.resume_embed,
        )
        print(
            f"Embedded {embedded['items']} item summaries and {embedded['sections']} section summaries."
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
