#!/usr/bin/env python3
"""Generate V3 document-node summaries for one item or the local library."""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_utils import load_dotenv_native
load_dotenv_native(ROOT)

from src.build_structure_summaries import (
    build_structure_summaries, embed_structure_summaries,
    structure_summaries_are_current,
)
from src.chunk_store import list_item_keys
from src.db_relations import get_item_processing_status, mark_artifact_status
from src.database_gate import validate_database_gate
from src.summary_core import set_summary_progress_callback
from src.v3_data_plane import V3_COLLECTION


class SummaryProgressReporter:
    """Thread-safe, line-oriented progress for long paid summary runs."""

    def __init__(self, total_items: int) -> None:
        self.total_items = total_items
        self.sent = 0
        self.responses = 0
        self.errors = 0
        self.completed = 0
        self.status_counts: dict[str, int] = {}
        self.lock = threading.Lock()

    def api_event(self, event: str, details: dict) -> None:
        with self.lock:
            if event == "request":
                self.sent += 1
                print(
                    f"[SUMMARY PROGRESS] API送信 {self.sent} / 応答 {self.responses} / "
                    f"失敗 {self.errors} / 処理中 {self.sent - self.responses - self.errors} / "
                    f"種別 {details.get('kind', '')}",
                    flush=True,
                )
            elif event == "response":
                self.responses += 1
                verification = details.get("verification") or {}
                print(
                    f"[SUMMARY PROGRESS] API応答 {self.responses}/{self.sent} / "
                    f"失敗 {self.errors} / 処理中 {self.sent - self.responses - self.errors} / 根拠確認 "
                    f"{verification.get('kept_sentences', 0)}/"
                    f"{verification.get('generated_sentences', 0)}文",
                    flush=True,
                )
            elif event == "error":
                self.errors += 1
                print(
                    f"[SUMMARY PROGRESS] API失敗 {self.errors} / 送信 {self.sent} / "
                    f"応答 {self.responses} / 処理中 {self.sent - self.responses - self.errors} / "
                    f"種別 {details.get('error', '')}",
                    flush=True,
                )

    def item_completed(self, result: dict) -> None:
        with self.lock:
            self.completed += 1
            status = str(result.get("status") or "unknown")
            self.status_counts[status] = self.status_counts.get(status, 0) + 1
            statuses = ", ".join(
                f"{key}={value}" for key, value in sorted(self.status_counts.items())
            )
            print(
                f"[SUMMARY PROGRESS] itemバッチ {self.completed}/{self.total_items} 完了 "
                f"({statuses}) / API送信 {self.sent} / 応答 {self.responses} / "
                f"失敗 {self.errors} / 処理中 {self.sent - self.responses - self.errors}",
                flush=True,
            )

    def embedding_progress(self, completed: int, total: int) -> None:
        with self.lock:
            print(
                f"[SUMMARY PROGRESS] 要約索引 {completed}/{total} 件を埋め込み済み",
                flush=True,
            )


def _embedding_item_keys(
    *, all_items: bool, limit: int, results: list[dict],
) -> set[str] | None:
    """Return the smallest safe summary-index scope for this run.

    ``None`` means a deliberate unlimited ``--all`` reconciliation.  A
    limited maintenance batch must never turn into a full-library re-embed,
    and current items need no index write at all.
    """
    successful = [row for row in results if row.get("status") != "failed"]
    if all_items and limit == 0 and successful:
        return None
    return {
        str(row.get("item_key") or "")
        for row in successful
        if row.get("status") != "skipped_current" and row.get("item_key")
    }


def _select_batch_keys(
    keys: list[str], *, all_items: bool, limit: int, force: bool,
    retry_failed: bool, mode: str,
) -> list[str]:
    """Apply a maintenance limit after removing already-current items."""
    selected = list(keys)
    if all_items and limit > 0 and not force and not retry_failed:
        selected = [
            key for key in selected
            if not structure_summaries_are_current(key, mode=mode)
        ]
    return selected[:limit] if limit > 0 else selected


def _embed_completed_summaries(
    args: argparse.Namespace, output: list[dict], succeeded: list[str],
    progress: SummaryProgressReporter,
) -> dict | None:
    scope = _embedding_item_keys(
        all_items=bool(args.all), limit=int(args.limit), results=output,
    )
    if not args.embed or scope == set():
        return None
    embedding = embed_structure_summaries(
        item_keys=scope,
        base_collection_name=args.collection,
        progress_callback=progress.embedding_progress,
    )
    marked_keys = succeeded if scope is None else sorted(scope)
    for key in marked_keys:
        mark_artifact_status(
            key, "summary_index", "success", processor_version="structure-v3-1",
            counts=embedding, fallback_kind="llm_node_summary_index",
        )
    return embedding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--item", action="append")
    selector.add_argument("--all", action="store_true")
    parser.add_argument("--mode", choices=("extractive", "llm"), default="extractive")
    parser.add_argument("--embed", action="store_true", help="Rebuild the searchable __sum_node collection after summaries")
    parser.add_argument("--force", action="store_true", help="Regenerate summaries even if the source fingerprint and prompt already match")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Report selected items without generating summaries")
    parser.add_argument(
        "--database-gate", type=Path,
        help="Passing --new-only database audit bound to the current DB generation. "
             "Required for non-dry-run LLM summaries.",
    )
    parser.add_argument("--retry-failed", action="store_true", help="Select retryable failed summary items only")
    parser.add_argument(
        "--collection", default=V3_COLLECTION,
        help="Source collection (V3 only).",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Concurrent items to summarize (LLM calls are I/O-bound; each item is an "
             "independent unit). Each worker opens its own DB connection. Default 1 "
             "(sequential, unchanged behavior).",
    )
    args = parser.parse_args()
    if args.collection != V3_COLLECTION:
        parser.error(f"--collection must be {V3_COLLECTION!r}; the legacy data plane is retired")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.mode == "llm" and not args.dry_run:
        if not args.database_gate:
            parser.error("--mode llm requires --database-gate from audit_v3_cutover.py --new-only")
        try:
            database_report = validate_database_gate(
                args.database_gate, collection_name=args.collection,
            )
        except RuntimeError as exc:
            parser.error(str(exc))
        if not args.collection:
            # The audited collection, rather than whichever collection happens
            # to be active in this shell, owns both summary inputs and index.
            args.collection = str(database_report["new_collection"])
    keys = list(dict.fromkeys(args.item or list_item_keys(collection_name=args.collection)))
    if args.retry_failed:
        keys = [
            key for key in keys
            if any(
                row.get("artifact_type") == "summary"
                and row.get("status") == "failed" and bool(row.get("retryable"))
                for row in get_item_processing_status(key)
            )
        ]
    if args.all and args.limit > 0 and not args.force and not args.retry_failed:
        print(
            f"[SUMMARY PROGRESS] {len(keys)} itemから要約更新が必要な資料を確認中...",
            flush=True,
        )
    keys = _select_batch_keys(
        keys, all_items=bool(args.all), limit=int(args.limit), force=bool(args.force),
        retry_failed=bool(args.retry_failed), mode=args.mode,
    )
    if args.all and args.limit > 0 and not args.force and not args.retry_failed:
        print(
            f"[SUMMARY PROGRESS] 更新対象 {len(keys)} item（上限 {args.limit}）",
            flush=True,
        )
    if args.dry_run:
        print(json.dumps({
            "dry_run": True, "items": keys, "count": len(keys), "mode": args.mode,
            "collection": args.collection, "embed": bool(args.embed),
            "canonical_data_modified": False,
        }, ensure_ascii=False, indent=2))
        return
    progress = SummaryProgressReporter(len(keys))
    set_summary_progress_callback(progress.api_event if args.mode == "llm" else None)
    print(
        f"[SUMMARY PROGRESS] 開始: item 0/{len(keys)} / workers={args.workers} / mode={args.mode}",
        flush=True,
    )
    def _process(key: str) -> dict:
        try:
            result = build_structure_summaries(
                key, mode=args.mode, force=args.force, collection_name=args.collection,
            )
            return result
        except Exception as exc:
            return {"item_key": key, "status": "failed", "error": str(exc)}

    output = []
    failures = 0
    succeeded: list[str] = []
    if args.workers == 1:
        for key in keys:
            result = _process(key)
            output.append(result)
            progress.item_completed(result)
            if result.get("status") == "failed":
                failures += 1
            else:
                succeeded.append(key)
    else:
        # Items are independent units of work; each build_structure_summaries()
        # call opens its own DB connection (safe under WAL) and makes its own
        # sequential LLM calls internally. Concurrency here overlaps the
        # network-I/O wait time across items, which dominates wall-clock time.
        lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_process, key): key for key in keys}
            for future in as_completed(futures):
                result = future.result()
                with lock:
                    output.append(result)
                    progress.item_completed(result)
                    if result.get("status") == "failed":
                        failures += 1
                    else:
                        succeeded.append(result.get("item_key") or futures[future])
    embedding = _embed_completed_summaries(args, output, succeeded, progress)
    print(json.dumps({
        "items": output, "embedding": embedding, "failed": failures,
        "succeeded": len(succeeded),
    }, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
