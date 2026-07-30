#!/usr/bin/env python3
"""Prepare, submit, resume, and collect the deferred Mistral OCR Batch queue."""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_utils import load_dotenv_native
from src.mistral_ocr_batch import (
    MistralBatchClient, evaluate_ocr_result, parse_batch_output, save_json_atomic,
    source_fingerprint, source_matches, write_batch_jsonl,
)
from src.mistral_ocr_extract import DEFAULT_BASE_URL, DEFAULT_MODEL, mistral_ocr_available
from src.zotero_source_localapi import ZoteroLocalAPI
from scripts.list_mistral_toc_candidates import REASON, build_queue_rows
from src.db_relations import get_artifact_processing_statuses

DEFAULT_STATE = ROOT / "data" / "mistral_ocr_batch_state.json"
DEFAULT_WORK = ROOT / "data" / "mistral_ocr_batches"
TERMINAL = {"SUCCESS", "FAILED", "CANCELLED", "TIMEOUT_EXCEEDED"}
DEFAULT_MAX_INPUT_BYTES = 100 * 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_state(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


async def resolve_candidates(item_filters: list[str] | None) -> list[tuple[dict[str, Any], Path]]:
    rows = build_queue_rows(get_artifact_processing_statuses(
        artifact_type="extraction", reason_code=REASON,
    ))
    if item_filters:
        allowed = set(item_filters)
        rows = [row for row in rows if row["item_key"] in allowed]
    api = ZoteroLocalAPI()
    data_dir = os.environ.get("ZOTERO_DATA_DIR")
    attachments = await api.list_normalized_attachments(
        zotero_data_dir=data_dir,
        pdf_cache_dir=str(ROOT / "data" / "pdf_cache"),
    )
    by_key = {attachment.attachmentKey: attachment for attachment in attachments}
    resolved: list[tuple[dict[str, Any], Path]] = []
    for row in rows:
        attachment = by_key.get(row["attachment_key"])
        if not attachment:
            print(f"[WARN] unresolved attachment={row['attachment_key']}", file=sys.stderr)
            continue
        source_path = Path(attachment.pdf_path).expanduser()
        batch_path = Path(str(row.get("batch_document_path") or source_path)).expanduser()
        if not source_path.is_file() or batch_path.suffix.casefold() != ".pdf" or not batch_path.is_file():
            print(
                f"[WARN] unresolved source/batch document attachment={row['attachment_key']} "
                f"source={source_path} batch={batch_path}", file=sys.stderr,
            )
            continue
        allowed_cloud, reason = mistral_ocr_available()
        if not allowed_cloud:
            print(f"[WARN] cloud policy blocked item={row['item_key']}: {reason}", file=sys.stderr)
            continue
        matches, mismatch_reason = source_matches(row, source_path)
        if not matches:
            print(
                f"[WARN] stale queue attachment={row['attachment_key']}: {mismatch_reason}",
                file=sys.stderr,
            )
            continue
        row = dict(row)
        row.update(source_fingerprint(source_path))
        row["source_path"] = str(source_path)
        row["batch_document_path"] = str(batch_path)
        row["pdf_path"] = str(batch_path)
        resolved.append((row, batch_path))
    return resolved


def estimated_batch_request_bytes(pdf_path: Path) -> int:
    """Conservative size estimate for one inline-PDF JSONL request.

    The Batch API request embeds the PDF as base64, whose payload is about 4/3
    of the source bytes.  Keep a small fixed allowance for JSON framing.
    """
    return math.ceil(pdf_path.stat().st_size / 3) * 4 + 4096


def select_input_batch(
    resolved: list[tuple[dict[str, Any], Path]], max_input_bytes: int,
) -> list[tuple[dict[str, Any], Path]]:
    """Choose a deterministic, bounded prefix without starving a large PDF."""
    selected: list[tuple[dict[str, Any], Path]] = []
    total = 0
    for row, pdf_path in resolved:
        estimate = estimated_batch_request_bytes(pdf_path)
        if selected and total + estimate > max_input_bytes:
            break
        selected.append((row, pdf_path))
        total += estimate
    return selected


def split_input_batches(
    resolved: list[tuple[dict[str, Any], Path]], max_input_bytes: int,
) -> list[list[tuple[dict[str, Any], Path]]]:
    """Partition every candidate into bounded, deterministic upload files."""
    remaining = list(resolved)
    batches: list[list[tuple[dict[str, Any], Path]]] = []
    while remaining:
        batch = select_input_batch(remaining, max_input_bytes)
        batches.append(batch)
        remaining = remaining[len(batch):]
    return batches


def client_from_env() -> MistralBatchClient:
    key = os.environ.get("MISTRAL_OCR_API_KEY", "").strip()
    if not key:
        raise RuntimeError("MISTRAL_OCR_API_KEY is not configured")
    return MistralBatchClient(
        key,
        base_url=os.environ.get("MISTRAL_OCR_BASE_URL", "").strip() or DEFAULT_BASE_URL,
        timeout=max(1.0, float(os.environ.get("MISTRAL_OCR_TIMEOUT_SEC", "120"))),
    )


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    resolved = asyncio.run(resolve_candidates(args.item))
    batches = split_input_batches(resolved, args.max_input_bytes)
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    work_dir = args.work_dir / run_id
    input_files: list[dict[str, Any]] = []
    for index, batch in enumerate(batches, start=1):
        input_path = work_dir / f"input-{index:03d}.jsonl"
        count = write_batch_jsonl(batch, input_path)
        input_files.append({
            "input_path": str(input_path),
            "candidate_count": count,
            "estimated_bytes": sum(estimated_batch_request_bytes(path) for _, path in batch),
        })
    state = {
        "schema_version": "mistral-ocr-batch-v1",
        "phase": "prepared",
        "created_at": utc_now(),
        "model": args.model,
        "work_dir": str(work_dir),
        "input_files": input_files,
        "candidates": [row for row, _ in resolved],
        "candidate_count": len(resolved),
        "input_batch_count": len(input_files),
        "max_input_bytes": args.max_input_bytes,
    }
    save_json_atomic(args.state, state)
    return state


def submit(args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    # A crash between create_job() succeeding and its result being persisted
    # used to leave phase="uploaded" with every input_file_id populated, so a
    # retry saw no pending uploads and called client.create_job() again --
    # submitting a second, duplicate billed batch job for the same documents
    # (found in code review, fixed 2026-07-30). phase="submitting" is written
    # just before the call so that exact crash window fails closed on retry
    # instead of silently resubmitting.
    if state.get("phase") == "submitting":
        raise RuntimeError(
            "state is in phase='submitting': a previous run may have already "
            "created a billed Mistral OCR batch job before crashing partway "
            "through recording that fact. Check the Mistral dashboard/API for "
            "a job matching this batch's uploaded input files before retrying -- "
            "resubmitting here could create a duplicate paid job. Once confirmed, "
            "edit the state file's phase back to 'uploaded' (or 'submitted' with "
            "the real job_id, if one was created) and retry."
        )
    if state.get("phase") not in {"prepared", "uploaded"}:
        raise RuntimeError(f"state must be prepared/uploaded before submit (phase={state.get('phase')})")
    input_files = state.get("input_files")
    if not isinstance(input_files, list):
        legacy_path = Path(str(state.get("input_path") or ""))
        input_files = [{"input_path": str(legacy_path)}] if legacy_path else []
        state["input_files"] = input_files
    if not input_files or not state.get("candidate_count"):
        raise RuntimeError("prepared batch has no candidates")
    for entry in input_files:
        if not Path(str(entry.get("input_path") or "")).is_file():
            raise RuntimeError("prepared batch input file is missing")
    for row in state.get("candidates") or []:
        source_path = Path(str(row.get("source_path") or row.get("pdf_path") or ""))
        matches, reason = source_matches(row, source_path)
        if not matches:
            raise RuntimeError(f"refusing stale batch attachment={row.get('attachment_key')}: {reason}")
        allowed_cloud, reason = mistral_ocr_available()
        if not allowed_cloud:
            raise RuntimeError(f"cloud policy changed item={row.get('item_key')}: {reason}")
    client = client_from_env()
    pending = [entry for entry in input_files if not entry.get("input_file_id")]
    if pending:
        with ThreadPoolExecutor(max_workers=args.upload_workers) as executor:
            futures = {
                executor.submit(client.upload, Path(str(entry["input_path"]))): entry
                for entry in pending
            }
            for future in as_completed(futures):
                entry = futures[future]
                entry["input_file_id"] = future.result()
                state.update({"phase": "uploaded", "uploaded_at": utc_now()})
                save_json_atomic(args.state, state)
    input_file_ids = [str(entry["input_file_id"]) for entry in input_files]
    state.update({"phase": "submitting", "submit_attempted_at": utc_now()})
    save_json_atomic(args.state, state)
    job = client.create_job(input_file_ids, model=str(state["model"]), timeout_hours=args.timeout_hours)
    state.update({
        "phase": "submitted", "submitted_at": utc_now(),
        "input_file_ids": input_file_ids, "job_id": str(job["id"]), "job": job,
    })
    save_json_atomic(args.state, state)
    return state


def status(args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    if not state.get("job_id"):
        raise RuntimeError("state has no submitted job_id")
    job = client_from_env().get_job(str(state["job_id"]))
    job_status = str(job.get("status") or "").upper()
    state.update({"job": job, "last_checked_at": utc_now()})
    if job_status in TERMINAL:
        state["phase"] = job_status.casefold()
    save_json_atomic(args.state, state)
    return state


def collect(args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    job = state.get("job") if isinstance(state.get("job"), dict) else {}
    if str(job.get("status") or "").upper() != "SUCCESS":
        state = status(args, state)
        job = state.get("job") if isinstance(state.get("job"), dict) else {}
    if str(job.get("status") or "").upper() != "SUCCESS":
        raise RuntimeError(f"batch is not successful (status={job.get('status')})")
    output_file_id = job.get("output_file") or job.get("output_file_id")
    if isinstance(output_file_id, dict):
        output_file_id = output_file_id.get("id")
    if not output_file_id:
        raise RuntimeError("successful batch has no output file")
    output_text = client_from_env().download_file(str(output_file_id))
    work_dir = Path(str(state["work_dir"]))
    output_path = work_dir / "output.jsonl"
    output_path.write_text(output_text, encoding="utf-8")
    results = parse_batch_output(output_text)
    adoption_rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    result_dir = work_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    for row in state.get("candidates") or []:
        attachment_key = str(row["attachment_key"])
        entry = results.get(attachment_key) or {"ok": False, "error": "missing batch output"}
        report: dict[str, Any] = {"attachment_key": attachment_key, "ok": bool(entry.get("ok"))}
        if entry.get("ok"):
            source_path = Path(str(row.get("source_path") or row["pdf_path"]))
            batch_path = Path(str(row.get("batch_document_path") or row["pdf_path"]))
            matches, mismatch_reason = source_matches(row, source_path)
            gate = evaluate_ocr_result(entry["body"], batch_path) if matches else {
                "passed": False, "reasons": [mismatch_reason],
            }
            report["gate"] = gate
            if gate["passed"]:
                result_path = result_dir / f"{attachment_key}.json"
                result_path.write_text(
                    json.dumps(entry["body"], ensure_ascii=False) + "\n", encoding="utf-8",
                )
                adoption_rows.append({
                    **row,
                    "target_engine": "mistral_ocr",
                    "mistral_result_path": str(result_path),
                    "batch_job_id": str(state["job_id"]),
                    "model": str(state["model"]),
                    "mistral_problem_pages": gate.get("problem_pages") or {},
                })
        else:
            report["error"] = entry.get("error")
        reports.append(report)
    adoption_queue = work_dir / "adoption_queue.json"
    save_json_atomic(adoption_queue, {
        "schema_version": "mistral-ocr-adoption-v1",
        "batch_job_id": str(state["job_id"]),
        "candidates": adoption_rows,
    })
    state.update({
        "phase": "collected", "collected_at": utc_now(),
        "output_path": str(output_path), "adoption_queue": str(adoption_queue),
        "reports": reports, "adoptable_count": len(adoption_rows),
    })
    save_json_atomic(args.state, state)
    return state


def main(argv: list[str] | None = None) -> int:
    load_dotenv_native(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--submit", action="store_true", help="Prepare if needed, then upload and submit.")
    actions.add_argument("--status", action="store_true", help="Refresh the saved job status.")
    actions.add_argument("--collect", action="store_true", help="Download results and build a gated adoption queue.")
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--item", action="append", help="Limit preparation to itemKey; repeatable.")
    parser.add_argument("--model", default=os.environ.get("MISTRAL_OCR_MODEL") or DEFAULT_MODEL)
    parser.add_argument("--timeout-hours", type=int, default=24)
    parser.add_argument(
        "--max-input-bytes", type=int,
        default=int(os.environ.get("MISTRAL_OCR_BATCH_MAX_INPUT_BYTES", DEFAULT_MAX_INPUT_BYTES)),
        help="Maximum estimated base64 JSONL upload size per Batch (default: 100 MiB).",
    )
    parser.add_argument(
        "--upload-workers", type=int,
        default=int(os.environ.get("MISTRAL_OCR_BATCH_UPLOAD_WORKERS", "3")),
        help="Maximum concurrent input-file uploads (default: 3).",
    )
    args = parser.parse_args(argv)
    if args.max_input_bytes < 1:
        parser.error("--max-input-bytes must be positive")
    if args.upload_workers < 1:
        parser.error("--upload-workers must be positive")
    state = load_state(args.state)
    if args.status:
        state = status(args, state)
    elif args.collect:
        state = collect(args, state)
    elif args.submit:
        terminal_phases = {value.casefold() for value in TERMINAL}
        if not state or state.get("phase") in terminal_phases or state.get("phase") == "collected":
            state = prepare(args)
        state = submit(args, state)
    else:
        state = prepare(args)
    print(json.dumps({
        "phase": state.get("phase"), "candidates": state.get("candidate_count", 0),
        "input_batches": state.get("input_batch_count", len(state.get("input_files") or [])),
        "job_id": state.get("job_id"), "adoptable": state.get("adoptable_count"),
        "state": str(args.state),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
