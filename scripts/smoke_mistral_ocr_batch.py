#!/usr/bin/env python3
"""Run one non-canonical PDF through the real Mistral OCR Batch API."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_utils import load_dotenv_native
from src.mistral_ocr_batch import (
    MistralBatchClient, evaluate_ocr_result, parse_batch_output, save_json_atomic,
    write_batch_jsonl,
)
from src.mistral_ocr_extract import (
    DEFAULT_BASE_URL, DEFAULT_MODEL, extract_chunks_from_mistral_ocr_result,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument("--max-wait-seconds", type=float, default=600)
    args = parser.parse_args()
    load_dotenv_native(ROOT)
    key = os.environ.get("MISTRAL_OCR_API_KEY", "").strip()
    if not key:
        raise RuntimeError("MISTRAL_OCR_API_KEY is not configured")
    model = os.environ.get("MISTRAL_OCR_MODEL", "").strip() or DEFAULT_MODEL
    client = MistralBatchClient(
        key,
        base_url=os.environ.get("MISTRAL_OCR_BASE_URL", "").strip() or DEFAULT_BASE_URL,
    )
    args.work_dir.mkdir(parents=True, exist_ok=True)
    input_jsonl = args.work_dir / "input.jsonl"
    write_batch_jsonl(
        [({"attachment_key": "synthetic-smoke"}, args.input.resolve())],
        input_jsonl,
    )
    state_path = args.work_dir / "state.json"
    previous = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    file_id = str(previous.get("input_file_id") or "")
    if not file_id:
        file_id = client.upload(input_jsonl)
        save_json_atomic(state_path, {
            "input": str(args.input.resolve()), "input_file_id": file_id,
            "status": "UPLOADED", "model": model,
        })
    job = client.create_job(file_id, model=model)
    save_json_atomic(state_path, {
        "input": str(args.input.resolve()), "input_file_id": file_id,
        "job_id": job["id"], "status": job.get("status"), "model": model,
    })
    deadline = time.monotonic() + args.max_wait_seconds
    while str(job.get("status") or "").upper() in {"QUEUED", "RUNNING"}:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Batch did not finish within {args.max_wait_seconds}s; job_id={job['id']}")
        time.sleep(max(1.0, args.poll_seconds))
        job = client.get_job(str(job["id"]))
        print(json.dumps({
            "job_id": job["id"], "status": job.get("status"),
            "succeeded": job.get("succeeded_requests"), "failed": job.get("failed_requests"),
        }), flush=True)
    if str(job.get("status") or "").upper() != "SUCCESS":
        raise RuntimeError(f"Batch failed: status={job.get('status')} errors={job.get('errors')}")
    output_file = job.get("output_file")
    output = client.download_file(str(output_file))
    (args.work_dir / "output.jsonl").write_text(output, encoding="utf-8")
    parsed = parse_batch_output(output)["synthetic-smoke"]
    if not parsed["ok"]:
        raise RuntimeError(f"OCR request failed inside successful batch: {parsed.get('error')}")
    result = parsed["body"]
    gate = evaluate_ocr_result(result, args.input)
    chunks, quality = extract_chunks_from_mistral_ocr_result(
        args.input, "synthetic-smoke",
        {"itemKey": "synthetic-smoke", "attachmentKey": "synthetic-smoke"},
        result, model=model,
    )
    report = {
        "job_id": job["id"], "status": job["status"], "gate": gate,
        "chunks": len(chunks), "parser": quality.get("parser"),
        "returned_pages": quality.get("total_pages"),
    }
    save_json_atomic(args.work_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if gate["passed"] and chunks else 2


if __name__ == "__main__":
    raise SystemExit(main())
