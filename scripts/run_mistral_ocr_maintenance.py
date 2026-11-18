#!/usr/bin/env python3
"""Advance one safe Mistral OCR Batch maintenance phase without prompting."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = Path(os.environ.get("MISTRAL_BATCH_STATE_PATH", "data/mistral_ocr_batch_state.json"))
if not STATE.is_absolute():
    STATE = ROOT / STATE
BATCH = ROOT / "scripts" / "run_mistral_ocr_batch.py"
STRUCTURE = ROOT / "scripts" / "rebuild_document_structure.py"
INDEXER = ROOT / "src" / "index_from_zotero.py"
RELATIONS_DB = ROOT / "data" / "relations.db"


def _load() -> dict:
    if not STATE.exists():
        return {}
    return json.loads(STATE.read_text(encoding="utf-8"))


def _run(*arguments: str, environment: dict | None = None) -> None:
    options = {"env": environment} if environment is not None else {}
    subprocess.run((sys.executable, *arguments), cwd=ROOT, check=True, **options)


def _save(state: dict) -> None:
    encoded = (json.dumps(state, ensure_ascii=False, indent=2) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=STATE.parent, delete=False) as stream:
        stream.write(encoded)
        stream.flush()
        candidate = Path(stream.name)
    candidate.replace(STATE)


def _queued_candidate_count(*, exclude_attachments: set[str] | None = None) -> int:
    """Return newly blocked extraction artifacts awaiting a Mistral batch."""
    connection = sqlite3.connect(
        f"{RELATIONS_DB.resolve().as_uri()}?mode=ro", uri=True, timeout=5,
    )
    try:
        rows = connection.execute(
            """
            SELECT DISTINCT attachment_key
            FROM artifact_processing_status
            WHERE artifact_type = 'extraction'
              AND status = 'blocked'
              AND reason_code = 'awaiting_mistral_ocr_batch'
              AND attachment_key <> ''
            """
        ).fetchall()
        return len({str(row[0]) for row in rows} - (exclude_attachments or set()))
    finally:
        connection.close()


def _adopt(state: dict, *, local_only: bool = False) -> None:
    if state.get("adoption_applied_at"):
        print("[情報] 品質検証合格分は採用済みです。", flush=True)
        return
    count = int(state.get("adoptable_count") or 0)
    if count == 0:
        print("[情報] 品質検証を通過した結果はありません。", flush=True)
        state["adoption_applied_at"] = datetime.now(timezone.utc).isoformat()
        _save(state)
        return
    queue = Path(str(state.get("adoption_queue") or ""))
    if not queue.is_absolute():
        queue = ROOT / queue
    if not queue.is_file():
        raise RuntimeError(f"Mistral adoption queue not found: {queue}")
    expected_job = state.get("job_id")
    print(f"[MISTRAL OCR] 品質検証合格 {count}件をV3へ採用します。", flush=True)
    if local_only:
        environment = dict(os.environ)
        for flag in ("PDF_AI_TOC_FAST_PATH_ENABLE", "OCR_LAYER_AUDIT_ENABLE",
                     "PDF_MISTRAL_TOC_QUEUE_ENABLE", "LLM_SUMMARIES_ENABLE",
                     "LLM_REFERENCE_EXTRACTION_ENABLE", "QUERY_EXPANSION_ENABLE"):
            environment[flag] = "0"
        _run(str(INDEXER), "--reocr-candidates", str(queue), "--progress", environment=environment)
        candidates = json.loads(queue.read_text(encoding="utf-8"))["candidates"]
        for item_key in sorted({row["item_key"] for row in candidates}):
            _run(str(STRUCTURE), "--item", item_key, environment=environment)
    else:
        _run(str(INDEXER), "--reocr-candidates", str(queue), "--progress")
        _run(str(STRUCTURE), "--all")
    state = _load()
    if state.get("job_id") != expected_job:
        raise RuntimeError("OCR Batch state changed during adoption; refusing to mark a different job as applied")
    state["adoption_applied_at"] = datetime.now(timezone.utc).isoformat()
    _save(state)


def _retryable_item_keys(state: dict) -> list[str]:
    return sorted({
        str(report.get("item_key") or "")
        for report in state.get("reports") or []
        if report.get("retryable") and report.get("item_key")
    })


def _submit_retryable(state: dict) -> bool:
    item_keys = _retryable_item_keys(state)
    if not item_keys:
        return False
    print(
        f"[MISTRAL OCR] 一時的なAPI失敗 {len(item_keys)}件を再送信します。",
        flush=True,
    )
    arguments = [str(BATCH), "--submit", "--state", str(STATE)]
    for item_key in item_keys:
        arguments.extend(("--item", item_key))
    for report in state.get("reports") or []:
        if report.get("attachment_key") and not report.get("retryable"):
            arguments.extend(("--exclude-attachment", str(report["attachment_key"])))
    _run(*arguments)
    print("[案内] 再試行Batchを送信しました。完了後にこの処理を再実行してください。", flush=True)
    return True


def _read_batch(action: str, *, collect_only: bool) -> bool:
    try:
        _run(str(BATCH), action, "--state", str(STATE))
    except subprocess.CalledProcessError:
        if not collect_only:
            raise
        print("[要確認] 送信済みOCR結果の確認・回収ができませんでした。再送せず、待機記録を保持して他の資料の更新を続けます。", flush=True)
        return False
    return True


def main(*, collect_only: bool = False) -> int:
    state = _load()
    phase = str(state.get("phase") or "").casefold()
    if phase in {"", "prepared", "uploaded"}:
        if collect_only:
            print("[情報] 完了済みOCRの回収だけを行います。新しい送信は行いません。", flush=True)
            return 0
        if phase == "prepared" and not state.get("candidate_count") and not _queued_candidate_count():
            print("[情報] Mistral OCRの待機資料はありません。外部送信は行いません。", flush=True)
            return 0
        _run(str(BATCH), "--submit", "--state", str(STATE))
        print("[案内] Batchを送信しました。完了後にこの処理を再実行してください。", flush=True)
        return 0
    if phase in {"submitted", "queued", "running", "in_progress"}:
        if not _read_batch("--status", collect_only=collect_only):
            return 0
        state = _load()
        phase = str(state.get("phase") or "").casefold()
        if phase in {"failed", "cancelled", "timeout_exceeded"}:
            if collect_only:
                print(f"[要確認] 既存OCR Batchは失敗しています（{phase}）。有料の再送信は行いません。", flush=True)
                return 0
            raise RuntimeError(f"Mistral Batch is in terminal failure state: {phase}")
        if phase != "success":
            print(f"[案内] Batchは処理中です（状態={phase or '不明'}）。", flush=True)
            return 0
    if phase == "success":
        if not _read_batch("--collect", collect_only=collect_only):
            return 0
        state = _load()
        phase = str(state.get("phase") or "").casefold()
    if phase == "collected":
        if not state.get("adoption_applied_at"):
            if collect_only:
                _adopt(state, local_only=True)
            else:
                _adopt(state)
            if int(state.get("adoptable_count") or 0):
                return 0
            state = _load()
        if collect_only:
            print("[情報] 回収・採用を確認しました。失敗資料や新しい待機資料の再送信は行いません。", flush=True)
            return 0
        if _submit_retryable(state):
            return 0
        processed = {str(row.get("attachment_key")) for row in
                     [*(state.get("candidates") or []), *(state.get("reports") or [])]
                     if row.get("attachment_key")} | set(state.get("review_required_attachments") or [])
        queued = (_queued_candidate_count(exclude_attachments=processed)
                  if processed else _queued_candidate_count())
        if queued:
            print(f"[MISTRAL OCR] 新しい待機資料 {queued}件をBatchへ送信します。", flush=True)
            arguments = [str(BATCH), "--submit", "--state", str(STATE)]
            for attachment_key in sorted(processed):
                arguments.extend(("--exclude-attachment", attachment_key))
            _run(*arguments)
            print("[案内] Batchを送信しました。完了後にこの処理を再実行してください。", flush=True)
        else:
            print("[情報] 品質検証合格分は採用済みで、新しい待機資料はありません。", flush=True)
        return 0
    if phase in {"failed", "cancelled", "timeout_exceeded"}:
        if collect_only:
            print(f"[要確認] OCR Batchは{phase}です。再送信には課金同意が必要です。", flush=True)
            return 0
        raise RuntimeError(f"Mistral Batch is in terminal failure state: {phase}")
    raise RuntimeError(f"Unknown Mistral Batch phase: {phase or '<empty>'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-only", action="store_true", help="Resume existing results without submitting paid work")
    raise SystemExit(main(collect_only=parser.parse_args().collect_only))
