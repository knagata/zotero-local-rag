#!/usr/bin/env python3
"""Advance one safe Mistral OCR Batch maintenance phase without prompting."""
from __future__ import annotations

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


def _run(*arguments: str) -> None:
    subprocess.run((sys.executable, *arguments), cwd=ROOT, check=True)


def _save(state: dict) -> None:
    encoded = (json.dumps(state, ensure_ascii=False, indent=2) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=STATE.parent, delete=False) as stream:
        stream.write(encoded)
        stream.flush()
        candidate = Path(stream.name)
    candidate.replace(STATE)


def _queued_candidate_count() -> int:
    """Return newly blocked extraction artifacts awaiting a Mistral batch."""
    connection = sqlite3.connect(
        f"{RELATIONS_DB.resolve().as_uri()}?mode=ro", uri=True, timeout=5,
    )
    try:
        row = connection.execute(
            """
            SELECT COUNT(*)
            FROM artifact_processing_status
            WHERE artifact_type = 'extraction'
              AND status = 'blocked'
              AND reason_code = 'awaiting_mistral_ocr_batch'
              AND attachment_key <> ''
            """
        ).fetchone()
        return int(row[0] if row else 0)
    finally:
        connection.close()


def _adopt(state: dict) -> None:
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
    print(f"[MISTRAL OCR] 品質検証合格 {count}件をV3へ採用します。", flush=True)
    _run(str(INDEXER), "--reocr-candidates", str(queue), "--progress")
    _run(str(STRUCTURE), "--all")
    state = _load()
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
    _run(*arguments)
    print("[案内] 再試行Batchを送信しました。完了後にこの処理を再実行してください。", flush=True)
    return True


def main() -> int:
    state = _load()
    phase = str(state.get("phase") or "").casefold()
    if phase in {"", "prepared", "uploaded"}:
        _run(str(BATCH), "--submit", "--state", str(STATE))
        print("[案内] Batchを送信しました。完了後にこの処理を再実行してください。", flush=True)
        return 0
    if phase in {"submitted", "queued", "running", "in_progress"}:
        _run(str(BATCH), "--status", "--state", str(STATE))
        state = _load()
        phase = str(state.get("phase") or "").casefold()
        if phase != "success":
            print(f"[案内] Batchは処理中です（状態={phase or '不明'}）。", flush=True)
            return 0
    if phase == "success":
        _run(str(BATCH), "--collect", "--state", str(STATE))
        state = _load()
        phase = str(state.get("phase") or "").casefold()
    if phase == "collected":
        if not state.get("adoption_applied_at"):
            _adopt(state)
            if int(state.get("adoptable_count") or 0):
                return 0
            state = _load()
        if _submit_retryable(state):
            return 0
        queued = _queued_candidate_count()
        if queued:
            print(f"[MISTRAL OCR] 新しい待機資料 {queued}件をBatchへ送信します。", flush=True)
            _run(str(BATCH), "--submit", "--state", str(STATE))
            print("[案内] Batchを送信しました。完了後にこの処理を再実行してください。", flush=True)
        else:
            print("[情報] 品質検証合格分は採用済みで、新しい待機資料はありません。", flush=True)
        return 0
    if phase in {"failed", "cancelled", "timeout_exceeded"}:
        raise RuntimeError(f"Mistral Batch is in terminal failure state: {phase}")
    raise RuntimeError(f"Unknown Mistral Batch phase: {phase or '<empty>'}")


if __name__ == "__main__":
    raise SystemExit(main())
