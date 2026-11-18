"""Index into the canonical plane while searches stay pinned to a saved copy."""
from __future__ import annotations

import os
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.env_utils import dotenv_values_native
from src.search_snapshot import load_snapshot, writer_environment
from src.v3_data_plane import resolve_configured_path


def resume_existing_ocr(environment: dict) -> None:
    configured = dotenv_values_native(ROOT)
    canonical = resolve_configured_path(ROOT, configured.get("CHROMA_DIR", ROOT / "data/chroma"))
    target = resolve_configured_path(ROOT, environment.get("CHROMA_DIR", canonical))
    if target.resolve() != canonical.resolve() and not environment.get("MISTRAL_BATCH_STATE_PATH"):
        print("[情報] 別データ領域の取込では、稼働側OCR状態を採用済みに変更しません。", flush=True)
        return
    state = resolve_configured_path(ROOT, environment.get(
        "MISTRAL_BATCH_STATE_PATH", ROOT / "data" / "mistral_ocr_batch_state.json"))
    if not state.exists():
        return
    try:
        batch = json.loads(state.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"[要確認] OCR状態を読めません。通常更新は続けますが、OCR待ちの未完了は監査に残します：{exc}", flush=True)
        return
    if not isinstance(batch, dict):
        print("[要確認] OCR状態の形式が不正です。OCR待ちを保持して通常更新を続けます。", flush=True)
        return
    if batch.get("phase") in {"submitted", "queued", "running", "in_progress", "success", "collected"}:
        print("[OCR結果確認] 送信済み結果を確認し、品質合格分を採用します。追加のOCR送信は行いません。", flush=True)
        subprocess.run([
            sys.executable, str(ROOT / "scripts" / "run_mistral_ocr_maintenance.py"), "--collect-only",
        ], cwd=ROOT, env=environment, check=True)


def main() -> int:
    # Preserve explicit caller overrides, especially hosted features forced
    # off for a local-only run. Reader paths must never leak to the writer.
    environment = {**dotenv_values_native(ROOT), **writer_environment(os.environ)}
    if load_snapshot(ROOT) is None:
        manifest = resolve_configured_path(ROOT, environment.get("MANIFEST_PATH", ROOT / "data" / "manifest_v3.json"))
        data = json.loads(manifest.read_text()) if manifest.exists() else {}
        if (data.get("hnsw_validated") is True and not data.get("inflight_attachments")
                and not data.get("post_index_pending")):
            subprocess.run([
                sys.executable, str(ROOT / "scripts" / "publish_search_snapshot.py"),
                "--initialize", "--restart-services",
            ], cwd=ROOT, env=environment, check=True)
    resume_existing_ocr(environment)
    snapshot = load_snapshot(ROOT)
    if snapshot:
        print(f"[検索継続] 更新中の検索には保存済みの{snapshot['attachments']}添付を使用します。", flush=True)
    else:
        print("[検索DB未作成] まだ検証済み索引がありません。今回の索引構築中は検索を保留します。", flush=True)
    return subprocess.run([
        sys.executable, str(ROOT / "src" / "index_from_zotero.py"), *sys.argv[1:],
    ], cwd=ROOT, env=environment, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
