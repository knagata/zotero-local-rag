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
