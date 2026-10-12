"""Bounded repairs of derived data, called with the indexing writer lock held.

Never extracts text, embeds, deletes chunks, approves coverage, or changes a gate.
The caller must run all three unchanged audits again after a repair.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path

STRUCTURE_REASONS = frozenset({
    "stale_structure_fingerprint", "stale_structure_version", "missing_v3_structure",
    "structure_attachment_coverage_mismatch", "incomplete_node_coverage",
})
MAX_ITEMS = 20
MAX_LEXICAL_ROWS = 10000


def repair_plan(report: dict, rows: list[dict], lexical_ids: list[str]) -> dict:
    """Only explicit known failures may authorize a derived-data repair."""
    targets = set()
    for item in report.get("items", []):
        reasons = set(item.get("failures", []))
        if reasons and reasons <= STRUCTURE_REASONS:
            targets.add(str(item.get("item_key") or ""))
    dangling = set(report.get("dangling_node_chunks", []))
    targets.update(str(row["metadata"].get("itemKey") or "")
                   for row in rows if row["id"] in dangling)
    targets.discard("")
    missing = []
    if {"chroma_lexical_id_mismatch", "lexical_content_mismatch"} & set(report.get("gate", {}).get("global_failures", [])):
        current = {row["id"] for row in rows}
        # Extra FTS IDs need investigation; never delete them automatically.
        if set(lexical_ids) <= current and len(lexical_ids) == len(set(lexical_ids)):
            existing = set(lexical_ids)
            stale = set(report.get("lexical_content_mismatches", []))
            missing = [row for row in rows if row["id"] not in existing or row["id"] in stale]
    if len(targets) > MAX_ITEMS or len(missing) > MAX_LEXICAL_ROWS:
        raise RuntimeError("自動修復の件数上限を超えています。対象を調査して限定修復してください。")
    return {"items": sorted(targets), "lexical_rows": missing}


def _backup(root: Path, rows: list[dict], lexical: Path, relations: Path) -> Path:
    """SQLite backup includes WAL; original Chroma rows preserve changed metadata."""
    destination = root / "data" / "backups" / f"audit-repair-{uuid.uuid4().hex}"
    resolved = destination.resolve()
    for source in (lexical, relations):
        if resolved == source.resolve() or resolved in source.resolve().parents:
            raise RuntimeError("修復バックアップと稼働DBのパスが重複しています。")
    destination.mkdir(parents=True, exist_ok=False)
    print(f"[自動修復] 修復前バックアップを作成・検証します: {destination}", flush=True)
    marker = destination / "INCOMPLETE"
    marker.write_text("backup pending\n")
    for label, source in (("lexical.sqlite3", lexical), ("relations.sqlite3", relations)):
        with closing(sqlite3.connect(f"{source.resolve().as_uri()}?mode=ro", uri=True)) as original:
            with closing(sqlite3.connect(destination / label)) as copy:
                original.backup(copy)
                if copy.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise RuntimeError("修復バックアップのSQLite検証に失敗しました。")
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True)
    saved = destination / "original-chunks.json"
    saved.write_text(payload, encoding="utf-8")
    if saved.read_text(encoding="utf-8") != payload:
        raise RuntimeError("修復バックアップの本文・メタデータ検証に失敗しました。")
    marker.unlink()
    return destination


def repair_derived_data(report: dict, root: Path) -> dict:
    from scripts.rebuild_document_structure import rebuild_item
    from src.chunk_store import get_collection_chunks
    from src.db_relations import DB_PATH
    from src.lexical_index import audit_content, list_chunk_ids, upsert_chunks
    from src.v3_data_plane import V3_COLLECTION, chroma_dir, lexical_path

    eligible = bool(report.get("dangling_node_chunks")) or any(
        set(item.get("failures", [])) and set(item["failures"]) <= STRUCTURE_REASONS
        for item in report.get("items", [])
    ) or bool({"chroma_lexical_id_mismatch", "lexical_content_mismatch"} & set(report.get("gate", {}).get("global_failures", [])))
    if not eligible:
        return {"repaired": False, "message": "安全に自動修復できる不整合はありません。"}
    rows = get_collection_chunks(chroma_dir=chroma_dir(root), collection_name=V3_COLLECTION)
    lexical = lexical_path(root)
    if "lexical_content_mismatch" in report.get("gate", {}).get("global_failures", []):
        report = {**report, "lexical_content_mismatches": audit_content(rows, path=lexical)[1]}
    plan = repair_plan(report, rows, list_chunk_ids(path=lexical))
    if not plan["items"] and not plan["lexical_rows"]:
        return {"repaired": False, "message": "安全に自動修復できる不整合はありません。"}
    run_id = f"audit-repair-{uuid.uuid4().hex}"
    # Every selected structure must pass preflight before the first write.
    for key in plan["items"]:
        preview = rebuild_item(key, dry_run=True, force=True, run_id=run_id,
                               collection_name=V3_COLLECTION)
        if (preview.get("source_refresh_error") or preview.get("status") == "blocked"
                or not preview.get("diagnostics", {}).get("valid")):
            raise RuntimeError(f"資料 {key} の修復前検査に失敗しました。原本・本文の対応を確認してください。")
    targets = set(plan["items"])
    saved_rows = [row for row in rows if row["metadata"].get("itemKey") in targets]
    saved_rows.extend(plan["lexical_rows"])
    backup = _backup(root, saved_rows, lexical, Path(DB_PATH))
    print(f"[自動修復] バックアップ: {backup}", flush=True)
    for key in plan["items"]:
        result = rebuild_item(key, dry_run=False, force=True, run_id=run_id,
                              collection_name=V3_COLLECTION)
        if result.get("source_refresh_error") or not result.get("diagnostics", {}).get("valid"):
            raise RuntimeError(f"資料 {key} の修復が完了していません。バックアップ: {backup}")
        print(f"[自動修復] 資料 {key} の文書構造・本文メタデータを同期しました。", flush=True)
    missing = plan["lexical_rows"]
    if missing:
        upsert_chunks([row["id"] for row in missing], [row["text"] for row in missing],
                      [row["metadata"] for row in missing], path=lexical)
        print(f"[自動修復] 保存済み本文から全文索引 {len(missing)}件を補完しました。", flush=True)
    return {"repaired": True, "items": plan["items"], "lexical_rows": len(missing),
            "backup": str(backup), "message": "自動修復を実行しました。全DB監査の合格を確認中です。"}
