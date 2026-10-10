"""Publish an isolated query copy without changing the indexing data plane."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.env_utils import environment_with_saved_dotenv
from src.indexing_lock import default_path, held
from src.search_snapshot import load_snapshot, pointer_path
from src.v3_data_plane import chroma_dir, lexical_path, manifest_path, resolve_configured_path

LABELS = ("dev.zotero-local-rag.remote-mcp", "dev.zotero-local-rag.citation-graph")


def clone_tree(source: Path, destination: Path) -> None:
    if any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError("Snapshot source contains a symbolic link")
    if sys.platform == "darwin":
        subprocess.run(["cp", "-cR", str(source), str(destination)], check=True)
    else:
        shutil.copytree(source, destination)


FTS_COMPAT_ERROR = "malformed inverted index for FTS5 table main.embedding_fulltext_search"


def native_nul_control_matches() -> bool:
    """Distinguish the Python/Chroma tokenizer mismatch with a fresh control."""
    import chromadb
    with tempfile.TemporaryDirectory(prefix="zotero-snapshot-fts-control-") as temporary:
        client = chromadb.PersistentClient(path=temporary)
        try:
            collection = client.create_collection("snapshot_control", embedding_function=None,
                                                  configuration={"hnsw": {"num_threads": 1}})
            collection.add(ids=["control"], documents=["abc\x00def ghijkl mnopqr"], embeddings=[[1., 0., 0.]])
        finally:
            client.close()
        with sqlite3.connect(Path(temporary) / "chroma.sqlite3") as db:
            return db.execute("PRAGMA quick_check").fetchall() == [(FTS_COMPAT_ERROR,)]


def verified_fts_compatibility(db: sqlite3.Connection) -> bool:
    count = db.execute("SELECT count(*) FROM embedding_fulltext_search_content").fetchone()[0]
    documents = db.execute("SELECT count(*) FROM embedding_metadata WHERE key='chroma:document'").fetchone()[0]
    nul = db.execute("SELECT count(*) FROM embedding_fulltext_search_content WHERE instr(c0,char(0))>0").fetchone()[0]
    mismatches = db.execute("""
        SELECT count(*) FROM embedding_fulltext_search_content f LEFT JOIN embedding_metadata m
        ON m.id=f.id AND m.key='chroma:document' WHERE m.string_value IS NULL OR f.c0 IS NOT m.string_value
    """).fetchone()[0]
    return bool(count == documents and nul and mismatches == 0 and native_nul_control_matches())


def sqlite_copy(source: Path, destination: Path, *, chroma: bool = False) -> str | None:
    with sqlite3.connect(f"{source.resolve().as_uri()}?mode=ro", uri=True) as original:
        with sqlite3.connect(destination) as copy:
            original.backup(copy)
            errors = copy.execute("PRAGMA quick_check").fetchall()
            if errors == [("ok",)]:
                return None
            if chroma and errors == [(FTS_COMPAT_ERROR,)] and verified_fts_compatibility(copy):
                note = "Python SQLite/Chroma FTS5 NUL tokenizer mismatch reproduced in a fresh native control; all stored FTS documents match metadata."
                print(f"[検索用DB] 互換性検証：{note}", flush=True)
                return note
            raise ValueError(f"SQLite snapshot verification failed: {source.name}: {errors[:3]}")


def expected_counts(directory: Path) -> dict[str, int]:
    with sqlite3.connect(f"{(directory / 'chroma.sqlite3').resolve().as_uri()}?mode=ro", uri=True) as db:
        return dict(db.execute("""
            SELECT c.name, count(e.id) FROM collections c
            JOIN segments s ON s.collection=c.id AND s.scope='METADATA'
            LEFT JOIN embeddings e ON e.segment_id=s.id GROUP BY c.name
        """))


def verify_native_copy(directory: Path, expected: dict[str, int]) -> None:
    import chromadb
    for reopen in (False, True):
        client = chromadb.PersistentClient(path=str(directory))
        try:
            for name, count in expected.items():
                collection = client.get_collection(name, embedding_function=None)
                if not reopen:
                    collection.modify(configuration={"hnsw": {"num_threads": 1}})
                if collection.count() != count:
                    raise ValueError("Snapshot native collection count differs from source")
                if count:
                    rows = collection.get(limit=1, include=["embeddings"])
                    vector = rows["embeddings"][0].tolist()
                    result = collection.query(query_embeddings=[vector], n_results=1, include=["distances"])
                    if (not result["ids"][0] or not math.isfinite(result["distances"][0][0])):
                        raise ValueError("Snapshot native query did not return a finite result")
        finally:
            client.close()


def loaded_services() -> list[str]:
    domain = f"gui/{os.getuid()}"
    return [label for label in LABELS if subprocess.run(
        ["launchctl", "print", f"{domain}/{label}"], capture_output=True, check=False,
    ).returncode == 0]


def stop_services(labels: list[str]) -> None:
    for label in labels:
        subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}/{label}"], check=True)


def restore_services(labels: list[str]) -> None:
    for label in labels:
        plist = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
        subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(plist)], check=True)


def publish(root: Path, *, initialize: bool = False, restart_services: bool = False) -> dict:
    with held(default_path(root)):
        previous = load_snapshot(root)
        prune_unused_copies(root)
        if initialize and previous is not None:
            return previous
        manifest = manifest_path(root)
        data = json.loads(manifest.read_text())
        if (data.get("hnsw_validated") is not True or data.get("inflight_attachments")
                or data.get("post_index_pending")):
            raise ValueError("Cannot publish an incomplete indexing generation")
        gate_path = root / "data" / "quality" / "server_database_gate.json"
        gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
        audit_passed = gate.get("gate", {}).get("passed") is True
        if not initialize and not audit_passed:
            raise ValueError("Refresh requires the unchanged full DB audit to pass")
        source = chroma_dir(root).resolve()
        home = (root / "data" / "search_snapshots").resolve()
        if source == home or home in source.parents or source in home.parents:
            raise ValueError("Indexing and search snapshot directories overlap")
        services = loaded_services() if restart_services else []
        stopped = []
        try:
            if previous is None:
                for label in services:
                    stop_services([label])
                    stopped.append(label)
            record = build_copy(root, home, source, manifest, data, audit_passed)
            temporary = pointer_path(root).with_suffix(f".{uuid.uuid4().hex}.tmp")
            temporary.write_text(json.dumps(record, indent=2) + "\n")
            os.replace(temporary, pointer_path(root))
        finally:
            if stopped:
                restore_services(stopped)
    if services and not stopped:
        for label in services:
            subprocess.run(["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{label}"], check=True)
    with held(default_path(root)):
        prune_unused_copies(root)
    return record


def prune_unused_copies(root: Path) -> list[str]:
    """Caller holds the writer lock; retain rollback and every open generation."""
    current = load_snapshot(root)
    inspector = shutil.which("lsof")
    if current is None or inspector is None:
        return []
    result = subprocess.run([inspector, "-n", "-F", "n"], capture_output=True, text=True, check=True)
    opened = [Path(line[1:]) for line in result.stdout.splitlines() if line.startswith("n/")]
    keep = {current["directory"], current.get("previous")}
    home = root / "data" / "search_snapshots"
    removed = []
    for directory in home.iterdir():
        if (directory.is_symlink() or not directory.is_dir() or str(directory.resolve()) in keep
                or any(directory.resolve() == path or directory.resolve() in path.parents for path in opened)):
            continue
        certificate = directory / "snapshot.json"
        if not certificate.is_file() or certificate.is_symlink():
            continue
        data = json.loads(certificate.read_text())
        if data.get("ready") is not True or data.get("generation") != directory.name:
            continue
        shutil.rmtree(directory)
        removed.append(directory.name)
        print(f"[検索用DB] 未使用の旧世代を削除しました：{directory.name}", flush=True)
    return removed


def require_copy_space(root: Path, source: Path, destination: Path) -> None:
    databases = [source / "chroma.sqlite3", lexical_path(root),
                 resolve_configured_path(root, os.environ.get("RELATIONS_DB_PATH", root / "data" / "relations.db"))]
    required = 1024 ** 3 + sum(
        p.stat().st_size for db in databases for p in (db, Path(str(db) + "-wal")) if p.exists())
    if sys.platform != "darwin":
        required += sum(p.stat().st_size for p in source.rglob("*") if p.is_file())
    free = shutil.disk_usage(destination).free
    if free < required:
        raise RuntimeError(f"検索DB公開に必要な空き容量が不足しています（空き {free / 1024**3:.1f} GiB、必要目安 {required / 1024**3:.1f} GiB）。旧検索DBは保持しています。空き容量を確保してから再実行してください。")


def build_copy(root: Path, home: Path, source: Path, manifest: Path, data: dict, audit_passed: bool) -> dict:
    generation = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    home.mkdir(parents=True, exist_ok=True)
    stage = home / f".partial-{generation}"
    destination = home / generation
    stage.mkdir()
    try:
        require_copy_space(root, source, home)
        print("[検索用DB] 更新前の索引と全文・構造DBを保存します。", flush=True)
        clone_tree(source, stage / "chroma")
        # Query entry points may touch SQLite operational state. Back up its live
        # logical contents and discard copied sidecars before any native open.
        for name in ("chroma.sqlite3", "chroma.sqlite3-wal", "chroma.sqlite3-shm"):
            (stage / "chroma" / name).unlink(missing_ok=True)
        compatibility = sqlite_copy(source / "chroma.sqlite3", stage / "chroma" / "chroma.sqlite3", chroma=True)
        sqlite_copy(lexical_path(root), stage / "lexical_v3.sqlite3")
        relations = resolve_configured_path(root, os.environ.get("RELATIONS_DB_PATH", root / "data" / "relations.db"))
        sqlite_copy(relations, stage / "relations.db")
        shutil.copy2(manifest, stage / "manifest_v3.json")
        counts = expected_counts(source)
        print(f"[検索用DB] 保存コピーの検索・再オープンを検証します：{counts}", flush=True)
        verify_native_copy(stage / "chroma", counts)
        record = {"generation": generation, "directory": str(destination), "ready": True,
                  "published_at": datetime.now(timezone.utc).isoformat(),
                  "attachments": len(data.get("files", {})), "collection_counts": counts,
                  "audit_passed": audit_passed, "previous": None, "sqlite_compatibility": compatibility}
        previous = load_snapshot(root)
        if previous:
            record["previous"] = previous["directory"]
        (stage / "snapshot.json").write_text(json.dumps(record, indent=2) + "\n")
        stage.rename(destination)
        return record
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize", action="store_true", help="Pin the current query-ready corpus once")
    parser.add_argument("--restart-services", action="store_true")
    args = parser.parse_args()
    os.environ.update(environment_with_saved_dotenv(ROOT))
    record = publish(ROOT, initialize=args.initialize, restart_services=args.restart_services)
    print(json.dumps({key: record[key] for key in ("generation", "attachments", "audit_passed")}), flush=True)


if __name__ == "__main__":
    main()
