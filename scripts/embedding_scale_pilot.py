"""Run a reproducible embedding scale pilot in an isolated Chroma plane.

The default command is intentionally explicit: use ``--fake`` for a fast,
deterministic check, or ``--real-bge`` when a local BGE-M3 snapshot is already
available.  Neither mode resolves or downloads a model.  The real pilot never
uses the active V3 paths; its collection and (for the compensation check) FTS
file live under the supplied temporary data-plane directory.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from embedder import EmbedderConfig, create_embedding_function, open_chroma_collection
from env_utils import load_dotenv_native
from index_batch import AttachmentBatch, replace_attachment_batch
from lexical_index import delete_by_attachment_keys, list_chunk_ids, upsert_chunks
from v3_data_plane import chroma_dir, lexical_path, manifest_path

load_dotenv_native(ROOT)

COLLECTION = "embedding_scale_pilot"
PILOT_ATTACHMENT = "PILOT-ATTACHMENT"


class PilotInterrupted(RuntimeError):
    """A deterministic stop used to verify idempotent resume."""


def synthetic_chunks(count: int) -> list[tuple[str, str, dict[str, Any]]]:
    """Return short, long, and Japanese/English mixed deterministic chunks."""
    short = "短い検索用チャンク。"
    mixed = "都市の記憶 and public space は、歩幅と視線の高さで変わる。"
    long = ("Long-form context preserves enough surrounding prose for a realistic "
            "embedding request. 日本語と English を同じ chunk に含めます。 " * 32).strip()
    patterns = (short, mixed, long)
    rows = []
    for index in range(max(1, count)):
        attachment = f"PILOT-{index // 3:04d}"
        rows.append((
            f"pilot-{index:05d}", patterns[index % len(patterns)], {
                "attachmentKey": attachment, "itemKey": f"ITEM-{index // 3:04d}",
                "source_type": "pilot", "lang": "ja+en" if index % 3 else "ja",
            },
        ))
    return rows


def _rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB, macOS reports bytes.
    return round(value / (1024 * 1024 if sys.platform == "darwin" else 1024), 2)


def _close(collection: Any) -> None:
    client = getattr(collection, "_chroma_client", None)
    close = getattr(client, "close", None)
    if callable(close):
        close()


def _active_paths() -> tuple[Path, Path, Path]:
    return (
        chroma_dir(ROOT).resolve(),
        lexical_path(ROOT).resolve(),
        manifest_path(ROOT).resolve(),
    )


def _assert_isolated_data_plane(root: Path) -> None:
    candidate = root.expanduser().resolve()
    for active in _active_paths():
        if candidate == active or candidate in active.parents or active in candidate.parents:
            raise ValueError(
                f"pilot data plane overlaps the active V3 plane: {candidate} / {active}"
            )


def _assert_safe_output(path: Path | None) -> None:
    if path is None:
        return
    candidate = path.expanduser().resolve()
    active_chroma, active_lexical, active_manifest = _active_paths()
    if (
        candidate == active_lexical
        or candidate == active_manifest
        or candidate == active_chroma
        or active_chroma in candidate.parents
    ):
        raise ValueError(f"pilot report path overlaps the active V3 plane: {candidate}")


def _open(root: Path, embedder: Callable[..., Any], sync_threshold: int):
    key = "CHROMA_HNSW_SYNC_THRESHOLD"
    previous = os.environ.get(key)
    os.environ[key] = str(sync_threshold)
    try:
        return open_chroma_collection(root / "chroma", COLLECTION, embedder)
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _tree_size(path: Path) -> int:
    return sum(candidate.stat().st_size for candidate in path.rglob("*") if candidate.is_file())


def _write_batches(
    collection: Any,
    rows: list[tuple[str, str, dict[str, Any]]],
    batch_size: int,
    interrupt_after: int | None = None,
) -> tuple[int, float]:
    existing = set(collection.get(include=[]).get("ids") or [])
    pending = [row for row in rows if row[0] not in existing]
    started = time.perf_counter()
    batches = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        collection.upsert(
            ids=[row[0] for row in batch], documents=[row[1] for row in batch],
            metadatas=[row[2] for row in batch],
        )
        batches += 1
        if interrupt_after is not None and batches >= interrupt_after:
            raise PilotInterrupted(f"stopped after {batches} batch(es)")
    elapsed = max(time.perf_counter() - started, 1e-9)
    return batches, elapsed


@contextmanager
def _isolated_paths(root: Path) -> Iterator[None]:
    """Make index_batch's lexical sidecar point inside the pilot plane."""
    values = {
        "INGEST_STRUCTURED_V3_ENABLE": "1",
        "LEXICAL_DB_PATH": str(root / "lexical_v3.sqlite3"),
    }
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _compensation_check(collection: Any, root: Path) -> dict[str, Any]:
    old = [("replace-old-0", "old generation", {"attachmentKey": PILOT_ATTACHMENT, "itemKey": "OLD"})]
    new = [(f"replace-new-{index}", f"new generation {index}", {
        "attachmentKey": PILOT_ATTACHMENT, "itemKey": "NEW",
    }) for index in range(3)]
    collection.upsert(ids=[old[0][0]], documents=[old[0][1]], metadatas=[old[0][2]])
    upsert_chunks([old[0][0]], [old[0][1]], [old[0][2]], path=root / "lexical_v3.sqlite3")
    calls = 0

    def delete_batch(col, keys, *, strict=False):
        del strict
        col.delete(where={"attachmentKey": {"$in": list(keys)}})
        delete_by_attachment_keys(keys, path=root / "lexical_v3.sqlite3")

    def upsert_batch(col, ids, docs, metas, **kwargs):
        nonlocal calls
        size = int(kwargs.get("subbatch_size") or len(ids))
        for start in range(0, len(ids), size):
            end = min(start + size, len(ids))
            col.upsert(ids=ids[start:end], documents=docs[start:end], metadatas=metas[start:end])
            upsert_chunks(ids[start:end], docs[start:end], metas[start:end], path=root / "lexical_v3.sqlite3")
            calls += 1
            if calls == 1:
                raise RuntimeError("pilot injected upsert failure")

    batch = AttachmentBatch.create(
        attachment_keys=[PILOT_ATTACHMENT], ids=[row[0] for row in new],
        documents=[row[1] for row in new], metadatas=[row[2] for row in new],
        expected_ids={PILOT_ATTACHMENT: {row[0] for row in new}},
        attachment_item_keys={PILOT_ATTACHMENT: "NEW"}, subbatch_size=2,
        show_progress=False, label="pilot compensation", context_label="pilot",
        strict_lexical=True,
    )
    with _isolated_paths(root):
        try:
            replace_attachment_batch(
                collection, batch, delete_batch=delete_batch, upsert_batch=upsert_batch,
                health_check=lambda col, _items, context_label: col.count(),
                verify_written=None,
            )
        except RuntimeError as exc:
            if "injected upsert failure" not in str(exc):
                raise
    actual = set(collection.get(where={"attachmentKey": PILOT_ATTACHMENT}, include=[])["ids"])
    lexical = set(list_chunk_ids(path=root / "lexical_v3.sqlite3"))
    return {"injected_failure": True, "restored_ids_match": actual == {old[0][0]},
            "vector_ids": sorted(actual), "lexical_ids": sorted(lexical),
            "stores_match": lexical == actual}


def run_pilot(
    root: Path, embedder: Callable[..., Any], *, batch_size: int = 8,
    sync_threshold: int = 100, chunk_count: int = 30, exercise_recovery: bool = True,
) -> dict[str, Any]:
    """Run the pilot and return a JSON-serializable measurement report."""
    if batch_size < 1 or sync_threshold < 1 or chunk_count < 1:
        raise ValueError("batch_size, sync_threshold, and chunk_count must be positive")
    if exercise_recovery and chunk_count <= batch_size:
        raise ValueError("recovery exercise requires chunk_count greater than batch_size")
    _assert_isolated_data_plane(root)
    root.mkdir(parents=True, exist_ok=True)
    rows = synthetic_chunks(chunk_count)
    started = time.perf_counter()
    collection = _open(root, embedder, sync_threshold)
    baseline_data_plane_bytes = _tree_size(root)
    interrupted: dict[str, Any] = {"enabled": False}
    try:
        write_started = time.perf_counter()
        if exercise_recovery:
            try:
                batches, elapsed = _write_batches(collection, rows, batch_size, 1)
            except PilotInterrupted:
                interrupted = {"enabled": True, "after_batches": 1, "count_before_resume": collection.count()}
                _close(collection)
                collection = _open(root, embedder, sync_threshold)
                resumed_batches, elapsed = _write_batches(collection, rows, batch_size)
                batches = 1 + resumed_batches
                interrupted["count_after_resume"] = collection.count()
                interrupted["ids_complete"] = set(collection.get(include=[])["ids"]) == {row[0] for row in rows}
        else:
            batches, elapsed = _write_batches(collection, rows, batch_size)
        elapsed = max(time.perf_counter() - write_started, 1e-9)
        count = collection.count()
        ids_before = set(collection.get(include=[])["ids"])
        _close(collection)
        collection = _open(root, embedder, sync_threshold)
        ids_after = set(collection.get(include=[])["ids"])
        query_vector = embedder([rows[0][1]])[0]
        query = collection.query(query_embeddings=[query_vector], n_results=min(3, count), include=["distances"])
        data_plane_bytes = _tree_size(root)
        incremental_bytes = max(0, data_plane_bytes - baseline_data_plane_bytes)
        compensation = _compensation_check(collection, root) if exercise_recovery else {"enabled": False}
        return {
            "collection": COLLECTION, "embedding_dimension": len(query_vector),
            "chunk_count": len(rows), "batch_size": batch_size,
            "hnsw_sync_threshold": sync_threshold, "batches": batches,
            "throughput_chunks_per_second": round(len(rows) / max(elapsed, 1e-9), 2),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "peak_rss_mb": _rss_mb(), "count_before_close": count,
            "count_after_reopen": len(ids_after), "ids_match_after_reopen": ids_before == ids_after,
            "data_plane_bytes": data_plane_bytes,
            "baseline_data_plane_bytes": baseline_data_plane_bytes,
            "incremental_data_plane_bytes": incremental_bytes,
            "incremental_bytes_per_chunk": round(incremental_bytes / count, 2),
            "hnsw_query": {"ids": query.get("ids", []), "distances": query.get("distances", [])},
            "interruption_resume": interrupted, "compensation": compensation,
            "isolated_data_plane": str(root),
        }
    finally:
        _close(collection)


def _embedder(args: argparse.Namespace):
    if args.fake:
        from tests.synthetic_library import deterministic_embedding_function
        return deterministic_embedding_function(dimensions=32), "fake"
    model = Path(args.model or ROOT / "data" / "models" / "bge-m3").expanduser()
    if not model.is_dir():
        raise SystemExit(f"BGE-M3 local model directory not found: {model}")
    config = EmbedderConfig("sentence_transformers", str(model), args.device)
    return create_embedding_function(config), "bge-m3"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fake", action="store_true", help="use deterministic 32-dim fake embeddings")
    mode.add_argument("--real-bge", action="store_true", help="use an existing local BGE-M3 snapshot")
    parser.add_argument("--model", help="local BGE-M3 directory (only with --real-bge)")
    parser.add_argument("--device", default="mps", choices=("mps", "cpu", "cuda"))
    parser.add_argument("--data-plane", type=Path, help="isolated output directory; defaults to a temporary directory")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sync-threshold", type=int, default=100)
    parser.add_argument("--chunks", type=int, default=30)
    parser.add_argument("--no-recovery", action="store_true", help="skip interruption/resume and compensation checks")
    parser.add_argument("--output", type=Path, help="write the JSON report to this path")
    args = parser.parse_args(argv)
    _assert_safe_output(args.output)
    embedder, profile = _embedder(args)
    if args.real_bge:
        print(f"Running explicit real pilot with {profile} on {args.device}", file=sys.stderr)
    import tempfile
    temporary = None
    root = args.data_plane
    if root is None:
        temporary = tempfile.TemporaryDirectory(prefix="zotero-rag-embedding-pilot-")
        root = Path(temporary.name)
    try:
        report = run_pilot(root, embedder, batch_size=args.batch_size, sync_threshold=args.sync_threshold,
                           chunk_count=args.chunks, exercise_recovery=not args.no_recovery)
        report["profile"] = profile
        rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        print(rendered)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
    finally:
        if temporary is not None:
            temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
