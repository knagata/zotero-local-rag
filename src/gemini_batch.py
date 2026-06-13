"""Gemini Batch Embedding pipeline for cost-optimized bulk indexing.

Uses the Gemini Batch API (client.batches.create_embeddings) which is:
  - Asynchronous: submit → poll → download
  - 50% cheaper than the online API ($0.075/1M input tokens)
  - Can take minutes to hours (documented max 24h)
  - Supports resume via persisted job state

For online queries (low volume, instant), use embedder.create_embedding_function().
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
JOB_STATE_PATH = DATA_DIR / "gemini_batch_job.json"
BATCH_JSONL_DIR = DATA_DIR / "gemini_batch_requests"

# --- Job state persistence (for resume) ---


def save_job_state(job_name: str, state: str, *, extra: dict | None = None) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "job_name": job_name,
        "state": state,
        "saved_at": time.time(),
    }
    if extra:
        data.update(extra)
    tmp = JOB_STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(JOB_STATE_PATH)


def load_job_state() -> dict | None:
    if not JOB_STATE_PATH.exists():
        return None
    try:
        return json.loads(JOB_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def clear_job_state() -> None:
    try:
        JOB_STATE_PATH.unlink(missing_ok=True)
    except Exception:
        pass


# --- Batch pipeline ---


def prepare_batch_requests(
    chunks: List[Tuple[str, str]],
    *,
    output_dimensionality: int | None = None,
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> Path:
    """Write chunk texts to a JSONL file for the Gemini batch API.

    Args:
        chunks: List of (chunk_id, text) tuples.
        output_dimensionality: Optional embedding dimension (128-3072).
        task_type: Gemini task type for embedding optimization.

    Returns:
        Path to the written JSONL file.
    """
    BATCH_JSONL_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    jsonl_path = BATCH_JSONL_DIR / f"batch_requests_{ts}.jsonl"

    total = len(chunks)
    print(f"[gemini-batch] Writing {total:,} chunk(s) to {jsonl_path} ...", file=sys.stderr)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i, (chunk_id, text) in enumerate(chunks):
            request_body: dict[str, Any] = {
                "content": {"parts": [{"text": text}]},
                "taskType": task_type,
            }
            if output_dimensionality:
                request_body["output_dimensionality"] = output_dimensionality

            line = {"key": chunk_id, "request": request_body}
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

            if (i + 1) % 5000 == 0:
                print(
                    f"[gemini-batch]   wrote {i + 1:,}/{total:,} requests ...",
                    file=sys.stderr,
                )

    size_mb = jsonl_path.stat().st_size / (1024 * 1024)
    print(f"[gemini-batch] JSONL ready: {total:,} requests, {size_mb:.1f} MB", file=sys.stderr)
    return jsonl_path


def _get_genai_client():
    """Lazy-import google.genai and create a client."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    from google import genai
    return genai.Client(api_key=api_key)


def submit_batch_job(
    jsonl_path: Path,
    model: str = "gemini-embedding-001",
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> dict:
    """Upload JSONL and submit a batch embedding job.

    Returns:
        dict with keys: job_name, model, jsonl_path, task_type
    """
    client = _get_genai_client()

    print(f"[gemini-batch] Uploading batch requests file ...", file=sys.stderr)
    uploaded = client.files.upload(file=str(jsonl_path), mime_type="application/x-jsonlines")
    print(f"[gemini-batch] Uploaded: {uploaded.name}", file=sys.stderr)

    print(
        f"[gemini-batch] Creating batch embedding job (model={model}, task_type={task_type}) ...",
        file=sys.stderr,
    )
    job = client.batches.create_embeddings(
        model=model,
        src={"file_name": uploaded.name},
    )
    job_name = job.name
    state = getattr(job, "state", "PENDING")
    print(f"[gemini-batch] Job created: {job_name}  state={state}", file=sys.stderr)

    result = {
        "job_name": job_name,
        "model": model,
        "jsonl_path": str(jsonl_path),
        "task_type": task_type,
    }
    save_job_state(job_name, state, extra=result)
    return result


def poll_batch_job(
    job_name: str,
    *,
    initial_interval: float = 30.0,
    max_interval: float = 120.0,
) -> dict:
    """Poll a batch job until it reaches a terminal state.

    Blocks with adaptive polling interval and progress display.
    Returns the job dict from the API.
    """
    client = _get_genai_client()

    poll_count = 0
    interval = initial_interval
    t_start = time.time()

    while True:
        job = client.batches.get(name=job_name)
        state = getattr(job, "state", "UNKNOWN")
        poll_count += 1
        elapsed = time.time() - t_start

        # Format elapsed time
        if elapsed < 60:
            elapsed_str = f"{elapsed:.0f}s"
        elif elapsed < 3600:
            elapsed_str = f"{elapsed / 60:.1f}m"
        else:
            elapsed_str = f"{elapsed / 3600:.1f}h"

        # Progress bar
        progress = ""
        try:
            if hasattr(job, "processed_count") and hasattr(job, "total_count"):
                pc = job.processed_count or 0
                tc = job.total_count or 0
                if tc > 0:
                    pct = pc / tc * 100
                    bar_w = 20
                    filled = int(bar_w * pc / tc)
                    bar = "█" * filled + "░" * (bar_w - filled)
                    progress = f" [{bar}] {pc:,}/{tc:,} ({pct:.1f}%)"
        except Exception:
            pass

        print(
            f"[gemini-batch] poll #{poll_count}  state={state}  "
            f"elapsed={elapsed_str}{progress}",
            file=sys.stderr,
        )

        save_job_state(job_name, state)

        if state in ("JOB_STATE_SUCCEEDED", "SUCCEEDED"):
            print(f"[gemini-batch] Job completed successfully.", file=sys.stderr)
            return _job_to_dict(job)

        if state in ("JOB_STATE_FAILED", "FAILED", "JOB_STATE_CANCELLED", "CANCELLED"):
            error_msg = ""
            try:
                error_msg = getattr(job, "error", "") or ""
            except Exception:
                pass
            raise RuntimeError(
                f"Gemini batch job {job_name} terminated with state={state}. {error_msg}"
            )

        if state in ("JOB_STATE_EXPIRED", "EXPIRED"):
            raise RuntimeError(
                f"Gemini batch job {job_name} expired. Re-run the indexer to submit a new job."
            )

        # Adaptive interval: faster polling early, slower later
        if poll_count >= 10:
            interval = min(interval * 1.2, max_interval)

        time.sleep(interval)


def _job_to_dict(job) -> dict:
    """Convert a genai job object to a plain dict."""
    d: dict[str, Any] = {}
    for attr in ("name", "state", "model", "create_time", "update_time", "end_time"):
        try:
            d[attr] = str(getattr(job, attr, ""))
        except Exception:
            d[attr] = ""
    return d


def download_batch_results(job: dict) -> dict[str, np.ndarray]:
    """Download and parse the batch job result file.

    Returns:
        {chunk_id: embedding_vector} mapping.
        Embedding vectors are float32 numpy arrays, L2-normalized.
    """
    client = _get_genai_client()
    job_name = job["job_name"]

    print(f"[gemini-batch] Downloading results for {job_name} ...", file=sys.stderr)

    # The result file name follows the pattern: <job_name>-out.jsonl
    result_file_name = f"{job_name}-out.jsonl"
    try:
        content = client.files.download(file=result_file_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download batch results for {job_name}: {e}\n"
            "The job may have completed but the result file is not yet available. "
            "Try re-running the indexer to resume."
        )

    if not content:
        # Retry after a short delay — occasional 0-byte race condition with GCS
        print("[gemini-batch] Result file empty (0 bytes), retrying after 10s ...", file=sys.stderr)
        time.sleep(10)
        try:
            content = client.files.download(file=result_file_name)
        except Exception:
            pass
        if not content:
            raise RuntimeError("Batch result file is empty. The batch job may have produced no output.")

    print(f"[gemini-batch] Downloaded {len(content):,} bytes, parsing ...", file=sys.stderr)

    embeddings: dict[str, np.ndarray] = {}
    line_count = 0
    error_count = 0

    for line in content.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            error_count += 1
            continue

        key = record.get("key", "")
        response = record.get("response") or {}
        emb_data = response.get("embeddings") or {}
        values = emb_data.get("values")

        if key and values:
            vec = np.array(values, dtype=np.float32)
            # Normalize to unit length for cosine similarity compatibility
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings[key] = vec
            line_count += 1
        else:
            error_count += 1

    print(
        f"[gemini-batch] Parsed {line_count:,} embeddings, {error_count} errors/empty",
        file=sys.stderr,
    )
    return embeddings


def ingest_embeddings(
    col,
    embeddings: dict[str, np.ndarray],
    chunks: List[Tuple[str, str, Dict[str, Any]]],
    batch_size: int = 128,
    show_progress: bool = False,
) -> int:
    """Insert pre-computed embeddings into ChromaDB in sub-batches.

    Only inserts chunks whose IDs are present in the embeddings dict.

    Uses upsert-first-then-cleanup ordering: new chunks are written before
    old ones are deleted, so a crash mid-ingest does not cause data loss.

    Returns:
        Number of chunks inserted.
    """
    # Collect unique attachment keys and new chunk data
    attachment_keys: set[str] = set()
    new_id_set: set[str] = set()
    ids: list[str] = []
    emb_list: list[np.ndarray] = []
    docs: list[str] = []
    metas: list[Dict[str, Any]] = []

    for chunk_id, text, md in chunks:
        if chunk_id not in embeddings:
            continue
        ak = md.get("attachmentKey")
        if ak:
            attachment_keys.add(ak)
        new_id_set.add(chunk_id)
        ids.append(chunk_id)
        emb_list.append(embeddings[chunk_id])
        docs.append(text)
        metas.append(md)

    if not ids:
        return 0

    # Collect old chunk IDs for the affected attachment keys
    old_ids: set[str] = set()
    for ak in attachment_keys:
        try:
            result = col.get(where={"attachmentKey": ak}, include=[])
            if result and result.get("ids"):
                old_ids.update(result["ids"])
        except Exception:
            pass

    # Upsert new chunks (atomic per sub-batch, safe for resume)
    total = len(ids)
    if show_progress:
        print(
            f"[gemini-batch] Ingesting {total:,} chunks into ChromaDB "
            f"(batch_size={batch_size}) ...",
            file=sys.stderr,
        )

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        if show_progress:
            print(
                f"[gemini-batch]   ↳ sub-batch {start + 1:,}-{end:,}/{total:,}",
                file=sys.stderr,
            )
        col.upsert(
            ids=ids[start:end],
            embeddings=[e.tolist() for e in emb_list[start:end]],
            documents=docs[start:end],
            metadatas=metas[start:end],
        )

    # Delete leftover old IDs that were not re-upserted
    leftover = old_ids - new_id_set
    if leftover:
        try:
            col.delete(ids=list(leftover))
        except Exception:
            pass

    return total
