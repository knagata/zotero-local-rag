"""Content-addressed archive of raw OCR engine responses.

Why this exists: a cloud OCR response costs money and (for the Batch API)
hours of turnaround, but everything downstream of it -- block normalisation,
zone assignment, chunking, structure recovery -- is our own Python and changes
often. Re-running ``--force-reparse`` after a chunking fix used to mean buying
the same OCR again. This module keeps the *raw* response so the expensive part
is paid once and the cheap part can be redone with current code any number of
times.

Two rules keep that from becoming the failure it is meant to prevent:

1. **Only raw engine output is stored.** Nothing that our own code derived ever
   enters this cache. A cached entry is therefore always upstream of chunking,
   so re-deriving from it cannot serve a stale *derived* artifact -- there is
   no derived artifact here to go stale. Quality gates and problem-page maps
   are recomputed from the raw body on every read, never cached.

2. **The request contract is part of the key.** Entries live under
   ``<engine>/<contract>/<sha256>.json`` where ``contract`` encodes the model
   and the request parameters that shape the response. Adding a parameter such
   as ``include_blocks`` changes the contract, so pre-change entries simply
   miss instead of being read as if they carried the new fields. That is what
   makes "skip a richer response for now" a reversible decision rather than a
   trap.

Identity is the sha256 of the source file's bytes, not its path, mtime, or
size. mtime is what ``mistral_ocr_batch.source_matches`` uses for a different
purpose (detecting edits between submit and adopt within one machine), and it
is deliberately not used here: Zotero sync gives the same document a new mtime
on every machine, so an mtime-keyed archive would miss everything the moment
the cache directory is copied. Content addressing makes ``data/ocr_cache/``
portable -- copy it alongside a synced Zotero library and it keeps hitting --
and makes a duplicate PDF filed under two items cost one OCR instead of two.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Bumped when the *request* we send changes shape (a new parameter, a
#: different document encoding). Not bumped for chunking/parsing changes --
#: those consume the cache, they do not invalidate it.
MISTRAL_REQUEST_CONTRACT = "req1"

SCHEMA_VERSION = "ocr-cache-1"

_SLUG_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")

_READ_CHUNK = 1024 * 1024


def cache_root(data_dir: Path) -> Path:
    """Root of the archive. ``OCR_CACHE_DIR`` overrides for tests/relocation."""
    override = os.environ.get("OCR_CACHE_DIR", "").strip()
    return Path(override).expanduser() if override else Path(data_dir) / "ocr_cache"


def source_digest(path: Path) -> str:
    """sha256 of the file's bytes, streamed so a large PDF is not held in RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(_READ_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def contract_slug(model: str, contract_version: str) -> str:
    """Directory name identifying one request contract.

    Both halves matter: the same parameters against a different model can
    return a different response shape (Mistral's ``blocks`` is populated only
    by OCR 4 and newer, and older models return an empty array rather than an
    error), so the model cannot be metadata-only.
    """
    raw = f"{str(model).strip() or 'unknown'}__{str(contract_version).strip() or 'v0'}"
    return _SLUG_UNSAFE_RE.sub("-", raw).strip("-") or "unknown"


def entry_path(
    data_dir: Path, *, engine: str, model: str, contract_version: str, digest: str,
) -> Path:
    return (
        cache_root(data_dir) / _SLUG_UNSAFE_RE.sub("-", engine)
        / contract_slug(model, contract_version) / f"{digest}.json"
    )


def load_result(
    data_dir: Path, *, engine: str, model: str, contract_version: str, digest: str,
) -> dict[str, Any] | None:
    """Return the archived raw response, or None when absent/unreadable.

    A damaged entry reads as a miss rather than an error: the caller's fallback
    is to fetch again, which is correct and merely costs what it would have
    cost anyway. Raising here would instead fail an ingest over a cache
    problem.
    """
    path = entry_path(
        data_dir, engine=engine, model=model,
        contract_version=contract_version, digest=digest,
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    # Guard against a digest-named file whose recorded identity disagrees --
    # a hand-edited or half-migrated entry must not be served as a match.
    if str(payload.get("source_sha256") or "") != digest:
        return None
    result = payload.get("result")
    return result if isinstance(result, dict) else None


def load_result_any_model(
    data_dir: Path, *, engine: str, contract_version: str, digest: str,
) -> tuple[str, dict[str, Any]] | None:
    """Find an archived response for this digest under any model directory.

    ``extract_chunks_from_pdf_with_mistral_ocr`` resolves its lookup model
    from ``MISTRAL_OCR_MODEL`` (or the default), but ``run_mistral_ocr_batch.py
    --model`` can archive a batch under a different, explicitly chosen model.
    Missing a real entry because the two call sites resolved the model
    differently would defeat the whole point of the archive, so a lookup
    falls back to whatever model the digest was actually stored under before
    concluding there is no cached response at all. Contract version is still
    exact-matched: it is what controls response *shape* (e.g. whether
    ``blocks`` was requested), not the model name.
    """
    engine_root = cache_root(data_dir) / _SLUG_UNSAFE_RE.sub("-", engine)
    suffix = f"__{str(contract_version).strip() or 'v0'}"
    try:
        contract_dirs = sorted(p for p in engine_root.iterdir() if p.is_dir())
    except OSError:
        return None
    for contract_dir in contract_dirs:
        if not contract_dir.name.endswith(suffix):
            continue
        candidate = contract_dir / f"{digest}.json"
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict) or str(payload.get("source_sha256") or "") != digest:
            continue
        result = payload.get("result")
        if isinstance(result, dict):
            return str(payload.get("model") or ""), result
    return None


def load_entry(
    data_dir: Path, *, engine: str, model: str, contract_version: str, digest: str,
) -> dict[str, Any] | None:
    """Full entry including provenance metadata (for logging and audits)."""
    path = entry_path(
        data_dir, engine=engine, model=model,
        contract_version=contract_version, digest=digest,
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def store_result(
    data_dir: Path, *, engine: str, model: str, contract_version: str, digest: str,
    result: dict[str, Any], attachment_key: str = "", title: str = "",
    source_size: int | None = None, source_path: str = "",
    batch_job_id: str = "", fetched_at: str | None = None,
) -> Path:
    """Archive one raw response atomically.

    Callers must only pass a response that already passed whatever quality
    gate applies to it: a rejected or truncated body archived here would be
    replayed forever, which is worse than paying for a refetch.

    The entry carries enough provenance (attachment key, title, model, fetch
    time) to be readable on its own, so a copied ``ocr_cache/`` directory can
    be audited without the manifest or the batch job's work directory -- the
    latter being exactly what today's ``results/`` files depend on and why they
    are unreachable once a work directory is cleaned up.
    """
    path = entry_path(
        data_dir, engine=engine, model=model,
        contract_version=contract_version, digest=digest,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "engine": engine,
        "model": model,
        "request_contract_version": contract_version,
        "source_sha256": digest,
        "source_size": source_size,
        "source_path": source_path,
        "attachment_key": attachment_key,
        "title": title,
        "batch_job_id": batch_job_id,
        "fetched_at": fetched_at or datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


__all__ = [
    "MISTRAL_REQUEST_CONTRACT", "SCHEMA_VERSION", "cache_root", "contract_slug",
    "entry_path", "load_entry", "load_result", "source_digest", "store_result",
]
