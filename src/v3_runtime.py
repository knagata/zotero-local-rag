"""Run-level compatibility and provenance guards for structured V3 ingestion."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


PIPELINE_CONTRACT_VERSION = "structured-v3-1"
CHUNK_SCHEME_VERSION = 3


def code_fingerprint(paths: Iterable[Path]) -> str:
    """Hash the exact Python sources that define one ingestion run."""
    digest = hashlib.sha256()
    for path in sorted({Path(value).resolve() for value in paths}, key=str):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\n")
    return f"sha256:{digest.hexdigest()}"


def pipeline_payload(
    embedder: Mapping[str, Any], *, collection: str, run_code_fingerprint: str,
) -> dict[str, Any]:
    compatibility = {
        "pipeline_contract_version": PIPELINE_CONTRACT_VERSION,
        "chunk_scheme": CHUNK_SCHEME_VERSION,
        "embedding_fingerprint": embedder.get("embedding_fingerprint"),
        "embedding_dim": embedder.get("embedding_dim"),
        "normalize_embeddings": embedder.get("normalize_embeddings"),
        "model_state_fingerprint": embedder.get("model_state_fingerprint"),
    }
    fingerprint = hashlib.sha256(
        json.dumps(compatibility, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "v3-pipeline-config-1",
        "collection": collection,
        **compatibility,
        "pipeline_fingerprint": f"sha256:{fingerprint}",
        "run_code_fingerprint": run_code_fingerprint,
        "embedder": dict(embedder),
    }


def ensure_pipeline_config(
    path: Path, runtime: Mapping[str, Any], *, existing_chunk_count: int,
) -> tuple[dict[str, Any], bool]:
    """Create or validate the V3 config; never permit incompatible appends."""
    created = not path.exists()
    if not created:
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("pipeline_fingerprint") != runtime.get("pipeline_fingerprint"):
            raise RuntimeError(
                "V3 pipeline/embedder configuration differs from the existing collection; "
                "refusing to mix embeddings in one collection."
            )
        return dict(stored), False

    payload = dict(runtime)
    payload["adopted_existing_chunks"] = int(existing_chunk_count)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return payload, True


def bind_manifest_pipeline(
    manifest: dict[str, Any], pipeline_fingerprint: str, *, adopt_existing: bool,
) -> bool:
    """Bind a manifest and every file entry to one compatible pipeline.

    ``adopt_existing`` fires whenever ``ensure_pipeline_config`` finds no
    config file to compare against -- at V3 launch, but also every time a
    fresh collection gets its own config, including the next rebuild. It
    stamps every file entry with the current fingerprint on trust, not on
    measurement: nothing here re-derives the entry's actual fingerprint from
    its recorded chunks to check the claim. Once stamped, the same fingerprint
    is what the reprocessing-skip check compares against, so an adopted entry
    reads as "verified current" forever after -- the act of stamping removed
    the only signal that could have shown the stamp was a guess (2026-07-28).

    Adopted entries are marked as such (``pipeline_fingerprint_adopted``) so a
    future audit can tell trust from measurement. The skip check is left
    reading only ``pipeline_fingerprint``, unchanged, so this adds an audit
    trail without silently forcing a reprocess of every adopted file.
    """
    previous = str(manifest.get("pipeline_fingerprint") or "")
    if previous and previous != pipeline_fingerprint:
        raise RuntimeError(
            "V3 manifest pipeline fingerprint differs from the runtime; refusing to continue."
        )
    changed = previous != pipeline_fingerprint
    manifest["pipeline_fingerprint"] = pipeline_fingerprint
    files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    for entry in files.values():
        if not isinstance(entry, dict):
            continue
        existing = str(entry.get("pipeline_fingerprint") or "")
        if existing and existing != pipeline_fingerprint:
            raise RuntimeError("V3 manifest contains files produced by a different pipeline.")
        if not existing:
            if not adopt_existing:
                raise RuntimeError("V3 manifest file is missing its pipeline fingerprint.")
            entry["pipeline_fingerprint"] = pipeline_fingerprint
            entry["pipeline_fingerprint_adopted"] = True
            changed = True
    return changed


def assert_code_unchanged(paths: Iterable[Path], expected: str) -> None:
    actual = code_fingerprint(paths)
    if actual != expected:
        raise RuntimeError(
            "Ingestion source files changed while this run was active; stopping before the next write."
        )


__all__ = [
    "CHUNK_SCHEME_VERSION", "PIPELINE_CONTRACT_VERSION", "assert_code_unchanged",
    "bind_manifest_pipeline", "code_fingerprint", "ensure_pipeline_config", "pipeline_payload",
]
