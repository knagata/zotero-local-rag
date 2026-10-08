#!/usr/bin/env python3
"""Verify an isolated stored-vector recovery before any activation.

The original snapshot is read through SQLite only. Native reads and query
probes are limited to the candidate, with an explicit close/reopen check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from contextlib import ExitStack, closing
from pathlib import Path

import numpy as np

ROUNDING_LIMIT = 5 * np.finfo(np.float32).eps
QUERY_ROUNDING_LIMIT = 10 * np.finfo(np.float32).eps


def logical_signature(directory: Path, name: str) -> dict:
    with closing(sqlite3.connect(
        f"{(directory / 'chroma.sqlite3').resolve().as_uri()}?mode=ro", uri=True,
    )) as connection:
        row = connection.execute("""
            SELECT s.id, c.dimension FROM segments s JOIN collections c ON c.id=s.collection
            WHERE c.name=? AND s.scope='METADATA'
        """, (name,)).fetchone()
        if row is None:
            raise ValueError("collection metadata segment is missing")
        segment, dimension = row
        digest = hashlib.sha256()
        ids = metadata = 0
        for (key,) in connection.execute(
            "SELECT embedding_id FROM embeddings WHERE segment_id=? ORDER BY embedding_id", (segment,),
        ):
            digest.update(json.dumps(key, ensure_ascii=False).encode() + b"\n")
            ids += 1
        digest.update(b"METADATA\0")
        for values in connection.execute("""
            SELECT e.embedding_id, m.key, m.string_value, m.int_value, m.float_value, m.bool_value
            FROM embeddings e JOIN embedding_metadata m ON m.id=e.id
            WHERE e.segment_id=? ORDER BY e.embedding_id, m.key
        """, (segment,)):
            digest.update(json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode() + b"\n")
            metadata += 1
        return {"ids": ids, "metadata_rows": metadata, "dimension": dimension, "sha256": digest.hexdigest()}


def verify_collection(collection, vectors, name, proof):
    dimension = proof["dimension"]
    if collection.count() != proof["count"]:
        raise ValueError("native collection count differs from stored vectors")
    cursor = vectors.execute("SELECT id, vector FROM vectors WHERE collection=? ORDER BY id", (name,))
    verified, max_difference = 0, 0.0
    digest, samples = hashlib.sha256(), []
    while batch := cursor.fetchmany(512):
        keys = [row[0] for row in batch]
        result = collection.get(ids=keys, include=["embeddings"])
        returned = result["ids"]
        if len(returned) != len(keys) or set(returned) != set(keys):
            raise ValueError("native reads do not return every stored-vector ID")
        positions = {key: position for position, key in enumerate(returned)}
        actual = np.asarray(result["embeddings"], dtype="<f4")
        if actual.shape != (len(keys), dimension) or not np.isfinite(actual).all():
            raise ValueError("native vector dimensions/finiteness differ")
        actual = actual[[positions[key] for key in keys]]
        expected = np.asarray([np.frombuffer(blob, dtype="<f4") for _, blob in batch])
        difference = float(np.max(np.abs(actual - expected)))
        if difference > ROUNDING_LIMIT:
            raise ValueError("native vectors changed beyond float32 cosine-normalization rounding")
        max_difference = max(max_difference, difference)
        for key, blob in batch:
            digest.update(json.dumps(key, ensure_ascii=False).encode() + b"\0" + blob)
        if verified % 8192 == 0:
            samples.append(expected[0].tolist())
        verified += len(keys)
        if verified % 8192 == 0 or verified == proof["count"]:
            print(f"VERIFY {name} {verified}/{proof['count']}", flush=True)
    if verified != proof["count"] or digest.hexdigest() != proof["id_vector_sha256"]:
        raise ValueError("stored-vector artifact hash/count changed")
    for vector in samples:
        check_query(collection, vector)
    return {"count": verified, "max_abs_float32_difference": max_difference,
            "query_probes": len(samples)}, samples[:1]


def check_query(collection, vector):
    result = collection.query(query_embeddings=[vector], n_results=1, include=["distances"])
    distances = result.get("distances")
    if not result["ids"] or not result["ids"][0] or not distances or not distances[0]:
        raise ValueError("native vector query returned no result")
    distance = distances[0][0]
    # Chroma's float32 cosine distance can be slightly negative for an exact
    # self-query. A 1024-dimensional vector measured -6 eps from accumulation
    # rounding, so use a small symmetric numeric tolerance here.
    if not np.isfinite(distance) or not -QUERY_ROUNDING_LIMIT <= distance <= QUERY_ROUNDING_LIMIT:
        raise ValueError("stored-vector self query failed")


def verify_copy(source: Path, vectors: Path, candidate: Path) -> dict:
    source, vectors, candidate = source.resolve(), vectors.resolve(), candidate.resolve()
    if source == candidate or source in candidate.parents or candidate in source.parents:
        raise ValueError("native verification must use a separate candidate")
    if not (source.parent / "RAW_SNAPSHOT_COMPLETE").is_file():
        raise ValueError("source must be a completed immutable raw snapshot")
    proof = json.loads(vectors.with_suffix(".json").read_text())
    build = json.loads((candidate / "stored-vector-build.json").read_text())
    if proof.get("complete") is not True or build.get("complete") is not True:
        raise ValueError("vector export/build is not complete")
    if set(proof["collections"]) != set(build["collections"]):
        raise ValueError("collection set changed during recovery")
    report = {"verified": False, "collections": {}}
    destination = candidate.parent / f"{candidate.name}-verification.json"
    destination.write_text(json.dumps(report) + "\n")
    for directory in (source, candidate):
        with closing(sqlite3.connect(
            f"{(directory / 'chroma.sqlite3').as_uri()}?mode=ro", uri=True,
        )) as connection:
            names = {row[0] for row in connection.execute("SELECT name FROM collections")}
        if names != set(proof["collections"]):
            raise ValueError("actual collection set differs from the recovery proof")
    for name, expected in proof["collections"].items():
        if build["collections"][name]["input_vector_sha256"] != expected["id_vector_sha256"]:
            raise ValueError("build input-vector hash differs from the export")
        original, rebuilt = logical_signature(source, name), logical_signature(candidate, name)
        if original != rebuilt or original["ids"] != expected["count"]:
            raise ValueError("ID/document/metadata/dimension content differs from the snapshot")
        report["collections"][name] = {"logical_signature": original}
        print("LOGICAL_CONTENT_MATCH", name, original, flush=True)
    for path in source.glob("*.json"):
        if path.read_bytes() != (candidate / path.name).read_bytes():
            raise ValueError("pipeline configuration file changed")
    import chromadb
    reopen_samples = {}
    with ExitStack() as resources:
        vector_db = resources.enter_context(closing(sqlite3.connect(f"{vectors.as_uri()}?mode=ro", uri=True)))
        client = chromadb.PersistentClient(path=str(candidate))
        resources.callback(client.close)
        for name, expected in proof["collections"].items():
            collection = client.get_collection(name, embedding_function=None)
            result, samples = verify_collection(collection, vector_db, name, expected)
            report["collections"][name].update(result)
            reopen_samples[name] = samples
    client = chromadb.PersistentClient(path=str(candidate))
    try:
        for name, samples in reopen_samples.items():
            collection = client.get_collection(name, embedding_function=None)
            if collection.count() != proof["collections"][name]["count"]:
                raise ValueError("reopened native count differs")
            for vector in samples:
                check_query(collection, vector)
    finally:
        client.close()
    report.update(verified=True, reopened=True, float32_rounding_limit=float(ROUNDING_LIMIT))
    destination.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    verify_copy(args.source, args.vectors, args.candidate)


if __name__ == "__main__":
    main()
