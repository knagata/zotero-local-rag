#!/usr/bin/env python3
"""Export stored vectors without loading native HNSW, including pending WAL.

Only immutable, trusted Chroma snapshots are accepted. This creates a separate
SQLite artifact and never repairs or opens the source with a native client.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
import sqlite3
import struct
from pathlib import Path

import numpy as np

HEADER = struct.Struct("<iQQQQQQiIQQQdQ")


class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError("snapshot metadata must contain primitive values only")


def read_persisted_vectors(segment: Path, dimension: int) -> dict[str, bytes]:
    header = HEADER.unpack((segment / "header.bin").read_bytes())
    version, _, capacity, count, stride, label_offset, data_offset = header[:7]
    if (version != 1 or not 0 < count <= capacity or dimension <= 0
            or label_offset - data_offset != dimension * 4
            or label_offset + 8 != stride or data_offset < 4):
        raise ValueError("unsupported or inconsistent HNSW header")
    binary = segment / "data_level0.bin"
    if binary.stat().st_size < count * stride:
        raise ValueError("truncated HNSW vector data")
    metadata = PrimitiveUnpickler(io.BytesIO(
        (segment / "index_metadata.pickle").read_bytes(),
    )).load()
    id_to_label = metadata["id_to_label"]
    label_to_id = metadata["label_to_id"]
    if (len(id_to_label) != len(label_to_id)
            or any(label_to_id.get(label) != key for key, label in id_to_label.items())):
        raise ValueError("HNSW label mappings disagree")
    raw = np.memmap(binary, mode="r", dtype="u1")
    labels = np.ndarray((count,), dtype="<u8", buffer=raw,
                        offset=label_offset, strides=(stride,))
    order = np.argsort(labels)
    sorted_labels = labels[order]
    if len(np.unique(sorted_labels)) != count:
        raise ValueError("duplicate physical HNSW labels")
    vectors = np.ndarray((count, dimension), dtype="<f4", buffer=raw,
                         offset=data_offset, strides=(stride, 4))
    result = {}
    for key, label in id_to_label.items():
        position = int(np.searchsorted(sorted_labels, label))
        if position >= count or int(sorted_labels[position]) != label:
            raise ValueError("live HNSW label has no physical vector")
        vector = vectors[int(order[position])]
        validate_vector(vector, dimension)
        result[str(key)] = vector.tobytes()
    return result


def validate_vector(vector: np.ndarray, dimension: int) -> None:
    if vector.shape != (dimension,) or not np.isfinite(vector).all():
        raise ValueError("vector dimension or finiteness validation failed")
    if not np.any(vector):
        raise ValueError("zero vector cannot represent cosine embeddings")


def replay_wal(connection, collection_id, segment_id, vectors, dimension):
    row = connection.execute(
        "SELECT seq_id FROM max_seq_id WHERE segment_id=?", (segment_id,),
    ).fetchone()
    if row is None or not isinstance(row[0], int):
        raise ValueError("persisted vector sequence boundary is missing")
    topic = f"persistent://default/default/{collection_id}"
    topics = {item[0] for item in connection.execute(
        "SELECT DISTINCT topic FROM embeddings_queue WHERE topic LIKE ?",
        (f"%/{collection_id}",),
    )}
    if topics - {topic}:
        raise ValueError("unsupported WAL topic namespace")
    pending = connection.execute("""
        SELECT operation, id, vector, encoding FROM embeddings_queue
        WHERE topic=? AND seq_id>? ORDER BY seq_id
    """, (topic, row[0]))
    replayed = 0
    for operation, key, blob, encoding in pending:
        replayed += 1
        if operation == 3:
            vectors.pop(key, None)
        elif operation in (0, 1, 2):
            if operation == 0 and key in vectors:
                continue
            if operation == 1 and key not in vectors:
                continue
            if blob is not None:
                if encoding != "FLOAT32" or len(blob) != dimension * 4:
                    raise ValueError("unsupported WAL vector encoding")
                validate_vector(np.frombuffer(blob, dtype="<f4"), dimension)
                vectors[key] = blob
        else:
            raise ValueError("unknown WAL operation")
    return replayed


def export_snapshot(source: Path, output: Path) -> dict:
    source = source.resolve()
    output = output.resolve()
    if not (source.parent / "RAW_SNAPSHOT_COMPLETE").is_file():
        raise ValueError("source must be a completed immutable raw snapshot")
    if source == output or source in output.parents or output.exists():
        raise ValueError("output must be a new artifact outside the source snapshot")
    original = sqlite3.connect(f"{(source / 'chroma.sqlite3').as_uri()}?mode=ro", uri=True)
    destination = sqlite3.connect(output)
    try:
        original.execute("BEGIN")
        destination.execute("""CREATE TABLE vectors (
            collection TEXT, id TEXT, vector BLOB, PRIMARY KEY(collection, id))""")
        report = {"complete": False, "collections": {}}
        for collection_id, name, dimension in original.execute(
            "SELECT id, name, dimension FROM collections ORDER BY name",
        ):
            segment = original.execute(
                "SELECT id FROM segments WHERE collection=? AND scope='VECTOR'",
                (collection_id,),
            ).fetchone()
            if segment is None:
                raise ValueError("collection has no vector segment")
            vectors = read_persisted_vectors(source / segment[0], dimension)
            replayed = replay_wal(original, collection_id, segment[0], vectors, dimension)
            canonical_rows = [row[0] for row in original.execute("""
                SELECT e.embedding_id FROM embeddings e JOIN segments s ON s.id=e.segment_id
                WHERE s.collection=? AND s.scope='METADATA'
            """, (collection_id,))]
            canonical = set(canonical_rows)
            if len(canonical_rows) != len(canonical):
                raise ValueError("duplicate canonical embedding IDs")
            if set(vectors) != canonical:
                raise ValueError(f"stored-vector IDs do not match canonical IDs for {name}")
            digest = hashlib.sha256()
            for key in sorted(vectors):
                blob = vectors[key]
                digest.update(json.dumps(key, ensure_ascii=False).encode() + b"\0" + blob)
                destination.execute("INSERT INTO vectors VALUES (?, ?, ?)", (name, key, blob))
            destination.commit()
            report["collections"][name] = {
                "count": len(vectors), "dimension": dimension,
                "wal_operations": replayed, "id_vector_sha256": digest.hexdigest(),
            }
            print(name, report["collections"][name], flush=True)
        report["complete"] = True
        output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        destination.close()
        original.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_snapshot(args.source, args.output)


if __name__ == "__main__":
    main()
