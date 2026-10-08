#!/usr/bin/env python3
"""Build an isolated Chroma copy using verified stored vectors, without embedding.

No activation, manifest change, extraction or hosted API call is performed.
The output must be new and separate from the immutable source snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from contextlib import ExitStack, closing
from pathlib import Path

import numpy as np


def _value(row):
    for value in row[:3]:
        if value is not None:
            return value
    if row[3] is not None:
        return bool(row[3])
    raise ValueError("unsupported null metadata")


def stored_records(connection, collection_id, keys):
    segment = connection.execute(
        "SELECT id FROM segments WHERE collection=? AND scope='METADATA'", (collection_id,),
    ).fetchone()
    if segment is None:
        raise ValueError("source metadata segment is missing")
    placeholders = ",".join("?" for _ in keys)
    rows = connection.execute(f"""
        SELECT e.embedding_id, em.key, em.string_value, em.int_value,
               em.float_value, em.bool_value
        FROM embeddings e JOIN embedding_metadata em ON em.id=e.id
        WHERE e.segment_id=? AND e.embedding_id IN ({placeholders})
    """, (segment[0], *keys))
    records = {key: {"document": None, "metadata": {}} for key in keys}
    for key, field, *values in rows:
        value = _value(values)
        if field == "chroma:document":
            records[key]["document"] = value
        else:
            records[key]["metadata"][field] = value
    if any(not isinstance(row["document"], str) or not row["metadata"]
           for row in records.values()):
        raise ValueError("source document or metadata is incomplete")
    return records


def _configuration(schema_text):
    schema = json.loads(schema_text)
    config = schema["keys"]["#embedding"]["float_list"]["vector_index"]["config"]
    if config["space"] != "cosine":
        raise ValueError("only the current cosine generation is supported")
    hnsw = dict(config["hnsw"])
    hnsw.update(space=config["space"], num_threads=1, sync_threshold=10000)
    return {"hnsw": hnsw}


def build_copy(source: Path, vectors: Path, output: Path, *, batch_size=512) -> dict:
    source, vectors, output = source.resolve(), vectors.resolve(), output.resolve()
    if not (source.parent / "RAW_SNAPSHOT_COMPLETE").is_file():
        raise ValueError("source must be a completed immutable raw snapshot")
    if not 1 <= batch_size <= 512:
        raise ValueError("batch_size must be between 1 and 512")
    if output.exists() or source == output or source in output.parents:
        raise ValueError("output must be a new directory outside the source snapshot")
    proof = json.loads(vectors.with_suffix(".json").read_text())
    if proof.get("complete") is not True:
        raise ValueError("stored vector export is not complete")
    import chromadb
    report = {"complete": False, "collections": {}}
    with ExitStack() as resources:
        source_db = resources.enter_context(closing(sqlite3.connect(
            f"{(source / 'chroma.sqlite3').as_uri()}?mode=ro", uri=True,
        )))
        vector_db = resources.enter_context(closing(sqlite3.connect(f"{vectors.as_uri()}?mode=ro", uri=True)))
        client = chromadb.PersistentClient(path=str(output))
        resources.callback(client.close)
        collections = source_db.execute(
            "SELECT id, name, dimension, schema_str FROM collections ORDER BY name",
        ).fetchall()
        if {row[1] for row in collections} != set(proof["collections"]):
            raise ValueError("collection set differs from the vector export")
        for collection_id, name, dimension, schema in collections:
            metadata = {row[0]: _value(row[1:]) for row in source_db.execute(
                "SELECT key, str_value, int_value, float_value, bool_value "
                "FROM collection_metadata WHERE collection_id=?", (collection_id,),
            )}
            collection = client.create_collection(
                name, embedding_function=None, metadata=metadata or None,
                configuration=_configuration(schema),
            )
            digest, written = hashlib.sha256(), 0
            cursor = vector_db.execute(
                "SELECT id, vector FROM vectors WHERE collection=? ORDER BY id", (name,),
            )
            while batch := cursor.fetchmany(batch_size):
                keys = [row[0] for row in batch]
                records = stored_records(source_db, collection_id, keys)
                embeddings = []
                for key, blob in batch:
                    if len(blob) != dimension * 4:
                        raise ValueError("stored vector dimension differs from collection")
                    digest.update(json.dumps(key, ensure_ascii=False).encode() + b"\0" + blob)
                    embeddings.append(np.frombuffer(blob, dtype="<f4"))
                collection.add(
                    ids=keys, embeddings=np.asarray(embeddings),
                    documents=[records[key]["document"] for key in keys],
                    metadatas=[records[key]["metadata"] for key in keys],
                )
                written += len(batch)
                print(f"BUILD {name} {written}/{proof['collections'][name]['count']}", flush=True)
            expected = proof["collections"][name]
            if written != expected["count"] or digest.hexdigest() != expected["id_vector_sha256"]:
                raise ValueError("stored vector export hash/count changed")
            if collection.count() != written:
                raise ValueError("rebuilt collection count differs")
            report["collections"][name] = {"count": written, "input_vector_sha256": digest.hexdigest()}
        for path in source.glob("*.json"):
            shutil.copy2(path, output / path.name)
    report["complete"] = True
    (output / "stored-vector-build.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_copy(args.source, args.vectors, args.output)


if __name__ == "__main__":
    main()
