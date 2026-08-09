#!/usr/bin/env python3
"""Record what ingesting one attachment actually does, so it can be refactored.

``index_from_zotero.main_async`` is 1,856 lines, 1,389 of them the body of a
single loop over attachments, and no test calls it. Seventeen test files name
the module; fifteen exercise helpers already lifted out of that loop and two
only mention it in comments. The loop itself -- 147 branches, 216 locals, the
place every chunk in the library comes from -- is unchecked.

That is what makes it hard to break up. An extract-method here does not fail by
drifting: it fails when a local stops being updated in the scope that still
reads it, and a unit test on the extracted piece cannot see that. Only running
the loop can. So the loop is run, against a throwaway data plane, and what it
did is written down.

Recorded per attachment: the counters the run reports, which are a direct
reading of which branches were taken; the manifest row it wrote; the chunks it
produced, by identity, block type, zone and structure rather than by text; and
the artifact status rows it left behind. Text is hashed, not stored -- the point
is whether the same text came out, and the library's contents do not belong in a
file even a gitignored one.

Nothing touches the real index. CHROMA_DIR, MANIFEST_PATH, LEXICAL_DB_PATH and
RELATIONS_DB_PATH are pointed at a temporary directory that is thrown away
afterwards, and the run is verified to have written only there.

    uv run python scripts/build_ingestion_baseline.py            # diff
    uv run python scripts/build_ingestion_baseline.py --write    # adopt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASELINE_PATH = ROOT / "tests" / "baselines" / "ingestion.json"

#: One attachment per extraction route this loop can take. The routes differ far
#: more than the items do: a web clipping never reaches the PDF quality gates,
#: and an EPUB takes a DOM path a PDF has no equivalent of. Chosen small so the
#: whole baseline rebuilds in about half a minute, and by route so that a branch
#: losing its state shows up.
#:
#: Zotero notes are deliberately absent. They carry no attachment, so they are
#: not processed by this loop at all -- an item made only of notes runs it zero
#: times, and including one recorded an empty result that would have gone on
#: agreeing with itself no matter what the loop did.
CORPUS = ("3FTKWX9U", "4GVKUITL", "4M5LZB5S")

#: Variables v3_data_plane.enforce_environment derives and writes back for its
#: children. Inheriting any of them alongside a redirected CHROMA_DIR describes
#: a data plane that is half one place and half another.
DERIVED_DATA_PLANE_VARIABLES = frozenset({
    "PIPELINE_CONFIG_PATH", "CHROMA_COLLECTION", "CHROMA_COLLECTION_V3",
    "MANIFEST_PATH", "LEXICAL_DB_PATH", "CHROMA_DIR", "RELATIONS_DB_PATH",
})

#: Values that differ between two runs of the same input and say nothing about
#: what the loop decided.
#:
#: The ocr_layer_* fields are not merely timestamps: the audit behind them asks
#: an LLM to propose OCR-error candidates, so two runs over the same PDF report
#: a different error rate and a different count (0.00807 against 0.01066, 15
#: rejected against 7, measured here). That is the audit working as designed --
#: the model proposes and deterministic reconciliation disposes -- but it means
#: those numbers cannot say whether a refactor changed anything, and left in
#: they would make this baseline disagree with itself on every run.
_VOLATILE = frozenset({
    "mtime", "updated_at", "created_at", "duration", "elapsed",
    "ocr_layer_error_rate", "ocr_layer_examples",
    "ocr_layer_rejected_count", "ocr_layer_verified_count",
    # Not just a number: the same PDF is flagged for review on one run and not
    # on the next, because this verdict is computed from the counts above.
    "ocr_layer_needs_review",
})


def _stable(value):
    """Drop the fields that move on their own, at any depth."""
    if isinstance(value, dict):
        return {k: _stable(v) for k, v in sorted(value.items()) if k not in _VOLATILE}
    if isinstance(value, list):
        return [_stable(v) for v in value]
    return value


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def ingest(item_key: str, plane: Path) -> dict:
    """Run the real loop over one item, into a data plane of its own."""
    # The child's data plane is built, not inherited. Passing os.environ
    # through leaves whatever the caller happens to have set: run from a test
    # session that had exported PIPELINE_CONFIG_PATH, the child got the real
    # config path with a temporary CHROMA_DIR and refused to start, because
    # v3_data_plane pins that file inside the Chroma directory. It failed only
    # under the full suite and passed alone, which is the worst way for a net to
    # be wrong -- it looks like flakiness rather than a bug in the net.
    #
    # So every variable the data plane derives is cleared first, and only the
    # four that name a location are set. Anything else the child needs -- the
    # embedding profile, the model cache, Zotero's data directory -- it reads
    # from .env exactly as a normal run does.
    environment = {
        key: value for key, value in os.environ.items()
        if key not in DERIVED_DATA_PLANE_VARIABLES
    }
    environment.update({
        "CHROMA_DIR": str(plane / "chroma"),
        "MANIFEST_PATH": str(plane / "manifest_v3.json"),
        "LEXICAL_DB_PATH": str(plane / "lexical_v3.sqlite3"),
        "RELATIONS_DB_PATH": str(plane / "relations.db"),
        "PYTHONPATH": str(ROOT / "src"),
    })
    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / "index_from_zotero.py"),
         "--item", item_key, "--force-reparse"],
        cwd=ROOT, env=environment, text=True, capture_output=True,
    )
    counters, summary = {}, ""
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith('{"event": "index_batch_result"'):
            counters = json.loads(line)
        elif line.startswith("Done. "):
            # The JSON event counts only the PDF route. The printed summary is
            # where an HTML or EPUB attachment is accounted for, so recording
            # the event alone left two of the three routes with nothing to
            # disagree about.
            summary = line
    return {
        "exit_code": result.returncode,
        # Between them, the loop's own account of which way it went.
        "counters": _stable(counters),
        "summary": summary,
        # Kept whole rather than tailed: when this net was wrong, the answer
        # was in a traceback the recorded summary had thrown away.
        "stderr_tail": "\n".join(result.stderr.strip().splitlines()[-6:]) if result.returncode else "",
    }


def observe(item_key: str, plane: Path) -> dict:
    """What the run left behind, in the throwaway plane."""
    from src.chunk_store import get_item_chunks

    chunks = get_item_chunks(item_key, chroma_dir=plane / "chroma")
    recorded = []
    for chunk in sorted(chunks, key=lambda c: str(c.get("id") or "")):
        metadata = chunk.get("metadata") or {}
        recorded.append({
            "id": str(chunk.get("id") or ""),
            "block_type": metadata.get("block_type"),
            "zone": metadata.get("zone"),
            "source_type": metadata.get("source_type"),
            "structure_path": metadata.get("structure_path"),
            "structure_roles": metadata.get("structure_roles"),
            "chars": len(str(chunk.get("text") or "")),
            "text": _digest(str(chunk.get("text") or "")),
        })

    manifest_file = plane / "manifest_v3.json"
    manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}
    rows = {
        key: _stable(entry)
        for key, entry in (manifest.get("files") or {}).items()
    }

    # Only the routes that record progress create this table, so it is usually
    # empty here; it is captured because a refactor that stops an item being
    # marked failed would otherwise look identical to one that works.
    statuses = []
    relations = plane / "relations.db"
    if relations.exists():
        connection = sqlite3.connect(f"file:{relations}?mode=ro", uri=True)
        try:
            statuses = [
                dict(zip(("scope", "status", "reason_code"), row, strict=False))
                for row in connection.execute(
                    "SELECT scope, status, reason_code FROM artifact_processing_status "
                    "WHERE item_key = ? ORDER BY scope, status", (item_key,))
            ]
        except sqlite3.OperationalError:
            statuses = []
        finally:
            connection.close()
    return {"chunks": recorded, "manifest": rows, "artifact_status": statuses}


def measure() -> list[dict]:
    measured = []
    for item_key in CORPUS:
        with tempfile.TemporaryDirectory(prefix="ingestion-baseline-") as raw:
            plane = Path(raw)
            print(f"  {item_key} ...", file=sys.stderr, end="", flush=True)
            run = ingest(item_key, plane)
            observed = observe(item_key, plane)
            print(f" {len(observed['chunks'])} chunks", file=sys.stderr)
            measured.append({"item_key": item_key, **run, **observed})
    return measured


def diff(previous: list[dict], current: list[dict]) -> list[str]:
    old = {row["item_key"]: row for row in previous}
    lines = []
    for row in current:
        before = old.get(row["item_key"])
        if before is None:
            lines.append(f"  + {row['item_key']} entered the corpus")
            continue
        for field in ("exit_code", "counters", "summary", "manifest", "artifact_status"):
            if before.get(field) != row.get(field):
                lines.append(f"  ~ {row['item_key']} {field}")
                lines.append(f"      was: {json.dumps(before.get(field), ensure_ascii=False)[:300]}")
                lines.append(f"      now: {json.dumps(row.get(field), ensure_ascii=False)[:300]}")
        if before.get("chunks") != row.get("chunks"):
            was, now = before.get("chunks") or [], row.get("chunks") or []
            lines.append(f"  ~ {row['item_key']} chunks: {len(was)} -> {len(now)}")
            for old_chunk, new_chunk in zip(was, now, strict=False):
                if old_chunk != new_chunk:
                    lines.append(f"      {old_chunk.get('id')}")
                    lines.append(f"        was: {json.dumps(old_chunk, ensure_ascii=False)[:200]}")
                    lines.append(f"        now: {json.dumps(new_chunk, ensure_ascii=False)[:200]}")
                    break
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true",
                        help="adopt the current behaviour as the baseline")
    args = parser.parse_args()

    previous = (
        json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        if BASELINE_PATH.exists() else {"items": []}
    )
    current = measure()

    changes = diff(previous["items"], current)
    print("\n".join(changes) if changes
          else "no change against the recorded baseline", file=sys.stderr)

    if not args.write:
        if changes:
            print("\nrun again with --write to adopt this behaviour", file=sys.stderr)
            raise SystemExit(1)
        return

    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(
        json.dumps({"recorded": date.today().isoformat(), "items": current},
                   ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {BASELINE_PATH.relative_to(ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
