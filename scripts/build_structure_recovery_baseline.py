#!/usr/bin/env python3
"""Record what flat-PDF structure recovery currently produces, book by book.

Every change to the recovery rules so far has been checked by counting how many
documents came out non-flat. That number is nearly useless on its own: the run
that introduced a contents region which swallowed a book's first 171 pages
scored *better* than the run before it, because a wrong tree counts the same as
a right one. The trees themselves have to be looked at, and looking at 84 of
them by hand every time is not something anyone will keep doing.

So they are recorded here instead, and ``tests/test_structure_recovery_corpus``
re-derives them and compares. A rule change then shows up as a diff over real
books rather than as a number, and one that quietly turns a chapter list into a
notes index cannot pass.

Only the *output* is stored. Freezing the input rows as a fixture would make the
test runnable without the library, but the corpus is 31,068 rows and 2.5 MB --
five times the largest file in the repository -- and the only way to shrink it
is to drop the ordinary body-text rows, which is exactly where the last round of
faults lived (a plain sentence reading as a part opener). A baseline of trees is
20 KB and costs nothing.

    uv run python scripts/build_structure_recovery_baseline.py            # diff
    uv run python scripts/build_structure_recovery_baseline.py --write    # adopt
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.env_utils import load_dotenv_native  # noqa: E402

load_dotenv_native(ROOT)

from src.chunk_store import get_item_chunks  # noqa: E402
from src.flat_structure_diagnostics import diagnose_flat_item  # noqa: E402
from src.source_structure_refresh import (  # noqa: E402
    _refresh_pdf_rows_from_numbered_body_headings,
)

BASELINE_PATH = ROOT / "tests" / "baselines" / "structure_recovery.json"

#: The reason code whose attachments this recovery path exists to serve. A flat
#: PDF with body headings the extractor kept but never assembled into a tree.
CORPUS_REASON = "pdf_body_heading_recovery_candidate"


def recovered_paths(rows: list[dict], attachment_key: str) -> list[list[str]] | None:
    """The boundaries the recovery would lay down, in order, or None if it declines.

    Consecutive chunks carrying the same path collapse to one entry: the shape of
    the tree is what matters here, not how many chunks each node holds.
    ``source_path`` reaches only the report dict, never a decision, so a
    placeholder keeps this independent of whether the PDF is still on disk.
    """
    result = _refresh_pdf_rows_from_numbered_body_headings(
        rows, attachment_key, Path("<not read>"),
    )
    if not result:
        return None
    paths: list[list[str]] = []
    previous: tuple[str, ...] | None = None
    for row in result[0]:
        current = tuple((row.get("metadata") or {}).get("structure_path") or ())
        if current and current != previous:
            paths.append(list(current))
        previous = current
    return paths


def _attachment_rows(item_key: str, attachment_key: str) -> list[dict]:
    return [
        row for row in get_item_chunks(item_key)
        if (row.get("metadata") or {}).get("attachmentKey") == attachment_key
    ]


def measure(entries: list[dict]) -> list[dict]:
    """Run the recovery over each listed attachment and record its tree."""
    measured = []
    for entry in entries:
        rows = _attachment_rows(entry["item_key"], entry["attachment_key"])
        measured.append({
            "item_key": entry["item_key"],
            "attachment_key": entry["attachment_key"],
            "title": entry.get("title", ""),
            "chunks": len(rows),
            "paths": recovered_paths(rows, entry["attachment_key"]),
        })
    measured.sort(key=lambda row: (row["item_key"], row["attachment_key"]))
    return measured


def _flat_item_keys() -> list[str]:
    """Items whose canonical tree is still the flat fallback.

    The same selection ``scripts/diagnose_flat_structures.py`` makes, which is
    where this corpus is defined.
    """
    from src.db_relations import get_db_connection

    connection = get_db_connection()
    try:
        return [str(row[0]) for row in connection.execute(
            "SELECT item_key FROM document_structures "
            "WHERE status = 'flat_fallback' ORDER BY item_key"
        )]
    finally:
        connection.close()


def discover(recorded: list[dict]) -> list[dict]:
    """Every attachment this recovery path answers for, old entries kept.

    Membership is defined by an item still being flat, but a book leaves that
    set the moment its recovered tree is written to the index -- which would
    drop precisely the books whose trees are worth pinning. So the recorded
    entries are carried forward unconditionally and discovery only adds to them.
    A book that has been rebuilt is still a valid input to this function, and
    its tree must not change silently afterwards.
    """
    entries = {
        (row["item_key"], row["attachment_key"]): {
            "item_key": row["item_key"], "attachment_key": row["attachment_key"],
            "title": row.get("title", ""),
        }
        for row in recorded
    }
    for item_key in _flat_item_keys():
        chunks = get_item_chunks(item_key)
        if not chunks:
            continue
        for attachment in diagnose_flat_item(item_key, chunks)["attachments"]:
            if attachment["reason_code"] != CORPUS_REASON:
                continue
            entries.setdefault((item_key, attachment["attachment_key"]), {
                "item_key": item_key,
                "attachment_key": attachment["attachment_key"],
                "title": attachment.get("title", ""),
            })
    return list(entries.values())


def describe(row: dict) -> str:
    if row["paths"] is None:
        return "        (flat -- recovery declined)"
    return "\n".join(f"        {' / '.join(path)}" for path in row["paths"])


def diff(previous: list[dict], current: list[dict]) -> list[str]:
    old = {(row["item_key"], row["attachment_key"]): row for row in previous}
    new = {(row["item_key"], row["attachment_key"]): row for row in current}
    lines = []
    for key in sorted(old.keys() | new.keys()):
        before, after = old.get(key), new.get(key)
        if before is None:
            lines.append(f"  + {key[0]} entered the corpus")
            lines.append(describe(after))
            continue
        if after is None:  # pragma: no cover -- discover() never drops an entry
            lines.append(f"  - {key[0]} left the corpus")
            continue
        if before["paths"] != after["paths"]:
            lines.append(f"  ~ {key[0]}  {after.get('title', '')[:52]}")
            lines.append("      was:")
            lines.append(describe(before))
            lines.append("      now:")
            lines.append(describe(after))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true",
                        help="adopt the current output as the baseline")
    args = parser.parse_args()

    previous = (
        json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        if BASELINE_PATH.exists() else {"items": []}
    )
    entries = discover(previous["items"])
    current = measure(entries)

    recovered = sum(1 for row in current if row["paths"] is not None)
    print(f"{len(current)} attachments in the corpus; {recovered} recover a tree",
          file=sys.stderr)

    changes = diff(previous["items"], current)
    if changes:
        print("\n".join(changes), file=sys.stderr)
    else:
        print("no change against the recorded baseline", file=sys.stderr)

    if not args.write:
        if changes:
            print("\nrun again with --write to adopt these trees", file=sys.stderr)
            raise SystemExit(1)
        return

    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(
        json.dumps(
            {"reason_code": CORPUS_REASON, "recorded": date.today().isoformat(),
             "attachments": len(current), "recovered": recovered, "items": current},
            ensure_ascii=False, indent=1,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {BASELINE_PATH.relative_to(ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
