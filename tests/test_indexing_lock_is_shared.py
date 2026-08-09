"""The lock has one implementation, one path, and covers every writer of it.

It used to live inside ``index_from_zotero`` where nothing else could reach it,
so re-OCR adoption -- which rewrites the same four stores the indexer does --
took no lock at all. An adoption running beside an indexer could overwrite
either generation, and an MCP query landing between the two could read half of
one. Nothing in the code said so; the two writers simply did not know about
each other.

What is checked here is the arrangement rather than the mechanics (those are in
``test_indexing_lock.py``): both writers resolve the same path from the same
function, holding it excludes a second holder, and the hold is released when
the body raises -- which is the difference between a crashed run and a search
index that stays unavailable until someone deletes a file by hand.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import indexing_lock  # noqa: E402


def test_holding_the_lock_excludes_a_second_holder(tmp_path):
    lock = tmp_path / "indexing.lock"
    with indexing_lock.held(lock):
        with pytest.raises(SystemExit, match="別のインデクサー"):
            indexing_lock.acquire(lock)


def test_the_hold_is_released_when_the_body_raises(tmp_path):
    lock = tmp_path / "indexing.lock"
    with pytest.raises(RuntimeError):
        with indexing_lock.held(lock):
            raise RuntimeError("the adoption failed")
    assert not lock.exists(), "a failed unit of work kept the lock"
    indexing_lock.release(indexing_lock.acquire(lock))


def test_every_writer_asks_the_same_function_where_the_lock_is():
    # Two spellings of data/indexing.lock is the drift this project has already
    # paid for with the manifest and the Chroma directory.
    spelled_out = []
    for relative in ("src/index_from_zotero.py", "scripts/run_reocr_queue.py"):
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == "indexing.lock":
                spelled_out.append(f"{relative}:{node.lineno}")
    assert not spelled_out, (
        "the lock path is written out instead of being asked of "
        "indexing_lock.default_path:\n  " + "\n  ".join(spelled_out)
    )


def test_the_adoption_command_holds_the_lock_around_the_adoption():
    # The register's item 1: --adopt mutates Chroma, the lexical index, the
    # manifest and the structure database. Reading the call site is how this
    # can be checked without running an adoption against a real library.
    source = (ROOT / "scripts" / "run_reocr_queue.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    adoptions = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "adopt_prepared_reocr"
    ]
    assert adoptions, "the adoption call moved; this check no longer looks at it"
    held_ranges = [
        (node.lineno, node.end_lineno) for node in ast.walk(tree)
        if isinstance(node, ast.With)
        and any(
            getattr(item.context_expr.func, "id", "") == "indexing_lock_held"
            for item in node.items if isinstance(item.context_expr, ast.Call)
        )
    ]
    for call in adoptions:
        assert any(start <= call.lineno <= end for start, end in held_ranges), (
            f"adopt_prepared_reocr at line {call.lineno} runs without the "
            "indexing lock, so an indexer can overwrite what it writes"
        )
