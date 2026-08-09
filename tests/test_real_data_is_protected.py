"""The two walls that stand between a test and the user's library.

Written after 2026-08-09, when a test run purged 574 items, 205,538 citations
and 41,133 references out of the real ``relations.db``. Two separate things
were missing, and each would have been enough on its own:

* the test harness let a run reach the real data directory at all;
* the purge itself had no idea that deleting the whole library was unusual,
  even though the attachment side of the same run refuses exactly that, with a
  comment explaining why.

Both are checked here, by doing the thing rather than by describing it. A guard
nobody has ever seen fail is a guard nobody knows is connected.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import pytest

from tests.conftest import PROTECTED, RealDataWriteAttempted

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def test_writing_into_the_real_data_directory_fails():
    with pytest.raises(RealDataWriteAttempted):
        open(PROTECTED / "a-test-should-never-write-this", "w").close()


def test_appending_to_a_real_file_fails():
    # Append is how the citation mapper's debug log reached data/ unnoticed:
    # it creates nothing visible and truncates nothing, so it left no trace.
    with pytest.raises(RealDataWriteAttempted):
        open(PROTECTED / "mapping_debug.log", "a").close()


def test_reading_the_real_library_is_still_allowed():
    # The corpus tests re-derive their baselines from the indexed library, so
    # the wall has to stop writes without stopping reads.
    listing = sorted(PROTECTED.glob("*")) if PROTECTED.exists() else []
    assert isinstance(listing, list)
    if (PROTECTED / "manifest_v3.json").exists():
        (PROTECTED / "manifest_v3.json").read_text(encoding="utf-8")


def test_opening_a_real_database_read_write_fails():
    if not (PROTECTED / "relations.db").exists():
        pytest.skip("no relations database on this machine")
    with pytest.raises(RealDataWriteAttempted):
        sqlite3.connect(str(PROTECTED / "relations.db"))


def test_the_relations_database_a_test_gets_is_not_the_real_one():
    # The default that means most tests never meet the wall at all.
    configured = Path(os.environ["RELATIONS_DB_PATH"]).resolve()
    assert PROTECTED != configured and PROTECTED not in configured.parents


def _seeded(path: Path, item_keys: list[str]) -> None:
    import db_relations

    os.environ["RELATIONS_DB_PATH"] = str(path)
    db_relations.DB_PATH = str(path)
    connection = db_relations.get_db_connection()
    try:
        connection.executemany(
            "INSERT INTO item_citation_status (item_key, s2_status) VALUES (?, 'matched')",
            [(key,) for key in item_keys],
        )
        connection.commit()
    finally:
        connection.close()


def test_a_purge_of_almost_everything_is_refused(tmp_path, capsys, monkeypatch):
    # The incident, reproduced: a caller that can only see three items must not
    # be able to delete the relations of the other five hundred.
    import db_relations

    database = tmp_path / "relations.db"
    monkeypatch.setattr(db_relations, "DB_PATH", str(database), raising=False)
    _seeded(database, [f"ITEM{index:04d}" for index in range(500)])

    counts = db_relations.purge_removed_items({"ITEM0001", "ITEM0002", "ITEM0003"})

    assert counts.get("refused") == 497
    assert counts["item_citation_status"] == 0
    assert "Refusing to purge" in capsys.readouterr().err
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        remaining = connection.execute(
            "SELECT COUNT(*) FROM item_citation_status"
        ).fetchone()[0]
    assert remaining == 500, "rows were deleted despite the refusal"


def test_an_ordinary_deletion_still_goes_through(tmp_path, monkeypatch):
    # The guard has to leave the normal case alone, or it will be turned off.
    import db_relations

    database = tmp_path / "relations.db"
    monkeypatch.setattr(db_relations, "DB_PATH", str(database), raising=False)
    keys = [f"ITEM{index:04d}" for index in range(500)]
    _seeded(database, keys)

    counts = db_relations.purge_removed_items(set(keys[:-5]))

    assert "refused" not in counts
    assert counts["item_citation_status"] == 5


def test_a_deliberate_bulk_purge_can_still_say_so(tmp_path, monkeypatch):
    import db_relations

    database = tmp_path / "relations.db"
    monkeypatch.setattr(db_relations, "DB_PATH", str(database), raising=False)
    _seeded(database, [f"ITEM{index:04d}" for index in range(500)])

    counts = db_relations.purge_removed_items(set(), force=True)

    assert counts["item_citation_status"] == 500


def test_a_refused_purge_is_not_reported_as_a_purge(capfd):
    """The refusal has to read as a refusal.

    The guard returns its count under "refused", and the run summed every value
    in that dict: a refused purge made the total non-zero and printed "Purged
    removed items ... =0, =0, =0" -- announcing a purge at the moment one was
    prevented, which is the sentence an operator would act on.
    """
    import index_from_zotero

    index_from_zotero._report_purge({
        "item_citation_status": 0, "global_citations": 0, "global_references": 0,
        "refused": 571,
    })
    # capfd, not capsys: the progress output goes to sys.__stderr__, which
    # capsys does not see, so a capsys-based assertion reads as an empty string
    # and passes for the wrong reason.
    refused_output = capfd.readouterr().err
    assert "Refused to purge 571" in refused_output
    assert "Purged removed items" not in refused_output, (
        "a refused purge is announced as a purge that happened"
    )

    index_from_zotero._report_purge({
        "item_citation_status": 2, "global_citations": 7, "global_references": 1,
    })
    assert "Purged removed items" in capfd.readouterr().err


def test_the_confirming_bulk_command_can_still_purge():
    # purge_orphans confirms every candidate against Zotero individually and
    # applies its own ratio guard, so it has to be able to get past the guard
    # inside purge_removed_items -- which the guard's own comment says is the
    # way to do a deliberate bulk removal.
    import ast

    source = (ROOT / "scripts" / "purge_orphans.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", "") == "purge_removed_items"
    ]
    assert calls, "the call moved; this check no longer looks at it"
    for call in calls:
        assert any(
            keyword.arg == "force" and getattr(keyword.value, "value", False) is True
            for keyword in call.keywords
        ), (
            f"purge_removed_items at line {call.lineno} runs without force, so "
            "the guard refuses it and the script reports an empty result as success"
        )
