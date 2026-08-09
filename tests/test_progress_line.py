"""The line a run prints as it reaches each attachment.

Lifted out of main_async's loop, and checked here rather than by the ingestion
corpus: that check records counters, the manifest and the chunks, and says
nothing about what the run printed. A block the net cannot see is exactly the
kind that needs its own cases before it is moved.

What it has to get right is identification. An attachment with no title falls
back to its filename and then to the file on disk, because a progress line that
names none of them tells whoever is watching a long run nothing at all.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from index_from_zotero import _progress_line  # noqa: E402


def _attachment(**overrides) -> SimpleNamespace:
    return SimpleNamespace(**{
        "attachmentKey": "ATT12345", "parentItemKey": "ITEM6789",
        "title": "Imperial Eyes", "filename": "imperial-eyes.pdf", **overrides,
    })


def _line(attachment=None, *, source_type="pdf", file_path=Path("/store/on-disk.pdf")):
    return _progress_line(
        attachment or _attachment(), index=3, total=97,
        source_type=source_type, file_path=file_path,
    )


def test_the_line_carries_position_identity_and_title():
    assert _line() == (
        "[PROGRESS] (3/97) attachment=ATT12345 item=ITEM6789 type=pdf Imperial Eyes"
    )


def test_a_missing_title_falls_back_to_the_filename():
    assert _line(_attachment(title="")).endswith("imperial-eyes.pdf")


def test_a_missing_title_and_filename_fall_back_to_the_file_on_disk():
    line = _line(_attachment(title="  ", filename=None), file_path=Path("/store/x/scan.pdf"))
    assert line.endswith("scan.pdf")


def test_an_attachment_with_no_filename_attribute_is_still_named():
    # Not every attachment record carries one; the fallback must not raise.
    attachment = SimpleNamespace(attachmentKey="ATT", parentItemKey="ITEM", title="")
    assert _progress_line(
        attachment, index=1, total=1, source_type="epub",
        file_path=Path("/store/book.epub"),
    ).endswith("book.epub")


def test_a_long_title_is_cut_to_fit_a_terminal_line():
    line = _line(_attachment(title="x" * 200))
    assert line.endswith("..." )
    assert "x" * 77 + "..." in line


def test_a_title_of_exactly_the_limit_is_left_alone():
    line = _line(_attachment(title="y" * 80))
    assert line.endswith("y" * 80)
    assert "..." not in line


def test_a_top_level_attachment_says_so_rather_than_showing_a_dash():
    # No parent is ordinary -- a PDF filed at the top level of the library --
    # and it is also what a lost parent looks like, so the line says which
    # rather than printing a bare "-" that reads as missing data.
    assert "item=- (orphan?)" in _line(_attachment(parentItemKey=None))
