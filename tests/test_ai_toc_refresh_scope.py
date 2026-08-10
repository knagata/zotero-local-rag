"""Resolved AI-TOC refresh targets must be nonempty and PDF-only."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from index_from_zotero import _validate_ai_toc_refresh_targets


def _attachment(name: str, source_type: str):
    return SimpleNamespace(
        attachmentKey=name.upper(),
        contentType=("application/epub+zip" if source_type == "epub" else None),
        pdf_path=Path(f"source.{source_type}"),
        source_type=source_type,
    )


def test_refresh_accepts_a_resolved_pdf_target():
    _validate_ai_toc_refresh_targets(
        SimpleNamespace(refresh_ai_toc=True), [_attachment("pdf", "pdf")],
    )


def test_refresh_rejects_an_empty_resolved_scope():
    with pytest.raises(SystemExit, match="resolved no PDF attachments"):
        _validate_ai_toc_refresh_targets(
            SimpleNamespace(refresh_ai_toc=True), [],
        )


@pytest.mark.parametrize("source_type", ["epub", "html"])
def test_refresh_rejects_a_resolved_non_pdf_target(source_type: str):
    with pytest.raises(SystemExit, match="resolved non-PDF attachments"):
        _validate_ai_toc_refresh_targets(
            SimpleNamespace(refresh_ai_toc=True),
            [_attachment(source_type, source_type)],
        )


def test_refresh_rejects_mixed_item_attachments_before_any_reindex():
    with pytest.raises(SystemExit, match="EPUB"):
        _validate_ai_toc_refresh_targets(
            SimpleNamespace(refresh_ai_toc=True),
            [_attachment("pdf", "pdf"), _attachment("epub", "epub")],
        )


def test_ordinary_ingestion_does_not_apply_the_refresh_scope_guard():
    _validate_ai_toc_refresh_targets(
        SimpleNamespace(refresh_ai_toc=False), [_attachment("epub", "epub")],
    )
