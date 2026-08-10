from __future__ import annotations

from unittest.mock import patch

import pytest

from src import pdf_extract


def test_finalization_preserves_chunks_and_adds_defects_without_mutating_input():
    chunks = [
        ("ATT:p1:para0:part0", "first body", {"page": 1}),
        ("ATT:p1:para1:part0", "second body", {"page": 1}),
    ]
    quality = {"total_pages": 1}
    defects = {"has_text_defects": True, "defect_reason": "synthetic"}

    with patch.object(pdf_extract, "detect_text_defects", return_value=defects) as detect:
        finalized_chunks, finalized_quality = pdf_extract._finalize_pdf_output(
            chunks, quality,
        )

    assert finalized_chunks is chunks
    assert finalized_quality == {"total_pages": 1, **defects}
    assert quality == {"total_pages": 1}
    detect.assert_called_once_with("first body\nsecond body")


def test_finalization_rejects_duplicate_chunk_ids_before_defect_detection():
    chunks = [
        ("duplicate", "first", {}),
        ("duplicate", "second", {}),
    ]

    with patch.object(pdf_extract, "detect_text_defects") as detect:
        with pytest.raises(RuntimeError, match=r"Duplicate chunk ids generated \(1\)"):
            pdf_extract._finalize_pdf_output(chunks, {})

    detect.assert_not_called()
