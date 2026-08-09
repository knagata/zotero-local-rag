"""What a run says and records when an attachment yields no chunks.

Lifted out of main_async's loop and checked here, because the ingestion corpus
cannot reach it: all three of its attachments extract successfully, so a change
to this path would leave that check green. The same reason the progress line
has its own cases.

The behaviour worth pinning is that the failure is *legible afterwards*. A
zero-chunk attachment never enters the manifest, so the run moves on and the
file looks untouched; unless the reason is printed here it cannot be recovered
later (2026-08-02, diagnosing 3QHTRQN7).
"""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import index_from_zotero  # noqa: E402

ATTACHMENT = SimpleNamespace(attachmentKey="ATT12345", parentItemKey="ITEM6789")


def _report(**overrides):
    """Run the reporter, returning (printed line, mark_artifact_status calls)."""
    calls = []
    kwargs = {
        "scope_item_key": "ITEM6789", "source_type": "html",
        "file_path": Path("/store/ATT12345/page.html"), "quality": {},
        "forced_mistral": False, "mtime": 100.0, "size": 2048, **overrides,
    }
    captured = io.StringIO()
    with patch.object(index_from_zotero, "mark_artifact_status",
                      side_effect=lambda *a, **k: calls.append((a, k))), \
            patch.object(sys, "__stderr__", captured), redirect_stderr(captured):
        index_from_zotero._report_empty_extraction(ATTACHMENT, **kwargs)
    return captured.getvalue(), calls


def test_the_attachment_and_its_file_are_named():
    line, _ = _report()
    assert "attachment=ATT12345" in line
    assert "type=html" in line
    assert "/store/ATT12345/page.html" in line


def test_the_reason_is_printed_when_the_extractor_recorded_one():
    line, _ = _report(quality={"failure_reason": "no_dom_blocks"})
    assert "reason=no_dom_blocks" in line


def test_no_reason_is_printed_rather_than_a_misleading_none():
    # Extractors other than the HTML one set no such key, and "reason=None"
    # reads as a recorded reason that happens to be empty.
    line, _ = _report(source_type="pdf", quality={})
    assert "reason=" not in line


def test_an_ordinary_failure_is_recorded_as_failed_and_retryable():
    _, calls = _report()
    (positional, keywords), = calls
    assert positional == ("ITEM6789", "extraction", "failed")
    assert keywords["reason_code"] == "no_chunks"
    assert keywords["retryable"] is True


def test_an_empty_mistral_batch_is_blocked_rather_than_failed():
    # The candidate is deliberately kept for a later batch, so calling this a
    # failure would invite a retry of the one thing already known not to work.
    _, calls = _report(forced_mistral=True)
    (positional, keywords), = calls
    assert positional == ("ITEM6789", "extraction", "blocked")
    assert keywords["retryable"] is False
    assert keywords["reason_code"] == index_from_zotero.MISTRAL_TOC_QUEUE_REASON
    assert keywords["counts"] == {"source_mtime": 100.0, "source_size": 2048}


def test_exactly_one_status_is_written_either_way():
    for forced in (True, False):
        _, calls = _report(forced_mistral=forced)
        assert len(calls) == 1
