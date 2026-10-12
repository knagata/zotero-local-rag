"""What the run owes an attachment, decided in one place.

``_source_verdict`` is the first piece lifted out of ``main_async``'s
1,389-line loop. The corpus check beside it runs the loop for real and would
catch most ways of breaking this, but not all: it never passes --check-quality,
so the branch that re-reads quality on request is invisible to it. These are the
cases stated directly, and they are what CI enforces -- the corpus check needs
the library and skips without it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from index_from_zotero import _source_verdict  # noqa: E402
from index_run import ReparseDecision  # noqa: E402

FINGERPRINT = "sha256:pipeline"


def _args(**overrides) -> argparse.Namespace:
    return argparse.Namespace(**{
        "retry_failed": False, "force_reparse": False, "check_quality": False,
        **overrides,
    })


def _indexed(**overrides) -> dict:
    """A manifest row for a file that was indexed by this pipeline."""
    return {
        "mtime": 100.0, "size": 2048, "quality": {"is_scanned": False},
        "pipeline_fingerprint": FINGERPRINT, **overrides,
    }


def _verdict(args=None, previous=None, *, mtime=100.0, size=2048,
             inflight=frozenset(), reparse=ReparseDecision()):
    return _source_verdict(
        args or _args(), attachment_key="ATT", previous=previous,
        file_path=Path("/nonexistent/source.pdf"), mtime=mtime, size=size,
        pipeline_fingerprint=FINGERPRINT,
        inflight=set(inflight), reparse=reparse,
    )


def test_a_file_never_seen_before_is_indexed():
    assert _verdict(previous=None).action == "index"


def test_an_unchanged_file_that_is_already_understood_is_skipped():
    assert _verdict(previous=_indexed()).action == "skip"


@pytest.mark.parametrize("changed", [{"mtime": 101.0}, {"size": 4096}])
def test_a_file_whose_stat_moved_is_indexed(changed):
    assert _verdict(previous=_indexed(**changed)).action == "index"


def test_quality_is_read_when_the_row_has_none():
    # The bytes are unchanged, so there is nothing to extract, but nothing is
    # known about them either.
    row = _indexed()
    del row["quality"]
    assert _verdict(previous=row).action == "quality_only"


def test_quality_is_read_again_when_it_is_asked_for():
    # Not covered by the corpus check, which never passes the flag.
    assert _verdict(_args(check_quality=True), _indexed()).action == "quality_only"


def test_an_attachment_a_batch_still_holds_is_indexed():
    # In flight means an OCR batch was submitted for it; its manifest row
    # describes the source, not an adopted result, so it is not settled.
    assert _verdict(previous=_indexed(), inflight={"ATT"}).action == "index"


def test_a_row_from_another_pipeline_is_indexed():
    assert _verdict(previous=_indexed(pipeline_fingerprint="sha256:older")).action == "index"


@pytest.mark.parametrize("flag", ["retry_failed", "force_reparse"])
def test_the_invocation_can_demand_extraction(flag):
    assert _verdict(_args(**{flag: True}), _indexed()).action == "index"


@pytest.mark.parametrize(
    "reparse",
    [ReparseDecision(force_docling=True), ReparseDecision(force_ndlocr=True),
     ReparseDecision(force_mistral=True)],
)
def test_a_forced_parser_demands_extraction(reparse):
    assert _verdict(previous=_indexed(), reparse=reparse).action == "index"


def test_a_hash_that_could_not_be_taken_leaves_the_stat_match_standing():
    # The hash is attempted only when mtime and size already match a row that
    # carried one. Here the file cannot be read, so no hash comes back, and
    # _source_content_unchanged compares signatures only when both sides have
    # one -- so the stat match decides and the attachment is treated as
    # unchanged. Recorded as the behaviour it is rather than the behaviour that
    # might be wanted: an unreadable file would fail extraction anyway, and
    # changing this without evidence would only move where it fails.
    verdict = _verdict(previous=_indexed(content_signature="sha256:whatever"))
    assert verdict.signature is None
    assert verdict.action == "skip"


def test_the_hash_is_not_taken_when_the_row_carries_none():
    # Nothing to compare it against, so the stat match is the whole answer and
    # the file is not read at all -- which is why a missing file is fine here.
    assert _verdict(previous=_indexed()).signature is None


def test_renamed_unsigned_source_is_reindexed():
    assert _verdict(previous=_indexed(pdf_path="/old.pdf")).action == "index"


def test_verified_rename_synchronizes_and_saves(tmp_path, monkeypatch):
    import index_from_zotero as indexer
    from manifest import content_signature
    path = tmp_path / "new.pdf"
    path.write_bytes(b"verified contents")
    stat = path.stat()
    previous = _indexed(pdf_path=str(tmp_path / "old.pdf"), mtime=stat.st_mtime,
                        size=stat.st_size, content_signature=content_signature(path, stat.st_size))
    verdict = indexer._source_verdict(
        _args(), attachment_key="ATT", previous=previous, file_path=path,
        mtime=stat.st_mtime, size=stat.st_size, pipeline_fingerprint=FINGERPRINT,
        inflight=set(), reparse=ReparseDecision(),
    )
    saved = []
    monkeypatch.setattr(indexer, "save_manifest", lambda p, m: saved.append(m))
    manifest = {"files": {"ATT": previous}}
    assert verdict.action == "skip"
    assert indexer._sync_unchanged_source_path(previous, path, verdict, manifest) == verdict.signature
    assert previous["pdf_path"] == str(path)
    assert saved == [manifest]
    indexer._sync_unchanged_source_path(previous, path, verdict, manifest)
    assert len(saved) == 1
