from pathlib import Path
from subprocess import CompletedProcess
from unittest import mock

from src.pdf_extract import _extract_pages_with_poppler


def test_poppler_extraction_preserves_page_boundaries():
    completed = CompletedProcess(
        args=["pdftotext"], returncode=0,
        stdout="First page\fSecond page\f", stderr="",
    )
    with (
        mock.patch("src.pdf_extract._find_pdftotext", return_value="/usr/bin/pdftotext"),
        mock.patch("subprocess.run", return_value=completed) as run,
    ):
        pages = _extract_pages_with_poppler(Path("sample.pdf"))
    assert pages == ["First page", "Second page"]
    assert "-layout" in run.call_args.args[0]


def test_poppler_failure_is_fail_closed():
    completed = CompletedProcess(
        args=["pdftotext"], returncode=1, stdout="", stderr="broken",
    )
    with (
        mock.patch("src.pdf_extract._find_pdftotext", return_value="/usr/bin/pdftotext"),
        mock.patch("subprocess.run", return_value=completed),
    ):
        assert _extract_pages_with_poppler(Path("sample.pdf")) == []
