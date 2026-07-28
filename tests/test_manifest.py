"""Tests for content_signature, the fingerprint mtime/size alone could not give.

The skip check that decides whether to re-parse a source compared only mtime
and size. A file replaced at the same path with the same byte-count -- a
corrected scan re-saved from the same source, a sync tool that does not always
preserve mtime -- kept its stale text indefinitely, and nothing could detect
it: the manifest recorded no information a replacement would change
(2026-07-28).
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from manifest import CONTENT_SIGNATURE_SAMPLE_BYTES, content_signature  # noqa: E402


class ContentSignatureTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)

    def _write(self, name: str, data: bytes) -> Path:
        path = Path(self.tempdir.name) / name
        path.write_bytes(data)
        return path

    def test_identical_content_yields_the_same_signature(self):
        a = self._write("a.pdf", b"same content")
        b = self._write("b.pdf", b"same content")
        self.assertEqual(
            content_signature(a, a.stat().st_size), content_signature(b, b.stat().st_size))

    def test_different_content_of_the_same_size_yields_a_different_signature(self):
        # The exact case mtime+size alone cannot see: a same-size replacement.
        a = self._write("a.pdf", b"aaaaaaaaaa")
        b = self._write("b.pdf", b"bbbbbbbbbb")
        self.assertEqual(a.stat().st_size, b.stat().st_size)
        self.assertNotEqual(
            content_signature(a, a.stat().st_size), content_signature(b, b.stat().st_size))

    def test_a_change_beyond_the_sampled_window_is_still_caught_via_size(self):
        # Content differing only in the untouched middle of a large file would
        # be invisible to the sampled bytes alone; the signature also binds
        # the declared size, so at minimum a size change is never missed, and
        # in practice most real replacements differ near the start (front
        # matter) or end (trailer/colophon), which the sample does see.
        big = self._write("big.pdf", b"x" * (CONTENT_SIGNATURE_SAMPLE_BYTES * 3))
        smaller = self._write("smaller.pdf", b"x" * (CONTENT_SIGNATURE_SAMPLE_BYTES * 3 - 1))
        self.assertNotEqual(
            content_signature(big, big.stat().st_size),
            content_signature(smaller, smaller.stat().st_size),
        )

    def test_a_change_within_the_sampled_head_or_tail_is_caught_on_a_large_file(self):
        base = b"H" * CONTENT_SIGNATURE_SAMPLE_BYTES + b"M" * 10 + b"T" * CONTENT_SIGNATURE_SAMPLE_BYTES
        changed_head = b"X" + base[1:]
        changed_tail = base[:-1] + b"X"
        a = self._write("a.pdf", base)
        b = self._write("b.pdf", changed_head)
        c = self._write("c.pdf", changed_tail)
        sig_a = content_signature(a, a.stat().st_size)
        self.assertNotEqual(sig_a, content_signature(b, b.stat().st_size))
        self.assertNotEqual(sig_a, content_signature(c, c.stat().st_size))

    def test_is_deterministic_across_repeated_calls(self):
        a = self._write("a.pdf", b"stable content")
        self.assertEqual(
            content_signature(a, a.stat().st_size), content_signature(a, a.stat().st_size))


if __name__ == "__main__":
    unittest.main()
