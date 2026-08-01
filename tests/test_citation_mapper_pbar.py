from __future__ import annotations

import io
import unittest

from src.citation_mapper import _pbar


class _FakeTTY(io.StringIO):
    def isatty(self) -> bool:
        return True


class _FakeFile(io.StringIO):
    def isatty(self) -> bool:
        return False


class PbarTests(unittest.TestCase):
    """A file (log capture) must not receive the same \\r-joined stream a
    real terminal does -- that previously turned a 100+ item enumeration
    into one sprawling line and pushed some citation-network logs past
    10MB (2026-08-02)."""

    def test_tty_output_uses_carriage_return_in_place_updates(self):
        f = _FakeTTY()
        for i in range(1, 11):
            _pbar(i, 10, "citing", file=f)
        out = f.getvalue()
        self.assertEqual(out.count("\n"), 1)  # only the final frame ends the line
        self.assertEqual(out.count("\r"), 9)
        self.assertIn("[", out)  # the bar itself only renders for a tty

    def test_non_tty_output_is_a_handful_of_real_lines(self):
        f = _FakeFile()
        for i in range(1, 107):
            _pbar(i, 106, "epub ref", file=f)
        out = f.getvalue()
        lines = out.splitlines()
        # Every emitted line is a complete, independent line -- none of them
        # rely on a later \r to become readable.
        self.assertNotIn("\r", out)
        self.assertLessEqual(len(lines), 12)
        self.assertTrue(lines[-1].startswith("        epub ref: 106/106"))

    def test_non_tty_output_always_includes_the_first_frame(self):
        # Matches write_batch_jsonl's force-printed count==1 in
        # src/mistral_ocr_batch.py: a reader tailing the log should see
        # confirmation that an enumeration started, not just its milestones.
        f = _FakeFile()
        for i in range(1, 107):
            _pbar(i, 106, "epub ref", file=f)
        out = f.getvalue()
        self.assertIn("epub ref: 1/106", out)

    def test_non_tty_output_always_includes_the_final_frame(self):
        # The 100% milestone must survive the throttle even when total isn't
        # a clean multiple of the step size.
        f = _FakeFile()
        for i in range(1, 8):
            _pbar(i, 7, "citing", file=f)
        out = f.getvalue()
        self.assertIn("7/7 (100%)", out)

    def test_non_tty_output_is_dramatically_smaller_than_the_old_behavior(self):
        f = _FakeFile()
        for i in range(1, 501):
            _pbar(i, 500, "epub ref", file=f)
        out = f.getvalue()
        # Old behavior wrote one growing frame per call with no real newline
        # between them; this must stay small and line-based regardless of
        # how large the enumeration is.
        self.assertLess(len(out.splitlines()), 15)
        self.assertLess(len(out), 2000)


if __name__ == "__main__":
    unittest.main()
