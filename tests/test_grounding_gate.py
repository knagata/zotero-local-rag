from __future__ import annotations

import unittest

from scripts.run_grounding_gate import _safe_error


class GroundingGateTests(unittest.TestCase):
    def test_safe_error_does_not_copy_submitted_source(self):
        error = RuntimeError(
            "user\nPRIVATE SOURCE TEXT\nERROR: You've hit your usage limit.\n"
            "try again later"
        )
        summary = _safe_error(error)
        self.assertIn("usage limit", summary)
        self.assertNotIn("PRIVATE SOURCE TEXT", summary)


if __name__ == "__main__":
    unittest.main()
