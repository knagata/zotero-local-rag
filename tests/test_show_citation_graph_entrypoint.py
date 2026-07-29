from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CitationGraphEntrypointTests(unittest.TestCase):
    def test_direct_script_entrypoint_can_import_src(self):
        completed = subprocess.run(
            [sys.executable, "citation_graph/server.py", "--help"],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--no-open", completed.stdout)


if __name__ == "__main__":
    unittest.main()
