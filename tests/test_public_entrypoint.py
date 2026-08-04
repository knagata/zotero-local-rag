from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PublicEntrypointTests(unittest.TestCase):
    def test_entrypoint_adapter_import_has_no_runtime_side_effects(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import src.cli; "
                "assert 'rag_mcp_server' not in sys.modules; "
                "assert 'chromadb' not in sys.modules",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
