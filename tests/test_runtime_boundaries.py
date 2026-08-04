from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.runtime_config import bounded_env_int


class RuntimeConfigTests(unittest.TestCase):
    def test_invalid_and_out_of_range_values_are_bounded(self):
        with patch.dict(os.environ, {"LIMIT": "invalid"}):
            self.assertEqual(bounded_env_int("LIMIT", 20, minimum=1, maximum=100), 20)
        with patch.dict(os.environ, {"LIMIT": "999999"}):
            self.assertEqual(bounded_env_int("LIMIT", 20, minimum=1, maximum=100), 100)
        with patch.dict(os.environ, {"LIMIT": "-5"}):
            self.assertEqual(bounded_env_int("LIMIT", 20, minimum=1, maximum=100), 1)


class DebugLogBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import through the supported direct-module compatibility path.
        import sys
        source = str(Path(__file__).resolve().parents[1] / "src")
        if source not in sys.path:
            sys.path.insert(0, source)
        global rag_mcp_server
        import rag_mcp_server

    def test_debug_logs_are_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MCP_DEBUG_LOGS_ENABLE", None)
            result = rag_mcp_server.get_debug_logs(100)
        self.assertIn("disabled", result["error"])
        self.assertNotIn("log_path", result)

    def test_enabled_debug_logs_are_capped_and_do_not_expose_the_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            path.write_text("".join(f"line-{i}\n" for i in range(600)), encoding="utf-8")
            with patch.object(rag_mcp_server, "_LOG_PATH", str(path)), \
                    patch.dict(os.environ, {"MCP_DEBUG_LOGS_ENABLE": "1"}):
                result = rag_mcp_server.get_debug_logs(100000)
        self.assertEqual(result["returned_lines"], 500)
        self.assertNotIn("log_path", result)


if __name__ == "__main__":
    unittest.main()
