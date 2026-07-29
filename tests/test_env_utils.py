from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.env_utils import load_dotenv_native


class EnvUtilsTests(unittest.TestCase):
    def test_policy_file_adds_values_without_overriding_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text("EXISTING=file\n", encoding="utf-8")
            (root / ".env.policy").write_text(
                "EXISTING=policy\nEXTRACT_EXCLUDE_TAGS=no-cloud\n", encoding="utf-8"
            )
            with patch.dict(os.environ, {"EXISTING": "process"}, clear=True):
                load_dotenv_native(root)
                self.assertEqual(os.environ["EXISTING"], "process")
                self.assertEqual(os.environ["EXTRACT_EXCLUDE_TAGS"], "no-cloud")

    def test_inline_comments_are_removed_but_hashes_inside_values_are_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text(
                "COMMENTED=value # explanation\n"
                "HASHED=abc#123\n"
                'QUOTED="value # retained" # explanation\n',
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                load_dotenv_native(root)
                self.assertEqual(os.environ["COMMENTED"], "value")
                self.assertEqual(os.environ["HASHED"], "abc#123")
                self.assertEqual(os.environ["QUOTED"], "value # retained")


if __name__ == "__main__":
    unittest.main()
