from __future__ import annotations

import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from src.v3_data_plane import resolve_configured_path

ROOT = Path(__file__).resolve().parents[1]


class ResolveConfiguredPathTests(unittest.TestCase):
    """The canonical rule: expand ~, and resolve relatives against the root.

    Both halves have been re-derived incorrectly by hand before. A copy in
    chunk_store.py missed .expanduser(), so a ~-prefixed CHROMA_DIR became a
    literal "~" subdirectory there and the real home directory everywhere else
    (2026-07-30). The other half bit next: .env stores MANIFEST_PATH and
    LEXICAL_DB_PATH as *relative* paths, and a bare Path() resolves those
    against the process's cwd, so scripts run from outside the project root
    silently addressed files that did not exist (2026-08-04).
    """

    def test_relative_paths_resolve_against_the_project_root_not_cwd(self):
        resolved = resolve_configured_path(ROOT, "data/manifest_v3.json")
        self.assertEqual(resolved, ROOT / "data" / "manifest_v3.json")
        self.assertTrue(resolved.is_absolute())

    def test_tilde_is_expanded(self):
        with tempfile_home() as home:
            resolved = resolve_configured_path(ROOT, "~/chroma-data")
            self.assertEqual(resolved, Path(home) / "chroma-data")
            self.assertNotIn("~", str(resolved))

    def test_absolute_paths_are_left_alone(self):
        self.assertEqual(
            resolve_configured_path(ROOT, "/var/data/chroma"), Path("/var/data/chroma"),
        )


class ModuleLevelPathsIgnoreCwdTests(unittest.TestCase):
    """Modules must resolve the same paths from any working directory.

    .env ships relative values, so this is not hypothetical: before the fix,
    importing verify_zotero_reconciliation from /tmp pointed MANIFEST_PATH at
    /tmp/data/manifest_v3.json.
    """

    def _resolved_from(self, cwd: Path) -> dict[str, str]:
        script = (
            "import sys, json;"
            f"sys.path[:0] = [{str(ROOT)!r}, {str(ROOT / 'scripts')!r}];"
            "import importlib;"
            "from src.chunk_store import DEFAULT_CHROMA_DIR;"
            "m = importlib.import_module('verify_zotero_reconciliation');"
            "v = importlib.import_module('verify_against_source');"
            "print(json.dumps({"
            "  'manifest': str(m.MANIFEST_PATH),"
            "  'source_manifest': str(v.MANIFEST),"
            "  'chroma': str(DEFAULT_CHROMA_DIR),"
            "}))"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=cwd,
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        import json
        return json.loads(result.stdout.strip().splitlines()[-1])

    def test_paths_are_identical_from_the_root_and_from_elsewhere(self):
        from_root = self._resolved_from(ROOT)
        from_tmp = self._resolved_from(Path("/tmp"))
        self.assertEqual(from_root, from_tmp)
        for key, value in from_root.items():
            self.assertTrue(
                value.startswith(str(ROOT)), f"{key} escaped the project root: {value}",
            )


class EnvVarNameTests(unittest.TestCase):
    """Path env vars must be spelled the way .env.example spells them.

    scripts/run_reocr_queue.py read MANIFEST_V3_PATH and LEXICAL_V3_DB_PATH --
    names that appeared nowhere else in the repo or in any .env -- so it always
    fell through to its hardcoded defaults and silently ignored the configured
    paths (2026-08-04).
    """

    CANONICAL = {"CHROMA_DIR", "MANIFEST_PATH", "LEXICAL_DB_PATH", "RELATIONS_DB_PATH"}
    #: Misspellings that have actually shipped, kept as an explicit denylist so
    #: a re-introduction fails loudly rather than degrading quietly.
    RETIRED = {"MANIFEST_V3_PATH", "LEXICAL_V3_DB_PATH", "CHROMA_DIR_V3"}

    #: Matches an actual environment read, not a mention. The names appear in
    #: prose in the comments that explain the bug, and flagging those would
    #: make the fix's own explanation un-writable.
    ENV_READ = re.compile(
        r'os\.(?:environ\.get|getenv)\(\s*["\'](\w+)["\']|os\.environ\[\s*["\'](\w+)["\']'
    )

    def test_no_module_reads_a_retired_path_variable_name(self):
        offenders = []
        for directory in ("src", "scripts", "citation_graph"):
            for path in (ROOT / directory).rglob("*.py"):
                text = path.read_text(encoding="utf-8")
                for match in self.ENV_READ.finditer(text):
                    name = match.group(1) or match.group(2)
                    if name in self.RETIRED:
                        line = text.count("\n", 0, match.start()) + 1
                        offenders.append(f"{path.relative_to(ROOT)}:{line} reads {name}")
        self.assertEqual(offenders, [], "\n".join(offenders))

    def test_the_detector_would_catch_a_reintroduction(self):
        # Without this, a regex that silently matches nothing would make the
        # check above pass forever.
        found = {
            m.group(1) or m.group(2)
            for m in self.ENV_READ.finditer(
                'os.environ.get("MANIFEST_V3_PATH", x)\n'
                'os.environ["LEXICAL_V3_DB_PATH"]\n'
                'os.getenv("CHROMA_DIR")\n'
                '# a comment mentioning MANIFEST_V3_PATH must not match\n'
            )
        }
        self.assertEqual(found, {"MANIFEST_V3_PATH", "LEXICAL_V3_DB_PATH", "CHROMA_DIR"})


class _tempfile_home:
    def __enter__(self):
        import tempfile
        self._dir = tempfile.TemporaryDirectory()
        self._patch = patch.dict(os.environ, {"HOME": self._dir.name})
        self._patch.start()
        return self._dir.name

    def __exit__(self, *exc):
        self._patch.stop()
        self._dir.cleanup()
        return False


def tempfile_home():
    return _tempfile_home()


if __name__ == "__main__":
    unittest.main()
