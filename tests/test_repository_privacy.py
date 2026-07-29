from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_OUTPUT_PREFIXES = (
    "evaluations/outline_gold_v1/",
    "evaluations/v3_cutover/",
)
PRIVATE_OUTPUT_NAMES = {
    "evaluations/ocr_bakeoff_v3/sample_suggestions.json",
    "evaluations/ocr_bakeoff_v3/selected_samples.json",
}


def _tracked_paths() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [
        value.decode("utf-8", errors="surrogateescape")
        for value in result.stdout.split(b"\0")
        if value
    ]


class RepositoryPrivacyTests(unittest.TestCase):
    def test_private_evaluation_outputs_are_not_tracked(self):
        tracked = _tracked_paths()
        private = [
            path for path in tracked
            if path in PRIVATE_OUTPUT_NAMES
            or path.startswith(PRIVATE_OUTPUT_PREFIXES)
            or path.startswith("evaluations/source_verification_")
            or path.startswith("evaluations/v3_cutover_audit")
            or path.startswith("evaluations/legacy_retirement_baseline_")
        ]
        self.assertEqual(private, [])

    def test_tracked_text_does_not_contain_local_zotero_paths(self):
        leaks: list[str] = []
        home_marker = "/Users/" + "knag"
        zotero_marker = "Zotero/" + "storage/"
        for relative in _tracked_paths():
            path = ROOT / relative
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            if home_marker in text or zotero_marker in text:
                leaks.append(relative)
        self.assertEqual(leaks, [])


if __name__ == "__main__":
    unittest.main()
