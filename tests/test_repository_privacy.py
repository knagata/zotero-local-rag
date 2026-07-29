from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_EVALUATION_PATHS = {
    "evaluations/ocr_bakeoff_v3/README.md",
    "evaluations/ocr_bakeoff_v3/annotation.example.json",
    "evaluations/ocr_bakeoff_v3/annotation.schema.json",
    "evaluations/ocr_bakeoff_v3/annotations/embedded_text_pair.json",
    "evaluations/ocr_bakeoff_v3/annotations/en_two_column.json",
    "evaluations/ocr_bakeoff_v3/annotations/ja_horizontal.json",
    "evaluations/ocr_bakeoff_v3/annotations/ja_vertical.json",
    "evaluations/ocr_bakeoff_v3/annotations/no_outline.json",
    "evaluations/ocr_bakeoff_v3/annotations/notes_bibliography_book.json",
    "evaluations/ocr_bakeoff_v3/annotations/scanned_pair.json",
    "evaluations/ocr_bakeoff_v3/annotations/tables_math.json",
    "evaluations/ocr_bakeoff_v3/manifest.json",
    "evaluations/ocr_bakeoff_v3/manifest.schema.json",
    "evaluations/ocr_bakeoff_v3/results/comparison.json",
    "evaluations/ocr_bakeoff_v3/results/comparison.md",
    "evaluations/ocr_bakeoff_v3/results/routing_proposal.md",
    "evaluations/v3_character_ratio_outlier_review_20260727.md",
}
LOCAL_PATH_FIXTURES = {"tests/test_compare_ocr_bakeoff_reports.py"}


def _tracked_paths() -> list[str]:
    if not (ROOT / ".git").exists():
        raise unittest.SkipTest("repository privacy checks require a Git checkout")
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
        unexpected = [
            path for path in tracked
            if path.startswith("evaluations/")
            and path not in PUBLIC_EVALUATION_PATHS
        ]
        self.assertEqual(unexpected, [])

    def test_tracked_text_does_not_contain_local_zotero_paths(self):
        leaks: list[str] = []
        # Split generic local markers so this test does not flag its own source.
        home_marker = "/" + "Users/"
        zotero_marker = "Zotero/" + "storage/"
        for relative in _tracked_paths():
            path = ROOT / relative
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            if (
                relative not in LOCAL_PATH_FIXTURES
                and (home_marker in text or zotero_marker in text)
            ):
                leaks.append(relative)
        self.assertEqual(leaks, [])


if __name__ == "__main__":
    unittest.main()
