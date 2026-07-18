from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"\[[^]]*\]\(([^)]+)\)")


class DocumentationTests(unittest.TestCase):
    def test_readme_stays_a_short_entry_point(self):
        lines = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
        self.assertLessEqual(len(lines), 120)
        self.assertIn("## まず使ってみる", lines)
        self.assertIn("## ドキュメント", lines)

    def test_all_local_markdown_links_resolve(self):
        missing: list[str] = []
        for source in [ROOT / "README.md", *(ROOT / "docs").glob("*.md")]:
            text = source.read_text(encoding="utf-8")
            for target in MARKDOWN_LINK.findall(text):
                target = target.split("#", 1)[0]
                if not target or "://" in target or target.startswith("mailto:"):
                    continue
                destination = (source.parent / target).resolve()
                if not destination.exists():
                    missing.append(f"{source.relative_to(ROOT)} -> {target}")
        self.assertEqual(missing, [])

    def test_documentation_is_scoped_to_claude_on_mac(self):
        public_docs = "\n".join(
            path.read_text(encoding="utf-8")
            for path in [ROOT / "README.md", *(ROOT / "docs").glob("*.md")]
        )
        for unsupported in ("Cursor", "Zed", "Windows", ".bat"):
            self.assertNotIn(unsupported, public_docs)

    def test_claude_guide_exists_at_the_server_resource_path(self):
        self.assertTrue((ROOT / "docs" / "claude-guide.md").is_file())


if __name__ == "__main__":
    unittest.main()
