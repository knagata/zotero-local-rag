from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DashboardSecurityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (ROOT / "dashboard.html").read_text(encoding="utf-8")

    def test_task_fields_are_escaped_before_inner_html_rendering(self):
        self.assertIn("${escapeHtml(task.title)}", self.html)
        self.assertIn("${escapeHtml(task.note)}", self.html)
        self.assertIn("${escapeHtml(st.text)}", self.html)
        self.assertNotIn(">${task.title}</div>", self.html)
        self.assertNotIn(">${task.note}</div>", self.html)
        self.assertNotIn(">${st.text}</span>", self.html)

    def test_file_names_are_not_interpolated_into_inline_handlers(self):
        self.assertNotIn('onclick="openFileModal(', self.html)
        self.assertNotIn('onclick="openNewFileModal(', self.html)
        self.assertIn("card.addEventListener('click'", self.html)
        self.assertIn("escapeAttribute(file.name)", self.html)

    def test_dynamic_attribute_values_use_attribute_escaping(self):
        self.assertIn("function escapeAttribute(text)", self.html)
        self.assertNotIn('data-search="${escapeHtml(', self.html)


if __name__ == "__main__":
    unittest.main()
