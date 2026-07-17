from __future__ import annotations

import unittest

from src.reference_text import is_short_form_reference


class ReferenceTextTests(unittest.TestCase):
    def test_detects_references_that_depend_on_previous_entries(self):
        for value in (
            "21 Ibid., p. 46.", "Ibid. (thesis 4), 12.", "Lacan quoted in ibid., 388.",
            "同書、46頁", "同前。", "前掲書、20頁", "Smith, op. cit., p. 4",
        ):
            self.assertTrue(is_short_form_reference(value), value)

    def test_explicit_reference_is_not_short_form(self):
        self.assertFalse(is_short_form_reference(
            "Dork Zabunyan, Scottie ou Irène ?, Les cinémas de Gilles Deleuze, 2011."
        ))
