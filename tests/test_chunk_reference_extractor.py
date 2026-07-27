from __future__ import annotations

import unittest
from unittest.mock import patch

from src import chunk_reference_extractor
from src.chunk_reference_extractor import (
    _split_numbered_entries, _split_reference_lines, extract_references_from_chunks,
)


class SplitReferenceLinesTests(unittest.TestCase):
    def test_newline_separated_entries_are_kept(self):
        text = (
            "Abramson DB (2020) Ancient resilience. Urban Studies 57.\n\n"
            "Aguiar D (2023) Transforming studies. JPS 50."
        )
        self.assertEqual(len(_split_reference_lines(text)), 2)

    def test_flattened_numbered_list_is_split(self):
        text = (
            "1. Smith J (2020) Cities and theory. Journal X 5: 1-20. "
            "2. Jones A (2019) Urban forms. Book Y. "
            "3. Lee B (2021) Notes on method. Press Z."
        )
        entries = _split_reference_lines(text)
        self.assertEqual(len(entries), 3)
        self.assertTrue(entries[1].startswith("2. Jones A"))

    def test_single_reference_with_numbers_is_not_over_split(self):
        text = "Barrat, J. (2013). Our Final Invention, pp. 12-45. See also p. 200. Press."
        self.assertEqual(len(_split_reference_lines(text)), 1)

    def test_non_sequential_numbers_do_not_trigger_split(self):
        # Page/volume numbers that are not an ascending 1,2,3 run must not split.
        self.assertIsNone(_split_numbered_entries("See 45. and also 12. in the volume."))


class ExtractReferencesFromChunksTests(unittest.TestCase):
    def test_reference_and_note_zones_with_deterministic_note_linkage(self):
        chunks = [
            {"id": "IT:body:1", "text": "body prose", "metadata": {"zone": "body"}},
            {"id": "IT:bib:1", "text": "Smith J (2020) A book. Press.",
             "metadata": {"zone": "bibliography"}},
            # A note carries the body citer recorded at ingestion (Part C).
            {"id": "IT:note:1", "text": "See Jones 2019 for discussion of the point.",
             "metadata": {"zone": "footnote", "citing_chunk_id": "IT:body:1"}},
            # An unlinked note anchors to its own chunk.
            {"id": "IT:note:2", "text": "An endnote entry with enough text to keep.",
             "metadata": {"zone": "endnote"}},
        ]
        with patch.object(chunk_reference_extractor, "get_item_chunks", return_value=chunks):
            refs = extract_references_from_chunks("IT")
        by_zone = {r["source_zone"]: r for r in refs}
        self.assertEqual(set(by_zone), {"bibliography", "footnote", "endnote"})
        self.assertEqual(by_zone["bibliography"]["citing_chunk_id"], "IT:bib:1")
        self.assertEqual(by_zone["footnote"]["citing_chunk_id"], "IT:body:1")  # Part C linkage
        self.assertEqual(by_zone["endnote"]["citing_chunk_id"], "IT:note:2")   # unlinked fallback

    def test_body_zone_chunks_are_ignored(self):
        chunks = [{"id": "IT:body:1", "text": "just body text here", "metadata": {"zone": "body"}}]
        with patch.object(chunk_reference_extractor, "get_item_chunks", return_value=chunks):
            self.assertEqual(extract_references_from_chunks("IT"), [])


if __name__ == "__main__":
    unittest.main()
