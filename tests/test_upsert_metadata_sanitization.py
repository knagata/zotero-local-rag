from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import index_from_zotero as module  # noqa: E402


class UpsertMetadataSanitizationTests(unittest.TestCase):
    """Chroma rejects an empty-list metadata value outright. Some PyMuPDF
    layout fallback paths legitimately emit e.g. source_block_indices=[] for a
    standalone chunk that never gets merged with a neighbor -- a pre-existing
    latent bug surfaced by scanned-page patch batches (E2c, dev-notes/current/77).
    `_upsert_in_subbatches` must strip such values before calling col.upsert().
    """

    def test_empty_list_metadata_values_are_stripped_before_upsert(self):
        col = MagicMock()
        col.count.return_value = 1
        metas = [
            {"page": 1, "source_block_indices": []},
            {"page": 2, "source_block_indices": [3, 4], "tags": []},
        ]
        with patch.object(module, "upsert_lexical_chunks"), \
                patch.object(module, "relieve_memory_pressure"):
            module._upsert_in_subbatches(
                col, ["a", "b"], ["doc a", "doc b"], metas,
                subbatch_size=10, show_progress=False, label="test",
            )
        col.upsert.assert_called_once()
        passed_metas = col.upsert.call_args.kwargs["metadatas"]
        self.assertNotIn("source_block_indices", passed_metas[0])
        self.assertEqual(passed_metas[1]["source_block_indices"], [3, 4])
        self.assertNotIn("tags", passed_metas[1])

    def test_metadata_without_empty_lists_is_unchanged(self):
        col = MagicMock()
        col.count.return_value = 1
        metas = [{"page": 1, "chapter": "Intro"}]
        with patch.object(module, "upsert_lexical_chunks"), \
                patch.object(module, "relieve_memory_pressure"):
            module._upsert_in_subbatches(
                col, ["a"], ["doc a"], metas,
                subbatch_size=10, show_progress=False, label="test",
            )
        passed_metas = col.upsert.call_args.kwargs["metadatas"]
        self.assertEqual(passed_metas[0], {"page": 1, "chapter": "Intro"})


if __name__ == "__main__":
    unittest.main()
