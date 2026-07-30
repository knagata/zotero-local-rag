from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))  # index_from_zotero imports its siblings flat

from src.index_from_zotero import _delete_by_attachment_keys


class FailingCollection:
    def delete(self, *, where):
        raise RuntimeError("transient Chroma delete failure")


class DeleteByAttachmentKeysTests(unittest.TestCase):
    def test_strict_true_propagates_a_chroma_delete_failure(self):
        # 2026-07-30 regression: the stale-attachment cleanup loop in main()
        # relied on strict=False's fail-open swallowing this exception, so a
        # transient delete failure was silently counted as a successful
        # deletion and the manifest entry was dropped anyway -- orphaning the
        # rows with no manifest entry left to retry them from. strict=True is
        # what that call site now uses specifically so this failure surfaces.
        with patch("src.index_from_zotero.delete_lexical_attachments"):
            with self.assertRaises(RuntimeError):
                _delete_by_attachment_keys(FailingCollection(), ["ATT"], strict=True)

    def test_strict_false_still_swallows_a_chroma_delete_failure(self):
        # Other callers of this helper intentionally keep the fail-open
        # default; that behavior must not change as a side effect of this fix.
        with patch("src.index_from_zotero.delete_lexical_attachments"):
            _delete_by_attachment_keys(FailingCollection(), ["ATT"], strict=False)


if __name__ == "__main__":
    unittest.main()
