"""A whole data plane in a temporary directory, in this process.

The stores a run writes to are reached three different ways: ``v3_data_plane``
reads the environment on every call, ``chunk_store`` takes an explicit path,
and ``index_from_zotero`` resolves its own set once per run. A test that
redirects only one of them gets the dangerous shape -- most of the run in the
temporary directory and one store in the real library -- so this redirects all
of them together, and ``test_ingest_paths_seam`` checks that it still does.

Environment first, then the run's paths derived from it: that ordering is what
makes one helper cover both mechanisms.
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


@contextmanager
def temporary_data_plane(root: Path):
    """Point Chroma, the lexical index, the manifest and the caches at ``root``.

    Yields the resolved ``IngestPaths``. The names are the ones the plane
    insists on (``manifest_v3.json``, ``lexical_v3.sqlite3``) because
    ``v3_data_plane`` refuses anything else -- that guard is part of what is
    being exercised, not something to work around.
    """
    import index_from_zotero

    chroma = root / "chroma"
    chroma.mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)
    environment = {
        "INGEST_STRUCTURED_V3_ENABLE": "1",
        "CHROMA_COLLECTION": "zotero_paragraphs_v3",
        "CHROMA_DIR": str(chroma),
        "LEXICAL_DB_PATH": str(root / "lexical_v3.sqlite3"),
        "MANIFEST_PATH": str(root / "manifest_v3.json"),
        "PIPELINE_CONFIG_PATH": str(chroma / "embedder_config_v3.json"),
        "PDF_CACHE_DIR": str(root / "pdf_cache"),
    }
    with patch.dict(os.environ, environment):
        resolved = index_from_zotero.IngestPaths.from_environment(root)
        with index_from_zotero.use_paths(resolved) as plane:
            yield plane
