from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "verify_against_source", ROOT / "scripts" / "verify_against_source.py",
)
MODULE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MODULE)


def _connection():
    connection = MagicMock()
    connection.execute.return_value = []
    return connection


def test_cli_uses_the_explicit_manifest_and_chroma_paths():
    manifest = Path("/tmp/server-manifest.json")
    chroma_dir = Path("/tmp/server-chroma")
    observed: dict[str, Path] = {}

    def collection_rows(_collection: str, *, chroma_dir: Path):
        observed["chroma_dir"] = chroma_dir
        return []

    def manifest_loader(path: Path):
        observed["manifest"] = path
        return {"files": {}}

    with (
        patch.object(MODULE, "_collection_rows", side_effect=collection_rows),
        patch.object(MODULE, "load_manifest", side_effect=manifest_loader),
        patch.object(MODULE, "get_db_connection", return_value=_connection()),
        patch.object(sys, "argv", [
                "verify_against_source.py", "--collection", "zotero_paragraphs_v3",
            "--manifest", str(manifest), "--chroma-dir", str(chroma_dir),
        ]),
    ):
        assert MODULE.main() == 0

    assert observed == {"chroma_dir": chroma_dir, "manifest": manifest}


def test_cli_fails_when_a_readable_pdf_page_is_missing_from_the_index():
    with TemporaryDirectory() as directory:
        source = Path(directory) / "source.pdf"
        source.touch()
        with (
            patch.object(MODULE, "_collection_rows", return_value=[]),
            patch.object(MODULE, "load_manifest", return_value={
                "files": {"ATT": {"pdf_path": str(source), "title": "Source"}},
            }),
            patch.object(MODULE, "source_page_chars", return_value={1: 100}),
            patch.object(MODULE, "get_db_connection", return_value=_connection()),
            patch.object(
                sys, "argv",
                ["verify_against_source.py", "--collection", "zotero_paragraphs_v3"],
            ),
        ):
            assert MODULE.main() == 2
