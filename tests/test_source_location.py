import pytest

from src.manifest import content_signature
from src.source_location import resolve_source_path


def fixture_source(tmp_path):
    folder = tmp_path / "ABCD1234"
    folder.mkdir()
    path = folder / "new.pdf"
    path.write_bytes(b"source bytes")
    entry = {"pdf_path": str(folder / "old.pdf"), "size": path.stat().st_size,
             "content_signature": content_signature(path, path.stat().st_size)}
    return path, entry


def test_rename_is_resolved_within_attachment(tmp_path):
    path, entry = fixture_source(tmp_path)
    assert resolve_source_path("ABCD1234", entry, storage_root=tmp_path) == path
    assert entry["pdf_path"].endswith("old.pdf")  # audit remains read-only
    entry["pdf_path"] = str(path)
    assert resolve_source_path("ABCD1234", entry) == path


@pytest.mark.parametrize("failure", ["changed", "ambiguous", "unsigned", "symlink", "invalid"])
def test_unsafe_relocation_fails(tmp_path, failure):
    path, entry = fixture_source(tmp_path)
    if failure == "changed":
        path.write_bytes(b"changed byte")
    elif failure == "ambiguous":
        (path.parent / "duplicate.pdf").write_bytes(path.read_bytes())
    elif failure == "unsigned":
        entry.pop("content_signature")
    elif failure == "symlink":
        target = tmp_path / "external.pdf"
        path.rename(target)
        path.symlink_to(target)
    key = "../ABCD1234" if failure == "invalid" else "ABCD1234"
    with pytest.raises(FileNotFoundError):
        resolve_source_path(key, entry, storage_root=tmp_path)
