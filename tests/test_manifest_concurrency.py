"""Two writers, one manifest.

The manifest is the list of what is indexed. A reader that finds an attachment
missing from it re-indexes the attachment; a reader that finds one present skips
it. So a manifest that loses entries costs work, and a manifest that gains
entries whose chunks were never written costs a permanent gap -- neither of
which announces itself, because the file is still valid JSON afterwards.

Every writer used to build the same ``<name>.json.tmp`` path and rename it into
place. Two of them wrote into one file. Worse than the interleaving is the
lost update above it: load, work for a minute, save. Whoever saves second
publishes a manifest that never contained the other's changes, and nothing
compares the two.

These run real concurrency -- threads for the same process, subprocesses for
different ones -- because a lock is exactly the thing that passes a
single-threaded test while being absent.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from src.manifest import _write, load_manifest, save_manifest, updating  # noqa: E402


def test_concurrent_writers_never_publish_a_partial_manifest(tmp_path):
    manifest_path = tmp_path / "manifest_v3.json"
    save_manifest(manifest_path, {"version": 1, "files": {}, "notes": {}})
    payloads = [
        {"version": 1, "notes": {}, "files": {f"KEY{index:04d}": {"size": index}
                                              for index in range(400)}}
        for _ in range(8)
    ]
    corrupt: list[str] = []

    def write(payload):
        for _ in range(20):
            save_manifest(manifest_path, payload)
            try:
                json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as failure:  # noqa: BLE001 - the finding is the point
                corrupt.append(str(failure))

    threads = [threading.Thread(target=write, args=(payload,)) for payload in payloads]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not corrupt, f"a reader saw a partial manifest: {corrupt[:3]}"
    assert len(load_manifest(manifest_path)["files"]) == 400


def test_a_writer_outside_the_lock_still_cannot_publish_a_partial_file(tmp_path):
    """The guarantee the temporary inode carries, and the lock does not.

    ``flock`` binds only the callers that take it. A script that writes the
    manifest through the plain function -- or a future path that forgets --
    used to share one ``.json.tmp`` with everyone else, so two of them wrote
    into a single file and whichever renamed second published a mixture of the
    two. With an inode per write, an unlocked writer can still lose a race, but
    it cannot publish something that was never a whole manifest.
    """
    manifest_path = tmp_path / "manifest_v3.json"
    save_manifest(manifest_path, {"version": 1, "files": {}, "notes": {}})
    corrupt: list[str] = []

    def write_without_the_lock(size: int):
        payload = {"version": 1, "notes": {},
                   "files": {f"K{i:05d}": {"size": size} for i in range(size)}}
        for _ in range(15):
            _write(manifest_path, payload)
            try:
                json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as failure:  # noqa: BLE001 - the finding is the point
                corrupt.append(str(failure))

    threads = [threading.Thread(target=write_without_the_lock, args=(size,))
               for size in (50, 800, 50, 800)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not corrupt, f"a reader saw a partial manifest: {corrupt[:3]}"


def test_a_write_that_dies_leaves_nothing_a_later_writer_could_inherit(tmp_path):
    """The half of the temporary file the lock cannot cover.

    Interleaving two unlocked writers is a race that a test can only sometimes
    provoke, so this asks the deterministic half instead: what a failed write
    leaves behind. With one shared ``.json.tmp`` it left a partial file under
    the name every other writer was about to use.
    """
    manifest_path = tmp_path / "manifest_v3.json"
    save_manifest(manifest_path, {"version": 1, "files": {"KEEP": {}}, "notes": {}})
    with patch("src.manifest.os.replace", side_effect=RuntimeError("crashed mid-write")):
        with pytest.raises(RuntimeError):
            _write(manifest_path, {"version": 1, "files": {"NEW": {}}, "notes": {}})
    leftovers = sorted(p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp"))
    assert not leftovers, f"a failed write left {leftovers} for the next writer"
    assert set(load_manifest(manifest_path)["files"]) == {"KEEP"}


def test_a_read_modify_write_does_not_lose_a_concurrent_change(tmp_path):
    # The failure save_manifest alone cannot prevent: both cycles load the same
    # manifest, and the second save publishes a version that never had the
    # first's entry in it.
    manifest_path = tmp_path / "manifest_v3.json"
    save_manifest(manifest_path, {"version": 1, "files": {}, "notes": {}})
    started = threading.Barrier(2)

    def add(key: str):
        started.wait()
        with updating(manifest_path) as manifest:
            manifest["files"][key] = {"size": 1}
            # Long enough that an unserialized pair would certainly overlap.
            threading.Event().wait(0.05)

    threads = [threading.Thread(target=add, args=(key,)) for key in ("FIRST", "SECOND")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    files = load_manifest(manifest_path)["files"]
    assert set(files) == {"FIRST", "SECOND"}, (
        f"one writer's entry was lost: {sorted(files)}"
    )


def test_the_body_failing_leaves_the_manifest_untouched(tmp_path):
    manifest_path = tmp_path / "manifest_v3.json"
    save_manifest(manifest_path, {"version": 1, "files": {"KEEP": {}}, "notes": {}})
    try:
        with updating(manifest_path) as manifest:
            manifest["files"].clear()
            raise RuntimeError("the repair failed")
    except RuntimeError:
        pass
    assert set(load_manifest(manifest_path)["files"]) == {"KEEP"}


def test_writers_in_different_processes_serialize(tmp_path):
    # Threads share a lock object; processes share only the file, which is what
    # the maintenance scripts and the indexer actually are.
    manifest_path = tmp_path / "manifest_v3.json"
    save_manifest(manifest_path, {"version": 1, "files": {}, "notes": {}})
    program = (
        "import sys; sys.path.insert(0, %r);"
        "from src.manifest import updating;"
        "from pathlib import Path;"
        "import time;"
        "key = sys.argv[1];"
        "path = Path(sys.argv[2]);"
        "exec('with updating(path) as m:\\n m[\"files\"][key] = {\"size\": 1}\\n time.sleep(0.2)')"
    ) % str(ROOT)
    processes = [
        subprocess.Popen([sys.executable, "-c", program, key, str(manifest_path)])
        for key in ("ONE", "TWO", "THREE")
    ]
    for process in processes:
        assert process.wait(timeout=60) == 0

    assert set(load_manifest(manifest_path)["files"]) == {"ONE", "TWO", "THREE"}
