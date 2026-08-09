"""What ingesting one attachment does, checked against what it used to do.

``index_from_zotero.main_async`` is 1,856 lines, 1,389 of them one loop over
attachments, and no test calls it. This is the net that has to exist before that
loop can be broken up, because the way an extract-method fails here is not
gradual: a local stops being updated in the scope that still reads it, and no
test of the extracted piece can see that. Only running the loop can.

Each attachment is ingested into a data plane of its own in a temporary
directory, and what came out is compared with what came out before: the
counters the run printed, which are its own account of which branch it took;
the manifest row; every chunk by identity, block type, zone and structure; and
any artifact status rows. The real index is never opened.

Verified by breaking the loop on purpose. Dropping one ``updated_html += ...``
shows up as a changed summary line; dropping ``block_type`` from a chunk's
metadata shows up as the chunk that lost it. An earlier attempt at the first of
those was not caught -- it removed a counter for a branch this corpus does not
take -- which is the limit worth stating plainly: this net covers the three
extraction routes below and says nothing about the ones it does not walk.

Regenerate with::

    uv run python scripts/build_ingestion_baseline.py            # diff
    uv run python scripts/build_ingestion_baseline.py --write    # adopt

Needs the library, the attachments on disk and the embedding model, so it skips
where those are absent. About half a minute for the three items.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from scripts.build_ingestion_baseline import BASELINE_PATH, CORPUS, ingest, observe


def _zotero_is_reachable() -> bool:
    """Whether Zotero's local API is answering.

    Ingestion enumerates the library over it, and refuses to treat an
    unreachable Zotero as an empty one -- rightly, since that would delete
    everything. So with Zotero closed this check cannot run, and saying so is
    the honest outcome: a red test here would mean "your Zotero is not open",
    which is not a fact about the code. It surfaced as intermittent failure
    until the child's stderr was printed and said so in one line.
    """
    import socket
    from urllib.parse import urlparse

    from src.zotero_source_localapi import local_api_base

    parsed = urlparse(local_api_base())
    try:
        with socket.create_connection(
            (parsed.hostname or "127.0.0.1", parsed.port or 23119), timeout=1.5,
        ):
            return True
    except OSError:
        return False


_needs_zotero = pytest.mark.skipif(
    not _zotero_is_reachable(),
    reason="needs Zotero's local API; ingestion reads the library through it",
)

_needs_baseline = pytest.mark.skipif(
    not BASELINE_PATH.exists(),
    reason=f"{BASELINE_PATH.name} is generated locally; run the builder with --write",
)


def _baseline() -> dict[str, dict]:
    payload = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return {row["item_key"]: row for row in payload["items"]}


@_needs_baseline
def test_the_recorded_corpus_covers_every_route_the_builder_names():
    # A route that quietly leaves the corpus takes its coverage with it, and the
    # remaining entries go on passing. One item was dropped for the opposite
    # reason -- it had no attachment, so the loop ran zero times and its empty
    # result agreed with itself no matter what the loop did.
    assert set(_baseline()) == set(CORPUS)


@_needs_baseline
@_needs_zotero
@pytest.mark.parametrize("item_key", CORPUS)
def test_ingesting_an_attachment_does_what_it_did(item_key: str):
    recorded = _baseline()[item_key]
    with tempfile.TemporaryDirectory(prefix="ingestion-check-") as raw:
        plane = Path(raw)
        run = ingest(item_key, plane)
        observed = observe(item_key, plane)
        # Again into the same plane and without --force-reparse, which is the
        # only way the unchanged-source decision is exercised at all: with the
        # flag set it always answers "index".
        wanted = ("exit_code", "counters", "summary")
        run["second_pass"] = {
            k: v for k, v in ingest(item_key, plane, force=False).items() if k in wanted
        }
        run["quality_pass"] = {
            k: v for k, v in
            ingest(item_key, plane, force=False, check_quality=True).items()
            if k in wanted
        }

    differences = []
    if run.get("exit_code"):
        differences.append("  the run failed:\n" + run.get("stderr_tail", ""))
    for field in ("exit_code", "counters", "summary", "second_pass", "quality_pass"):
        if recorded.get(field) != run.get(field):
            differences.append(
                f"  {field}:\n    was: {recorded.get(field)}\n    now: {run.get(field)}"
            )
    for field in ("manifest", "artifact_status"):
        if recorded.get(field) != observed.get(field):
            differences.append(
                f"  {field}:\n"
                f"    was: {json.dumps(recorded.get(field), ensure_ascii=False)[:400]}\n"
                f"    now: {json.dumps(observed.get(field), ensure_ascii=False)[:400]}"
            )
    if recorded.get("chunks") != observed.get("chunks"):
        was, now = recorded.get("chunks") or [], observed.get("chunks") or []
        differences.append(f"  chunks: {len(was)} -> {len(now)}")
        for before, after in zip(was, now, strict=False):
            if before != after:
                differences.append(
                    f"    {before.get('id')}\n"
                    f"      was: {json.dumps(before, ensure_ascii=False)}\n"
                    f"      now: {json.dumps(after, ensure_ascii=False)}"
                )
                break

    if differences:
        pytest.fail(
            f"ingesting {item_key} no longer does what was recorded. If the "
            "change was intended, read each difference before adopting it with "
            "the builder's --write:\n" + "\n".join(differences),
            pytrace=False,
        )
