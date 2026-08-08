#!/usr/bin/env python3
"""Compare Zotero's own attachment listing against the manifest. Read-only.

Every existing check compares the manifest, Chroma, and the FTS index against
*each other*. None of them compares against Zotero, the one thing that
actually decides what should exist. 75QYJJYK -- a linked_url attachment whose
failure to resolve was silently swallowed -- was the single Zotero-eligible
attachment missing from the manifest, and nothing but a manual audit noticed
(2026-07-28). This is that check, made runnable on demand instead of by hand.

Only reports; never writes to the manifest, Chroma, or Zotero. Deliberate
removal of a truly deleted item is scripts/purge_orphans.py's job, which
confirms each candidate against Zotero individually before deleting anything.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_utils import load_dotenv_native  # noqa: E402

load_dotenv_native(ROOT)

from src.manifest import load_manifest  # noqa: E402
from src.v3_data_plane import manifest_path  # noqa: E402
from src.zotero_source_localapi import (  # noqa: E402
    ZoteroLocalAPI, classify_attachment_source_type,
)

# .env の MANIFEST_PATH は相対パス。素の Path() だとCWD基準になり、
# プロジェクトルート以外から起動すると別のファイルを見にいく（2026-08-04）。
MANIFEST_PATH = manifest_path(ROOT)


async def eligible_zotero_attachments(api: ZoteroLocalAPI) -> dict[str, dict]:
    """Every Zotero attachment of a type this project indexes, keyed by attachment key.

    Uses the raw listing (list_pdf_attachments), not iter_normalized_attachments:
    the latter already drops attachments whose local file could not be
    resolved, which is exactly the population a reconciliation check needs to
    see, not the population it should take for granted.
    """
    # A partial inventory can agree with a partial manifest.  This script is
    # used as a rebuild gate, so accepting an unproven listing would defeat
    # the only comparison with Zotero itself.
    raw_attachments = await api.list_pdf_attachments(require_complete=True)
    eligible: dict[str, dict] = {}
    for raw in raw_attachments:
        key, data = ZoteroLocalAPI._unwrap_item(raw)
        if data.get("itemType") != "attachment":
            continue
        source_type = classify_attachment_source_type(data.get("contentType"), data.get("filename"))
        if source_type is None:
            continue
        eligible[key] = {
            "source_type": source_type,
            "linkMode": data.get("linkMode"),
            "contentType": data.get("contentType"),
            "filename": data.get("filename"),
            "parentItem": data.get("parentItem"),
        }
    return eligible


async def main_async(args: argparse.Namespace, api: ZoteroLocalAPI | None = None) -> int:
    api = api or ZoteroLocalAPI()
    eligible = await eligible_zotero_attachments(api)
    manifest_path = Path(getattr(args, "manifest", MANIFEST_PATH))
    manifest_keys = set((load_manifest(manifest_path).get("files") or {}).keys())

    # ``linked_url`` is intentionally not indexed: Zotero stores no local
    # file for it and the Local API's /file endpoint cannot supply one.  Keep
    # it visible in the report, but do not make an otherwise complete local
    # corpus impossible to approve.  Any other eligible attachment is a
    # required local source and its absence must fail closed.
    unindexable = {
        key: value for key, value in eligible.items()
        if str(value.get("linkMode") or "").casefold() == "linked_url"
    }
    required = {key: value for key, value in eligible.items() if key not in unindexable}
    missing = sorted(set(required) - manifest_keys)
    report = {
        "manifest_path": str(manifest_path.resolve()),
        "zotero_eligible_attachments": len(eligible),
        "required_attachments": len(required),
        "unindexable_linked_url_attachments": [
            {"attachment_key": key, "reason_code": "linked_url_no_local_file", **unindexable[key]}
            for key in sorted(unindexable)
        ],
        "manifest_attachments": len(manifest_keys),
        "missing_required_from_manifest": [
            {"attachment_key": key, **required[key]} for key in missing
        ],
    }
    report["passed"] = not missing
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(
        {"passed": report["passed"], "zotero_eligible_attachments": len(eligible),
         "required_attachments": len(required), "manifest_attachments": len(manifest_keys),
         "missing_required_count": len(missing), "unindexable_linked_url_count": len(unindexable)},
        ensure_ascii=False,
    ))
    return 0 if report["passed"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=MANIFEST_PATH,
        help="V3 manifest to reconcile (defaults to MANIFEST_PATH or data/manifest_v3.json).",
    )
    parser.add_argument("--output", type=Path, help="Write the full report as JSON.")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
