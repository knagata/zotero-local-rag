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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_utils import load_dotenv_native  # noqa: E402

load_dotenv_native(ROOT)

from src.manifest import load_manifest  # noqa: E402
from src.zotero_source_localapi import (  # noqa: E402
    ZoteroLocalAPI, classify_attachment_source_type,
)

MANIFEST_PATH = ROOT / "data" / "manifest_v3.json"


async def eligible_zotero_attachments(api: ZoteroLocalAPI) -> dict[str, dict]:
    """Every Zotero attachment of a type this project indexes, keyed by attachment key.

    Uses the raw listing (list_pdf_attachments), not iter_normalized_attachments:
    the latter already drops attachments whose local file could not be
    resolved, which is exactly the population a reconciliation check needs to
    see, not the population it should take for granted.
    """
    raw_attachments = await api.list_pdf_attachments()
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
    manifest_keys = set((load_manifest(MANIFEST_PATH).get("files") or {}).keys())

    missing = sorted(set(eligible) - manifest_keys)
    report = {
        "zotero_eligible_attachments": len(eligible),
        "manifest_attachments": len(manifest_keys),
        "missing_from_manifest": [
            {"attachment_key": key, **eligible[key]} for key in missing
        ],
    }
    report["passed"] = not missing
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(
        {"passed": report["passed"], "zotero_eligible_attachments": len(eligible),
         "manifest_attachments": len(manifest_keys), "missing_count": len(missing)},
        ensure_ascii=False,
    ))
    return 0 if report["passed"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write the full report as JSON.")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
