#!/usr/bin/env python3
"""Request and persist Zotero 10 Local API write authorization."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from env_utils import load_dotenv_native  # noqa: E402
from zotero_import import ZoteroLocalWriter  # noqa: E402


async def main() -> None:
    load_dotenv_native(ROOT)
    print("Zoteroの確認画面で「常に許可」を選んでください。資料は追加されません。", flush=True)
    result = await ZoteroLocalWriter(timeout=300).authorize()
    if result["web_api"]:
        print("Zotero Web API fallback is configured.")
    elif result["remembered"]:
        print("Zotero Local API write authorization was saved.")
    else:
        print("One-time authorization was granted but not saved; run again and choose Always Allow.")


if __name__ == "__main__":
    asyncio.run(main())
