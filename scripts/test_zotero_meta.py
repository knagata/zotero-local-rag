import asyncio
import os
import sys
from pathlib import Path

# Setup Path
ROOT = Path("/Users/knag/Documents/GitHub/zotero-local-rag")
sys.path.insert(0, str(ROOT / "src"))

from rag_mcp_server import list_recent_items, get_item_details

async def test():
    print("=== Test 1: Listing recent items ===")
    try:
        items = await list_recent_items(limit=10)
        print(f"Fetched {len(items)} recent items.")
        if items:
            for i, it in enumerate(items[:3]):
                print(f"  [{i}] {it.get('key')} - {it.get('title')}")
            
            # Use the first key for Test 2
            test_key = items[0].get('key')
            print(f"\n=== Test 2: Getting details for key {test_key} ===")
            details = await get_item_details(test_key)
            print(f"Item Type: {details.get('data', {}).get('itemType')}")
            print(f"Full Title: {details.get('data', {}).get('title')}")
            print(f"Tags: {details.get('data', {}).get('tags')}")
        else:
            print("No items found in library.")
    except Exception as e:
        print(f"❌ Error: {e}")
        if "ConnectionRefusedError" in str(e) or "ConnectError" in str(e):
            print("💡 Hint: Ensure Zotero desktop app is running.")

if __name__ == "__main__":
    asyncio.run(test())
