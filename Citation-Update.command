#!/bin/bash

# Change the working directory to the folder containing this script
cd "$(dirname "$0")"

# Ensure common paths are included so `uv` can be found when double-clicking from GUI
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin:$HOME/.cargo/bin"

echo "========================================"
echo "   Zotero Local RAG - Citation Update"
echo "========================================"
echo ""
echo "Semantic Scholar API: rate-limited across all processes"
echo "Help: CITATION_UPDATE_GUIDE.md (メニューの選び方・エラー回復の詳細)"
echo ""

echo "How would you like to process citations?"
echo "  1) Update a specific item by ID  (skips if already mapped)"
echo "  2) Force re-update a specific item by ID  (always re-processes)"
echo "  3) Update ALL items  (新規・エラー分のみ処理 / 通常はこれ)"
echo "  4) Force re-update ALL items  (re-processes everything)"
echo "  5) Resume skipped EPUB refs  (バジェットを増やして skipped を再解決)"
echo ""
read -p "Enter your choice (1-5): " choice

echo ""
case "$choice" in
    1)
        read -p "Enter the Item ID: " item_id
        if [ -n "$item_id" ]; then
            echo "Starting update for item: $item_id"
            uv run src/update_citations.py --item "$item_id"
        else
            echo "No ID provided. Exiting."
        fi
        ;;
    2)
        read -p "Enter the Item ID: " item_id
        if [ -n "$item_id" ]; then
            echo "Force-updating item: $item_id"
            uv run src/update_citations.py --item "$item_id" --force
        else
            echo "No ID provided. Exiting."
        fi
        ;;
    3)
        echo "Starting bulk update. (Already-mapped items will be skipped)"
        uv run src/update_citations.py --all
        ;;
    4)
        echo "Starting force bulk update. (All items will be re-processed)"
        echo "Warning: This may take a very long time for large libraries."
        read -p "Are you sure? (y/N): " confirm
        if [ "$confirm" == "y" ] || [ "$confirm" == "Y" ]; then
            uv run src/update_citations.py --all --force
        else
            echo "Cancelled."
        fi
        ;;
    5)
        read -p "S2 lookup budget per item (default 200): " budget
        budget="${budget:-200}"
        echo "Resolving skipped EPUB refs with budget=${budget}..."
        uv run src/update_citations.py --resume-skipped --epub-budget "$budget"
        ;;
    *)
        echo "Invalid choice. Exiting."
        ;;
esac

echo ""
uv run scripts/db_summary.py

echo "Done. Press any key to exit..."
read -n 1 -s
