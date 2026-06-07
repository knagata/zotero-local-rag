#!/bin/bash

# Change the working directory to the folder containing this script
cd "$(dirname "$0")"

# Ensure common paths are included so `uv` can be found when double-clicking from GUI
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin:$HOME/.cargo/bin"

echo "========================================"
echo "   Zotero Local RAG - Citation Update"
echo "========================================"
echo ""
echo "Semantic Scholar API: 1 req/sec (rate-limited across all processes)"
echo ""

echo "How would you like to process citations?"
echo "  1) Update a specific item by ID  (skips if already mapped)"
echo "  2) Force re-update a specific item by ID  (always re-processes)"
echo "  3) Update ALL items  (skips already-mapped items)"
echo "  4) Force re-update ALL items  (re-processes everything)"
echo ""
read -p "Enter your choice (1-4): " choice

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
    *)
        echo "Invalid choice. Exiting."
        ;;
esac

echo ""
echo "Done. Press any key to exit..."
read -n 1 -s
