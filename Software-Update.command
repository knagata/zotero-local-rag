#!/bin/bash

cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"

export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin:$HOME/.cargo/bin"

REPO_ZIP="https://github.com/knagata/zotero-local-rag/archive/refs/heads/main.zip"
TMP_ZIP="/tmp/zotero-local-rag-update.zip"
TMP_DIR="/tmp/zotero-local-rag-update"
EXTRACTED="$TMP_DIR/zotero-local-rag-main"

echo "========================================"
echo "   Zotero Local RAG - Updater"
echo "========================================"
echo ""
echo "Downloading and updating to the latest version from GitHub."
echo ".env and data/ (indexes/models) will be preserved."
echo ""
read -p "Do you want to continue? [Y/n]: " ans
ans=$(echo "$ans" | tr '[:upper:]' '[:lower:]')
if [ "$ans" = "n" ]; then
    echo "Update cancelled."
    read -p "Press Enter to exit..."
    exit 0
fi

echo ""
echo "[1/4] Downloading latest version..."
rm -f "$TMP_ZIP"
if ! curl -L --progress-bar -o "$TMP_ZIP" "$REPO_ZIP"; then
    echo ""
    echo "[!] Download failed. Please check your internet connection."
    read -p "Press Enter to exit..."
    exit 1
fi

echo ""
echo "[2/4] Extracting..."
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"
if ! unzip -q "$TMP_ZIP" -d "$TMP_DIR"; then
    echo "[!] Extraction failed."
    read -p "Press Enter to exit..."
    exit 1
fi

if [ ! -d "$EXTRACTED" ]; then
    echo "[!] Extracted folder not found: $EXTRACTED"
    read -p "Press Enter to exit..."
    exit 1
fi

echo "[3/4] Updating files (.env and data/ are preserved)..."
# rsync: preserve .env, data/, .venv/, .claude/ (user data)
rsync -a \
    --exclude='.env' \
    --exclude='data/' \
    --exclude='.venv/' \
    --exclude='.claude/' \
    --exclude='.git/' \
    "$EXTRACTED/" "$SCRIPT_DIR/"

# Ensure update scripts are executable
chmod +x "$SCRIPT_DIR/Software-Update.command"
chmod +x "$SCRIPT_DIR/Setup.command"
chmod +x "$SCRIPT_DIR/Maintenance-Widget.command"

echo "[4/4] Cleaning up temporary files..."
rm -rf "$TMP_DIR" "$TMP_ZIP"

echo ""
echo "========================================"
echo "   Update Complete!"
echo "========================================"
echo ""
echo "Please restart Claude Desktop to apply the changes."
echo ""
read -p "Do you want to run the setup wizard now? [y/N]: " run_idx
run_idx=$(echo "$run_idx" | tr '[:upper:]' '[:lower:]')
if [ "$run_idx" = "y" ]; then
    echo ""
    uv run scripts/setup_wizard.py
fi

echo ""
read -p "Do you want to run a quality check on the existing index? [y/N]: " run_chk
run_chk=$(echo "$run_chk" | tr '[:upper:]' '[:lower:]')
if [ "$run_chk" = "y" ]; then
    echo ""
    uv run src/index_from_zotero.py --check-quality --progress
fi

echo ""
echo "You can safely close this terminal window."
