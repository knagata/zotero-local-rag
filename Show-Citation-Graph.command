#!/bin/bash

# Change the working directory to the folder containing this script
cd "$(dirname "$0")"

# Ensure common paths are included so `uv` can be found when double-clicking from GUI
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin:$HOME/.cargo/bin"

echo "========================================"
echo "   Zotero Local RAG - Citation Graph"
echo "========================================"
echo ""
echo "グラフを構築中です。ブラウザが自動的に開きます..."
echo "終了するには Ctrl+C を押してください。"
echo ""

uv run citation_graph/server.py

echo ""
echo "サーバーを終了しました。"
