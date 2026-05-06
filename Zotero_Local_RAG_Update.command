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
echo "最新バージョンをGitHubからダウンロードして更新します。"
echo ".env と data/ (インデックス・モデル) は保持されます。"
echo ""
read -p "続けますか？ [Y/n]: " ans
ans=$(echo "$ans" | tr '[:upper:]' '[:lower:]')
if [ "$ans" = "n" ]; then
    echo "更新をキャンセルしました。"
    read -p "Enterを押して終了..."
    exit 0
fi

echo ""
echo "[1/4] 最新バージョンをダウンロード中..."
rm -f "$TMP_ZIP"
if ! curl -L --progress-bar -o "$TMP_ZIP" "$REPO_ZIP"; then
    echo ""
    echo "[!] ダウンロードに失敗しました。インターネット接続を確認してください。"
    read -p "Enterを押して終了..."
    exit 1
fi

echo ""
echo "[2/4] 展開中..."
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"
if ! unzip -q "$TMP_ZIP" -d "$TMP_DIR"; then
    echo "[!] 展開に失敗しました。"
    read -p "Enterを押して終了..."
    exit 1
fi

if [ ! -d "$EXTRACTED" ]; then
    echo "[!] 展開後のフォルダが見つかりません: $EXTRACTED"
    read -p "Enterを押して終了..."
    exit 1
fi

echo "[3/4] ファイルを更新中（.env と data/ は保持）..."
# rsync: preserve .env, data/, .venv/, .claude/ (user data)
rsync -a \
    --exclude='.env' \
    --exclude='data/' \
    --exclude='.venv/' \
    --exclude='.claude/' \
    --exclude='.git/' \
    "$EXTRACTED/" "$SCRIPT_DIR/"

# Ensure update script itself is executable
chmod +x "$SCRIPT_DIR/Zotero_Local_RAG_Update.command"
chmod +x "$SCRIPT_DIR/Zotero_Local_RAG_Setup.command"

echo "[4/4] 一時ファイルを削除中..."
rm -rf "$TMP_DIR" "$TMP_ZIP"

echo ""
echo "========================================"
echo "   更新完了！"
echo "========================================"
echo ""
echo "Claude Desktopを再起動すると変更が反映されます。"
echo ""
read -p "埋め込みインデクサーを再実行しますか？ [y/N]: " run_idx
run_idx=$(echo "$run_idx" | tr '[:upper:]' '[:lower:]')
if [ "$run_idx" = "y" ]; then
    echo ""
    uv run scripts/setup_wizard.py
fi

echo ""
echo "このウィンドウは閉じて構いません。"
