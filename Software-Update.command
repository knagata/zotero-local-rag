#!/bin/bash

cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"

export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin:$HOME/.cargo/bin"

REPO_ZIP="https://github.com/knagata/zotero-local-rag/archive/refs/heads/main.zip"
TMP_ZIP="/tmp/zotero-local-rag-update.zip"
TMP_DIR="/tmp/zotero-local-rag-update"
EXTRACTED="$TMP_DIR/zotero-local-rag-main"

echo "========================================"
echo "   Zotero Local RAG - ソフトウェア更新"
echo "========================================"
echo ""
echo "GitHubから最新版をダウンロードして更新します。"
echo ".envとdata/（索引・モデル）は保持されます。"
echo ""
read -p "続行しますか？ [Y/n]: " ans
ans=$(echo "$ans" | tr '[:upper:]' '[:lower:]')
if [ "$ans" = "n" ]; then
    echo "更新をキャンセルしました。"
    read -p "Enterを押すと終了します..."
    exit 0
fi

echo ""
echo "[1/4] 最新版をダウンロードしています..."
rm -f "$TMP_ZIP"
if ! curl -L --progress-bar -o "$TMP_ZIP" "$REPO_ZIP"; then
    echo ""
    echo "[!] ダウンロードに失敗しました。インターネット接続を確認してください。"
    read -p "Enterを押すと終了します..."
    exit 1
fi

echo ""
echo "[2/4] 展開しています..."
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"
if ! unzip -q "$TMP_ZIP" -d "$TMP_DIR"; then
    echo "[!] 展開に失敗しました。"
    read -p "Enterを押すと終了します..."
    exit 1
fi

if [ ! -d "$EXTRACTED" ]; then
    echo "[!] 展開後のフォルダが見つかりません: $EXTRACTED"
    read -p "Enterを押すと終了します..."
    exit 1
fi

echo "[3/4] ファイルを更新しています（.envとdata/は保持）..."
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

echo "[4/4] 一時ファイルを削除しています..."
rm -rf "$TMP_DIR" "$TMP_ZIP"

echo ""
echo "========================================"
echo "   更新が完了しました"
echo "========================================"
echo ""
echo "変更を反映するためClaude Desktopを再起動してください。"
echo ""
read -p "続けてセットアップウィザードを実行しますか？ [y/N]: " run_idx
run_idx=$(echo "$run_idx" | tr '[:upper:]' '[:lower:]')
if [ "$run_idx" = "y" ]; then
    echo ""
    uv run scripts/setup_wizard.py
fi

echo ""
read -p "既存索引の品質チェックを実行しますか？ [y/N]: " run_chk
run_chk=$(echo "$run_chk" | tr '[:upper:]' '[:lower:]')
if [ "$run_chk" = "y" ]; then
    echo ""
    uv run src/index_from_zotero.py --check-quality --progress
fi

echo ""
echo "このターミナルウィンドウは閉じて構いません。"
