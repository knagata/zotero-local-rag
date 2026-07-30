#!/bin/bash

# Change the working directory to the folder containing this script
cd "$(dirname "$0")"

# Ensure common paths are included so `uv` can be found when double-clicking from GUI
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin:$HOME/.cargo/bin"

echo "Zoteroセットアップウィザードを起動します..."

# Off by default -- an initial build can run for a long time and is exactly
# the kind of work worth being able to debug after the fact (2026-07-30).
save_log=0
read -r -p "ログを保存しますか？ [y/N]: " log_answer
case "$log_answer" in
    y|Y|yes|Yes|YES) save_log=1 ;;
esac
if [[ "$save_log" == "1" ]]; then
    log_dir="${MAINTENANCE_LOG_DIR:-data/logs}"
    mkdir -p "$log_dir"
    log_file="$log_dir/setup_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "$log_file") 2>&1
    echo "[情報] ログを保存します: $log_file"
    echo ""
fi

# Automatically download dependencies and run the wizard via `uv`.  The
# wizard itself offers to build and audit the DB once configuration is
# saved, so a failure here must not be followed by a misleading "close
# the window" message.
uv run scripts/setup_wizard.py
status=$?

echo ""
if [[ "$status" -ne 0 ]]; then
    echo "セットアップウィザードがエラー終了しました（終了コード: $status）。"
    echo "上記のメッセージを確認し、対応してから再実行してください。"
else
    echo "このターミナルウィンドウは閉じて構いません。"
fi
exit "$status"
