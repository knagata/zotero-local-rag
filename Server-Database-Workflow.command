#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin:$HOME/.cargo/bin"

gate_path="${SERVER_DB_GATE_PATH:-data/quality/server_database_gate.json}"
summary_audit_path="${SERVER_SUMMARY_AUDIT_PATH:-data/quality/server_summary_audit.json}"
collection="${CHROMA_COLLECTION:-zotero_paragraphs_v3}"
manifest="${MANIFEST_PATH:-data/manifest_v3.json}"
lexical_db="${LEXICAL_DB_PATH:-data/lexical_v3.sqlite3}"
pipeline_config="${PIPELINE_CONFIG_PATH:-data/chroma/embedder_config_v3.json}"

# This workflow is deliberately V3-only. Never let a missing server flag turn
# its destructive rebuild phase into the legacy reset path.
export INGEST_STRUCTURED_V3_ENABLE=1
export CHROMA_COLLECTION="$collection"
export MANIFEST_PATH="$manifest"
export LEXICAL_DB_PATH="$lexical_db"

echo "============================================================"
echo " Zotero Local RAG - Server Database Workflow"
echo "============================================================"
echo "各フェーズは別々に実行します。要約はDB監査合格前には開始できません。"
echo ""
echo "  1) DBをゼロから構築（階層要約は実行しない）"
echo "  2) DBを監査し、要約実行用の合格証明を作成"
echo "  3) 階層AI要約を生成・索引化（API料金あり）"
echo "  4) 階層要約と要約索引を監査"
echo ""

phase="${SERVER_WORKFLOW_PHASE:-}"
if [[ -z "$phase" ]]; then
    read -r -p "実行するフェーズ [1-4]: " phase
fi

run_step() {
    local label="$1"
    shift
    echo ""
    echo ">> $label"
    "$@"
    echo "[完了] $label"
}

case "$phase" in
    1)
        echo ""
        echo "[注意] 対象V3 DBを初期化して再構築します。階層要約APIは呼びません。"
        echo "       AI目次推定が有効な場合は、構造復元用のLLM APIを使うことがあります。"
        read -r -p "続行するには REBUILD と入力してください: " confirmation
        if [[ "$confirmation" != "REBUILD" ]]; then
            echo "キャンセルしました。"
            exit 0
        fi
        run_step "V3 DBゼロ再構築" uv run src/index_from_zotero.py --rebuild --progress
        run_step "文書構造の決定的再確認" uv run python scripts/rebuild_document_structure.py --all
        echo ""
        echo "[次] フェーズ2を別に実行してDBを監査してください。"
        ;;
    2)
        run_step "V3 DB完全監査" uv run python scripts/audit_v3_cutover.py \
            --new-only --new-collection "$collection" --manifest "$manifest" \
            --lexical-db "$lexical_db" --pipeline-config "$pipeline_config" \
            --output "$gate_path"
        echo ""
        echo "[合格] 現在のDB世代に結び付いた要約実行gateを作成しました: $gate_path"
        ;;
    3)
        if [[ ! -f "$gate_path" ]]; then
            echo "[停止] DB監査gateがありません。先にフェーズ2を合格させてください: $gate_path"
            exit 2
        fi
        echo ""
        echo "[課金確認] DeepSeek APIで階層要約を生成します。DB変更後の古いgateは自動拒否されます。"
        read -r -p "続行するには SUMMARIZE と入力してください: " confirmation
        if [[ "$confirmation" != "SUMMARIZE" ]]; then
            echo "キャンセルしました。"
            exit 0
        fi
        summary_workers="${SUMMARY_BACKFILL_WORKERS:-20}"
        run_step "階層AI要約生成・要約索引構築" uv run python scripts/build_structure_summaries.py \
            --all --mode llm --workers "$summary_workers" --embed \
            --collection "$collection" --database-gate "$gate_path"
        echo ""
        echo "[次] フェーズ4を別に実行して要約と索引を監査してください。"
        ;;
    4)
        run_step "階層要約・要約索引監査" uv run python scripts/audit_structure_summaries.py \
            --collection "$collection" --output "$summary_audit_path"
        echo ""
        echo "[合格] 仕様確定に進めます: $summary_audit_path"
        ;;
    *)
        echo "1〜4のいずれかを指定してください。"
        exit 2
        ;;
esac
