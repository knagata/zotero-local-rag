#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin:$HOME/.cargo/bin"

gate_path="${SERVER_DB_GATE_PATH:-data/quality/server_database_gate.json}"
summary_audit_path="${SERVER_SUMMARY_AUDIT_PATH:-data/quality/server_summary_audit.json}"
zotero_audit_path="${SERVER_ZOTERO_AUDIT_PATH:-data/quality/server_zotero_reconciliation.json}"
source_audit_path="${SERVER_SOURCE_AUDIT_PATH:-data/quality/server_source_verification.json}"
collection="${CHROMA_COLLECTION:-zotero_paragraphs_v3}"
manifest="${MANIFEST_PATH:-data/manifest_v3.json}"
lexical_db="${LEXICAL_DB_PATH:-data/lexical_v3.sqlite3}"
chroma_dir="${CHROMA_DIR:-data/chroma}"
pipeline_config="${PIPELINE_CONFIG_PATH:-$chroma_dir/embedder_config_v3.json}"

# This workflow is deliberately V3-only. Never let a missing server flag turn
# its destructive rebuild phase into the legacy reset path.
if [[ "$collection" != "zotero_paragraphs_v3" ]]; then
    echo "[停止] 旧collectionまたは任意collectionは使用できません: $collection"
    exit 2
fi
if [[ "$(basename "$manifest")" != "manifest_v3.json" ]]; then
    echo "[停止] 旧manifestは使用できません: $manifest"
    exit 2
fi
if [[ "$(basename "$lexical_db")" != "lexical_v3.sqlite3" ]]; then
    echo "[停止] 旧FTSは使用できません: $lexical_db"
    exit 2
fi
if [[ "$pipeline_config" != "$chroma_dir/embedder_config_v3.json" ]]; then
    echo "[停止] pipeline configはV3 Chromaディレクトリ内に固定されています。"
    exit 2
fi
export INGEST_STRUCTURED_V3_ENABLE=1
export CHROMA_COLLECTION="$collection"
export MANIFEST_PATH="$manifest"
export LEXICAL_DB_PATH="$lexical_db"
export CHROMA_DIR="$chroma_dir"

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
        # A failed re-audit must never leave a previous passing gate available
        # for phase 3.  The gate is recreated only after every phase-2 check.
        if [[ -e "$gate_path" ]]; then
            echo "[情報] 前回のDB監査gateを無効化します: $gate_path"
            rm -f -- "$gate_path"
        fi
        run_step "Zotero実在庫とmanifestの完全照合" uv run python scripts/verify_zotero_reconciliation.py \
            --manifest "$manifest" --output "$zotero_audit_path"
        run_step "原PDFと索引の照合" uv run python scripts/verify_against_source.py \
            --collection "$collection" --manifest "$manifest" --chroma-dir "$chroma_dir" \
            --output "$source_audit_path"
        run_step "V3 DB完全監査" uv run python scripts/audit_v3_cutover.py \
            --new-only --new-collection "$collection" --manifest "$manifest" \
            --lexical-db "$lexical_db" --pipeline-config "$pipeline_config" \
            --zotero-report "$zotero_audit_path" --source-report "$source_audit_path" \
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
