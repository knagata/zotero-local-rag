#!/bin/bash
set -uo pipefail

cd "$(dirname "$0")"
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin:$HOME/.cargo/bin"

echo "========================================"
echo " Zotero Local RAG - Maintenance Update"
echo "========================================"
echo ""
echo "日常更新を次の順番でまとめて実行します。"
echo "Enterを押すと既定の「実行」を選択します。"
echo ""

ask_enabled() {
    local prompt="$1"
    local answer
    read -r -p "$prompt [Y/n]: " answer
    case "$answer" in
        n|N|no|No|NO) return 1 ;;
        *) return 0 ;;
    esac
}

run_library=0
run_summaries=0
run_citations=0
review_relations=0

if ask_enabled "1. ライブラリを差分更新する"; then
    run_library=1
fi
if ask_enabled "2. 要約を差分更新する（ローカル抽出後、DeepSeek APIでAI要約）"; then
    run_summaries=1
fi
if ask_enabled "3. Citation Networkの未処理・エラー分を更新する"; then
    run_citations=1
fi
if ask_enabled "4. 品質報告をAI判定し、曖昧な例だけ確認する"; then
    review_relations=1
fi

echo ""
echo "実行予定:"
if [[ "$run_library" == "1" ]]; then echo "  ✓ ライブラリ更新"; fi
if [[ "$run_summaries" == "1" ]]; then echo "  ✓ 抽出型要約・DeepSeek AI要約更新"; fi
if [[ "$run_citations" == "1" ]]; then echo "  ✓ Citation Network更新"; fi
if [[ "$review_relations" == "1" ]]; then echo "  ✓ 品質報告の自動判定・例外確認"; fi

if [[ "$run_library" == "0" && "$run_summaries" == "0" && "$run_citations" == "0" && "$review_relations" == "0" ]]; then
    echo "  （選択なし）"
    echo ""
    echo "更新を行わず終了します。"
    exit 0
fi

echo ""
if ! ask_enabled "この内容で開始する"; then
    echo "更新をキャンセルしました。"
    exit 0
fi

run_step() {
    local label="$1"
    shift
    echo ""
    echo "========================================================================"
    echo ">> $label"
    echo "========================================================================"
    if "$@"; then
        echo "[完了] $label"
    else
        local code=$?
        echo "[エラー] ${label}（終了コード: ${code}）"
        echo "安全のため後続処理を実行せず終了します。"
        exit "$code"
    fi
}

if [[ "$run_library" == "1" ]]; then
    run_step "ライブラリ差分更新" uv run src/index_from_zotero.py --progress
    run_step "文書構造v2の差分更新" uv run python scripts/rebuild_document_structure.py --all
fi

if [[ "$run_summaries" == "1" ]]; then
    if [[ "$run_library" == "0" ]]; then
        run_step "文書構造v2の差分確認" uv run python scripts/rebuild_document_structure.py --all
    fi
    run_step "ローカル抽出型要約更新" uv run python -m src.build_summaries
    run_step "DeepSeek AI要約更新" uv run python scripts/build_deepseek_summaries.py \
        --output data/quality/maintenance-summary-report.json
    run_step "DeepSeek 構造化事例更新" uv run python scripts/build_deepseek_cases.py \
        --output data/quality/maintenance-case-report.json
fi

if [[ "$run_citations" == "1" ]]; then
    run_step "Citation Network更新" uv run src/update_citations.py --all
fi

if [[ "$review_relations" == "1" ]]; then
    run_step "品質報告のAI自動判定" uv run python scripts/triage_quality_reports.py
    run_step "曖昧な引用関係レポート確認" uv run python scripts/review_relation_reports.py
    run_step "曖昧な要約品質レポート確認" uv run python scripts/review_summary_quality_reports.py
    run_step "曖昧な事例品質レポート確認" uv run python scripts/review_case_quality_reports.py
fi

echo ""
echo "========================================"
echo " 選択した日常更新が完了しました"
echo "========================================"
