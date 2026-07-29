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

# The library owner may make local maintenance non-interactive. Paid API steps
# (hierarchical summaries and cloud OCR) remain explicit per-run opt-ins.
maintenance_auto_approve="${MAINTENANCE_AUTO_APPROVE:-1}"
if [[ "$maintenance_auto_approve" == "1" ]]; then
    echo "[情報] MAINTENANCE_AUTO_APPROVE=1: ローカル更新を自動許可します（有料API処理は除外）。"
    echo ""
fi

ask_enabled() {
    local prompt="$1"
    local answer
    if [[ "$maintenance_auto_approve" == "1" ]]; then
        echo "$prompt [自動許可]"
        return 0
    fi
    read -r -p "$prompt [Y/n]: " answer
    case "$answer" in
        n|N|no|No|NO) return 1 ;;
        *) return 0 ;;
    esac
}

ask_disabled() {
    local prompt="$1"
    local answer
    if [[ "$maintenance_auto_approve" == "1" ]]; then
        echo "$prompt [自動実行の対象外]"
        return 1
    fi
    read -r -p "$prompt [y/N]: " answer
    case "$answer" in
        y|Y|yes|Yes|YES) return 0 ;;
        *) return 1 ;;
    esac
}

run_library=0
run_summaries=0
run_citations=0
review_relations=0
run_mistral_batch=0
mistral_state_path="${MISTRAL_BATCH_STATE_PATH:-data/mistral_ocr_batch_state.json}"

if ask_enabled "1. ライブラリを差分更新する"; then
    run_library=1
fi
# Paid summaries are never included by auto-approval. They require an explicit
# per-run opt-in and a server database gate produced after a successful audit.
if [[ "$maintenance_auto_approve" == "1" ]]; then
    echo "2. 要約を差分更新する（DeepSeek API料金あり） [自動実行の対象外]"
elif ask_disabled "2. 要約を差分更新する（DeepSeek API料金あり・DB監査合格後のみ）"; then
    run_summaries=1
fi
if ask_enabled "3. Citation Networkの未処理・エラー分を更新する"; then
    run_citations=1
fi
if ask_enabled "4. 品質報告をAI判定し、曖昧な例だけ確認する"; then
    review_relations=1
fi
if ask_disabled "5. Mistral OCR Batchを送信・回収・品質確認・採用する（クラウド送信を含む）"; then
    run_mistral_batch=1
fi

echo ""
echo "実行予定:"
if [[ "$run_library" == "1" ]]; then echo "  ✓ ライブラリ更新"; fi
if [[ "$run_summaries" == "1" ]]; then echo "  ✓ 文書構造要約・検索索引の更新"; fi
if [[ "$run_citations" == "1" ]]; then echo "  ✓ Citation Network更新"; fi
if [[ "$review_relations" == "1" ]]; then echo "  ✓ 品質報告の自動判定・例外確認"; fi
if [[ "$run_mistral_batch" == "1" ]]; then echo "  ✓ Mistral OCR Batch処理"; fi

if [[ "$run_library" == "0" && "$run_summaries" == "0" && "$run_citations" == "0" && "$review_relations" == "0" && "$run_mistral_batch" == "0" ]]; then
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

mistral_state_value() {
    local key="$1"
    /usr/bin/python3 - "$mistral_state_path" "$key" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(0)
try:
    value = json.loads(path.read_text(encoding="utf-8")).get(sys.argv[2], "")
except (OSError, ValueError):
    value = ""
print(value if value is not None else "")
PY
}

mark_mistral_adoption_applied() {
    /usr/bin/python3 - "$mistral_state_path" <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

path = Path(sys.argv[1])
state = json.loads(path.read_text(encoding="utf-8"))
state["adoption_applied_at"] = datetime.now(timezone.utc).isoformat()
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
tmp.replace(path)
PY
}

run_mistral_batch_step() {
    local phase
    local queue
    local adoptable
    local adopted_at=""
    phase="$(mistral_state_value phase)"

    case "$phase" in
        "")
            run_step "Mistral OCR Batch送信" uv run python scripts/run_mistral_ocr_batch.py --submit --state "$mistral_state_path"
            echo ""
            echo "[案内] Mistral Batchを送信しました。処理完了後にMaintenance-Widget.commandを再度起動し、"
            echo "       この項目を明示的に許可してください。次回は結果を回収し、品質gate合格分だけをV3へ採用します。"
            ;;
        prepared|uploaded)
            run_step "Mistral OCR Batch送信を再開" uv run python scripts/run_mistral_ocr_batch.py --submit --state "$mistral_state_path"
            echo ""
            echo "[案内] Mistral Batchを送信しました。処理完了後にWidgetを再起動してこの項目を許可してください。"
            ;;
        submitted|queued|running|in_progress)
            run_step "Mistral OCR Batch状態確認" uv run python scripts/run_mistral_ocr_batch.py --status --state "$mistral_state_path"
            phase="$(mistral_state_value phase)"
            if [[ "$phase" != "success" ]]; then
                echo "[案内] Batchはまだ完了していません（phase=${phase:-unknown}）。完了後にWidgetを再起動してください。"
                return 0
            fi
            run_step "Mistral OCR Batch結果回収・品質確認" uv run python scripts/run_mistral_ocr_batch.py --collect --state "$mistral_state_path"
            ;;
        success)
            run_step "Mistral OCR Batch結果回収・品質確認" uv run python scripts/run_mistral_ocr_batch.py --collect --state "$mistral_state_path"
            ;;
        collected)
            ;;
        failed|cancelled|timeout_exceeded)
            echo "[注意] Mistral Batchは終了状態です（phase=$phase）。状態ファイルを確認し、必要なら個別に再送信してください。"
            return 0
            ;;
        *)
            echo "[注意] Mistral Batchの未知の状態です（phase=$phase）。安全のため何も実行しません。"
            return 0
            ;;
    esac

    adopted_at="$(mistral_state_value adoption_applied_at)"
    if [[ -n "$adopted_at" ]]; then
        echo "[情報] このBatchの採用は既に完了しています（$adopted_at）。"
        return 0
    fi
    queue="$(mistral_state_value adoption_queue)"
    adoptable="$(mistral_state_value adoptable_count)"
    if [[ ! "$adoptable" =~ ^[0-9]+$ || "$adoptable" == "0" ]]; then
        echo "[情報] 品質gateを通過したMistral結果はありません。採用は行いません。"
        return 0
    fi
    if [[ -z "$queue" || ! -f "$queue" ]]; then
        echo "[エラー] Mistral adoption queueが見つかりません: ${queue:-<empty>}"
        return 1
    fi
    run_step "Mistral OCR品質gate合格分をV3へ採用（${adoptable}件）" \
        uv run src/index_from_zotero.py --reocr-candidates "$queue" --progress
    mark_mistral_adoption_applied
    run_step "Mistral OCR採用分の文書構造V3更新" \
        uv run python scripts/rebuild_document_structure.py --all
    echo "[完了] Mistral OCRの品質gate合格分をV3へ採用しました。"
}

if [[ "$run_library" == "1" ]]; then
    run_step "ライブラリ差分更新" uv run src/index_from_zotero.py --progress
    run_step "文書構造V3の差分更新" uv run python scripts/rebuild_document_structure.py --all
fi

if [[ "$run_mistral_batch" == "1" ]]; then
    run_mistral_batch_step
fi

if [[ "$run_summaries" == "1" ]]; then
    if [[ "$run_library" == "0" ]]; then
        run_step "文書構造V3の差分確認" uv run python scripts/rebuild_document_structure.py --all
    fi
    # バッチ単位のLLM要約生成を標準機能として実行する。1回のメンテナンスでは
    # 有限バッチ（既定10件）だけ処理し、複数並列ワーカー（既定10）でDeepSeek APIへ
    # 同時にリクエストする。バッチ件数・並列度は環境変数で変更できる
    # （SUMMARY_BACKFILL_BATCH_SIZE / SUMMARY_BACKFILL_WORKERS）。
    # R1差分スキップにより、既にfingerprint一致の要約はLLM呼び出しゼロでskipされるため、
    # ウィジェットを繰り返し実行すれば未処理分から順にバッチが進む。
    # 大規模一括backfillは scripts/detached_summary_backfill.py を別途使う。
    summary_batch_size="${SUMMARY_BACKFILL_BATCH_SIZE:-10}"
    summary_workers="${SUMMARY_BACKFILL_WORKERS:-10}"
    summary_database_gate="${SUMMARY_DATABASE_GATE:-data/quality/server_database_gate.json}"
    if [[ ! -f "$summary_database_gate" ]]; then
        echo "[エラー] DB監査gateがありません。先にServer-Database-Workflow.commandの"
        echo "         フェーズ2を合格させてください: $summary_database_gate"
        exit 2
    fi
    # 全件LLM backfillは「pilotバッチ → 費用レポート → ユーザー承認」の後に限る。
    # 承認マーカーが無い間も、Yを押した場合にバッチ単位でpilot実行できるようにする。
    if [[ -f "data/quality/summary_backfill_approved" ]]; then
        run_step "DeepSeek 文書構造V3要約・検索索引更新（差分・${summary_batch_size}件バッチ・${summary_workers}並列）" \
            uv run python scripts/build_structure_summaries.py --all --mode llm --limit "$summary_batch_size" --workers "$summary_workers" --embed \
            --database-gate "$summary_database_gate"
    else
        echo ""
        echo "[注意] 全件LLM要約backfillは未承認です（data/quality/summary_backfill_approved が無い）。"
        echo "       安全のため${summary_batch_size}件のバッチのみ実行します（SUMMARY_BACKFILL_BATCH_SIZEで変更可）。"
        echo "       承認後にマーカーを作成すると、以後は差分の残り分にバッチが順次進みます。"
        run_step "DeepSeek 文書構造V3要約 ${summary_batch_size}件バッチ" \
            uv run python scripts/build_structure_summaries.py --all --mode llm --limit "$summary_batch_size" --workers "$summary_workers" --embed \
            --database-gate "$summary_database_gate"
    fi
fi

if [[ "$run_citations" == "1" ]]; then
    run_step "Citation Network更新" uv run src/update_citations.py --all
fi

if [[ "$review_relations" == "1" ]]; then
    run_step "品質報告のAI自動判定" uv run python scripts/triage_quality_reports.py
    run_step "曖昧な引用関係レポート確認" uv run python scripts/review_relation_reports.py
    run_step "曖昧な要約品質レポート確認" uv run python scripts/review_summary_quality_reports.py
fi

# 仕様§8: メンテナンス末尾に状態台帳サマリを表示する（read-only）。
# blocked（Mistral queue候補・cloud拒否）・truncated・failed/retryableの溜まりを毎回可視化。
echo ""
echo "========================================================================"
echo ">> 未解決の処理状態サマリ（read-only）"
echo "========================================================================"
uv run python scripts/list_artifact_status.py --unresolved-only || true

echo ""
echo "========================================"
echo " 選択した日常更新が完了しました"
echo "========================================"
