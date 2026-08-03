#!/bin/bash
set -uo pipefail

cd "$(dirname "$0")"
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin:$HOME/.cargo/bin"

# Runs-end summary (spec: "何が起きたか一目で分かる、次にやることがあれば
# それも分かる" -- a long scrolling transcript otherwise makes both easy to
# miss, as it did for the citation-network step ending silently and for the
# Mistral batch's "come back later" notice getting buried, 2026-08-01/02).
# SUMMARY_ROWS holds one "icon|label|detail" entry per step outcome;
# NEXT_ACTIONS holds plain strings for anything the operator still needs to
# do. Both print_summary (normal end) and run_step's failure path (abort)
# render them, so the operator sees this even when the run stops early.
RUN_START_TIME=$(date +%s)
SUMMARY_ROWS=()
NEXT_ACTIONS=()


# ASCII Unit Separator: unlike "|", this cannot appear in a label/detail
# string typed or interpolated here (paths, timestamps, state names), so
# record_summary's encoding can't be corrupted by a value containing it.
SUMMARY_FIELD_SEP=$'\x1f'

record_summary() {
    SUMMARY_ROWS+=("$1${SUMMARY_FIELD_SEP}$2${SUMMARY_FIELD_SEP}${3:-}")
}

record_next_action() {
    NEXT_ACTIONS+=("$1")
}

fmt_elapsed() {
    local total="$1" h m s
    h=$((total / 3600)); m=$(((total % 3600) / 60)); s=$((total % 60))
    if [[ "$h" -gt 0 ]]; then printf "%dh%dm%ds" "$h" "$m" "$s"
    elif [[ "$m" -gt 0 ]]; then printf "%dm%ds" "$m" "$s"
    else printf "%ds" "$s"; fi
}

print_summary() {
    echo ""
    echo "========================================"
    echo " 実行サマリー"
    echo "========================================"
    if [[ ${#SUMMARY_ROWS[@]} -eq 0 ]]; then
        echo "  （実行した項目はありません）"
    else
        local row status label detail
        for row in "${SUMMARY_ROWS[@]}"; do
            IFS="$SUMMARY_FIELD_SEP" read -r status label detail <<< "$row"
            if [[ -n "$detail" ]]; then
                echo "  ${status} ${label}  (${detail})"
            else
                echo "  ${status} ${label}"
            fi
        done
    fi
    echo "  合計所要時間: $(fmt_elapsed $(( $(date +%s) - RUN_START_TIME )))"

    if [[ ${#NEXT_ACTIONS[@]} -gt 0 ]]; then
        echo ""
        echo "次にやること:"
        local action
        for action in "${NEXT_ACTIONS[@]}"; do
            echo "  - ${action}"
        done
    fi
}

echo "========================================"
echo " Zotero Local RAG - メンテナンス更新"
echo "========================================"
echo ""
echo "日常更新を次の順番でまとめて実行します。"
echo "Enterを押すと既定の「実行」を選択します。"
echo ""

# The library owner may make local maintenance non-interactive (e.g. for a
# cron/launchd job) by exporting MAINTENANCE_AUTO_APPROVE=1 in that job's own
# environment. Paid API steps (hierarchical summaries and cloud OCR) remain
# explicit per-run opt-ins regardless. Defaults to 0 (ask), matching
# .env.example -- this script never sources .env itself (only the uv-run
# Python subprocesses it launches do), so an interactive double-click launch
# must default to asking here or every question (including "save a log?")
# silently never fires (found 2026-07-31: the fallback here was "1" while
# .env.example documented "0", so a plain double-click always ran fully
# unattended with paid steps silently skipped instead of offered).
maintenance_auto_approve="${MAINTENANCE_AUTO_APPROVE:-0}"
if [[ "$maintenance_auto_approve" == "1" ]]; then
    echo "[情報] MAINTENANCE_AUTO_APPROVE=1: ローカル更新を自動許可します（有料API処理は除外）。"
    echo ""
fi

# Off by default -- routine runs don't need a permanent transcript, but
# embedding/audit work is exactly the kind of thing worth being able to
# debug after the terminal window is long gone (2026-07-30). Auto-approve
# runs (cron/launchd) never see this prompt, so they need the env var
# instead if logging is wanted there.
save_log=0
if [[ "$maintenance_auto_approve" == "1" ]]; then
    if [[ "${MAINTENANCE_SAVE_LOG:-0}" == "1" ]]; then
        save_log=1
    fi
else
    read -r -p "ログを保存しますか？ [y/N]: " log_answer
    case "$log_answer" in
        y|Y|yes|Yes|YES) save_log=1 ;;
    esac
fi
if [[ "$save_log" == "1" ]]; then
    log_dir="${MAINTENANCE_LOG_DIR:-data/logs}"
    mkdir -p "$log_dir"
    log_file="$log_dir/maintenance_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "$log_file") 2>&1
    echo "[情報] ログを保存します: $log_file"
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

# The audit is free and non-destructive (Zotero照合＋原本照合、read-only), so
# unlike summaries/Mistral it is safe to auto-approve -- but only when it is
# actually needed. Its default tracks whether a passing gate already exists:
# the first run (no gate yet, or a prior audit failed and cleared it) defaults
# to yes, every run after a passing audit defaults to no. This mirrors what a
# careful operator would type by hand instead of asking them to remember it.
gate_path="${SERVER_DB_GATE_PATH:-data/quality/server_database_gate.json}"

# scripts/run_db_audit.py writes these two reports as its first two checks
# run, regardless of whether the overall audit passes -- gate_path (above) is
# written only by its *third* check, and only once the first two already
# passed. A message that points a failed-audit operator at gate_path for
# "details" sends them to a file that was never created, since a failure at
# either of these first two checks (the common case) means the third check
# -- the one that would write it -- never ran (found 2026-08-03, from an
# operator confused that the file the widget told them to check didn't
# exist). These two are what actually hold the failure detail in that case.
zotero_audit_path="${SERVER_ZOTERO_AUDIT_PATH:-data/quality/server_zotero_reconciliation.json}"
source_audit_path="${SERVER_SOURCE_AUDIT_PATH:-data/quality/server_source_verification.json}"

# scripts/audit_v3_cutover.py writes this file unconditionally -- including
# on failure, with "passed": false -- before it raises on that failure. So
# the file's mere existence is not evidence the audit passed; only its
# `gate.passed` field is (matches src/database_gate.py's own check).
gate_passes() {
    [[ -f "$gate_path" ]] || return 1
    /usr/bin/python3 - "$gate_path" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as f:
        data = json.load(f)
except (OSError, ValueError):
    sys.exit(1)
gate = data.get("gate") if isinstance(data, dict) else None
sys.exit(0 if isinstance(gate, dict) and gate.get("passed") is True else 1)
PY
}

audit_default_yes=1
if gate_passes; then
    audit_default_yes=0
fi

ask_audit() {
    local prompt="$1"
    local answer
    if [[ "$audit_default_yes" == "1" ]]; then
        if [[ "$maintenance_auto_approve" == "1" ]]; then
            echo "$prompt [自動許可: 監査合格証明が未作成/失効のため]"
            return 0
        fi
        read -r -p "$prompt [Y/n]: " answer
        case "$answer" in
            n|N|no|No|NO) return 1 ;;
            *) return 0 ;;
        esac
    else
        if [[ "$maintenance_auto_approve" == "1" ]]; then
            echo "$prompt [自動実行の対象外: 監査合格証明は最新]"
            return 1
        fi
        read -r -p "$prompt [y/N]: " answer
        case "$answer" in
            y|Y|yes|Yes|YES) return 0 ;;
            *) return 1 ;;
        esac
    fi
}

# Shared by both paid-summary blocks below: true only once the audit
# succeeded and a fresh gate exists. Records why it was skipped otherwise
# instead of duplicating that logic in each block.
require_gate() {
    local step_label="$1"
    if [[ "$audit_failed" == "1" ]]; then
        echo "[エラー] この実行のDB監査が不合格のため、${step_label}はスキップします。"
        summary_blocked=1
        record_summary "－" "$step_label" "スキップ: この実行のDB監査が不合格"
        record_next_action "「${step_label}」はDB監査の不合格によりスキップされました。原因を解消し「2. DBを監査する」を合格させてから、この項目を再実行してください。"
        return 1
    fi
    if ! gate_passes; then
        echo "[エラー] DB監査の合格証明がありません。上の「2. DBを監査する」を許可して"
        echo "         先に合格させてください: $gate_path"
        summary_blocked=1
        record_summary "－" "$step_label" "スキップ: DB監査の合格証明なし"
        record_next_action "「${step_label}」はDB監査の合格証明が無いためスキップされました。「2. DBを監査する」を許可して先に合格させてから、この項目を再実行してください: $gate_path"
        return 1
    fi
    return 0
}

run_library=0
run_audit=0
run_summaries=0
run_bulk_summary=0
run_citations=0
run_mistral_batch=0
# A requested summary step that could not run for lack of a passing gate is
# reported at the very end rather than aborting mid-run: the citation update
# and the unresolved-status listing do not depend on the audit, so killing the
# script here would silently drop work the operator explicitly asked for.
summary_blocked=0
# Set when this run's audit failed. Tracked separately from the gate file
# because run_db_audit.py can die before it invalidates a previous gate,
# which would otherwise let a stale gate authorise paid summaries in the very
# run the operator was told would generate none.
audit_failed=0
mistral_state_path="${MISTRAL_BATCH_STATE_PATH:-data/mistral_ocr_batch_state.json}"

if ask_enabled "1. ライブラリを差分更新する"; then
    run_library=1
fi
if ask_audit "2. DBを監査する（Zotero本体との突き合わせを含む・非破壊。要約の実行に必要）"; then
    run_audit=1
fi
# Paid summaries are never included by auto-approval. They require an explicit
# per-run opt-in and a server database gate produced after a successful audit.
if [[ "$maintenance_auto_approve" == "1" ]]; then
    echo "3. 要約を差分更新する（DeepSeek API料金あり） [自動実行の対象外]"
elif ask_disabled "3. 要約を差分更新する（DeepSeek API料金あり・DB監査合格後のみ・少量バッチ）"; then
    run_summaries=1
fi
if [[ "$maintenance_auto_approve" == "1" ]]; then
    echo "4. 全件要約を一括生成する（DeepSeek API課金・重い処理） [自動実行の対象外]"
elif ask_disabled "4. 全件要約を一括生成する（DeepSeek API課金・DB監査合格後のみ・SUMMARIZE確認・重い処理）"; then
    run_bulk_summary=1
fi
if ask_enabled "5. 引用ネットワークの未処理・エラー分を更新する"; then
    run_citations=1
fi
echo "6. 品質報告のAI判定 [退役: 旧要約DBを参照するため実行不可]"
if ask_disabled "7. Mistral OCRバッチを送信・回収・品質確認・採用する（クラウド送信を含む）"; then
    run_mistral_batch=1
fi

echo ""
echo "実行予定:"
if [[ "$run_library" == "1" ]]; then echo "  ✓ ライブラリ更新"; fi
if [[ "$run_audit" == "1" ]]; then echo "  ✓ DB監査"; fi
if [[ "$run_summaries" == "1" ]]; then echo "  ✓ 文書構造要約・検索索引の更新（差分）"; fi
if [[ "$run_bulk_summary" == "1" ]]; then echo "  ✓ 全件要約の一括生成"; fi
if [[ "$run_citations" == "1" ]]; then echo "  ✓ 引用ネットワーク更新"; fi
if [[ "$run_mistral_batch" == "1" ]]; then echo "  ✓ Mistral OCRバッチ処理"; fi

if [[ "$run_library" == "0" && "$run_audit" == "0" && "$run_summaries" == "0" \
        && "$run_bulk_summary" == "0" && "$run_citations" == "0" && "$run_mistral_batch" == "0" ]]; then
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
    local t0 elapsed
    t0=$(date +%s)
    echo ""
    echo "========================================================================"
    echo ">> $label"
    echo "========================================================================"
    if "$@"; then
        elapsed=$(( $(date +%s) - t0 ))
        echo "[完了] $label"
        record_summary "✓" "$label" "$(fmt_elapsed "$elapsed")"
    else
        local code=$?
        elapsed=$(( $(date +%s) - t0 ))
        echo "[エラー] ${label}（終了コード: ${code}）"
        echo "安全のため後続処理を実行せず終了します。"
        record_summary "✗" "$label" "終了コード ${code}, $(fmt_elapsed "$elapsed")"
        record_next_action "「${label}」がエラーで停止しました（終了コード ${code}）。上記の出力を確認して原因に対処し、再実行してください。"
        print_summary
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
            run_step "Mistral OCRバッチ送信" uv run python scripts/run_mistral_ocr_batch.py --submit --state "$mistral_state_path"
            echo ""
            echo "[案内] Mistralバッチを送信しました。処理完了後にMaintenance-Widget.commandを再度起動し、"
            echo "       この項目を明示的に許可してください。次回は結果を回収し、品質検証の合格分だけをV3へ採用します。"
            record_next_action "Mistral OCRバッチを送信しました。クラウド側の処理完了後、Maintenance-Widget.commandを再度実行し「7. Mistral OCRバッチ」を許可すると結果を回収します。"
            ;;
        prepared|uploaded)
            run_step "Mistral OCRバッチ送信を再開" uv run python scripts/run_mistral_ocr_batch.py --submit --state "$mistral_state_path"
            echo ""
            echo "[案内] Mistralバッチを送信しました。処理完了後にメンテナンス画面を再起動してこの項目を許可してください。"
            record_next_action "Mistral OCRバッチを送信しました。クラウド側の処理完了後、Maintenance-Widget.commandを再度実行し「7. Mistral OCRバッチ」を許可すると結果を回収します。"
            ;;
        submitted|queued|running|in_progress)
            run_step "Mistral OCRバッチ状態確認" uv run python scripts/run_mistral_ocr_batch.py --status --state "$mistral_state_path"
            phase="$(mistral_state_value phase)"
            if [[ "$phase" != "success" ]]; then
                echo "[案内] バッチはまだ完了していません（状態=${phase:-不明}）。完了後にウィジェットを再起動してください。"
                record_summary "－" "Mistral OCRバッチ" "実行中（状態=${phase:-不明}）"
                record_next_action "Mistral OCRバッチがまだ完了していません（状態=${phase:-不明}）。完了後にMaintenance-Widget.commandを再度実行してください。"
                return 0
            fi
            run_step "Mistral OCRバッチ結果回収・品質確認" uv run python scripts/run_mistral_ocr_batch.py --collect --state "$mistral_state_path"
            ;;
        success)
            run_step "Mistral OCRバッチ結果回収・品質確認" uv run python scripts/run_mistral_ocr_batch.py --collect --state "$mistral_state_path"
            ;;
        collected)
            record_summary "－" "Mistral OCRバッチ" "既に結果回収済み"
            ;;
        failed|cancelled|timeout_exceeded)
            echo "[注意] Mistralバッチは終了状態です（状態=${phase}）。状態ファイルを確認し、必要なら個別に再送信してください。"
            record_summary "⚠" "Mistral OCRバッチ" "終了状態=$phase"
            record_next_action "Mistral OCRバッチが失敗/中断しました（状態=${phase}）。状態ファイル（${mistral_state_path}）を確認し、必要なら個別に再送信してください。"
            return 0
            ;;
        *)
            echo "[注意] Mistralバッチの未知の状態です（状態=${phase}）。安全のため何も実行しません。"
            record_summary "⚠" "Mistral OCRバッチ" "未知の状態=$phase"
            record_next_action "Mistral OCRバッチの状態（${phase}）が想定外です。状態ファイル（${mistral_state_path}）を確認してください。"
            return 0
            ;;
    esac

    adopted_at="$(mistral_state_value adoption_applied_at)"
    if [[ -n "$adopted_at" ]]; then
        echo "[情報] このバッチの採用は既に完了しています（${adopted_at}）。"
        record_summary "－" "Mistral OCR採用" "既に完了（${adopted_at}）"
        return 0
    fi
    queue="$(mistral_state_value adoption_queue)"
    adoptable="$(mistral_state_value adoptable_count)"
    if [[ ! "$adoptable" =~ ^[0-9]+$ || "$adoptable" == "0" ]]; then
        echo "[情報] 品質検証を通過したMistral結果はありません。採用は行いません。"
        record_summary "－" "Mistral OCR採用" "品質検証通過分なし"
        return 0
    fi
    if [[ -z "$queue" || ! -f "$queue" ]]; then
        echo "[エラー] Mistral採用キューが見つかりません: ${queue:-<空>}"
        record_summary "✗" "Mistral OCR採用" "採用キューが見つからない"
        record_next_action "Mistral採用キューが見つかりません（${queue:-<空>}）。状態ファイル（${mistral_state_path}）を確認してください。"
        return 1
    fi
    run_step "Mistral OCR品質検証の合格分をV3へ採用（${adoptable}件）" \
        uv run src/index_from_zotero.py --reocr-candidates "$queue" --progress
    mark_mistral_adoption_applied
    run_step "Mistral OCR採用分の文書構造V3更新" \
        uv run python scripts/rebuild_document_structure.py --all
    echo "[完了] Mistral OCRの品質検証合格分をV3へ採用しました。"
}

if [[ "$run_library" == "1" ]]; then
    run_step "ライブラリ差分更新" uv run src/index_from_zotero.py --progress
    run_step "文書構造V3の差分更新" uv run python scripts/rebuild_document_structure.py --all
fi

if [[ "$run_mistral_batch" == "1" ]]; then
    run_mistral_batch_step
fi

# A summary step needs an up-to-date document structure even when the library
# update itself was skipped this run. This must happen before the audit below
# -- it writes document_structures, which is an input to the gate's fingerprint
# -- not inside the summary blocks, where it would run after the audit instead.
if [[ "$run_library" == "0" && ( "$run_summaries" == "1" || "$run_bulk_summary" == "1" ) ]]; then
    run_step "文書構造V3の差分確認" uv run python scripts/rebuild_document_structure.py --all
fi

# The audit runs after every step that writes canonical data (library update,
# Mistral adoption, the structure refresh above) and before every step that
# consumes its gate. Auditing earlier would bind the gate to a DB generation
# this same run then replaces, and build_structure_summaries.py would reject
# it as stale.
if [[ "$run_audit" == "1" ]]; then
    echo ""
    echo "========================================================================"
    echo ">> DB監査（Zotero本体との突き合わせ・原本照合・非破壊）"
    echo "========================================================================"
    # A failed audit must not abort the rest of routine maintenance -- only
    # the paid summary steps depend on its result. Citations and the library
    # update are independent of it. The gate file alone is not a sufficient
    # signal: run_db_audit.py can fail before it gets far enough to invalidate
    # a previous gate, so the failure is recorded explicitly here instead.
    if uv run python scripts/run_db_audit.py; then
        echo "[完了] DB監査"
        record_summary "✓" "DB監査" ""
    else
        audit_failed=1
        echo "[警告] DB監査が不合格でした。要約の生成はこの実行では行われません。"
        # gate_path is only written by the audit's third/last check, which
        # never runs if either of these first two failed (the common case) --
        # pointing here at gate_path would send the operator to a file that
        # does not exist yet.
        echo "       詳細は上記の出力、または次のレポートを確認してください:"
        echo "       ${zotero_audit_path}"
        echo "       ${source_audit_path}"
        record_summary "⚠" "DB監査" "不合格"
        record_next_action "DB監査が不合格でした。上記の出力、または ${zotero_audit_path} / ${source_audit_path} を確認して原因に対処し、監査に合格させてください。"
    fi
fi

if [[ "$run_summaries" == "1" ]]; then
    # バッチ単位のLLM要約生成を標準機能として実行する。1回のメンテナンスでは
    # 有限バッチ（既定10件）だけ処理し、複数並列ワーカー（既定10）でDeepSeek APIへ
    # 同時にリクエストする。バッチ件数・並列度は環境変数で変更できる
    # （SUMMARY_BACKFILL_BATCH_SIZE / SUMMARY_BACKFILL_WORKERS）。
    # R1差分スキップにより、既にfingerprint一致の要約はLLM呼び出しゼロでskipされるため、
    # ウィジェットを繰り返し実行すれば未処理分から順にバッチが進む。
    # 全件を一度に済ませたい場合は下の「4. 全件要約を一括生成する」を使う。
    summary_batch_size="${SUMMARY_BACKFILL_BATCH_SIZE:-10}"
    summary_workers="${SUMMARY_BACKFILL_WORKERS:-10}"
    if require_gate "要約の差分更新"; then
        run_step "DeepSeek 文書構造V3要約・検索索引更新（差分・${summary_batch_size}件バッチ・${summary_workers}並列）" \
            uv run python scripts/build_structure_summaries.py --all --mode llm --limit "$summary_batch_size" --workers "$summary_workers" --embed \
            --database-gate "$gate_path"
    fi
fi

if [[ "$run_bulk_summary" == "1" ]]; then
    echo ""
    echo "========================================================================"
    echo ">> 全件要約の一括生成（DeepSeek API課金・重い処理）"
    echo "========================================================================"
    if require_gate "全件要約の一括生成"; then
        echo "[課金確認] DeepSeek APIで全件の階層要約を生成します。DB変更後の古い合格証明は"
        echo "           自動拒否されます。"
        read -r -p "続行するには SUMMARIZE と入力してください（スキップする場合はEnter）: " bulk_confirmation
        if [[ "$bulk_confirmation" != "SUMMARIZE" ]]; then
            echo "全件要約の一括生成をキャンセルしました。"
            record_summary "－" "全件要約の一括生成" "キャンセル（未確認）"
        else
            bulk_summary_workers="${SUMMARY_BULK_WORKERS:-20}"
            run_step "DeepSeek 階層AI要約生成・要約索引構築（全件・${bulk_summary_workers}並列）" \
                uv run python scripts/build_structure_summaries.py --all --mode llm \
                --workers "$bulk_summary_workers" --embed --database-gate "$gate_path"
            run_step "階層要約・要約索引監査" \
                uv run python scripts/audit_structure_summaries.py \
                --output "${SERVER_SUMMARY_AUDIT_PATH:-data/quality/server_summary_audit.json}"
        fi
    fi
fi

if [[ "$run_citations" == "1" ]]; then
    run_step "引用ネットワーク更新" uv run src/update_citations.py --all
fi

# 仕様§8: メンテナンス末尾に状態台帳サマリを表示する（read-only）。
# blocked（Mistral queue候補・cloud拒否）・truncated・failed/retryableの溜まりを毎回可視化。
echo ""
echo "========================================================================"
echo ">> 未解決の処理状態サマリ（読み取り専用）"
echo "========================================================================"
artifact_status_json="$(uv run python scripts/list_artifact_status.py --unresolved-only)" || artifact_status_json=""
if [[ -n "$artifact_status_json" ]]; then
    echo "$artifact_status_json"
    unresolved_count="$(printf '%s' "$artifact_status_json" | /usr/bin/python3 -c '
import json, sys
try:
    print(int(json.load(sys.stdin).get("unresolved_count") or 0))
except (ValueError, TypeError):
    print(0)
' 2>/dev/null)"
    if [[ "$unresolved_count" =~ ^[0-9]+$ && "$unresolved_count" -gt 0 ]]; then
        record_next_action "未解決の処理状態が${unresolved_count}件あります。上記の「未解決の処理状態サマリ」を確認してください。"
    fi
fi

# Chromaデータベース健全性チェック（読み取り専用、FTS5破損のみ自動修復）。
# 中断された書き込み（強制終了・SIGSEGV等）がFTS5の内部索引だけを本体データと
# 無関係に破損させる場合があり、これまで気づく手段がなかった（手作業調査で発覚、
# 2026-08-03）。孤立セグメントディレクトリ（中断されたSetupの残骸）も同時に
# 報告するが、実害がないため削除はしない。
echo ""
echo "========================================================================"
echo ">> Chromaデータベース健全性チェック（読み取り専用、FTS5破損のみ自動修復）"
echo "========================================================================"
chroma_health_path="${SERVER_CHROMA_HEALTH_PATH:-data/quality/server_chroma_health.json}"
# No `|| chroma_health_json=""` here: unlike list_artifact_status.py (always
# exits 0), this script deliberately exits 2 when a problem is found -- the
# exact case that must not be swallowed. `set -u`/`pipefail` are active but
# not `-e`, so a nonzero exit here does not abort the run either way.
chroma_health_json="$(uv run python scripts/check_chroma_health.py --output "$chroma_health_path")"
if [[ -n "$chroma_health_json" ]]; then
    echo "$chroma_health_json"
    chroma_health_line="$(printf '%s' "$chroma_health_json" | /usr/bin/python3 -c "
import json, sys
sep = '$SUMMARY_FIELD_SEP'
try:
    d = json.load(sys.stdin)
except (ValueError, TypeError):
    d = {}
passed = bool(d.get('passed'))
repaired = bool(d.get('fts_repair_attempted'))
orphans = len(d.get('orphaned_segment_dirs') or [])
parts = []
if repaired:
    parts.append('FTS5破損を自動修復')
if orphans:
    parts.append(f'孤立セグメント{orphans}件')
if not parts:
    parts.append('問題なし')
print(sep.join(['✓' if passed else '✗', ', '.join(parts), '1' if passed else '0']))
" 2>/dev/null)"
    IFS="$SUMMARY_FIELD_SEP" read -r chroma_icon chroma_detail chroma_passed <<< "$chroma_health_line"
    record_summary "${chroma_icon:-✗}" "Chroma健全性チェック" "${chroma_detail:-}"
    if [[ "$chroma_passed" != "1" ]]; then
        record_next_action "Chromaデータベースの整合性チェックで問題が見つかりました（FTS5以外の破損の可能性があります）。${chroma_health_path} を確認してください。"
    fi
fi

# 実効設定のスナップショットをログへ残す（read-only）。
# 2台で挙動が違うとき、その場の設定が分からないと原因究明が手作業の総当たりに
# なる。PDF_SCANNED_PAGE_PATCH_ENABLE が片方だけ未設定だったために1週間分の
# ページ内容が静かに失われた件（2026-08-03）は、まさにこれが無くて発見が
# 遅れた。APIキーは値を出さず「設定済み・長さN」だけを表示するので、この
# ログはそのまま共有できる。
config_snapshot_path="${SERVER_CONFIG_SNAPSHOT_PATH:-data/quality/effective_config.json}"
mkdir -p "$(dirname "${config_snapshot_path}")"
# if/else にするのは `cmd && A || B` だと A が失敗したときにも B が走るため。
# stderr は捨てない: 取得できなかった理由が分からなければ、この診断機能自体が
# 目的を果たさない。
if uv run python scripts/show_effective_config.py --json > "$config_snapshot_path"; then
    record_summary "－" "実効設定スナップショット" "${config_snapshot_path}"
else
    record_summary "⚠" "実効設定スナップショット" "取得できませんでした"
fi

echo ""
echo "========================================"
echo " 選択した日常更新が完了しました"
echo "========================================"

if [[ "$summary_blocked" == "1" ]]; then
    echo ""
    echo "[注意] DB監査に合格していないため、要約の生成はスキップしました。"
    echo "       他の更新は最後まで実行済みです。「2. DBを監査する」を合格させてから"
    echo "       要約を再実行してください: $gate_path"
fi

print_summary

if [[ "$summary_blocked" == "1" ]]; then
    exit 2
fi
