#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
source "$ROOT/scripts/lib/load_dotenv.zsh"
load_dotenv_file "$ROOT/.env"
load_dotenv_file "$ROOT/.env.policy"
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
MAX_HOURS="${NIGHTLY_MAX_HOURS:-5}"
MAX_ITEMS="${NIGHTLY_MAX_ITEMS:-20}"
SUMMARY_MODEL="${NIGHTLY_SUMMARY_MODEL:-deepseek-v4-flash}"
SUMMARY_FALLBACK_MODEL="${NIGHTLY_SUMMARY_FALLBACK_MODEL:-deepseek-v4-pro}"
SUMMARY_WORKERS="${NIGHTLY_SUMMARY_WORKERS:-10}"
REOCR_LLM="${NIGHTLY_REOCR_LLM:-deepseek:deepseek-v4-pro}"
LOG="$ROOT/data/nightly_summaries.log"
REOCR_CANDIDATES="${NIGHTLY_REOCR_CANDIDATES:-$ROOT/data/quality/reocr-candidates.json}"
REOCR_MAX_ITEMS="${NIGHTLY_REOCR_MAX_ITEMS:-2}"
MIN_WEEKLY_REMAINING="${NIGHTLY_MIN_WEEKLY_REMAINING_PERCENT:-20}"
LOCK_DIR="$ROOT/data/nightly_summaries.lock"
STOP_FILE="${NIGHTLY_STOP_FILE:-$ROOT/data/nightly.stop}"

if [[ "${1:-}" == "--check" ]]; then
  [[ -x "$PYTHON" ]] || { echo "Python not executable: $PYTHON"; exit 1; }
  [[ -n "${DEEPSEEK_API_KEY:-}" ]] || { echo "DEEPSEEK_API_KEY is not set"; exit 1; }
  if [[ "${NIGHTLY_REOCR_ENABLE:-0}" == "1" ]]; then
    command -v ndlocr-lite >/dev/null || { echo "ndlocr-lite not found"; exit 1; }
    [[ -f "$REOCR_CANDIDATES" ]] || { echo "re-OCR candidates not found: $REOCR_CANDIDATES"; exit 1; }
    if [[ "$REOCR_LLM" == codex_cli:* ]]; then
      command -v codex >/dev/null || { echo "codex CLI not found"; exit 1; }
    fi
  fi
  echo "nightly prerequisites: ok (execution not started; NIGHTLY_ENABLE=${NIGHTLY_ENABLE:-0})"
  exit 0
fi

if [[ "${NIGHTLY_ENABLE:-0}" != "1" ]]; then
  echo "Nightly execution is disabled. Set NIGHTLY_ENABLE=1 after the quality gate." >&2
  exit 2
fi

mkdir -p "$ROOT/data"
exec > >(tee -a "$LOG") 2>&1
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  LOCK_PID="$(<"$LOCK_DIR/pid" 2>/dev/null || true)"
  if [[ "$LOCK_PID" == <-> ]] && kill -0 "$LOCK_PID" 2>/dev/null; then
    echo "[$(date -Iseconds)] nightly summaries skipped: already running (pid=$LOCK_PID)"
    exit 0
  fi
  rm -f "$LOCK_DIR/pid"
  rmdir "$LOCK_DIR" 2>/dev/null || true
  mkdir "$LOCK_DIR"
fi
echo $$ > "$LOCK_DIR/pid"
cleanup_lock() {
  rm -f "$LOCK_DIR/pid"
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup_lock EXIT INT TERM HUP
echo "[$(date -Iseconds)] nightly summaries start pid=$$"
cd "$ROOT"
if [[ "${NIGHTLY_REOCR_ENABLE:-0}" == "1" ]]; then
  if [[ "$REOCR_LLM" == codex_cli:* ]] && ! "$PYTHON" -m src.codex_quota --min-remaining-percent "$MIN_WEEKLY_REMAINING"; then
    echo "[$(date -Iseconds)] nightly re-OCR skipped: weekly quota floor or quota unavailable"
  else
  "$PYTHON" scripts/run_reocr_queue.py \
    --candidates "$REOCR_CANDIDATES" --limit "$REOCR_MAX_ITEMS" --llm "$REOCR_LLM" \
    --output data/nightly_reocr_report.json
  fi
fi
caffeinate -i "$PYTHON" scripts/build_deepseek_summaries.py \
  --model "$SUMMARY_MODEL" --fallback-model "$SUMMARY_FALLBACK_MODEL" \
  --workers "$SUMMARY_WORKERS" --max-hours "$MAX_HOURS" --max-items "$MAX_ITEMS" \
  --stop-file "$STOP_FILE" --output data/nightly_deepseek_summary_report.json
echo "[$(date -Iseconds)] nightly summaries end"
"$PYTHON" scripts/nightly_report.py \
  --since-hours "$MAX_HOURS" --output data/nightly_report.json
