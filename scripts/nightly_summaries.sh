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
LLM="${NIGHTLY_LLM:-codex_cli:gpt-5.6-luna}"
LOG="$ROOT/data/nightly_summaries.log"
REOCR_CANDIDATES="${NIGHTLY_REOCR_CANDIDATES:-$ROOT/data/quality/reocr-candidates.json}"
REOCR_MAX_ITEMS="${NIGHTLY_REOCR_MAX_ITEMS:-2}"
MIN_WEEKLY_REMAINING="${NIGHTLY_MIN_WEEKLY_REMAINING_PERCENT:-20}"
LOCK_DIR="$ROOT/data/nightly_summaries.lock"

if [[ "${1:-}" == "--check" ]]; then
  [[ -x "$PYTHON" ]] || { echo "Python not executable: $PYTHON"; exit 1; }
  command -v codex >/dev/null || { echo "codex CLI not found"; exit 1; }
  if [[ "${NIGHTLY_REOCR_ENABLE:-0}" == "1" ]]; then
    command -v ndlocr-lite >/dev/null || { echo "ndlocr-lite not found"; exit 1; }
    [[ -f "$REOCR_CANDIDATES" ]] || { echo "re-OCR candidates not found: $REOCR_CANDIDATES"; exit 1; }
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
if ! "$PYTHON" -m src.codex_quota --min-remaining-percent "$MIN_WEEKLY_REMAINING"; then
  echo "[$(date -Iseconds)] nightly summaries skipped: weekly quota floor or quota unavailable"
  exit 0
fi
if [[ "${NIGHTLY_REOCR_ENABLE:-0}" == "1" ]]; then
  "$PYTHON" scripts/run_reocr_queue.py \
    --candidates "$REOCR_CANDIDATES" --limit "$REOCR_MAX_ITEMS" --llm "$LLM" \
    --output data/nightly_reocr_report.json
fi
caffeinate -i "$PYTHON" -m src.build_summaries \
  --mode llm --llm "$LLM" --stop-on-rate-limit \
  --max-hours "$MAX_HOURS" --max-items "$MAX_ITEMS" \
  --min-weekly-remaining-percent "$MIN_WEEKLY_REMAINING"
echo "[$(date -Iseconds)] nightly summaries end"
"$PYTHON" scripts/nightly_report.py \
  --since-hours "$MAX_HOURS" --output data/nightly_report.json
