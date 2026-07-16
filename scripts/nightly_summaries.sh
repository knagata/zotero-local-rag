#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
MAX_HOURS="${NIGHTLY_MAX_HOURS:-5}"
MAX_ITEMS="${NIGHTLY_MAX_ITEMS:-20}"
LLM="${NIGHTLY_LLM:-codex_cli:auto}"
LOG="$ROOT/data/nightly_summaries.log"

if [[ "${1:-}" == "--check" ]]; then
  [[ -x "$PYTHON" ]] || { echo "Python not executable: $PYTHON"; exit 1; }
  command -v codex >/dev/null || { echo "codex CLI not found"; exit 1; }
  echo "nightly prerequisites: ok (execution not started)"
  exit 0
fi

if [[ "${NIGHTLY_ENABLE:-0}" != "1" ]]; then
  echo "Nightly execution is disabled. Set NIGHTLY_ENABLE=1 after the quality gate." >&2
  exit 2
fi

mkdir -p "$ROOT/data"
exec >>"$LOG" 2>&1
echo "[$(date -Iseconds)] nightly summaries start"
cd "$ROOT"
caffeinate -i "$PYTHON" -m src.build_summaries \
  --mode llm --llm "$LLM" --stop-on-rate-limit \
  --max-hours "$MAX_HOURS" --max-items "$MAX_ITEMS"
"$PYTHON" scripts/nightly_report.py \
  --since-hours "$MAX_HOURS" --output data/nightly_report.json
echo "[$(date -Iseconds)] nightly summaries end"
