#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
if [ -f "env.sh" ]; then
  # shellcheck disable=SC1091
  source "env.sh"
fi

echo "Updating local extractive summaries and hierarchical indexes..."
uv run python -m src.build_summaries
echo "Summary update completed."
