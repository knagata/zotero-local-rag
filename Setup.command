#!/bin/bash

# Change the working directory to the folder containing this script
cd "$(dirname "$0")"

# Ensure common paths are included so `uv` can be found when double-clicking from GUI
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin:$HOME/.cargo/bin"

echo "Launching Zotero Setup Wizard..."

# Automatically download dependencies and run the wizard via `uv`
uv run scripts/setup_wizard.py

echo ""
echo "You can safely close this terminal window."
