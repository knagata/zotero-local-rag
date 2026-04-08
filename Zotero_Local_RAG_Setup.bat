@echo off
title Zotero Local RAG Setup
REM Change directory to where the batch file is located
cd /d "%~dp0"

echo Launching Zotero Setup Wizard...

REM Run the interactive python wizard using uv
uv run scripts/setup_wizard.py

echo.
echo Setup script finished.
pause
