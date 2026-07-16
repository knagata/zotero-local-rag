@echo off
setlocal
cd /d "%~dp0"
echo Updating local extractive summaries and hierarchical indexes...
uv run python -m src.build_summaries
if errorlevel 1 (
  echo Summary update failed.
  exit /b 1
)
echo Summary update completed.
