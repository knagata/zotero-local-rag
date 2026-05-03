@echo off
title Zotero Local RAG Setup
cd /d "%~dp0"

REM Add common uv install paths so it can be found without a full shell session
set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"

echo Launching Zotero Setup Wizard...

uv run scripts/setup_wizard.py

echo.
echo You can safely close this window.
pause
