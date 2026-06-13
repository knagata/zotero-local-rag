@echo off
setlocal

cd /d "%~dp0"

echo ========================================
echo    Zotero Local RAG - Citation Graph
echo ========================================
echo.
echo グラフを構築中です。ブラウザが自動的に開きます...
echo 終了するには Ctrl+C を押してください。
echo.

uv run scripts\show_citation_graph.py

echo.
echo サーバーを終了しました。
pause
