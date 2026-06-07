@echo off
title Zotero Local RAG Library Update
cd /d "%~dp0"
set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"

echo ========================================
echo    Zotero Local RAG - Library Update
echo ========================================
echo.
echo Please select an action to perform:
echo   [1] Sync library and run embedding (index new/modified files)
echo   [2] Run quality check only (detect scanned pages or corrupted text)
echo   [3] Run high-fidelity re-indexing on scanned/corrupted PDFs (requires Docling)
echo   [4] Cancel
echo.
set "choice=1"
set /p "choice=Choice [1-4, Default: 1]: "

if "%choice%"=="1" (
    echo.
    set "use_doc=n"
    set /p "use_doc=Use high-fidelity parsing (Docling) for all PDFs? [y/N]: "
    if /i "%use_doc%"=="y" (
        echo.
        echo >> Syncing library and running high-fidelity embedding indexer (Docling)...
        uv run src/index_from_zotero.py --progress --use-docling
    ) else (
        echo.
        echo >> Syncing library and running standard embedding indexer...
        uv run src/index_from_zotero.py --progress
    )
) else if "%choice%"=="2" (
    echo.
    echo >> Running quality check on existing index...
    uv run src/index_from_zotero.py --check-quality --progress
) else if "%choice%"=="3" (
    echo.
    echo >> Re-indexing scanned/corrupted PDFs using high-fidelity parser (Docling)...
    uv run src/index_from_zotero.py --reparse-corrupted --progress
) else (
    echo Operation cancelled.
)

echo.
pause
