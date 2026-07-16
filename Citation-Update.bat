@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo ========================================
echo    Zotero Local RAG - Citation Update
echo ========================================
echo.
echo Semantic Scholar API: rate-limited across all processes
echo Help: CITATION_UPDATE_GUIDE.md (menu guide and error recovery)
echo.

echo How would you like to process citations?
echo   1) Update a specific item by ID  (skips if already mapped)
echo   2) Force re-update a specific item by ID  (always re-processes)
echo   3) Update ALL items  (new and error items only / recommended)
echo   4) Force re-update ALL items  (re-processes everything)
echo   5) Resume skipped EPUB refs  (budget wo fuyashite skipped wo saikai)
echo   6) Preview PDF/HTML reference extraction for one item (no DB writes)
echo   7) Extract and save PDF/HTML references for one item
echo.
set /p choice="Enter your choice (1-7): "

echo.
if "%choice%"=="1" (
    set /p item_id="Enter the Item ID: "
    if not "!item_id!"=="" (
        echo Starting update for item: !item_id!
        uv run src\update_citations.py --item "!item_id!"
    ) else (
        echo No ID provided. Exiting.
    )
) else if "%choice%"=="2" (
    set /p item_id="Enter the Item ID: "
    if not "!item_id!"=="" (
        echo Force-updating item: !item_id!
        uv run src\update_citations.py --item "!item_id!" --force
    ) else (
        echo No ID provided. Exiting.
    )
) else if "%choice%"=="3" (
    echo Starting bulk update. (Already-mapped items will be skipped)
    uv run src\update_citations.py --all
) else if "%choice%"=="4" (
    echo Starting force bulk update. (All items will be re-processed)
    echo Warning: This may take a very long time for large libraries.
    set /p confirm="Are you sure? (y/N): "
    if /i "!confirm!"=="y" (
        uv run src\update_citations.py --all --force
    ) else (
        echo Cancelled.
    )
) else if "%choice%"=="5" (
    set /p budget="S2 lookup budget per item (default 200): "
    if "!budget!"=="" set budget=200
    echo Resolving skipped EPUB refs with budget=!budget!...
    uv run src\update_citations.py --resume-skipped --epub-budget "!budget!"
) else if "%choice%"=="6" (
    set /p item_id="Enter the Item ID: "
    uv run python -m src.extract_references --item "!item_id!"
) else if "%choice%"=="7" (
    set /p item_id="Enter the Item ID: "
    echo Reference text may be sent to the configured LLM_EXTRACT provider.
    set /p confirm="Continue and save resolved edges? (y/N): "
    if /i "!confirm!"=="y" (
        uv run python -m src.extract_references --item "!item_id!" --commit
    ) else (
        echo Cancelled.
    )
) else (
    echo Invalid choice. Exiting.
)

echo.
pause
