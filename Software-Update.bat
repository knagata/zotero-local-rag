@echo off
title Zotero Local RAG Updater
cd /d "%~dp0"
set "SCRIPT_DIR=%~dp0"
set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"

set "REPO_ZIP=https://github.com/knagata/zotero-local-rag/archive/refs/heads/main.zip"
set "TMP_ZIP=%TEMP%\zotero-local-rag-update.zip"
set "TMP_DIR=%TEMP%\zotero-local-rag-update"
set "EXTRACTED=%TMP_DIR%\zotero-local-rag-main"

echo ========================================
echo    Zotero Local RAG - Updater
echo ========================================
echo.
echo Downloading and updating to the latest version from GitHub.
echo .env and data\ (indexes/models) will be preserved.
echo.
set /p "ans=Do you want to continue? [Y/n]: "
if /i "%ans%"=="n" (
    echo Update cancelled.
    pause
    exit /b 0
)

echo.
echo [1/4] Downloading latest version...
if exist "%TMP_ZIP%" del "%TMP_ZIP%"
powershell -NoProfile -Command "Invoke-WebRequest -Uri '%REPO_ZIP%' -OutFile '%TMP_ZIP%'" 2>&1
if not exist "%TMP_ZIP%" (
    echo [!] Download failed. Please check your internet connection.
    pause
    exit /b 1
)

echo.
echo [2/4] Extracting...
if exist "%TMP_DIR%" rmdir /s /q "%TMP_DIR%"
mkdir "%TMP_DIR%"
powershell -NoProfile -Command "Expand-Archive -Path '%TMP_ZIP%' -DestinationPath '%TMP_DIR%' -Force"
if not exist "%EXTRACTED%" (
    echo [!] Extracted folder not found.
    pause
    exit /b 1
)

echo [3/4] Updating files (.env and data\ are preserved)...
REM robocopy: /e=subfolders /xd=exclude dirs /xf=exclude files
robocopy "%EXTRACTED%" "%SCRIPT_DIR%" /e ^
    /xd ".env" "data" ".venv" ".claude" ".git" ^
    /xf ".env" ^
    /NFL /NDL /NJH /NJS >nul

echo [4/4] Cleaning up temporary files...
rmdir /s /q "%TMP_DIR%"
del "%TMP_ZIP%"

echo.
echo ========================================
echo    Update Complete!
echo ========================================
echo.
echo Please restart Claude Desktop to apply the changes.
echo.
set /p "run_idx=Do you want to run the setup wizard now? [y/N]: "
if /i "%run_idx%"=="y" (
    echo.
    uv run scripts/setup_wizard.py
)

echo.
set /p "run_chk=Do you want to run a quality check on the existing index? [y/N]: "
if /i "%run_chk%"=="y" (
    echo.
    uv run src/index_from_zotero.py --check-quality --progress
)

echo.
echo You can safely close this window.
pause
