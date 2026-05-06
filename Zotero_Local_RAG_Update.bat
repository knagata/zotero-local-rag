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
echo 最新バージョンをGitHubからダウンロードして更新します。
echo .env と data\ (インデックス・モデル) は保持されます。
echo.
set /p "ans=続けますか？ [Y/n]: "
if /i "%ans%"=="n" (
    echo 更新をキャンセルしました。
    pause
    exit /b 0
)

echo.
echo [1/4] 最新バージョンをダウンロード中...
if exist "%TMP_ZIP%" del "%TMP_ZIP%"
powershell -NoProfile -Command "Invoke-WebRequest -Uri '%REPO_ZIP%' -OutFile '%TMP_ZIP%'" 2>&1
if not exist "%TMP_ZIP%" (
    echo [!] ダウンロードに失敗しました。インターネット接続を確認してください。
    pause
    exit /b 1
)

echo.
echo [2/4] 展開中...
if exist "%TMP_DIR%" rmdir /s /q "%TMP_DIR%"
mkdir "%TMP_DIR%"
powershell -NoProfile -Command "Expand-Archive -Path '%TMP_ZIP%' -DestinationPath '%TMP_DIR%' -Force"
if not exist "%EXTRACTED%" (
    echo [!] 展開後のフォルダが見つかりません。
    pause
    exit /b 1
)

echo [3/4] ファイルを更新中（.env と data\ は保持）...
REM robocopy: /e=サブフォルダ含む /xd=除外ディレクトリ /xf=除外ファイル /NFL /NDL=ログ簡略化
robocopy "%EXTRACTED%" "%SCRIPT_DIR%" /e ^
    /xd ".env" "data" ".venv" ".claude" ".git" ^
    /xf ".env" ^
    /NFL /NDL /NJH /NJS >nul

echo [4/4] 一時ファイルを削除中...
rmdir /s /q "%TMP_DIR%"
del "%TMP_ZIP%"

echo.
echo ========================================
echo    更新完了！
echo ========================================
echo.
echo Claude Desktop を再起動すると変更が反映されます。
echo.
set /p "run_idx=埋め込みインデクサーを再実行しますか？ [y/N]: "
if /i "%run_idx%"=="y" (
    echo.
    uv run scripts/setup_wizard.py
)

echo.
echo このウィンドウは閉じて構いません。
pause
