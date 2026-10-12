@echo off
chcp 65001 >nul
setlocal
title "Geekatplay LoRA Maker - Verify Training"
cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Run install.bat first.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat

echo ==============================================================
echo   Verifying that LoRA training really runs on this machine
echo   (20 real steps on your SDXL checkpoint, or a tiny test model)
echo ==============================================================
echo.
python verify_training.py %*
set RC=%errorlevel%
echo.
if %RC% equ 0 (
    echo [OK] Training verified. Use the "Smoke test" preset in the Studio for the same check on your own dataset.
) else (
    echo [FAIL] Verification failed with code %RC%. Copy the output above into the Diagnostics console or an issue.
)
pause
exit /b %RC%
