@echo off
chcp 65001 >nul
setlocal
title "Geekatplay LoRA Maker Studio - Vladimir Chopine"
cd /d "%~dp0"

echo ==============================================================
echo   Geekatplay LoRA Maker Studio v2.0.0
echo   Universal AI LoRA Assets Maker and Multi-Agent Studio
echo   Vladimir Chopine (Geekatplay Studio)
echo   https://github.com/GeekatplayStudio/LoraMaker.git
echo ==============================================================
echo.

if exist "venv\Scripts\activate.bat" goto :use_venv

echo [WARN] Virtual environment 'venv' was not found.
goto :check_packages

:use_venv
call venv\Scripts\activate.bat
echo [INFO] Activated virtual environment (venv).

:check_packages
python -c "import uvicorn, fastapi" >nul 2>&1
if %errorlevel% equ 0 goto :launch

echo [WARN] Required studio packages (FastAPI, Uvicorn) are not installed!
echo Please run install.bat first to configure your environment.
echo.
set /p RUN_SETUP="Would you like to run install.bat now? (Y/N) [default: Y]: "
if /i "%RUN_SETUP%"=="" set RUN_SETUP=Y
if /i "%RUN_SETUP%"=="Y" (
    call install.bat
    exit /b 0
)
echo Attempting to launch anyway...

:launch
echo [INFO] Launching studio backend and opening browser at http://127.0.0.1:7860 ...
echo [INFO] Press CTRL+C in this console to stop the server.
echo.

python run_app.py %*

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application exited with error code %errorlevel%.
    echo Check the terminal output above or the logs in the console.
    pause
)
