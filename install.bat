@echo off
chcp 65001 >nul
setlocal
title "Geekatplay LoRA Maker - Setup and Installation"
cd /d "%~dp0"

echo ==============================================================
echo   Geekatplay LoRA Maker - Automated Installation
echo   Geekatplay Studio - Vladimir Chopine
echo   https://github.com/GeekatplayStudio/LoraMaker.git
echo ==============================================================
echo.

:: 1. Check Python installation
where python >nul 2>&1
if %errorlevel% equ 0 goto :check_py_version

echo [ERROR] Python was not found in your system PATH!
echo Please install Python 3.10, 3.11, 3.12 or 3.13 from:
echo https://www.python.org/downloads/
echo NOTE: Ensure you check "Add Python to PATH" during installation.
echo.
pause
exit /b 1

:check_py_version
python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if %errorlevel% equ 0 goto :py_ok

echo [ERROR] Python 3.10 or higher is required.
echo Found:
python --version
echo Please upgrade Python and try again.
echo.
pause
exit /b 1

:py_ok
for /f "tokens=*" %%i in ('python --version') do set PY_VER=%%i
echo [INFO] Detected %PY_VER%
python -c "import sys; exit(0 if sys.version_info < (3, 14) else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN] Python 3.14+ detected. PyTorch CUDA wheels and Kohya sd-scripts are
    echo [WARN] validated on Python 3.10 - 3.13. If package installs fail, install 3.12.
)

:: 2. Setup Virtual Environment
if exist "venv\Scripts\activate.bat" goto :venv_exists

echo [INFO] Creating Python virtual environment in .\venv ...
python -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment 'venv'.
    pause
    exit /b 1
)
echo [INFO] Virtual environment created successfully.
goto :activate_venv

:venv_exists
echo [INFO] Existing virtual environment found in .\venv.

:activate_venv
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

:: 3. Upgrade pip, wheel, setuptools
echo [INFO] Upgrading pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel --quiet

:: 4. Check for NVIDIA CUDA GPU
echo.
echo [INFO] Detecting GPU and compute capabilities...
where nvidia-smi >nul 2>&1
if %errorlevel% equ 0 goto :install_cuda_torch

echo [WARN] nvidia-smi not detected. Installing CPU PyTorch (training needs an NVIDIA GPU)...
pip install --upgrade torch torchvision
goto :install_reqs

:install_cuda_torch
echo [INFO] NVIDIA GPU hardware detected via nvidia-smi.
echo [INFO] Installing PyTorch with CUDA 12.6 compute support (driver 560+)...
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu126
if %errorlevel% neq 0 (
    echo [WARN] CUDA 12.6 wheel install failed; retrying with CUDA 12.8 build...
    pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128
)

:install_reqs
echo.
echo [INFO] Installing required studio packages from requirements.txt...
pip install --upgrade -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Encountered an error installing packages from requirements.txt.
    pause
    exit /b 1
)

:: 5. Validation Healthcheck
echo.
echo [INFO] Verifying installation and environment health...
python -c "import torch; print('  [OK] PyTorch:', torch.__version__, '| CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None (CPU)')"
python -c "import diffusers, transformers, peft, cv2, fastapi, uvicorn, safetensors, ollama; print('  [OK] Diffusers', diffusers.__version__, '| Transformers', transformers.__version__, '| PEFT', peft.__version__, '| OpenCV, Ollama, FastAPI loaded cleanly!')"
python -c "import bitsandbytes; print('  [OK] bitsandbytes', bitsandbytes.__version__, '(8-bit AdamW available)')" 2>nul || echo   [WARN] bitsandbytes not importable; trainer will use standard AdamW.
python -c "from app.services.training_service import TrainingService as T; r=[str(x) for x in T.script_roots() if (x/'sdxl_train_network.py').is_file()]; print('  [OK] Kohya sd-scripts:', r[0]) if r else print('  [INFO] Kohya sd-scripts not found; SDXL uses the built-in Diffusers/PEFT trainer (set KOHYA_ROOT to use Kohya).')" 2>nul

echo.
echo [INFO] To prove training really works on this machine, run:  verify_training.bat

echo.
echo ==============================================================
echo   Geekatplay LoRA Maker installation completed successfully!
echo   Launch anytime by running: start.bat
echo ==============================================================
echo.

set /p LAUNCH="Would you like to launch the Studio right now? (Y/N) [default: Y]: "
if /i "%LAUNCH%"=="" set LAUNCH=Y
if /i "%LAUNCH%"=="Y" (
    call start.bat
)
