# Geekatplay LoRA Maker - PowerShell Automated Installer
# Geekatplay Studio - Vladimir Chopine
# Repository: https://github.com/GeekatplayStudio/LoraMaker.git

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "  ⚡ Geekatplay LoRA Maker - Automated Installation" -ForegroundColor Yellow
Write-Host "  🎨 Geekatplay Studio - Vladimir Chopine" -ForegroundColor White
Write-Host "  ⭐ https://github.com/GeekatplayStudio/LoraMaker.git" -ForegroundColor Gray
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check Python
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[ERROR] Python was not found in your system PATH!" -ForegroundColor Red
    Write-Host "Please install Python 3.10+ from https://www.python.org/downloads/ and check 'Add Python to PATH'." -ForegroundColor Yellow
    exit 1
}

$pyVersion = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
Write-Host "[INFO] Detected Python $pyVersion" -ForegroundColor Green

# Verify version >= 3.10
$isCompatible = & python -c "import sys; print(sys.version_info >= (3, 10))"
if ($isCompatible.Trim() -ne "True") {
    Write-Host "[ERROR] Python 3.10 or higher is required. Found Python $pyVersion." -ForegroundColor Red
    exit 1
}

# 2. Virtual Environment
$venvPath = Join-Path $PSScriptRoot "venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "[INFO] Creating virtual environment in .\venv ..." -ForegroundColor Cyan
    & python -m venv venv
    Write-Host "[INFO] Virtual environment created successfully." -ForegroundColor Green
} else {
    Write-Host "[INFO] Existing virtual environment found in .\venv." -ForegroundColor Cyan
}

# 3. Activate Virtual Environment
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
    Write-Host "[INFO] Virtual environment activated." -ForegroundColor Green
} else {
    Write-Host "[WARN] Could not find Activate.ps1, proceeding with venv python executable..." -ForegroundColor Yellow
}

$venvPython = Join-Path $venvPath "Scripts\python.exe"
if (-not (Test-Path $venvPython)) { $venvPython = "python" }

# 4. Upgrade pip and wheel
Write-Host "[INFO] Upgrading pip, setuptools, and wheel..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip setuptools wheel --quiet

# 5. Check NVIDIA GPU
Write-Host "`n[INFO] Detecting GPU and compute capabilities..." -ForegroundColor Cyan
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    Write-Host "[INFO] NVIDIA GPU hardware detected via nvidia-smi." -ForegroundColor Green
    Write-Host "[INFO] Installing PyTorch with CUDA 12.4 compute support..." -ForegroundColor Cyan
    & $venvPython -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
} else {
    Write-Host "[WARN] nvidia-smi not detected. Installing standard PyTorch..." -ForegroundColor Yellow
    & $venvPython -m pip install torch torchvision torchaudio
}

# 6. Install dependencies
Write-Host "`n[INFO] Installing studio packages from requirements.txt..." -ForegroundColor Cyan
& $venvPython -m pip install -r requirements.txt

# 7. Verification
Write-Host "`n[INFO] Verifying environment health..." -ForegroundColor Cyan
& $venvPython -c "import torch; print(f'  [OK] PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''CPU''}')"
& $venvPython -c "import diffusers, transformers, cv2, fastapi, uvicorn, safetensors, ollama; print('  [OK] Diffusers, Transformers, OpenCV, Ollama, & FastAPI loaded cleanly!')"

Write-Host "`n==============================================================" -ForegroundColor Green
Write-Host "  ✨ Geekatplay LoRA Maker installation completed successfully!" -ForegroundColor Green
Write-Host "  Launch anytime with: .\start.bat or .\start.ps1" -ForegroundColor Yellow
Write-Host "==============================================================" -ForegroundColor Green
Write-Host ""
