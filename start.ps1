# Geekatplay LoRA Maker - PowerShell Launcher
# Geekatplay Studio - Vladimir Chopine
# Repository: https://github.com/GeekatplayStudio/LoraMaker.git

[CmdletBinding()]
param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 7860,
    [switch]$NoBrowser,
    [switch]$Reload,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location -Path $PSScriptRoot

Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "  ⚡ Geekatplay LoRA Maker Studio v2.0.0" -ForegroundColor Yellow
Write-Host "  🎨 Universal AI LoRA Assets Maker & Multi-Agent Creative Studio" -ForegroundColor White
Write-Host "  👤 Vladimir Chopine (Geekatplay Studio)" -ForegroundColor Gray
Write-Host "  ⭐ https://github.com/GeekatplayStudio/LoraMaker.git" -ForegroundColor Gray
Write-Host "==============================================================" -ForegroundColor Cyan

# Check for venv
$venvPath = Join-Path $PSScriptRoot "venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if (Test-Path $venvPython) {
    Write-Host "[INFO] Using virtual environment python ($venvPython)" -ForegroundColor Green
    $activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        & $activateScript
    }
    $py = $venvPython
} else {
    Write-Host "[WARN] Virtual environment 'venv' not found." -ForegroundColor Yellow
    $py = "python"
}

Write-Host "[INFO] Launching studio backend and opening browser at http://$HostAddress`:$Port ..." -ForegroundColor Cyan
Write-Host "[INFO] Press CTRL+C in this console window to gracefully stop the server.`n" -ForegroundColor Gray

$argsList = @("run_app.py", "--host", $HostAddress, "--port", $Port)
if ($NoBrowser) { $argsList += "--no-browser" }
if ($Reload) { $argsList += "--reload" }
if ($ExtraArgs) { $argsList += $ExtraArgs }

& $py $argsList
