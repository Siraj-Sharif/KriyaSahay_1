#!/usr/bin/env pwsh
<#
.SYNOPSIS
    NeuroGrip Launcher - Starts Python CV backend + Electron frontend
.DESCRIPTION
    Single-command launcher for the complete NeuroGrip application stack.
    Starts Python CV backend with mock serial, then Electron desktop app.
.NOTES
    Current serial mode: mock (no ESP32 connected)
    To switch to COM9 later: change $SerialMode = "mock" to $SerialMode = "enabled" and $SerialPort = "COM9"
#>

param(
    [string]$CameraIndex = "0",
    [string]$SerialMode = "mock",       # mock | enabled | disabled
    [string]$SerialPort = "COM9",       # Used when SerialMode = "enabled"
    [int]$SerialBaud = 115200
)

$ProjectRoot = $PSScriptRoot
$VenvPython = "$ProjectRoot\.venv\Scripts\python.exe"
$ElectronBin = "$ProjectRoot\electron\node_modules\.bin\electron.cmd"
$ElectronDir = "$ProjectRoot\electron"

# Kill any existing processes on port 8765
function Stop-ExistingBridge {
    $listener = netstat -ano | Select-String "127.0.0.1:8765.*LISTENING"
    if ($listener) {
        $pid = ($listener -split '\s+')[-1]
        if ($pid -match '^\d+$') {
            Write-Host "Stopping existing process on port 8765 (PID: $pid)..."
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
            Start-Sleep 1
        }
    }
}

# Start Python CV Backend
function Start-PythonBackend {
    $args = @(
        "-m", "neurogrip",
        "--camera", $CameraIndex,
        "--serial", $SerialMode,
        "--no-gui"
    )
    if ($SerialMode -eq "enabled") {
        $args += "--serial-port", $SerialPort, "--baudrate", $SerialBaud
    }

    Write-Host "Starting Python CV backend..." -ForegroundColor Cyan
    Write-Host "  Command: $VenvPython $($args -join ' ')"

    $process = Start-Process -FilePath $VenvPython -ArgumentList $args -WorkingDirectory $ProjectRoot -PassThru -WindowStyle Hidden
    return $process
}

# Start Electron Frontend
function Start-ElectronFrontend {
    Write-Host "Preparing React UI for Electron..." -ForegroundColor Cyan

    $prepareScript = Join-Path $ElectronDir "scripts\prepare.cjs"

    if (-not (Test-Path $prepareScript)) {
        throw "Electron prepare script not found: $prepareScript"
    }

    # Build Vite UI and copy dist/index.html into electron/renderer/
    & node $prepareScript --build

    if ($LASTEXITCODE -ne 0) {
        throw "UI preparation failed with exit code $LASTEXITCODE"
    }

    $rendererIndex = Join-Path $ElectronDir "renderer\index.html"

    if (-not (Test-Path $rendererIndex)) {
        throw "Renderer was not created: $rendererIndex"
    }

    Write-Host "UI prepared successfully." -ForegroundColor Green
    Write-Host "  Renderer: $rendererIndex" -ForegroundColor White

    Write-Host "Starting Electron frontend..." -ForegroundColor Cyan
    Write-Host "  Command: $ElectronBin $ElectronDir"

    $process = Start-Process `
        -FilePath $ElectronBin `
        -ArgumentList $ElectronDir `
        -WorkingDirectory $ElectronDir `
        -PassThru

    return $process
}

# Main
Write-Host "=======================================" -ForegroundColor Green
Write-Host "       NeuroGrip Launcher v1.0         " -ForegroundColor Green
Write-Host "  Python CV + Electron + React Dash    " -ForegroundColor Green
Write-Host "=======================================" -ForegroundColor Green
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host ("  Camera:     {0}" -f $CameraIndex) -ForegroundColor White
Write-Host ("  Serial:     {0}" -f $SerialMode) -ForegroundColor White
if ($SerialMode -eq "enabled") {
    $portMsg = '  Port:       ' + $SerialPort + ' @ ' + $SerialBaud + ' baud'
    Write-Host $portMsg -ForegroundColor White
}
Write-Host ""

Stop-ExistingBridge

$pythonProcess = Start-PythonBackend
Write-Host ("Python backend started (PID: {0})" -f $pythonProcess.Id) -ForegroundColor Green

# Give Python time to initialize and bind to port 8765
Write-Host "Waiting for Python backend to initialize..." -ForegroundColor Yellow
Start-Sleep 4

$electronProcess = Start-ElectronFrontend
Write-Host ("Electron frontend started (PID: {0})" -f $electronProcess.Id) -ForegroundColor Green

Write-Host ""
Write-Host "=======================================" -ForegroundColor Green
Write-Host "NeuroGrip is running!" -ForegroundColor Green
Write-Host "  - Dashboard: Electron window" -ForegroundColor White
Write-Host ("  - Camera:     Index {0}" -f $CameraIndex) -ForegroundColor White
Write-Host ("  - Serial:     {0}" -f $SerialMode) -ForegroundColor White
Write-Host "  - TCP Bridge: 127.0.0.1:8765" -ForegroundColor White
Write-Host "=======================================" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop both processes..." -ForegroundColor Gray

# Wait for either process to exit
try {
    $pythonProcess | Wait-Process -ErrorAction SilentlyContinue
    $electronProcess | Wait-Process -ErrorAction SilentlyContinue
}
finally {
    Write-Host "Shutting down..." -ForegroundColor Yellow
    if (-not $pythonProcess.HasExited) { Stop-Process -Id $pythonProcess.Id -Force -ErrorAction SilentlyContinue }
    if (-not $electronProcess.HasExited) { Stop-Process -Id $electronProcess.Id -Force -ErrorAction SilentlyContinue }
    Write-Host "Done." -ForegroundColor Green
}