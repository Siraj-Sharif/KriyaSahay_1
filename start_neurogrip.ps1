#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Launches the Kriya Sahay desktop application.
.DESCRIPTION
    Electron owns startup, readiness checks and shutdown of the one Python backend.
    This script intentionally does not start a separate Python process or assume a
    camera index / serial port.
#>

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$ElectronDir = Join-Path $ProjectRoot "electron"

if (-not (Test-Path (Join-Path $ElectronDir "node_modules"))) {
    throw "Electron dependencies are missing. Run npm install in '$ElectronDir'."
}
if (-not (Test-Path (Join-Path $ProjectRoot "node_modules"))) {
    throw "UI dependencies are missing. Run npm install in '$ProjectRoot'."
}

Write-Host "Starting Kriya Sahay desktop app..." -ForegroundColor Cyan
Write-Host "Electron will supervise the single Python pipeline process." -ForegroundColor Gray
Write-Host "Camera and serial devices are discovered from the running backend." -ForegroundColor Gray

Push-Location $ElectronDir
try {
    & npm start
    if ($LASTEXITCODE -ne 0) {
        throw "Kriya Sahay exited with code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
