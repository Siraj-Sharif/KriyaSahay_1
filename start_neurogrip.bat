@echo off
REM NeuroGrip Launcher - Starts Python CV backend + Electron frontend
REM Current serial mode: mock (no ESP32 connected)
REM To switch to COM9 later: change SERIAL_MODE=mock to SERIAL_MODE=enabled and SERIAL_PORT=COM9

set PROJECT_ROOT=D:\NeuroGrip_Project_New
set VENV_PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe
set ELECTRON_BIN=%PROJECT_ROOT%\electron\node_modules\.bin\electron.cmd
set ELECTRON_DIR=%PROJECT_ROOT%\electron

set CAMERA_INDEX=0
set SERIAL_MODE=mock
set SERIAL_PORT=COM9
set SERIAL_BAUD=115200

echo ╔═══════════════════════════════════════════════╗
echo ║         NeuroGrip Launcher v1.0              ║
echo ║   Python CV + Electron + React Dashboard     ║
echo ╚═══════════════════════════════════════════════╝
echo.
echo Configuration:
echo   Camera:     %CAMERA_INDEX%
echo   Serial:     %SERIAL_MODE%
if "%SERIAL_MODE%"=="enabled" echo   Port:       %SERIAL_PORT% @ %SERIAL_BAUD% baud
echo.

REM Kill any existing processes on port 8765
for /f "tokens=5" %%a in ('netstat -ano ^| findstr "127.0.0.1:8765.*LISTENING"') do (
    echo Stopping existing process on port 8765 (PID: %%a)...
    taskkill /F /PID %%a >nul 2>&1
    timeout /t 1 /nobreak >nul
)

echo Starting Python CV backend...
if "%SERIAL_MODE%"=="enabled" (
    start "NeuroGrip Python" /B %VENV_PYTHON% -m neurogrip --camera %CAMERA_INDEX% --serial %SERIAL_MODE% --serial-port %SERIAL_PORT% --baudrate %SERIAL_BAUD% --no-gui
) else (
    start "NeuroGrip Python" /B %VENV_PYTHON% -m neurogrip --camera %CAMERA_INDEX% --serial %SERIAL_MODE% --no-gui
)

echo Python backend started. Waiting for initialization...
timeout /t 5 /nobreak >nul

echo Starting Electron frontend...
start "NeuroGrip Electron" "" %ELECTRON_BIN% %ELECTRON_DIR%

echo.
echo ═══════════════════════════════════════════════
echo NeuroGrip is running!
echo   • Dashboard:  Electron window
echo   • Camera:     Index %CAMERA_INDEX%
echo   • Serial:     %SERIAL_MODE%
echo   • TCP Bridge: 127.0.0.1:8765
echo ═══════════════════════════════════════════════
echo.
echo Press Ctrl+C in the Python console to stop the backend.
pause