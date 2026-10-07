@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
set "ELECTRON_DIR=%PROJECT_ROOT%electron"

if not exist "%ELECTRON_DIR%\node_modules" (
    echo Electron dependencies are missing.
    echo Run "npm install" in the electron folder first.
    exit /b 1
)
if not exist "%PROJECT_ROOT%\node_modules" (
    echo UI dependencies are missing.
    echo Run "npm install" in the project root first.
    exit /b 1
)

echo Starting Kriya Sahay desktop app...
echo Electron will supervise the single Python pipeline process.
echo Camera and serial devices are discovered from the running backend.
echo.
pushd "%ELECTRON_DIR%"
call npm start
set "APP_EXIT=%ERRORLEVEL%"
popd
exit /b %APP_EXIT%
