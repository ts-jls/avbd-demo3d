@echo off
setlocal
title AVBD Launcher
pushd "%~dp0"

rem Double-click to launch the AVBD demo: physics server + three.js viewer.
rem Optional argument: scene name, e.g.  launch-viewer.cmd "Sphere Pour 5000 on Cylinders"

set EXE=build-webgpu-dawn-nmake3\avbd_headless_server.exe
if not exist "%EXE%" (
    echo Headless server not built: %EXE%
    echo Build it first from a VS x64 prompt: cmake --build build-webgpu-dawn-nmake3
    pause
    exit /b 1
)

rem Leftover server instances (crashed sessions, benchmarks) hold the bridge
rem port and would push the new server off 8765; clear them out.
tasklist /fi "imagename eq avbd_headless_server.exe" | find /i "avbd_headless_server.exe" >nul
if not errorlevel 1 (
    echo Stopping leftover avbd_headless_server.exe instances...
    taskkill /f /im avbd_headless_server.exe >nul
)

if not exist "viewer\node_modules" (
    echo Installing viewer dependencies...
    pushd viewer
    call npm install
    if errorlevel 1 (
        popd
        pause
        exit /b 1
    )
    popd
)

set SCENE=%~1

echo Starting physics server (webgpu-avbd)...
if defined SCENE (
    start "AVBD server" cmd /k "cd /d "%~dp0viewer" && npm run server:gpu -- --scene "%SCENE%""
) else (
    start "AVBD server" cmd /k "cd /d "%~dp0viewer" && npm run server:gpu"
)

echo Starting viewer dev server...
start "AVBD viewer" cmd /k "cd /d "%~dp0viewer" && npm run dev"

echo Opening browser at http://127.0.0.1:5173/ ...
timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:5173/"

echo.
echo Close the "AVBD server" and "AVBD viewer" windows to stop everything.
timeout /t 5 >nul
popd
endlocal
