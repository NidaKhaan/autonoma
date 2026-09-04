@echo off
title Autonoma — Self-Driving AI System
color 0A

echo ============================================
echo   AUTONOMA — Vision-Based Autonomous AI
echo ============================================
echo.

:: Change to project directory (wherever launch.bat lives)
cd /d "%~dp0"

:: Activate virtual environment
echo [1/3] Activating Python environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Could not activate venv. Run setup again.
    pause
    exit /b 1
)

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in venv.
    pause
    exit /b 1
)

echo [2/3] Starting Unity project...

set UNITY_PROJECT=D:\autonoma\unity\Autonoma
set UNITY_EXE=C:\Program Files\Unity\Hub\Editor\6000.4.8f1\Editor\Unity.exe

if exist "%UNITY_EXE%" (
    start "" "%UNITY_EXE%" -projectPath "%UNITY_PROJECT%"
    echo Unity launching... waiting 8 seconds for it to load.
    timeout /t 8 /nobreak >nul
) else (
    echo [WARN] Unity.exe not found at expected path.
    echo [WARN] Please open Unity manually and press Enter here to continue.
    pause
)

echo [3/3] Starting Autonoma AI Brain...
echo.
echo ============================================
echo  Dashboard will open in a separate window.
echo  Unity must be in Play mode for connection.
echo  Press Ctrl+C here to stop the AI brain.
echo ============================================
echo.

python main.py

echo.
echo [Autonoma] AI Brain stopped.
pause