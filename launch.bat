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
echo Activating Python environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Could not activate venv. Run setup again.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Open the Unity project manually in Unity Hub
echo  and press Play before continuing.
echo  Dashboard will open in a separate window.
echo  Press Ctrl+C here to stop the AI brain.
echo ============================================
echo.

python main.py

echo.
echo [Autonoma] AI Brain stopped.
pause