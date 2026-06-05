@echo off
title JARVIS
cd /d "%~dp0"

echo.
echo  ==========================================
echo    J.A.R.V.I.S  ^|  Starting up...
echo  ==========================================
echo.

:: Open the browser after a short delay (runs in parallel)
start "" cmd /c "timeout /t 4 /nobreak >nul && start http://localhost:5000"

:: Start the server (this keeps the window alive)
python app.py

:: If server exits, pause so the user can see any error message
echo.
echo  JARVIS has stopped.
pause
