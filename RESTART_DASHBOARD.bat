@echo off
echo ======================================================================
echo Restarting Dashboard
echo ======================================================================
echo.
echo Stopping any running Python processes...
taskkill /F /IM python.exe 2>nul
timeout /t 2 /nobreak >nul

echo.
echo Starting Enhanced Dashboard...
echo.

cd /d "%~dp0"
python enhanced_dashboard.py

pause
