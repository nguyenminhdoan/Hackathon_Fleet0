@echo off
echo ======================================================================
echo Starting Enhanced Dashboard
echo ======================================================================
echo.
echo Dashboard will be available at: http://localhost:8000/dashboard
echo Press Ctrl+C to stop
echo.
echo ======================================================================

cd /d "%~dp0"
python enhanced_dashboard.py

pause
