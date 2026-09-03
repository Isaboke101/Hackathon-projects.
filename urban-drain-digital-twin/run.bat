@echo off
REM ===========================================================================
REM  Urban Drain Digital Twin - start the dashboard
REM
REM  Run setup.bat once first. After that, this is the only file you need.
REM ===========================================================================

if not exist ".venv\" (
    echo  [X] Not set up yet. Run setup.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo.
echo  ============================================
echo   URBAN DRAIN DIGITAL TWIN
echo  ============================================
echo.
echo   Dashboard : http://127.0.0.1:8000
echo   API docs  : http://127.0.0.1:8000/docs
echo.
echo   Press CTRL+C to stop.
echo.

REM Open the browser after a short pause, so the server is up first.
start "" /b cmd /c "timeout /t 3 >nul && start http://127.0.0.1:8000"

python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
