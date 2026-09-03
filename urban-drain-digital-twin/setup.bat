@echo off
setlocal enabledelayedexpansion
REM ===========================================================================
REM  Urban Drain Digital Twin - one-time setup for Windows
REM
REM  Run from PowerShell with:  .\setup.bat      (or just double-click it)
REM
REM  Creates a virtual environment, installs dependencies, builds the drainage
REM  network, generates training data and trains the models. 3-5 minutes.
REM ===========================================================================

echo.
echo  ============================================
echo   URBAN DRAIN DIGITAL TWIN - SETUP
echo  ============================================
echo.

REM ---------------------------------------------------------------------------
REM  Step 1 - find a Python that has prebuilt packages available.
REM
REM  This is the step that bit us. Scientific packages (numpy, pandas,
REM  scikit-learn) ship as precompiled "wheels" built for specific Python
REM  versions. On a Python that is too new, no wheel exists yet, so pip tries to
REM  compile from C source and fails unless you have Microsoft C++ Build Tools.
REM
REM  So: prefer a Python in the range that definitely has wheels.
REM ---------------------------------------------------------------------------
set PYEXE=

REM Try the Windows Python launcher for a known-good version, newest first.
for %%V in (3.13 3.12 3.11 3.10) do (
    if "!PYEXE!"=="" (
        py -%%V -c "import sys" >nul 2>&1
        if !errorlevel! equ 0 (
            set PYEXE=py -%%V
            echo  [1/5] Using Python %%V via the py launcher.
        )
    )
)

REM Fall back to whatever "python" is, if it is in a supported range.
if "!PYEXE!"=="" (
    python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>&1
    if !errorlevel! equ 0 (
        set PYEXE=python
        for /f "tokens=2" %%v in ('python --version 2^>^&1') do echo  [1/5] Using Python %%v.
    )
)

REM Nothing suitable found - explain exactly what to do.
if "!PYEXE!"=="" (
    echo.
    echo  [X] No suitable Python found.
    echo.
    python --version 2>nul
    if errorlevel 1 (
        echo      Python is not installed, or not on your PATH.
    ) else (
        echo      Your Python is outside the range 3.10 - 3.13.
        echo.
        echo      The scientific packages this project needs ^(numpy, pandas,
        echo      scikit-learn^) do not yet publish prebuilt files for it, so pip
        echo      would try to compile them from source and fail.
    )
    echo.
    echo      FIX: install Python 3.12 from
    echo           https://www.python.org/downloads/release/python-3128/
    echo           Tick "Add Python to PATH" during installation.
    echo           Then run this script again - it will find 3.12 automatically.
    echo.
    pause
    exit /b 1
)

REM ---------------------------------------------------------------------------
REM  Step 2 - virtual environment.
REM  If one already exists but was built with the wrong Python, rebuild it.
REM ---------------------------------------------------------------------------
if exist ".venv\" (
    .venv\Scripts\python.exe -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>&1
    if errorlevel 1 (
        echo  [2/5] Existing environment uses an unsupported Python - rebuilding...
        rmdir /s /q .venv
    ) else (
        echo  [2/5] Virtual environment already exists.
    )
)

if not exist ".venv\" (
    echo  [2/5] Creating virtual environment...
    %PYEXE% -m venv .venv
    if errorlevel 1 (
        echo  [X] Could not create the virtual environment.
        pause
        exit /b 1
    )
)

call .venv\Scripts\activate.bat

REM ---------------------------------------------------------------------------
REM  Step 3 - dependencies.
REM  Upgrading pip first matters: old pip versions do not recognise newer wheel
REM  formats and will silently fall back to a source build.
REM ---------------------------------------------------------------------------
echo  [3/5] Installing dependencies (this is the slow part)...
python -m pip install --upgrade pip setuptools wheel --quiet
REM --only-binary=:all: forbids source builds outright. If a prebuilt package
REM is genuinely unavailable we get one clear line instead of a 20-line C
REM compiler error that looks like a code problem but is not.
python -m pip install --only-binary=:all: -r requirements.txt

if errorlevel 1 (
    echo.
    echo  [X] Installation failed.
    echo.
    echo      Scroll up and read the FIRST error, not the last one.
    echo.
    echo      If you see "Unknown compiler" or "Microsoft Visual C++ 14.0
    echo      is required", pip tried to build a package from source because
    echo      no prebuilt version exists for your Python. Install Python 3.12
    echo      and run this script again:
    echo           https://www.python.org/downloads/release/python-3128/
    echo.
    echo      If you see timeouts or "Could not find a version", it really is
    echo      the network - check your connection and retry.
    echo.
    pause
    exit /b 1
)

REM ---------------------------------------------------------------------------
REM  Step 4 - build the network and generate the training data.
REM ---------------------------------------------------------------------------
echo  [4/5] Building drainage network and generating training data...
python -m backend.network
if errorlevel 1 goto :buildfail
python -m backend.dataset
if errorlevel 1 goto :buildfail

REM ---------------------------------------------------------------------------
REM  Step 5 - train the models.
REM ---------------------------------------------------------------------------
echo  [5/5] Training the AI models...
python -m backend.train
if errorlevel 1 goto :buildfail

echo.
echo  ============================================
echo   SETUP COMPLETE
echo  ============================================
echo.
echo   Now run:  run.bat
echo   Then open http://127.0.0.1:8000
echo.
pause
exit /b 0

:buildfail
echo.
echo  [X] A build step failed. Check the error above.
echo      Make sure you are running this from the project folder - the one
echo      that contains the "backend" folder.
echo.
pause
exit /b 1
