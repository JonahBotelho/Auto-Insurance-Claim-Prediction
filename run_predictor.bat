@echo off
setlocal
cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo Creating local virtual environment in .venv...
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 -m venv .venv
    ) else (
        python -m venv .venv
    )
    if errorlevel 1 (
        echo Failed to create .venv. Ensure Python 3.10 or newer is installed and on PATH.
        pause
        exit /b 1
    )
)

"%VENV_PY%" -c "import lightgbm, catboost, sklearn, pandas, scipy, joblib, threadpoolctl" >nul 2>nul
if errorlevel 1 (
    echo Installing inference dependencies from src\predict_requirements.txt...
    "%VENV_PY%" -m pip install -r src\predict_requirements.txt
    if errorlevel 1 (
        echo Failed to install dependencies from src\predict_requirements.txt.
        pause
        exit /b 1
    )
)

"%VENV_PY%" -m src.predict_ui %*
if errorlevel 1 (
    echo.
    echo Local predictor exited with an error.
    pause
    exit /b 1
)
