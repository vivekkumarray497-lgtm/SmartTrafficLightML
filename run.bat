@echo off
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.10+ and try again.
  pause
  exit /b 1
)
if not exist venv\Scripts\python.exe (
  echo Creating virtual environment...
  python -m venv venv
  if errorlevel 1 goto :error
  echo Installing dependencies...
  venv\Scripts\python.exe -m pip install --upgrade pip
  venv\Scripts\python.exe -m pip install -r requirements.txt
  if errorlevel 1 goto :error
)
venv\Scripts\python.exe run.py
pause
exit /b 0
:error
echo.
echo Setup failed. Check your internet connection and Python installation.
pause
exit /b 1
