@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
  echo Project virtual-environment Python was not found:
  echo "%PYTHON%"
  echo Run install_windows.bat first.
  pause
  exit /b 1
)

echo Using project Python:
echo "%PYTHON%"

"%PYTHON%" -c "import flask; print('Flask:', flask.__file__)" >nul 2>nul
if errorlevel 1 (
  echo Flask is not installed in this project's virtual environment.
  echo Install dependencies using:
  echo "%PYTHON%" -m pip install -r "%~dp0requirements.txt"
  pause
  exit /b 1
)

if not exist "models\stable-diffusion-v1-5\model_index.json" (
  echo Model files were not found. Run download_model.bat while online.
  pause
  exit /b 1
)

"%PYTHON%" app.py
if errorlevel 1 (
  echo.
  echo The app stopped with an error. Review the message above.
  pause
  exit /b 1
)
pause
