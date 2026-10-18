@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
  echo Run install_windows.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
if not exist "models\stable-diffusion-v1-5\model_index.json" (
  echo Model files were not found. Run download_model.bat while online.
  pause
  exit /b 1
)
python app.py
pause
