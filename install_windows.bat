@echo off
setlocal
cd /d "%~dp0"
echo Creating Python virtual environment...
py -3.11 -m venv .venv
if errorlevel 1 (
  echo Could not create venv. Install Python 3.11 x64 from https://www.python.org/downloads/windows/
  echo During installation, enable "Add python.exe to PATH".
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo Installing CPU-only PyTorch. This can take a while.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
  echo PyTorch install failed.
  pause
  exit /b 1
)
echo Installing app requirements...
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Requirements install failed.
  pause
  exit /b 1
)
echo.
echo Installation finished. Run download_model.bat next.
pause
