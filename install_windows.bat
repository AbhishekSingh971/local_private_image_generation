@echo off
setlocal
cd /d "%~dp0"

echo Creating Python 3.11 virtual environment...
py -3.11 -m venv .venv
if errorlevel 1 (
  echo Could not create venv. Install Python 3.11 x64 from https://www.python.org/downloads/windows/
  echo During installation, enable "Add python.exe to PATH".
  pause
  exit /b 1
)

set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo Virtual-environment Python was not created:
  echo "%PYTHON%"
  pause
  exit /b 1
)

echo Upgrading pip...
"%PYTHON%" -m pip install --upgrade pip
if errorlevel 1 (
  echo pip upgrade failed.
  pause
  exit /b 1
)

echo Installing matching CPU-only PyTorch and TorchVision packages...
echo Pair: torch 2.14.1 + torchvision 0.29.1
"%PYTHON%" -m pip install --no-cache-dir torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
  echo PyTorch/TorchVision install failed.
  pause
  exit /b 1
)

echo Installing app requirements...
"%PYTHON%" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
  echo Requirements install failed.
  pause
  exit /b 1
)

echo Verifying PyTorch and TorchVision native operators...
"%PYTHON%" -c "import torch, torchvision; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); from torchvision.ops import nms; print('TorchVision native ops OK')"
if errorlevel 1 (
  echo.
  echo PyTorch/TorchVision verification failed.
  echo If Windows reports that an Application Control policy blocked a DLL, check Windows Code Integrity events.
  pause
  exit /b 1
)

echo.
echo Installation finished. Run download_model.bat next.
pause
