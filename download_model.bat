@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
  echo Run install_windows.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
echo.
echo This downloads the Stable Diffusion 1.5 model files for local use.
echo Review and accept the model terms on its Hugging Face model page first:
echo https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
echo If access requires a Hugging Face account/token, sign in using huggingface-cli login.
echo.
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='stable-diffusion-v1-5/stable-diffusion-v1-5', local_dir='models/stable-diffusion-v1-5')"
if errorlevel 1 (
  echo Model download failed. Review the model page, license/access requirements, and internet connection.
  pause
  exit /b 1
)
echo Model downloaded. After setup, inference uses the local model directory.
pause
