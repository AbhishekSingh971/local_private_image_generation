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
echo Downloading AnimateDiff motion-adapter weights for local video inference.
echo Review model details and license before continuing:
echo https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-2
echo CPU-only PyTorch cannot run video generation; NVIDIA CUDA is required.
echo.
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='guoyww/animatediff-motion-adapter-v1-5-2', local_dir='models/animatediff-motion-adapter-v1-5-2')"
if errorlevel 1 (
  echo Download failed. Review the model card, access requirements and internet connection.
  pause
  exit /b 1
)
echo Motion adapter downloaded. Video inference uses this local directory.
pause
