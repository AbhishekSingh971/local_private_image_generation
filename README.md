# Local Private Image AI (Windows, CPU)

A minimal Flask UI for local image-to-image generation. It accepts an input image and prompt, runs Stable Diffusion 1.5 img2img on CPU, and saves results locally.

## Requirements

- Windows 10/11, 64-bit
- Python 3.11 x64
- 8 GB RAM minimum (16 GB is strongly preferred)
- Several GB of free SSD space for Python packages and model files
- Internet only for initial dependency/model downloads

## Install

1. Review the model card and license: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
2. Double-click `install_windows.bat`.
3. Double-click `download_model.bat` while online. If the model host requires access approval or a token, follow the model card's instructions.
4. Double-click `run_local.bat`.
5. Open `http://127.0.0.1:5000` in your browser.

The app binds to localhost only. Do not change it to `0.0.0.0` if you want to keep it local.

## Offline operation

After dependencies and model files are fully downloaded, `run_local.bat` uses the local model directory. Do not delete the `.venv` or `models/stable-diffusion-v1-5` folder. No external inference API is used. The app disables Hugging Face telemetry environment variables and asks Diffusers to load from local files only.

## 8 GB RAM caution

CPU inference is slow. Close browsers and other memory-heavy apps before generating. Begin with 5–10 steps and a small image. If the process fails due to memory, this model may not fit comfortably on your PC. Do not assume a model can run quickly merely because it fits on disk.

## Data locations

- Uploaded images are processed in memory; this starter does not persist the original upload.
- Generated results: `data/outputs/`
- Model files: `models/stable-diffusion-v1-5/`

This starter has no authentication because it binds to localhost. Do not expose it to the internet or local network without adding access controls and hardening.

## Troubleshooting

- `py -3.11` not found: install Python 3.11 x64 from https://www.python.org/downloads/windows/
- Model access/download error: check model card terms, repository availability, and any access requirements.
- Out of memory: close other apps, choose 5 steps, or use a smaller model / machine with more RAM.
- First generation may take a long time while CPU kernels warm up.
