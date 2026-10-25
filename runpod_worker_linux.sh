#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ -f .env ]]; then
  set -a
  # This file is local-only and must never be committed.
  source .env
  set +a
fi

if [[ -z "${REMOTE_API_TOKEN:-}" ]]; then
  echo "Set REMOTE_API_TOKEN to a strong random secret before starting the worker." >&2
  exit 1
fi

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("This environment does not have CUDA enabled. Choose a CUDA-enabled NVIDIA GPU Pod.")
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))
PY

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [[ ! -f models/stable-diffusion-v1-5/model_index.json ]]; then
  python - <<'PY'
import os
from huggingface_hub import snapshot_download
token = os.getenv("HF_TOKEN") or None
snapshot_download(
    repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    local_dir="models/stable-diffusion-v1-5",
    token=token,
)
PY
fi

if [[ ! -f models/animatediff-motion-adapter-v1-5-2/config.json ]]; then
  python - <<'PY'
import os
from huggingface_hub import snapshot_download
token = os.getenv("HF_TOKEN") or None
snapshot_download(
    repo_id="guoyww/animatediff-motion-adapter-v1-5-2",
    local_dir="models/animatediff-motion-adapter-v1-5-2",
    token=token,
)
PY
fi

exec python gpu_worker.py
