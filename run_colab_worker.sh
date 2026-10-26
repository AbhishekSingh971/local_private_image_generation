#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Colab provides the CUDA-enabled PyTorch runtime. Install app dependencies
# without replacing Colab's preinstalled CUDA-enabled torch build.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [[ -z "${REMOTE_API_TOKEN:-}" ]]; then
  echo "Set REMOTE_API_TOKEN in the Colab environment before starting the worker." >&2
  exit 1
fi

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("Colab has no CUDA GPU. In Runtime > Change runtime type, select a GPU accelerator, then reconnect.")
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))
PY

# Download model weights onto the current Colab runtime if missing.
# Runtime storage is ephemeral; downloads may need repeating after a reset.
if [[ ! -f models/stable-diffusion-v1-5/model_index.json ]]; then
  python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=os.getenv("SD_MODEL_ID") or "stable-diffusion-v1-5/stable-diffusion-v1-5",
    local_dir="models/stable-diffusion-v1-5",
    token=os.getenv("HF_TOKEN") or None,
)
PY
fi

if [[ ! -f models/animatediff-motion-adapter-v1-5-2/config.json ]]; then
  python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="guoyww/animatediff-motion-adapter-v1-5-2",
    local_dir="models/animatediff-motion-adapter-v1-5-2",
    token=os.getenv("HF_TOKEN") or None,
)
PY
fi

exec python gpu_worker.py
