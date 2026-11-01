#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
: "${REMOTE_API_TOKEN:?Set REMOTE_API_TOKEN to a long random secret first}"
PORT="${GPU_WORKER_PORT:-8000}"
export GPU_WORKER_PORT="$PORT"

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable. Choose GHive's PyTorch + Jupyter workload with an NVIDIA GPU.")
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))
PY

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is no longer available after dependency installation; restore the GHive CUDA PyTorch environment.")
print("CUDA verified:", torch.cuda.get_device_name(0))
PY

if [[ ! -f models/stable-diffusion-v1-5/model_index.json ]]; then
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download("stable-diffusion-v1-5/stable-diffusion-v1-5", local_dir="models/stable-diffusion-v1-5", token=os.getenv("HF_TOKEN") or None)
PY
fi
if [[ ! -f models/animatediff-motion-adapter-v1-5-2/config.json ]]; then
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download("guoyww/animatediff-motion-adapter-v1-5-2", local_dir="models/animatediff-motion-adapter-v1-5-2", token=os.getenv("HF_TOKEN") or None)
PY
fi

BIN_DIR="$HOME/.local/bin"
mkdir -p "$BIN_DIR"
if [[ ! -x "$BIN_DIR/cloudflared" ]]; then
  case "$(uname -m)" in
    x86_64|amd64) ASSET="cloudflared-linux-amd64" ;;
    aarch64|arm64) ASSET="cloudflared-linux-arm64" ;;
    *) echo "Unsupported cloudflared architecture: $(uname -m)" >&2; exit 1 ;;
  esac
  curl -fL "https://github.com/cloudflare/cloudflared/releases/latest/download/$ASSET" -o "$BIN_DIR/cloudflared"
  chmod +x "$BIN_DIR/cloudflared"
fi
WORKER_LOG="$PWD/ghive_worker.log"
python gpu_worker.py >"$WORKER_LOG" 2>&1 &
WORKER_PID=$!
cleanup(){ kill "$WORKER_PID" 2>/dev/null || true; wait "$WORKER_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
for _ in $(seq 1 60); do
  kill -0 "$WORKER_PID" 2>/dev/null || { cat "$WORKER_LOG" >&2; exit 1; }
  curl -fsS -H "Authorization: Bearer $REMOTE_API_TOKEN" "http://127.0.0.1:$PORT/api/remote/health" >/dev/null && break
  sleep 2
done
echo "Worker ready on port $PORT. Starting outbound Cloudflare Quick Tunnel."
echo "Copy the https://*.trycloudflare.com URL into your Windows .env as GPU_WORKER_URL."
echo "GHive does not provide public instance endpoints; this tunnel is a best-effort workaround."
echo "Keep this terminal open. Stop the GHive rental in its dashboard when finished."
"$BIN_DIR/cloudflared" tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate
