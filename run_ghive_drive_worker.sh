#!/usr/bin/env bash
# GHive worker using the user's Google Drive SD 1.5 model.
# Does not download SD 1.5 from Hugging Face and does not copy the full Drive folder.
set -Eeuo pipefail
cd "$(dirname "$0")"

: "${REMOTE_API_TOKEN:?Set REMOTE_API_TOKEN to a long random secret first}"
export GPU_WORKER_PORT="${GPU_WORKER_PORT:-8000}"
export SD_MODEL_DRIVE_REMOTE="${SD_MODEL_DRIVE_REMOTE:-gdrive:Local Ai/models/stable-diffusion-v1-5/vae}"
export SD_MODEL_CACHE_DIR="${SD_MODEL_CACHE_DIR:-$HOME/.cache/local_private_image_generation/sd15}"
export FACE_FINETUNE_DRIVE_REMOTE="${FACE_FINETUNE_DRIVE_REMOTE:-gdrive:Local Ai/models/face-finetunes}"

echo "Checking GHive CUDA..."
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable. Start a GHive NVIDIA GPU workload.")
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))
PY

echo "Preparing selective Google Drive model cache..."
export SD_MODEL_PATH="$SD_MODEL_CACHE_DIR"
python drive_model_cache.py

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
  if curl -fsS -H "Authorization: Bearer $REMOTE_API_TOKEN" "http://127.0.0.1:$GPU_WORKER_PORT/api/remote/health" >/dev/null; then
    break
  fi
  sleep 2
done

curl -fsS -H "Authorization: Bearer $REMOTE_API_TOKEN" "http://127.0.0.1:$GPU_WORKER_PORT/api/remote/health" || {
  cat "$WORKER_LOG" >&2
  echo "GPU worker health check failed." >&2
  exit 1
}
echo
echo "Worker ready. Starting Cloudflare Quick Tunnel."
echo "Copy the https://*.trycloudflare.com URL into your Windows .env as GPU_WORKER_URL."
echo "Keep this terminal open. Stop the GHive rental in its dashboard when finished."
"$BIN_DIR/cloudflared" tunnel --url "http://127.0.0.1:$GPU_WORKER_PORT" --no-autoupdate
