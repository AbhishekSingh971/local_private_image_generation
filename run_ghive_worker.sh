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

# Download only the Diffusers files used by this app. The upstream SD 1.5
# repository also contains many duplicate .bin/.ckpt/fp16 variants totalling
# tens of GB; downloading the entire repository wastes time and disk space.
python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

model_dir = Path("models/stable-diffusion-v1-5")
required = [
    "model_index.json",
    "feature_extractor/preprocessor_config.json",
    "scheduler/scheduler_config.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
allowed = required.copy()
keep_weights = {
    "text_encoder/model.safetensors",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
}

# Remove large files left by an earlier unfiltered/interrupted snapshot.
# Keep only the three standard SafeTensors weights selected below.
if model_dir.exists():
    for path in model_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(model_dir).as_posix()
        if path.suffix in {".bin", ".ckpt", ".safetensors"} and rel not in keep_weights:
            path.unlink(missing_ok=True)

if not all((model_dir / name).is_file() for name in required):
    print("Downloading only the required Stable Diffusion 1.5 Diffusers files...")
    snapshot_download(
        repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
        local_dir=str(model_dir),
        allow_patterns=allowed,
        token=os.getenv("HF_TOKEN") or None,
    )
else:
    print("Stable Diffusion 1.5 model files are already complete.")

missing = [name for name in required if not (model_dir / name).is_file()]
if missing:
    raise SystemExit("Stable Diffusion model is incomplete: " + ", ".join(missing))
print("Stable Diffusion 1.5 model files verified.")

motion_dir = Path("models/animatediff-motion-adapter-v1-5-2")
motion_required = ["config.json", "diffusion_pytorch_model.safetensors"]
if motion_dir.exists():
    for path in motion_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(motion_dir).as_posix()
        if (
            path.suffix in {".bin", ".ckpt", ".safetensors"}
            and rel not in {"diffusion_pytorch_model.safetensors"}
        ):
            path.unlink(missing_ok=True)

if not all((motion_dir / name).is_file() for name in motion_required):
    print("Downloading the required AnimateDiff motion adapter files...")
    snapshot_download(
        repo_id="guoyww/animatediff-motion-adapter-v1-5-2",
        local_dir=str(motion_dir),
        allow_patterns=motion_required,
        token=os.getenv("HF_TOKEN") or None,
    )
else:
    print("AnimateDiff motion adapter files are already complete.")

missing = [name for name in motion_required if not (motion_dir / name).is_file()]
if missing:
    raise SystemExit("AnimateDiff adapter is incomplete: " + ", ".join(missing))
print("AnimateDiff motion adapter files verified.")
PY

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
