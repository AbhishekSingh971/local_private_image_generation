# Google Colab GPU worker setup

This lets the existing Private Face Studio app use a Google Colab GPU through the same authenticated worker API previously used by RunPod. The app remains on your computer (or your own web host); only image/video generation and LoRA training requests are sent to the Colab worker.

## Important limitations

- A free Colab runtime is temporary and may disconnect, time out, or reclaim its GPU. It is not a 24/7 production server.
- The public HTTPS tunnel URL must be reachable by your app and the shared token must be kept secret. A stable hostname does not keep a Colab runtime alive.
- Model weights, files and LoRAs stored on a Colab VM may disappear when that VM is reset. Use Google Drive for reusable model cache if appropriate, but avoid putting a public tunnel or token in a notebook shared with others.
- Colab GPU availability and limits vary by account and time. Video generation and LoRA training are heavier than image generation and may exceed the available VRAM/runtime.
- Requests and selected face photos are sent from your local app to the cloud worker when the cloud GPU option is selected. Do not upload data you do not have permission to process.

## 1. Create a Colab GPU runtime

1. Open https://colab.research.google.com/ and create a notebook.
2. Select **Runtime → Change runtime type → T4 GPU** (or another GPU Colab offers).
3. Run this diagnostic cell. Do not start the worker unless CUDA is `True`.

```python
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
```

If it reports no CUDA, change the runtime type to a GPU and reconnect. Do not pip-install CPU-only PyTorch over Colab's CUDA build.

## 2. Get the code and install dependencies

Run these cells in order. A Git branch with this setup is available; replace the branch name if you choose to use another branch.

```bash
%cd /content
!git clone --branch feature/colab-gpu-worker https://github.com/AbhishekSingh971/local_private_image_generation.git
%cd /content/local_private_image_generation
```

Create a high-entropy token in a separate cell. Do not commit it or print it in notebook output.

```python
import secrets
REMOTE_API_TOKEN = secrets.token_urlsafe(40)
print("Token generated in this Colab session. Copy it securely to your local .env; do not share this notebook.")
```

Keep the token variable in notebook memory; avoid writing it into the repository. Set it for the worker's child process:

```python
import os
os.environ["REMOTE_API_TOKEN"] = REMOTE_API_TOKEN
os.environ["GPU_WORKER_PORT"] = "8000"
```

Run the launcher in a dedicated cell:

```bash
!bash run_colab_worker.sh
```

That command occupies the cell while the worker is running. Keep the cell running. The script installs the app dependencies, checks CUDA, downloads Stable Diffusion 1.5 and AnimateDiff weights if missing, and starts the authenticated worker. First setup may take a while and downloads require enough disk space. If the Hugging Face repository requires accepted model terms, accept them on the model page and supply a token through a private Colab secret/environment variable.

**Do not run a second copy of the worker.** Stop the worker cell with the stop button before restarting it. A cell that is still running must be stopped before you start another worker in the same runtime.

## 3. Give the worker a public HTTPS endpoint

The worker listens on port 8000 inside Colab. Your local app cannot connect to `localhost:8000` on Colab, so it needs an HTTPS tunnel.

### Option A — ngrok Dev Domain (simplest stable hostname)

1. Create a free ngrok account and reserve/use the free static Dev Domain offered for your account.
2. Get an ngrok authtoken from its dashboard. Keep it private.
3. Install ngrok in a separate Colab cell using the official ngrok Colab instructions, then configure the token without printing it.
4. Start an HTTP tunnel for port 8000 using that reserved Dev Domain. Keep this tunnel cell running too.
5. Copy the HTTPS URL ngrok displays. A reserved Dev Domain can keep the same hostname across restarts, but the Colab VM and worker must still be running.

Example command shape (use your own actual reserved domain; this is not a working URL):

```bash
ngrok http --domain=YOUR-RESERVED-DOMAIN.ngrok-free.app 8000
```

Use the current ngrok dashboard/docs for exact account-specific setup and domain name. Do not use an ephemeral URL if you require the same hostname on every restart.

### Option B — Cloudflare Tunnel

A quick tunnel can be useful for a temporary test, but its generated hostname changes when restarted. A fixed named tunnel typically requires a domain managed in your Cloudflare account, so it does not meet the no-domain constraint. Don't assume a stable hostname means the tunnel persists after Colab disconnects.

## 4. Configure the local app

In your local repository, copy `.env.example` to `.env` if you have not already done so. Keep your existing `DATABASE_URL` and other private settings. Set:

```dotenv
GPU_WORKER_URL=https://YOUR-RESERVED-DOMAIN.ngrok-free.app
GPU_WORKER_TOKEN=PASTE_THE_SAME_REMOTE_API_TOKEN_HERE

# Backward compatibility for older app versions. These can stay configured if using RunPod.
RUNPOD_WORKER_URL=
RUNPOD_WORKER_TOKEN=
```

Restart the local Flask app after changing environment variables. The app should report the remote worker as a cloud GPU when the current app code supports the generic worker configuration. Keep the URL and token out of Git and screenshots.

## 5. Verify connectivity and run a small test

The worker health route is authenticated. From a local shell, substitute the real URL and token without putting the token in source control:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" https://YOUR-RESERVED-DOMAIN.ngrok-free.app/api/remote/health
```

A healthy response includes `"status":"ok"`, `"cuda_available":true`, `gpu_name`, and `vram_gb`. HTTP 401 means the tokens don't match; HTTP 503 usually means the worker has no available CUDA GPU or the worker is not ready.

Then in Private Face Studio, turn on GPU acceleration and select the cloud GPU. Try a low-step image first. The local app saves the returned image/video and metadata to its configured storage. For LoRA training, a worker reset can discard model files unless you have backed them up or persisted them.

## 6. Stop safely and restart

- Stop the worker cell and tunnel cell when finished.
- The free Colab VM can disconnect at any time; do not rely on it for unattended background jobs.
- When restarting, verify CUDA, start the worker once, start the tunnel, and check the same reserved HTTPS URL.
- If the session is reset, install dependencies and download weights may need to run again. Save finished LoRA bundles and any important outputs to durable storage before the runtime ends.

## Architecture note

This deployment uses `gpu_worker.py` on Colab, the existing authenticated `/api/remote/*` API, and a tunnel to port 8000. It does not embed a GPU in the local app, keep the GPU warm, bypass Colab limits, or make the free runtime always-on.
