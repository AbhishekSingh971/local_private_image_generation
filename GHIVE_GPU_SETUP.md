# GHive GPU worker setup

1. Rent an NVIDIA GPU using GHive's **PyTorch + Jupyter** workload. Prefer 24 GB VRAM for face-LoRA training and video generation.
2. Open a JupyterLab terminal and clone the main branch:

   ```bash
   git clone --branch main https://github.com/AbhishekSingh971/local_private_image_generation.git
   cd local_private_image_generation
   ```

3. Create a worker secret and launch:

   ```bash
   export REMOTE_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
   export GPU_WORKER_PORT=8000
   bash run_ghive_worker.sh
   ```

4. Copy the HTTPS `trycloudflare.com` URL printed by cloudflared. Keep the terminal open.
5. In the Windows app's private `.env`, set `GPU_WORKER_URL` to that URL and `REMOTE_API_TOKEN` to the exact same secret. Restart `run_local.bat`.
6. Enable GPU acceleration, select Cloud GPU, and test a small image.

## Critical networking limitation

GHive currently provides JupyterLab and terminal access but does not provide public URLs for arbitrary ports on rented instances. This script attempts an outbound Cloudflare Quick Tunnel; it is not an official GHive integration and is not guaranteed to work on every rental. If outbound tunnel traffic is blocked, the local app cannot directly reach this custom worker through a GHive-provided endpoint. The JupyterLab URL is not the worker URL.

GHive rental filesystems are erased when rentals stop. PostgreSQL/Supabase retains the app's persisted profile/media data and LoRA ZIP backups, but model weights and active worker files may need to be downloaded/restored on a new rental. GHive bills by the second; stop the rental from its dashboard when finished. Use a Secure-badged host for sensitive workloads where possible and do not send face photos or secrets unless you accept the third-party-host privacy risk.

References: [GHive workloads](https://ghive.co.in/workloads), [help](https://ghive.co.in/help), [pricing](https://ghive.co.in/pricing).
