# GHive GPU worker setup

1. Rent an NVIDIA GPU using GHive's **PyTorch + Jupyter** workload. Prefer 24 GB VRAM for face-LoRA training and video generation.
2. Open a JupyterLab terminal and clone the main branch:

   ```bash
   git clone --branch feature/ghive-runtime-integration https://github.com/AbhishekSingh971/local_private_image_generation.git
   cd local_private_image_generation
   ```

3. Create a worker secret and launch:

   ```bash
   export REMOTE_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
   printf 'Copy this token into your local .env (keep it private): %s\n' "$REMOTE_API_TOKEN"
   export GPU_WORKER_PORT=8000
   bash run_ghive_worker.sh
   ```

4. Copy the HTTPS `trycloudflare.com` URL printed by cloudflared. Keep the terminal open.
5. Configure the application that will call the worker:
   - **Local Windows:** set `GPU_WORKER_URL` and `REMOTE_API_TOKEN` in the private `.env`, then restart `run_local.bat`.
   - **Render:** add the same `GPU_WORKER_URL` and `REMOTE_API_TOKEN` values in the Render service's Environment settings, then restart/redeploy the service.
   - You may configure both environments with the same values. Neither application needs CUDA installed locally.
6. Enable GPU acceleration and select the remote GHive / Colab GPU, then test a small image.

## Critical networking limitation

GHive currently provides JupyterLab and terminal access but does not provide public URLs for arbitrary ports on rented instances. This script attempts an outbound Cloudflare Quick Tunnel; it is not an official GHive integration and is not guaranteed to work on every rental. If outbound tunnel traffic is blocked, the local app cannot directly reach this custom worker through a GHive-provided endpoint. The JupyterLab URL is not the worker URL.

GHive rental filesystems are erased when rentals stop. PostgreSQL/Supabase retains the app's persisted profile/media data and LoRA ZIP backups, but model weights and active worker files may need to be downloaded/restored on a new rental. GHive bills by the second; stop the rental from its dashboard when finished. Use a Secure-badged host for sensitive workloads where possible and do not send face photos or secrets unless you accept the third-party-host privacy risk.

For long-running video and training requests, keep in mind that the current worker API performs some generation calls synchronously. A free Quick Tunnel or the app host can time out on unusually long jobs; verify your actual workload before relying on it for production.

References: [GHive workloads](https://ghive.co.in/workloads), [help](https://ghive.co.in/help), [pricing](https://ghive.co.in/pricing).
