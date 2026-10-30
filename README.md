# Private Face Studio

A local-first Windows/Flask application for generating photos, training separate face-identity LoRAs, managing a persistent face library, and optionally using a remote CUDA GPU through the authenticated cloud worker API. GHive can host the authenticated remote CUDA worker for this custom pipeline.

## Features

- Local text-to-image and image-to-image generation with selectable 512×768 portrait, 512×512 square, or 768×512 landscape composition.
- Image-generation presets: standard, photorealistic, and full-body photorealistic. The full-body preset prioritizes head-to-toe framing, adds relevant negative terms for cropped images/anatomy, and discourages clothing when the prompt requests nude adult fine-art imagery.
- Separate sidebar pages for overview, image generation, video generation, face model training, generated images and generated videos.
- PostgreSQL persistence for face profiles, face-training photos, trained LoRA ZIP backups and generated media bytes plus prompts, backend, timestamps and titles.
- Local working copies of per-profile photo datasets under `data/face_profiles/<profile-id>/dataset/`; adding a second person's photos does not mix them with the first person's dataset.
- Image/video library CRUD: preview, search, edit title and prompts, download original media bytes, and delete from the database and local output copy.
- Add model-training photos either by uploading local files or by selecting saved generated images from the image library.
- Generate a **nine-target face-angle dataset** from one source image: front (0°), middle-left/right (±30°), three-quarter left/right (about ±55°), left/right profiles (±90°), and looking up/down (pitch ±18°). Angle jobs use a synthetic target-face pose map with an SD 1.5 OpenPose ControlNet and optional full-face IP-Adapter identity guidance; a pose estimator stores target and actual yaw/pitch and labels unverifiable or mismatched outputs **Needs review**.
- Generate a separate **ten-target facial-expression dataset**: neutral, gentle smile, broad smile, laughing, sad, angry, surprised, fearful, disgusted and confused. Expressions are conditioned to preserve the source head direction where possible and are not mixed with the angle target set.
- Face-angle and expression generation run as **server-managed background jobs** with persisted progress, cancellation and failed-target records. The user can navigate between pages without stopping the job. Each output is saved immediately to the image library and its training group before the next target starts. Cancellation stops at the next supported denoising check and preserves completed results. Failed-target retry submits only failed targets, not previously saved views.
- A separate Training datasets page displays grouped source/angle photos over time and links back to the selected face profile for retraining and later image generation. Dataset groups can be renamed or deleted as a whole. Individual grouped photos can be previewed, relabeled, replaced by a local image, or deleted; local and saved-library photos can also be added to an existing group. Deleting a group/photo removes its training-dataset copy and metadata, but leaves the original generated-library image intact.
- Reusable LoRA adapters stored under `models/face_loras/<profile-id>/`.
- Face selector for photo and video generation.
- A GPU acceleration toggle that is OFF by default: OFF forces local CPU; ON checks actual local CUDA and connected cloud GPU availability.
- A backend selector that appears when acceleration is on: supported local System GPU, connected cloud GPU (GHive), and CPU fallback. Auto mode prefers a usable local CUDA GPU first, then the configured cloud GPU, then CPU if neither GPU is available.
- Authenticated provider-neutral cloud worker endpoints for face training, photo generation, video generation and status polling.
- Local output copies saved under `data/outputs/` and `data/videos/`.

Use your own face or images of a person who has given permission. A LoRA adapter is not a guarantee of perfect identity consistency, and identity may drift between video frames.

## Supabase PostgreSQL configuration

Use Supabase as the PostgreSQL provider for persistent face profiles, face-training photos, trained LoRA ZIP backups, and generated image/video media. The application already reads the database connection from the `DATABASE_URL` environment variable, so changing the private `.env` value switches future database operations to Supabase; no Supabase SDK or separate API key is required.

1. Create a Supabase project and open **Connect** in the project dashboard.
2. Copy the PostgreSQL connection URI. For a local Flask app, use the **Session pooler** URI if the direct database host is not reachable from your network; otherwise the direct URI is also supported.
3. Put the URI in your private `.env` file as `DATABASE_URL=...`. Keep the password private and URL-encode reserved characters in it. Keep the required `sslmode=require` option.
4. Restart the Flask app and check the app's database status. The app creates its tables in the configured Supabase database when it first connects.

The expected URI shape is provider-specific, so copy the exact URI from the Supabase dashboard rather than typing the host or password manually. Do not commit `.env`, paste its credentials into issues, or use the Supabase `service_role` key as a database password.

**Important:** changing `DATABASE_URL` does not automatically copy old records from a previous PostgreSQL database into Supabase. New writes go to Supabase after the app restarts; existing source data remains in the old database unless you explicitly migrate it.

### Migrate an existing PostgreSQL database to Supabase

The repository includes `migrate_postgres_to_supabase.py` for a non-destructive copy of the application's database tables.

1. Back up both databases first.
2. Set `OLD_DATABASE_URL` to the old PostgreSQL database URI and `DATABASE_URL` to the Supabase PostgreSQL URI in your private shell environment. Do not commit either URI or share passwords.
3. Install the project requirements, then run `python migrate_postgres_to_supabase.py`.
4. Review the per-table counts. The script copies parent tables before dependent tables and uses `ON CONFLICT DO NOTHING`: rows conflicting with a primary key or unique constraint are skipped, never overwritten. It reports inserted and skipped/conflicting counts.
5. Verify profiles, training photos, jobs, model bundles, and generated assets in the Supabase-backed app before retiring the old database.

The script is safe to rerun for ordinary primary-key/unique-constraint duplicates, but it is not a substitute for a database backup. It does not delete source records. It copies only the application's known tables and matching columns; review any schema changes or custom tables separately. Keep both connection strings private.



On first library access, existing generated files already present under `data/outputs/` and `data/videos/` are imported into the library when missing from the database, so older locally generated files can appear alongside new results. Their old prompts may not be recoverable, so those records use an imported/unknown metadata label.

The database stores generated image/video file bytes, profile photo bytes, and trained LoRA ZIP backups in PostgreSQL binary columns, in addition to metadata. Local LoRA model folders remain on the compute host so Diffusers can load them directly. **Supabase free-tier database storage is limited, and generated media can consume it quickly**; monitor usage and keep local backups. If the library grows substantially, consider storing large media files in Supabase Storage and keeping only their metadata in PostgreSQL as a separate future change.

`DATABASE_URL` must point to PostgreSQL/Supabase. The app does not use a local SQLite database.

## Local Windows setup

Requirements: Windows 10/11 64-bit, Python 3.11 x64, PostgreSQL reachable from the computer, and several GB of free disk. CPU inference can be very slow; 16 GB RAM is preferable to 8 GB for basic local generation.

1. Review the Stable Diffusion 1.5 model card and license: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
2. Set DATABASE_URL in .env before the first run. If you previously used a JSON profile registry, the app migrates it and its stored face photos to PostgreSQL.
3. Run install_windows.bat.
3. Run `download_model.bat` while online.
4. Run `run_local.bat`.
5. Open http://127.0.0.1:5000.

The Windows installer installs CPU-only PyTorch. This is for local development and slow image generation; it does not enable CUDA training or local video generation. The local web app binds to 127.0.0.1 only.

### Select a compatible photorealistic checkpoint

The original Stable Diffusion 1.5 checkpoint often produces less convincing human faces and skin than a checkpoint fine-tuned for photography. The app now honors an explicit `SD_MODEL_ID` Hugging Face Diffusers repository even if the bundled base folder exists, or an existing local Diffusers folder via `SD_MODEL_PATH`. Add the setting to your private `.env` and restart both the app and GHive worker. Review each model's license, compatibility, and access requirements before use. Leave both variables blank to use the bundled model. Changing the base checkpoint may require retraining face LoRAs against that checkpoint.

## Use the multipage workspace

- Image generation: open Generate image from the sidebar, select a trained face, choose the composition and image preset, optionally upload a reference image, enter a concise prompt and save a title. Portrait is 512×768 by default to provide more room for full-length subjects; inference defaults to 25 steps and can be reduced for faster tests. Successful output is stored in PostgreSQL and appears in Generated images. Stable Diffusion 1.5 has a short CLIP text context, so keep essential subject, clothing/nudity, body, and framing terms near the start. A preset can improve the wording and framing, but cannot guarantee a particular result.
- Video generation: open Generate video, select the face/profile trained on the selected backend, choose frame count and steps, and generate the clip. Successful output is stored in PostgreSQL and appears in Generated videos.
- Face models: create isolated identity profiles, add photos from your computer, reuse saved generated images, or select **Face angles** / **Expressions** and upload one source image to start a background job. Angles queues 9 precise yaw/pitch targets; Expressions queues 10 expression targets. A shared progress banner and Cancel button remain available across routes; the Start button stays disabled while a job is active. Every successful output is saved immediately and appears in Training datasets while later targets continue. A failed target can be retried from the global job banner without regenerating successful targets. Pose conditioning improves the requested view, but visual review is still required because SD 1.5 can invent unseen face details and identity/pose estimates are approximate. Each profile can include an optional face/training description of up to 200 words; the trigger token remains a short identifier. Stable Diffusion 1.5 uses a fixed text-encoder context window, so the trainer rotates through short description chunks across training steps instead of sending all 200 words in one prompt. At least 5 photos are required to start training; 15–30 varied photos are a better starting point.
- Training datasets: separate filters show **Face angles**, **Expressions**, and **Body type · global**. Photo cards show automatic pose-validation status, estimated yaw/pitch when available, and the reason a photo needs review. A prompt-edit preview remains non-destructive until **Save new photo** is clicked; replacing an image resets its pose status to Needs review. Retrain the associated identity after reviewing and accepting dataset updates.
- Libraries: use the separate Generated images and Generated videos pages to search across all saved items, page through the results, preview, edit the title/prompt/negative prompt, download the original media bytes, or delete an item. Deletion removes the database record and the local output copy. The generated-image picker on Face models can also search and page through saved image outputs when selecting training photos.
- **Live generation progress:** image and video pages show the current phase and real Diffusers denoising step count while generation runs. A browser poll updates the progress bar roughly every 0.8 seconds; the GHive worker exposes its own authenticated progress endpoint so cloud generations show their actual worker-side steps too. Model loading, final encoding, and database save appear as separate phases, so the percentage is an approximate overall indicator while the step counter reflects the pipeline callback.
- **Background dataset jobs:** job ID, target manifest, target progress, cancellation state, per-target failures and pose-review records are persisted in the database. Navigation does not cancel a job; only stopping the app/worker process interrupts execution. After an app restart, a stale active job is marked interrupted and completed photos remain saved rather than claiming the generation continued.
- Each training job reads only its selected profile's local working dataset and writes an adapter to models/face_loras/<profile-id>/. When PostgreSQL is configured, successful local training also writes a ZIP backup of the adapter to the database. A completed GHive training job is similarly backed up to the database when the local app polls its final status.
- Select the same face in the image or video panel. Include the displayed prompt token in your prompt, such as photo of face_a1b2c3d4 person, natural studio light.

Local LoRA training requires a CUDA-enabled NVIDIA GPU. Your local Windows venv will normally report `torch.cuda.is_available() == False`; on that machine, select a connected cloud GPU worker (GHive) after configuring it.

### Existing ungrouped dataset note

Older versions stored all uploaded photos in `data/face_dataset/`. New named profiles use `data/face_profiles/<profile-id>/dataset/`. Review and manually copy any old images into the correct new profile through the UI; the app deliberately will not guess which person those old images belong to.

## GHive rented GPU mode (optional, OFF by default)

The local Windows application remains the controller and database/library. GHive supplies the remote NVIDIA GPU where this repository's own Stable Diffusion, AnimateDiff, ControlNet, IP-Adapter and face-LoRA pipeline runs. This differs from GHive's hosted image-generation API: that API uses GHive-managed model tiers and does not run this repository's custom face-profile LoRAs.

**Networking limitation:** GHive currently provides JupyterLab/terminal access but does not provide public URLs for ports exposed by rented instances. This project therefore attempts an outbound Cloudflare Quick Tunnel from inside the GHive instance. This is not an official GHive integration or guaranteed route; verify that outbound tunnel traffic is permitted for your rental. If it is blocked, the local app cannot directly call this custom worker through a GHive-provided endpoint. Do not mistake the JupyterLab URL for a worker URL.

### A. Rent and prepare a GHive GPU

1. Open [GHive](https://ghive.co.in/), sign in, and open **Browse GPUs**.
2. Choose **PyTorch + Jupyter** and an NVIDIA GPU. Prefer 24 GB VRAM (RTX 3090/4090) for LoRA training and video generation. Smaller GPUs may need reduced settings.
3. Start the rental and open its JupyterLab terminal.
4. Clone the repository's main branch:

       git clone --branch main https://github.com/AbhishekSingh971/local_private_image_generation.git
       cd local_private_image_generation

5. Generate a private worker secret and start the setup:

       export REMOTE_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
       export GPU_WORKER_PORT=8000
       bash run_ghive_worker.sh

The script checks CUDA, installs project requirements, downloads the SD 1.5 base model and AnimateDiff adapter if absent, starts the authenticated worker and launches cloudflared. Copy the HTTPS `trycloudflare.com` URL printed by cloudflared. Keep the terminal/tunnel running while the local app uses the GPU. The URL changes when the Quick Tunnel restarts.

### B. Configure Windows

1. Copy `.env.example` to `.env` if needed. Preserve your existing `DATABASE_URL` and `HF_TOKEN`.
2. Set `GPU_WORKER_URL` to the exact HTTPS tunnel base URL (without `/api/remote/health`).
3. Set `REMOTE_API_TOKEN` to the exact same secret generated in the GHive terminal.
4. Restart `run_local.bat`.
5. Enable **GPU acceleration** and choose **GHive GPU** (or Auto). The app authenticates to the worker and checks CUDA, GPU name and VRAM.
6. Test a small image generation first, then test LoRA training and video generation.

Example local `.env` entries:

       GPU_WORKER_URL=https://your-random-name.trycloudflare.com
       REMOTE_API_TOKEN=the-same-long-random-secret
       GPU_WORKER_PORT=8000

Never commit `.env`, paste the token into chat, or expose port 8000 without HTTPS and bearer-token authentication. The local app sends only the selected job's prompt/reference image or face-training photos to the worker.

### C. Storage, privacy and costs

- GHive rental workspaces are erased when the rental stops. Download anything that is not already persisted before stopping.
- PostgreSQL/Supabase stores supported face profiles, source photos, generated media and completed LoRA ZIP backups. Model weights and active LoRA files are still needed on the worker filesystem while jobs run; a fresh rental may need model downloads and adapter restoration/retraining.
- GHive workloads run on third-party host machines. Prefer Secure-badged hosts for sensitive workloads; do not upload secrets or face photos unless you accept the privacy risk and have permission to process them.
- Billing is per second while the rental is running. Stop the rental in GHive's dashboard when finished; closing the local app does not stop it.
- GHive's hosted image API is a separate pay-per-output service and is not a drop-in replacement for this worker's custom face-LoRA pipeline.

Official references: [GHive workloads](https://ghive.co.in/workloads), [GHive help](https://ghive.co.in/help), [GHive pricing](https://ghive.co.in/pricing).

## Face-angle dataset: identity-first training strategy

The nine-angle workflow is pose-guided synthesis, not literal rotation of a 2D photograph. A single image contains no true texture for the hidden side of the nose, cheek, ear, or hair; rotating a 2D face mesh can estimate target geometry but cannot recover unseen pixels. The application combines a rotated MediaPipe landmark map, image-to-image conditioning, and IP-Adapter / profile LoRA identity guidance when available. Pose estimates are approximate; every generated view must be validated before it is trusted as training data.

### Why the previous samples became distorted

- The old middle-angle targets used img2img strength 0.62, three-quarter targets 0.74, and profile targets 0.82. Higher img2img strength injects more noise and allows more of the original face to be redrawn, so identity can drift and eyes, lips, and makeup can become corrupted.
- ControlNet strength 1.35 could overpower image/identity conditioning, especially because the rotated MediaPipe mesh is a pseudo-3D estimate rather than a complete 3D face reconstruction.
- Pose validation checks estimated yaw/pitch, not whether the person still looks like the same identity. A pose-matched but distorted face must still be rejected manually.

### Mitigations committed

- Reduced img2img strength for middle angles to 0.38, three-quarter views to 0.46, side profiles to 0.52, and up/down views to 0.38.
- Reduced default face-pose ControlNet conditioning scale to 0.75 (override with `FACE_POSE_CONTROLNET_SCALE`).
- Increased default full-face IP-Adapter scale from 0.45 to 0.65 (override with `FACE_IDENTITY_IP_ADAPTER_SCALE`).
- Kept pose validation and review states. These changes reduce the chance of distortion but cannot guarantee perfect novel-view synthesis from one photo.

### Research comparison and recommended long-term design

1. **Identity LoRA / DreamBooth** learns a person-specific appearance token from several real, diverse views. It is not itself a 3D rotation engine; with only one frontal image it has no real examples of the hidden side. The original DreamBooth method uses few-shot subject images plus prior-preservation to reduce overfitting ([paper](https://arxiv.org/abs/2208.12242)).
2. **IP-Adapter / face-ID conditioning** supplies identity features at inference time. Pair it with pose control; neither prompt text nor identity embeddings alone enforce accurate geometry.
3. **3D face reconstruction / morphable models** estimate shape and pose, then render a proxy face at the target yaw/pitch. A 3D mesh provides more coherent geometry than rotating a 2D landmark cloud, but unseen skin/hair texture still needs generative completion.
4. **Multi-view novel-view synthesis** methods such as CAfCA report strong results from as few as three casual views, using a learned volumetric prior plus sparse per-person fitting ([Google Research summary](https://research.google/pubs/cafca-high-quality-novel-view-synthesis-of-expressive-faces-from-casual-few-shot-captures/)). This is a better target architecture if users can supply 3+ real views.
5. **MediaPipe landmarks and transformation matrices** are useful for estimating pose and driving effects, but the 478 landmarks and matrix are not a complete identity-preserving 3D texture model ([documentation](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)).

Recommended next engineering stage: add a dedicated 3D face reconstruction stage (for example, a FLAME/3DMM-based estimator), render target-pose geometry, then run low-strength diffusion with face-ID conditioning. Train the identity LoRA only on real source images plus generated views that pass both pose checks and a face-identity similarity check. Do not automatically train on every generated output or photos marked **Needs review**. A single image cannot uniquely determine the true unseen side of a face; for reliable nine-angle identity, ask for multiple real views whenever possible.