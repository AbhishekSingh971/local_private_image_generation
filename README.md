# Private Face Studio

A local-first Windows/Flask application for generating photos, training separate face-identity LoRAs, managing a persistent face library, and optionally using a remote CUDA GPU through the authenticated cloud worker API. RunPod remains supported; Google Colab can also host the temporary GPU worker.

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
- A backend selector that appears when acceleration is on: supported local System GPU, connected cloud GPU (Colab or RunPod), and CPU fallback. Auto mode prefers a usable local CUDA GPU first, then the configured cloud GPU, then CPU if neither GPU is available.
- Authenticated provider-neutral cloud worker endpoints for face training, photo generation, video generation and status polling.
- Local output copies saved under `data/outputs/` and `data/videos/`.

Use your own face or images of a person who has given permission. A LoRA adapter is not a guarantee of perfect identity consistency, and identity may drift between video frames.

## PostgreSQL configuration

Create a PostgreSQL database before starting the app. Copy .env.example to .env if you do not already have a private .env, then set DATABASE_URL to the connection URI for your database. Supported examples are postgresql://USER:PASSWORD@HOST:5432/DATABASE and a provider URL that includes its required SSL options. The app normalizes the standard postgresql:// format to the psycopg 3 driver. If the password contains reserved URL characters, URL-encode them.

Keep the real connection URL only in your local .env; never commit it or paste its credentials into an issue. The application creates the required tables when it connects. When DATABASE_URL is configured, it imports existing profiles and face photos from the previous JSON/filesystem layout into PostgreSQL the first time the profile store is read. The original files are left in place as a working cache.

On first library access, existing generated files already present under `data/outputs/` and `data/videos/` are imported into the library when missing from the database, so older locally generated files can appear alongside new results. Their old prompts may not be recoverable, so those records use an imported/unknown metadata label.

The database stores generated image/video file bytes in PostgreSQL binary columns, as well as media metadata. Profile photo bytes are stored in PostgreSQL and mirrored as working files for the training script. Trained LoRA model folders remain on the compute host so Diffusers can load them directly, and ZIP backups are saved in PostgreSQL after successful training when DATABASE_URL is configured. Binary storage can increase database size quickly, so monitor database storage and backups.

If DATABASE_URL is empty, the app shows a storage warning and uses a local SQLite development database for media instead; this is not the intended PostgreSQL setup for persistent use.

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

The original Stable Diffusion 1.5 checkpoint often produces less convincing human faces and skin than a checkpoint fine-tuned for photography. The app now honors an explicit `SD_MODEL_ID` Hugging Face Diffusers repository even if the bundled base folder exists, or an existing local Diffusers folder via `SD_MODEL_PATH`. Add the setting to your private `.env` and restart both the app and RunPod worker. Review each model's license, compatibility, and access requirements before use. Leave both variables blank to use the bundled model. Changing the base checkpoint may require retraining face LoRAs against that checkpoint.

## Use the multipage workspace

- Image generation: open Generate image from the sidebar, select a trained face, choose the composition and image preset, optionally upload a reference image, enter a concise prompt and save a title. Portrait is 512×768 by default to provide more room for full-length subjects; inference defaults to 25 steps and can be reduced for faster tests. Successful output is stored in PostgreSQL and appears in Generated images. Stable Diffusion 1.5 has a short CLIP text context, so keep essential subject, clothing/nudity, body, and framing terms near the start. A preset can improve the wording and framing, but cannot guarantee a particular result.
- Video generation: open Generate video, select the face/profile trained on the selected backend, choose frame count and steps, and generate the clip. Successful output is stored in PostgreSQL and appears in Generated videos.
- Face models: create isolated identity profiles, add photos from your computer, reuse saved generated images, or select **Face angles** / **Expressions** and upload one source image to start a background job. Angles queues 9 precise yaw/pitch targets; Expressions queues 10 expression targets. A shared progress banner and Cancel button remain available across routes; the Start button stays disabled while a job is active. Every successful output is saved immediately and appears in Training datasets while later targets continue. A failed target can be retried from the global job banner without regenerating successful targets. Pose conditioning improves the requested view, but visual review is still required because SD 1.5 can invent unseen face details and identity/pose estimates are approximate. Each profile can include an optional face/training description of up to 200 words; the trigger token remains a short identifier. Stable Diffusion 1.5 uses a fixed text-encoder context window, so the trainer rotates through short description chunks across training steps instead of sending all 200 words in one prompt. At least 5 photos are required to start training; 15–30 varied photos are a better starting point.
- Training datasets: separate filters show **Face angles**, **Expressions**, and **Body type · global**. Photo cards show automatic pose-validation status, estimated yaw/pitch when available, and the reason a photo needs review. A prompt-edit preview remains non-destructive until **Save new photo** is clicked; replacing an image resets its pose status to Needs review. Retrain the associated identity after reviewing and accepting dataset updates.
- Libraries: use the separate Generated images and Generated videos pages to search across all saved items, page through the results, preview, edit the title/prompt/negative prompt, download the original media bytes, or delete an item. Deletion removes the database record and the local output copy. The generated-image picker on Face models can also search and page through saved image outputs when selecting training photos.
- **Live generation progress:** image and video pages show the current phase and real Diffusers denoising step count while generation runs. A browser poll updates the progress bar roughly every 0.8 seconds; the RunPod worker exposes its own authenticated progress endpoint so cloud generations show their actual worker-side steps too. Model loading, final encoding, and database save appear as separate phases, so the percentage is an approximate overall indicator while the step counter reflects the pipeline callback.
- **Background dataset jobs:** job ID, target manifest, target progress, cancellation state, per-target failures and pose-review records are persisted in the database. Navigation does not cancel a job; only stopping the app/worker process interrupts execution. After an app restart, a stale active job is marked interrupted and completed photos remain saved rather than claiming the generation continued.
- Each training job reads only its selected profile's local working dataset and writes an adapter to models/face_loras/<profile-id>/. When PostgreSQL is configured, successful local training also writes a ZIP backup of the adapter to the database. A completed RunPod training job is similarly backed up to the database when the local app polls its final status.
- Select the same face in the image or video panel. Include the displayed prompt token in your prompt, such as photo of face_a1b2c3d4 person, natural studio light.

Local LoRA training requires a CUDA-enabled NVIDIA GPU. Your local Windows venv will normally report `torch.cuda.is_available() == False`; on that machine, select a connected cloud GPU worker (Colab or RunPod) after configuring it.

### Existing ungrouped dataset note

Older versions stored all uploaded photos in `data/face_dataset/`. New named profiles use `data/face_profiles/<profile-id>/dataset/`. Review and manually copy any old images into the correct new profile through the UI; the app deliberately will not guess which person those old images belong to.

## Google Colab GPU mode (free runtime, optional)

You can experiment with the same authenticated `gpu_worker.py` worker from a Colab notebook. Start with [COLAB_GPU_SETUP.md](COLAB_GPU_SETUP.md) and [run_colab_worker.sh](run_colab_worker.sh). The app code supports the existing `/api/remote/*` endpoints, with `GPU_WORKER_URL` and `REMOTE_API_TOKEN` for the generic worker connection; legacy RunPod-specific environment variables are no longer used.

**Important:** Google's current Colab FAQ warns that free managed runtimes may be terminated when used as a web service for content generation. This worker exposes an HTTP image/video-generation API, so free Colab is not a reliable supported deployment target; use this section for limited experimentation only and check Google's [current policy](https://research.google.com/colaboratory/faq.html) before using it.

Colab's free runtime is temporary and may be disconnected or reclaim its GPU. A stable ngrok Dev Domain can keep the same HTTPS hostname, but it does not keep the Colab session or tunnel alive. Model weights and LoRA files on Colab storage can be lost when the runtime is reset; save trained adapters and important outputs to durable storage. Do not expose the worker without a strong `REMOTE_API_TOKEN` and HTTPS.

## RunPod GPU mode (optional, OFF by default)

The UI toggle routes training/generation requests to a worker that you already deployed. It **does not create, purchase, start, stop, or terminate** a Pod. You must manually choose an RTX 3090 (24 GB VRAM) Pod in RunPod if available; availability and prices vary. A running Pod can incur charges even when idle. Stop it from RunPod when finished, and use a persistent volume if you need model weights and face profiles to survive Pod replacement. RunPod's exposed-port proxy uses a URL like `https://<POD_ID>-8000.proxy.runpod.net`; see the official docs: https://docs.runpod.io/pods/overview and https://docs.runpod.io/pods/connect-to-a-pod.

### A. Prepare the GPU Pod

1. In RunPod, deploy a CUDA-enabled NVIDIA GPU Pod. Select RTX 3090 if it is available for your region/account.
2. Expose HTTP port `8000` and mount a persistent volume, recommended at `/workspace`.
3. Clone this repository's `feature/face-identity-lora` branch into the persistent workspace, for example:

       cd /workspace
       git clone --branch feature/face-identity-lora https://github.com/AbhishekSingh971/local_private_image_generation.git
       cd local_private_image_generation

4. Confirm that the Pod sees its GPU:

       python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU')"

   The CUDA result must be `True`. Use a RunPod PyTorch/CUDA template; the local Windows CPU-only installer is not suitable on the Pod.

5. Create a long random secret. Set `REMOTE_API_TOKEN` in the Pod environment to that secret, then run:

       bash runpod_worker_linux.sh

   The script installs app requirements, downloads the local SD 1.5 base model and AnimateDiff motion adapter (downloads may require accepting their model terms and setting `HF_TOKEN`), then starts the authenticated worker on port 8000. Do not expose this worker without its token.

6. In RunPod, confirm the Pod is running and note the exposed port URL. It should follow:

For background face-angle/expression jobs, pose-conditioned generation and Cancel support, the worker must be running the latest `feature/face-identity-lora` code. It uses the authenticated `/api/remote/generation-progress/<task-id>` and `/api/remote/generation-cancel/<task-id>` endpoints. Pull this branch and restart `runpod_worker_linux.sh`; updating only the Windows app is not enough for cloud jobs. The first angle job downloads the SD 1.5 OpenPose ControlNet and IP-Adapter weights if they are not already cached, so leave enough disk space and expect first-run model loading to take longer.

       https://<POD_ID>-8000.proxy.runpod.net

### B. Configure your Windows app

1. Copy `.env.example` to `.env` in the local project folder.
2. Set `GPU_WORKER_URL` to the HTTPS base URL for the worker (no `/api/remote/health` suffix).
3. Set `REMOTE_API_TOKEN` to the exact same secret used by `gpu_worker.py` on the worker.
4. Restart `run_local.bat`.
5. Toggle **Enable GPU acceleration** on. The app checks whether this Python environment can use a local CUDA GPU and probes the authenticated RunPod worker.
6. The selector lists only detected, usable devices plus Auto and an explicit CPU fallback. Auto prefers the local System GPU, then the connected cloud GPU. If Auto cannot reach a GPU, generation returns a clear error instead of silently running on CPU.
7. Turn the toggle off to force local CPU for subsequent requests.

Keep `.env` private. Do not commit it or paste a real token into an issue/chat. `.env` and face data are ignored by Git. The local app forwards face photos only when you submit a training job with the RunPod toggle on; generation requests send the prompt and, if used, the reference image to the configured worker.

### RunPod persistence and storage

The worker stores the uploaded face datasets and generated LoRAs on the Pod's filesystem under the project directory. Put the repository and its `data/` and `models/` directories on a persistent volume if you need them to survive a Pod replacement. Back up important profiles and LoRA weights before terminating a Pod.

## Training details

- Training script: `train_face_lora.py`
- A profile is trained independently by ID, token and dataset.
- Example direct command (only on CUDA-enabled hardware, after photos have been saved to a profile):

      python train_face_lora.py --profile-id <32-character-profile-id> --output-name <same-profile-id> --token <profile-token> --steps 800 --resolution 512 --mixed-precision fp16

- Default web UI training values are 800 steps and 512px. Lower to 384px if VRAM is tight. More steps do not automatically produce a better likeness; inspect results and adjust dataset/parameters.
- The app uses a job ID and status polling for RunPod training so the browser request does not need to remain open for the entire training run.

## Security and privacy

- The browser toggle defaults OFF every time the page loads. OFF means local CPU.
- The sidebar separates Overview, Image generation, Video generation, Face models, Generated images and Generated videos.
- PostgreSQL stores generated media bytes, training photos, face profile metadata and ZIP backups of trained LoRA bundles. Working LoRA folders stay on compute hosts for direct model loading.
- When ON, actual availability is detected server-side. The automatic choice prefers a usable local CUDA GPU, then an authenticated RunPod GPU, then CPU fallback; an explicitly chosen unavailable GPU is rejected rather than silently used.
- When local and cloud GPUs are both available, the UI lets you switch between them. A local GPU typically avoids cloud usage fees; a cloud RTX 3090 may be faster than a lower-powered local GPU.
- The toggle never provisions a GPU or changes RunPod billing state; it only chooses where submitted requests run.
- Worker APIs require `Authorization: Bearer <REMOTE_API_TOKEN>`; use a strong random token and HTTPS.
- Initial package/model downloads use the internet. Local inference otherwise runs on the local model. RunPod mode sends the specific task's input data to that Pod.
- Uploaded face images are re-encoded as JPEG to discard EXIF metadata and original filenames.
- This starter app is designed for personal localhost use. Do not expose the local Flask UI to the public internet.
- Check all underlying model licenses before publishing or using outputs commercially.

## Troubleshooting

- **Flask import fails:** install dependencies with `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.
- **Windows DLL blocked:** inspect the Windows Code Integrity event log. Do not disable application control or download replacement DLLs from unofficial sites.
- **Local training says CUDA unavailable:** expected with the CPU-only Windows install. Configure the RunPod worker instead.
- **GPU worker shows not configured:** fill `GPU_WORKER_URL` and `REMOTE_API_TOKEN` in `.env` (use the same token on the worker), then restart the app.
- **Cloud GPU health request fails:** check the worker runtime, exposed HTTP port 8000, HTTPS URL, shared `REMOTE_API_TOKEN`, and worker logs.
- **Model download fails:** review model terms and access requirements and check any required Hugging Face token.


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