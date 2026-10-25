# Private Face Studio

A local-first Windows/Flask application for generating photos, training separate face-identity LoRAs, managing a persistent face library, and optionally using a RunPod NVIDIA GPU (such as an RTX 3090) for training and generation.

## Features

- Local text-to-image and image-to-image generation with Stable Diffusion 1.5.
- Separate sidebar pages for overview, image generation, video generation, face model training, generated images and generated videos.
- PostgreSQL persistence for face profiles, face-training photos, trained LoRA ZIP backups and generated media bytes plus prompts, backend, timestamps and titles.
- Local working copies of per-profile photo datasets under `data/face_profiles/<profile-id>/dataset/`; adding a second person's photos does not mix them with the first person's dataset.
- Image/video library CRUD: preview, search, edit title and prompts, download original media bytes, and delete from the database and local output copy.
- Add model-training photos either by uploading local files or by selecting saved generated images from the image library.
- Reusable LoRA adapters stored under `models/face_loras/<profile-id>/`.
- Face selector for photo and video generation.
- A GPU acceleration toggle that is OFF by default: OFF forces local CPU; ON checks actual local CUDA and RunPod GPU availability.
- A backend selector that appears when acceleration is on: supported local System GPU, connected RunPod GPU, and CPU fallback. Auto mode prefers a usable local CUDA GPU first, then the configured cloud GPU, then CPU if neither GPU is available.
- Authenticated RunPod worker endpoints for face training, photo generation, video generation and status polling.
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

## Use the multipage workspace

- Image generation: open Generate image from the sidebar, select a trained face, optionally upload a reference image, enter prompts and save a title. Successful output is stored in PostgreSQL and appears in Generated images.
- Video generation: open Generate video, select the face/profile trained on the selected backend, choose frame count and steps, and generate the clip. Successful output is stored in PostgreSQL and appears in Generated videos.
- Face models: create isolated identity profiles, add photos from your computer, or use the Generated images picker to add previously saved outputs as training photos. Each profile can include an optional face/training description of up to 200 words; the trigger token remains a short identifier. Stable Diffusion 1.5 uses a fixed text-encoder context window, so the trainer rotates through short description chunks across training steps instead of sending all 200 words in one prompt. At least 5 photos are required to start training; 15–30 varied photos are a better starting point.
- Libraries: use the separate Generated images and Generated videos pages to search across all saved items, page through the results, preview, edit the title/prompt/negative prompt, download the original media bytes, or delete an item. Deletion removes the database record and the local output copy. The generated-image picker on Face models can also search and page through saved image outputs when selecting training photos.
- **Live generation progress:** image and video pages show the current phase and real Diffusers denoising step count while generation runs. A browser poll updates the progress bar roughly every 0.8 seconds; the RunPod worker exposes its own authenticated progress endpoint so cloud generations show their actual worker-side steps too. Model loading, final encoding, and database save appear as separate phases, so the percentage is an approximate overall indicator while the step counter reflects the pipeline callback.
- Each training job reads only its selected profile's local working dataset and writes an adapter to models/face_loras/<profile-id>/. When PostgreSQL is configured, successful local training also writes a ZIP backup of the adapter to the database. A completed RunPod training job is similarly backed up to the database when the local app polls its final status.
- Select the same face in the image or video panel. Include the displayed prompt token in your prompt, such as photo of face_a1b2c3d4 person, natural studio light.

Local LoRA training requires a CUDA-enabled NVIDIA GPU. Your local Windows venv will normally report `torch.cuda.is_available() == False`; on that machine, turn on RunPod after you configure a running GPU worker.

### Existing ungrouped dataset note

Older versions stored all uploaded photos in `data/face_dataset/`. New named profiles use `data/face_profiles/<profile-id>/dataset/`. Review and manually copy any old images into the correct new profile through the UI; the app deliberately will not guess which person those old images belong to.

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

For live generation progress, the worker must be running the latest `feature/face-identity-lora` code, which adds the authenticated `/api/remote/generation-progress/<task-id>` endpoint and diffusion-step callbacks. If this Pod was deployed from an older commit, pull the latest branch in its workspace and restart `runpod_worker_linux.sh`; updating only the Windows app is not enough for cloud progress.

       https://<POD_ID>-8000.proxy.runpod.net

### B. Configure your Windows app

1. Copy `.env.example` to `.env` in the local project folder.
2. Set `RUNPOD_WORKER_URL` to the HTTPS port-8000 URL from the Pod.
3. Set `RUNPOD_WORKER_TOKEN` to the exact same secret used as `REMOTE_API_TOKEN` on the Pod.
4. Restart `run_local.bat`.
5. Toggle **Enable GPU acceleration** on. The app checks whether this Python environment can use a local CUDA GPU and probes the authenticated RunPod worker.
6. The selector only lists available backends. When both GPUs are available, the default is the local System GPU to avoid cloud charges, but you can switch to the RunPod GPU. If local CUDA is unavailable, the connected RunPod GPU is selected automatically; CPU remains an explicit fallback.
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
- **RunPod toggle shows not configured:** fill `RUNPOD_WORKER_URL` and `RUNPOD_WORKER_TOKEN` in `.env`, then restart the local app.
- **RunPod health request fails:** check Pod status, exposed HTTP port 8000, URL, shared token and worker terminal logs.
- **Model download fails:** review model terms and access requirements and check any required Hugging Face token.
