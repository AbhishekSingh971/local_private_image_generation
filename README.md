# Local Private Image AI

A local-first Windows/Flask project for Stable Diffusion 1.5 image generation, reusable one-person face identities (LoRA), and optional short AnimateDiff videos.

## What is supported

- Text-to-image generation from a prompt.
- Image-to-image edits with an optional reference image.
- Upload and store a local dataset of one person's face photos.
- Train a reusable SD 1.5 LoRA from the dataset with a CUDA-enabled NVIDIA GPU.
- Select a trained LoRA in the web app to generate more photos with that identity.
- Optional short video generation with AnimateDiff and the same SD 1.5 LoRA on a compatible NVIDIA GPU.

A LoRA is an identity adapter, not a new base model. Identity consistency varies with dataset quality, prompts, poses and lighting; it is not guaranteed in every image or video frame.

## Hardware: read before installing

The app can run on CPU, but CPU diffusion is slow and can exceed available RAM. An 8 GB RAM laptop with Intel Iris Xe graphics is suitable for preparing the dataset and developing the web app, not practical for face-LoRA training or video generation.

- **Photo generation:** CPU works slowly; 16 GB RAM is preferable to 8 GB.
- **LoRA training:** requires an NVIDIA GPU and CUDA-enabled PyTorch. A GPU with 12 GB or more VRAM is a more comfortable starting point; exact needs vary by batch size, resolution and settings.
- **Video generation:** requires CUDA-enabled PyTorch, an NVIDIA GPU, and additional model weights. More VRAM is helpful. The bundled Windows installer intentionally installs CPU-only PyTorch, so it will not enable training or video generation.

The app does not automatically send your dataset to a cloud service. If you choose to train on a rented GPU, you must transfer the photos and model files yourself; the photos will then be processed outside your computer.

## Setup on Windows

1. Use Python 3.11 x64.
2. Review and accept the Stable Diffusion 1.5 model terms: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
3. Run install_windows.bat. This installs CPU-only PyTorch for local app development and basic CPU inference.
4. Run download_model.bat while online. It downloads SD 1.5 to models/stable-diffusion-v1-5.
5. Run run_local.bat.
6. Open http://127.0.0.1:5000.

The app binds to localhost only. Do not change its host to 0.0.0.0 if you want it private to this computer.

### If you have an NVIDIA GPU

Install a CUDA-enabled PyTorch build that matches your environment using the official PyTorch installation selector. Do not assume the CPU-only PyTorch installed by install_windows.bat will use the GPU. Verify with:

    python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

The last value should be True before training or video generation.

## Train a face identity

Only use your own images or images of a person who has given permission for this use.

1. In the web app, upload 15–30 clear, varied photos of the same person. They are saved under data/face_dataset. The app re-encodes them to strip EXIF metadata.
2. Review the dataset and remove duplicates, group shots, strong filters, sunglasses and blurry photos if needed.
3. Activate the project environment. Training requires CUDA-enabled PyTorch.
4. Run:

    python train_face_lora.py --output-name my_face --token sks --steps 800 --mixed-precision fp16

The trainer reads the base model and images from local disk only. It saves the LoRA to models/face_loras/my_face. Training defaults to 512px resolution and batch size 1. If you run out of GPU memory, try --resolution 384 or a compatible GPU with more VRAM; tune settings rather than blindly increasing steps.

Example prompt: "photo of sks person, outdoor portrait, natural window light". The token is not a magic identity lock. Inspect results and improve the dataset/settings if identity consistency is poor.

After training, click **Refresh face models** in the web UI and select your trained identity. With a reference image the app uses image-to-image; without one it uses text-to-image.

## Optional video generation

1. Review the AnimateDiff motion-adapter model card and license: https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-2
2. Run download_video_model.bat while online. The adapter is saved under models/animatediff-motion-adapter-v1-5-2.
3. Ensure the local SD 1.5 base model exists and torch.cuda.is_available() returns True.
4. Select a trained face model in the Video section and request a short clip.

Video generation uses AnimateDiff with the locally stored SD 1.5 model, motion adapter and optional face LoRA. Results are saved under data/videos. Short clips can take several minutes, and the face can drift across frames. The current CPU-only setup cannot run this feature.

## Local storage and privacy

- Face dataset: data/face_dataset/
- Generated photos: data/outputs/
- Generated videos: data/videos/
- Base model: models/stable-diffusion-v1-5/
- Face LoRAs: models/face_loras/
- Motion adapter: models/animatediff-motion-adapter-v1-5-2/

Generated files, uploads and model weights are ignored by Git. Do not commit private photos, LoRA weights, model files or generated media. The web app does not use a remote inference API; initial software/model downloads do require internet.

## Limitations

- The UI has no authentication and is intended for localhost use only.
- Video generation requires compatible GPU software and model weights.
- No workflow guarantees exact face identity in every generated frame.
- Review model terms before publishing or using outputs commercially.
