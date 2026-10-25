import gc
import io
import os
import re
import time
import uuid
import threading
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from flask import Flask, request, render_template, send_from_directory, jsonify
from PIL import Image, ImageOps, UnidentifiedImageError

APP_DIR = Path(__file__).resolve().parent
INPUT_DIR = APP_DIR / "data" / "inputs"
OUTPUT_DIR = APP_DIR / "data" / "outputs"
DATASET_DIR = APP_DIR / "data" / "face_dataset"
VIDEO_OUTPUT_DIR = APP_DIR / "data" / "videos"
MODEL_DIR = APP_DIR / "models" / "stable-diffusion-v1-5"
LORA_DIR = APP_DIR / "models" / "face_loras"
MOTION_DIR = APP_DIR / "models" / "animatediff-motion-adapter-v1-5-2"
for directory in (INPUT_DIR, OUTPUT_DIR, DATASET_DIR, VIDEO_OUTPUT_DIR, MODEL_DIR, LORA_DIR):
    directory.mkdir(parents=True, exist_ok=True)

Image.MAX_IMAGE_PIXELS = 40_000_000
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
MODEL_ID = os.environ.get("SD_MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5")

pipe = None
pipeline_mode = None
active_lora = None
model_error = None
video_pipe = None
active_video_lora = None
video_model_error = None
model_lock = threading.Lock()
generation_lock = threading.Lock()


def safe_model_name(value):
    value = (value or "").strip()
    if not value:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
        raise ValueError("Invalid face model name.")
    return value


def list_face_models():
    if not LORA_DIR.exists():
        return []
    result = []
    for folder in sorted(LORA_DIR.iterdir()):
        if folder.is_dir() and (
            (folder / "pytorch_lora_weights.safetensors").is_file()
            or (folder / "pytorch_lora_weights.bin").is_file()
        ):
            result.append(folder.name)
    return result


def dataset_count():
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sum(1 for p in DATASET_DIR.iterdir() if p.is_file() and p.suffix.lower() in exts)


def save_input_image(file_storage):
    if not file_storage or not file_storage.filename:
        return None
    try:
        with Image.open(file_storage.stream) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise ValueError("Use a valid PNG, JPG, or WEBP image.")
    return ImageOps.fit(image, (512, 512), method=Image.Resampling.LANCZOS)


def store_dataset_images(files):
    selected = [item for item in files if item and item.filename]
    if not selected:
        raise ValueError("Choose at least one face photo.")
    if len(selected) > 30:
        raise ValueError("Upload at most 30 photos per batch.")
    prepared = []
    for item in selected:
        raw = item.stream.read(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("Each photo must be 8 MB or smaller.")
        try:
            with Image.open(io.BytesIO(raw)) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.load()
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            raise ValueError("A file was not a valid supported image.")
        if image.width < 128 or image.height < 128:
            raise ValueError("Each photo must be at least 128 × 128 pixels.")
        prepared.append(image)
    # Re-encode locally to strip EXIF metadata and ignore original filenames.
    for image in prepared:
        image.save(DATASET_DIR / f"face_{uuid.uuid4().hex}.jpg", "JPEG", quality=95, optimize=True)
    return len(prepared)


def release_image_pipeline():
    global pipe, pipeline_mode, active_lora
    pipe = None
    pipeline_mode = None
    active_lora = None
    gc.collect()


def get_pipeline(mode, face_model):
    global pipe, pipeline_mode, active_lora, model_error
    global video_pipe, active_video_lora
    import torch
    from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline

    with model_lock:
        if pipe is None or pipeline_mode != mode:
            release_image_pipeline()
            video_pipe = None
            active_video_lora = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            local_model = MODEL_DIR if (MODEL_DIR / "model_index.json").is_file() else MODEL_ID
            cls = StableDiffusionImg2ImgPipeline if mode == "img2img" else StableDiffusionPipeline
            kwargs = {
                "torch_dtype": torch.float32,
                "safety_checker": None,
                "requires_safety_checker": False,
            }
            if local_model == MODEL_DIR:
                kwargs["local_files_only"] = True
            try:
                loaded = cls.from_pretrained(local_model, **kwargs).to("cpu")
                loaded.enable_attention_slicing("max")
                try:
                    loaded.enable_vae_slicing()
                except Exception:
                    pass
                pipe, pipeline_mode, active_lora = loaded, mode, None
                model_error = None
            except Exception as exc:
                model_error = str(exc)
                raise

        if active_lora != face_model:
            if active_lora is not None:
                pipe.unload_lora_weights()
                active_lora = None
            if face_model:
                path = LORA_DIR / face_model
                if not path.is_dir() or face_model not in list_face_models():
                    raise ValueError("That face model is missing or is not a completed LoRA.")
                pipe.load_lora_weights(str(path))
                active_lora = face_model
        return pipe


def get_video_pipeline(face_model):
    global video_pipe, active_video_lora, video_model_error
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Video generation requires an NVIDIA GPU and CUDA-enabled PyTorch. "
            "This CPU-only setup cannot run AnimateDiff."
        )
    if not (MOTION_DIR / "config.json").is_file():
        raise RuntimeError(
            "The AnimateDiff motion adapter is missing. Review its model card and run "
            "download_video_model.bat while online."
        )
    if not (MODEL_DIR / "model_index.json").is_file():
        raise RuntimeError("The local SD 1.5 model is missing. Run download_model.bat first.")

    from diffusers import AnimateDiffPipeline, MotionAdapter
    with model_lock:
        release_image_pipeline()
        adapter = None
        if video_pipe is None:
            adapter = MotionAdapter.from_pretrained(
                str(MOTION_DIR), local_files_only=True, torch_dtype=torch.float16
            )
            loaded = AnimateDiffPipeline.from_pretrained(
                str(MODEL_DIR), motion_adapter=adapter, local_files_only=True,
                torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False,
            )
            loaded.enable_vae_slicing()
            loaded.enable_model_cpu_offload()
            video_pipe = loaded
            active_video_lora = None
            video_model_error = None

        if active_video_lora != face_model:
            if active_video_lora is not None:
                video_pipe.unload_lora_weights()
                active_video_lora = None
            if face_model:
                path = LORA_DIR / face_model
                if not path.is_dir() or face_model not in list_face_models():
                    raise ValueError("That face model is missing or is not a completed LoRA.")
                video_pipe.load_lora_weights(str(path))
                active_video_lora = face_model
        return video_pipe


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify(error="Upload too large. Keep the combined request under 64 MB."), 413


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify(
        status="ok",
        image_model_loaded=pipe is not None,
        image_model_error=model_error,
        face_dataset_images=dataset_count(),
        face_models=list_face_models(),
        video_model_loaded=video_pipe is not None,
        video_adapter_ready=(MOTION_DIR / "config.json").is_file(),
        video_model_error=video_model_error,
        note="Inference stays local; video generation requires CUDA-enabled NVIDIA GPU.",
    )


@app.get("/api/face-models")
def api_face_models():
    return jsonify(models=list_face_models(), dataset_images=dataset_count())


@app.post("/api/face-dataset")
def api_face_dataset():
    try:
        added = store_dataset_images(request.files.getlist("photos"))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(status="completed", added=added, total=dataset_count(),
                   message="Photos saved locally. Review the dataset before training.")


@app.post("/generate")
def generate():
    prompt = (request.form.get("prompt") or "").strip()
    negative_prompt = (request.form.get("negative_prompt") or "").strip()
    try:
        face_model = safe_model_name(request.form.get("face_model"))
        strength = float(request.form.get("strength", "0.45"))
        steps = int(request.form.get("steps", "10"))
    except (ValueError, TypeError) as exc:
        return jsonify(error=str(exc) or "Invalid generation settings."), 400
    if not prompt:
        return jsonify(error="Enter a prompt describing the image you want."), 400
    if len(prompt) > 1200 or len(negative_prompt) > 600:
        return jsonify(error="Prompt is too long."), 400
    if not 0.2 <= strength <= 0.95 or not 5 <= steps <= 25:
        return jsonify(error="Strength must be 0.20–0.95 and steps must be 5–25."), 400
    try:
        image = save_input_image(request.files.get("image"))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    if face_model and face_model not in list_face_models():
        return jsonify(error="Choose a completed face model from the list."), 400
    if not generation_lock.acquire(blocking=False):
        return jsonify(error="Another generation is running. Please wait."), 429
    try:
        try:
            import torch
            mode = "img2img" if image is not None else "txt2img"
            pipeline = get_pipeline(mode, face_model)
            kwargs = {
                "prompt": prompt, "negative_prompt": negative_prompt or None,
                "num_inference_steps": steps, "guidance_scale": 7.0,
                "generator": torch.Generator(device="cpu").manual_seed(int(time.time_ns() % (2**31 - 1))),
                "height": 512, "width": 512,
            }
            if image is not None:
                kwargs.update(image=image, strength=strength)
            result = pipeline(**kwargs).images[0]
        except Exception as exc:
            message = str(exc)
            if "out of memory" in message.lower() or "memory" in message.lower():
                message = "The model ran out of memory. Close other apps and retry with fewer steps."
            return jsonify(error=f"Generation failed: {message}"), 500
        filename = f"{uuid.uuid4().hex}.png"
        result.save(OUTPUT_DIR / filename, format="PNG", optimize=True)
        return jsonify(status="completed", image_url=f"/outputs/{filename}",
                       filename=filename, mode=mode, face_model=face_model,
                       message="Generated locally on this computer.")
    finally:
        generation_lock.release()


@app.post("/generate-video")
def generate_video():
    prompt = (request.form.get("prompt") or "").strip()
    negative_prompt = (request.form.get("negative_prompt") or "").strip()
    try:
        face_model = safe_model_name(request.form.get("face_model"))
        steps = int(request.form.get("steps", "12"))
        frames = int(request.form.get("frames", "16"))
    except (ValueError, TypeError) as exc:
        return jsonify(error=str(exc) or "Invalid video settings."), 400
    if not prompt or len(prompt) > 1000 or len(negative_prompt) > 600:
        return jsonify(error="Enter a video prompt within the allowed length."), 400
    if not 5 <= steps <= 20 or frames not in (8, 16, 24):
        return jsonify(error="Choose 5–20 steps and 8, 16, or 24 frames."), 400
    if face_model and face_model not in list_face_models():
        return jsonify(error="Choose a completed face model from the list."), 400
    if not generation_lock.acquire(blocking=False):
        return jsonify(error="Another generation is running. Please wait."), 429
    try:
        try:
            import torch
            from diffusers.utils import export_to_video
            pipeline = get_video_pipeline(face_model)
            generator = torch.Generator(device="cuda").manual_seed(int(time.time_ns() % (2**31 - 1)))
            result = pipeline(
                prompt=prompt, negative_prompt=negative_prompt or None,
                num_frames=frames, num_inference_steps=steps, guidance_scale=7.0,
                generator=generator,
            )
            filename = f"{uuid.uuid4().hex}.mp4"
            export_to_video(result.frames[0], str(VIDEO_OUTPUT_DIR / filename), fps=8)
        except Exception as exc:
            message = str(exc)
            if "out of memory" in message.lower():
                message = "The GPU ran out of memory. Try fewer frames or a smaller workload."
            return jsonify(error=f"Video generation failed: {message}"), 500
        return jsonify(status="completed", video_url=f"/videos/{filename}", filename=filename,
                       face_model=face_model, message="Video generated locally.")
    finally:
        generation_lock.release()


@app.get("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)


@app.get("/videos/<path:filename>")
def videos(filename):
    return send_from_directory(VIDEO_OUTPUT_DIR, filename, as_attachment=False, mimetype="video/mp4")


if __name__ == "__main__":
    print("\nLocal Private Image AI")
    print("Open http://127.0.0.1:5000 in your browser.")
    print("Only this computer can access the app. Do not change host to 0.0.0.0.")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
