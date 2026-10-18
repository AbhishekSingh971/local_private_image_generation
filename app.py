import os
# Set conservative CPU threading before importing torch.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import io
import time
import uuid
import threading
from pathlib import Path
from flask import Flask, request, render_template, send_from_directory, jsonify
from PIL import Image, ImageOps, UnidentifiedImageError

APP_DIR = Path(__file__).resolve().parent
INPUT_DIR = APP_DIR / "data" / "inputs"
OUTPUT_DIR = APP_DIR / "data" / "outputs"
MODEL_DIR = APP_DIR / "models" / "stable-diffusion-v1-5"
for folder in (INPUT_DIR, OUTPUT_DIR, MODEL_DIR):
    folder.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12 MB upload limit

# This app intentionally binds to localhost only when started using run_local.bat.
# Model: Stable Diffusion v1.5 (img2img). You must accept the model's license
# on its model page before downloading it.
MODEL_ID = os.environ.get("SD_MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5")
pipe = None
model_error = None
model_lock = threading.Lock()
generation_lock = threading.Lock()
is_loading = False

def get_pipeline():
    global pipe, model_error, is_loading
    with model_lock:
        if pipe is not None:
            return pipe
        is_loading = True
        try:
            import torch
            from diffusers import StableDiffusionImg2ImgPipeline
            torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))
            # Local model cache only after initial download. Set local_files_only=True
            # so inference never attempts to contact the model host.
            local_model = MODEL_DIR if any(MODEL_DIR.iterdir()) else MODEL_ID
            kwargs = {
                "torch_dtype": torch.float32,
                "safety_checker": None,
                "requires_safety_checker": False,
            }
            if local_model == MODEL_DIR:
                kwargs["local_files_only"] = True
            loaded = StableDiffusionImg2ImgPipeline.from_pretrained(local_model, **kwargs)
            loaded = loaded.to("cpu")
            loaded.enable_attention_slicing("max")
            try:
                loaded.enable_vae_slicing()
            except Exception:
                pass
            pipe = loaded
            model_error = None
            return pipe
        except Exception as exc:
            model_error = str(exc)
            raise
        finally:
            is_loading = False


def save_uploaded_image(file_storage):
    if not file_storage or not file_storage.filename:
        raise ValueError("Please choose an input image.")

    try:
        image = Image.open(file_storage.stream)
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.load()
    except (UnidentifiedImageError, OSError, ValueError):
        raise ValueError(
            "That file is not a supported image. Try PNG, JPG, or WEBP."
        )

    # Preserve the original width and height.
    return image


@app.get("/")
def index():
    return render_template("index.html")

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": pipe is not None,
        "model_error": model_error,
        "model_path": str(MODEL_DIR),
        "note": "Inference runs locally on CPU. First model download requires internet."
    })

@app.post("/generate")
def generate():
    prompt = (request.form.get("prompt") or "").strip()
    negative_prompt = (request.form.get("negative_prompt") or "").strip()
    strength_raw = request.form.get("strength", "0.25")
    steps_raw = request.form.get("steps", "25")
    if not prompt:
        return jsonify(error="Enter a prompt describing the edit you want."), 400
    if len(prompt) > 1200 or len(negative_prompt) > 600:
        return jsonify(error="Prompt is too long."), 400
    try:
        strength = float(strength_raw)
        steps = int(steps_raw)
    except ValueError:
        return jsonify(error="Invalid strength or step count."), 400
    if not 0.2 <= strength <= 0.95:
        return jsonify(error="Strength must be between 0.20 and 0.95."), 400
    if not 5 <= steps <= 25:
        return jsonify(error="Steps must be between 5 and 25 for this CPU setup."), 400

    upload = request.files.get("image")
    try:
        init_image = save_uploaded_image(upload)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    # One CPU generation at a time to reduce memory pressure.
    if not generation_lock.acquire(blocking=False):
        return jsonify(error="A generation is already running. Please wait for it to finish."), 429

    try:
        generator = None
        try:
            import torch
            generator = torch.Generator(device="cpu").manual_seed(int(time.time_ns() % (2**31 - 1)))
            pipeline = get_pipeline()
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                image=init_image,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=7.0,
                generator=generator,
            ).images[0]
        except Exception as exc:
            message = str(exc)
            if "out of memory" in message.lower() or "memory" in message.lower():
                message = (
                    "The model ran out of available memory. Close other apps and retry with "
                    "a smaller input image or fewer steps. With 8 GB RAM, this model may exceed "
                    "your PC's available memory."
                )
            return jsonify(error=f"Generation failed: {message}"), 500

        result_id = f"{uuid.uuid4().hex}.png"
        result.save(OUTPUT_DIR / result_id, format="PNG", optimize=True)
        return jsonify(
            status="completed",
            image_url=f"/outputs/{result_id}",
            filename=result_id,
            message="Generated locally on this computer."
        )
    finally:
        generation_lock.release()

@app.get("/outputs/<path:filename>")
def outputs(filename):
    # Only serve generated output files, never arbitrary paths.
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)

if __name__ == "__main__":
    print("\nLocal Private Image AI")
    print("Open http://127.0.0.1:5000 in your browser.")
    print("Only this computer can access the app. Do not change host to 0.0.0.0.")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
